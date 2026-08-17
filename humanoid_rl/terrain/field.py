"""Rough ground: one static heightfield, generated once, baked into the model, never mutated.

Phase 4. The design is deliberately the smallest thing that is genuinely rough ground:
ONE band-limited field, shared identically by every model in the domain-randomisation pool,
read-only for the life of the run. Terrain variety comes from spawning each environment at a
different (x, y), not from giving each environment its own terrain.

Three failures are avoided BY CONSTRUCTION rather than by remembering to be careful, and all
three are silent -- they corrupt conclusions instead of crashing:

1. `vec_env` resets and `mj_forward`s against `self.model`, which is `model_pool[0]`, while
   it STEPS `pool[env_model[i]]`. Per-pool-entry terrain would evaluate the first observation
   of every episode -- every touch and framepos sensor -- against a different heightfield from
   the one the episode then runs on.
2. `build_model_pool` returns a pool of ONE whenever domain randomisation is off, which is
   exactly how the trainer builds its evaluation env and its video env. Per-entry terrain
   would mean training on rough ground and then evaluating and filming on whatever entry 0
   happened to hold. That is the E16 shape: a measurement configured differently from the
   thing it measures.
3. `render.py` hands the Renderer `env.model`, i.e. pool[0], while the env being filmed may
   be stepping another entry.

Because the field is never mutated, the whole render problem recorded in DESIGN.md:434-446
dissolves. `mujoco.Renderer` genuinely has no `update_hfield` (verified against the installed
3.11.0), so a field edited after the renderer exists would render byte-identically stale --
but a field baked in before compile is uploaded by `MjrContext` construction like any other
asset, and `render.py` builds a fresh Renderer per video inside a `with` block. No
`mjr_uploadHField`, no private `GLContext`, no dedicated render thread.

INSTRUMENTATION BUG #10, found while building this and the reason `TerrainField` cannot be
constructed from an authored array. **MuJoCo's compiler renormalises heightfield data to
exactly [0, 1]**: it subtracts the minimum and divides by the range. An authored grid with
min -0.352642 and max 0.647358 arrives in the compiled model as min 0.0, max 1.0. A height
reader built from the array you wrote is therefore wrong by a constant offset -- measured at
64.75 mm, with max and rms equal, which is the signature of a pure offset and would have
presented as "the terrain reward is subtly wrong everywhere" with no error anywhere.

So the compiled model is the single source of truth, and the only constructor takes one.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

FLOOR_GEOM = "floor"


@dataclass
class TerrainConfig:
    """Every number here is pinned to a measurement; see the notes on each field."""

    enabled: bool = False

    #: Grid spacing in metres. THE cost dial: collision cost is set by cell size alone and is
    #: independent of the field's extent. Measured single-thread `mj_step` ratio against a
    #: plane, same humanoid, same pose: 0.200 m -> 1.22x, 0.150 m -> 1.26x, 0.100 m -> 1.79x,
    #: 0.075 m -> 1.97x, 0.050 m -> 3.25x.
    #:
    #: 0.10 m is chosen on the MINIMUM contact count, not the median. At 0.15 m a box foot
    #: rests on a median of 3 contact points but a minimum of ONE, and a foot on one contact
    #: cannot transmit ankle torque from the ground -- a policy learning against that at a
    #: measurable fraction of footfalls is learning against a degenerate foot. At 0.10 m the
    #: minimum is 2. Below 0.075 m the disturbance stops growing while the cost keeps going.
    cell: float = 0.10

    #: Half-extent in metres, so the field spans [-half_extent, +half_extent] on both axes.
    #: Extent is FREE in physics (cost tracks cells under the foot, not grid size), so it is
    #: sized purely by the travel envelope: max forward command 1.5 m/s x 20.0 s episode =
    #: 30.0 m, discounted by the measured achieved/commanded ratio ~0.79 to ~23.8 m. Spawns
    #: are confined to +-12 m, so the worst reachable radius is ~38 m against 40.
    #: Off the field there is no geom at all and the humanoid falls into the void, which
    #: would read as an ordinary fall -- hence the Oracle invariant rather than a comment.
    half_extent: float = 40.0

    #: Peak-to-peak relief in metres over the spawn square.
    #:
    #: Chosen against the gait clock's own tolerance rather than picked. The reward's
    #: `gait_phase` term is 27.8% of the budget and has a `stance_transition_width` of 0.07
    #: cycles = 78 ms at 0.9 Hz. Converting stride-to-stride ground change at p95 into a
    #: touchdown timing error at the descent speed the swing apex implies:
    #:
    #:   p2p 1.75 cm -> 20.0 ms (0.26x the clock tolerance), open-loop PD hold 23/48
    #:   p2p 3.50 cm -> 38.9 ms (0.50x),                     PD hold 16/48
    #:   p2p 5.25 cm -> 60.5 ms (0.78x),                     PD hold 11/48
    #:   p2p 7.00 cm -> 77.8 ms (1.00x),                     PD hold  7/48
    #:
    #: 7.00 cm is the CEILING: past it the terrain, not the policy, is what loses the gait
    #: clock, and a run there would be measuring the ground. Run 1 takes 0.75 of the ceiling.
    #: Flat ground holds 41/48 by the same open-loop test, so 5.25 cm is a measured 2.4x
    #: reduction in open-loop stability, not an adjective.
    amplitude_p2p: float = 0.0525

    #: Two octaves of smoothed Gaussian noise. The short correlation is the measured stride
    #: length, so the ground changes meaningfully between one footfall and the next -- which
    #: is what makes it rough rather than merely sloped.
    correlation_short: float = 0.35
    correlation_long: float = 1.20
    long_weight: float = 0.25
    clip_sigma: float = 2.5

    #: A flat disc at the origin, for exactly one reason: `_compute_standing_height` and
    #: `_verify_pd_holds_pose` both run the humanoid at x = y = 0, so this makes `prepare()`
    #: byte-identical to a plane run and keeps `standing_height` from depending on terrain.
    flat_disc_radius: float = 2.0

    #: Spawn square half-width. See `half_extent` for the margin arithmetic.
    spawn_half_extent: float = 12.0

    seed: int = 0

    def rows(self) -> int:
        """Grid side. Odd, so a sample lands exactly on the origin."""
        return int(round(2.0 * self.half_extent / self.cell)) + 1


def generate(cfg: TerrainConfig) -> np.ndarray:
    """A band-limited rough field in METRES, shape (rows, rows), before compilation.

    Band-limited rather than white: white noise at a 0.10 m cell puts a step discontinuity
    under every footfall, which is not rough ground, it is gravel the size of the foot.
    """
    n = cfg.rows()
    rng = np.random.default_rng(cfg.seed)
    raw = rng.standard_normal((n, n)).astype(np.float64)

    field = np.zeros((n, n), dtype=np.float64)
    for corr, weight in ((cfg.correlation_short, 1.0 - cfg.long_weight),
                         (cfg.correlation_long, cfg.long_weight)):
        field += weight * _box_smooth(raw, max(1, int(round(corr / cfg.cell))))

    std = float(field.std())
    if std > 0.0:
        field /= std
    np.clip(field, -cfg.clip_sigma, cfg.clip_sigma, out=field)

    # The flat disc is applied BEFORE the peak-to-peak rescale, so the configured amplitude
    # describes the terrain the humanoid actually walks on rather than being diluted by a
    # flat region whose size is a separate decision.
    axis = np.linspace(-cfg.half_extent, cfg.half_extent, n)
    xx, yy = np.meshgrid(axis, axis)          # xx varies along columns, yy along rows
    r = np.hypot(xx, yy)
    if cfg.flat_disc_radius > 0.0:
        # Smooth over one correlation length so the disc edge is not itself an obstacle.
        blend = np.clip((r - cfg.flat_disc_radius) / max(cfg.correlation_long, 1e-6), 0.0, 1.0)
        field *= blend * blend * (3.0 - 2.0 * blend)      # smoothstep

    span = float(field.max() - field.min())
    if span > 0.0:
        field *= cfg.amplitude_p2p / span
    return np.ascontiguousarray(field - field.min(), dtype=np.float32)


def _box_smooth(a: np.ndarray, half: int) -> np.ndarray:
    """Separable box blur by summed-area table. Cheap, and its spectrum is good enough."""
    if half <= 0:
        return a.copy()
    out = a
    for axis in (0, 1):
        c = np.cumsum(np.pad(out, [(half + 1, half) if i == axis else (0, 0)
                                   for i in range(2)], mode="edge"), axis=axis)
        lo = np.take(c, np.arange(0, out.shape[axis]), axis=axis)
        hi = np.take(c, np.arange(2 * half + 1, out.shape[axis] + 2 * half + 1), axis=axis)
        out = (hi - lo) / (2 * half + 1)
    return out


class TerrainField:
    """Reads terrain height out of a COMPILED model. There is no other constructor.

    That is the whole point: see instrumentation bug #10 in the module docstring. Building
    this from the array handed to the compiler gives a reader that is wrong by a constant,
    silently, everywhere.
    """

    def __init__(self, model: mujoco.MjModel, floor_geom_id: int,
                 spawn_half_extent: float | None = None) -> None:
        hf = int(model.geom_dataid[floor_geom_id])
        if hf < 0:
            raise ValueError(
                f"geom {floor_geom_id} is not a heightfield; geom_dataid is {hf}. "
                "TerrainField must be built from a compiled model whose floor was converted."
            )
        self.nrow = int(model.hfield_nrow[hf])
        self.ncol = int(model.hfield_ncol[hf])
        rx, ry, z_scale, _base = (float(v) for v in model.hfield_size[hf])
        self.radius_x, self.radius_y, self.z_scale = rx, ry, z_scale
        self.pos_z = float(model.geom_pos[floor_geom_id, 2])
        adr = int(model.hfield_adr[hf])
        # Copy: hfield_data is a live view of the model, and this object outlives the caller's
        # assumptions about who owns it.
        self.grid = np.array(
            model.hfield_data[adr:adr + self.nrow * self.ncol], dtype=np.float64
        ).reshape(self.nrow, self.ncol)
        # Carried here so the task never re-derives it and the two can never disagree. The
        # margin against `radius_x` is what keeps an episode from walking off the field, where
        # there is no geom at all and the humanoid falls into the void -- which would be
        # recorded as an ordinary fall and read as a policy failure.
        self.spawn_half_extent = (
            float(spawn_half_extent) if spawn_half_extent is not None
            else 0.3 * min(self.radius_x, self.radius_y)
        )

    def height_at(self, x, y):
        """World z of the surface directly under each (x, y). Exact, not interpolated.

        Reproduces MuJoCo's own triangulation -- each cell split along the diagonal joining
        (row i, col j) to (row i+1, col j+1) -- so it agrees with `mj_ray` to 0.00000 mm.
        Bilinear on the same field is off by 0.63 mm, and the opposite diagonal by 1.06 mm;
        exactness is free here, and a reader that disagrees with the physics is precisely the
        instrumentation bug this project keeps finding.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Grid coordinates. Column index runs along x, row index along y.
        gx = (x + self.radius_x) / (2.0 * self.radius_x) * (self.ncol - 1)
        gy = (y + self.radius_y) / (2.0 * self.radius_y) * (self.nrow - 1)
        gx = np.clip(gx, 0.0, self.ncol - 1 - 1e-9)
        gy = np.clip(gy, 0.0, self.nrow - 1 - 1e-9)

        j = np.floor(gx).astype(np.int64)
        i = np.floor(gy).astype(np.int64)
        u = gx - j
        v = gy - i
        j1 = np.minimum(j + 1, self.ncol - 1)
        i1 = np.minimum(i + 1, self.nrow - 1)

        a = self.grid[i, j]        # (i,   j)
        b = self.grid[i, j1]       # (i,   j+1)
        c = self.grid[i1, j]       # (i+1, j)
        e = self.grid[i1, j1]      # (i+1, j+1)

        upper = a + (e - c) * u + (c - a) * v
        lower = a + (b - a) * u + (e - b) * v
        return self.pos_z + self.z_scale * np.where(v >= u, upper, lower)

    def probe_max(self, x, y, offsets: np.ndarray):
        """Highest surface point over a set of body-frame (dx, dy) probes around each (x, y).

        This is the spawn rule, and it is the load-bearing change of the whole design.
        Spawning at the unmodified nominal root height on rough ground drives 30,295 N under
        one foot on the very first frame -- 61.7x body weight against a 9.82 N contact
        threshold, so `foot_contact` is trivially true and `foot_force` (which the tasks read
        in body-weight units) is off by a factor of sixty on the first observation.
        Offsetting by the height under the ROOT alone is not enough: it still leaves 32.2 BW,
        because a 17.7 x 9.0 cm box foot straddles cells the root does not. Taking the
        maximum over both footprints brings it to a median of 0.0 N and a peak of 1.80 BW.
        """
        x = np.asarray(x, dtype=np.float64)[:, None]
        y = np.asarray(y, dtype=np.float64)[:, None]
        h = self.height_at(x + offsets[None, :, 0], y + offsets[None, :, 1])
        return h.max(axis=1)


def foot_probe_offsets(prepared) -> np.ndarray:
    """Body-frame (dx, dy) probes covering both footprints at the nominal pose.

    ONE definition, used by the task's spawn rule and by every tool that wants to place the
    humanoid on terrain the way training does. Two implementations of this would drift, and
    the symptom would be a preflight picture of a pose the run never produces -- which is
    the instrument-disagrees-with-the-thing failure this project has hit nine times.
    """
    model = prepared.model
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = prepared.default_qpos
    mujoco.mj_forward(model, data)
    root_xy = np.asarray(prepared.default_qpos[0:2], dtype=float)
    pts = []
    for g in np.asarray(prepared.foot_geom_ids, dtype=int):
        centre = np.asarray(data.geom_xpos[g][:2], dtype=float) - root_xy
        hx, hy = float(model.geom_size[g][0]), float(model.geom_size[g][1])
        for dx in (-hx, 0.0, hx):
            for dy in (-hy, 0.0, hy):
                pts.append(centre + np.array([dx, dy]))
    return np.asarray(pts, dtype=float)


def place_on_terrain(prepared, x: float, y: float) -> np.ndarray:
    """`default_qpos` moved to (x, y) and lifted onto the surface, exactly as training does.

    Any tool that wants "the humanoid, standing where an episode would start it" calls this
    rather than reimplementing the lift.
    """
    q = np.array(prepared.default_qpos, dtype=float)
    q[0], q[1] = float(x), float(y)
    q[2] += float(prepared.terrain.probe_max(
        np.array([x]), np.array([y]), foot_probe_offsets(prepared))[0])
    return q


def inject(spec: mujoco.MjSpec, grid_m: np.ndarray, half_extent: float,
           pos_z: float) -> mujoco.MjSpec:
    """Turn the scene's `floor` plane into a heightfield carrying `grid_m` (in metres).

    Everything that matters about the floor geom is preserved and asserted by the preflight:
    `condim` 3, friction, the checker material (so relief is legible in video), the geom's id
    and its name -- the name matters because two `mj_name2id(..., "floor")` lookups drive the
    standing-height probe and friction domain randomisation, and a silent -1 there would
    randomise the friction of whatever geom happens to be last.
    """
    if grid_m.ndim != 2 or grid_m.shape[0] != grid_m.shape[1]:
        raise ValueError(f"terrain grid must be square, got {grid_m.shape}")
    try:
        geom = spec.geom(FLOOR_GEOM)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"scene has no geom named {FLOOR_GEOM!r}; terrain injection needs it, and so do "
            "the standing-height probe and friction randomisation"
        ) from exc
    if int(geom.condim) != 3:
        raise ValueError(f"floor condim is {geom.condim}, expected 3")

    elevation = float(grid_m.max() - grid_m.min())
    # A perfectly flat field would give elevation 0, which MuJoCo rejects. Guard it here
    # rather than let a zero-amplitude config fail at compile with an opaque message.
    elevation = max(elevation, 1e-6)

    hf = spec.add_hfield()
    hf.name = "rough"
    hf.nrow = int(grid_m.shape[0])
    hf.ncol = int(grid_m.shape[1])
    # size = (radius_x, radius_y, elevation_z, base_z). base_z is solid material hanging
    # below, so the field is never a shell the humanoid can fall through at an edge.
    hf.size = [half_extent, half_extent, elevation, 1.0]
    hf.userdata = np.asarray(grid_m, dtype=np.float64).ravel()

    geom.type = mujoco.mjtGeom.mjGEOM_HFIELD
    geom.hfieldname = "rough"
    geom.pos = [0.0, 0.0, pos_z]
    return spec


def bake(model_path, cfg: TerrainConfig, build_spec) -> tuple[mujoco.MjSpec, np.ndarray]:
    """Two-pass compile that puts the surface at the origin exactly on z = 0.

    Two passes are needed because the compiler renormalises the data to [0, 1] (bug #10), so
    the world z of the field's flat anchor is not knowable until after a compile. Pass one
    compiles at `pos_z = 0` and reads the surface height at the origin off the compiled
    model; pass two shifts the geom down by exactly that.

    This is what makes `prepare()` behave identically to a plane run: `_compute_standing_height`
    probes at x = y = 0 and `_verify_pd_holds_pose` starts the humanoid there, so both land on
    the flat disc at z = 0 and `standing_height` comes out bit-identical to flat ground.
    """
    grid = generate(cfg)

    spec = inject(build_spec(model_path), grid, cfg.half_extent, 0.0)
    model = spec.compile()
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    h00 = float(TerrainField(model, floor_id).height_at(np.zeros(1), np.zeros(1))[0])

    spec = inject(build_spec(model_path), grid, cfg.half_extent, -h00)
    return spec, grid
