"""Check the terrain before any training touches it, and render a picture to look at.

    python scripts/preflight_terrain.py --config configs/terrain.yaml
    python scripts/preflight_terrain.py --config configs/terrain.yaml --out /tmp/terrain.png

Exit code is 1 if any check fails, so this can gate a launch.

Why this exists, in the project's own words: "a reference clip is never trained against until
it has been executed, frame by frame, by the same servos the policy will use, and looked at
by a person" (E47). Terrain is the same class of object -- a thing the run assumes and never
verifies -- and it is worse, because a wrong heightfield produces no error anywhere. It
produces a slightly wrong reward everywhere.

Every check compares two INDEPENDENTLY specified quantities. A check that compares the config
to itself is a rubber stamp, which is the note written at the top of scripts/oracle.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.envs.model_prep import prepare  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402

GREEN, RED, DIM, RESET = "\033[92m", "\033[91m", "\033[2m", "\033[0m"
failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    mark = f"{GREEN}[ok]{RESET}" if ok else f"{RED}[XX]{RESET}"
    print(f"{mark} {name}: {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "terrain.yaml")
    ap.add_argument("--out", type=Path, default=Path("/tmp/terrain_preflight.png"))
    ap.add_argument("--envs", type=int, default=256)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    tcfg = cfg.terrain
    if not tcfg.enabled:
        print("terrain.enabled is false; nothing to check")
        return 0

    print(f"\n=== terrain preflight: {args.config.name} ===\n")
    print(f"{DIM}cell {tcfg.cell} m, half-extent {tcfg.half_extent} m, "
          f"grid {tcfg.rows()}x{tcfg.rows()}, p2p {tcfg.amplitude_p2p * 100:.2f} cm, "
          f"spawn +-{tcfg.spawn_half_extent} m, seed {tcfg.seed}{RESET}\n")

    model_path = REPO_ROOT / cfg.env.model_path
    flat = prepare(model_path)
    rough = prepare(model_path, terrain=tcfg)
    tf = rough.terrain
    m = rough.model
    fid = rough.floor_geom_id

    # --- the scene survived injection -------------------------------------------------
    check("floor is a heightfield",
          int(m.geom_type[fid]) == int(mujoco.mjtGeom.mjGEOM_HFIELD),
          f"geom {fid} type is {mujoco.mjtGeom(m.geom_type[fid]).name}")
    check("floor keeps its name and id",
          fid == flat.floor_geom_id,
          f"{fid} == flat {flat.floor_geom_id}; friction randomisation writes this index")
    check("contact parameters preserved",
          int(m.geom_condim[fid]) == 3
          and np.allclose(m.geom_friction[fid], flat.model.geom_friction[flat.floor_geom_id]),
          f"condim {m.geom_condim[fid]}, friction {m.geom_friction[fid]}")
    check("checker material survives",
          int(m.geom_matid[fid]) == int(flat.model.geom_matid[flat.floor_geom_id]),
          f"matid {m.geom_matid[fid]}; without it the relief is invisible in video")
    check("render buffer still 1280x960",
          m.vis.global_.offwidth == 1280 and m.vis.global_.offheight == 960,
          f"{m.vis.global_.offwidth}x{m.vis.global_.offheight} (DESIGN.md:440)")

    # --- the model the humanoid is measured against did not move ----------------------
    check("standing height bit-identical to flat",
          rough.standing_height == flat.standing_height,
          f"{rough.standing_height!r}")
    check("nominal spawn bit-identical to flat",
          rough.default_qpos[2] == flat.default_qpos[2],
          f"{rough.default_qpos[2]!r}")
    h00 = float(tf.height_at(np.zeros(1), np.zeros(1))[0])
    check("surface at the origin is exactly 0",
          abs(h00) < 1e-9, f"{h00:.15f} (the two-pass bake anchor)")

    # --- the height reader agrees with the PHYSICS, not with itself -------------------
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    rng = np.random.default_rng(12345)
    P = 400
    S = tcfg.spawn_half_extent
    px, py = rng.uniform(-S, S, P), rng.uniform(-S, S, P)
    mine = tf.height_at(px, py)
    gid = np.zeros(1, dtype=np.int32)
    ray = np.full(P, np.nan)
    for k in range(P):
        d = mujoco.mj_ray(m, data, np.array([px[k], py[k], 5.0]),
                          np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
        if d >= 0:
            ray[k] = 5.0 - d
    good = ~np.isnan(ray)
    err_mm = np.abs(mine[good] - ray[good]) * 1000.0
    check("height reader matches mj_ray",
          err_mm.max() < 1e-3,
          f"max {err_mm.max():.6f} mm, rms {np.sqrt((err_mm ** 2).mean()):.6f} mm "
          f"over {good.sum()} points -- this is what makes the reward measure the real surface")

    # --- the field is as rough as the config claims, measured on the compiled model ---
    hs = tf.height_at(px, py)
    p2p = float(np.ptp(hs))
    check("relief matches the configured amplitude",
          abs(p2p - tcfg.amplitude_p2p) < 0.2 * tcfg.amplitude_p2p,
          f"measured p2p {p2p * 100:.2f} cm over the spawn square vs configured "
          f"{tcfg.amplitude_p2p * 100:.2f} cm")
    # Stride-to-stride change, BY PATCH, because the field is deliberately heterogeneous.
    #
    # The check this replaces required a single p95 under 3.5 cm, derived from the gait-clock
    # ceiling that E57 refuted by measurement (gait_phase falls only 13.6% at nearly 3x that
    # ceiling, and what rough ground actually costs is speed). It also could not express the
    # thing the field now exists to have: VARIATION. A uniform ceiling passes a field that is
    # the same everywhere, which is the failure E57 was fixing.
    stride = 0.30
    patches = [(a, b) for a in np.linspace(-S * 0.9, S * 0.9, 9)
               for b in np.linspace(-S * 0.9, S * 0.9, 9)]
    dz_patch = []
    for cx, cy in patches:
        x = cx + rng.uniform(-1.2, 1.2, 200)
        y = cy + rng.uniform(-1.2, 1.2, 200)
        th = rng.uniform(-np.pi, np.pi, 200)
        h0 = tf.height_at(x, y)
        h1 = tf.height_at(x + stride * np.cos(th), y + stride * np.sin(th))
        dz_patch.append(float(np.percentile(np.abs(h1 - h0), 95)))
    dz_patch = np.array(dz_patch) * 100.0
    lo, hi = float(np.percentile(dz_patch, 5)), float(np.percentile(dz_patch, 95))
    check("the field contains genuinely FLAT ground",
          lo < 2.0,
          f"5th-percentile patch changes {lo:.2f} cm per 0.30 m step "
          f"(the uniform 5.25 cm field measured 1.62 cm everywhere)")
    check("the field contains genuinely ROUGH ground",
          hi > 4.0,
          f"95th-percentile patch changes {hi:.2f} cm "
          f"(the uniform 14 cm field measured 4.33 cm, at 35.9% zero-shot falls)")
    check("the roughest ground is inside what has been measured",
          float(dz_patch.max()) < 8.0,
          f"worst patch {dz_patch.max():.2f} cm; the 20 cm homogeneous field measured 6.19 cm "
          f"at 64.1% zero-shot falls, and nothing rougher has ever been measured here")
    check("difficulty actually varies across the field",
          hi / max(lo, 1e-6) > 2.5,
          f"roughest/flattest ratio {hi / max(lo, 1e-6):.1f}x -- an episode travels ~24 m and "
          f"must cross regimes, not sit in one")

    # --- the spawn rule, which is the load-bearing change ------------------------------
    env = ThreadedVecEnv(str(model_path), LocomotionTask(), args.envs, num_workers=4,
                         max_episode_steps=500, seed=7, terrain=tcfg)
    env.reset()
    s = env.state
    weight = float(-m.opt.gravity[2] * mujoco.mj_getTotalmass(m))
    force = s.foot_force.sum(axis=1)
    check("spawn does not bury the humanoid",
          np.median(force) < 0.5 * weight and np.percentile(force, 95) < 3.0 * weight,
          f"first-frame foot force median {np.median(force) / weight:.2f} BW, "
          f"p95 {np.percentile(force, 95) / weight:.2f} BW, max {force.max() / weight:.2f} BW "
          f"(no lift measured 50.7 BW; root-only lift 19.1 BW)")
    check("spawn scatter covers the square",
          np.ptp(s.qpos[:, 0]) > 1.5 * S and np.ptp(s.qpos[:, 1]) > 1.5 * S,
          f"x spans {np.ptp(s.qpos[:, 0]):.1f} m, y spans {np.ptp(s.qpos[:, 1]):.1f} m")
    check("every spawn is inside the field with margin",
          float(np.abs(s.qpos[:, :2]).max()) < tf.radius_x - 24.0,
          f"worst |xy| at spawn {np.abs(s.qpos[:, :2]).max():.1f} m, field half-extent "
          f"{tf.radius_x:.1f} m; an episode can travel ~24 m and off the field is a void")
    check("ground under the root is actually varied",
          np.ptp(s.ground_z) > 0.5 * tcfg.amplitude_p2p,
          f"ground_z spans {np.ptp(s.ground_z) * 100:.2f} cm across {args.envs} spawns")

    # --- it still walks: the terrain must not break the engine ------------------------
    rngw = np.random.default_rng(0)
    for _ in range(50):
        env.step(rngw.uniform(-0.3, 0.3, (args.envs, env.nu)).astype(np.float32))
    finite = bool(np.isfinite(env.state.qpos).all() and np.isfinite(env.state.qvel).all())
    check("50 steps produce no NaN", finite, "qpos and qvel finite")
    env.close()

    # --- and finally, a picture, because metrics have hidden three failures here -------
    try:
        _render(rough, args.out)
        check("rendered a frame to look at", args.out.exists(), f"{args.out}")
    except Exception as exc:  # noqa: BLE001
        check("rendered a frame to look at", False, f"render failed: {exc}")

    print()
    if failures:
        print(f"{RED}{len(failures)} check(s) failed: {', '.join(failures)}{RESET}")
        print("Terrain is a silent failure mode. Do not train on a field that fails a check.")
        return 1
    print(f"{GREEN}All checks passed.{RESET} Now LOOK at {args.out} before launching -- "
          "the checks prove the numbers agree, not that the ground looks like ground.")
    return 0


def _render(prepared, out: Path) -> None:
    """One frame of the humanoid standing on the terrain, from a low angle.

    Low on purpose: from above, a 5 cm relief on an 80 m field is invisible, and a picture
    that cannot show the thing it was taken to show is worse than no picture.
    """
    from humanoid_rl.terrain import place_on_terrain

    m = prepared.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    # The SAME placement the training spawn uses. Lifting by the height under the root
    # instead buries the feet 12.5 mm into the surface, and a preflight picture of a pose
    # the run never produces is worse than no picture.
    d.qpos[:] = place_on_terrain(prepared, 6.0, 6.0)
    mujoco.mj_forward(m, d)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [6.0, 6.0, float(d.qpos[2]) - 0.5]
    cam.distance, cam.azimuth, cam.elevation = 4.5, 135.0, -8.0
    with mujoco.Renderer(m, 720, 1280) as r:
        r.update_scene(d, cam)
        pixels = r.render()
    try:
        import imageio.v2 as imageio

        imageio.imwrite(out, pixels)
    except ImportError:
        from PIL import Image

        Image.fromarray(pixels).save(out)


if __name__ == "__main__":
    raise SystemExit(main())
