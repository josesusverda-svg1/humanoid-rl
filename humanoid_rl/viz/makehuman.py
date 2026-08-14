"""Read the three MakeHuman CC0 data files, in MuJoCo's coordinate convention.

Deliberately not a MakeHuman integration. The application is a GUI written against an older
Python and it is not needed: the three files below are plain data, and this module is all the
"toolchain" required to get a rigged human mesh into this project.

    third_party/makehuman/base.obj              19,158 vertices, 18,486 quads
    third_party/makehuman/default.mhskel        163 bones, 326 named joints (JSON)
    third_party/makehuman/default_weights.mhw   per-bone vertex weights (JSON)

All three carry `"license": "CC0"` as a field inside the file, so provenance travels with the
data. That matters for this project specifically, which already had to reject LAFAN1,
LocoMuJoCo and SMPL over redistribution and NoDerivatives clauses.

Two representation details that are easy to get wrong and are handled here:

* **A joint is a vertex group, not a coordinate.** `default.mhskel` defines each joint as a
  list of vertex indices into `base.obj`; the joint's position is their centroid. The
  skeleton is therefore derived from the mesh and is guaranteed consistent with it.
* **Coordinate conventions differ.** MakeHuman is x=left, y=up, z=forward in decimetres.
  MuJoCo is x=forward, y=left, z=up in metres. That is the permutation (x, y, z) ->
  (z, x, y) plus a scale, applied once on load so nothing downstream has to remember it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Morph targets applied to the base mesh, as {filename: weight}.
#:
#: MakeHuman's base.obj is deliberately androgynous, ageless and unmuscled, because it is the
#: neutral origin that every body shape is morphed away from. Rendered as-is it reads as
#: gaunt and unwell. A target file is a sparse list of per-vertex offsets, `index dx dy dz`,
#: applied as `vertex += weight * delta`, so they compose linearly and can be blended.
#:
#: These two are the macro axes: one sets sex and age, the other musculature at average body
#: weight. Both are CC0, stated in their own file headers.
#: The muscle weight is deliberately 2.0, past the 1.0 the slider allows. MakeHuman's
#: "maximum muscle" is athletic-normal, not muscular: at 1.0 it thickens the thigh by only
#: 10%. Because a target is a plain list of vertex offsets the weight extrapolates linearly,
#: and a side-by-side of 1.0 / 2.0 / 3.0 is unambiguous: 1.0 is soft, 2.0 has real definition
#: through the back, shoulders and legs, and 3.0 goes lumpy and distorted. Do not raise it.
DEFAULT_TARGETS: dict[str, float] = {
    "caucasian-female-young.target": 1.0,
    "universal-female-young-maxmuscle-averageweight.target": 2.0,
    "bodyshapes-elvs-fem-neat-hourglass.target": 0.6,
}

#: Decimetres to metres. MakeHuman's base mesh measures 16.95 units head to toe, which is a
#: 1.695 m adult under this factor.
MAKEHUMAN_TO_METRES = 0.1


@dataclass
class MakeHumanBody:
    """The mesh, its skeleton and its skin weights, in MuJoCo axes and metres."""

    #: (n_vert, 3) vertex positions.
    vert: np.ndarray
    #: (n_face, 3) triangle indices. MuJoCo rejects quads outright, so the source quads are
    #: split on load: "Error: Face data must be multiple of 3".
    face: np.ndarray
    #: bone name -> (head, tail) positions.
    bone: dict[str, tuple[np.ndarray, np.ndarray]]
    #: bone name -> (vertex indices, weights).
    weight: dict[str, tuple[np.ndarray, np.ndarray]]

    @property
    def n_vert(self) -> int:
        return int(self.vert.shape[0])

    def height(self) -> float:
        return float(self.vert[:, 2].max() - self.vert[:, 2].min())


def _to_mujoco(points: np.ndarray, scale: float) -> np.ndarray:
    """MakeHuman (x=left, y=up, z=forward) to MuJoCo (x=forward, y=left, z=up)."""
    points = np.atleast_2d(points)
    return np.stack([points[:, 2], points[:, 0], points[:, 1]], axis=1) * scale


#: OBJ groups to render. base.obj is not just a body: it also carries MakeHuman's *helper*
#: geometry, which the application hides and which exists only to fit clothing and hair onto
#: the model. Rendering all of it puts the humanoid in a floor-length dress (`helper-skirt`),
#: gives it spikes for hair (`helper-hair`) and scatters small cubes over it (the `joint-*`
#: groups, which are the vertex cubes that define joint centres). Keeping `body` alone is
#: what MakeHuman itself shows.
BODY_GROUPS = ("body",)


def _read_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertex indices and offsets from a MakeHuman .target file, skipping its `#` header."""
    index: list[int] = []
    delta: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        index.append(int(parts[0]))
        delta.append([float(x) for x in parts[1:4]])
    return np.asarray(index, dtype=np.int64), np.asarray(delta, dtype=np.float64)


def apply_targets(vert: np.ndarray, directory: Path, targets: dict[str, float]) -> np.ndarray:
    """Morph the base mesh. Offsets are in MakeHuman's own axes, so apply before converting."""
    out = vert.copy()
    for name, weight in targets.items():
        path = directory / "targets" / name
        if not path.exists() or weight == 0.0:
            continue
        index, delta = _read_target(path)
        if index.size:
            out[index] += weight * delta
    return out


def relax(vert: np.ndarray, face: np.ndarray, iterations: int = 6, rate: float = 0.35,
          inflate_back: float = 0.34) -> np.ndarray:
    """Gently relax the surface, to repair a morph that has been pushed past its range.

    The muscle target is applied at weight 2.0, well beyond the 1.0 MakeHuman's own slider
    allows, because 1.0 is merely athletic. Extrapolating a linear morph that far makes parts
    of the surface pass through each other, which renders as thin torn slivers across the
    chest, deltoids and cheek. A side-by-side of neutral / 1.0 / 2.0 shows the tearing
    growing with the weight, so it is the extrapolation and not the mesh.

    This is Taubin's lambda-mu smoothing: a shrinking pass followed by a smaller inflating
    pass. Plain Laplacian smoothing would flatten exactly the muscle definition we went out
    of range to get, whereas alternating the sign removes the high-frequency self-intersection
    while leaving the low-frequency shape intact.
    """
    if iterations <= 0:
        return vert
    src = np.concatenate([face[:, 0], face[:, 1], face[:, 2], face[:, 1], face[:, 2], face[:, 0]])
    dst = np.concatenate([face[:, 1], face[:, 2], face[:, 0], face[:, 0], face[:, 1], face[:, 2]])
    degree = np.bincount(src, minlength=len(vert)).astype(float)[:, None]
    degree[degree == 0.0] = 1.0

    out = vert.copy()
    for _ in range(iterations):
        for step in (rate, -inflate_back):
            neighbour = np.zeros_like(out)
            np.add.at(neighbour, src, out[dst])
            out += step * (neighbour / degree - out)
    return out


def _read_obj(path: Path, keep: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangulated faces from a Wavefront OBJ, restricted to some groups.

    base.obj is 100% quads, each of which becomes two triangles by the standard fan split.
    Vertices outside the kept groups are dropped and the face indices remapped, since MuJoCo
    requires every skin vertex to carry positive weight and unused vertices would otherwise
    have to be padded with fake influences.
    """
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    active = True
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            active = line.split(maxsplit=1)[1].strip() in keep
        elif line.startswith("f ") and active:
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):  # fan, correct for triangles and quads alike
                faces.append((idx[0], idx[k], idx[k + 1]))

    vertices = np.asarray(verts, dtype=np.float64)
    face = np.asarray(faces, dtype=np.int32)
    if not face.size:
        raise ValueError(f"no faces in groups {keep}; the OBJ groups are named differently")

    used = np.unique(face)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return vertices[used], remap[face], used


def load(
    directory: str | Path,
    scale: float = MAKEHUMAN_TO_METRES,
    keep_groups: tuple[str, ...] = BODY_GROUPS,
    targets: dict[str, float] | None = None,
    relax_passes: int = 6,
) -> MakeHumanBody:
    """Load the three files and return them in MuJoCo axes and metres.

    `targets` morphs the neutral base mesh into an actual body; see DEFAULT_TARGETS. Pass an
    empty dict for the unmorphed base.
    """
    directory = Path(directory)
    _, face, kept = _read_obj(directory / "base.obj", keep_groups)
    # Joints and weights index the ORIGINAL vertex numbering, so both need remapping onto
    # the kept subset. Joint definitions deliberately use the full mesh, including helper
    # vertices, so they are resolved against it before anything is dropped.
    full_vert = np.asarray(
        [[float(x) for x in line.split()[1:4]]
         for line in (directory / "base.obj").read_text().splitlines()
         if line.startswith("v ")],
        dtype=np.float64,
    )
    # Morph first, so joint centres are read off the ACTUAL body. Reading them off the
    # neutral mesh instead would leave the skeleton fitted to a different shape than the one
    # being rendered, and the limbs would sit slightly outside their own bones.
    full_vert = apply_targets(
        full_vert, directory, DEFAULT_TARGETS if targets is None else targets
    )
    raw_vert = full_vert[kept]
    remap = np.full(len(full_vert), -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept))

    skel = json.loads((directory / "default.mhskel").read_text())
    joints = skel["joints"]

    def joint_position(name: str) -> np.ndarray:
        # A joint is a group of mesh vertices; its position is their centroid. Resolved on
        # the full mesh, because joint cubes are themselves helper geometry.
        return full_vert[np.asarray(joints[name], dtype=np.int64)].mean(axis=0)

    bone: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, record in skel["bones"].items():
        head = _to_mujoco(joint_position(record["head"]), scale)[0]
        tail = _to_mujoco(joint_position(record["tail"]), scale)[0]
        bone[name] = (head, tail)

    raw_weight = json.loads((directory / "default_weights.mhw").read_text())["weights"]
    weight = {}
    for name, pairs in raw_weight.items():
        ids = remap[np.asarray([v for v, _ in pairs], dtype=np.int64)]
        w = np.asarray([w for _, w in pairs], dtype=np.float64)
        live = ids >= 0  # drop weights that referred to discarded helper vertices
        if live.any():
            weight[name] = (ids[live], w[live])

    # Relax AFTER the joint centres have been read, so the skeleton is unaffected by it.
    return MakeHumanBody(
        vert=relax(_to_mujoco(raw_vert, scale), face, iterations=relax_passes),
        face=face,
        bone=bone,
        weight=weight,
    )
