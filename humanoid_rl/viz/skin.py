"""Wrap a human mesh around this project's humanoid, as a render-only MuJoCo skin.

The humanoid is a stack of capsules. It walks well and reads as a robot. MuJoCo's `skin`
element fixes the appearance without touching anything else: it is consumed only by the
visualiser, so collision geoms, masses, inertias and the trained policy are all untouched.
`scripts/build_skin.py --verify` asserts that rather than trusting it, by comparing a
1000-step trajectory with and without the skin.

THE PROBLEM THIS MODULE SOLVES

MakeHuman's mesh is rigged to MakeHuman's skeleton: 163 bones, in an A-pose, with its own
limb lengths. This humanoid has 15 bodies, stands in a different pose, and has its own
proportions. Binding one to the other needs two things:

1. **Weight aggregation.** Every MakeHuman bone is assigned to one of the 15 bodies and its
   per-vertex weights summed there. Fingers collapse into the hand, the five spine bones and
   the face bones collapse into torso and head.
2. **Geometric fitting.** Each body defines a segment (its origin to its child's origin),
   and so does the corresponding pair of MakeHuman joints. Per segment we solve the
   similarity transform (rotation, uniform scale, translation) taking one to the other, then
   move the mesh with exactly the linear-blend skinning MuJoCo itself will use at runtime.
   The result is a mesh built for THIS humanoid's proportions, not a generic human scaled to
   roughly fit, which is what makes the arms sit right despite the A-pose mismatch.

After fitting, binding is trivial and exact: the mesh is already in world coordinates at the
standing pose, so `bindpos`/`bindquat` are just the bodies' world poses in that same pose,
which is precisely what MuJoCo's `mjv_updateActiveSkin` expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_rl.viz.makehuman import MakeHumanBody

#: Which body each MakeHuman bone drives, resolved by name. Order matters: the first
#: matching pattern wins, so specific prefixes precede general ones. `{}` is filled with the
#: MakeHuman side suffix (L/R) and the matching project-side prefix (left/right).
_LIMB_RULES: list[tuple[tuple[str, ...], str]] = [
    # Hands swallow every finger and metacarpal bone; the humanoid has no articulated hand.
    (("finger", "metacarpal", "wrist", "palm"), "{side}_hand"),
    (("lowerarm",), "{side}_lower_arm"),
    (("upperarm", "shoulder", "clavicle", "deltoid"), "{side}_upper_arm"),
    (("toe", "foot"), "{side}_foot"),
    (("lowerleg",), "{side}_shin"),
    (("upperleg",), "{side}_thigh"),
    # pelvis.L/.R are the hip bones themselves, part of the trunk rather than the leg.
    (("pelvis",), "pelvis"),
    (("breast",), "torso"),
]

#: Trunk and head bones, which carry no side suffix. Everything facial rides on the head.
_TRUNK_RULES: list[tuple[tuple[str, ...], str | None]] = [
    # `root` is a rig helper, not anatomy: its 16 vertices are scattered from the ankles
    # (mesh z -0.845) to the hip. Weighting them to the pelvis stretched a spike of mesh
    # from foot to waist. Excluded, and picked up by the orphan pass below instead.
    (("root",), None),
    (("spine05", "spine04"), "pelvis"),
    (("spine03", "spine02", "spine01"), "torso"),
    (("neck",), "head"),
]

#: A segment is (this body, its child) on our side, and two MakeHuman joints on theirs. The
#: MakeHuman points were chosen to be the anatomically equivalent joint centres, verified by
#: measurement: the resulting hip-to-shoulder lengths agree to 6% and thigh lengths to 0.3%.
#: `None` as the child means a terminal segment, whose endpoint is derived from the body's
#: own collision geometry instead.
#:
#: The fourth entry is only read for terminal segments and gives the direction the segment
#: points, in the body's own frame. It has to be stated rather than inferred: the obvious
#: inference, "away from the parent body", sends the foot straight DOWN, because the shin sits
#: directly above it, and the resulting rotation tipped the whole foot mesh under the floor.
#: A foot points forward, a head points up, and a hand simply continues the forearm.
SEGMENTS: dict[str, tuple[str | None, str, str, tuple[float, float, float] | None]] = {
    "pelvis": ("torso", "spine05:head", "spine02:head", None),
    # Ends at the SHOULDER LINE, not the head. Ending it at the head gave the torso a scale
    # of 0.72, which squashed the chest and, worse, put the mesh's shoulders at z=1.291 while
    # the arm bodies they bind to sit at 1.359. Seven centimetres of disagreement across the
    # deltoid is what produced the peaked, creased shoulders. Against the shoulder line the
    # scale comes out at 1.00 and the mesh shoulder lands on the joint it pivots about.
    "torso": ("left_upper_arm|right_upper_arm", "spine02:head",
              "upperarm01.L:head|upperarm01.R:head", None),
    "head": (None, "neck01:head", "head:tail", (0.0, 0.0, 1.0)),
    "{side}_upper_arm": ("{side}_lower_arm", "upperarm01.{S}:head", "lowerarm01.{S}:head", None),
    "{side}_lower_arm": ("{side}_hand", "lowerarm01.{S}:head", "wrist.{S}:head", None),
    "{side}_hand": (None, "wrist.{S}:head", "finger3-2.{S}:head", None),
    "{side}_thigh": ("{side}_shin", "upperleg01.{S}:head", "lowerleg01.{S}:head", None),
    "{side}_shin": ("{side}_foot", "lowerleg01.{S}:head", "foot.{S}:head", None),
    "{side}_foot": (None, "foot.{S}:head", "toe1-1.{S}:head", (1.0, 0.0, 0.0)),
}

#: Terminal segments take their scale from their parent limb instead of from their own
#: length. Our hand is a 4 cm sphere against MakeHuman's 12.8 cm hand, so fitting scale from
#: length shrank both hands to 31% of size. A hand is the same scale as the forearm it is on.
SCALE_PARENT: dict[str, str] = {
    "{side}_hand": "{side}_lower_arm",
    "{side}_foot": "{side}_shin",
    "head": "torso",
}

#: Skeleton hierarchy on our side, child -> parent. Used to place segments by chaining from
#: the pelvis outward, so the mesh stays connected while keeping its own proportions.
PARENTS: dict[str, str | None] = {
    "pelvis": None,
    "torso": "pelvis",
    "head": "torso",
    "{side}_upper_arm": "torso",
    "{side}_lower_arm": "{side}_upper_arm",
    "{side}_hand": "{side}_lower_arm",
    "{side}_thigh": "pelvis",
    "{side}_shin": "{side}_thigh",
    "{side}_foot": "{side}_shin",
}

_SIDES = (("left", "L"), ("right", "R"))


def _classify(bone: str) -> str | None:
    """The body a MakeHuman bone belongs to, or None if it should be ignored."""
    lowered = bone.lower()
    for side, suffix in _SIDES:
        if lowered.endswith(f".{suffix.lower()}"):
            stem = lowered[: -(len(suffix) + 1)]
            for prefixes, target in _LIMB_RULES:
                if stem.startswith(prefixes):
                    return target.format(side=side)
            # Any remaining sided bone is facial (oculi, orbicularis, ear, brow, ...).
            return "head"
    for prefixes, target in _TRUNK_RULES:
        if lowered.startswith(prefixes):
            return target
    # Unsided leftovers are the skull, jaw, tongue and teeth.
    return "head"


def _expand(template: str, side: str, suffix: str) -> str:
    return template.format(side=side, S=suffix)


def _segment_table() -> dict[str, tuple[str | None, str, str]]:
    """SEGMENTS with the `{side}` templates expanded into concrete body names."""
    table: dict[str, tuple[str | None, str, str, tuple[float, float, float] | None]] = {}
    for key, (child, head, tail, direction) in SEGMENTS.items():
        if "{side}" not in key:
            table[key] = (child, head, tail, direction)
            continue
        for side, suffix in _SIDES:
            table[_expand(key, side, suffix)] = (
                _expand(child, side, suffix) if child else None,
                _expand(head, side, suffix),
                _expand(tail, side, suffix),
                direction,
            )
    return table


def _parents() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, parent in PARENTS.items():
        if "{side}" not in key:
            out[key] = parent
            continue
        for side, suffix in _SIDES:
            out[_expand(key, side, suffix)] = (
                _expand(parent, side, suffix) if parent and "{side}" in parent else parent
            )
    return out


def _scale_parents() -> dict[str, str]:
    out: dict[str, str] = {}
    for key, parent in SCALE_PARENT.items():
        if "{side}" not in key:
            out[key] = parent
            continue
        for side, suffix in _SIDES:
            out[_expand(key, side, suffix)] = _expand(parent, side, suffix)
    return out


def _smooth_weights(
    weights: np.ndarray, face: np.ndarray, iterations: int, rate: float
) -> np.ndarray:
    """Diffuse skin weights across mesh neighbours, to widen the blend at each joint.

    MakeHuman's weights are authored for a 163-bone rig. Collapsed onto 15 bodies they get
    blockier: most vertices end up rigidly owned by one body, so at a joint two very
    different transforms meet across a single edge and the surface creases. It shows worst
    at the shoulder, where the source A-pose and this humanoid's arms-down pose differ by
    about 40 degrees.

    Averaging each vertex's weights with its neighbours spreads that transition over a band
    of vertices instead of an edge, which is the standard mitigation and costs nothing at
    runtime: it changes only which bodies influence a vertex, not how many are supported.
    """
    if iterations <= 0:
        return weights
    # Undirected edge list from the triangles, both directions, for a simple umbrella
    # (uniform Laplacian) average.
    edges = np.concatenate([face[:, [0, 1]], face[:, [1, 2]], face[:, [2, 0]]])
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(src, minlength=len(weights)).astype(float)[:, None]
    degree[degree == 0.0] = 1.0

    for _ in range(iterations):
        neighbour_sum = np.zeros_like(weights)
        np.add.at(neighbour_sum, src, weights[dst])
        weights = weights + rate * (neighbour_sum / degree - weights)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights


def _depth(name: str, parents: dict[str, str | None]) -> int:
    """Distance from the pelvis, so segments can be placed parents-first."""
    depth, node = 0, name
    while parents.get(node) is not None:
        node, depth = parents[node], depth + 1
    return depth


def _mh_point(body: MakeHumanBody, spec: str) -> np.ndarray:
    """A MakeHuman joint position. `a:head|b:head` averages several, for midlines."""
    points = []
    for part in spec.split("|"):
        name, which = part.split(":")
        head, tail = body.bone[name]
        points.append(head if which == "head" else tail)
    return np.mean(points, axis=0)


def _our_point(model: mujoco.MjModel, data: mujoco.MjData, spec: str) -> np.ndarray:
    """A body position by name. `a|b` averages several, so a pair of shoulders gives the
    midline point between them, which no single body provides."""
    return np.mean(
        [data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in spec.split("|")],
        axis=0,
    )


def _terminal_tail(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    local_direction: tuple[float, float, float] | None,
) -> np.ndarray:
    """Where a terminal segment ends: the far side of its geoms along a stated direction.

    `local_direction` is in the body's own frame, so it rotates with the body and stays
    correct in any pose. `None` means "keep going the way the parent limb was already
    pointing", which is right for a hand on the end of a forearm.

    Only the direction matters here. The LENGTH this produces is not used to set scale for
    these segments; see SCALE_PARENT for why.
    """
    origin = data.xpos[bid]
    rot = data.xmat[bid].reshape(3, 3)
    if local_direction is None:
        direction = origin - data.xpos[model.body_parentid[bid]]
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-9 else rot @ np.array([0.0, 0.0, -1.0])
    else:
        direction = rot @ np.asarray(local_direction, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)

    # Reach of the body's geoms along that direction, so the segment spans the actual part.
    reach = 0.0
    for g in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
        offset = float(np.dot(data.geom_xpos[g] - origin, direction))
        reach = max(reach, offset + float(model.geom_rbound[g]))
    return origin + direction * max(reach, 1e-3)


def _similarity(src: tuple[np.ndarray, np.ndarray], dst: tuple[np.ndarray, np.ndarray]):
    """Rotation, uniform scale and translation taking segment `src` onto segment `dst`.

    Rotation is the minimal one carrying the source direction to the target direction, so
    twist about the bone axis is left unchanged. That is what we want: both skeletons stand
    upright and face the same way, so the identity twist is already correct, and inventing
    one from a single bone vector would only introduce error.
    """
    a0, a1 = src
    b0, b1 = dst
    u, v = a1 - a0, b1 - b0
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if lu < 1e-9 or lv < 1e-9:
        return np.eye(3), 1.0, b0 - a0
    u, v = u / lu, v / lv

    axis = np.cross(u, v)
    sin, cos = float(np.linalg.norm(axis)), float(np.dot(u, v))
    if sin < 1e-12:
        rot = np.eye(3) if cos > 0 else -np.eye(3)
    else:
        axis = axis / sin
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + sin * K + (1.0 - cos) * (K @ K)

    scale = lv / lu
    return rot, scale, b0 - scale * rot @ a0


@dataclass
class FittedSkin:
    """Everything `MjSpec.add_skin` needs, plus the numbers to judge the fit by."""

    vert: np.ndarray  # (n_vert, 3) world positions at the bind pose
    face: np.ndarray  # (n_face, 3)
    bodies: list[str]
    bindpos: np.ndarray  # (n_bone, 3)
    bindquat: np.ndarray  # (n_bone, 4)
    vertid: list[np.ndarray]
    vertweight: list[np.ndarray]
    report: dict[str, float]


def fit(
    model: mujoco.MjModel,
    body: MakeHumanBody,
    qpos: np.ndarray,
    *,
    weight_floor: float = 1e-4,
    smoothing: int = 6,
    smoothing_rate: float = 0.5,
    preserve_proportions: bool = True,
) -> FittedSkin:
    """Fit the mesh to this model's proportions and bind it to the model's bodies.

    Args:
        model: The humanoid. Only read from.
        body: The MakeHuman mesh, skeleton and weights.
        qpos: Pose to bind in. Any pose works as long as the mesh is fitted in the same one,
            which it is, since both come from the single `mj_forward` below. It must
            genuinely be the STANDING pose though: `model.qpos0` is not, it leaves the root
            at z=0 with the humanoid sunk to its waist in the floor, and fitting there
            produces a mesh a metre underground. Pass `model_prep.prepare(...).default_qpos`,
            which is the pose with the feet resting on the ground. Defaulting to `qpos0`
            would silently do the wrong thing, so there is no default.
        smoothing: Laplacian smoothing passes over the weights. 0 reproduces MakeHuman's
            weights exactly and creases at the shoulders; 6 is enough to remove the crease
            without softening the elbows and knees into rubber.
        weight_floor: Weights below this are dropped, after which each vertex is
            renormalised. Trims the long tail of negligible influences that would otherwise
            triple the bone-vertex count for no visual difference.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    lowest_foot = min(
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f)][2]
        for f in ("left_foot", "right_foot")
    )
    if lowest_foot > 0.5:
        raise ValueError(
            f"fit() needs the STANDING pose, but the feet are at z={lowest_foot:.2f} m. "
            "This looks like model.qpos0; pass prepare(...).default_qpos instead."
        )

    table = _segment_table()
    bodies = [b for b in table if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]
    missing = [b for b in table if b not in bodies]
    if missing:
        raise ValueError(f"model has no bodies named {missing}")
    index_of = {name: i for i, name in enumerate(bodies)}

    # --- weights, aggregated from 163 MakeHuman bones onto our bodies ---
    weights = np.zeros((body.n_vert, len(bodies)))
    unmapped: set[str] = set()
    for bone, (ids, w) in body.weight.items():
        target = _classify(bone)
        if target is None or target not in index_of:
            unmapped.add(bone)
            continue
        np.add.at(weights[:, index_of[target]], ids, w)

    # Smooth before thresholding, so the floor trims the tail the diffusion leaves behind
    # rather than the diffusion re-creating it.
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    weights = _smooth_weights(weights, body.face, smoothing, smoothing_rate)

    weights[weights < weight_floor] = 0.0
    total = weights.sum(axis=1, keepdims=True)
    # MuJoCo rejects a skin outright if any vertex has no influence: "vertex N must have
    # positive total weight in skin". Attach any orphan to its nearest body instead.
    orphans = np.flatnonzero(total[:, 0] <= 0.0)
    if orphans.size:
        # Give each orphan the body of its nearest already-weighted vertex. Comparing against
        # body positions instead would be wrong: those live in world coordinates at standing
        # height, while the mesh is still in MakeHuman's hip-centred frame at this point, so
        # every orphan would snap to whichever body happened to be near the origin.
        anchored = np.flatnonzero(total[:, 0] > 0.0)
        for v in orphans:
            nearest = anchored[np.argmin(np.linalg.norm(body.vert[anchored] - body.vert[v], axis=1))]
            weights[v] = weights[nearest]
        total = weights.sum(axis=1, keepdims=True)
    weights /= total

    # --- per-segment similarity transforms ---
    rotations = np.zeros((len(bodies), 3, 3))
    scales = np.zeros(len(bodies))
    offsets = np.zeros((len(bodies), 3))
    scale_parent = _scale_parents()
    parents = _parents()

    if preserve_proportions:
        # ONE scale for the whole body, so the mesh keeps its own human proportions and only
        # its POSE is taken from the humanoid.
        #
        # Fitting each segment to its own bone length is the obvious thing to do and it is
        # wrong here, because this humanoid is not proportioned like a person: its head body
        # sits at z=1.339 while its shoulders sit at 1.359, so the head is mounted AT
        # shoulder height and the skeleton has no neck at all. Stretching the mesh onto that
        # produced a human with no neck. The bones still drive the motion, joint for joint;
        # they just no longer dictate how long a neck is.
        #
        # The scale is set by hip-to-ground so the feet meet the floor, which is the one
        # proportion that has to agree with the physics or the humanoid appears to hover.
        hip = _mh_point(body, table["pelvis"][1])
        scale = float(
            (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][2])
            / max(hip[2] - body.vert[:, 2].min(), 1e-9)
        )

        # Place segments outward from the pelvis. Each takes its DIRECTION from the humanoid
        # (so it follows the real joint angles) and its LENGTH and origin offset from the
        # mesh (so proportions survive).
        order = sorted(index_of, key=lambda n: _depth(n, parents))
        for name in order:
            i = index_of[name]
            child, mh_head, mh_tail, direction = table[name]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            our_tail = (
                _terminal_tail(model, data, bid, direction) if child is None
                else _our_point(model, data, child)
            )
            source = (_mh_point(body, mh_head), _mh_point(body, mh_tail))
            rot, _, _ = _similarity(source, (data.xpos[bid].copy(), our_tail))

            parent = parents[name]
            if parent is None:
                anchor = data.xpos[bid].copy()  # pelvis pins the whole body to the humanoid
            else:
                j = index_of[parent]
                anchor = scales[j] * rotations[j] @ source[0] + offsets[j]
            rotations[i], scales[i] = rot, scale
            offsets[i] = anchor - scale * rot @ source[0]
    else:
        # Parents first, so an inherited scale is already computed when it is needed.
        order = sorted(index_of, key=lambda n: n in scale_parent)
        for name in order:
            i = index_of[name]
            child, mh_head, mh_tail, direction = table[name]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            our_head = data.xpos[bid].copy()
            our_tail = (
                _terminal_tail(model, data, bid, direction) if child is None
                else _our_point(model, data, child)
            )
            source = (_mh_point(body, mh_head), _mh_point(body, mh_tail))
            rot, scale, _ = _similarity(source, (our_head, our_tail))
            if name in scale_parent:
                scale = scales[index_of[scale_parent[name]]]
            rotations[i], scales[i] = rot, scale
            offsets[i] = our_head - scale * rot @ source[0]

    # --- move the mesh, by the same linear blend MuJoCo will use at runtime ---
    # (n_vert, n_bone, 3): every body's transform applied to every vertex, then averaged by
    # weight. Vertices spanning a joint therefore blend smoothly instead of tearing.
    transformed = np.einsum("bij,vj->vbi", rotations * scales[:, None, None], body.vert)
    transformed += offsets[None, :, :]
    fitted = np.einsum("vb,vbi->vi", weights, transformed)

    # --- bind ---
    # The mesh is now in world coordinates at exactly this pose, so the bind pose is simply
    # each body's world pose here. At runtime MuJoCo computes R = xquat * conj(bindquat) and
    # t = xpos - R @ bindpos, which are identity and zero in this pose by construction.
    bindpos = np.zeros((len(bodies), 3))
    bindquat = np.zeros((len(bodies), 4))
    for name, i in index_of.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        bindpos[i] = data.xpos[bid]
        bindquat[i] = data.xquat[bid]

    vertid = [np.flatnonzero(weights[:, i] > 0.0).astype(np.int32) for i in range(len(bodies))]
    vertweight = [weights[ids, i].astype(np.float32) for i, ids in enumerate(vertid)]

    report = {
        "vertices": float(body.n_vert),
        "faces": float(len(body.face)),
        "bones": float(len(bodies)),
        "bone_vertex_pairs": float(sum(v.size for v in vertid)),
        "mean_bones_per_vertex": float(sum(v.size for v in vertid) / body.n_vert),
        "unmapped_makehuman_bones": float(len(unmapped)),
        "orphan_vertices_reattached": float(orphans.size),
        "weight_smoothing_passes": float(smoothing),
        "preserve_proportions": float(preserve_proportions),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "fitted_height": float(fitted[:, 2].max() - fitted[:, 2].min()),
        "fitted_lowest_z": float(fitted[:, 2].min()),
    }
    return FittedSkin(
        vert=fitted,
        face=body.face,
        bodies=bodies,
        bindpos=bindpos,
        bindquat=bindquat,
        vertid=vertid,
        vertweight=vertweight,
        report=report,
    )


#: Skin shading. Flat matte diffuse is what makes an untextured body read as a corpse: real
#: skin has a soft broad highlight, and without any specular term the muscle definition the
#: mesh actually has is invisible, because nothing catches the light. A little reflectance
#: keeps it from looking like plastic.
SKIN_MATERIAL = dict(rgba=(0.86, 0.66, 0.54, 1.0), specular=0.25, shininess=0.28,
                     reflectance=0.03)


def attach(spec: mujoco.MjSpec, skin: FittedSkin, *, rgba=(0.86, 0.66, 0.54, 1.0),
           inflate: float = 0.0, group: int = 0) -> None:
    """Add the fitted skin to a spec, in place.

    Every array must be passed FLAT. A 2-D (N, 3) numpy array raises
    `RuntimeError: Unable to cast Python instance of type <class 'numpy.ndarray'> to C++
    type '?'`, which is an unhelpful message for a shape problem, so `.ravel()` is applied
    here rather than left to callers.
    """
    material = spec.add_material()
    material.name = "skin"
    material.rgba = list(SKIN_MATERIAL["rgba"])
    material.specular = SKIN_MATERIAL["specular"]
    material.shininess = SKIN_MATERIAL["shininess"]
    material.reflectance = SKIN_MATERIAL["reflectance"]

    sk = spec.add_skin()
    sk.name = "human"
    sk.material = "skin"
    sk.vert = skin.vert.astype(np.float32).ravel()
    sk.face = skin.face.astype(np.int32).ravel()
    sk.bodyname = list(skin.bodies)
    sk.bindpos = skin.bindpos.astype(np.float32).ravel()
    sk.bindquat = skin.bindquat.astype(np.float32).ravel()
    sk.vertid = [v.astype(np.int32) for v in skin.vertid]
    sk.vertweight = [w.astype(np.float32) for w in skin.vertweight]
    sk.rgba = list(rgba)
    sk.inflate = inflate
    sk.group = group
