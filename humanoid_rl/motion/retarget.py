"""Retarget mocap onto the humanoid, using MuJoCo's own kinematics.

**Retargeting** means transferring motion recorded from a human of one size and skeleton
onto a simulated character with different proportions and a different joint structure. It
cannot be a direct copy of joint angles: the source skeleton has different bone lengths,
different joint definitions, and often a different rotation convention.

The approach here is marker-based inverse kinematics. Pick a set of correspondences (source
left foot to model left foot, and so on), scale the source skeleton to the model's
proportions, then for each frame solve for the joint angles that put the model's bodies as
close as possible to the scaled source positions.

**Inverse kinematics (IK)**: given desired positions for some body parts, find the joint
angles that achieve them. Solved here by damped least squares, iterating

    dq = J^T (J J^T + lambda*I)^-1 * error

where J is the Jacobian relating joint velocities to marker velocities. The damping term
`lambda` is what keeps the solve stable near singular configurations (a fully extended leg,
for instance), where an undamped solve demands enormous joint velocities for a tiny gain.

Why MuJoCo rather than a retargeting library: `mj_jacBody` and `mj_integratePos` give exact
Jacobians and correct quaternion integration for the free root, against the *same* model
that will be trained. Joint limits come for free. It avoids importing a dependency that
would need its own robot description and its own audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_rl.motion.bvh import BvhMotion


@dataclass
class MarkerPair:
    """One source-to-model correspondence."""

    source: str  # BVH joint name (matched leniently, see BvhMotion.index_of)
    body: str  # MuJoCo body name
    weight: float = 1.0
    #: Which body segment this marker belongs to, for per-segment scaling. Humans and
    #: humanoid models rarely share proportions: the 100STYLE subject has a torso-to-leg
    #: ratio of 0.71 against our model's 0.55. Scaling everything by leg length left the
    #: head target 16 cm above anywhere the model could reach (measured: 17.8 cm residual
    #: on the head, 9.4 cm on the chest, against 2 to 3 cm on every leg marker), so the
    #: unreachable head marker fought the rest of the solve and distorted the spine.
    segment: str = "lower"  # "lower" (root, legs), "upper" (spine, head), "arm"


#: Correspondences for the 100STYLE skeleton onto the MimicKit humanoid.
#:
#: Verified against the real files rather than guessed: 100STYLE names its joints
#: Hips / Chest..Chest4 / Neck / Head / {Left,Right}{Collar,Shoulder,Elbow,Wrist} /
#: {Left,Right}{Hip,Knee,Ankle,Toe}. An earlier guess at generic names (LeftFoot, LeftLeg)
#: matched nothing, which is why `scripts/retarget.py --inspect` exists.
#:
#: Feet and pelvis carry the most weight: foot placement is what makes a gait read as
#: correct, and the pelvis anchors the body. Hands matter for arm swing but must not
#: outvote the legs, so they are weighted low.
MARKERS_100STYLE: list[MarkerPair] = [
    MarkerPair("Hips", "pelvis", 3.0, segment="lower"),
    MarkerPair("LeftAnkle", "left_foot", 3.0, segment="lower"),
    MarkerPair("RightAnkle", "right_foot", 3.0, segment="lower"),
    MarkerPair("LeftKnee", "left_shin", 1.5, segment="lower"),
    MarkerPair("RightKnee", "right_shin", 1.5, segment="lower"),
    MarkerPair("LeftHip", "left_thigh", 1.0, segment="lower"),
    MarkerPair("RightHip", "right_thigh", 1.0, segment="lower"),
    MarkerPair("Chest3", "torso", 1.5, segment="upper"),
    MarkerPair("Head", "head", 1.0, segment="upper"),
    MarkerPair("LeftElbow", "left_lower_arm", 0.5, segment="arm"),
    MarkerPair("RightElbow", "right_lower_arm", 0.5, segment="arm"),
    MarkerPair("LeftWrist", "left_hand", 0.6, segment="arm"),
    MarkerPair("RightWrist", "right_hand", 0.6, segment="arm"),
]

DEFAULT_MARKERS = MARKERS_100STYLE


def axis_transform(up: str = "y", forward: str = "-z") -> np.ndarray:
    """Rotation mapping a source coordinate convention onto MuJoCo's (X forward, Z up).

    Mocap formats disagree on which axis points up and which way the subject walks. BVH is
    almost always Y-up; 100STYLE subjects travel along -Z. MuJoCo is Z-up and, by our
    convention, X forward.

    The returned matrix is checked to be a proper rotation (determinant +1). That check is
    not pedantry: a transform with determinant -1 is a mirror, and a mirrored retarget
    swaps left and right limbs. The motion still looks plausible frame by frame, so the bug
    survives visual inspection and only shows up as a policy with a permanently asymmetric
    gait, which is close to undebuggable after the fact.
    """
    basis = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}

    def vec(spec: str) -> np.ndarray:
        sign = -1.0 if spec.startswith("-") else 1.0
        return sign * basis[spec.lstrip("+-").lower()]

    up_v, fwd_v = vec(up), vec(forward)
    left_v = np.cross(up_v, fwd_v)  # completes a right-handed set with X forward, Z up
    matrix = np.stack([fwd_v, left_v, up_v])  # rows map source -> (x, y, z) in MuJoCo

    det = float(np.linalg.det(matrix))
    if not np.isclose(abs(det), 1.0) or det < 0:
        raise ValueError(
            f"axis_transform({up=}, {forward=}) gives determinant {det:.3f}. A negative "
            "determinant mirrors the skeleton and silently swaps left and right limbs."
        )
    return matrix


@dataclass
class RetargetConfig:
    iterations: int = 24
    #: Damping for the least-squares solve. Too small oscillates near singularities, too
    #: large converges slowly. 0.05 is stable across the clips tested.
    damping: float = 0.05
    step_size: float = 0.8
    #: Stop early once the weighted RMS marker error falls below this, in metres.
    tolerance: float = 0.008
    #: Lift or drop the whole clip so the lowest foot just touches the ground. Retargeting
    #: a scaled skeleton almost always leaves the feet slightly floating or sunk, and a
    #: clip whose feet never reach the floor teaches a discriminator that hovering is
    #: normal.
    ground_align: bool = True
    #: Resample the clip to this rate. Should match the control rate so reference frames
    #: line up with policy steps without interpolation at training time.
    target_fps: float = 50.0
    #: Source coordinate convention. 100STYLE is Y-up with the subject travelling along -Z.
    source_up: str = "y"
    source_forward: str = "-z"
    #: Remove foot skating by correcting the root trajectory. Measured on a 100STYLE walk
    #: before this existed: planted feet slid at 0.134 m/s while the body travelled at
    #: 0.588 m/s, so 23% of the body's displacement came from the feet sliding.
    #:
    #: That is invisible in a kinematic playback (it looks like walking) but it is not
    #: physically reproducible: a planted foot grips the ground. Simulating the reference
    #: joint angles covered only 63% of the reference distance, and raising the PD gains
    #: eightfold changed that to 65% while octupling peak torque. The reference itself was
    #: the problem, not the controller.
    fix_foot_skating: bool = True
    #: A foot within this height of its own lowest point counts as planted.
    contact_margin: float = 0.015

    #: Savitzky-Golay window applied to the solved trajectory before velocities are
    #: computed. 0 disables it.
    #:
    #: Necessary because each frame's IK is solved independently, so consecutive frames
    #: settle into slightly different local optima. The positions look fine, but
    #: differentiating that roughness produced joint velocities up to 64 rad/s, which no
    #: human joint reaches. Measured on a walk clip: 18% of frames had a velocity
    #: sum-of-squares above 50 against a median of 13, and the second difference of joint
    #: angles peaked at 63 degrees per frame squared.
    #:
    #: A window of 9 cuts peak velocity by 4.6x while moving the pose by only 0.145
    #: degrees. That matters beyond the tracking reward: the Phase 3 discriminator sees
    #: velocities, and artifacts like these are exactly what it would learn to detect
    #: instead of gait style.
    smooth_window: int = 9
    smooth_polyorder: int = 2


@dataclass
class MotionClip:
    """A retargeted clip: model-space joint trajectories at a fixed rate."""

    qpos: np.ndarray  # (n_frames, nq)
    qvel: np.ndarray  # (n_frames, nv)
    fps: float
    name: str = ""
    #: Per-frame weighted RMS marker error in metres. The honest measure of retarget
    #: quality, and the thing to check before trusting a clip.
    residual: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            qpos=self.qpos,
            qvel=self.qvel,
            fps=self.fps,
            name=self.name,
            residual=self.residual,
        )

    @staticmethod
    def load(path) -> "MotionClip":
        d = np.load(path, allow_pickle=False)
        return MotionClip(
            qpos=d["qpos"],
            qvel=d["qvel"],
            fps=float(d["fps"]),
            name=str(d["name"]) if "name" in d else "",
            residual=d["residual"] if "residual" in d else np.zeros(0),
        )


class Retargeter:
    """Solves mocap frames onto a MuJoCo model by marker IK."""

    def __init__(
        self,
        model: mujoco.MjModel,
        default_qpos: np.ndarray,
        markers: list[MarkerPair] | None = None,
        config: RetargetConfig | None = None,
    ) -> None:
        self.model = model
        self.default_qpos = default_qpos.copy()
        self.cfg = config or RetargetConfig()
        self.markers = markers or DEFAULT_MARKERS
        self.data = mujoco.MjData(model)

        self.body_ids: list[int] = []
        self.weights: list[float] = []
        self.resolved: list[MarkerPair] = []
        for pair in self.markers:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pair.body)
            if bid < 0:
                raise ValueError(f"model has no body named {pair.body!r}")
            self.body_ids.append(bid)
            self.weights.append(pair.weight)
            self.resolved.append(pair)

        self.n_markers = len(self.body_ids)
        self._jac_pos = np.zeros((3, model.nv))
        self._jac_rot = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ scaling

    def _model_span(self, body_a: str, body_b: str) -> float:
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.default_qpos
        mujoco.mj_forward(self.model, data)
        ia = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_a)
        ib = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_b)
        return float(np.linalg.norm(data.xpos[ia] - data.xpos[ib]))

    def estimate_scales(
        self, motion: BvhMotion, positions: np.ndarray
    ) -> dict[str, float]:
        """Separate scales for the lower and upper body.

        A single uniform scale cannot fit both, because humans and humanoid models have
        different torso-to-leg ratios (0.71 for this dataset's subject against 0.55 for our
        model). Scaling by legs alone leaves the head unreachable; scaling by height alone
        wrecks stride length and ground clearance, which are what make a gait read as
        correct.

        Lower-body scale comes from hips-to-ankle and also sets the root trajectory, so
        stride length and body height stay physically consistent. Upper-body scale comes
        from hips-to-head and applies only to markers above the pelvis.

        Medians over frames, to be robust to a few badly tracked frames.
        """

        def source_span(a: str, b: str) -> float:
            try:
                ia, ib = motion.index_of(a), motion.index_of(b)
            except KeyError:
                return 0.0
            return float(np.median(np.linalg.norm(positions[:, ia] - positions[:, ib], axis=1)))

        lower_src = source_span("Hips", "LeftAnkle")
        upper_src = source_span("Hips", "Head")
        arm_src = source_span("LeftShoulder", "LeftWrist")
        lower = self._model_span("pelvis", "left_foot") / lower_src if lower_src > 1e-6 else 1.0
        upper = self._model_span("pelvis", "head") / upper_src if upper_src > 1e-6 else lower
        # Arms hang off the torso but have their own length ratio. Scaling them by the
        # torso ratio left the elbows 8 cm out, the worst marker in the solve.
        arm = self._model_span("left_upper_arm", "left_hand") / arm_src if arm_src > 1e-6 else upper
        return {"lower": lower, "upper": upper, "arm": arm}

    # Kept for backwards compatibility with anything reading a single scale.
    def model_limb_span(self) -> float:
        return self._model_span("pelvis", "left_foot")

    # ------------------------------------------------------------------ IK

    def _solve_frame(self, targets: np.ndarray, qpos: np.ndarray) -> tuple[np.ndarray, float]:
        """Damped least-squares IK for one frame, warm-started from `qpos`."""
        model, data = self.model, self.data
        data.qpos[:] = qpos
        weights = np.repeat(np.asarray(self.weights), 3)

        error = np.zeros(3 * self.n_markers)
        jacobian = np.zeros((3 * self.n_markers, model.nv))
        rms = float("inf")

        for _ in range(self.cfg.iterations):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            for k, bid in enumerate(self.body_ids):
                error[3 * k : 3 * k + 3] = targets[k] - data.xpos[bid]
                mujoco.mj_jacBody(model, data, self._jac_pos, self._jac_rot, bid)
                jacobian[3 * k : 3 * k + 3, :] = self._jac_pos

            weighted_error = error * weights
            rms = float(np.sqrt(np.mean(weighted_error**2)))
            if rms < self.cfg.tolerance:
                break

            jac_w = jacobian * weights[:, None]
            # dq = J^T (J J^T + lambda I)^-1 e. Solving in marker space rather than joint
            # space because there are far fewer markers than degrees of freedom, so this
            # is the smaller and better-conditioned system.
            lhs = jac_w @ jac_w.T + (self.cfg.damping**2) * np.eye(3 * self.n_markers)
            dq = jac_w.T @ np.linalg.solve(lhs, weighted_error)

            mujoco.mj_integratePos(model, data.qpos, dq, self.cfg.step_size)
            self._clamp_joints(data.qpos)

        return data.qpos.copy(), rms

    def _clamp_joints(self, qpos: np.ndarray) -> None:
        """Hold every hinge inside its anatomical range.

        The IK has no notion of joint limits, so without this it will happily fold a knee
        backwards to reach a marker, producing a clip that is unreachable by the physical
        model and would poison an imitation reward.
        """
        model = self.model
        for j in range(model.njnt):
            if not model.jnt_limited[j] or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            adr = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            qpos[adr] = min(max(qpos[adr], lo), hi)

    # ------------------------------------------------------------------ driver

    def retarget(self, motion: BvhMotion, name: str = "") -> MotionClip:
        """Retarget a whole BVH clip, resampled to the configured rate."""
        cfg = self.cfg

        # Resample source frames onto the target rate before solving, so we solve exactly
        # the frames we keep rather than solving 120 Hz and throwing most of it away.
        n_out = max(2, int(round(motion.duration * cfg.target_fps)))
        src_idx = np.clip(
            np.round(np.linspace(0, motion.n_frames - 1, n_out)).astype(int),
            0,
            motion.n_frames - 1,
        )
        positions = motion.forward_kinematics(src_idx)

        # Map BVH joints to marker slots once, tolerating missing ones.
        source_idx: list[int] = []
        keep: list[int] = []
        for k, pair in enumerate(self.resolved):
            try:
                source_idx.append(motion.index_of(pair.source))
                keep.append(k)
            except KeyError:
                continue
        if len(keep) < 4:
            raise ValueError(
                f"only {len(keep)} of {self.n_markers} markers matched the source skeleton "
                f"{motion.names[:12]}...; the marker map needs adjusting for this dataset"
            )
        # Restrict to matched markers for this clip.
        self.body_ids = [self.body_ids[k] for k in keep]
        self.weights = [self.weights[k] for k in keep]
        self.n_markers = len(keep)

        scales = self.estimate_scales(motion, positions)
        lower_scale = scales["lower"]
        # Convert the source convention (Y-up, travel along -Z) into MuJoCo's (Z-up,
        # X forward) with a verified proper rotation.
        rotation = axis_transform(cfg.source_up, cfg.source_forward)
        world = positions[:, source_idx, :] @ rotation.T

        # Per-segment scaling, applied about the hips. The root trajectory follows the
        # lower-body scale so stride length and body height stay physically consistent,
        # while upper-body markers are scaled about the pelvis by the torso ratio.
        hips_slot = next(
            (i for i, k in enumerate(keep) if self.resolved[k].segment == "lower"
             and self.resolved[k].body == "pelvis"),
            0,
        )
        root = world[:, hips_slot : hips_slot + 1, :]
        relative = world - root
        seg_scale = np.array(
            [scales.get(self.resolved[k].segment, lower_scale) for k in keep]
        )[None, :, None]
        targets_all = root * lower_scale + relative * seg_scale

        qpos_out = np.zeros((n_out, self.model.nq))
        residual = np.zeros(n_out)
        qpos = self.default_qpos.copy()

        for f in range(n_out):
            targets = targets_all[f]
            # Seed the root translation from the target pelvis so the solver starts close.
            qpos[0:3] = targets[0] if self.resolved[keep[0]].body == "pelvis" else qpos[0:3]
            qpos, rms = self._solve_frame(targets, qpos)
            qpos_out[f] = qpos
            residual[f] = rms

        # Order matters. Ground alignment must come FIRST: de-skating detects stance from
        # real foot-floor contacts, and a clip that has not been dropped onto the ground has
        # no contacts to find. Getting this backwards made de-skating silently do nothing.
        if cfg.ground_align:
            qpos_out[:, 2] += self._ground_offset(qpos_out)

        if cfg.fix_foot_skating:
            qpos_out = self._deskate(qpos_out, cfg.contact_margin)

        if cfg.smooth_window and qpos_out.shape[0] > cfg.smooth_window:
            qpos_out = self._smooth(qpos_out, cfg.smooth_window, cfg.smooth_polyorder)
            # Smoothing nudges height slightly, so re-seat the clip on the ground.
            if cfg.ground_align:
                qpos_out[:, 2] += self._ground_offset(qpos_out)

        qvel_out = self._finite_difference(qpos_out, cfg.target_fps)
        return MotionClip(
            qpos=qpos_out, qvel=qvel_out, fps=cfg.target_fps, name=name, residual=residual
        )

    def _deskate(self, qpos: np.ndarray, margin: float) -> np.ndarray:
        """Shift the root so feet in stance stay put in world space.

        Stance is detected from the **exact lowest point of each foot geom** against an
        absolute ground threshold, which is only meaningful because ground alignment now
        runs first and is itself exact.

        Two earlier attempts failed and are worth recording. A relative height heuristic
        (foot within 3 cm of its own lowest point over the clip) mislabelled stance and left
        18% slip. Real `mj_forward` contacts found *nothing*, because MuJoCo only generates
        a contact on actual penetration and an exactly-aligned clip merely grazes the floor.

        The correction shortens the clip's travel, which is correct: the original distance
        was partly fictional, produced by feet sliding along the ground.
        """
        from humanoid_rl.envs.model_prep import _geom_lowest_z

        model = self.model
        sides: dict[int, int] = {}
        for side, name in enumerate(("left_foot", "right_foot")):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                return qpos
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid:
                    sides[g] = side

        data = mujoco.MjData(model)
        n = qpos.shape[0]
        anchor = np.zeros((n, 2, 2))
        planted = np.zeros((n, 2), dtype=bool)

        for f in range(n):
            data.qpos[:] = qpos[f]
            mujoco.mj_forward(model, data)
            lowest = np.full(2, np.inf)
            pos = np.zeros((2, 2))
            for g, side in sides.items():
                z = _geom_lowest_z(model, data, g)
                if z < lowest[side]:
                    lowest[side] = z
                    pos[side] = data.geom_xpos[g][:2]
            for side in (0, 1):
                if lowest[side] < margin:
                    planted[f, side] = True
                    anchor[f, side] = pos[side]

        correction = np.zeros((n, 2))
        running = np.zeros(2)
        for f in range(1, n):
            both = planted[f] & planted[f - 1]
            if both.any():
                running = running - (anchor[f, both] - anchor[f - 1, both]).mean(axis=0)
            correction[f] = running

        if n > 9:
            from scipy.signal import savgol_filter

            correction = savgol_filter(correction, 9, 2, axis=0)

        out = qpos.copy()
        out[:, 0:2] += correction
        return out

    def _foot_geom_ids(self) -> list[int]:
        model = self.model
        ids = []
        for name in ("left_foot", "right_foot"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            ids += [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        return ids

    @staticmethod
    def _smooth(qpos: np.ndarray, window: int, polyorder: int) -> np.ndarray:
        """Low-pass the solved trajectory to remove frame-to-frame IK jitter.

        Root position and joint angles are filtered directly. The root quaternion is
        filtered componentwise and then renormalised, which is valid here only because the
        corrections are tiny; a general quaternion path would need spherical interpolation.
        """
        from scipy.signal import savgol_filter

        window = window if window % 2 == 1 else window + 1
        out = qpos.copy()
        out[:, 0:3] = savgol_filter(qpos[:, 0:3], window, polyorder, axis=0)
        out[:, 7:] = savgol_filter(qpos[:, 7:], window, polyorder, axis=0)

        quat = savgol_filter(qpos[:, 3:7], window, polyorder, axis=0)
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        out[:, 3:7] = quat / np.maximum(norm, 1e-9)
        return out

    def _ground_offset(self, qpos: np.ndarray) -> float:
        """Vertical shift so the lowest point of the body over the clip rests on z = 0.

        Uses the exact per-geom-type lowest point, not `geom_rbound`. That distinction is
        not cosmetic: `geom_rbound` is a *bounding sphere* radius, which for the foot box is
        sqrt(0.0885^2 + 0.045^2 + 0.0275^2) = 10.4 cm against a true flat half-extent of
        2.75 cm. Using it lifted every clip about 7.6 cm, so the feet never touched the
        ground at all: contact-based stance detection found zero contacts, and physics
        simulation began with the humanoid airborne and dropping.

        Also samples every frame. The previous version sampled about 60 of 1499, which can
        miss the true minimum entirely.
        """
        from humanoid_rl.envs.model_prep import _geom_lowest_z

        model = self.model
        data = mujoco.MjData(model)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        lowest = np.inf
        for f in range(qpos.shape[0]):
            data.qpos[:] = qpos[f]
            mujoco.mj_forward(model, data)
            for g in range(model.ngeom):
                if g == floor:
                    continue
                lowest = min(lowest, _geom_lowest_z(model, data, g))
        return -lowest if np.isfinite(lowest) else 0.0

    def _finite_difference(self, qpos: np.ndarray, fps: float) -> np.ndarray:
        """Velocities from positions, using MuJoCo's own quaternion-aware differencing.

        A naive `np.diff` on qpos is wrong for the free root: subtracting quaternions
        component-wise is not an angular velocity. `mj_differentiatePos` handles the
        manifold correctly, which matters because the discriminator in Phase 3 sees
        velocities and would otherwise learn to detect our differencing bug.
        """
        model = self.model
        dt = 1.0 / fps
        qvel = np.zeros((qpos.shape[0], model.nv))
        buf = np.zeros(model.nv)
        for f in range(1, qpos.shape[0]):
            mujoco.mj_differentiatePos(model, buf, dt, qpos[f - 1], qpos[f])
            qvel[f] = buf
        if qpos.shape[0] > 1:
            qvel[0] = qvel[1]
        return qvel
