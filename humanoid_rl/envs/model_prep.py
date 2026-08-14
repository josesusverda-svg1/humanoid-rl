"""Prepare a humanoid MJCF for reinforcement learning.

Two things happen here, and both are done at load time in code rather than by editing the
XML. That keeps `amp_humanoid_mimickit.xml` byte-identical to upstream, which matters
because the Phase 3 mocap retargeting tooling is written against that exact file.

1. **Torque motors become PD position servos.**

   The upstream model uses direct torque actuators. Learning locomotion from raw torques is
   substantially harder than learning target joint *angles*, because the policy has to
   discover the inverse dynamics of its own body before it can even stand. Every strong
   humanoid locomotion result (DeepMimic, AMP, legged-gym) uses PD position control, so
   Phase 3 needs it anyway.

   **PD controller** (proportional-derivative): given a target angle, it applies a torque
   proportional to the angle error minus a term proportional to the joint velocity. The
   first pulls the joint toward the target, the second damps oscillation.

   Crucially this is configured *inside MuJoCo*, using position actuators, rather than
   computed in Python. MuJoCo then runs the PD at the full 200 Hz physics rate while the
   policy sets targets at 50 Hz. Computing it in Python would force stepping physics one
   step at a time, which Phase 0 measured as 2.6x slower (see DESIGN.md section 2.2).

   MuJoCo actuator force is `gear * (gain * ctrl + bias)`. For a position servo,
   `gain = kp` and `bias = -kp*q - kv*qd`, giving `force = kp*(ctrl - q) - kv*qd` once gear
   is set to 1. Setting gear to 1 makes kp and kv directly interpretable in N*m/rad.

2. **A natural standing pose is defined and its height measured.**

   The upstream model's zero pose is a T-pose floating at z=0. Policy actions are expressed
   as offsets from a nominal standing pose, so that nominal pose has to exist and the root
   height at which the feet rest on the ground has to be measured, not guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# Optional per-joint-group override of the PD gains, in N*m/rad and N*m*s/rad.
#
# Normally left empty. By default the gains are read straight out of the model, for a
# reason worth recording: this MJCF specifies `stiffness` and `damping` on every joint
# (abdomen 1000, hip 500, knee 500, shoulder 400, elbow 300, ankle 400, neck 100, damping
# always exactly kp/10). Those are not passive material properties, they are DeepMimic-style
# PD gains. Isaac Gym, which the upstream model targets, uses a DOF's stiffness and damping
# directly as its position-drive gains.
#
# MuJoCo interprets the same fields differently: as a passive spring pulling the joint
# toward `springref` (zero here, the T-pose) plus a damper. Left in place they fight the
# position servo. The first attempt at this file kept them and the arms would not leave the
# T-pose: a constant 300 N*m actuator torque balanced exactly against the spring, holding
# the shoulder at 14 degrees instead of the commanded 83.
#
# So the fix is to move those numbers from the passive springs into the actuators, where
# they mean what their author intended, and zero the springs.
GAIN_OVERRIDES: dict[str, tuple[float, float]] = {}

#: Multiplier applied to the gains read from the model. Above 1.0 tracks targets more
#: tightly at the cost of stiffer, less compliant contact.
GAIN_SCALE = 1.0

#: Body whose own up-axis defines "torso upright", and the body used for head height.
#: Both guard against the fold-forward-at-the-waist failure described in
#: build_model_with_foot_sensors.
UPRIGHT_BODY = "torso"
HEAD_BODY = "head"

#: Bodies whose world positions are published every step. These are the extremities of the
#: kinematic chain, so matching them constrains everything upstream. Used by the tracking
#: reward now and by the AMP discriminator observation in Phase 3.
KEY_BODIES = ("left_foot", "right_foot", "left_hand", "right_hand")

# Nominal standing pose, in radians, for joints that are not zero.
# Verified visually: shoulder_x with opposite signs brings the arms down to the sides from
# the T-pose. Slight elbow and knee flexion gives a relaxed, human stance rather than a
# locked-out mannequin, and a softly bent knee is also much easier to initiate gait from.
DEFAULT_POSE: dict[str, float] = {
    "right_shoulder_x": np.radians(83.0),
    "left_shoulder_x": np.radians(-83.0),
    "right_elbow": np.radians(15.0),
    "left_elbow": np.radians(-15.0),
    "right_knee": np.radians(12.0),
    "left_knee": np.radians(12.0),
    "right_hip_y": np.radians(-6.0),
    "left_hip_y": np.radians(-6.0),
    "right_ankle_y": np.radians(-6.0),
    "left_ankle_y": np.radians(-6.0),
}


@dataclass
class PreparedModel:
    """A loaded model plus everything the environment needs to drive it."""

    model: mujoco.MjModel
    #: Full nq initial state, root included, with the feet resting on the ground.
    default_qpos: np.ndarray
    #: Nominal joint angles only, length nu, in actuator order.
    default_joint_pos: np.ndarray
    #: Per-joint scale mapping a policy action in [-1, 1] to an angle offset, length nu.
    action_scale: np.ndarray
    #: Root height at which the humanoid stands with its feet on the ground.
    standing_height: float
    #: Body ids of the feet, used for contact-based reward terms and fall detection.
    foot_body_ids: np.ndarray
    #: Geom ids of the feet.
    foot_geom_ids: np.ndarray
    #: Geom id of the ground plane.
    floor_geom_id: int
    #: Body ids that should never touch the ground. Contact here means a fall.
    non_foot_body_ids: np.ndarray
    #: Indices into `data.sensordata` giving the normal force under each foot, in newtons.
    foot_touch_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    #: Start index into `data.sensordata` of each foot's world-frame linear velocity (3).
    foot_linvel_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    #: Index into `data.sensordata` of the torso up-axis (3), or -1 if the model has none.
    torso_zaxis_adr: int = -1
    #: Index into `data.sensordata` of the head position (3), or -1 if absent.
    head_pos_adr: int = -1
    #: Start indices into `data.sensordata` of each key body's world position (3 each).
    key_body_adr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    #: Names of the key bodies, in the same order as `key_body_adr`.
    key_body_names: tuple[str, ...] = ()
    #: Head height while standing in the nominal pose, used to normalise the head-drop
    #: termination so it transfers to a differently proportioned humanoid.
    standing_head_height: float = 0.0
    joint_names: list[str] = field(default_factory=list)

    @property
    def nu(self) -> int:
        return self.model.nu


def build_model_with_foot_sensors(
    model_path: str | Path, margin: float = 0.01
) -> tuple[mujoco.MjModel, list[str]]:
    """Compile the scene, adding a touch sensor under each foot.

    Foot contact drives several of the most useful gait reward terms (air time, foot slip)
    and gives a far better fall signal than a root-height threshold. The obvious way to get
    it is to read `data.contact` after each step, but that is per-environment Python inside
    the worker threads, which Phase 0 measured as the one thing that must never happen in
    the hot loop (see DESIGN.md section 2.4).

    A touch sensor instead writes a scalar normal force into `data.sensordata`, a flat
    array the worker copies out at the same near-zero cost as `qpos`. Verified against
    ground truth: standing gives 245.7 N under each foot, summing to 491 N, which is the
    body weight to the newton, and 0 N when airborne.

    `cfrc_ext` was tried first and is not an option: MuJoCo leaves it zeroed unless
    something in the model requires `mj_rnePostConstraint`, which nothing here did.

    Sensors are added through MjSpec rather than by editing the XML, so the upstream body
    file stays byte-identical and the Phase 3 mocap tooling keeps working.
    """
    spec = mujoco.MjSpec.from_file(str(model_path))
    foot_names = sorted(b.name for b in spec.bodies if re.search(r"foot", b.name or ""))
    if not foot_names:
        raise ValueError(f"no bodies matching 'foot' in {model_path}")

    for foot in foot_names:
        body = spec.body(foot)
        geoms = list(body.geoms)
        if not geoms:
            raise ValueError(f"foot body {foot!r} has no geom to wrap with a touch site")
        geom = geoms[0]

        # A touch sensor reports the force on contacts falling inside its site's volume, so
        # the site must slightly enclose the foot geom.
        site = body.add_site()
        site.name = f"{foot}_touch"
        site.type = mujoco.mjtGeom.mjGEOM_BOX
        site.pos = np.asarray(geom.pos, dtype=float)
        site.size = np.asarray(geom.size, dtype=float) + margin
        # Invisible: it exists for sensing, and drawing it would obscure the foot in videos.
        site.rgba = [0.0, 0.0, 0.0, 0.0]

        sensor = spec.add_sensor()
        sensor.name = f"{foot}_touch"
        sensor.type = mujoco.mjtSensor.mjSENS_TOUCH
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = f"{foot}_touch"

        # World-frame linear velocity of the foot, for the foot-slip penalty. Same
        # rationale as the touch sensor: it arrives as a flat array the worker copies for
        # free, instead of per-environment kinematics work in the hot loop.
        vel = spec.add_sensor()
        vel.name = f"{foot}_linvel"
        vel.type = mujoco.mjtSensor.mjSENS_FRAMELINVEL
        vel.objtype = mujoco.mjtObj.mjOBJ_SITE
        vel.objname = f"{foot}_touch"

    # Upper-body sensors. These exist because of a failure found by watching a video: with
    # uprightness and height measured only at the pelvis, the policy learned to fold forward
    # at the abdomen and lurch along with its torso horizontal and its head near the ground.
    # The pelvis stayed level at the right height throughout, so every reward term was
    # satisfied. Measuring the torso's own up-axis and the head's height closes that hole.
    body_names = {b.name for b in spec.bodies}
    for body_name in KEY_BODIES:
        if body_name not in body_names:
            continue
        s = spec.add_sensor()
        s.name = f"{body_name}_pos"
        s.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
        s.objtype = mujoco.mjtObj.mjOBJ_BODY
        s.objname = body_name
    if UPRIGHT_BODY in body_names:
        s = spec.add_sensor()
        s.name = "torso_zaxis"
        s.type = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
        s.objtype = mujoco.mjtObj.mjOBJ_BODY
        s.objname = UPRIGHT_BODY
    if HEAD_BODY in body_names:
        s = spec.add_sensor()
        s.name = "head_pos"
        s.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
        s.objtype = mujoco.mjtObj.mjOBJ_BODY
        s.objname = HEAD_BODY

    return spec.compile(), foot_names


def _gains_for(
    name: str, overrides: dict[str, tuple[float, float]], from_model: tuple[float, float]
) -> tuple[float, float]:
    for pattern, kp_kd in overrides.items():
        if re.search(pattern, name):
            return kp_kd
    return from_model


def to_position_control(
    model: mujoco.MjModel,
    overrides: dict[str, tuple[float, float]] | None = None,
    gain_scale: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Rewrite every actuator in place as a PD position servo.

    Gains are taken from each joint's own `stiffness` and `damping` (see the note on
    GAIN_OVERRIDES for why), then those passive springs and dampers are zeroed so the
    actuator is the only thing driving the joint.

    After this call, `data.ctrl[i]` is a *target joint angle in radians*, not a torque.

    Returns the gains actually applied, keyed by joint name, for logging.
    """
    overrides = overrides or GAIN_OVERRIDES
    # Resolved at CALL time, not bound as a default argument. A default of `GAIN_SCALE`
    # captures the value at function-definition time, so reassigning the module global
    # silently does nothing, which is exactly what a gain sweep hit.
    gain_scale = GAIN_SCALE if gain_scale is None else gain_scale
    applied: dict[str, tuple[float, float]] = {}

    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint{i}"
        dof = model.jnt_dofadr[joint_id]

        model_kp = float(model.jnt_stiffness[joint_id])
        model_kv = float(model.dof_damping[dof])
        if model_kp <= 0.0:
            raise ValueError(
                f"joint {name!r} has no stiffness to derive PD gains from. "
                "Add an entry to GAIN_OVERRIDES for it."
            )
        kp, kv = _gains_for(name, overrides, (model_kp, model_kv))
        kp *= gain_scale
        kv *= gain_scale
        applied[name] = (kp, kv)

        # Position servo: force = gear * (gain*ctrl + bias), with gain = kp and
        # bias = -kp*q - kv*qd, so force = kp*(target - q) - kv*qd.
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[i] = 0.0
        model.actuator_biasprm[i] = 0.0
        model.actuator_gainprm[i, 0] = kp
        model.actuator_biasprm[i, 1] = -kp
        model.actuator_biasprm[i, 2] = -kv
        # gear 1 keeps kp and kv in real physical units (N*m/rad, N*m*s/rad).
        model.actuator_gear[i, 0] = 1.0

        # Hand the passive spring and damper over to the actuator.
        model.jnt_stiffness[joint_id] = 0.0
        model.dof_damping[dof] = 0.0

        # Clamp targets to the joint's own limits, so a diverged policy cannot command a
        # target outside the anatomy and fight the joint limit constraint forever.
        if model.jnt_limited[joint_id]:
            model.actuator_ctrlrange[i] = model.jnt_range[joint_id]
            model.actuator_ctrllimited[i] = 1

    # Damping now lives in the actuator bias. The implicitfast integrator treats actuator
    # damping implicitly, which stays stable at much higher gains than explicit Euler and
    # is also slightly faster.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return applied


def _body_ids_matching(model: mujoco.MjModel, pattern: str) -> np.ndarray:
    out = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if re.search(pattern, name):
            out.append(b)
    return np.array(out, dtype=np.int32)


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """World z of the lowest point of one geom, exactly for boxes and tightly for others."""
    pos_z = float(data.geom_xpos[geom_id, 2])
    size = model.geom_size[geom_id]
    gtype = model.geom_type[geom_id]

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        # The lowest corner of an oriented box: project its half-extents onto world -z.
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        drop = abs(rot[2, 0]) * size[0] + abs(rot[2, 1]) * size[1] + abs(rot[2, 2]) * size[2]
        return pos_z - drop
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos_z - float(size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # Half-length along the geom's local z, plus the cap radius.
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        return pos_z - (abs(rot[2, 2]) * float(size[1]) + float(size[0]))
    return pos_z - float(model.geom_rbound[geom_id])


def _compute_standing_height(model: mujoco.MjModel, default_qpos: np.ndarray) -> float:
    """Root height at which the lowest point of the body rests exactly on z = 0.

    Computed geometrically rather than by dropping the humanoid and letting it settle.
    Dropping is unreliable here for a reason worth recording: a humanoid holding a *fixed*
    joint configuration cannot balance. It is an inverted pendulum, so any settling test
    long enough to remove the landing impact is also long enough for it to topple. Keeping
    balance is the policy's job, not the PD controller's.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = default_qpos
    probe_height = 1.0
    data.qpos[2] = probe_height
    mujoco.mj_forward(model, data)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    lowest = min(
        _geom_lowest_z(model, data, g) for g in range(model.ngeom) if g != floor_id
    )
    return probe_height - lowest


def _verify_pd_holds_pose(
    model: mujoco.MjModel, default_qpos: np.ndarray, seconds: float = 0.4
) -> tuple[float, float]:
    """Check the PD is stiff enough to hold the nominal pose against gravity.

    This deliberately does *not* test balance. It starts the humanoid already standing and
    measures, over a short window before tipping can dominate, how far the joints sag from
    their targets and how far the root sinks. Large values here mean the gains are too soft.

    Returns (max joint tracking error in radians, root sag in metres).
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = default_qpos
    mujoco.mj_forward(model, data)

    targets = np.empty(model.nu)
    qadr = np.empty(model.nu, dtype=int)
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        qadr[i] = model.jnt_qposadr[joint_id]
        targets[i] = default_qpos[qadr[i]]
    data.ctrl[:] = targets

    start_z = float(data.qpos[2])
    mujoco.mj_step(model, data, nstep=int(seconds / model.opt.timestep))

    if not np.isfinite(data.qpos).all():
        raise RuntimeError("simulation diverged while holding the nominal pose")
    error = float(np.abs(data.qpos[qadr] - targets).max())
    sag = start_z - float(data.qpos[2])
    return error, sag


def prepare(
    model_path: str | Path,
    *,
    gains: dict[str, tuple[float, float]] | None = None,
    pose: dict[str, float] | None = None,
    action_scale_fraction: float = 0.6,
    gain_scale: float | None = None,
) -> PreparedModel:
    """Load an MJCF and return it configured for position-controlled RL.

    Args:
        model_path: Path to the scene XML (body plus ground).
        gains: Regex-to-(kp, kv) map. Defaults to DEFAULT_GAINS.
        pose: Joint-name-to-angle nominal pose. Defaults to DEFAULT_POSE.
        action_scale_fraction: Fraction of each joint's half-range that a full-scale
            action (+-1) may command as an offset from the nominal pose. Scaling per joint
            rather than using one global constant matters because ranges differ by 4x
            across this body: 0.5 rad is a gentle nudge for a shoulder and the entire
            travel of an ankle.
    """
    model, foot_names = build_model_with_foot_sensors(model_path)
    to_position_control(model, gains, gain_scale)

    pose = DEFAULT_POSE if pose is None else pose
    joint_names: list[str] = []
    default_qpos = np.zeros(model.nq)
    default_qpos[3] = 1.0  # identity quaternion (w, x, y, z)

    default_joint_pos = np.zeros(model.nu)
    action_scale = np.zeros(model.nu)

    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint{i}"
        joint_names.append(name)

        angle = float(pose.get(name, 0.0))
        lo, hi = model.jnt_range[joint_id]
        if model.jnt_limited[joint_id]:
            angle = float(np.clip(angle, lo, hi))
            action_scale[i] = action_scale_fraction * (hi - lo) * 0.5
        else:
            action_scale[i] = action_scale_fraction

        default_qpos[model.jnt_qposadr[joint_id]] = angle
        default_joint_pos[i] = angle

    standing_height = _compute_standing_height(model, default_qpos)
    # Start a hair above contact so episodes do not begin inside the floor, which would
    # produce a large spurious contact impulse on the very first step.
    default_qpos[2] = standing_height + 0.002

    tracking_error, sag = _verify_pd_holds_pose(model, default_qpos)
    if tracking_error > 0.09 or sag > 0.03:
        raise RuntimeError(
            f"PD gains are too soft: joints sag {np.degrees(tracking_error):.1f} deg and the "
            f"root sinks {sag * 100:.1f} cm while merely holding the nominal pose. "
            "Raise GAIN_SCALE or add an entry to GAIN_OVERRIDES."
        )

    foot_bodies = _body_ids_matching(model, r"foot")
    if foot_bodies.size == 0:
        raise ValueError("no bodies matching 'foot' found; contact rewards need them")
    foot_geoms = np.array(
        [g for g in range(model.ngeom) if model.geom_bodyid[g] in set(foot_bodies.tolist())],
        dtype=np.int32,
    )
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Every body except the feet and the world. Ground contact on any of these is a fall,
    # which is a far more reliable termination signal than a root height threshold alone
    # (a humanoid can drop its root low in a legitimate deep stride).
    foot_set = set(foot_bodies.tolist())
    non_foot = np.array(
        [b for b in range(1, model.nbody) if b not in foot_set], dtype=np.int32
    )

    def _sensor_adr(suffix: str) -> np.ndarray:
        return np.array(
            [
                model.sensor_adr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{f}_{suffix}")
                ]
                for f in foot_names
            ],
            dtype=np.int32,
        )

    touch_adr = _sensor_adr("touch")
    linvel_adr = _sensor_adr("linvel")

    def _named_adr(name: str) -> int:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(model.sensor_adr[sid]) if sid >= 0 else -1

    torso_adr = _named_adr("torso_zaxis")
    head_adr = _named_adr("head_pos")
    key_names = tuple(b for b in KEY_BODIES if _named_adr(f"{b}_pos") >= 0)
    key_adr = np.array([_named_adr(f"{b}_pos") for b in key_names], dtype=np.int32)

    standing_head = 0.0
    if head_adr >= 0:
        probe = mujoco.MjData(model)
        mujoco.mj_resetData(model, probe)
        probe.qpos[:] = default_qpos
        mujoco.mj_forward(model, probe)
        standing_head = float(probe.sensordata[head_adr + 2])

    return PreparedModel(
        model=model,
        default_qpos=default_qpos,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        standing_height=standing_height,
        foot_body_ids=foot_bodies,
        foot_geom_ids=foot_geoms,
        floor_geom_id=int(floor_id),
        non_foot_body_ids=non_foot,
        foot_touch_adr=touch_adr,
        foot_linvel_adr=linvel_adr,
        torso_zaxis_adr=torso_adr,
        head_pos_adr=head_adr,
        key_body_adr=key_adr,
        key_body_names=key_names,
        standing_head_height=standing_head,
        joint_names=joint_names,
    )
