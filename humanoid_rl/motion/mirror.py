"""Reflect a retargeted clip across the sagittal plane, to make the reference symmetric.

Why this exists, concretely. AMP reproduces whatever distribution it is shown. The mocap
here is a handful of clips from one performer, and a real person's walk is never perfectly
even: measured on this library, the left and right stance times differ by a factor of 1.32.
The discriminator learns that asymmetry as part of "looking human", so the policy is
rewarded for reproducing it, and it duly did, at a left/right stance ratio of 1.90.

The first attempt at a fix added a symmetry loss to PPO, penalising the policy when its
action on a mirrored state was not the mirror of its action. That is satisfiable by a policy
that ignores its input and holds a symmetric pose, and at coefficient 2.0 the policy found
exactly that: a 61 cm wide two-footed brace, 90% double support, 6 cm of travel per foot
strike, scoring 0.91 on the old symmetry metric while not walking at all.

Mirroring the reference instead fixes the cause. If every clip appears alongside its
reflection, the reference distribution is exactly symmetric by construction, so a
left-favouring gait is no longer any more "human" than a right-favouring one, and the
discriminator stops rewarding it. Nothing is added to the policy objective, so there is no
degenerate solution to find: a brace still looks nothing like walking and still scores badly
on style.

Conventions, all following MuJoCo (x forward, y left, z up), reflecting across the plane
y = 0:

* **root position** (x, y, z) becomes (x, -y, z).
* **root orientation** (w, x, y, z) becomes (w, -x, y, -z). Reflection conjugates the
  rotation, R -> M R M with M = diag(1, -1, 1). Pitch (about y) is unchanged; roll and yaw
  reverse, which is what those sign flips encode.
* **root linear velocity** is a polar vector: (vx, vy, vz) becomes (vx, -vy, vz).
* **root angular velocity** is an axial vector, so it picks up an extra sign:
  (wx, wy, wz) becomes (-wx, wy, -wz). This holds whether it is expressed in the world frame
  or the body frame, because the body frame is reflected too and the two sign changes
  cancel.
* **joint angles and velocities** are permuted left-to-right and sign-flipped by axis, using
  the permutation `envs.mirror` derives from the model itself.
"""

from __future__ import annotations

import mujoco
import numpy as np

from humanoid_rl.envs.mirror import MirrorSpec
from humanoid_rl.motion.retarget import MotionClip


def _joint_slices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel addresses of the hinge joints, in the order `envs.mirror` uses.

    Read from the model rather than assumed to be contiguous after the free joint, because
    an assumption that happens to hold today is exactly the kind of thing that silently
    mirrors the wrong joints if the model gains a slide joint later.
    """
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    return (
        np.array([model.jnt_qposadr[j] for j in hinges], dtype=np.int64),
        np.array([model.jnt_dofadr[j] for j in hinges], dtype=np.int64),
    )


def mirror_qpos_qvel(
    qpos: np.ndarray, qvel: np.ndarray, model: mujoco.MjModel, spec: MirrorSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Reflect a (T, nq) / (T, nv) trajectory across the sagittal plane."""
    q = qpos.copy()
    v = qvel.copy()
    qadr, vadr = _joint_slices(model)

    # Root translation: negate the lateral component.
    q[:, 1] *= -1.0
    # Root orientation: conjugate the rotation by the reflection.
    q[:, 4] *= -1.0  # quaternion x
    q[:, 6] *= -1.0  # quaternion z
    # Root velocity: polar then axial.
    v[:, 1] *= -1.0
    v[:, 3] *= -1.0
    v[:, 5] *= -1.0

    # Joints: swap left for right, then flip the sign of roll and yaw axes.
    q[:, qadr] = q[:, qadr][:, spec.qpos_perm] * spec.qpos_sign
    v[:, vadr] = v[:, vadr][:, spec.qpos_perm] * spec.qpos_sign
    return q, v


def mirror_clip(clip: MotionClip, model: mujoco.MjModel, spec: MirrorSpec) -> MotionClip:
    """A left-right reflected copy of a clip, named with a `_mirror` suffix."""
    qpos, qvel = mirror_qpos_qvel(clip.qpos, clip.qvel, model, spec)
    return MotionClip(
        qpos=qpos,
        qvel=qvel,
        fps=clip.fps,
        name=f"{clip.name}_mirror" if clip.name else "mirror",
        # Retarget residual is a property of the fit, unchanged by reflecting it.
        residual=clip.residual.copy(),
    )


def asymmetry(qpos: np.ndarray, model: mujoco.MjModel) -> float:
    """Left/right stance-time ratio of a trajectory, as a single number.

    1.0 is perfectly even. Used to report what mirroring actually bought, rather than
    asserting that it must have worked.
    """
    data = mujoco.MjData(model)
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    heights = np.zeros((len(qpos), 2))
    for f, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        heights[f] = (data.xpos[left, 2], data.xpos[right, 2])
    # A foot within 3 cm of its own lowest point over the clip counts as loaded.
    loaded = heights < heights.min(axis=0, keepdims=True) + 0.03
    lo, hi = np.sort(loaded.mean(axis=0))
    return float(hi / max(lo, 1e-6))
