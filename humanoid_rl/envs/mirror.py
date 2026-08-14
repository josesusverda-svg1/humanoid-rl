"""Left-right mirroring, for enforcing gait symmetry.

A humanoid is bilaterally symmetric, so walking straight ahead should look the same in a
mirror. Nothing in a standard RL setup says so, and policies routinely converge on a
one-sided gait: one leg drives while the other acts as a passive strut. Measured on this
project's AMP policy before any symmetry pressure was applied:

    stance fraction   left 0.58   right 0.30   (ratio 1.90)
    mean air time     left 0.046s right 0.134s (ratio 2.9)
    ankle pitch range left 10.0   right 5.0    (ratio 1.99)

None of the usual metrics notice. Fall rate, foot slip, style reward and uprightness all
aggregate over both legs, so a perfectly one-sided gait scores exactly like a symmetric one.
It was spotted by a human watching a video.

This module builds the two permutations needed to state "mirrored input should give
mirrored output":

* an **observation** mirror, mapping a state to its reflection across the sagittal plane
* an **action** mirror, doing the same for the policy's output

Both are derived from the model itself (joint names for the left-right pairing, joint axes
for the sign flips) rather than hardcoded, so they stay correct if the humanoid changes.

Convention: MuJoCo is x forward, y left, z up. Reflecting across the sagittal plane negates
y. A rotation about x (roll) or z (yaw) flips sign under that reflection; a rotation about
y (pitch) does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class MirrorSpec:
    """Index permutations and sign flips that reflect a state or action left-to-right."""

    #: Permutation over actuators, and the sign applied after permuting.
    action_perm: np.ndarray
    action_sign: np.ndarray
    #: Permutation and signs over the proprioceptive observation block.
    obs_perm: np.ndarray
    obs_sign: np.ndarray
    #: Permutation and signs in QPOS order, for the joint blocks of the observation.
    #: Distinct from the actuator-order permutation: this model declares the hip as
    #: (x, y, z) in its joint list but (x, z, y) in its actuator list, so the two orders
    #: are NOT interchangeable. Using one for the other silently mirrors the wrong joints.
    qpos_perm: np.ndarray
    qpos_sign: np.ndarray
    #: Width of the observation block this spec covers. Anything beyond it (the task
    #: observation) is handled separately by the caller.
    obs_width: int

    def mirror_action(self, action: np.ndarray) -> np.ndarray:
        return action[..., self.action_perm] * self.action_sign

    def mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        out[..., : self.obs_width] = (
            obs[..., self.obs_perm] * self.obs_sign
        )
        return out


def _partner_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _axis_sign(axis: np.ndarray) -> float:
    """Sign a hinge rotation picks up when reflected across the sagittal plane.

    Rotations about x (roll) and z (yaw) reverse; rotations about y (pitch) do not.
    Determined from the dominant component of the joint's own axis, so it is correct
    regardless of naming.
    """
    dominant = int(np.argmax(np.abs(axis)))
    return 1.0 if dominant == 1 else -1.0


def build_mirror_spec(
    model: mujoco.MjModel,
    n_joint_pos: int,
    n_joint_vel: int,
    n_feet: int,
    n_actions: int,
) -> MirrorSpec:
    """Derive the mirror permutations for a model and this project's observation layout.

    The proprioceptive layout, from `vec_env._compute_obs`, is:
        joint angles | joint velocities | gravity(3) | lin vel(3) | ang vel(3) |
        previous action | foot contact
    """
    def build(names: list[str], axes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        index_of = {n: i for i, n in enumerate(names)}
        perm = np.arange(len(names))
        sign = np.ones(len(names))
        for i, name in enumerate(names):
            perm[i] = index_of.get(_partner_name(name), i)
            sign[i] = _axis_sign(axes[i])
        return perm, sign

    # Actuator order, for mirroring actions.
    act_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        or f"joint{i}"
        for i in range(model.nu)
    ]
    act_axes = [model.jnt_axis[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    perm, sign = build(act_names, act_axes)

    # qpos order, for mirroring the joint blocks of the observation. These orders differ on
    # this model, which is exactly the kind of mismatch that produces a confidently wrong
    # symmetry loss.
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    q_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}" for j in hinges]
    q_axes = [model.jnt_axis[j] for j in hinges]
    q_perm, q_sign = build(q_names, q_axes)

    # Feet are ordered as the model lists them; mirroring swaps the pair.
    foot_perm = np.arange(n_feet)
    if n_feet == 2:
        foot_perm = np.array([1, 0])

    blocks: list[np.ndarray] = []
    signs: list[np.ndarray] = []
    cursor = 0

    def add(local_perm: np.ndarray, local_sign: np.ndarray) -> None:
        nonlocal cursor
        blocks.append(local_perm + cursor)
        signs.append(local_sign)
        cursor += local_perm.size

    add(q_perm, q_sign)  # joint angles, in qpos order
    add(q_perm, q_sign)  # joint velocities, same order
    # Gravity, linear velocity and angular velocity are body-frame 3-vectors. Reflecting
    # across the sagittal plane negates the y component of a position-like vector, and
    # negates x and z of an angular velocity (an axial vector behaves the opposite way).
    add(np.arange(3), np.array([1.0, -1.0, 1.0]))  # gravity
    add(np.arange(3), np.array([1.0, -1.0, 1.0]))  # linear velocity
    add(np.arange(3), np.array([-1.0, 1.0, -1.0]))  # angular velocity (axial)
    add(perm, sign)  # previous action
    add(foot_perm, np.ones(n_feet))  # foot contact flags

    return MirrorSpec(
        action_perm=perm,
        action_sign=sign,
        qpos_perm=q_perm,
        qpos_sign=q_sign,
        obs_perm=np.concatenate(blocks),
        obs_sign=np.concatenate(signs),
        obs_width=int(cursor),
    )


def verify(spec: MirrorSpec, rng: np.random.Generator, n: int = 64) -> dict[str, float]:
    """Self-check: mirroring twice must be the identity.

    This is the property that catches a wrong sign or a bad pairing, both of which
    otherwise produce a policy that is confidently and subtly wrong.
    """
    obs = rng.normal(size=(n, spec.obs_width))
    act = rng.normal(size=(n, spec.action_perm.size))
    obs_err = float(np.abs(spec.mirror_obs(spec.mirror_obs(obs)) - obs).max())
    act_err = float(np.abs(spec.mirror_action(spec.mirror_action(act)) - act).max())
    return {"obs_involution_error": obs_err, "action_involution_error": act_err}
