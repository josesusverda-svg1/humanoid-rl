"""Task interface, batch-oriented.

A *task* is everything that makes one training objective different from another: what the
agent observes beyond its own body state, how it is rewarded, and when an episode ends.
The physics engine and the parallel stepping machinery know nothing about any of it.

That separation is the point. Adding a future skill (jump over an obstacle, climb a box)
means writing a new Task, adding a terrain generator, and listing reward terms. It must
never mean touching `humanoid_rl/envs/vec_env.py`.

Why every callback takes the WHOLE batch
----------------------------------------
The obvious design gives each task a per-environment callback. It was measured on this
machine and it is roughly five times slower, for a reason specific to threaded Python.

With ten worker threads, Python bytecode is serialised by the GIL, so *total* Python work
per batch step is what matters, not per-thread work. Measured marginal costs per
environment step under ten-way contention:

    gather qpos/qvel into a batch array      0.09 us   (free)
    numpy action clip and scale              4.90 us
    numpy-scalar reward arithmetic          13.37 us
    four mju_rotVecQuat pybind11 calls      23.78 us   (worst)

A pybind11 call that costs well under a microsecond single-threaded costs about six
microseconds under contention. Multiply by 1024 environments and per-environment Python
becomes the entire runtime, dwarfing the physics it was meant to support.

So the split is: worker threads do C work only (`mj_step`) and copy raw state out, which
is free. Every piece of arithmetic then happens once, vectorised over all environments, on
the main thread. Task authors write numpy over `(num_envs, ...)` arrays, which is both
faster and usually clearer than a scalar loop.

Terminology
-----------
* Observation: the vector the policy sees. Split into a *proprioceptive* part (the body
  itself, identical for every task) and a *task* part (goals, commands, targets).
* Termination: the episode ended through the agent's own behaviour (it fell). The
  resulting state genuinely has zero value.
* Truncation: the episode hit a time limit. The state still had value, so the value
  estimate must be bootstrapped. Conflating the two is a classic and costly RL bug, so
  they stay separate everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BatchState:
    """State of every environment, as batched arrays. Row `i` is environment `i`.

    Populated once per step by the engine. The derived fields are computed vectorised so
    that tasks never need a quaternion routine of their own, and never pay the pybind11
    cost of MuJoCo's scalar math helpers.
    """

    # --- raw physics state, copied out of MjData by the workers ---
    qpos: np.ndarray  # (N, nq) full generalised position, root free joint first
    qvel: np.ndarray  # (N, nv) full generalised velocity
    ctrl: np.ndarray  # (N, nu) target joint angles actually applied, in radians
    action: np.ndarray  # (N, nu) raw policy output, clipped to [-1, 1]
    prev_action: np.ndarray  # (N, nu) previous step's raw action
    episode_step: np.ndarray  # (N,) steps elapsed in the current episode
    #: (N, n_feet) normal force under each foot in newtons, from touch sensors.
    foot_force: np.ndarray
    #: (N, n_feet) True where that foot is bearing load. See `contact_force_threshold`.
    #: (N, nu) torque actually applied by the actuators this step. The hardware-proven
    #: reward stacks all penalise torque squared, the physical effort, rather than the
    #: position-command magnitude this project once penalised, which is a different
    #: quantity entirely (a large command against a spring can cost little torque and a
    #: small command against gravity a lot).
    torque: np.ndarray
    #: (N, nu) hinge joint velocities from the PREVIOUS control step, for the dof_acc
    #: penalty: all three reference robots penalise joint acceleration (about -1e-7 per
    #: (rad/s^2)^2), the canonical anti-vibration term. Its absence here is plausibly why
    #: learning ever found action tremor profitable.
    prev_joint_vel: np.ndarray
    foot_contact: np.ndarray
    #: (N, n_feet) consecutive seconds each foot has been off the ground. The single most
    #: useful signal for shaping a stepping gait rather than a shuffle.
    foot_air_time: np.ndarray
    #: (N, n_feet) True only on the step a foot touches down. Paired with `foot_air_time`
    #: this gives the standard air-time reward, which pays out once per footfall in
    #: proportion to how long that foot was swinging.
    foot_first_contact: np.ndarray
    #: (N, n_feet, 3) world-frame linear velocity of each foot. Horizontal speed while a
    #: foot is loaded is slip, which is what separates walking from skating.
    foot_lin_vel: np.ndarray
    #: (N,) world z component of the torso's own up-axis. 1.0 is fully upright, 0.0 is
    #: horizontal. Distinct from `gravity_body`, which describes only the pelvis: a
    #: humanoid can hold its pelvis level while folding its torso flat, and this is the
    #: term that notices.
    torso_upright: np.ndarray
    #: (N, 3) the torso's own z-axis in WORLD coordinates.
    #:
    #: `torso_upright` is only this vector's z component, i.e. cos(tilt), which is SIGN-BLIND:
    #: a 48 degree forward stoop and a 48 degree backward arch give the identical number. That
    #: blindness cost this project a full analysis cycle, during which a backward fold was
    #: diagnosed, reported and nearly "fixed" as a forward lean. Any task that cares which way
    #: the torso is bent must use this and rotate it into the heading frame.
    torso_zaxis: np.ndarray
    #: (N,) world height of the head, normalised against its standing height.
    head_height_ratio: np.ndarray
    #: (N, n_key, 3) world positions of the key bodies (feet and hands). Used by the
    #: tracking reward and, in Phase 3, by the AMP discriminator observation.
    key_body_pos: np.ndarray

    # --- derived, computed vectorised by the engine every step ---
    root_pos: np.ndarray  # (N, 3) world position of the pelvis
    root_height: np.ndarray  # (N,) convenience view of root_pos[:, 2]
    gravity_body: np.ndarray  # (N, 3) gravity direction in the body frame
    lin_vel_body: np.ndarray  # (N, 3) linear velocity in the body frame
    ang_vel_body: np.ndarray  # (N, 3) angular velocity in the body frame
    heading: np.ndarray  # (N,) yaw of the root in the world XY plane, radians
    #: (N,) world z of the ground surface directly under the root. **Identically 0.0 on a
    #: plane**, which is what makes every terrain-aware reward term an exact algebraic no-op
    #: on flat ground rather than an approximate one -- the flat results stay bit-comparable
    #: and a warm start across the change is exact.
    #:
    #: Computed once here rather than looked up by each consumer. There are seven consumers,
    #: and E29's lesson is that a guarantee proved for one term does not transfer to another
    #: term in the same function: seven independent lookups would drift apart.
    ground_z: np.ndarray
    #: (N, n_key) the same, under each key body. A humanoid's foot is not above its pelvis,
    #: and on rough ground the difference is the whole point.
    key_ground_z: np.ndarray
    #: Seconds of simulated time per control step, so tasks can express rewards in
    #: physical units rather than in steps.
    dt: float = 0.02

    #: Free-form per-task state, keyed by name. Each value is an array whose first
    #: dimension is num_envs. The velocity-command task keeps its commands here; the
    #: waypoint task will keep courses and target indices here.
    task_state: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_envs(self) -> int:
        return self.qpos.shape[0]

    def to_local_frame(self, world_xy: np.ndarray) -> np.ndarray:
        """Rotate world-frame XY vectors, shape (N, 2), into each humanoid's heading frame.

        Goal-conditioned policies must see targets in their own frame. Feeding world
        coordinates instead forces the network to learn the rotation itself, which wastes
        capacity and generalises badly to headings unseen in training.
        """
        c, s = np.cos(-self.heading), np.sin(-self.heading)
        return np.stack(
            [c * world_xy[:, 0] - s * world_xy[:, 1], s * world_xy[:, 0] + c * world_xy[:, 1]],
            axis=1,
        )


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` from the world frame into the frame defined by `quat`, batched.

    Args:
        quat: (N, 4) unit quaternions in MuJoCo order (w, x, y, z).
        vec: (N, 3) vectors in the world frame.

    Returns:
        (N, 3) vectors in the local frame.

    Implemented directly in numpy rather than by calling `mujoco.mju_rotVecQuat` per
    environment. That pybind11 call measured 23.78 us per environment step across four
    uses under ten-thread contention, which was the single largest cost in the loop.
    """
    w = quat[:, 0:1]
    u = quat[:, 1:4]
    # Rotation by the conjugate: v*(2w^2 - 1) - 2w*(u x v) + 2u*(u . v)
    return (
        vec * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(u, vec)
        + 2.0 * u * np.sum(u * vec, axis=1, keepdims=True)
    )


def quat_to_heading(quat: np.ndarray) -> np.ndarray:
    """Yaw angle in the world XY plane for a batch of quaternions, shape (N,).

    Taken from the rotated forward axis rather than a full Euler conversion, which avoids
    gimbal problems when the humanoid pitches forward while walking.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    return np.arctan2(fwd_y, fwd_x)


class Task(ABC):
    """Base class for every training objective.

    Every method receives the full `BatchState` and writes into preallocated output
    arrays. Implementations should be pure numpy over the batch dimension, and should
    avoid Python loops over environments.

    Threading: all of these run on the main thread, single-threaded, between physics
    phases. No locking is needed anywhere in a Task.
    """

    #: Names of the individual reward components, logged separately so the dashboard can
    #: show which term is actually driving behaviour. Nearly every locomotion debugging
    #: session ends up being "one term quietly dominated everything else".
    reward_term_names: tuple[str, ...] = ()

    @property
    @abstractmethod
    def task_obs_dim(self) -> int:
        """Width of the task-specific observation appended to the proprioceptive vector."""

    @abstractmethod
    def init_state(self, state: BatchState, rng: np.random.Generator) -> None:
        """Allocate this task's arrays inside `state.task_state`. Called once at startup."""

    @abstractmethod
    def reset_batch(
        self, state: BatchState, indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Re-randomise task state for the environments listed in `indices`.

        Called after physics has been reset for those environments. `indices` is an
        integer array and may be empty. Use fancy indexing, not a loop.
        """

    @abstractmethod
    def observe_batch(self, state: BatchState, out: np.ndarray) -> None:
        """Write the (N, task_obs_dim) task observation into `out`, a preallocated array."""

    @abstractmethod
    def reward_batch(self, state: BatchState, terms: np.ndarray) -> np.ndarray:
        """Return the (N,) reward, and write each component into `terms`, shape (N, n_terms).

        The returned scalar per environment is what the agent optimises. The components
        exist purely for logging and must not influence training.
        """

    @abstractmethod
    def terminated_batch(self, state: BatchState) -> np.ndarray:
        """Return a (N,) boolean array: True where the episode ended through behaviour."""

    def success_batch(self, state: BatchState) -> np.ndarray:
        """Return a (N,) boolean array: True where the objective was achieved.

        Default is all False. The waypoint task overrides this to mean "completed the
        entire course", which is the definition the project is graded on.
        """
        return np.zeros(state.num_envs, dtype=bool)

    def reset_pose(
        self, state: BatchState, indices: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Optional ABSOLUTE initial state for the environments in `indices`.

        Returns `(qpos, qvel)` with shapes (len(indices), nq) and (len(indices), nv), used
        verbatim instead of the nominal pose. Distinct from `reset_noise`, which perturbs
        the nominal pose: motion tracking must begin *in* a reference frame, not near the
        default stance, and expressing that as a perturbation of the default would be both
        awkward and wrong.

        Takes precedence over `reset_noise` when both are provided.
        """
        return None

    def reset_noise(
        self, state: BatchState, indices: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Optional initial-state randomisation for the environments in `indices`.

        Returns `(qpos_noise, qvel_noise)` with shapes (len(indices), nq) and
        (len(indices), nv), added to the default pose by the worker threads. Return None
        for no randomisation.

        This is also the hook where Phase 3 will implement reference state initialisation
        (starting episodes from poses sampled out of the mocap data), which is one of the
        highest-value tricks in motion imitation.
        """
        return None

    def on_batch_end(self, state: BatchState, metrics: dict[str, float]) -> dict[str, float]:
        """Hook run once per step, on the main thread, for cross-environment bookkeeping.

        The right place for curriculum level updates and running success-rate estimates.
        Returns extra scalars to merge into the training metrics.
        """
        return {}

    def mirror_task_obs(self, task_obs):
        """Reflect this task's observation block left-to-right, or None if not supported.

        The proprioceptive block is mirrored generically by `envs/mirror.py`, but the task
        block is task-specific: a velocity command mirrors by negating its sideways and
        yaw components, while a waypoint mirrors by negating the target's y coordinate.
        Returning None disables the symmetry loss for this task rather than silently
        applying a wrong transform.

        Accepts and returns a torch tensor, since this runs inside the PPO update.
        """
        return None

    def action_offset(self, state: BatchState) -> np.ndarray | None:
        """Optional per-step baseline that a zero action corresponds to, shape (N, nu).

        By default a zero action means the nominal standing pose. A motion-tracking task
        overrides this to mean *the current reference pose*, which turns the policy's job
        from "reproduce the whole trajectory" into "correct the small errors between the
        reference and what physics actually does". That is a far easier learning problem
        and it is what DeepMimic-style controllers do.

        Return None to keep the nominal pose.
        """
        return None

    def eval_metrics(self, state: BatchState) -> dict[str, float]:
        """Task-specific scalars recorded during deterministic evaluation.

        Called once per evaluation step and averaged by the evaluator, so return
        instantaneous batch means, not accumulations. This is how a task reports the
        quality measures that matter to it specifically (command tracking error now,
        gait discriminator score and waypoints reached in later phases) without the
        evaluator needing to know anything about the task.
        """
        return {}
