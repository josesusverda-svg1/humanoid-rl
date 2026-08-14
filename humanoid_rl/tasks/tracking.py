"""DeepMimic-style motion tracking. Stage 1 of the Phase 3 build order.

The agent is rewarded for physically reproducing a specific reference clip, frame by frame.
This is **not** the final objective. It exists as a bring-up check, and the research was
blunt about skipping it being how people end up with an AMP run that never produces a clean
gait.

What it validates, while failures are still diagnosable:

* the PD gains can actually track a real human trajectory
* contact parameters let the feet push off rather than skid or stick
* the retargeted clips are physically reachable, not just visually plausible
* reference state initialization and early termination are wired correctly

If the humanoid cannot track one walk clip, no adversarial prior will save it. The
difference from AMP is the reward: tracking asks "are you in the same pose as frame 412",
AMP asks "does this look like the kind of thing a human does". Tracking is much easier to
learn and much less general, which is exactly what makes it a good first test.

Reward follows DeepMimic (Peng et al. 2018): a weighted sum of exponential kernels on pose,
velocity, end-effector position and centre of mass. Every term is bounded in (0, 1], so no
single one can dominate, and the exponential gives its strongest gradient near a match,
which is where precision matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humanoid_rl.motion.library import MotionLibrary
from humanoid_rl.tasks.base import BatchState, Task


@dataclass
class TrackingConfig:
    """DeepMimic tracking weights and kernel widths.

    Weights are the paper's: pose dominates, with end-effector position second. Kernel
    widths are the paper's too, and they are not interchangeable with the weights: the
    width sets how quickly reward falls off with error, the weight sets how much that term
    counts once computed.
    """

    w_pose: float = 0.65
    w_vel: float = 0.10
    w_end_effector: float = 0.15
    w_root: float = 0.15

    k_pose: float = 2.0
    k_vel: float = 0.1
    k_end_effector: float = 40.0
    k_root: float = 10.0

    #: An extra upright term. Not in DeepMimic, added because our termination checks it and
    #: a policy that tracks joint angles while slowly toppling should be told earlier.
    w_upright: float = 0.1

    #: End the episode when the pose has diverged this far (mean per-joint radians). Early
    #: termination is called crucial by DeepMimic's ablations: without it the character
    #: falls over and mimes the motion on the ground, a very stable local optimum.
    max_pose_error: float = 1.2
    #: End the episode if the root drifts this far from the reference, in metres.
    max_root_error: float = 0.9
    terminate_torso_upright: float = 0.4

    #: Number of future reference frames included in the observation, and how far apart.
    #: Seeing ahead turns tracking from reactive into anticipatory, which matters because a
    #: walking gait has to commit to a footfall before it happens.
    n_future_frames: int = 2
    future_stride: int = 5


class TrackingTask(Task):
    """Track a reference clip frame by frame."""

    reward_term_names = ("pose", "vel", "end_effector", "root_pos", "upright")

    def __init__(self, library: MotionLibrary, config: TrackingConfig | None = None) -> None:
        self.lib = library
        self.cfg = config or TrackingConfig()
        self.n_key = library.key_local.shape[1]

    @property
    def task_obs_dim(self) -> int:
        # Per future frame: joint targets, key body positions, root height and gravity.
        n_joints = self.lib.qpos.shape[1] - 7
        per_frame = n_joints + self.n_key * 3 + 1 + 3
        # Plus phase, as sin and cos so it is continuous across the clip loop.
        return per_frame * self.cfg.n_future_frames + 2

    # ------------------------------------------------------------------ lifecycle

    def init_state(self, state: BatchState, rng: np.random.Generator) -> None:
        n = state.num_envs
        state.task_state["ref_index"] = np.zeros(n, dtype=np.int64)
        state.task_state["ref_start"] = np.zeros(n, dtype=np.int64)

    def reset_batch(self, state: BatchState, indices: np.ndarray, rng) -> None:
        """Reference state initialization: start at a random frame of a random clip."""
        if indices.size == 0:
            return
        starts = self.lib.sample_starts(indices.size, rng, min_remaining=8)
        state.task_state["ref_start"][indices] = starts
        state.task_state["ref_index"][indices] = starts

    def reset_pose(self, state: BatchState, indices: np.ndarray, rng):
        """Start the humanoid *in* the sampled reference frame.

        The other half of reference state initialization, and what makes tracking learnable
        at all. Starting from the nominal stance and asking the agent to reach frame 412 of
        a walk cycle within one step is not a learnable problem; starting *at* frame 412
        and asking it to continue is.

        Returns an absolute state rather than a perturbation, which is why the engine grew
        a `reset_pose` hook distinct from `reset_noise`.
        """
        if indices.size == 0:
            return None
        starts = state.task_state["ref_index"][indices]
        qpos = self.lib.qpos[starts].copy()
        qvel = self.lib.qvel[starts].copy()

        # The world XY is deliberately NOT zeroed. An earlier version did, to keep episodes
        # near the origin, and it broke tracking completely: `terminated_batch` compares the
        # humanoid's position against the reference's absolute world position, so zeroing
        # one side started every episode tens of metres "away" from its own reference and
        # tripped the 0.9 m termination on the very first step. Measured: 6 of 8
        # environments terminated at step 0.
        #
        # Keeping the reference's world position costs nothing (each environment has its own
        # MjData, and MuJoCo is untroubled by large coordinates) and removes a whole class
        # of frame-mismatch bug.
        return qpos, qvel

    # ------------------------------------------------------------------ observation

    def _future_indices(self, state: BatchState) -> np.ndarray:
        """Clamped indices of the future reference frames, shape (N, n_future)."""
        cfg = self.cfg
        base = state.task_state["ref_index"]
        offsets = (np.arange(cfg.n_future_frames) + 1) * cfg.future_stride
        idx = base[:, None] + offsets[None, :]
        # Clamp within each clip so we never read across a clip boundary into a completely
        # unrelated motion, which would produce a nonsense target for one frame.
        clip = self.lib.clip_of(base)
        last = (self.lib.clip_start[clip] + self.lib.clip_len[clip] - 1)[:, None]
        return np.minimum(idx, last)

    def observe_batch(self, state: BatchState, out: np.ndarray) -> None:
        lib = self.lib
        idx = self._future_indices(state)
        n, f = idx.shape
        flat = idx.reshape(-1)

        joints = lib.qpos[flat, 7:].reshape(n, f, -1)
        keys = lib.key_local[flat].reshape(n, f, -1)
        height = lib.qpos[flat, 2].reshape(n, f, 1)
        gravity = lib.gravity_body[flat].reshape(n, f, 3)
        block = np.concatenate([joints, keys, height, gravity], axis=2).reshape(n, -1)

        # Phase within the clip, as sin and cos so it is continuous at the wrap point.
        base = state.task_state["ref_index"]
        clip = lib.clip_of(base)
        phase = (base - lib.clip_start[clip]) / np.maximum(1, lib.clip_len[clip])
        out[:, : block.shape[1]] = block
        out[:, block.shape[1]] = np.sin(2 * np.pi * phase)
        out[:, block.shape[1] + 1] = np.cos(2 * np.pi * phase)

    def action_offset(self, state: BatchState) -> np.ndarray:
        """A zero action means "hold the current reference pose".

        Without this the PD targets the nominal stance while the reward asks for a walk
        cycle, so the humanoid accelerates away from the motion it is being scored on from
        the very first step. Measured before this was added: the velocity reward term sat
        at exactly 0.0.

        Only the actuated joints, in actuator order, which for this model is qpos[7:].
        """
        return self.lib.qpos[state.task_state["ref_index"], 7:]

    # ------------------------------------------------------------------ reward

    def reward_batch(self, state: BatchState, terms: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        lib = self.lib
        idx = state.task_state["ref_index"]

        # Pose: joint angle error, excluding the free root.
        joint_err = state.qpos[:, 7:] - lib.qpos[idx, 7:]
        pose_sq = np.sum(np.square(joint_err), axis=1)

        # Velocity: joint velocity error, excluding the root.
        vel_err = state.qvel[:, 6:] - lib.qvel[idx, 6:]
        vel_sq = np.sum(np.square(vel_err), axis=1)

        # End effectors, compared in the heading-local frame so the reward does not care
        # which compass direction the character happens to face.
        key_actual = self._key_local(state)
        key_sq = np.sum(np.square(key_actual - lib.key_local[idx]), axis=(1, 2))

        # Full 3D root position, not just height. An earlier version compared only
        # `root_height`, which made the reward blind to horizontal drift: a humanoid 60 cm
        # off course scored identically to one perfectly on it. Nothing pulled the root
        # back except the 0.9 m termination, a sparse signal arriving only after the
        # episode was already lost. Measured symptom: root error sat at 30 cm across 41M
        # steps while pose error, episode length and return all improved.
        root_sq = np.sum(np.square(state.root_pos - lib.qpos[idx, :3]), axis=1)

        upright = np.clip(state.torso_upright, 0.0, 1.0)

        terms[:, 0] = cfg.w_pose * np.exp(-cfg.k_pose * pose_sq)
        terms[:, 1] = cfg.w_vel * np.exp(-cfg.k_vel * vel_sq)
        terms[:, 2] = cfg.w_end_effector * np.exp(-cfg.k_end_effector * key_sq)
        terms[:, 3] = cfg.w_root * np.exp(-cfg.k_root * root_sq)
        terms[:, 4] = cfg.w_upright * upright
        return terms.sum(axis=1)

    def _key_local(self, state: BatchState) -> np.ndarray:
        """Key body positions of the actual humanoid, in its heading-local frame.

        Compared in the heading-local frame so the reward measures *pose*, not compass
        direction. Published by the engine from framepos sensors, so this is an array
        lookup rather than per-environment forward kinematics.
        """
        root = state.root_pos[:, None, :]
        c, s = np.cos(-state.heading), np.sin(-state.heading)
        rel = state.key_body_pos - root
        out = np.empty_like(rel)
        out[:, :, 0] = c[:, None] * rel[:, :, 0] - s[:, None] * rel[:, :, 1]
        out[:, :, 1] = s[:, None] * rel[:, :, 0] + c[:, None] * rel[:, :, 1]
        out[:, :, 2] = rel[:, :, 2]
        return out

    # ------------------------------------------------------------------ progression

    def on_batch_end(self, state: BatchState, metrics: dict[str, float]) -> dict[str, float]:
        """Advance the reference playhead by one frame for every environment."""
        idx = state.task_state["ref_index"]
        clip = self.lib.clip_of(idx)
        last = self.lib.clip_start[clip] + self.lib.clip_len[clip] - 1
        state.task_state["ref_index"] = np.minimum(idx + 1, last)
        return {}

    def terminated_batch(self, state: BatchState) -> np.ndarray:
        cfg = self.cfg
        idx = state.task_state["ref_index"]

        pose_err = np.abs(state.qpos[:, 7:] - self.lib.qpos[idx, 7:]).mean(axis=1)
        root_err = np.linalg.norm(state.qpos[:, :3] - self.lib.qpos[idx, :3], axis=1)
        diverged = ~np.isfinite(state.qpos).all(axis=1)
        return (
            (pose_err > cfg.max_pose_error)
            | (root_err > cfg.max_root_error)
            | (state.torso_upright < cfg.terminate_torso_upright)
            | diverged
        )

    def success_batch(self, state: BatchState) -> np.ndarray:
        """Success means reaching the end of the clip without diverging."""
        idx = state.task_state["ref_index"]
        return self.lib.frames_left(idx) <= 1

    def eval_metrics(self, state: BatchState) -> dict[str, float]:
        idx = state.task_state["ref_index"]
        pose_err = np.abs(state.qpos[:, 7:] - self.lib.qpos[idx, 7:]).mean(axis=1)
        root_err = np.linalg.norm(state.qpos[:, :3] - self.lib.qpos[idx, :3], axis=1)
        return {
            "pose_error_rad": float(pose_err.mean()),
            "root_error_m": float(root_err.mean()),
            "torso_upright": float(state.torso_upright.mean()),
        }
