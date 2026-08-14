"""Velocity-command locomotion with an adversarial motion prior. Phase 3 Stage 2.

The task reward is the same velocity-command objective Phase 2 trained on. What changes is
that a discriminator, trained on real human mocap, also scores how human the motion looks,
and that score becomes half the reward.

    r = 0.5 * r_task + 0.5 * r_style

The division of labour matters and mirrors the rest of the codebase: this class does the
numpy work (rolling observation history, task reward) on the main thread, and the trainer
does the torch work (discriminator forward and update) on the GPU. The style reward is
blended in by the trainer, because computing it here would mean a GPU round-trip inside
what is otherwise pure numpy.

Two AMP details that are easy to get wrong and are handled here:

* **Reference state initialization.** Episodes begin from a random frame of a random clip,
  in that pose, not from the nominal stance. DeepMimic's ablations call RSI crucial and AMP
  inherits it.
* **Seeding the observation history.** After an RSI reset the AMP observation *history*
  must come from the clip's preceding frames, not from zeros or from repeated copies of the
  current frame. Miss this and the discriminator sees a fabricated transition on every
  single reset, which at these environment counts is a large fraction of its negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from humanoid_rl.algos.amp import build_amp_features
from humanoid_rl.motion.library import MotionLibrary
from humanoid_rl.tasks.base import BatchState
from humanoid_rl.tasks.locomotion import LocomotionConfig, LocomotionTask


@dataclass
class AMPLocomotionConfig(LocomotionConfig):
    """Velocity-command settings plus AMP-specific initialisation."""

    #: Fraction of episodes that begin from a reference frame rather than the nominal
    #: stance. Some nominal-stance starts are kept so the policy still learns to set off
    #: from standing, which is what Phase 5 waypoint episodes will begin from.
    rsi_prob: float = 0.85
    #: Fraction of RSI episodes whose velocity command is taken from the reference clip's
    #: own motion at that frame. Starting mid-stride at 1 m/s while being commanded to walk
    #: backwards is a contradiction the policy cannot satisfy, and it injects noise into
    #: early training.
    reference_command_prob: float = 0.5
    #: Reward blend. The AMP paper uses 0.5 / 0.5 for every task.
    task_reward_weight: float = 0.5
    style_reward_weight: float = 0.5


class AMPLocomotionTask(LocomotionTask):
    """Velocity-command walking, with the AMP observation history maintained per step."""

    def __init__(
        self,
        library: MotionLibrary,
        config: AMPLocomotionConfig | None = None,
        n_obs_frames: int = 2,
    ) -> None:
        super().__init__(config or AMPLocomotionConfig())
        self.lib = library
        self.n_obs_frames = n_obs_frames
        # Per-frame features of every reference frame, precomputed once.
        self.ref_features = build_amp_features(library.qpos, library.qvel, library.key_local)
        self.per_frame = self.ref_features.shape[1]

    @property
    def amp_obs_dim(self) -> int:
        return self.per_frame * self.n_obs_frames

    # ------------------------------------------------------------------ lifecycle

    def init_state(self, state: BatchState, rng: np.random.Generator) -> None:
        super().init_state(state, rng)
        n = state.num_envs
        # Oldest frame first, newest last, so flattening gives (s_{t-1}, s_t).
        state.task_state["amp_history"] = np.zeros(
            (n, self.n_obs_frames, self.per_frame), dtype=np.float32
        )
        state.task_state["rsi_frame"] = np.full(n, -1, dtype=np.int64)

    def reset_batch(self, state: BatchState, indices: np.ndarray, rng) -> None:
        """Sample the velocity command, drawing it from the reference where RSI applies."""
        if indices.size == 0:
            return
        super().reset_batch(state, indices, rng)

        frames = state.task_state["rsi_frame"][indices]
        use_ref = (frames >= 0) & (rng.random(indices.size) < self.cfg.reference_command_prob)
        if not use_ref.any():
            return

        sel = indices[use_ref]
        ref = self.lib.qvel[state.task_state["rsi_frame"][sel]]
        quat = self.lib.qpos[state.task_state["rsi_frame"][sel], 3:7]
        # Reference root velocity expressed in its own heading frame, matching how the
        # command is interpreted.
        from humanoid_rl.tasks.base import quat_to_heading

        yaw = quat_to_heading(quat)
        c, s = np.cos(-yaw), np.sin(-yaw)
        command = np.stack(
            [c * ref[:, 0] - s * ref[:, 1], s * ref[:, 0] + c * ref[:, 1], ref[:, 5]], axis=1
        )
        state.task_state["command"][sel] = np.clip(command, -2.0, 2.0)

    def reset_pose(self, state: BatchState, indices: np.ndarray, rng):
        """Reference state initialization, and seeding of the AMP observation history."""
        if indices.size == 0:
            return None

        n = indices.size
        use_rsi = rng.random(n) < self.cfg.rsi_prob
        frames = np.full(n, -1, dtype=np.int64)
        if use_rsi.any():
            # Need enough history *behind* the sampled frame to seed the observation.
            frames[use_rsi] = self.lib.sample_starts(
                int(use_rsi.sum()), rng, min_remaining=4
            ) + self.n_obs_frames
            frames[use_rsi] = np.minimum(frames[use_rsi], self.lib.total_frames - 2)
        state.task_state["rsi_frame"][indices] = frames

        history = state.task_state["amp_history"]
        for slot in range(self.n_obs_frames):
            # slot 0 is the oldest, so it reads furthest back in the clip.
            offset = self.n_obs_frames - 1 - slot
            src = np.maximum(frames - offset, 0)
            rows = indices[use_rsi]
            if rows.size:
                history[rows, slot] = self.ref_features[src[use_rsi]]

        if not use_rsi.any():
            return None

        # Non-RSI environments fall back to the nominal stance, which the engine supplies.
        qpos = np.repeat(state.qpos[indices[:1]] * 0.0, n, axis=0)
        qvel = np.zeros((n, state.qvel.shape[1]))
        default = self._nominal_qpos(state)
        qpos[:] = default
        qpos[use_rsi] = self.lib.qpos[frames[use_rsi]]
        qvel[use_rsi] = self.lib.qvel[frames[use_rsi]]
        return qpos, qvel

    def _nominal_qpos(self, state: BatchState) -> np.ndarray:
        cached = state.task_state.get("_nominal_qpos")
        if cached is None:
            raise RuntimeError(
                "AMPLocomotionTask needs the engine to publish its nominal pose as "
                "task_state['_nominal_qpos']"
            )
        return cached

    # ------------------------------------------------------------------ per step

    def on_batch_end(self, state: BatchState, metrics: dict[str, float]) -> dict[str, float]:
        """Roll the AMP observation history forward by one frame.

        Calls up first: the base task advances the gait clock here, and skipping that would
        freeze the clock at its reset value and silently disable the gait-phase reward.
        """
        super().on_batch_end(state, metrics)
        history = state.task_state["amp_history"]
        history[:, :-1] = history[:, 1:]
        history[:, -1] = self._features_now(state)
        return {}

    def _features_now(self, state: BatchState) -> np.ndarray:
        root = state.root_pos[:, None, :]
        c, s = np.cos(-state.heading), np.sin(-state.heading)
        rel = state.key_body_pos - root
        key_local = np.empty_like(rel)
        key_local[:, :, 0] = c[:, None] * rel[:, :, 0] - s[:, None] * rel[:, :, 1]
        key_local[:, :, 1] = s[:, None] * rel[:, :, 0] + c[:, None] * rel[:, :, 1]
        key_local[:, :, 2] = rel[:, :, 2]
        return build_amp_features(state.qpos, state.qvel, key_local)

    def amp_observation(self, state: BatchState) -> np.ndarray:
        """(N, amp_obs_dim) flattened history, oldest frame first."""
        history = state.task_state["amp_history"]
        return history.reshape(history.shape[0], -1)
