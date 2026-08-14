"""Motion library: all retargeted clips in one flat, index-addressable structure.

Two things drive the design.

**Flat concatenation.** Every clip is stored end to end in one array, with a per-clip start
offset. Reference lookups then become a single numpy fancy-index across the whole batch of
environments, instead of a Python loop over per-clip objects. This is the same rule that
governs `vec_env.py`: per-environment Python is the thing that kills throughput.

**Precomputed derived quantities.** A tracking reward needs end-effector positions and the
centre of mass of the *reference* pose, which normally means running forward kinematics on
the reference every step, for every environment. Instead FK is run once per frame when the
library is built and the results are cached. About 12,000 `mj_forward` calls costing a
couple of seconds at startup, in exchange for making the reward a pure array lookup.

End-effector positions are stored in the reference's own **heading-local frame**, so a clip
walking north and the same clip walking east produce identical numbers. Without that, a
tracking reward would penalise a policy for facing the wrong compass direction, which has
nothing to do with whether its gait matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from humanoid_rl.motion.retarget import MotionClip
from humanoid_rl.tasks.base import quat_rotate_inverse, quat_to_heading

#: Bodies whose positions the tracking reward matches. Hands and feet are the standard
#: DeepMimic and AMP choice: they are the extremities of the kinematic chain, so matching
#: them constrains everything upstream, and they are what a viewer actually looks at.
DEFAULT_KEY_BODIES = ("left_foot", "right_foot", "left_hand", "right_hand")


@dataclass
class MotionLibrary:
    """Every reference clip, flattened, with derived quantities cached."""

    qpos: np.ndarray  # (total_frames, nq)
    qvel: np.ndarray  # (total_frames, nv)
    #: (total_frames, n_key, 3) key body positions relative to the root, heading-removed.
    key_local: np.ndarray
    #: (total_frames, 3) centre of mass relative to the root, heading-removed.
    com_local: np.ndarray
    #: (total_frames, 3) gravity direction in the reference's body frame.
    gravity_body: np.ndarray
    clip_start: np.ndarray  # (n_clips,) index of each clip's first frame
    clip_len: np.ndarray  # (n_clips,)
    names: list[str]
    fps: float
    key_bodies: tuple[str, ...]

    @property
    def n_clips(self) -> int:
        return len(self.names)

    @property
    def total_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps

    # ------------------------------------------------------------------ construction

    @staticmethod
    def build(
        clip_dir: str | Path,
        model: mujoco.MjModel,
        key_bodies: tuple[str, ...] = DEFAULT_KEY_BODIES,
        include: list[str] | None = None,
        mirror: bool = False,
    ) -> "MotionLibrary":
        """Load every .npz clip in a directory and precompute derived quantities.

        Args:
            clip_dir: Directory of clips written by `scripts/retarget.py`.
            model: The model the clips were retargeted onto. Used for forward kinematics.
            key_bodies: Bodies whose positions the tracking reward will match.
            include: Optional list of clip name substrings to keep. Useful for training on
                walking only while leaving runs in the directory.
            mirror: Also include a left-right reflection of every clip. This makes the
                reference distribution exactly symmetric, so an AMP discriminator trained on
                it cannot reward the performer's own left/right imbalance (measured at 1.32
                here) as part of looking human. See `motion.mirror` for why this is
                preferred to adding a symmetry term to the policy objective.
        """
        clip_dir = Path(clip_dir)
        paths = sorted(clip_dir.glob("*.npz"))
        if include:
            paths = [p for p in paths if any(tag in p.stem for tag in include)]
        if not paths:
            raise FileNotFoundError(
                f"no motion clips in {clip_dir}"
                + (f" matching {include}" if include else "")
                + ". Run scripts/retarget.py first."
            )

        clips = [MotionClip.load(p) for p in paths]
        fps_values = {round(c.fps, 3) for c in clips}
        if len(fps_values) > 1:
            raise ValueError(f"clips disagree on frame rate: {fps_values}")

        names = [c.name or p.stem for c, p in zip(clips, paths)]
        if mirror:
            # Mirrored before concatenation, so every derived quantity below (key body
            # positions, centre of mass, gravity direction) is recomputed by the same
            # forward-kinematics pass rather than reflected by hand.
            from humanoid_rl.envs.mirror import build_mirror_spec
            from humanoid_rl.motion.mirror import mirror_clip

            spec = build_mirror_spec(
                model,
                n_joint_pos=model.nq - 7,
                n_joint_vel=model.nv - 6,
                n_feet=2,
                n_actions=model.nu,
            )
            reflected = [mirror_clip(c, model, spec) for c in clips]
            clips = clips + reflected
            names = names + [f"{n}_mirror" for n in names]

        qpos = np.concatenate([c.qpos for c in clips], axis=0)
        qvel = np.concatenate([c.qvel for c in clips], axis=0)
        lengths = np.array([c.n_frames for c in clips], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

        key_ids = []
        for name in key_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"model has no body named {name!r}")
            key_ids.append(bid)

        n = qpos.shape[0]
        key_local = np.zeros((n, len(key_ids), 3))
        com_local = np.zeros((n, 3))
        gravity_body = np.zeros((n, 3))
        down = np.array([0.0, 0.0, -1.0])

        data = mujoco.MjData(model)
        for f in range(n):
            data.qpos[:] = qpos[f]
            mujoco.mj_forward(model, data)
            root_pos = data.qpos[0:3]
            quat = data.qpos[3:7].copy()

            # Remove heading so the same motion in any compass direction is identical.
            yaw = float(quat_to_heading(quat[None, :])[0])
            c, s = np.cos(-yaw), np.sin(-yaw)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

            for k, bid in enumerate(key_ids):
                key_local[f, k] = rot @ (data.xpos[bid] - root_pos)
            com_local[f] = rot @ (np.asarray(data.subtree_com[0]) - root_pos)
            gravity_body[f] = quat_rotate_inverse(quat[None, :], down[None, :])[0]

        return MotionLibrary(
            qpos=qpos,
            qvel=qvel,
            key_local=key_local,
            com_local=com_local,
            gravity_body=gravity_body,
            clip_start=starts,
            clip_len=lengths,
            names=names,
            fps=float(clips[0].fps),
            key_bodies=tuple(key_bodies),
        )

    # ------------------------------------------------------------------ sampling

    def sample_starts(
        self, n: int, rng: np.random.Generator, min_remaining: int = 2
    ) -> np.ndarray:
        """Sample `n` random start frames, uniformly over all usable frames.

        This is **reference state initialization (RSI)**: episodes begin at a random point
        in a random clip rather than always at the beginning. DeepMimic's own ablations
        call it crucial. Without it the agent must learn the motion strictly in order, and
        never sees the middle of a stride until it has mastered the start, which makes
        long clips almost unlearnable.

        Sampling uniformly over concatenated frames rather than picking a clip then a frame
        weights each clip by its length, which is what we want: a 30 second clip should
        contribute more starts than a 12 second one.
        """
        usable = self.clip_len - min_remaining
        if np.any(usable <= 0):
            raise ValueError("some clips are shorter than min_remaining frames")
        # Cumulative usable lengths give a length-weighted uniform draw in one shot.
        cumulative = np.cumsum(usable)
        picks = rng.integers(0, cumulative[-1], size=n)
        clip_idx = np.searchsorted(cumulative, picks, side="right")
        offset = picks - np.where(clip_idx > 0, cumulative[clip_idx - 1], 0)
        return self.clip_start[clip_idx] + offset

    def frames_left(self, index: np.ndarray) -> np.ndarray:
        """How many frames remain in the clip containing each global index."""
        clip_idx = np.searchsorted(self.clip_start, index, side="right") - 1
        return self.clip_start[clip_idx] + self.clip_len[clip_idx] - index - 1

    def clip_of(self, index: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.clip_start, index, side="right") - 1

    def summary(self) -> str:
        lines = [
            f"{self.n_clips} clips, {self.total_frames:,} frames, "
            f"{self.duration:.0f}s at {self.fps:.0f} fps",
            f"key bodies: {', '.join(self.key_bodies)}",
        ]
        for i, name in enumerate(self.names):
            lines.append(f"  {name:<18s} {self.clip_len[i]:>5d} frames "
                         f"{self.clip_len[i] / self.fps:>5.1f}s")
        return "\n".join(lines)
