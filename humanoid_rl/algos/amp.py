"""Adversarial Motion Priors (AMP), Peng et al. 2021. Phase 3 Stage 2.

Where motion *tracking* asks "are you in the same pose as frame 412", AMP asks the much
looser and more useful question "does this look like the kind of thing a human does". A
discriminator is trained to tell real mocap transitions from the policy's own, and its
verdict becomes part of the reward. The policy is then free to solve the task however it
likes, as long as the result stays on the manifold of human-looking motion.

That looseness is exactly what the project needs: the reference clips are all flat-ground
walking, but Phase 4 puts the humanoid on slopes and steps, where frame-by-frame tracking
has nothing to say.

Implementation follows the **paper**, not the widely-copied Isaac Gym config. That matters
more than it sounds: the paper trained on 16 CPU cores with 4096 samples per update and a
256 discriminator batch, so its Table 4 already *is* the small-batch CPU configuration.
`HumanoidAMPPPO.yaml` is a later retune for 4096 GPU environments and does not transfer.

Terminology
-----------
* **Discriminator**: a network scoring how real a motion transition looks. Trained against
  mocap (real) and policy rollouts (fake).
* **Style reward**: the discriminator's score, folded into the agent's reward.
* **Gradient penalty**: a term penalising large discriminator gradients on real data. AMP's
  own ablation calls it the single most vital stabiliser; without it the paper reports
  "large performance fluctuations" and visible artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from humanoid_rl.algos.networks import RunningMeanStd
from humanoid_rl.motion.library import MotionLibrary
from humanoid_rl.tasks.base import BatchState, quat_to_heading


@dataclass
class AMPConfig:
    """AMP paper Table 4 values, adapted where our model differs."""

    #: Frames of history in one discriminator observation. The paper, IsaacGymEnvs and
    #: IsaacLab all use 2. MimicKit's current configs use 10, which is a later change and
    #: not the published method. More frames give a richer style signal but also more room
    #: for the discriminator to latch onto simulation-specific temporal signatures such as
    #: contact chatter.
    n_obs_frames: int = 2

    #: Discriminator trunk. Unanimous across the paper and every implementation.
    hidden: tuple[int, ...] = (1024, 512)

    #: Reward blend. The paper uses 0.5 / 0.5 for every task. This is the primary tuning
    #: knob and is genuinely sensitive: Disney's published sweep found 0.4 best once safety
    #: was scored alongside style.
    task_reward_weight: float = 0.5
    style_reward_weight: float = 0.5

    #: Gradient penalty on real data. The paper's term carries a factor of w_gp/2, so an
    #: effective multiplier of 5, which is what NVIDIA configs write directly as 5.
    grad_penalty_coef: float = 5.0
    #: L2 on the final logit layer only.
    logit_reg_coef: float = 0.01
    weight_decay: float = 1.0e-4

    learning_rate: float = 1.0e-4
    #: Updates per PPO iteration, with its own optimiser rather than a joint loss, so the
    #: discriminator learning rate is independently tunable. This is the paper's design.
    n_epochs: int = 2
    batch_size: int = 512

    #: Policy transitions kept for training the discriminator. The paper's value; the 1e6
    #: used by IsaacGymEnvs is scaled for far larger rollouts than ours.
    replay_size: int = 100_000
    #: Once the buffer is full, each new sample is kept with this probability. Without it a
    #: large buffer is dominated by the most recent policy and the discriminator overfits.
    replay_keep_prob: float = 0.01

    #: Use the paper's bounded least-squares reward rather than the unbounded BCE variant
    #: shipped by NVIDIA, skrl and MimicKit. An unbounded style reward is far easier to
    #: blow up against a fixed-scale task reward, and we have small batches and no ability
    #: to brute-force through instability.
    bounded_reward: bool = True


def amp_observation_dim(n_joints: int, n_key_bodies: int, n_frames: int) -> int:
    """Width of one AMP observation.

    Per frame: root height (1), root rotation as tangent-normal (6), heading-local root
    linear velocity (3), heading-local root angular velocity (3), joint angles, joint
    velocities, and key body positions in the heading-local frame.
    """
    per_frame = 1 + 6 + 3 + 3 + n_joints + n_joints + n_key_bodies * 3
    return per_frame * n_frames


def _tangent_normal(quat: np.ndarray) -> np.ndarray:
    """Represent rotations as two rotated basis vectors, shape (N, 6).

    Preferred over quaternions for network input because it is continuous: quaternions have
    a double cover (q and -q are the same rotation), so a network fed raw quaternions must
    learn to treat two distant inputs as identical. The tangent-normal form has no such
    discontinuity.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    tangent = np.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], axis=1
    )
    normal = np.stack(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], axis=1
    )
    return np.concatenate([tangent, normal], axis=1)


def _heading_inverse_quat(quat: np.ndarray) -> np.ndarray:
    """Quaternion that removes the yaw component of `quat`, shape (N, 4)."""
    yaw = quat_to_heading(quat)
    half = -0.5 * yaw
    out = np.zeros_like(quat)
    out[:, 0] = np.cos(half)
    out[:, 3] = np.sin(half)
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def build_amp_features(
    qpos: np.ndarray,
    qvel: np.ndarray,
    key_local: np.ndarray,
    local_root_obs: bool = True,
) -> np.ndarray:
    """One frame of AMP features from raw state, shape (N, per_frame).

    Args:
        qpos: (N, nq) with a free root first.
        qvel: (N, nv).
        key_local: (N, n_key, 3) key body positions already in the heading-local frame.
        local_root_obs: Remove heading from the root rotation.

    **`local_root_obs=True` is the single highest-value detail in this file.** The shipped
    Isaac config sets it False, which leaves root rotation in world coordinates and teaches
    the discriminator a heading-*dependent* notion of style. For a policy that must walk
    toward arbitrary waypoints that is fatal: it would be penalised for facing the wrong
    compass direction, which has nothing to do with whether its gait looks human.
    """
    quat = qpos[:, 3:7]
    heading_inv = _heading_inverse_quat(quat)

    root_rot = _quat_mul(heading_inv, quat) if local_root_obs else quat
    rot_features = _tangent_normal(root_rot)

    # Root velocities into the heading-local frame, which the paper does unconditionally.
    def rotate(vec: np.ndarray) -> np.ndarray:
        w = heading_inv[:, 0:1]
        u = heading_inv[:, 1:4]
        return (
            vec * (2.0 * w * w - 1.0)
            + 2.0 * w * np.cross(u, vec)
            + 2.0 * u * np.sum(u * vec, axis=1, keepdims=True)
        )

    return np.concatenate(
        [
            qpos[:, 2:3],  # root height. See the terrain note in DESIGN.md section 10.
            rot_features,
            rotate(qvel[:, 0:3]),
            rotate(qvel[:, 3:6]),
            qpos[:, 7:],  # joint angles
            qvel[:, 6:],  # joint velocities
            key_local.reshape(key_local.shape[0], -1),
        ],
        axis=1,
    ).astype(np.float32)


class Discriminator(nn.Module):
    """Scores how human a state transition looks. One scalar logit, no output activation."""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (1024, 512)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.logit = nn.Linear(prev, 1)

        # Output-layer initialisation from the reference implementation.
        nn.init.uniform_(self.logit.weight, -1.0, 1.0)
        nn.init.zeros_(self.logit.bias)

        #: Discriminator observations get their OWN normaliser, separate from the policy's.
        #: Sharing one would couple two very different distributions and is not what any
        #: reference implementation does.
        self.obs_rms = RunningMeanStd((obs_dim,))

    def forward(self, obs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        x = self.obs_rms.normalize(obs) if normalize else obs
        return self.logit(self.trunk(x)).squeeze(-1)

    @torch.no_grad()
    def style_reward(self, obs: torch.Tensor, bounded: bool = True) -> torch.Tensor:
        """Turn the discriminator's verdict into a reward.

        The paper's bounded least-squares form, `max(0, 1 - 0.25*(D-1)^2)`, lies in [0, 1].
        The alternative used by NVIDIA-lineage code, `-log(1 - sigmoid(D))`, is unbounded
        above (capped near 18 in practice) and is far easier to blow up against a
        fixed-scale task reward.
        """
        d = self.forward(obs)
        if bounded:
            return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
        prob = torch.sigmoid(d)
        return -torch.log(torch.clamp(1.0 - prob, min=1e-4)) * 2.0


class MotionTransitionSampler:
    """Draws real (state, next-state) pairs from the reference library.

    Sampled as *transitions* within a single clip, never across a boundary, because a pair
    spanning two unrelated motions is not a real transition and would teach the
    discriminator that teleporting is human.
    """

    def __init__(self, library: MotionLibrary, config: AMPConfig) -> None:
        self.lib = library
        self.cfg = config
        self.n_frames = config.n_obs_frames

        # Precompute per-frame features once. The reference set is fixed, so recomputing
        # them every update would be pure waste.
        self.features = build_amp_features(library.qpos, library.qvel, library.key_local)
        self.per_frame = self.features.shape[1]

        # Frames with enough history and future inside their own clip.
        span = self.n_frames
        starts = []
        for i in range(library.n_clips):
            lo = int(library.clip_start[i])
            hi = lo + int(library.clip_len[i])
            starts.append(np.arange(lo, hi - span))
        self.valid = np.concatenate(starts)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, obs_dim) real AMP observations."""
        base = rng.choice(self.valid, size=n, replace=True)
        offsets = np.arange(self.n_frames)
        idx = base[:, None] + offsets[None, :]
        return self.features[idx.reshape(-1)].reshape(n, -1)


class TransitionReplay:
    """Ring buffer of policy transitions, with subsampled insertion.

    Once full, each new sample is kept only with `replay_keep_prob`. That is what stops a
    large buffer being dominated by the most recent policy: without it the discriminator
    sees only what the policy is doing right now and overfits to it, which is one of the
    documented routes to discriminator collapse.
    """

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator) -> None:
        self.data = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.cursor = 0
        self.rng = rng

    def add(self, obs: np.ndarray, keep_prob: float) -> None:
        if self.size < self.capacity:
            n = min(obs.shape[0], self.capacity - self.size)
            self.data[self.size : self.size + n] = obs[:n]
            self.size += n
            obs = obs[n:]
            if obs.shape[0] == 0:
                return
        if keep_prob < 1.0:
            mask = self.rng.random(obs.shape[0]) < keep_prob
            obs = obs[mask]
        for chunk in np.array_split(obs, max(1, obs.shape[0] // 4096 + 1)):
            if chunk.shape[0] == 0:
                continue
            end = self.cursor + chunk.shape[0]
            if end <= self.capacity:
                self.data[self.cursor : end] = chunk
            else:
                split = self.capacity - self.cursor
                self.data[self.cursor :] = chunk[:split]
                self.data[: end - self.capacity] = chunk[split:]
            self.cursor = end % self.capacity

    def sample(self, n: int) -> np.ndarray | None:
        if self.size == 0:
            return None
        return self.data[self.rng.integers(0, self.size, size=n)]


class AMPTrainer:
    """Owns the discriminator, its optimiser, the replay buffer and the reference sampler."""

    def __init__(
        self,
        library: MotionLibrary,
        n_joints: int,
        n_key_bodies: int,
        device: torch.device,
        config: AMPConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = config or AMPConfig()
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.obs_dim = amp_observation_dim(n_joints, n_key_bodies, self.cfg.n_obs_frames)
        self.discriminator = Discriminator(self.obs_dim, self.cfg.hidden).to(device)
        self.optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.reference = MotionTransitionSampler(library, self.cfg)
        if self.reference.per_frame * self.cfg.n_obs_frames != self.obs_dim:
            raise ValueError(
                f"AMP observation width mismatch: reference gives "
                f"{self.reference.per_frame * self.cfg.n_obs_frames}, model implies "
                f"{self.obs_dim}. The reference clips and the model disagree on joint or "
                f"key body count."
            )
        self.replay = TransitionReplay(self.cfg.replay_size, self.obs_dim, self.rng)

    # ------------------------------------------------------------------ observation

    def observe(self, state: BatchState, history: np.ndarray) -> np.ndarray:
        """Build the current AMP observation from a rolling feature history.

        `history` is (N, n_frames, per_frame), maintained by the caller.
        """
        return history.reshape(history.shape[0], -1)

    def features_now(self, state: BatchState) -> np.ndarray:
        root = state.root_pos[:, None, :]
        c, s = np.cos(-state.heading), np.sin(-state.heading)
        rel = state.key_body_pos - root
        key_local = np.empty_like(rel)
        key_local[:, :, 0] = c[:, None] * rel[:, :, 0] - s[:, None] * rel[:, :, 1]
        key_local[:, :, 1] = s[:, None] * rel[:, :, 0] + c[:, None] * rel[:, :, 1]
        key_local[:, :, 2] = rel[:, :, 2]
        return build_amp_features(state.qpos, state.qvel, key_local)

    # ------------------------------------------------------------------ training

    def update(self, policy_obs: np.ndarray) -> dict[str, float]:
        """One discriminator update against real data and policy data.

        Returns diagnostics. `accuracy` is the one to watch: a discriminator that separates
        real from fake too well saturates the style reward and kills its own gradient. Above
        roughly 90 percent means trouble.
        """
        cfg = self.cfg
        self.replay.add(policy_obs, cfg.replay_keep_prob)

        stats = {k: [] for k in ("loss", "grad_penalty", "acc_real", "acc_fake", "logit_real", "logit_fake")}
        for _ in range(cfg.n_epochs):
            real = torch.from_numpy(self.reference.sample(cfg.batch_size, self.rng)).to(self.device)
            fake_np = self.replay.sample(cfg.batch_size)
            if fake_np is None:
                continue
            fake = torch.from_numpy(fake_np).to(self.device)

            self.discriminator.obs_rms.update(torch.cat([real, fake], dim=0))

            # The gradient penalty is taken with respect to the NORMALISED observation.
            # Getting this backwards scales the penalty by the observation variance, which
            # is a silent and very hard-to-spot error.
            real_norm = self.discriminator.obs_rms.normalize(real).requires_grad_(True)
            d_real = self.discriminator(real_norm, normalize=False)
            d_fake = self.discriminator(fake)

            # Least-squares GAN: regress to +1 on real, -1 on policy.
            loss = ((d_real - 1.0) ** 2).mean() + ((d_fake + 1.0) ** 2).mean()

            grad = torch.autograd.grad(d_real.sum(), real_norm, create_graph=True)[0]
            grad_penalty = grad.pow(2).sum(dim=-1).mean()
            loss = loss + cfg.grad_penalty_coef * grad_penalty

            loss = loss + cfg.logit_reg_coef * self.discriminator.logit.weight.pow(2).sum()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                stats["loss"].append(loss.item())
                stats["grad_penalty"].append(grad_penalty.item())
                stats["acc_real"].append((d_real > 0).float().mean().item())
                stats["acc_fake"].append((d_fake < 0).float().mean().item())
                stats["logit_real"].append(d_real.mean().item())
                stats["logit_fake"].append(d_fake.mean().item())

        out = {f"amp/{k}": float(np.mean(v)) for k, v in stats.items() if v}
        if "amp/acc_real" in out:
            out["amp/accuracy"] = 0.5 * (out["amp/acc_real"] + out["amp/acc_fake"])
        return out

    @torch.no_grad()
    def style_reward(self, amp_obs: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(amp_obs).to(self.device)
        return self.discriminator.style_reward(obs, self.cfg.bounded_reward).cpu().numpy()
