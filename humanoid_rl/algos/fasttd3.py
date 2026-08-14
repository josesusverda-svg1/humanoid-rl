"""FastTD3: off-policy TD3 with a distributional critic, for massively parallel simulation.

Why this exists alongside PPO rather than replacing it. PPO throws every transition away
after a handful of gradient steps, so all 500M environment steps of a run are used once. TD3
keeps them in a replay buffer and reuses them, which is why FastTD3 solves HumanoidBench
tasks in under 3 hours on one A100 where earlier methods took tens of hours or never solved
them (arXiv:2505.22642), and why the same family trains a sim2real G1 in 15 minutes on one
RTX 4090 (arXiv:2512.01996). The lever is sample efficiency, not faster steps, which is
exactly the lever that matters here: physics is 57% of our iteration time.

WHAT CARRIES OVER. Everything except the learning algorithm. `ThreadedVecEnv`, the reward
function, the observations and the humanoid model are untouched, because off-policy RL does
not care where transitions come from. That containment is the entire reason this is worth
trying before a port to Isaac Lab, which would mean rewriting the environment.

Differences from textbook TD3, all four from the paper:

* massively parallel environments feeding one buffer
* very large batches (they use 32,768)
* a distributional critic (C51, Bellemare et al. 2017) instead of a scalar Q
* clipped double Q on the DISTRIBUTIONS, taking whichever critic's mean is lower

Deliberately NOT included: layer normalisation. The FastTD3 authors tested it and report it
degraded performance. The later 15-minute paper says layer norm was necessary for stability
in its refinement of FastSAC. They disagree, so it is a flag defaulting to off (the FastTD3
finding, since this is a FastTD3 implementation) rather than a silent choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastTD3Config:
    """Defaults are the FastTD3 authors' IsaacLab settings unless noted.

    The value support is the exception and MUST be set from the reward scale of this project,
    not copied. Their IsaacLab preset is v_min/v_max = -10/+10 because IsaacLab rewards are
    tiny; ours run about 2.8 per step, so the discounted return at gamma 0.99 approaches
    2.8/0.01 = 280. A support of +/-10 would put every good state off the top of the scale,
    the critic would saturate, and every state would look equally excellent. See
    `support_covers_return_range` in humanoid_rl/oracle/invariants.py, which fails the run
    before it starts if these disagree with gamma and the reward weights.
    """

    #: C51 support, sized from THIS reward function, not copied from a reference.
    #:
    #: First set to [-50, 350] from the measured per-step reward of 3.27, and the Oracle
    #: rejected it before the first run: the reward WEIGHTS allow 6.4 per step, so the
    #: reachable discounted return is 640, not 327. A support must cover the best case a
    #: policy could reach, not the mean of the policy that happens to exist today, or
    #: improvement past the current level is invisible to the critic.
    v_min: float = -100.0
    v_max: float = 800.0
    #: 401, not the authors' 251. Their IsaacLab preset spans [-10, 10], so 251 atoms give
    #: 0.08 of resolution. Our range is 900 wide, and 251 atoms would put 3.6 of return in
    #: one atom, over half a single step's best reward: a whole step of improvement could
    #: then fail to move the target at all.
    num_atoms: int = 401

    gamma: float = 0.99
    #: 0.1, which is 100x the usual TD3 value of 0.005. Large batches and many parallel
    #: environments make the critic estimate far less noisy, so the target can chase it.
    tau: float = 0.1

    actor_hidden: tuple[int, ...] = (512, 256, 128)
    critic_hidden: tuple[int, ...] = (1024, 512, 256)
    use_layer_norm: bool = False

    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    #: 8,192, not the authors' 32,768, and the reason is our chip. Measured by
    #: scripts/bench_fasttd3.py on this M3 Max:
    #:
    #:     batch     ms/update   us/sample   end-to-end env steps/s
    #:      4,096        19.9       4.86            46,083
    #:      8,192        36.9       4.51            33,321
    #:     32,768       147.0       4.48            11,941
    #:     65,536       342.7       5.23             5,577
    #:
    #: The us/sample column is FLAT. MPS is already saturated at 4,096, so cost scales
    #: linearly with batch and a large batch buys nothing back. On an A100 that column falls
    #: steeply, which is exactly why the authors can afford 32,768; the number does not
    #: transfer across that difference in hardware. Copying it unchecked is the same mistake
    #: that gave this project an 8-second episode.
    #:
    #: What IS fixed is the replay ratio (samples processed per environment step). At ratio
    #: 16 every split costs the same wall clock, so batch size becomes a free choice about
    #: gradient quality: 8,192 x 8 updates and 32,768 x 2 both take ~344 ms per iteration,
    #: but the former takes four times as many optimiser steps. 8K is also exactly the
    #: ceiling the 15-minute paper reports as "consistently improving performance".
    batch_size: int = 8192
    #: Gradient steps per environment step, giving a replay ratio of 8192*8/4096 = 16
    #: samples per environment step. PPO on this project runs at 5. The paper's range is
    #: "2 to 8 updates per 128 to 4096 parallel environment steps".
    num_updates: int = 8
    #: Delayed policy updates, the second D in TD3.
    policy_frequency: int = 2

    #: Target-policy smoothing: noise added to the target action so the critic cannot
    #: exploit a sharp peak.
    policy_noise: float = 0.001
    noise_clip: float = 0.5
    #: Exploration noise. Each environment draws its own std in this range and keeps it,
    #: so the batch always contains both careful and reckless behaviour.
    std_min: float = 0.001
    std_max: float = 0.4

    #: Transitions kept per environment. Total buffer is this times num_envs; at 4096 envs
    #: and obs_dim 108, 1024 per env is 4.2M transitions and about 4.1 GB.
    buffer_size_per_env: int = 1024
    #: Environment steps collected before any gradient step, so the first batch is not drawn
    #: from a nearly empty buffer.
    learning_starts: int = 10


def _mlp(inp: int, hidden: tuple[int, ...], out: int, layer_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = inp
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy. tanh output, so actions are always in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: FastTD3Config) -> None:
        super().__init__()
        self.net = _mlp(obs_dim, cfg.actor_hidden, act_dim, cfg.use_layer_norm)
        # Small final layer, so the initial policy is near zero action (the standing pose)
        # rather than a random extreme that terminates the episode instantly.
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class DistributionalCritic(nn.Module):
    """Twin critics, each predicting a distribution over returns rather than a mean.

    A scalar critic reports one number and cannot express "usually 300, occasionally 0
    because it falls". The categorical critic puts probability mass on a fixed grid of
    return values, so bimodality survives, and TD3's overestimation bias has much less room
    to work with. This is the single change the FastTD3 paper credits most.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: FastTD3Config) -> None:
        super().__init__()
        self.num_atoms = cfg.num_atoms
        self.q1 = _mlp(obs_dim + act_dim, cfg.critic_hidden, cfg.num_atoms, cfg.use_layer_norm)
        self.q2 = _mlp(obs_dim + act_dim, cfg.critic_hidden, cfg.num_atoms, cfg.use_layer_norm)
        self.register_buffer(
            "support", torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms)
        )

    def logits(self, obs: torch.Tensor, act: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def expected(self, obs: torch.Tensor, act: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean of each critic's return distribution, which is the ordinary Q value."""
        l1, l2 = self.logits(obs, act)
        return (
            (F.softmax(l1, dim=-1) * self.support).sum(-1),
            (F.softmax(l2, dim=-1) * self.support).sum(-1),
        )


def project_distribution(
    rewards: torch.Tensor,
    keep_going: torch.Tensor,
    next_probs: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Categorical projection: move a shifted, scaled distribution back onto the grid.

    The Bellman update maps each support point z to r + gamma*z, which lands between grid
    points. The mass at each new location is split between the two neighbouring atoms in
    proportion to how close it is to each. This is the standard C51 projection (Bellemare
    et al. 2017, Algorithm 1) and it is the only fiddly part of the method.

    `keep_going` is 1.0 where the episode continued and 0.0 where it TERMINATED. It must not
    be zero for a time-limit truncation: the humanoid that is walking fine when the clock
    runs out has a real future worth bootstrapping from, and treating that as death teaches
    it that surviving 20 seconds is worthless.
    """
    batch, num_atoms = next_probs.shape
    v_min, v_max = float(support[0]), float(support[-1])
    delta = (v_max - v_min) / (num_atoms - 1)

    target_z = rewards[:, None] + gamma * keep_going[:, None] * support[None, :]
    target_z = target_z.clamp(v_min, v_max)
    b = (target_z - v_min) / delta
    lower = b.floor().clamp(0, num_atoms - 1)
    upper = b.ceil().clamp(0, num_atoms - 1)
    # When b lands exactly on an atom, floor == ceil and the two weights below would both be
    # zero, silently discarding that mass. Nudge them apart in that case.
    lower = torch.where((upper > 0) & (lower == upper), lower - 1, lower)
    upper = torch.where((lower < num_atoms - 1) & (lower == upper), upper + 1, upper)

    projected = torch.zeros_like(next_probs)
    projected.scatter_add_(1, lower.long(), next_probs * (upper - b))
    projected.scatter_add_(1, upper.long(), next_probs * (b - lower))
    return projected


class ReplayBuffer:
    """Flat ring buffer, kept on the training device.

    Stored as one contiguous block per field rather than a list of transitions, because the
    only access pattern is "gather 32,768 random indices" and that wants a single indexing
    operation, not a Python loop.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        #: 1.0 where the episode continued OR was truncated by the time limit, 0.0 only where
        #: it genuinely terminated. See project_distribution for why the distinction matters.
        self.keep_going = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.position = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def add(self, obs, actions, rewards, next_obs, keep_going) -> None:
        """Insert one step from every environment at once."""
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.position) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.keep_going[idx] = keep_going
        self.position = int((self.position + n) % self.capacity)
        self.full = self.full or self.position < n

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, len(self), (batch_size,), device=self.device)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.keep_going[idx])


class FastTD3:
    """The agent: two critics, a delayed actor, target copies of both, and a buffer."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: FastTD3Config,
                 num_envs: int, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim, cfg).to(device)
        self.critic = DistributionalCritic(obs_dim, act_dim, cfg).to(device)
        self.actor_target = Actor(obs_dim, act_dim, cfg).to(device)
        self.critic_target = DistributionalCritic(obs_dim, act_dim, cfg).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in list(self.actor_target.parameters()) + list(self.critic_target.parameters()):
            p.requires_grad_(False)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        self.buffer = ReplayBuffer(
            cfg.buffer_size_per_env * num_envs, obs_dim, act_dim, device
        )
        # One fixed exploration std per environment, drawn once. A single shared std makes
        # every environment equally reckless at the same moment; a spread means the batch
        # always contains some careful behaviour to learn the value of.
        self.explore_std = torch.empty(num_envs, 1, device=device).uniform_(
            cfg.std_min, cfg.std_max
        )
        self.updates = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        action = self.actor(obs)
        if explore:
            action = action + torch.randn_like(action) * self.explore_std
        return action.clamp(-1.0, 1.0)

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        obs, actions, rewards, next_obs, keep_going = self.buffer.sample(cfg.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)
            l1, l2 = self.critic_target.logits(next_obs, next_actions)
            p1, p2 = F.softmax(l1, dim=-1), F.softmax(l2, dim=-1)
            support = self.critic_target.support
            # Clipped double Q, applied per sample: keep the WHOLE distribution belonging to
            # whichever critic is more pessimistic in expectation. Averaging the two
            # distributions instead would smear a bimodal return into a unimodal one and
            # throw away the reason for using a distributional critic at all.
            q1, q2 = (p1 * support).sum(-1), (p2 * support).sum(-1)
            chosen = torch.where((q1 <= q2)[:, None], p1, p2)
            target = project_distribution(rewards, keep_going, chosen, support, cfg.gamma)

        logits1, logits2 = self.critic.logits(obs, actions)
        # Cross-entropy against the projected target, which is the categorical analogue of
        # the squared TD error.
        critic_loss = (
            -(target * F.log_softmax(logits1, dim=-1)).sum(-1).mean()
            - (target * F.log_softmax(logits2, dim=-1)).sum(-1).mean()
        )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        metrics = {"critic_loss": float(critic_loss.detach())}
        self.updates += 1

        if self.updates % cfg.policy_frequency == 0:
            q1_pred, _ = self.critic.expected(obs, self.actor(obs))
            actor_loss = -q1_pred.mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            metrics["actor_loss"] = float(actor_loss.detach())
            metrics["q_value"] = float(q1_pred.mean().detach())
            self._soft_update()
        return metrics

    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for net, target in ((self.actor, self.actor_target),
                            (self.critic, self.critic_target)):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(p, alpha=tau)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.updates = int(state.get("updates", 0))


def suggested_support(reward_per_step: float, gamma: float,
                      headroom: float = 1.25) -> tuple[float, float]:
    """A value support that actually covers the returns this reward function can produce.

    The failure this prevents is quiet and total. If v_max sits below the achievable
    discounted return, every good state piles its mass on the top atom, the critic reports
    the same value for "walking beautifully" and "barely upright", and the actor gets no
    gradient to tell them apart. Nothing in the loss curve looks wrong.
    """
    ceiling = reward_per_step / max(1.0 - gamma, 1e-9)
    return -0.15 * ceiling * headroom, ceiling * headroom
