"""Actor-critic networks and observation normalisation.

Deliberately small networks (two hidden layers of 512). Locomotion policies do not need
depth, and Phase 0 measured that the physics simulation, not the network, dominates
runtime. Making these larger would slow training without improving the gait.

Terminology:
* Actor: the network that outputs actions. Here it outputs the *mean* of a Gaussian
  distribution over actions, and exploration comes from sampling around that mean.
* Critic: the network that estimates how much future reward a state is worth. Used to
  reduce the variance of the policy gradient, never to pick actions.
* Observation normalisation: rescaling each observation dimension to roughly zero mean and
  unit variance. Without it, dimensions with large numeric ranges (joint velocities) swamp
  small ones (gravity direction), and training is far less stable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


class RunningMeanStd(nn.Module):
    """Tracks a running mean and variance per observation dimension.

    Uses Chan et al.'s parallel variance algorithm, which merges the statistics of a new
    batch into the running estimate without storing history and without the numerical
    drift of a naive incremental update.

    Registered as buffers rather than plain attributes so they are saved in the checkpoint
    and restored on resume. Getting this wrong is a classic bug: the policy resumes with
    the right weights but the wrong input scaling, and performance collapses in a way that
    looks like the checkpoint itself is corrupt.
    """

    #: NO ceiling on the sample count, and the absence is load-bearing.
    #:
    #: A cap of 1e6 was added here once, reasoning that a warm start onto a shifted
    #: distribution should adapt its statistics quickly. Measured consequence: a warm start
    #: from a 49M-sample run had its per-batch drift accelerated 50x, the statistics
    #: re-converged to the new task within ~40 iterations, and the policy's inputs were
    #: re-centred by up to 2.2 standard deviations under weights trained against the old
    #: values. The warm-started walker scored 12% falls at evaluation 50 and was at 100%
    #: by evaluation 100, destroyed by its own normaliser. The slow drift of an uncapped
    #: count is what lets policy and statistics co-evolve gently, which is the entire
    #: reason warm starts survive. If a future change genuinely needs fresh statistics,
    #: that is a deliberate reset, not a cap.
    COUNT_MAX: float = float("inf")

    #: Floor on the per-channel variance. A channel that is constant (a newly added
    #: observation slot not yet populated, or a sensor that has not moved) drives the
    #: variance toward zero, and `normalize` then divides by sqrt(var + 1e-8) ~= 1e-4, a
    #: gain of roughly 9,500. The first time that channel carries a real value it explodes
    #: and slams into the clip bound. The floor costs nothing on channels that vary.
    VAR_MIN: float = 1.0e-4

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total)

        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(m2 / total, min=self.VAR_MIN))
        self.count.copy_(torch.clamp(total, max=self.COUNT_MAX))

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalise and clip. Clipping bounds the damage from a rare extreme observation."""
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip)


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), act_cls()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Sequential, final_gain: float) -> None:
    """Orthogonal initialisation, with a deliberately tiny gain on the output layer.

    Standard practice for on-policy RL. A small final gain makes the initial policy output
    near-zero actions, so the humanoid starts by standing roughly still instead of
    thrashing. Starting with large random torques wastes the early episodes and can put
    the policy in a bad basin it never leaves.
    """
    linears = [m for m in module if isinstance(m, nn.Linear)]
    for layer in linears[:-1]:
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(linears[-1].weight, gain=final_gain)
    nn.init.zeros_(linears[-1].bias)


class ActorCritic(nn.Module):
    """Gaussian policy plus value function, with separate (non-shared) trunks.

    Separate trunks rather than a shared body: for locomotion this is the more robust
    choice, because the value function needs to track a fast-changing reward scale while
    the policy should change slowly. Sharing makes the value loss perturb the policy
    through the shared weights, which shows up as sudden unexplained gait regressions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        actor_hidden: tuple[int, ...] = (512, 512),
        critic_hidden: tuple[int, ...] = (512, 512),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        log_std_max: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}, expected {list(_ACTIVATIONS)}")

        self.actor = _mlp(obs_dim, actor_hidden, act_dim, activation)
        self.critic = _mlp(obs_dim, critic_hidden, 1, activation)
        _orthogonal_init(self.actor, final_gain=0.01)
        _orthogonal_init(self.critic, final_gain=1.0)

        # State-independent exploration noise, learned as a free parameter. Standard for
        # locomotion: a state-dependent std tends to collapse early, killing exploration
        # before the policy has found a gait.
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_noise_std))))
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.act_dim = act_dim
        # Ceiling on exploration noise. exp(0) = 1.0 is already large against an action
        # range of [-1, 1]; letting it grow deepens the saturation regime that the
        # bounds loss in ppo.py exists to prevent.
        self.log_std_max = float(log_std_max)

    # ------------------------------------------------------------------ helpers

    def _distribution(self, obs_norm: torch.Tensor) -> Normal:
        mean = self.actor(obs_norm)
        # Bound the std so exploration can neither explode nor vanish entirely.
        # NOTE: clamping here is a GRADIENT SINK, not a bound. torch.clamp passes no
        # gradient outside its range, so any component that ends up beyond the limit stops
        # receiving updates entirely and is frozen for the rest of training. That is not
        # hypothetical: every one of these 28 components sat at +0.0037 against a ceiling of
        # 0.0 for 610 iterations, inherited through a warm start, reading as a perfectly
        # steady std of 1.0000 in the logs while being simply dead. The exploration noise
        # cost 49% of the per-step reward the whole time.
        #
        # The clamp is kept for numerical safety, but the parameter is now also clamped IN
        # PLACE after each optimiser step (see PPO.update), so it can never drift outside
        # the range in the first place and always has live gradient.
        std = torch.exp(self.log_std.clamp(-5.0, self.log_std_max)).expand_as(mean)
        return Normal(mean, std)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(obs)

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, noise_scale: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for rollout collection.

        `noise_scale` is a per-environment multiplier on the exploration noise, shape (N,)
        or (N, 1). It exists because a policy whose objective only ever sees NOISY rollouts
        will happily learn to use the noise itself as a stabiliser, and this project's did,
        twice: first as 62 Hz tremor, then, after an 8 Hz action filter removed that
        channel, again as sub-cutoff dither. Deterministic evaluation collapsed both times
        while training looked excellent. Collecting a fraction of environments at near-zero
        noise puts deterministic-quality trajectories INTO the batch, so the mean policy's
        failures finally generate gradient.

        The log probability is deliberately computed under the policy's own distribution,
        not the scaled one. That makes the scaled environments mildly off-policy (their
        actions are drawn from a narrower Gaussian than the density used in the ratio),
        which is the standard, bounded bias of this technique; the PPO clip contains it.
        The alternative, correcting the density per environment, would make the ratio
        exactly 1 for near-deterministic samples and remove the very gradient this exists
        to create.

        Returns:
            action, log_prob, value. The action is unclipped; the environment clips it, so
            that the log probability stays consistent with what was actually sampled.
        """
        obs_norm = self.normalize_obs(obs)
        dist = self._distribution(obs_norm)
        if noise_scale is None:
            action = dist.sample()
        else:
            scale = noise_scale.reshape(-1, 1).to(obs.device)
            action = dist.mean + torch.randn_like(dist.mean) * dist.stddev * scale
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs_norm).squeeze(-1)
        return action, log_prob, value

    @torch.no_grad()
    @torch.no_grad()
    def clamp_log_std(self) -> None:
        """Pull log_std back inside its range, in place.

        Called after every optimiser step and after loading weights. Without it a value
        outside the range is a permanently frozen parameter, because the clamp in the
        forward pass zeroes its gradient.
        """
        self.log_std.clamp_(-5.0, self.log_std_max)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """The distribution mean, with no exploration noise. Used for evaluation and video."""
        return self.actor(self.normalize_obs(obs))

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.normalize_obs(obs)).squeeze(-1)

    # ------------------------------------------------------------------ training

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the current policy, for the PPO update.

        Returns:
            log_prob, entropy, value, action_mean.
        """
        obs_norm = self.normalize_obs(obs)
        dist = self._distribution(obs_norm)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs_norm).squeeze(-1)
        return log_prob, entropy, value, dist.mean

    @property
    def action_std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-5.0, self.log_std_max))
