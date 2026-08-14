"""Proximal Policy Optimization, with the details that matter for locomotion.

PPO is the standard on-policy algorithm for legged and humanoid locomotion. The core idea
is simple: improve the policy using collected experience, but clip the update so the new
policy never moves too far from the one that gathered the data, because that data stops
being valid once the policy changes much.

Three implementation details here are worth calling out, because getting any of them wrong
produces a policy that trains but never walks well:

1. **Termination and truncation are handled differently.** Falling over means the future is
   genuinely worth zero. Hitting the episode time limit does not, so the value function
   must be bootstrapped from the final observation. Treating a time limit as a real
   termination teaches the humanoid that surviving long is punished, which is a
   surprisingly common and very destructive bug.

2. **Adaptive learning rate driven by KL divergence.** Rather than a fixed schedule, the
   learning rate is adjusted every update to hold the policy change near a target. This is
   what the ETH legged-robot line of work uses and it is far more robust across reward
   scales than any hand-tuned decay.

3. **Observation normalisation is frozen for a whole iteration.** Statistics are updated
   once, after the update completes. If the normaliser shifted between collecting an
   action and re-evaluating it, the importance ratio would start away from 1 even for an
   unchanged policy, which silently corrupts the clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from humanoid_rl.algos.networks import ActorCritic


@dataclass
class PPOConfig:
    """Hyperparameters. Mirrored one-for-one in the YAML config."""

    # --- rollout ---
    horizon: int = 24  # steps collected per environment per iteration
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE bias-variance tradeoff

    # --- optimisation ---
    learning_rate: float = 1.0e-3
    num_epochs: int = 5
    num_minibatches: int = 4
    clip_ratio: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    clip_value_loss: bool = True
    #: Penalty on the policy MEAN straying outside the valid action range.
    #:
    #: Without it the environment clips actions and nothing pushes back, so driving the
    #: mean far out of bounds becomes a free "maximum effort" strategy. Measured on a run
    #: without this term: mean |action| reached 2.3 with a maximum of 22.8, and 71% of
    #: components sat outside [-1, 1]. Training looked healthy because exploration noise
    #: still modulated the saturated mean, but the deterministic evaluation policy became a
    #: near-constant bang-bang controller and its fall rate went from 30% to 100% while the
    #: training curve kept rising. The policy had learned to use its own noise as part of
    #: the controller.
    #:
    #: 10.0 matches the value IsaacGymEnvs uses for the same purpose.
    bounds_loss_coef: float = 10.0
    #: Weight on the mirror-symmetry loss, penalising ||pi(s) - M(pi(M(s)))||^2.
    #:
    #: A bilaterally symmetric body walking straight ahead should look the same in a
    #: mirror, but nothing in standard RL says so and policies routinely converge on a
    #: one-sided gait. Measured on this project's AMP policy: left foot in stance 0.58 of
    #: the time against the right's 0.30, and the right leg airborne 2.9x longer. Every
    #: aggregate metric was blind to it; a human watching a video caught it.
    #:
    #: 0.0 disables. 1.0 to 4.0 is the useful range.
    #:
    #: DEPRECATED in favour of `symmetry_augment` below, on evidence. At 2.0 this term
    #: produced a policy that satisfied it perfectly and did not walk: a 61 cm wide
    #: two-footed brace, both feet loaded 90% of the time, 6 cm of travel per foot strike,
    #: dragged forward by ground slip. The reason is structural rather than a bad weight.
    #: The penalty is exactly zero for any policy whose output does not depend on its input,
    #: so "hold a symmetric pose" is a global optimum of this term, and it is far easier to
    #: reach than walking. Prefer data augmentation, which has no such solution.
    symmetry_loss_coef: float = 0.0
    #: Add the left-right reflection of every transition to the PPO batch.
    #:
    #: States the same belief as the loss above, that a mirrored state deserves a mirrored
    #: action, but enforces it through the data rather than through a penalty. The reflected
    #: transition is a genuine transition for a bilaterally symmetric body, so this is
    #: unbiased, and it doubles the effective batch. Crucially there is nothing to game: a
    #: constant symmetric policy gains no return on either copy, so the brace is not a
    #: solution.
    #:
    #: The reflected sample reuses the original's stored log probability, which is exact
    #: when the policy is equivariant and approximate before then. The error shrinks as the
    #: policy becomes symmetric, which is the thing being optimised.
    symmetry_augment: bool = False
    #: Fraction of environments that collect rollouts at `exploit_noise_scale` times the
    #: exploration noise instead of the full amount. Their trajectories show the batch what
    #: the near-deterministic policy actually does, which is the only mechanism that stops
    #: PPO optimising a noise-stabilised gait the deterministic policy cannot reproduce.
    #: Found the hard way: without it, deterministic evaluation collapsed to 100% falls
    #: twice while noisy training returns looked excellent.
    exploit_env_fraction: float = 0.25
    exploit_noise_scale: float = 0.1
    #: Updates at the start of training during which ONLY the critic learns. The policy,
    #: entropy, bounds and symmetry terms are all skipped, and so is the KL-adaptive
    #: learning rate (KL is meaningless while the actor is frozen).
    #:
    #: Exists for warm starts. A warm-started run pairs a policy that already performs with
    #: a critic whose value estimates describe a DIFFERENT reward function, so the first
    #: advantages are garbage with conviction. Measured: the first update after warm start
    #: reached KL 1.66 against a target of 0.01, one hundred sixty times over, teleporting
    #: the policy off its stability manifold before anything else had a say. The critic
    #: re-fits within roughly ten updates; this makes the policy wait for it.
    critic_warmup_updates: int = 0

    #: Upper limit on log std. exp(0) = 1.0, which is already large relative to an action
    #: range of [-1, 1]. Letting it grow further just deepens the saturation regime.
    log_std_max: float = 0.0

    # --- adaptive learning rate ---
    adaptive_lr: bool = True
    desired_kl: float = 0.01
    lr_min: float = 1.0e-5
    lr_max: float = 1.0e-2

    # --- normalisation ---
    normalize_advantage: bool = True
    obs_clip: float = 10.0


class RolloutBuffer:
    """Fixed-size on-policy storage, shaped (horizon, num_envs, ...).

    Kept on the training device. At 24 x 4096 x 75 floats this is about 29 MB, which on a
    96 GB unified memory machine is free, and keeping it resident avoids a host transfer
    on every minibatch.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.horizon, self.num_envs = horizon, num_envs
        self.device = device
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)  # noqa: E731

        self.obs = z(horizon, num_envs, obs_dim)
        self.actions = z(horizon, num_envs, act_dim)
        self.log_probs = z(horizon, num_envs)
        self.values = z(horizon, num_envs)
        self.rewards = z(horizon, num_envs)
        # Value of the final observation, non-zero only where an episode was truncated.
        self.truncated_values = z(horizon, num_envs)
        self.terminated = torch.zeros(horizon, num_envs, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(horizon, num_envs, dtype=torch.bool, device=device)

        self.advantages = z(horizon, num_envs)
        self.returns = z(horizon, num_envs)
        self._step = 0

    def reset(self) -> None:
        self._step = 0
        self.truncated_values.zero_()

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        t = self._step
        if t >= self.horizon:
            raise RuntimeError("RolloutBuffer overflow: call reset() between iterations")
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self._step += 1

    def set_truncated_values(self, t: int, idx: torch.Tensor, values: torch.Tensor) -> None:
        self.truncated_values[t, idx] = values

    @torch.no_grad()
    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        """Generalised Advantage Estimation, walking backwards through the rollout.

        The subtlety is what "the value of the next state" means at an episode boundary,
        because autoreset means the stored next observation belongs to a *new* episode:

            terminated -> 0, the future really is worthless
            truncated  -> V(final observation), the episode was cut off artificially
            otherwise  -> the stored value of the next step
        """
        adv = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.horizon)):
            next_values = last_values if t == self.horizon - 1 else self.values[t + 1]
            # Where the episode ended, replace the next-step value appropriately.
            next_values = torch.where(self.truncated[t], self.truncated_values[t], next_values)
            next_values = torch.where(
                self.terminated[t], torch.zeros_like(next_values), next_values
            )

            delta = self.rewards[t] + gamma * next_values - self.values[t]
            # An episode boundary of either kind breaks the GAE recursion, because the
            # advantage of a later step in a different episode is unrelated.
            not_done = (~(self.terminated[t] | self.truncated[t])).float()
            adv = delta + gamma * lam * not_done * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def minibatches(self, num_minibatches: int, generator: torch.Generator):
        """Yield shuffled flat minibatches over the whole rollout."""
        batch_size = self.horizon * self.num_envs
        if batch_size % num_minibatches != 0:
            raise ValueError(
                f"batch size {batch_size} not divisible by num_minibatches {num_minibatches}"
            )
        mb_size = batch_size // num_minibatches
        perm = torch.randperm(batch_size, device=self.device, generator=generator)

        flat_obs = self.obs.reshape(batch_size, -1)
        flat_act = self.actions.reshape(batch_size, -1)
        flat_lp = self.log_probs.reshape(batch_size)
        flat_val = self.values.reshape(batch_size)
        flat_adv = self.advantages.reshape(batch_size)
        flat_ret = self.returns.reshape(batch_size)

        for i in range(num_minibatches):
            idx = perm[i * mb_size : (i + 1) * mb_size]
            yield flat_obs[idx], flat_act[idx], flat_lp[idx], flat_val[idx], flat_adv[idx], flat_ret[idx]


class PPO:
    """The PPO update. Owns the optimiser and the adaptive learning rate."""

    def __init__(
        self,
        policy: ActorCritic,
        config: PPOConfig,
        device: torch.device,
        seed: int = 0,
        mirror=None,
        task=None,
    ) -> None:
        self.policy = policy
        # Mirror machinery for the symmetry loss. Held as tensors so the loss stays on the
        # GPU with the rest of the update.
        self.mirror = mirror
        self.task = task
        if mirror is not None and (config.symmetry_loss_coef > 0.0 or config.symmetry_augment):
            self._obs_perm = torch.as_tensor(mirror.obs_perm, dtype=torch.long, device=device)
            self._obs_sign = torch.as_tensor(mirror.obs_sign, dtype=torch.float32, device=device)
            self._act_perm = torch.as_tensor(mirror.action_perm, dtype=torch.long, device=device)
            self._act_sign = torch.as_tensor(mirror.action_sign, dtype=torch.float32, device=device)
            self._obs_width = mirror.obs_width
        self.cfg = config
        self.device = device
        self.learning_rate = config.learning_rate
        self._updates_done = 0
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def _layer_snapshot(self) -> dict[str, torch.Tensor]:
        """A copy of every weight matrix, to diff against after the update.

        Reported so the training loop is inspectable rather than a black box: "the policy
        changed" is not a useful statement, but "layer 1 of the actor moved 0.4% while the
        output layer moved 0.02%" says which part of the network is actually learning right
        now. Weights only, not biases, and one clone per layer per iteration, which is
        negligible next to a rollout.
        """
        return {
            name: param.detach().clone()
            for name, param in self.policy.named_parameters()
            if param.dim() == 2
        }

    def _layer_change(self, before: dict[str, torch.Tensor]) -> dict[str, float]:
        """Relative movement of each layer over one update: ||dW|| / ||W||.

        Relative rather than absolute, because layers differ in size by an order of
        magnitude and the raw norms are not comparable. A relative number answers the
        question that matters: which part of the network is being rewritten.
        """
        out: dict[str, float] = {}
        after = dict(self.policy.named_parameters())
        for name, old in before.items():
            new = after[name].detach()
            denom = float(old.norm()) or 1.0
            out[f"weight_change/{name}"] = float((new - old).norm()) / denom
        return out

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
            "bounds_loss": [],
            "action_mean_abs": [],
            "action_oob_frac": [],
            "symmetry_loss": [],
        }

        advantages = buffer.advantages
        before_update = self._layer_snapshot()

        if cfg.normalize_advantage:
            # Normalised over the whole rollout, not per minibatch, so every minibatch
            # sees the same scale and the effective step size stays consistent.
            adv_mean, adv_std = advantages.mean(), advantages.std()
            buffer.advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        for _ in range(cfg.num_epochs):
            for minibatch in buffer.minibatches(cfg.num_minibatches, self.generator):
                if cfg.symmetry_augment and self.mirror is not None:
                    minibatch = self._augment_minibatch(minibatch)
                obs_b, act_b, old_lp_b, old_val_b, adv_b, ret_b = minibatch

                log_prob, entropy, value, action_mean = self.policy.evaluate(obs_b, act_b)
                warming_up = self._updates_done < cfg.critic_warmup_updates

                # --- clipped policy objective ---
                ratio = torch.exp(log_prob - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value loss, optionally clipped the same way ---
                if cfg.clip_value_loss:
                    value_clipped = old_val_b + (value - old_val_b).clamp(
                        -cfg.clip_ratio, cfg.clip_ratio
                    )
                    value_loss = torch.max(
                        (value - ret_b).pow(2), (value_clipped - ret_b).pow(2)
                    ).mean()
                else:
                    value_loss = (value - ret_b).pow(2).mean()

                entropy_mean = entropy.mean()

                # Soft bound on the action mean. Quadratic only outside [-1, 1], so it is
                # exactly zero for a well-behaved policy and costs nothing until the mean
                # starts escaping the range the environment can actually apply.
                over = (action_mean - 1.0).clamp(min=0.0)
                under = (action_mean + 1.0).clamp(max=0.0)
                bounds_loss = (over.pow(2) + under.pow(2)).sum(-1).mean()

                symmetry_loss = torch.zeros((), device=obs_b.device)
                if cfg.symmetry_loss_coef > 0.0 and self.mirror is not None:
                    symmetry_loss = self._symmetry_loss(obs_b, action_mean)

                if warming_up:
                    # Critic only. The actor receives no gradient at all, so the policy that
                    # collected these rollouts is exactly the policy that exits the warm-up.
                    loss = cfg.value_loss_coef * value_loss
                else:
                    loss = (
                        policy_loss
                        + cfg.value_loss_coef * value_loss
                        - cfg.entropy_coef * entropy_mean
                        + cfg.bounds_loss_coef * bounds_loss
                        + cfg.symmetry_loss_coef * symmetry_loss
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                # Keep log_std inside its range so it always has live gradient. The clamp in
                # the forward pass bounds the value but kills the gradient of anything that
                # escapes, which froze exploration for an entire run.
                self.policy.clamp_log_std()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator. The naive (old_lp - lp).mean()
                    # is unbiased but noisy enough to make the adaptive LR oscillate.
                    log_ratio = log_prob - old_lp_b
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()

                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy_mean.item())
                    stats["approx_kl"].append(approx_kl.item())
                    stats["clip_fraction"].append(clip_frac.item())
                    stats["grad_norm"].append(grad_norm.item())
                    stats["bounds_loss"].append(bounds_loss.item())
                    # The diagnostic that would have caught the saturation collapse
                    # immediately: how far outside [-1, 1] the policy mean actually sits.
                    stats["action_mean_abs"].append(action_mean.abs().mean().item())
                    stats["action_oob_frac"].append(
                        (action_mean.abs() > 1.0).float().mean().item()
                    )
                    stats["symmetry_loss"].append(float(symmetry_loss))

            if cfg.adaptive_lr and self._updates_done >= cfg.critic_warmup_updates:
                self._adapt_learning_rate(float(np.mean(stats["approx_kl"][-cfg.num_minibatches :])))

        self._updates_done += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["critic_warmup"] = float(self._updates_done <= cfg.critic_warmup_updates)
        out["learning_rate"] = self.learning_rate
        # NOT explained variance, despite looking exactly like it. `compute_returns` sets
        # returns = advantages + values, so (returns - values) IS the advantage tensor and
        # this reduces to 1 - Var(A)/Var(V+A). It reports how small advantages are relative
        # to the spread of V across states, and reads +0.95 on data whose honest
        # Monte-Carlo explained variance is +0.72. It CANNOT detect a miscalibrated critic.
        #
        # Kept under an honest name because the quantity is still useful: it falls when
        # advantages grow relative to the value spread, which is what a policy finding new
        # behaviour looks like. A real explained variance needs Monte-Carlo returns over
        # completed episodes, which this buffer does not hold; `scripts/critic_report.py`
        # measures that offline. With horizon 24 and gamma 0.99, gamma^24 = 0.79 of the
        # target's discounted mass is the bootstrap V(s) itself, so the two cannot agree.
        with torch.no_grad():
            values = buffer.values.reshape(-1)
            advantage = buffer.advantages.reshape(-1)
            out["advantage_share"] = float(
                advantage.var() / torch.clamp((values + advantage).var(), min=1e-8)
            )
        out["action_std"] = self.policy.action_std.mean().item()
        out.update(self._layer_change(before_update))
        return out

    def _mirror_obs(self, obs: torch.Tensor) -> torch.Tensor | None:
        """Reflect an observation batch left-to-right, or None if the task cannot.

        The proprioceptive block permutes by the model-derived spec. The task block is
        task-specific (a velocity command mirrors by negating its lateral and yaw parts, a
        waypoint by reflecting its offset), so it is delegated. A task that returns None is
        declaring it does not know how to mirror itself, and the caller must then skip
        mirroring entirely rather than apply a transform that is silently wrong.
        """
        mirrored = obs.clone()
        mirrored[:, : self._obs_width] = obs[:, self._obs_perm] * self._obs_sign
        if obs.shape[1] > self._obs_width:
            task_block = self.task.mirror_task_obs(obs[:, self._obs_width :])
            if task_block is None:
                return None
            mirrored[:, self._obs_width :] = task_block
        return mirrored

    def _augment_minibatch(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Append the left-right reflection of every sample in a minibatch.

        Observations and actions are reflected; advantage, return and value are unchanged,
        because reflecting a transition of a symmetric body does not change what it was
        worth. The stored log probability is reused for the reflected action, which is exact
        for an equivariant policy and approximate before then.
        """
        obs, act, old_lp, old_val, adv, ret = batch
        mirrored_obs = self._mirror_obs(obs)
        if mirrored_obs is None:
            return batch
        mirrored_act = act[:, self._act_perm] * self._act_sign
        return (
            torch.cat([obs, mirrored_obs]),
            torch.cat([act, mirrored_act]),
            torch.cat([old_lp, old_lp]),
            torch.cat([old_val, old_val]),
            torch.cat([adv, adv]),
            torch.cat([ret, ret]),
        )

    def _symmetry_loss(self, obs: torch.Tensor, action_mean: torch.Tensor) -> torch.Tensor:
        """||pi(s) - M_a(pi(M_s(s)))||^2, averaged over the batch.

        States what bilateral symmetry actually means: feeding the mirrored state should
        produce the mirrored action. Note the loss compares against the *current* forward
        pass's mean rather than recomputing it, so the extra cost is one additional forward
        pass per minibatch.

        Retained for comparison, but see the note on `symmetry_loss_coef`: this term has a
        degenerate optimum that data augmentation does not, and it found it.
        """
        mirrored_obs = self._mirror_obs(obs)
        if mirrored_obs is None:
            return torch.zeros((), device=obs.device)
        mirrored_mean = self.policy.actor(self.policy.normalize_obs(mirrored_obs))
        target = mirrored_mean[:, self._act_perm] * self._act_sign
        return (action_mean - target).pow(2).mean()

    def _adapt_learning_rate(self, kl: float) -> None:
        """Hold the per-update policy change near `desired_kl`.

        Too large a KL means the update moved further than the collected data can justify,
        so slow down. Too small means we are leaving progress on the table, so speed up.
        The 1.5x factor and 2x deadband are the standard values from legged-gym.
        """
        cfg = self.cfg
        if kl > cfg.desired_kl * 2.0:
            self.learning_rate = max(cfg.lr_min, self.learning_rate / 1.5)
        elif kl < cfg.desired_kl / 2.0 and kl > 0.0:
            self.learning_rate = min(cfg.lr_max, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    # ------------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "learning_rate": self.learning_rate}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.learning_rate = state.get("learning_rate", self.cfg.learning_rate)
