"""Deterministic evaluation, run separately from training.

Why this is separate rather than reusing the training statistics: during training the policy
acts by *sampling* from its Gaussian, so the reported return includes exploration noise, and
autoreset means episodes are constantly being cut and restarted. Neither reflects how the
policy would actually perform if you deployed it.

Evaluation instead uses the distribution mean (no exploration), a fixed seed, and its own
environment pool, so the number is reproducible and comparable across checkpoints. That is
also the number the "best" checkpoint should be selected on.

This module is where Phase 3 will hook offscreen video rendering, since it already owns a
clean deterministic rollout of a full episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from humanoid_rl.algos.networks import ActorCritic
from humanoid_rl.envs.vec_env import ThreadedVecEnv


@dataclass
class EvalResult:
    """Summary of a deterministic evaluation."""

    num_episodes: int
    episode_return: float
    episode_return_std: float
    episode_length: float
    success_rate: float
    fall_rate: float
    #: Mean planar speed in the humanoid's own frame, in m/s. A policy that scores well
    #: while barely moving is a very common local optimum, and this exposes it immediately.
    mean_speed: float
    #: Extra task-specific metrics, merged in from Task.eval_metrics.
    task_metrics: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self, prefix: str = "eval/") -> dict[str, float]:
        out = {f"{prefix}{k}": v for k, v in asdict(self).items() if k != "task_metrics"}
        out.update({f"{prefix}{k}": v for k, v in self.task_metrics.items()})
        return out


@torch.no_grad()
def evaluate(
    env: ThreadedVecEnv,
    policy: ActorCritic,
    device: torch.device,
    *,
    num_episodes: int = 32,
    max_steps: int = 2000,
) -> EvalResult:
    """Run the policy deterministically and score the FIRST episode from each environment.

    One episode per environment, and no more. This is not a detail, it is the whole
    correctness argument, and getting it wrong silently poisoned every eval number in this
    project for months.

    The previous version ran until `num_episodes` episodes had finished, counting whichever
    finished first. Under autoreset the first episodes to finish are, by construction, the
    SHORT ones, which is to say the falls. An environment that survives to the 1000-step
    limit contributes nothing unless enough of its neighbours survive too. The result was a
    metric with two modes and nothing in between, across 101 evals of one run:

        fall_rate 1.00, 32-34 episodes counted   ->  85 evals
        fall_rate 0.31, 64-71 episodes counted   ->  16 evals

    Both describe the same policy. The first is what gets reported whenever half the
    environments fall early: the loop hits its quota on falls alone and stops before a
    single survivor is counted. The second is what gets reported when fewer than half fall,
    because then all 64 truncate together at the step limit and land in the sample at once.

    The damage went past the log. `best.pt` is selected on `eval/episode_return`, which
    inherits the same bias, so returns read ~2,700 in the second mode against ~1,000 in the
    first. That gap is far larger than any real difference between checkpoints, which means
    checkpoint selection was mostly deciding which mode the eval happened to land in.

    Counting each environment exactly once removes the bias: every environment contributes
    one episode regardless of how long it lasted, so falls and survivors are sampled at the
    rate they actually occur.

    Args:
        env: A dedicated evaluation environment. Must not be the training environment,
            whose internal state would be destroyed by the reset.
        policy: The actor-critic. Only its mean action is used.
        num_episodes: Minimum environments that must report before the result is trusted.
            The sample size is `env.num_envs`, so keep the eval env at least this wide.
        max_steps: Hard cap on vectorised steps, so a policy that never terminates cannot
            stall training forever. Must exceed the episode limit or slow environments are
            dropped, which would reintroduce exactly the bias described above.
    """
    was_training = policy.training
    policy.eval()

    obs = env.reset()
    #: One slot per environment, flipped when that environment's first episode lands.
    counted = np.zeros(env.num_envs, dtype=bool)
    returns: list[float] = []
    lengths: list[float] = []
    successes: list[bool] = []
    falls: list[bool] = []
    speed_sum = 0.0
    speed_count = 0
    task_accum: dict[str, list[float]] = {}

    for _ in range(max_steps):
        obs_t = torch.from_numpy(obs).to(device)
        action = policy.act_deterministic(obs_t).cpu().numpy()
        result = env.step(action)
        obs = result.obs

        # Planar speed in the body frame, averaged over every environment and step.
        speed_sum += float(np.linalg.norm(env.state.lin_vel_body[:, :2], axis=1).mean())
        speed_count += 1

        for name, value in env.task.eval_metrics(env.state).items():
            task_accum.setdefault(name, []).append(value)

        # Only environments that have not yet reported. Later episodes from an environment
        # that already finished one are ignored, so a fast-failing environment cannot fill
        # the sample with repeats of its own failure.
        done_idx = np.flatnonzero(result.done & ~counted)
        if done_idx.size:
            counted[done_idx] = True
            returns.extend(result.episode_return[done_idx].tolist())
            lengths.extend(result.episode_length[done_idx].tolist())
            successes.extend(result.success[done_idx].astype(bool).tolist())
            # Terminated means it fell or otherwise failed. Truncated means it survived
            # to the time limit, which is the good outcome for a locomotion task.
            falls.extend(result.terminated[done_idx].astype(bool).tolist())

        if counted.all():
            break

    if was_training:
        policy.train()

    if not returns:
        # No episode finished within the step budget. Report zeros rather than raising,
        # so a single pathological evaluation cannot kill a multi-day training run.
        return EvalResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    if not counted.all():
        # Some environments never finished an episode inside max_steps. Whatever they were
        # doing, they were doing it for longer than everyone else, so dropping them biases
        # the result in exactly the direction this function was just fixed to avoid. Say so
        # rather than quietly averaging over the survivors.
        print(f"  [eval] WARNING: {int((~counted).sum())} of {env.num_envs} environments "
              f"did not finish an episode within max_steps={max_steps}; the reported "
              f"fall rate and episode length are biased low. Raise max_steps.", flush=True)
    elif len(returns) < num_episodes:
        print(f"  [eval] WARNING: sample is {len(returns)} episodes against a requested "
              f"{num_episodes}. The eval env has only {env.num_envs} environments and each "
              f"contributes one; widen eval.num_envs.", flush=True)

    return EvalResult(
        num_episodes=len(returns),
        episode_return=float(np.mean(returns)),
        episode_return_std=float(np.std(returns)),
        episode_length=float(np.mean(lengths)),
        success_rate=float(np.mean(successes)),
        fall_rate=float(np.mean(falls)),
        mean_speed=speed_sum / max(1, speed_count),
        task_metrics={k: float(np.mean(v)) for k, v in task_accum.items()},
    )
