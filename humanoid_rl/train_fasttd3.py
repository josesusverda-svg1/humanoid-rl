"""Off-policy training loop, deliberately separate from humanoid_rl/train.py.

Separate rather than a branch inside the PPO trainer, because the PPO path currently works
and is producing the only walking policy this project has. A shared trainer with `if algo ==`
scattered through it is how a working path acquires a bug from an experimental one.

What is shared, and it is nearly everything: the environment, the reward function, the
observations, the humanoid model, the evaluator, the metrics logger and the checkpoint
layout. Only the consumer of transitions differs.

    .venv/bin/python -m humanoid_rl.train_fasttd3 --config configs/fasttd3.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from humanoid_rl.algos.fasttd3 import FastTD3
from humanoid_rl.config import Config, resolve_device
from humanoid_rl.envs.vec_env import ThreadedVecEnv
from humanoid_rl.evaluate import evaluate
from humanoid_rl.logging_utils.metrics import RunLogger
from humanoid_rl.tasks.locomotion import LocomotionTask

REPO_ROOT = Path(__file__).resolve().parent.parent


class _PolicyShim(torch.nn.Module):
    """Presents a FastTD3 actor with the interface `evaluate()` expects.

    `evaluate` was written against ActorCritic and calls `act_deterministic`. Rather than
    fork the evaluator (and risk the two drifting, which is exactly how the biased-eval bug
    survived so long), the actor is wrapped so BOTH algorithms are scored by the same code
    on the same definition of a fall.
    """

    def __init__(self, actor: torch.nn.Module) -> None:
        super().__init__()
        self.actor = actor

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "fasttd3.yaml")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if cfg.run.algo != "fasttd3":
        raise SystemExit(f"config selects algo={cfg.run.algo!r}; use configs/fasttd3.yaml")
    device = torch.device(resolve_device(cfg.run.device))
    torch.manual_seed(cfg.run.seed)

    # RunLogger owns the run directory, so a FastTD3 run is laid out identically to a PPO
    # run and every existing tool (dashboard, gait_report, amp_readiness, the Oracle) reads
    # it without knowing which algorithm produced it.
    #
    # Saved BEFORE the environment is built, matching train.py. `prepare()` overwrites
    # target_height and terminate_height on the config with numpy scalars derived from the
    # model, and yaml refuses to serialise those. Saving first keeps them plain floats.
    logger = RunLogger(REPO_ROOT / cfg.run.output_dir, cfg.run.name)
    run_dir = logger.run_dir
    cfg.save(run_dir / "config.yaml")

    task = LocomotionTask(cfg.task)
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=cfg.env.num_envs,
        num_workers=cfg.env.num_workers, decimation=cfg.env.decimation,
        max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=cfg.run.seed,
        domain_rand=cfg.domain_rand,
    )
    agent = FastTD3(env.obs_dim, env.nu, cfg.fasttd3, cfg.env.num_envs, device)

    total_iters = args.iterations or (cfg.run.total_env_steps // cfg.env.num_envs)
    print(f"run dir : {run_dir}")
    print(f"device  : {device}   envs: {cfg.env.num_envs}   algo: fasttd3")
    print(f"buffer  : {agent.buffer.capacity:,} transitions "
          f"({agent.buffer.capacity * (2 * env.obs_dim + env.nu + 2) * 4 / 1e9:.1f} GB)")
    print(f"target  : {cfg.run.total_env_steps:,} env steps = {total_iters:,} iterations\n")

    obs = env.reset()
    eval_env: ThreadedVecEnv | None = None
    best_return = -float("inf")
    started = time.perf_counter()
    env_steps = 0

    for iteration in range(1, total_iters + 1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action = agent.act(obs_t, explore=True)
        result = env.step(action.cpu().numpy())
        env_steps += cfg.env.num_envs

        # next_obs must be the FINAL observation of the episode where one ended, not the
        # reset observation the env already returned in its place. Using the reset obs would
        # teach the critic that falling leads to a fresh standing start, which is the single
        # most damaging thing an off-policy buffer can be fed.
        next_obs = result.obs.copy()
        done = result.done
        if done.any():
            next_obs[done] = result.final_obs[done]
        # Bootstrap through a time-limit truncation but not through a real termination: a
        # humanoid still walking when the 20 s clock runs out has a genuine future.
        keep_going = (~result.terminated).astype(np.float32)

        agent.buffer.add(
            obs_t,
            action,
            torch.as_tensor(result.reward, dtype=torch.float32, device=device),
            torch.as_tensor(next_obs, dtype=torch.float32, device=device),
            torch.as_tensor(keep_going, dtype=torch.float32, device=device),
        )
        obs = result.obs

        metrics: dict[str, float] = {}
        if iteration >= cfg.fasttd3.learning_starts:
            for _ in range(cfg.fasttd3.num_updates):
                metrics = agent.update()

        # Intervals are used AS WRITTEN, not multiplied. An off-policy iteration is one
        # environment step, not a 24-step horizon, so the same number means something very
        # different here than in the PPO trainer; the config carries off-policy values.
        if iteration % max(cfg.log.log_interval_iterations, 1) == 0:
            elapsed = time.perf_counter() - started
            sps = env_steps / max(elapsed, 1e-9)
            logger.log_metrics(iteration, env_steps, {
                **metrics,
                "buffer_size": float(len(agent.buffer)),
                "env_steps_per_sec": sps,
            })
            print(f"it {iteration:>7,}/{total_iters:,}  steps {env_steps / 1e6:>7.2f}M  "
                  f"q {metrics.get('q_value', float('nan')):>8.1f}  "
                  f"critic {metrics.get('critic_loss', float('nan')):>7.3f}  "
                  f"{sps:>8,.0f} sps  "
                  f"eta {(total_iters - iteration) / max(iteration / elapsed, 1e-9) / 3600:>5.1f}h",
                  flush=True)

        if iteration % max(cfg.eval.interval_iterations, 1) == 0:
            if eval_env is None:
                eval_env = ThreadedVecEnv(
                    REPO_ROOT / cfg.env.model_path, task, num_envs=cfg.eval.num_envs,
                    num_workers=cfg.env.num_workers, decimation=cfg.env.decimation,
                    max_episode_steps=cfg.env.max_episode_steps,
                    action_filter_hz=cfg.env.action_filter_hz, seed=cfg.run.seed + 10_000,
                    domain_rand=replace(cfg.domain_rand, enabled=False),
                )
            shim = _PolicyShim(agent.actor)
            res = evaluate(eval_env, shim, device, num_episodes=cfg.eval.num_episodes,
                           max_steps=cfg.env.max_episode_steps * 2)
            logger.log_metrics(iteration, env_steps, res.to_flat_dict())
            print(f"  eval: return {res.episode_return:>8.1f}  falls {res.fall_rate:>5.1%}  "
                  f"len {res.episode_length:>6.0f}  speed {res.mean_speed:.2f}", flush=True)
            if res.episode_return > best_return:
                best_return = res.episode_return
                torch.save({"policy": agent.state_dict(), "iteration": iteration,
                            "env_steps": env_steps, "best_return": best_return},
                           logger.checkpoint_dir / "best.pt")

    env.close()
    if eval_env is not None:
        eval_env.close()
    print(f"\nfinished at iteration {total_iters:,}, {env_steps:,} env steps")
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
