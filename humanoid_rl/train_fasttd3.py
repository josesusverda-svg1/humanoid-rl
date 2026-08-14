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

from humanoid_rl.algos.async_collector import AsyncCollector
from humanoid_rl.algos.fasttd3 import FastTD3
from humanoid_rl.config import Config, resolve_device
from humanoid_rl.envs.vec_env import ThreadedVecEnv
from humanoid_rl.evaluate import evaluate
from humanoid_rl.logging_utils.metrics import RunLogger
from humanoid_rl.render import SHORT_SCHEDULE, build_render_env, render_episode
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
    render_env: ThreadedVecEnv | None = None
    best_return = -float("inf")
    started = time.perf_counter()
    env_steps = 0
    # Rolling window of finished training episodes. The PPO trainer logs these and every
    # dashboard chart of "how is it going" reads them; without them the run looks dead even
    # while it trains fine.
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    reward_term_sums = np.zeros(len(task.reward_term_names), dtype=np.float64)
    reward_term_count = 0
    evals_done = 0

    collector: AsyncCollector | None = None
    if cfg.fasttd3.async_collection:
        collector = AsyncCollector(env, agent.actor, device, agent.explore_std)
        collector.start()
        print("async collection ON: physics and gradients overlap\n", flush=True)

    def to_dev(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    for iteration in range(1, total_iters + 1):
        if collector is not None:
            # Ask for exactly one iteration's worth of fresh data. This is what pins the
            # replay ratio: whichever side is slower sets the pace, and the ratio of samples
            # processed to environment steps collected stays what the config asked for.
            for chunk in collector.drain(min_steps=cfg.env.num_envs):
                agent.buffer.add(to_dev(chunk.obs), to_dev(chunk.actions),
                                 to_dev(chunk.rewards), to_dev(chunk.next_obs),
                                 to_dev(chunk.keep_going))
                env_steps += chunk.obs.shape[0]
                if chunk.episode_returns.size:
                    recent_returns.extend(chunk.episode_returns.tolist())
                    recent_lengths.extend(chunk.episode_lengths.tolist())
                reward_term_sums += chunk.reward_terms
                reward_term_count += 1
            recent_returns = recent_returns[-2000:]
            recent_lengths = recent_lengths[-2000:]
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action = agent.act(obs_t, explore=True)
            result = env.step(action.cpu().numpy())
            env_steps += cfg.env.num_envs

            # next_obs must be the FINAL observation of the episode where one ended, not the
            # reset observation the env already returned in its place. Using the reset obs
            # would teach the critic that falling leads to a fresh standing start, which is
            # the single most damaging thing an off-policy buffer can be fed.
            next_obs = result.obs.copy()
            done = result.done
            if done.any():
                next_obs[done] = result.final_obs[done]
            # Bootstrap through a time-limit truncation but not a real termination: a
            # humanoid still walking when the 20 s clock runs out has a genuine future.
            keep_going = (~result.terminated).astype(np.float32)

            finished = np.flatnonzero(result.done)
            if finished.size:
                recent_returns.extend(result.episode_return[finished].tolist())
                recent_lengths.extend(result.episode_length[finished].tolist())
                recent_returns = recent_returns[-2000:]
                recent_lengths = recent_lengths[-2000:]
            reward_term_sums += result.reward_terms.mean(axis=0)
            reward_term_count += 1

            agent.buffer.add(obs_t, action, to_dev(result.reward), to_dev(next_obs),
                             to_dev(keep_going))
            obs = result.obs

        metrics: dict[str, float] = {}
        if iteration >= cfg.fasttd3.learning_starts:
            for _ in range(cfg.fasttd3.num_updates):
                metrics = agent.update()
            if collector is not None:
                # Publish the freshly-updated policy to the actor thread. Once per round is
                # standard; the actor is then at most one round stale, which off-policy
                # learning is entirely fine with.
                collector.sync_weights(agent.actor)

        # Intervals are used AS WRITTEN, not multiplied. An off-policy iteration is one
        # environment step, not a 24-step horizon, so the same number means something very
        # different here than in the PPO trainer; the config carries off-policy values.
        if iteration % max(cfg.log.log_interval_iterations, 1) == 0:
            elapsed = time.perf_counter() - started
            sps = env_steps / max(elapsed, 1e-9)
            terms = {}
            if reward_term_count:
                terms = {f"reward/{name}": float(reward_term_sums[i] / reward_term_count)
                         for i, name in enumerate(task.reward_term_names)}
            logger.log_metrics(iteration, env_steps, {
                **metrics, **terms,
                "episode_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "episode_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                "buffer_size": float(len(agent.buffer)),
                "action_std": float(agent.explore_std.mean()),
                "env_steps_per_sec": sps,
                # Actor idle time. Large means the GPU is behind and the replay ratio is too
                # high for this machine; near zero means physics is the limit, which is the
                # state we are trying to reach.
                **({"actor_blocked_frac": collector.blocked_seconds
                    / max(time.perf_counter() - started, 1e-9)} if collector else {}),
            })
            reward_term_sums[:] = 0.0
            reward_term_count = 0
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
            improved = res.episode_return > best_return
            if improved:
                best_return = res.episode_return
                torch.save({"policy": agent.state_dict(), "iteration": iteration,
                            "env_steps": env_steps, "best_return": best_return},
                           logger.checkpoint_dir / "best.pt")

            # Visualisation. Wrapped so a rendering failure can never end a training run,
            # which is the same rule the PPO trainer follows and the reason it has survived
            # several broken visualisers.
            evals_done += 1
            want_video = improved and cfg.eval.video_on_best
            want_video = want_video or (cfg.eval.video_every_n_evals > 0
                                        and evals_done % cfg.eval.video_every_n_evals == 0)
            # Built unconditionally, because the flip-book below runs on EVERY evaluation
            # even when no video is due. It costs 0.6 s and 80 KB against a video's 18 s and
            # 11 MB, which is what makes a flip-book of the whole run affordable.
            if render_env is None:
                render_env = build_render_env(
                    REPO_ROOT / cfg.env.model_path, task, seed=cfg.run.seed + 20_000)
            shim_eval = _PolicyShim(agent.actor)
            if want_video:
                try:
                    out = logger.video_dir / f"iter_{iteration:08d}_{'best' if improved else 'periodic'}.mp4"
                    t0 = time.perf_counter()
                    rendered = render_episode(
                        render_env, shim_eval, device, out, schedule=SHORT_SCHEDULE,
                        width=cfg.eval.video_width, height=cfg.eval.video_height,
                        fps=cfg.eval.video_fps)
                    logger.log_event("video", path=str(out.relative_to(logger.run_dir)),
                                     iteration=iteration, env_steps=env_steps,
                                     tag="best" if improved else "periodic",
                                     render_seconds=round(time.perf_counter() - t0, 1),
                                     **rendered.metadata)
                    print(f"  video {out.name} ({rendered.seconds:.0f}s footage, fell={rendered.fell})",
                          flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  video failed: {exc}", flush=True)
            try:
                from humanoid_rl.viz import skeleton
                cap = skeleton.capture(render_env, shim_eval, device)
                path = skeleton.write(
                    cap, logger.run_dir / "skeletons" / f"iter_{iteration:08d}.json",
                    iteration=iteration, env_steps=env_steps)
                logger.log_event("skeleton", path=str(path), iteration=iteration)
            except Exception as exc:  # noqa: BLE001
                print(f"  skeleton failed: {exc}", flush=True)

    if collector is not None:
        collector.stop()
    env.close()
    if eval_env is not None:
        eval_env.close()
    if render_env is not None:
        render_env.close()
    print(f"\nfinished at iteration {total_iters:,}, {env_steps:,} env steps")
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
