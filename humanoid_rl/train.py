"""Training entry point. One command starts a run, the same command resumes it.

    .venv/bin/python -m humanoid_rl.train --config configs/default.yaml
    .venv/bin/python -m humanoid_rl.train --config configs/default.yaml --resume runs/humanoid-...

Structure of one iteration:

    1. Collect `horizon` steps from every environment in parallel (CPU, 10 threads).
    2. Compute advantages with GAE.
    3. Run the PPO update in minibatches (Metal GPU).
    4. Log metrics as JSONL, which the dashboard tails live.

Phase 0 measured that steps 1 and 3 use independent hardware and barely interfere (CPU
retains 91.5%, GPU 100.3% when run concurrently), which is the basis for overlapping them
in a later optimisation pass. They run sequentially here, because correctness comes first
and asynchronous PPO subtly changes the on-policy data distribution.
"""

from __future__ import annotations

import os

# Must precede numpy and torch imports. Without it every worker spawns its own BLAS thread
# pool on top of our 10 physics threads and the machine oversubscribes badly.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import signal
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from humanoid_rl import hardware
from humanoid_rl.algos.networks import ActorCritic
from humanoid_rl.algos.ppo import PPO, RolloutBuffer
from humanoid_rl.config import Config, resolve_device
from humanoid_rl.envs.vec_env import ThreadedVecEnv
from humanoid_rl.evaluate import evaluate
from humanoid_rl.logging_utils.metrics import EpisodeStats, RunLogger
from humanoid_rl.render import SHORT_SCHEDULE, build_render_env, render_episode
from humanoid_rl.tasks.locomotion import LocomotionTask
from humanoid_rl.tasks.tracking import TrackingTask
from humanoid_rl.tasks.amp_locomotion import AMPLocomotionTask

REPO_ROOT = Path(__file__).resolve().parent.parent


class Trainer:
    """Owns the environment, policy, optimiser, logging and checkpointing for one run."""

    def __init__(self, config: Config, resume_dir: str | None = None) -> None:
        self.cfg = config
        self.hw = hardware.detect()

        # Reproducibility. Every stochastic component is seeded from the one config value.
        seed = config.run.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(resolve_device(config.run.device))
        self.logger = RunLogger(config.run.output_dir, config.run.name, resume_dir)
        if resume_dir is None:
            config.save(self.logger.run_dir / "config.yaml")

        self.task = self._build_task(config)
        self.env = ThreadedVecEnv(
            REPO_ROOT / config.env.model_path,
            self.task,
            num_envs=config.env.num_envs,
            num_workers=config.env.num_workers,
            decimation=config.env.decimation,
            max_episode_steps=config.env.max_episode_steps,
            action_filter_hz=config.env.action_filter_hz,
            action_scale_mode=config.env.action_scale_mode,
            seed=seed,
            domain_rand=config.domain_rand,
        )

        self.policy = ActorCritic(
            self.env.obs_dim,
            self.env.nu,
            actor_hidden=tuple(config.network.actor_hidden),
            critic_hidden=tuple(config.network.critic_hidden),
            activation=config.network.activation,
            init_noise_std=config.network.init_noise_std,
            log_std_max=config.ppo.log_std_max,
        ).to(self.device)

        # Set here, right after the policy exists and BEFORE any warm start, so that
        # init_policy_from's clamp_log_std enforces the floor on the loaded weights too.
        if config.ppo.explore_floor_dims:
            self.policy.set_explore_floor(
                config.ppo.explore_floor_dims, config.ppo.explore_floor)
            print(f"exploration floor: log_std >= {config.ppo.explore_floor:+.2f} "
                  f"(std {math.exp(config.ppo.explore_floor):.3f}) on action dims "
                  f"{list(config.ppo.explore_floor_dims)}")

        # Adversarial motion prior. Owns the discriminator, its optimiser, the policy
        # replay buffer and the reference sampler. Only built for the amp task.
        self.amp = None
        if config.run.task == "amp":
            from humanoid_rl.algos.amp import AMPTrainer

            self.amp = AMPTrainer(
                self.task.lib,
                self.env.nq - 7,
                self.env.n_key_bodies,
                self.device,
                config.amp,
                seed=seed,
            )
            if self.amp.obs_dim != self.task.amp_obs_dim:
                raise ValueError(
                    f"AMP observation width mismatch: trainer {self.amp.obs_dim}, "
                    f"task {self.task.amp_obs_dim}"
                )
            print(f"AMP: discriminator obs dim {self.amp.obs_dim}, "
                  f"blend {config.amp.task_reward_weight}/{config.amp.style_reward_weight}")

        mirror = None
        if config.ppo.symmetry_loss_coef > 0.0:
            from humanoid_rl.envs.mirror import build_mirror_spec, verify

            mirror = build_mirror_spec(
                self.env.model, self.env.n_joint_pos, self.env.n_joint_vel,
                self.env.n_feet, self.env.nu,
            )
            checks = verify(mirror, np.random.default_rng(0))
            if max(checks.values()) > 1e-9:
                raise RuntimeError(f"mirror spec failed its involution check: {checks}")
            print(f"symmetry loss on (coef {config.ppo.symmetry_loss_coef}), mirror verified")

        self.ppo = PPO(self.policy, config.ppo, self.device, seed=seed,
                       mirror=mirror, task=self.task)
        self.buffer = RolloutBuffer(
            config.ppo.horizon, self.env.num_envs, self.env.obs_dim, self.env.nu, self.device
        )
        self.stats = EpisodeStats()

        self.iteration = 0
        self.env_steps = 0
        self.best_return = -float("inf")
        self._stop_requested = False

        # Evaluation environment, created on first use. Deferred because it costs another
        # worker pool, and a short debug run may never evaluate at all. Its threads park on
        # a barrier when idle, so an unused pool costs no CPU.
        self.eval_env: ThreadedVecEnv | None = None
        self.eval_count = 0
        # Env-step mark for the next stick-figure capture. Set from the current step count
        # so a resumed run does not immediately fire one.
        self._next_skeleton = 0.0
        # Per-environment exploration scale, fixed by environment index. The first
        # exploit_env_fraction of environments run near-deterministic so the batch always
        # contains the mean policy's actual behaviour; the rest explore normally. Fixed
        # rather than resampled so an environment's replay statistics stay stationary.
        n = config.env.num_envs
        k = int(n * config.ppo.exploit_env_fraction)
        scale = torch.ones(n, device=self.device)
        scale[:k] = config.ppo.exploit_noise_scale
        self._noise_scale = scale
        #: Single-environment simulation used only for rendering. Built on first use.
        self.render_env: ThreadedVecEnv | None = None

        # Ctrl+C saves a checkpoint before exiting. On a multi-day run, losing hours of
        # progress to an impatient keystroke is an avoidable and infuriating failure.
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        self.logger.log_event(
            "run_start",
            device=str(self.device),
            chip=self.hw.chip,
            performance_cores=self.hw.performance_cores,
            gpu_cores=self.hw.gpu_cores,
            num_envs=self.env.num_envs,
            num_workers=self.env.num_workers,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.nu,
            total_iterations=config.total_iterations,
            steps_per_iteration=config.steps_per_iteration,
        )

    @staticmethod
    def _build_task(config: Config):
        """Construct the objective named by `run.task`."""
        kind = config.run.task
        if kind == "locomotion":
            return LocomotionTask(config.task)
        if kind == "getup":
            from humanoid_rl.tasks.getup import GetUpTask

            return GetUpTask(config.getup)
        if kind == "amp":
            from humanoid_rl.envs.model_prep import prepare
            from humanoid_rl.motion.library import MotionLibrary

            prepared = prepare(REPO_ROOT / config.env.model_path)
            library = MotionLibrary.build(
                REPO_ROOT / config.clip_dir,
                prepared.model,
                include=list(config.clip_include) or None,
                mirror=config.mirror_clips,
            )
            print(library.summary())
            return AMPLocomotionTask(library, config.amp_task, config.amp.n_obs_frames)
        if kind == "tracking":
            from humanoid_rl.envs.model_prep import prepare
            from humanoid_rl.motion.library import MotionLibrary

            prepared = prepare(REPO_ROOT / config.env.model_path)
            library = MotionLibrary.build(
                REPO_ROOT / config.clip_dir,
                prepared.model,
                include=list(config.clip_include) or None,
                mirror=config.mirror_clips,
            )
            print(library.summary())
            return TrackingTask(library, config.tracking)
        raise ValueError(
            f"unknown run.task {kind!r}, expected 'locomotion', 'tracking' or 'amp'"
        )

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        if self._stop_requested:
            raise KeyboardInterrupt("second interrupt, exiting immediately")
        print("\ninterrupt received, finishing this iteration then saving a checkpoint...")
        self._stop_requested = True

    # ------------------------------------------------------------------ rollout

    def collect_rollout(self, obs: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Collect `horizon` steps from every environment. Returns the next obs and timings."""
        cfg = self.cfg
        self.buffer.reset()
        t_env = 0.0
        t_policy = 0.0
        reward_term_sums = np.zeros(self.env.n_reward_terms)
        amp_batch: list[np.ndarray] = []
        style_sum = 0.0

        for t in range(cfg.ppo.horizon):
            t0 = time.perf_counter()
            obs_t = torch.from_numpy(obs).to(self.device)
            action, log_prob, value = self.policy.act(obs_t, self._noise_scale)
            action_np = action.cpu().numpy()
            t_policy += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = self.env.step(action_np)
            t_env += time.perf_counter() - t0

            # Adversarial motion prior: blend the discriminator's verdict into the reward
            # before it reaches the buffer. Done here rather than inside the task because
            # the discriminator is a torch module on the GPU, and the task layer is pure
            # numpy on the main thread.
            if self.amp is not None:
                amp_obs = self.task.amp_observation(self.env.state)
                style = self.amp.style_reward(amp_obs)
                res.reward[:] = (
                    self.cfg.amp.task_reward_weight * res.reward
                    + self.cfg.amp.style_reward_weight * style
                )
                amp_batch.append(amp_obs.copy())
                style_sum += float(style.mean())

            terminated = torch.from_numpy(res.terminated).to(self.device)
            # A mid-episode command change is a boundary for CREDIT ASSIGNMENT but not for
            # the episode: the humanoid keeps walking, but the objective it is being paid
            # for just changed discontinuously. Treating it as a truncation for GAE only
            # bootstraps V(s) across the change instead of asking the critic to have
            # predicted a reward scale it could not see coming. The environment's own
            # `done` is untouched, so no reset happens.
            boundary = res.truncated.copy()
            changed = self.env.state.task_state.get("command_changed")
            if changed is not None:
                boundary |= changed
            truncated = torch.from_numpy(boundary).to(self.device)
            self.buffer.add(
                obs_t,
                action,
                log_prob,
                value,
                torch.from_numpy(res.reward).to(self.device),
                terminated,
                truncated,
            )

            # A truncated episode was cut off by the time limit, so its future still has
            # value and must be bootstrapped from the final observation. A terminated one
            # genuinely ended, and contributes zero.
            if boundary.any():
                idx = np.flatnonzero(boundary)
                # For a real truncation the episode is over, so the final observation is
                # the one to bootstrap from. For a command boundary the episode continues,
                # so the CURRENT observation is the right one.
                source = np.where(res.truncated[idx, None], res.final_obs[idx], res.obs[idx])
                final = torch.from_numpy(source).to(self.device)
                self.buffer.set_truncated_values(
                    t, torch.from_numpy(idx).to(self.device), self.policy.value(final)
                )

            self.stats.add_batch(res.done, res.episode_return, res.episode_length, res.success)
            reward_term_sums += res.reward_terms.mean(axis=0)
            obs = res.obs
            self.env_steps += self.env.num_envs

        with torch.no_grad():
            last_values = self.policy.value(torch.from_numpy(obs).to(self.device))
        self.buffer.compute_returns(last_values, cfg.ppo.gamma, cfg.ppo.gae_lambda)

        timings = {"time_env": t_env, "time_policy": t_policy}
        if self.amp is not None:
            timings["amp/style_reward"] = style_sum / max(1, cfg.ppo.horizon)
            self._amp_batch = np.concatenate(amp_batch, axis=0) if amp_batch else None
        for name, value in zip(self.task.reward_term_names, reward_term_sums / cfg.ppo.horizon):
            timings[f"reward/{name}"] = float(value)
        return obs, timings

    # ------------------------------------------------------------------ evaluation

    def _get_eval_env(self) -> ThreadedVecEnv:
        """Build the evaluation environment once, on first use.

        Shares the Task *instance* with training deliberately. From Phase 4 the task owns
        the curriculum level, and evaluation should measure the policy at the difficulty it
        is currently being trained on, not at some independent one.
        """
        if self.eval_env is None:
            cfg = self.cfg
            self.eval_env = ThreadedVecEnv(
                REPO_ROOT / cfg.env.model_path,
                self.task,
                num_envs=cfg.eval.num_envs,
                num_workers=cfg.env.num_workers,
                decimation=cfg.env.decimation,
                max_episode_steps=cfg.env.max_episode_steps,
                action_filter_hz=cfg.env.action_filter_hz,
            action_scale_mode=cfg.env.action_scale_mode,
                # Fixed offset seed: evaluation is reproducible and independent of how far
                # the training environment's RNG has advanced.
                seed=cfg.run.seed + 10_000,
                # Evaluate on nominal dynamics with clean sensors. Randomised evaluation
                # would mix "did the policy improve" with "was this draw of physics easy",
                # making scores incomparable across checkpoints.
                domain_rand=replace(cfg.domain_rand, enabled=False),
            )
        return self.eval_env

    def _get_render_env(self) -> ThreadedVecEnv:
        if self.render_env is None:
            self.render_env = build_render_env(
                REPO_ROOT / self.cfg.env.model_path, self.task,
                seed=self.cfg.run.seed + 20_000,
                action_scale_mode=self.cfg.env.action_scale_mode,
            )
            # Guard, not a comment. Training, evaluation and rendering must share one action
            # mapping; if they drift, every video and every eval silently describes a robot
            # that was never trained. This exact class of fault has now appeared three times
            # in this project (episode length, actuator ordering, the overlay's command).
            import numpy as np

            if not np.allclose(self.render_env.prepared.action_scale,
                               self.env.prepared.action_scale):
                raise RuntimeError(
                    "render env action_scale differs from the training env: "
                    f"{self.render_env.prepared.action_scale[:3]} vs "
                    f"{self.env.prepared.action_scale[:3]}")
        return self.render_env

    def capture_skeleton(self) -> Path | None:
        """Multi-view stick figure of the current gait, as JSON.

        Deliberately separate from `render_video`: this costs about 0.6 s and 80 KB against
        that method's 18 s and 11 MB, so it can run often enough to form a flip-book of how
        the gait evolved. Failures are swallowed for the same reason as video: a
        visualisation must never end a multi-day run.
        """
        try:
            from humanoid_rl.viz import skeleton

            env = self._get_render_env()
            # A get-up starts on the floor, so the walking view set (six angles around an
            # upright body) mostly shows a silhouette. Use the sagittal-biased set and a
            # longer window: the rise takes several seconds, a stride takes one.
            extra = {}
            if self.cfg.run.task == "getup":
                extra = {"views": skeleton.GETUP_VIEWS, "seconds": 6.0, "settle": 0.0,
                         "max_frames": 90}
            capture = skeleton.capture(env, self.policy, self.device, **extra)
            path = skeleton.write(
                capture,
                self.logger.run_dir / "skeletons" / f"iter_{self.iteration:08d}.json",
                iteration=self.iteration,
                env_steps=self.env_steps,
            )
            self.logger.log_event("skeleton", path=str(path), iteration=self.iteration)
            return path
        except Exception as exc:  # noqa: BLE001 - never kill training over a visualisation
            print(f"skeleton capture failed: {exc}")
            return None

    def render_video(self, tag: str) -> Path | None:
        """Render one evaluation episode to mp4 in the run's videos directory.

        Runs on the main thread between iterations, while the physics workers are parked on
        their barrier, because a CGL context can only be current on one thread at a time.
        Encoding uses Apple's media engine, so it does not compete with the PPO update.

        A failure here must never take down a multi-day training run, so everything is
        caught and reported rather than raised.
        """
        cfg = self.cfg.eval
        try:
            env = self._get_render_env()
            out = self.logger.video_dir / f"iter_{self.iteration:08d}_{tag}.mp4"
            # The walking camera sits high and follows the heading, which for a body on the
            # floor frames mostly empty ground. Drop it and pull back for the get-up task.
            camera = None
            if self.cfg.run.task == "getup":
                from humanoid_rl.render import CameraConfig

                camera = CameraConfig(distance=3.0, elevation=-8.0, azimuth=100.0,
                                      height_offset=0.55)
            t0 = time.perf_counter()
            result = render_episode(
                env,
                self.policy,
                self.device,
                out,
                schedule=SHORT_SCHEDULE,
                camera=camera,
                width=cfg.video_width,
                height=cfg.video_height,
                fps=cfg.video_fps,
            )
            self.logger.log_event(
                "video",
                path=str(out.relative_to(self.logger.run_dir)),
                iteration=self.iteration,
                env_steps=self.env_steps,
                tag=tag,
                render_seconds=round(time.perf_counter() - t0, 1),
                **result.metadata,
            )
            print(f"  video {out.name}  ({result.seconds:.0f}s footage, "
                  f"{time.perf_counter() - t0:.0f}s to render, fell={result.fell})")
            self._prune_videos()
            return out
        except Exception as exc:  # noqa: BLE001 - never kill training over a video
            print(f"  video render failed ({type(exc).__name__}: {exc})")
            self.logger.log_event("video_failed", iteration=self.iteration, error=str(exc))
            return None

    def _prune_videos(self) -> None:
        """Keep the newest N periodic videos. Best-score videos are never pruned."""
        keep = self.cfg.eval.keep_last_videos
        periodic = sorted(self.logger.video_dir.glob("iter_*_periodic.mp4"))
        for old in periodic[:-keep] if len(periodic) > keep else []:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

    def run_evaluation(self) -> dict[str, float]:
        """Deterministic evaluation. Returns metrics already prefixed with 'eval/'."""
        cfg = self.cfg
        env = self._get_eval_env()
        t0 = time.perf_counter()
        result = evaluate(
            env,
            self.policy,
            self.device,
            num_episodes=cfg.eval.num_episodes,
            max_steps=cfg.env.max_episode_steps * 2,
        )
        self.eval_count += 1
        metrics = result.to_flat_dict()
        metrics["eval/seconds"] = time.perf_counter() - t0

        self.logger.log_event(
            "evaluation", iteration=self.iteration, env_steps=self.env_steps, **metrics
        )
        print(
            f"  eval  return {result.episode_return:>8.2f} +- {result.episode_return_std:<6.2f} "
            f"len {result.episode_length:>6.1f}  speed {result.mean_speed:.2f} m/s  "
            f"falls {result.fall_rate * 100:>5.1f}%  ({result.num_episodes} episodes)"
        )
        return metrics

    # ------------------------------------------------------------------ main loop

    def train(self, max_iterations: int | None = None) -> None:
        cfg = self.cfg
        total_iterations = max_iterations or cfg.total_iterations
        obs = self.env.reset()

        print(f"run dir : {self.logger.run_dir}")
        print(f"device  : {self.device}   envs: {self.env.num_envs}   "
              f"workers: {self.env.num_workers}")
        print(f"target  : {cfg.run.total_env_steps:,} env steps "
              f"= {total_iterations:,} iterations of {cfg.steps_per_iteration:,}")
        print()

        run_start = time.perf_counter()
        while self.iteration < total_iterations and not self._stop_requested:
            iter_start = time.perf_counter()

            obs, rollout_info = self.collect_rollout(obs)

            t0 = time.perf_counter()
            update_info = self.ppo.update(self.buffer)
            # Discriminator update, with its own optimiser rather than a joint loss, so its
            # learning rate is independently tunable. This is the AMP paper's design.
            if self.amp is not None and getattr(self, "_amp_batch", None) is not None:
                update_info.update(self.amp.update(self._amp_batch))
            t_update = time.perf_counter() - t0

            # Observation statistics are refreshed once per iteration, after the update, so
            # the normaliser stays fixed for the whole collect-and-update cycle. See the
            # note in ppo.py on why a shifting normaliser corrupts the importance ratio.
            self.policy.obs_rms.update(self.buffer.obs.reshape(-1, self.env.obs_dim))

            self.iteration += 1
            elapsed = time.perf_counter() - iter_start
            sps = cfg.steps_per_iteration / elapsed

            metrics: dict[str, Any] = {
                **self.stats.summary(),
                **update_info,
                **rollout_info,
                "env_steps_per_sec": sps,
                "iteration_time": elapsed,
                "time_update": t_update,
                "total_wall_hours": (time.perf_counter() - run_start) / 3600.0,
            }

            # Periodic deterministic evaluation. Its result is what "best" is judged on,
            # because the training return includes exploration noise and is not comparable
            # across checkpoints.
            if self.iteration % cfg.eval.interval_iterations == 0:
                eval_metrics = self.run_evaluation()
                metrics.update(eval_metrics)
                score = eval_metrics.get("eval/episode_return", -float("inf"))

                is_best = score > self.best_return
                if is_best:
                    self.best_return = score
                    self.save_checkpoint(best=True)

                # Videos: on a new best, and on a fixed cadence so a plateau still gets
                # looked at. The metrics alone hid a badly broken gait for 24.6M steps.
                every = cfg.eval.video_every_n_evals
                due = every > 0 and self.eval_count % every == 0
                if is_best and cfg.eval.video_on_best:
                    self.render_video("best")
                elif due:
                    self.render_video("periodic")

                # Stick figures on their own, much shorter, cadence. Measured in env steps
                # rather than evaluations so the spacing is comparable across runs with
                # different evaluation intervals.
                gap = cfg.eval.skeleton_every_m_steps * 1e6
                if gap > 0 and self.env_steps >= self._next_skeleton:
                    self._next_skeleton = self.env_steps + gap
                    self.capture_skeleton()

            if self.iteration % cfg.log.log_interval_iterations == 0:
                self.logger.log_metrics(self.iteration, self.env_steps, metrics)
                self._print_progress(metrics, total_iterations, sps)

            if self.iteration % cfg.log.checkpoint_interval_iterations == 0:
                self.save_checkpoint()

            # Thermal state is logged periodically: a laptop under multi-day load will
            # throttle, and without this a sudden throughput drop looks like a code bug.
            if self.iteration % 100 == 0:
                self.logger.log_event("thermal", **hardware.thermal_state())

        self.save_checkpoint()
        self.logger.log_event(
            "run_end",
            iterations=self.iteration,
            env_steps=self.env_steps,
            best_return=self.best_return,
            interrupted=self._stop_requested,
        )
        self.env.close()
        if self.eval_env is not None:
            self.eval_env.close()
        if self.render_env is not None:
            self.render_env.close()
        self.logger.close()
        print(f"\nfinished at iteration {self.iteration:,}, {self.env_steps:,} env steps")
        print(f"run dir: {self.logger.run_dir}")

    def _print_progress(self, m: dict[str, Any], total: int, sps: float) -> None:
        remaining = (total - self.iteration) * self.cfg.steps_per_iteration
        eta_h = remaining / sps / 3600.0 if sps > 0 else 0.0
        print(
            f"it {self.iteration:>6,}/{total:<7,} "
            f"steps {self.env_steps / 1e6:>7.2f}M  "
            f"ret {m['episode_return']:>8.2f}  "
            f"len {m['episode_length']:>6.1f}  "
            f"kl {m['approx_kl']:.4f}  "
            f"lr {m['learning_rate']:.2e}  "
            f"std {m['action_std']:.3f}  "
            f"{sps:>7,.0f} sps  "
            f"eta {eta_h:>5.1f}h"
        )

    # ------------------------------------------------------------------ checkpoints

    def save_checkpoint(self, best: bool = False) -> None:
        name = "best.pt" if best else f"iter_{self.iteration:08d}.pt"
        path = self.logger.checkpoint_dir / name
        torch.save(
            {
                "iteration": self.iteration,
                "env_steps": self.env_steps,
                "best_return": self.best_return,
                # obs_rms lives inside the policy state dict, so normalisation statistics
                # travel with the weights and a resumed run cannot mismatch them.
                "policy": self.policy.state_dict(),
                "ppo": self.ppo.state_dict(),
                "config": self.cfg.to_dict(),
            },
            path,
        )
        self.logger.log_event(
            "checkpoint",
            path=str(path.relative_to(self.logger.run_dir)),
            iteration=self.iteration,
            env_steps=self.env_steps,
            best=best,
            episode_return=self.best_return if best else None,
        )
        if not best:
            self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        keep = self.cfg.log.keep_last_checkpoints
        saved = sorted(self.logger.checkpoint_dir.glob("iter_*.pt"))
        for old in saved[:-keep] if len(saved) > keep else []:
            old.unlink(missing_ok=True)

    def init_policy_from(self, path: str | Path) -> None:
        """Warm-start the policy from another run's weights, keeping everything else fresh.

        Used to begin AMP from the Phase 2 walking policy. Starting adversarial training
        from a randomly initialised policy makes the discriminator's job trivial: it reached
        0.96 to 0.98 accuracy within a couple of hundred iterations, which saturates the
        bounded style reward near zero and leaves almost no gradient. A policy that already
        walks is much harder to distinguish from human mocap, so the adversarial game starts
        somewhere useful.

        Only the policy weights and observation normaliser are taken. The optimiser state,
        iteration count and the discriminator all start fresh, because they describe a
        different objective.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        incoming = state["policy"]
        current = self.policy.state_dict()
        # A wider observation is expected and recoverable: adding the gait clock grew the
        # observation from 98 to 100. Copy the old weights into the leading columns and leave
        # the new ones at zero, so the warm-started policy begins EXACTLY as it behaved
        # before, blind to the new inputs, and learns to use them from there. Anything else
        # that changes shape is a genuine mismatch and still raises.
        for key, tensor in list(incoming.items()):
            if key not in current or tensor.shape == current[key].shape:
                continue
            target = current[key]
            if tensor.dim() == 2 and tensor.shape[0] == target.shape[0] and tensor.shape[1] < target.shape[1]:
                grown = torch.zeros_like(target)
                grown[:, : tensor.shape[1]] = tensor
                incoming[key] = grown
            elif tensor.dim() == 1 and tensor.shape[0] < target.shape[0]:
                # Observation normaliser statistics. New inputs start at mean 0, variance 1,
                # which is what a fresh normaliser would hold anyway.
                grown = torch.ones_like(target) if key.endswith("var") else torch.zeros_like(target)
                grown[: tensor.shape[0]] = tensor
                incoming[key] = grown

        shape_mismatch = [
            k for k in incoming if k in current and incoming[k].shape != current[k].shape
        ]
        if shape_mismatch:
            raise ValueError(
                f"cannot warm-start from {path}: shape mismatch on {shape_mismatch}. "
                "The observation or action layout differs between the two runs."
            )
        missing = self.policy.load_state_dict(incoming, strict=False)
        # A warm start can inherit a log_std from outside the current clamp, at which point
        # the config's init_noise_std is silently ignored and the parameter is frozen. That
        # is exactly how this project ran 610 iterations at a std it never chose.
        self.policy.clamp_log_std()
        self.logger.log_event("init_from", path=str(path), source_iteration=state["iteration"])
        print(
            f"warm-started policy from {path} "
            f"(iteration {state['iteration']:,}, {state['env_steps'] / 1e6:.0f}M steps)"
            + (f", unmatched keys: {missing.missing_keys}" if missing.missing_keys else "")
        )

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(state["policy"])
        self.ppo.load_state_dict(state["ppo"])
        self.iteration = state["iteration"]
        self.env_steps = state["env_steps"]
        self.best_return = state.get("best_return", -float("inf"))
        print(f"resumed from {path}: iteration {self.iteration:,}, {self.env_steps:,} env steps")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml")
    ap.add_argument("--resume", type=Path, default=None, help="run directory to resume")
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="load policy weights from a checkpoint as a starting point (fresh optimiser, "
        "fresh iteration count). Used to start AMP from a policy that already walks.",
    )
    ap.add_argument("--iterations", type=int, default=None, help="override iteration count")
    ap.add_argument("--num-envs", type=int, default=None, help="override env count")
    ap.add_argument(
        "--eval-interval", type=int, default=None, help="override eval interval in iterations"
    )
    args = ap.parse_args()

    config = Config.load(args.config)
    if args.num_envs is not None:
        config.env.num_envs = args.num_envs
    if args.eval_interval is not None:
        config.eval.interval_iterations = args.eval_interval

    resume_dir = str(args.resume) if args.resume else None
    trainer = Trainer(config, resume_dir=resume_dir)

    if args.init_from:
        trainer.init_policy_from(args.init_from)

    if args.resume:
        ckpts = sorted((Path(args.resume) / "checkpoints").glob("iter_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints found in {args.resume}/checkpoints")
        trainer.load_checkpoint(ckpts[-1])

    trainer.train(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
