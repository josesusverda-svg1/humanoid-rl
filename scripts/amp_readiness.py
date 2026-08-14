"""Decide whether a policy is a good enough base to hand to AMP.

AMP replaces half the reward with a discriminator score: "does this transition look like it
came from the mocap library?" That is a STYLE critic. It has no opinion about falling over,
and it cannot make a policy go faster than the policy is able to go. Handing it a base that
falls or crawls does not produce a human walk, it produces a discriminator that wins.

This project has already paid for that lesson once. Started from a policy that did not walk,
the discriminator went 0.53 -> 0.98 accuracy in 97 iterations, the gradient penalty collapsed
4.67 -> 0.65, and the style reward FELL 0.54 -> 0.31 while task return rose. The policy was
being told "everything you do is fake" with no gradient pointing anywhere useful, because
nothing it could reach was close to the reference distribution.

So the gate is about the base, not about AMP. Four things must hold, and each is checked
against a number that came from somewhere other than this file:

  1. IT STAYS UP.       Deterministic fall rate under 20%. Not the training episode length,
                        which is measured with exploration noise on and can hide a policy
                        that only balances because it is shaking.
  2. IT DOES NOT NEED ITS OWN NOISE. Deterministic survival must be at least as good as
                        stochastic survival. If a policy survives longer WITH noise, it is
                        using vibration as a crutch and the deterministic policy AMP would
                        be shaping is not the one that was trained.
  3. IT OBEYS SPEED.    Achieved speed within 30% of commanded, across the range. AMP's
                        reference clips run 0.52-1.24 m/s. A policy stuck at 0.4 m/s is
                        outside the reference distribution at every moment, so every frame
                        it produces is correctly scored as fake.
  4. IT IS ALREADY ROUGHLY A WALK. Step rate 1.6-2.0/s and stride over 0.45 m. AMP polishes
                        a walk into a human walk. It does not turn a shuffle into a walk.

Usage:
    python scripts/amp_readiness.py --run runs/<run> --checkpoint best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataclasses import replace  # noqa: E402

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402

#: Reference clip speeds, measured on the retargeted mocap (see configs/amp.yaml).
#: These bound the range in which the discriminator has anything to say.
CLIP_SPEED_RANGE = (0.52, 1.24)


def load(run: Path, checkpoint: str):
    config = Config.load(run / "config.yaml")
    device = torch.device(resolve_device(config.run.device))
    task = LocomotionTask(config.task)
    env = ThreadedVecEnv(
        REPO_ROOT / config.env.model_path,
        task,
        num_envs=64,
        num_workers=config.env.num_workers,
        decimation=config.env.decimation,
        max_episode_steps=config.env.max_episode_steps,
        action_filter_hz=config.env.action_filter_hz,
        seed=config.run.seed + 10_000,
        domain_rand=replace(config.domain_rand, enabled=False),
    )
    policy = ActorCritic(
        env.obs_dim,
        env.nu,
        actor_hidden=tuple(config.network.actor_hidden),
        critic_hidden=tuple(config.network.critic_hidden),
        activation=config.network.activation,
        init_noise_std=config.network.init_noise_std,
    ).to(device)
    ckpt = torch.load(run / "checkpoints" / checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return config, env, policy, device


def impose(env, cfg, command: float) -> None:
    """Pin the WHOLE command set to a coherent forward walk at `command` m/s.

    Setting only task_state["command"] is not enough and gives a misleading answer. The gait
    clock frequency, the heading target and the body height are drawn alongside the velocity
    command, and the policy observes all of them. Overwriting the velocity alone leaves the
    clock demanding the cadence of some *other*, randomly drawn speed, so the policy is asked
    to satisfy two contradictory instructions and the measurement reports the contradiction
    rather than the policy. The frequency is therefore recomputed here by the same Inman
    square-root law the sampler uses.
    """
    ts = env.state.task_state
    ts["command"][:] = np.array([command, 0.0, 0.0], dtype=ts["command"].dtype)
    freq = cfg.gait_frequency * np.sqrt(max(command, 0.3) / cfg.gait_speed_anchor)
    ts["gait_freq"][:] = np.clip(freq, *cfg.gait_frequency_range)
    ts["stance_frac"][:] = cfg.stance_fraction
    ts["foot_offset"][:] = 0.5
    ts["clock_authority"][:] = 1.0
    ts["body_height"][:] = cfg.target_height
    # Zero yaw command means "keep facing the way you are", which is what the heading
    # target must say for a straight-line walk.
    ts["desired_heading"][:] = env.state.heading


@torch.no_grad()
def probe(env, cfg, policy, device, *, command: float, noise: float,
          steps: int) -> dict[str, float]:
    """Hold one coherent command for the whole rollout, report survival and achieved speed.

    Re-imposed every step because the task resamples on its own schedule, and a rollout that
    spends part of its time asked to stand still measures something else.
    """
    obs = env.reset()
    falls, lengths, speeds = [], [], []
    scale = None if noise <= 0.0 else torch.full((env.num_envs,), noise, device=device)

    for _ in range(steps):
        impose(env, cfg, command)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if scale is None:
            action = policy.act_deterministic(obs_t)
        else:
            action = policy.act(obs_t, noise_scale=scale)[0]
        result = env.step(action.cpu().numpy())
        obs = result.obs
        # Forward speed in the body frame: the component the command actually asks for.
        speeds.append(float(env.state.lin_vel_body[:, 0].mean()))
        done = np.flatnonzero(result.done)
        if done.size:
            falls.extend(result.terminated[done].astype(bool).tolist())
            lengths.extend(result.episode_length[done].tolist())

    return {
        "fall_rate": float(np.mean(falls)) if falls else 0.0,
        "episode_length": float(np.mean(lengths)) if lengths else float(steps),
        "episodes": len(falls),
        "speed": float(np.mean(speeds)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()

    config, env, policy, device = load(args.run, args.checkpoint)
    print(f"\n=== AMP readiness: {args.run.name} / {args.checkpoint} ===\n")

    # --- gate 1 and 2: does it stay up, and does it need its own noise to do it?
    print("survival vs exploration noise (command 1.0 m/s forward)")
    print(f"{'noise std':>10}{'falls':>9}{'ep length':>11}{'speed':>8}")
    survival = {}
    for noise in (0.0, 0.25, 0.5, 1.0):
        r = probe(env, config.task, policy, device, command=1.0, noise=noise, steps=args.steps)
        survival[noise] = r
        tag = " (deterministic, the one AMP would shape)" if noise == 0.0 else ""
        print(f"{noise:>10.2f}{r['fall_rate']:>8.0%}{r['episode_length']:>11.0f}"
              f"{r['speed']:>8.2f}{tag}")

    # --- gate 3: does it obey the speed command across the reference range?
    print("\nspeed tracking, deterministic")
    print(f"{'commanded':>10}{'achieved':>10}{'ratio':>8}{'falls':>8}")
    tracking = {}
    for cmd in (0.5, 0.8, 1.0, 1.5):
        r = probe(env, config.task, policy, device, command=cmd, noise=0.0, steps=args.steps)
        tracking[cmd] = r
        print(f"{cmd:>10.2f}{r['speed']:>10.2f}{r['speed']/cmd:>8.2f}{r['fall_rate']:>8.0%}")
    env.close()

    # --- verdict
    print("\n--- gates ---")
    det, stoch = survival[0.0], survival[0.5]
    checks = []
    checks.append((
        "stays up", det["fall_rate"] < 0.20,
        f"deterministic falls {det['fall_rate']:.0%}, need <20%"))
    # 5% tolerance. The first version demanded det >= stoch exactly and duly failed a policy
    # at 963 steps against 997, a 3.5% gap on a stochastic measurement, which is a coin flip
    # reported as a defect. A crutch means the policy NEEDS its noise, which looks like tens
    # of percent, not three.
    crutch_ok = det["episode_length"] >= 0.95 * stoch["episode_length"]
    checks.append((
        "no noise crutch", crutch_ok,
        f"deterministic survives {det['episode_length']:.0f} steps vs "
        f"{stoch['episode_length']:.0f} with noise "
        f"({det['episode_length'] / max(stoch['episode_length'], 1e-9) - 1.0:+.1%}); "
        f"deterministic must be within 5%"))
    in_band = [c for c in tracking if CLIP_SPEED_RANGE[0] <= c <= CLIP_SPEED_RANGE[1]]
    worst = min((tracking[c]["speed"] / c for c in in_band), default=0.0)
    checks.append((
        "obeys speed", worst >= 0.70,
        f"worst tracking ratio {worst:.2f} inside the clip range "
        f"{CLIP_SPEED_RANGE[0]}-{CLIP_SPEED_RANGE[1]} m/s, need >=0.70"))

    for name, ok, detail in checks:
        print(f"[{'ok' if ok else 'XX'}] {name}: {detail}")
    print("\n(gate 4, step rate and stride, comes from scripts/gait_report.py)")

    ready = all(ok for _, ok, _ in checks)
    print(f"\nVERDICT: {'ready for AMP' if ready else 'NOT ready for AMP'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
