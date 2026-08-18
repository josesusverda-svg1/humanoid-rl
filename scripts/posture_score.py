"""Measure torso posture under a HELD command, identically for every checkpoint.

Why not just read `eval/torso_upright` from metrics.jsonl. Two reasons, both of which would
bias a comparison between arms rather than merely add noise:

* The eval metric is sampled across whatever command mix the evaluation drew, and a treatment
  that changes the fall rate changes that mix. Slower commands read as more upright with no
  postural improvement whatsoever.
* `torso_upright` is read after `_do_resets` has already overwritten some rows, so the value
  is contaminated in proportion to how often episodes ended, which every treatment moves.

Holding one command for every checkpoint removes both. The number is then comparable across
arms by construction rather than by hope.

DIRECTION IS REPORTED SEPARATELY, and this is the important part. `torso_upright` is
`cos(tilt)`, which is SIGN-BLIND: a 48 degree backward fold and a 48 degree forward lean give
the identical number. This project spent a full analysis cycle believing the humanoid leaned
forward when it was folding backward and to the left. The fore/lateral decomposition below is
what makes that visible, so a "fix" that merely flips the fold cannot score as a win.

    python scripts/posture_score.py runs/lean-ctl-s0-* runs/lean-flr-s0-*
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402


def score(run: Path, checkpoint: str = "best.pt", steps: int = 500, settle: int = 100,
          command: float = 1.0, seed: int = 11) -> dict[str, float]:
    cfg = Config.load(run / "config.yaml")
    device = torch.device(resolve_device(cfg.run.device))
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, LocomotionTask(cfg.task), num_envs=64,
        num_workers=cfg.env.num_workers, decimation=cfg.env.decimation,
        max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=seed,
        domain_rand=replace(cfg.domain_rand, enabled=False),
    )
    policy = ActorCritic(
        env.obs_dim, env.nu, actor_hidden=tuple(cfg.network.actor_hidden),
        critic_hidden=tuple(cfg.network.critic_hidden), activation=cfg.network.activation,
        init_noise_std=cfg.network.init_noise_std,
    ).to(device)
    state = torch.load(run / "checkpoints" / checkpoint, map_location=device,
                       weights_only=False)
    policy.load_state_dict(state["policy"])
    policy.eval()

    adr = env._torso_adr  # noqa: SLF001 - full torso z-axis, not the z component alone
    obs = env.reset()
    up, fore, lat, spd, rew, fell = [], [], [], [], [], []
    with torch.no_grad():
        for t in range(steps):
            env.state.task_state["command"][:] = np.array([command, 0.0, 0.0], np.float32)
            action = policy.act_deterministic(
                torch.as_tensor(obs, dtype=torch.float32, device=device))
            result = env.step(action.cpu().numpy())
            obs = result.obs
            if t < settle:
                continue
            z = env._sensordata[:, adr:adr + 3]  # noqa: SLF001
            yaw = env.state.heading
            fore.append(np.cos(yaw) * z[:, 0] + np.sin(yaw) * z[:, 1])
            lat.append(-np.sin(yaw) * z[:, 0] + np.cos(yaw) * z[:, 1])
            up.append(z[:, 2].copy())
            spd.append(env.state.lin_vel_body[:, 0].copy())
            rew.append(result.reward.copy())
            fell.append(result.terminated.copy())
    env.close()

    u = float(np.concatenate(up).mean())
    f = float(np.concatenate(fore).mean())
    return {
        "torso_upright": u,
        "tilt_deg": float(np.degrees(np.arccos(np.clip(u, -1.0, 1.0)))),
        "fore": f,
        "lateral": float(np.concatenate(lat).mean()),
        "direction": "FORWARD" if f > 0.05 else ("BACKWARD" if f < -0.05 else "upright"),
        "speed": float(np.concatenate(spd).mean()),
        "reward_per_step": float(np.concatenate(rew).mean()),
        "terminations_per_1k": float(np.concatenate(fell).mean() * 1000.0),
        "iteration": float(state.get("iteration", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--command", type=float, default=1.0)
    args = ap.parse_args()

    print(f"{'run':<26}{'torso_up':>9}{'tilt':>7}{'dir':>10}{'fore':>8}"
          f"{'lat':>7}{'speed':>7}{'rew/step':>10}{'term/1k':>9}")
    for run in args.runs:
        if not (run / "checkpoints" / args.checkpoint).exists():
            print(f"{run.name:<26}  no {args.checkpoint}")
            continue
        s = score(run, args.checkpoint, command=args.command)
        print(f"{run.name[:26]:<26}{s['torso_upright']:>9.4f}{s['tilt_deg']:>6.1f}d"
              f"{s['direction']:>10}{s['fore']:>8.3f}{s['lateral']:>7.3f}"
              f"{s['speed']:>7.3f}{s['reward_per_step']:>10.4f}"
              f"{s['terminations_per_1k']:>9.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
