"""Which of the 13 standing conjuncts is blocking success, measured one at a time.

`standing_frac` being zero says nothing about WHY. All thirteen must hold at once, so a single
stubborn clause pins the whole thing at zero while the other twelve look healthy. This scores
each separately, worst first, so the next fix aims at the actual blocker instead of the most
recently discussed one.

Found this way, on the run that reached a pelvis height of 0.752 against a 0.745 threshold:
stance width held 0.3% of the time and per-foot load 7.5%, while total foot load held 47.7%.
The gap between 47.7 and 7.5 was the whole diagnosis: the weight was there, all on one leg.
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
from humanoid_rl.tasks.getup import GetUpTask  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    run = args.run or sorted((REPO_ROOT / "runs").glob("getup-*"))[-1]
    # A run has no best.pt until its first evaluation. Say so in one line rather than
    # dumping a traceback: this is called every ten minutes and a stack trace in that slot
    # trains the reader to stop looking at the output.
    # The NEWEST checkpoint, not best.pt.
    #
    # best.pt only moves when the eval return sets a record, so on a run that peaks early and
    # then degrades it freezes. Measured here: best.pt sat at iteration 300 while the run was
    # at 900, so nine minutes of "looking at the frames" were nine minutes of looking at a
    # 600-iteration-old policy while reporting current metrics beside it. Watching the wrong
    # object is worse than not watching.
    ckpts = sorted((run / "checkpoints").glob("iter_*.pt"))
    best = run / "checkpoints" / "best.pt"
    ckpt_path = ckpts[-1] if ckpts else (best if best.exists() else None)
    if ckpt_path is None:
        print(f"   no checkpoint in {run.name} yet (first evaluation not reached)")
        return 0
    cfg = Config.load(run / "config.yaml")
    g = cfg.getup
    device = torch.device(resolve_device(cfg.run.device))
    task = GetUpTask(g)
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=args.envs, num_workers=6,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=7,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode,
    )
    policy = ActorCritic(
        env.obs_dim, env.nu, actor_hidden=tuple(cfg.network.actor_hidden),
        critic_hidden=tuple(cfg.network.critic_hidden), activation=cfg.network.activation,
        init_noise_std=cfg.network.init_noise_std).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device,
                                      weights_only=False)["policy"])
    policy.eval()

    obs = env.reset()
    acc: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i in range(args.steps):
            action = policy.act_deterministic(
                torch.as_tensor(obs, dtype=torch.float32, device=device))
            obs = env.step(action.cpu().numpy()).obs
            if i < args.steps // 4:      # let the settle transient pass
                continue
            s = env.state
            fore, side = task._lean(s)   # noqa: SLF001
            bw = task._body_weight       # noqa: SLF001
            fz, hz = s.key_body_pos[:, 0:2, 2], s.key_body_pos[:, 2:4, 2]
            # HEIGHT-MASKED, matching _standing_parts (E34): the raw sensor reads multi-BW
            # ghosts from mid-air self-contact, and this script's whole job is per-clause
            # blocker attribution. Measured unmasked vs masked on the E32 launch policy:
            # clause 7 over-reported 405 vs 317 steps, clause 8 22 vs 12.
            F = np.where(fz <= g.u_foot_height, s.foot_force[:, :2], 0.0)
            knee = s.qpos[:, task._knee_qadr]  # noqa: SLF001
            rel = s.key_body_pos[:, 0, :2] - s.key_body_pos[:, 1, :2]
            sep = np.linalg.norm(rel, axis=1)
            clauses = {
                "1  pelvis high": s.root_height >= g.u_root_height_frac * task._standing_height,  # noqa: SLF001
                "2  head high": s.head_height_ratio >= g.u_head_ratio,
                "3  pelvis level": s.gravity_body[:, 2] <= g.u_pelvis_upright,
                "4  torso not inverted": s.torso_upright >= g.u_torso_upright,
                "5  not folded forward": np.abs(fore) <= g.u_lean,
                "6  not folded sideways": np.abs(side) <= g.u_lean,
                "7  feet carry 60%": F.sum(1) >= g.u_force_total_bw * bw,
                "8  each foot 20%": F.min(1) >= g.u_force_min_bw * bw,
                "9  feet on the floor": fz.max(1) <= g.u_foot_height,
                "10 hands not propping": hz.min(1) >= g.u_hand_height,
                "11 knees straight": knee.max(1) <= g.u_knee,
                "12 stance width": (sep >= g.u_sep_min) & (sep <= g.u_sep_max),
                "13 not ballistic": (np.linalg.norm(s.qvel[:, 0:3], axis=1) <= g.u_lin_speed)
                & (np.linalg.norm(s.qvel[:, 3:6], axis=1) <= g.u_ang_speed),
            }
            for k, v in clauses.items():
                acc[k] = acc.get(k, 0.0) + float(v.mean())
            n += 1
    env.close()

    for k, v in sorted(acc.items(), key=lambda x: x[1]):
        share = v / max(n, 1)
        bar = "#" * int(share * 30)
        print(f"   {k:<24}{share:>7.1%}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
