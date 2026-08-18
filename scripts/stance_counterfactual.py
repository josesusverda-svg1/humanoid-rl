"""Is the wide stance a cheap habit, or is it carrying the balance?

    python scripts/stance_counterfactual.py --run runs/final-s0-...

The reward's `feet_distance` corridor is [0.20, 0.45] and human walking is 0.10-0.15, so the
two do not overlap: to score 1.00 on this project's own human-likeness band the policy must
pay 0.225/step, 6.4% of its positive budget, forever. It sits at 0.33 instead, where the
penalty is exactly zero. That much is arithmetic and is already settled.

What arithmetic cannot say is whether narrowing is AFFORDABLE. If the splay is carrying the
balance, moving the corridor buys falls rather than human-likeness, and this project has
changed a reward on a plausible story and regretted it repeatedly. E23 is the precedent that
went the other way: it measured the counterfactual before touching the reward and found the
reward already preferred the thing it was about to be re-tuned for.

So this measures three things on the trained policy, with no training and no reward change:

  1. WHAT IT DOES.       Stance width over a normal rollout, and whether it ever narrows on
                         its own. A policy that transiently reaches 0.15 and survives has
                         already shown narrowing is feasible.
  2. WHETHER IT CAN.     Start episodes with the hips adducted so the feet begin narrow, and
                         measure survival. If it falls, the splay is load-bearing. If it
                         survives, it is not.
  3. WHETHER IT WANTS TO. From a narrow start, does the stance drift back out to 0.33? That
                         is the signature of a reward-driven preference rather than a
                         mechanical necessity, because nothing but the reward is pulling it.

The three answers are independent and the conclusion needs all of them: feasible + survives +
drifts back means the corridor is the binding constraint and moving it is justified. Falls
means it is not, and the corridor stays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask, _stance_width  # noqa: E402

W, LO, HI = -3.0, 0.20, 0.45


def penalty(sep: np.ndarray) -> np.ndarray:
    return W * (np.clip(LO - sep, 0.0, 0.1) + np.clip(sep - HI, 0.0, 0.3))


def build(cfg: Config, n: int, seed: int) -> ThreadedVecEnv:
    return ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path,
        LocomotionTask(cfg.task, seed=seed),
        num_envs=n,
        num_workers=4,
        decimation=cfg.env.decimation,
        max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz,
        action_scale_mode=cfg.env.action_scale_mode,
        seed=seed,
        domain_rand=None,
    )


def rollout(env, policy, device, steps: int, command: float,
            adduct: float = 0.0) -> dict:
    """One deterministic rollout at a held forward command.

    `adduct` closes both hips by that many radians at reset, which starts the feet narrow.
    It is applied ONCE, to the initial state only -- the policy is free to open them again
    from the first control step, and whether it does is the whole question.
    """
    obs = env.reset()
    st = env.state
    if adduct != 0.0:
        # hip_x is the abduction axis. Signs are opposite on the two legs, so closing both
        # means moving them toward each other rather than both in one direction.
        q = st.qpos.copy()
        q[:, 21] += adduct       # right_hip_x
        q[:, 28] -= adduct       # left_hip_x
        env.set_state(q, st.qvel.copy()) if hasattr(env, "set_state") else None
        for i, d in enumerate(env.datas):
            d.qpos[21] += adduct
            d.qpos[28] -= adduct
        obs = env._recompute_obs() if hasattr(env, "_recompute_obs") else obs

    st.task_state["command"][:, 0] = command
    st.task_state["command"][:, 1] = 0.0
    st.task_state["command"][:, 2] = 0.0

    widths, alive_at = [], np.full(env.num_envs, steps, dtype=int)
    dead = np.zeros(env.num_envs, dtype=bool)
    for t in range(steps):
        st.task_state["command"][:, 0] = command
        st.task_state["command"][:, 1] = 0.0
        st.task_state["command"][:, 2] = 0.0
        with torch.no_grad():
            a = policy.act_deterministic(torch.from_numpy(obs).to(device)).cpu().numpy()
        res = env.step(a)
        obs = res.obs
        widths.append(_stance_width(env.state).copy())
        newly = res.terminated & ~dead
        alive_at[newly] = t
        dead |= res.terminated
        if dead.all():
            break
    w = np.stack(widths)                       # (T, N)
    return {
        "width_mean": float(w[~np.isnan(w)].mean()),
        "width_min": float(np.percentile(w, 1)),
        "width_trace": w.mean(axis=1),
        "fell": float(dead.mean()),
        "steps_survived": float(alive_at.mean()),
        "max_steps": len(widths),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--command", type=float, default=1.0)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    cfg = Config.load(run / "config.yaml")
    ck = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu",
                    weights_only=False)
    device = torch.device("cpu")

    env = build(cfg, args.envs, seed=99)
    policy = ActorCritic(env.obs_dim, env.nu,
                         actor_hidden=cfg.network.actor_hidden,
                         critic_hidden=cfg.network.critic_hidden,
                         activation=cfg.network.activation,
                         init_noise_std=cfg.network.init_noise_std).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    print(f"\n=== stance counterfactual: {run.name}/{args.checkpoint}, "
          f"command {args.command} m/s, {args.envs} envs ===\n")

    print("1. WHAT IT DOES, unperturbed")
    base = rollout(env, policy, device, args.steps, args.command)
    print(f"   stance mean {base['width_mean']:.3f} m, 1st percentile {base['width_min']:.3f} m")
    print(f"   falls {base['fell'] * 100:.0f}%, survived {base['steps_survived']:.0f} of "
          f"{base['max_steps']} steps")
    print(f"   reward from the feet_distance term: {penalty(np.array([base['width_mean']]))[0]:+.3f}/step")
    print(f"   {'it never narrows on its own' if base['width_min'] > 0.20 else 'it DOES reach human width transiently'}")

    print("\n2. WHETHER IT CAN, hips adducted at reset")
    for rad in (0.10, 0.20, 0.30):
        r = rollout(env, policy, device, args.steps, args.command, adduct=rad)
        print(f"   adduct {rad:.2f} rad -> stance mean {r['width_mean']:.3f} m, "
              f"falls {r['fell'] * 100:>3.0f}%, survived {r['steps_survived']:>4.0f} steps")

    print("\n3. WHETHER IT WANTS TO: stance over time from the narrowest start")
    r = rollout(env, policy, device, args.steps, args.command, adduct=0.30)
    tr = r["width_trace"]
    marks = [0, len(tr) // 8, len(tr) // 4, len(tr) // 2, len(tr) - 1]
    print("   step  " + "  ".join(f"{m:>5d}" for m in marks))
    print("   width " + "  ".join(f"{tr[m]:>5.3f}" for m in marks))
    drift = tr[-1] - tr[0]
    print(f"\n   drift over the rollout: {drift:+.3f} m")
    print(f"   {'DRIFTS BACK OUT -- a preference, not a necessity' if drift > 0.02 else 'stays narrow -- the splay is not being actively sought'}")

    env.close()
    print(f"\n{'-' * 78}")
    print("Conclusion needs all three: feasible AND survives AND drifts back means the")
    print("corridor is the binding constraint. Falling at a narrow start means it is not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
