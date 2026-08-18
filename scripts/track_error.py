"""How far along the get-up film the policy actually rides. THE decision metric.

Two earlier metrics failed and both failures are instructive:

* `reward/track` in reward units is meaningless across kernel changes. Widening
  track_sigma_sq 2.0 -> 8.0 multiplied it 9x while the physical error went 27.1 -> 29.8 deg,
  i.e. slightly WORSE.
* Mean joint error in degrees is SURVIVORSHIP-BIASED once early termination exists: riders
  that drift are killed at 26 deg, so averaging over survivors reported a beautiful 5.9 deg
  while those survivors only reached 36% of the film.

What cannot be gamed is DISTANCE TRAVELLED along the reference before losing it: phase 1.0
is a completed get-up, by construction of the film (supine -> stand). Reported per episode,
not per step, so a rider that dies early counts once and honestly.

Reward-unit thresholds are worthless across kernel changes. E44 proved it: widening
track_sigma_sq 2.0 -> 8.0 multiplied the paid `track` by 9x while the physical error went
27.1 -> 29.8 deg, i.e. slightly WORSE. Degrees cannot be inflated by a coefficient.

    python scripts/track_error.py [--run runs/getup-...]
"""
from __future__ import annotations
import argparse, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.getup import GetUpTask  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    run = args.run or sorted(R.glob("runs/getup-*"))[-1]
    cks = sorted((run / "checkpoints").glob("iter_*.pt"))
    if not cks:
        print("no checkpoint yet"); return 0
    ck = cks[-1]
    cfg = Config.load(run / "config.yaml")
    dev = torch.device(resolve_device(cfg.run.device))

    class TrackOnly(GetUpTask):
        """Everything on film, spawned at phase 0: the whole rise is exercised."""
        def reset_pose(self, state, idx, rng):
            starts = self._ref_bounds[:-1]
            clip = rng.integers(0, len(starts), idx.size)
            at = starts[clip]
            ts = state.task_state
            ts["ref_clip"][idx] = clip
            ts["ref_step"][idx] = at
            ts["from_standing"][idx] = False
            ts["from_midrise"][idx] = True
            return self._ref_q[at].copy(), self._ref_v[at].copy()

    t = TrackOnly(cfg.getup)
    env = ThreadedVecEnv(
        R / cfg.env.model_path, t, num_envs=64, num_workers=6,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=5,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode)
    pol = ActorCritic(
        env.obs_dim, env.nu, actor_hidden=tuple(cfg.network.actor_hidden),
        critic_hidden=tuple(cfg.network.critic_hidden), activation=cfg.network.activation,
        init_noise_std=cfg.network.init_noise_std).to(dev)
    pol.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["policy"])
    pol.eval()

    obs = env.reset()
    n = 64
    best = np.zeros(n)          # furthest phase this attempt reached
    finished: list[float] = []  # one entry per completed attempt (death or clip end)
    degs: list[float] = []
    with torch.no_grad():
        for _ in range(args.steps):
            a = pol.act_deterministic(torch.as_tensor(obs, dtype=torch.float32, device=dev))
            was_on = env.state.task_state["ref_clip"] >= 0
            res = env.step(a.cpu().numpy())
            obs = res.obs
            s = env.state
            ts = s.task_state
            on = ts["ref_clip"] >= 0
            _ = was_on
            if on.any():
                rows = np.flatnonzero(on)
                at = ts["ref_step"][rows]
                clips = ts["ref_clip"][rows]
                starts = t._ref_bounds[clips]  # noqa: SLF001
                lens = np.maximum(t._ref_bounds[clips + 1] - starts, 1)  # noqa: SLF001
                ph = (at - starts) / lens
                best[rows] = np.maximum(best[rows], ph)
                err = np.sum((s.qpos[rows][:, t._qadr]  # noqa: SLF001
                              - t._ref_q[at][:, t._qadr]) ** 2, axis=1)  # noqa: SLF001
                degs.append(float(np.degrees(np.sqrt(err / t._qadr.size)).mean()))  # noqa: SLF001
            # An attempt ends when the env terminates (lost the film) or the clip runs out.
            # `on` is read BEFORE the step's autoreset, so a terminated env has already been
            # respawned by now; the previous attempt's best is still in `best` and is
            # harvested here. Only 1 attempt was recorded before this was fixed, because the
            # condition tested `on` from the wrong side of the reset.
            ended = res.terminated | res.truncated | (~on & (best > 0))
            for i in np.flatnonzero(ended):
                if best[i] > 0:
                    finished.append(float(best[i]))
                best[i] = 0.0
    env.close()
    f = np.array(finished) if finished else np.array([0.0])
    print(f"{ck.name}: РАССТОЯНИЕ ПО ФИЛЬМУ за попытку, {len(f)} попыток")
    print(f"  медиана {np.median(f):.2f}   среднее {f.mean():.2f}   "
          f"лучшие 10% {np.percentile(f, 90):.2f}   максимум {f.max():.2f}")
    print(f"  доехали до конца (>=0.95): {float((f >= 0.95).mean()):.1%}")
    if degs:
        print(f"  (справочно, ошибка выживших {np.mean(degs):.1f}°/сустав: смещена отбором)")
    print("  ПОРОГ РЕШЕНИЯ: медиана >= 0.60 к итерации 2500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
