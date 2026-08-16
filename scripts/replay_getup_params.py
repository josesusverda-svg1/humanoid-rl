"""Replay a searched get-up trajectory: per-conjunct blocker table, height trace, frames.

The search reports one number, the fraction of the hold spent standing. When that number is
0.01 while the pelvis sits at 0.874 of a 0.877 standing height, the number says the body is
up and the predicate disagrees, and the only useful question is WHICH of the thirteen
clauses disagrees. `getup_conjuncts.py` answers that for a trained policy; this answers it
for a searched trajectory, using the identical clause definitions.

    python scripts/replay_getup_params.py --family supine
    python scripts/replay_getup_params.py --family supine --out /tmp/getup_search.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from search_getup_trajectory import SearchTask  # noqa: E402


def clauses(task, state, g) -> dict[str, np.ndarray]:
    """The thirteen standing conjuncts, clause by clause. Same definitions as the task."""
    fore, side = task._lean(state)                                  # noqa: SLF001
    bw = task._body_weight                                          # noqa: SLF001
    fz, hz = state.key_body_pos[:, 0:2, 2], state.key_body_pos[:, 2:4, 2]
    F = np.where(fz <= g.u_foot_height, state.foot_force[:, :2], 0.0)
    knee = state.qpos[:, task._knee_qadr]                           # noqa: SLF001
    rel = state.key_body_pos[:, 0, :2] - state.key_body_pos[:, 1, :2]
    sep = np.linalg.norm(rel, axis=1)
    return {
        "1  pelvis high": state.root_height >= g.u_root_height_frac * task._standing_height,  # noqa: SLF001
        "2  head high": state.head_height_ratio >= g.u_head_ratio,
        "3  pelvis level": state.gravity_body[:, 2] <= g.u_pelvis_upright,
        "4  torso not inverted": state.torso_upright >= g.u_torso_upright,
        "5  not folded forward": np.abs(fore) <= g.u_lean,
        "6  not folded sideways": np.abs(side) <= g.u_lean,
        "7  feet carry 60%": F.sum(1) >= g.u_force_total_bw * bw,
        "8  each foot 20%": F.min(1) >= g.u_force_min_bw * bw,
        "9  feet on the floor": fz.max(1) <= g.u_foot_height,
        "10 hands not propping": hz.min(1) >= g.u_hand_height,
        "11 knees straight": knee.max(1) <= g.u_knee,
        "12 stance width": (sep >= g.u_sep_min) & (sep <= g.u_sep_max),
        "13 not ballistic": (np.linalg.norm(state.qvel[:, 0:3], axis=1) <= g.u_lin_speed)
        & (np.linalg.norm(state.qvel[:, 3:6], axis=1) <= g.u_ang_speed),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="supine")
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("/tmp/getup_search.png"))
    ap.add_argument("--shots", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.params or REPO_ROOT / f"data/fallen/getup_params_{args.family}.npz")
    params = d["params"]
    # Timing comes from the FILE, never from the current value of the search module's
    # constants. Replaying a searched trajectory under different segment lengths is not a
    # replay: the same waypoints stretched from a 3.75 s ramp to a 5.5 s one put the pelvis
    # at 0.085 where the search recorded 0.875.
    seg_seconds = float(d["seg_seconds"])
    hold_seconds = float(d["hold_seconds"])
    cfg = Config.load(REPO_ROOT / "configs" / "getup.yaml")
    g = cfg.getup

    task = SearchTask(replace(g, ref_path="", ball_enabled=False, standing_reset_frac=0.0,
                              midrise_reset_frac=0.0, rising_reset_frac=0.0,
                              track_reset_frac=0.0))
    # The start pose the search actually used, stored in the same file. Re-drawing it from
    # the seed would only work for the FIRST family: every later draw sits behind however many
    # normal deviates the search consumed, so a shared seed reproduces the sequence and not
    # any one element of it.
    task.start_q = d["start_q"]
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=1, num_workers=1,
        decimation=cfg.env.decimation, max_episode_steps=1_000_000,
        action_filter_hz=cfg.env.action_filter_hz, seed=args.seed,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode)
    task._exam_level = len(g.exam_levels) - 1                       # noqa: SLF001

    seg = max(1, int(round(seg_seconds / env.dt)))
    hold = max(1, int(round(hold_seconds / env.dt)))
    total = params.shape[0] * seg + hold
    model = env.prepared.model
    data = mujoco.MjData(model)
    shot_at = set(np.linspace(0, total - 1, args.shots).astype(int).tolist())

    env.reset()
    acc: dict[str, float] = {}
    n_hold = 0
    trace, speeds, imgs = [], [], []
    cur = np.zeros((1, env.nu), dtype=np.float32)
    with mujoco.Renderer(model, height=420, width=340) as renderer:
        for step in range(total):
            in_ramp = step < params.shape[0] * seg
            if in_ramp:
                w, t = divmod(step, seg)
                goal = params[w][None].astype(np.float32)
                if t == 0 and w > 0:
                    cur = params[w - 1][None].astype(np.float32)
                a = cur + (goal - cur) * ((t + 1) / seg)
            else:
                a = np.zeros((1, env.nu), dtype=np.float32)
            env.step(np.clip(a, -1.0, 1.0))
            s = env.state
            trace.append(float(s.root_height[0]))
            speeds.append(float(np.linalg.norm(s.qvel[0, 0:3])))
            if not in_ramp:
                for k, v in clauses(task, s, g).items():
                    acc[k] = acc.get(k, 0.0) + float(v[0])
                n_hold += 1
            if step in shot_at:
                data.qpos[:] = s.qpos[0]
                data.qvel[:] = s.qvel[0]
                mujoco.mj_forward(model, data)
                cam = mujoco.MjvCamera()
                cam.distance, cam.elevation, cam.azimuth = 3.2, -10.0, 110.0
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.55]
                renderer.update_scene(data, camera=cam)
                imgs.append((step * env.dt, float(s.root_height[0]),
                             renderer.render().copy()))
    env.close()

    print(f"{args.family}: {params.shape[0]} waypoints, ramp "
          f"{params.shape[0] * seg_seconds:.2f} s, hold {hold_seconds:.2f} s "
          f"at the nominal stand (timings read from the file)")
    ramp = params.shape[0] * seg
    print(f"pelvis: start {trace[0]:.3f}  peak {max(trace):.3f}  "
          f"end {trace[-1]:.3f}  (standing height {task._standing_height:.3f})")  # noqa: SLF001
    # How FAST it got up. A kip-up and a get-up end in the same pose and are told apart only
    # here: the first complete search won every starting orientation with a rise that moved
    # the pelvis 0.65 m in 0.6 s and then failed the "not ballistic" conjunct for a third of
    # the hold.
    print(f"root speed: peak during the rise {max(speeds[:ramp]):.2f} m/s, "
          f"peak during the hold {max(speeds[ramp:]):.2f} m/s")
    print("\nstanding conjuncts over the hold, worst first:")
    for k, v in sorted(acc.items(), key=lambda x: x[1]):
        share = v / max(n_hold, 1)
        print(f"   {k:<24}{share:>7.1%}  {'#' * int(share * 30)}")

    sheet = Image.new("RGB", (sum(im.shape[1] for _t, _z, im in imgs), imgs[0][2].shape[0]),
                      (255, 255, 255))
    x = 0
    for _t, _z, im in imgs:
        sheet.paste(Image.fromarray(im), (x, 0))
        x += im.shape[1]
    sheet.save(args.out)
    print("\nframes at " + "  ".join(f"{t:.1f}s z={z:.2f}" for t, z, _ in imgs))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
