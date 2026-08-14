"""Render a checkpoint to an mp4 you can actually watch.

    .venv/bin/python scripts/render.py                       # newest run, best checkpoint
    .venv/bin/python scripts/render.py --run runs/humanoid-... --checkpoint best.pt
    .venv/bin/python scripts/render.py --out /tmp/gait.mp4 --width 1280 --height 960

Safe to run while training is in progress: it builds its own single-environment simulation
and encodes on Apple's media engine, so it takes almost nothing from the training run.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.render import build_render_env, render_episode  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def newest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()]
    if not candidates:
        raise FileNotFoundError(f"no runs with a config.yaml found in {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: newest)")
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skin", action="store_true",
                    help="render the human skin instead of the collision capsules "
                         "(build it first with scripts/build_skin.py)")
    args = ap.parse_args()

    run_dir = args.run or newest_run(REPO_ROOT / "runs")
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        available = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
        raise FileNotFoundError(f"{ckpt_path} not found. Available: {available}")

    config = Config.load(run_dir / "config.yaml")

    # The skinned model is the same physics with a render-only skin attached, verified
    # bitwise identical by `build_skin.py --verify`, so swapping it in cannot change what
    # the policy does. Only its appearance changes.
    model_path = REPO_ROOT / config.env.model_path
    if args.skin:
        skinned = REPO_ROOT / "assets" / "humanoid_skinned.xml"
        if not skinned.exists():
            raise FileNotFoundError(
                f"{skinned} not found. Build it with:\n"
                "    .venv/bin/python scripts/build_skin.py --verify"
            )
        model_path = skinned
    device = torch.device(resolve_device(config.run.device))

    task = LocomotionTask(config.task)
    env = build_render_env(model_path, task, seed=args.seed)

    policy = ActorCritic(
        env.obs_dim,
        env.nu,
        actor_hidden=tuple(config.network.actor_hidden),
        critic_hidden=tuple(config.network.critic_hidden),
        activation=config.network.activation,
        init_noise_std=config.network.init_noise_std,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(state["policy"])

    out = args.out or (run_dir / "videos" / f"{ckpt_path.stem}_iter{state['iteration']:08d}.mp4")
    print(f"run        {run_dir.name}")
    print(f"checkpoint {args.checkpoint}  (iteration {state['iteration']:,}, "
          f"{state['env_steps']:,} env steps)")
    print(f"rendering  {args.width}x{args.height} @ {args.fps} fps -> {out}")

    t0 = time.perf_counter()
    try:
        result = render_episode(
            env,
            policy,
            device,
            out,
            width=args.width,
            height=args.height,
            fps=args.fps,
            overlay=not args.no_overlay,
        )
    finally:
        env.close()

    print(f"\ndone in {time.perf_counter() - t0:.1f}s")
    print(f"  {result.frames} frames, {result.seconds:.1f}s of footage")
    print(f"  mean speed {result.mean_speed:.2f} m/s, mean foot slip {result.mean_slip:.2f} m/s")
    print(f"  fell: {result.fell}" + (f" at t={result.fell_at:.1f}s" if result.fell else ""))
    print(f"  {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
