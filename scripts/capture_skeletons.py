"""Build a gait flip-book from checkpoints already on disk.

Two situations this covers, both real:

* A run started before skeleton capture existed, so its process will never write any. Its
  checkpoints are still there, and a capture can be reconstructed from each one.
* A live run whose trainer does not capture, where `--watch` polls for new checkpoints and
  captures them as they appear, without restarting the run.

A capture costs about 0.6 s, so rebuilding a whole run's flip-book takes seconds.

    python scripts/capture_skeletons.py --run runs/<run>
    python scripts/capture_skeletons.py --run runs/<run> --watch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.render import build_render_env  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402
from humanoid_rl.viz import skeleton  # noqa: E402


def build(run_dir: Path, seed: int):
    """The environment and an empty policy shell, reused across every checkpoint."""
    config = Config.load(run_dir / "config.yaml")
    device = torch.device(resolve_device(config.run.device))
    env = build_render_env(
        REPO_ROOT / config.env.model_path, LocomotionTask(config.task), seed=seed
    )
    policy = ActorCritic(
        env.obs_dim,
        env.nu,
        actor_hidden=tuple(config.network.actor_hidden),
        critic_hidden=tuple(config.network.critic_hidden),
        activation=config.network.activation,
        init_noise_std=config.network.init_noise_std,
    ).to(device)
    return config, device, env, policy


def capture_one(run_dir: Path, ckpt_path: Path, env, policy, device) -> Path | None:
    """One checkpoint to one capture, or None if the weights do not fit this environment."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        policy.load_state_dict(state["policy"])
    except RuntimeError as exc:
        # A checkpoint from before an observation change cannot be replayed in the current
        # environment. Skipped rather than guessed at, and said out loud.
        print(f"  skip {ckpt_path.name}: {str(exc).splitlines()[0]}")
        return None
    policy.eval()

    result = skeleton.capture(env, policy, device)
    out = run_dir / "skeletons" / f"iter_{int(state['iteration']):08d}.json"
    return skeleton.write(
        result, out, iteration=int(state["iteration"]), env_steps=int(state["env_steps"])
    )


def sweep(run_dir: Path, env, policy, device, *, quiet: bool = False) -> int:
    """Capture every checkpoint that does not already have one. Returns how many were made."""
    checkpoints = sorted((run_dir / "checkpoints").glob("iter_*.pt"))
    existing = {p.stem for p in (run_dir / "skeletons").glob("*.json")}
    made = 0
    for ckpt in checkpoints:
        # Checkpoint iter_00000650.pt maps to skeleton iter_00000650.json.
        if ckpt.stem in existing:
            continue
        started = time.perf_counter()
        path = capture_one(run_dir, ckpt, env, policy, device)
        if path:
            made += 1
            if not quiet:
                print(f"  {path.name}  {time.perf_counter() - started:.2f}s  "
                      f"{path.stat().st_size / 1024:.0f} KB")
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch", action="store_true",
                    help="keep polling for new checkpoints, for a live run")
    ap.add_argument("--poll", type=float, default=120.0)
    args = ap.parse_args()

    run_dir = args.run.resolve()
    _, device, env, policy = build(run_dir, args.seed)
    try:
        print(f"{run_dir.name}: building flip-book from checkpoints")
        made = sweep(run_dir, env, policy, device)
        print(f"{made} capture(s) written, "
              f"{len(list((run_dir / 'skeletons').glob('*.json')))} total")

        while args.watch:
            time.sleep(args.poll)
            made = sweep(run_dir, env, policy, device, quiet=False)
            if made:
                print(f"  +{made} new", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
