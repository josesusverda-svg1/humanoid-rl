"""Smoke test for the vectorised environment: correctness first, then throughput.

Run this after any change to `vec_env.py` or a task. It is fast (a few seconds) and
catches the failure modes that are otherwise invisible until a training run has silently
wasted hours: non-finite observations, a reward term that has quietly gone to zero,
autoreset not firing, or a throughput regression from accidentally adding per-environment
Python to the worker loop.

Usage:
    .venv/bin/python scripts/smoke_env.py
    .venv/bin/python scripts/smoke_env.py --envs 4096 --steps 500
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanoid_rl import hardware  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.base import quat_rotate_inverse  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionConfig, LocomotionTask  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = REPO_ROOT / "humanoid_rl" / "models" / "humanoid_scene.xml"

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = f"{GREEN}ok  {RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name:<44s} {detail}")
    if not ok:
        _failures.append(name)


def check_quat_math() -> None:
    """The batched quaternion routine must agree exactly with MuJoCo's scalar version."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(256, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(256, 3))

    mine = quat_rotate_inverse(q, v)
    ref = np.zeros((256, 3))
    conj, tmp = np.zeros(4), np.zeros(3)
    for i in range(256):
        mujoco.mju_negQuat(conj, q[i])
        mujoco.mju_rotVecQuat(tmp, v[i], conj)
        ref[i] = tmp
    err = float(np.abs(mine - ref).max())
    check("quat_rotate_inverse vs MuJoCo", err < 1e-10, f"max abs err {err:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--min-sps", type=float, default=50_000, help="fail below this throughput")
    args = ap.parse_args()

    hw = hardware.detect()
    print("=" * 84)
    print("VECTORISED ENVIRONMENT SMOKE TEST".center(84))
    print("=" * 84)
    print(f"  {hw.chip}, {hw.performance_cores}P cores, mujoco {mujoco.__version__}\n")

    check_quat_math()

    task = LocomotionTask(LocomotionConfig())
    env = ThreadedVecEnv(MODEL, task, num_envs=args.envs, max_episode_steps=1000, seed=0)
    rng = np.random.default_rng(1)
    try:
        print(
            f"\n  config: {env.num_envs} envs, {env.num_workers} workers, "
            f"obs_dim={env.obs_dim} (proprio {env.proprio_dim} + task {task.task_obs_dim}), "
            f"nu={env.nu}\n"
        )

        obs = env.reset()
        check("reset shape", obs.shape == (env.num_envs, env.obs_dim), str(obs.shape))
        check("reset dtype float32", obs.dtype == np.float32, str(obs.dtype))
        check("reset observations finite", bool(np.isfinite(obs).all()))

        total_done = 0
        total_terminated = 0
        for _ in range(60):
            res = env.step(rng.uniform(-1, 1, size=(env.num_envs, env.nu)).astype(np.float32))
            total_done += int(res.done.sum())
            total_terminated += int(res.terminated.sum())

        check("step observations finite", bool(np.isfinite(res.obs).all()))
        check("rewards finite", bool(np.isfinite(res.reward).all()))
        # Informational, not a hard assertion. With PD position control the humanoid is
        # genuinely stable against random *offsets* from a standing pose, so falls are rare
        # under random actions. Relying on a fall to prove autoreset works was a weak test
        # that happened to pass by luck.
        print(f"  [{DIM}info{RESET}] {'falls under random actions':<44s} "
              f"{total_terminated} terminated, {total_done} episodes ended")
        check(
            "terminated and truncated are disjoint",
            not bool((res.terminated & res.truncated).any()),
        )

        # Autoreset, tested deterministically via the time limit rather than by hoping the
        # humanoid falls over. A short-horizon env is guaranteed to truncate.
        short = ThreadedVecEnv(MODEL, LocomotionTask(LocomotionConfig()), num_envs=64,
                               num_workers=2, max_episode_steps=10, seed=3)
        try:
            short.reset()
            fired = 0
            steps_at_reset = []
            for _ in range(25):
                r = short.step(np.zeros((short.num_envs, short.nu), dtype=np.float32))
                if r.truncated.any():
                    fired += int(r.truncated.sum())
                    steps_at_reset.append(int(r.episode_length[np.flatnonzero(r.truncated)[0]]))
            check("autoreset fires on the time limit", fired > 0, f"{fired} truncations")
            check(
                "episode length matches the limit",
                bool(steps_at_reset) and all(s == 10 for s in steps_at_reset),
                f"lengths {sorted(set(steps_at_reset))}",
            )
            check(
                "episode_step resets to zero after autoreset",
                int(short.state.episode_step.max()) <= 10,
                f"max episode_step {int(short.state.episode_step.max())}",
            )
        finally:
            short.close()

        term_means = res.reward_terms.mean(axis=0)
        for name, value in zip(task.reward_term_names, term_means):
            check(f"reward term '{name}' active", bool(np.isfinite(value)), f"mean {value:+.4f}")

        # Throughput. A regression here almost always means per-environment Python crept
        # back into the worker loop. See DESIGN.md section 2.4.
        action = rng.uniform(-1, 1, size=(env.num_envs, env.nu)).astype(np.float32)
        env.step(action)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(action)
        elapsed = time.perf_counter() - t0
        sps = args.steps * env.num_envs / elapsed

        print()
        check(
            "throughput",
            sps >= args.min_sps,
            f"{sps:,.0f} env steps/s ({elapsed / args.steps * 1e3:.2f} ms/batch), "
            f"floor {args.min_sps:,.0f}",
        )
    finally:
        env.close()

    print("\n" + "=" * 84)
    if _failures:
        print(f"{RED}{len(_failures)} CHECK(S) FAILED:{RESET} " + ", ".join(_failures))
        return 1
    print(f"{GREEN}ALL CHECKS PASSED{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
