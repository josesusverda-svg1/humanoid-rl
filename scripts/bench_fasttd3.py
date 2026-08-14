"""Tune FastTD3's batch size and update ratio for THIS machine, by measuring.

The published FastTD3 hyperparameters were tuned on an A100, where a 32,768-sample update is
cheap and the physics runs on the same GPU. Our shape is the opposite: MuJoCo physics on ten
CPU performance cores, gradients on the M3 Max GPU through MPS, and MPS is the weaker half.
Copying a batch size across that gap is exactly the kind of unchecked borrowing that gave us
an 8-second episode and a 2-4 strides/s cadence.

What this measures, on the real networks at the real observation size:

* seconds per gradient update at each batch size, so we can see whether MPS actually
  amortises a large batch or just takes proportionally longer
* seconds per vectorised physics step at 4096 environments
* the resulting end-to-end throughput and the split between the two

    python scripts/bench_fasttd3.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.fasttd3 import FastTD3, FastTD3Config  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402

BATCHES = [4096, 8192, 16384, 32768, 65536]


def sync(device: torch.device) -> None:
    """MPS queues work asynchronously; without this the timer measures enqueue, not compute."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_updates(cfg: Config, device: torch.device, batch: int,
                 atoms: int, critic: tuple[int, ...]) -> float:
    td3 = replace(cfg.fasttd3, batch_size=batch, num_atoms=atoms, critic_hidden=critic,
                  buffer_size_per_env=max(batch // 256 + 4, 8))
    agent = FastTD3(108, 28, td3, num_envs=256, device=device)
    for _ in range(max(batch // 256 + 4, 8)):
        agent.buffer.add(torch.randn(256, 108, device=device),
                         torch.randn(256, 28, device=device).clamp(-1, 1),
                         torch.rand(256, device=device) * 3,
                         torch.randn(256, 108, device=device),
                         torch.ones(256, device=device))
    for _ in range(3):          # warm up: MPS compiles kernels on first use
        agent.update()
    sync(device)
    start = time.perf_counter()
    for _ in range(10):
        agent.update()
    sync(device)
    return (time.perf_counter() - start) / 10.0


def main() -> int:
    cfg = Config.load(REPO_ROOT / "configs" / "fasttd3.yaml")
    device = torch.device(resolve_device(cfg.run.device))
    print(f"device: {device}\n")

    # --- physics, which is the part we are NOT changing.
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, LocomotionTask(cfg.task), num_envs=cfg.env.num_envs,
        num_workers=cfg.env.num_workers, decimation=cfg.env.decimation,
        max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=0, domain_rand=cfg.domain_rand,
    )
    env.reset()
    zero = np.zeros((cfg.env.num_envs, env.nu), dtype=np.float32)
    for _ in range(5):
        env.step(zero)
    start = time.perf_counter()
    for _ in range(20):
        env.step(zero)
    physics = (time.perf_counter() - start) / 20.0
    env.close()
    print(f"physics: {physics * 1000:.1f} ms per vectorised step "
          f"({cfg.env.num_envs} envs) = {cfg.env.num_envs / physics:,.0f} env steps/s alone\n")

    # --- gradient updates at the published network size.
    print(f"{'batch':>8}{'ms/update':>12}{'us/sample':>12}{'end-to-end sps':>16}{'gpu share':>11}")
    best = None
    for batch in BATCHES:
        try:
            secs = time_updates(cfg, device, batch, cfg.fasttd3.num_atoms,
                                tuple(cfg.fasttd3.critic_hidden))
        except RuntimeError as exc:
            print(f"{batch:>8}  failed: {exc}")
            continue
        iteration = physics + secs * cfg.fasttd3.num_updates
        sps = cfg.env.num_envs / iteration
        share = secs * cfg.fasttd3.num_updates / iteration
        print(f"{batch:>8}{secs * 1000:>12.1f}{secs / batch * 1e6:>12.2f}"
              f"{sps:>16,.0f}{share:>10.0%}")
        if best is None or sps > best[1]:
            best = (batch, sps)

    print(f"\nPPO on this machine: 50,244 env steps/s, physics 57% of the time.")
    if best:
        print(f"Best FastTD3 batch here: {best[0]:,} at {best[1]:,.0f} env steps/s.")
        print("Read the us/sample column: if it is flat, the GPU is amortising the batch and "
              "bigger is free. If it rises, MPS is saturated and the large batch is pure cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
