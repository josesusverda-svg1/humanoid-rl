"""Phase 0 benchmark: where should the neural networks run?

Candidates: PyTorch on CPU, PyTorch on MPS (the Metal GPU backend), and Apple MLX.

    MPS (Metal Performance Shaders): Apple's GPU compute framework. PyTorch's "mps"
    device runs tensor operations on the 30-core GPU.
    MLX: Apple's own array framework, designed around unified memory, with lazy
    evaluation and no explicit host-to-device copies.

The naive assumption is "GPU is faster". For this workload that is not obvious. Our
networks are small multi-layer perceptrons (roughly 76 -> 512 -> 512 -> 21), and at
small batch sizes the fixed cost of launching a GPU kernel can exceed the arithmetic
itself, making the CPU win. The crossover batch size is what actually decides the
architecture, so we measure it rather than guess.

Two regimes are measured separately because they have different characteristics:

1. INFERENCE. Runs every control step on a (n_envs, obs_dim) batch. Latency-critical:
   it sits directly in the environment stepping loop, so overhead here is paid
   thousands of times per second.
2. TRAINING. The PPO update: forward, backward, optimiser step on a minibatch.
   Throughput-critical, larger batches, run in a burst between rollout collection.

Usage:
    .venv/bin/python scripts/bench_networks.py
    .venv/bin/python scripts/bench_networks.py --json bench_results/networks.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanoid_rl import hardware  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Network shape used throughout the project. Deliberately small, which is standard for
# locomotion policies and is exactly why the CPU-versus-GPU question is non-trivial.
OBS_DIM = 76
ACT_DIM = 21
HIDDEN = (512, 512)


@dataclass
class NetResult:
    backend: str
    mode: str  # "inference" or "train"
    batch: int
    per_call_ms: float
    calls_per_sec: float
    samples_per_sec: float
    notes: str = ""


def _bench_loop(fn, sync, warmup: int = 20, iters: int = 100) -> float:
    """Time `fn`, calling `sync` after each iteration so async backends are measured fairly.

    Both MPS and MLX are asynchronous: the call returns before the GPU has finished.
    Without an explicit synchronise the measurement would record queueing time only.
    """
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


# --------------------------------------------------------------------------------------
# PyTorch (CPU and MPS)
# --------------------------------------------------------------------------------------


def bench_torch(device: str, batches: list[int], train_batches: list[int]) -> list[NetResult]:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    results: list[NetResult] = []
    dev = torch.device(device)

    def sync() -> None:
        if device == "mps":
            torch.mps.synchronize()

    def make_mlp(out_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = OBS_DIM
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers).to(dev)

    actor = make_mlp(ACT_DIM)
    critic = make_mlp(1)

    # --- inference ---
    with torch.no_grad():
        for b in batches:
            x = torch.randn(b, OBS_DIM, device=dev)
            dt = _bench_loop(lambda: actor(x), sync)
            results.append(
                NetResult(
                    backend=f"torch-{device}",
                    mode="inference",
                    batch=b,
                    per_call_ms=dt * 1e3,
                    calls_per_sec=1.0 / dt,
                    samples_per_sec=b / dt,
                    notes="actor forward, no_grad",
                )
            )

    # --- inference including the numpy handoff ---
    # The environment produces numpy arrays on the CPU. This measures the real cost of
    # getting them to the network and the actions back, which is what the loop pays.
    with torch.no_grad():
        for b in batches:
            host = np.random.randn(b, OBS_DIM).astype(np.float32)

            def roundtrip() -> np.ndarray:
                t = torch.from_numpy(host).to(dev)
                out = actor(t)
                return out.cpu().numpy()

            dt = _bench_loop(roundtrip, sync)
            results.append(
                NetResult(
                    backend=f"torch-{device}",
                    mode="inference+transfer",
                    batch=b,
                    per_call_ms=dt * 1e3,
                    calls_per_sec=1.0 / dt,
                    samples_per_sec=b / dt,
                    notes="numpy -> device -> forward -> numpy",
                )
            )

    # --- training step ---
    opt = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    for b in train_batches:
        x = torch.randn(b, OBS_DIM, device=dev)
        adv = torch.randn(b, device=dev)
        ret = torch.randn(b, device=dev)
        old_lp = torch.randn(b, device=dev)

        def train_step() -> None:
            mean = actor(x)
            value = critic(x).squeeze(-1)
            logp = -(mean.pow(2).sum(-1))  # stand-in for a Gaussian log-prob
            ratio = torch.exp(logp - old_lp)
            pg = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            vf = (value - ret).pow(2).mean()
            loss = pg + 0.5 * vf
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        dt = _bench_loop(train_step, sync, warmup=10, iters=50)
        results.append(
            NetResult(
                backend=f"torch-{device}",
                mode="train",
                batch=b,
                per_call_ms=dt * 1e3,
                calls_per_sec=1.0 / dt,
                samples_per_sec=b / dt,
                notes="PPO-shaped fwd+bwd+AdamW",
            )
        )

    return results


# --------------------------------------------------------------------------------------
# MLX
# --------------------------------------------------------------------------------------


def bench_mlx(batches: list[int], train_batches: list[int]) -> list[NetResult]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(0)
    results: list[NetResult] = []

    class MLP(nn.Module):
        def __init__(self, out_dim: int) -> None:
            super().__init__()
            self.l1 = nn.Linear(OBS_DIM, HIDDEN[0])
            self.l2 = nn.Linear(HIDDEN[0], HIDDEN[1])
            self.l3 = nn.Linear(HIDDEN[1], out_dim)

        def __call__(self, x: mx.array) -> mx.array:
            x = nn.elu(self.l1(x))
            x = nn.elu(self.l2(x))
            return self.l3(x)

    actor = MLP(ACT_DIM)
    critic = MLP(1)
    mx.eval(actor.parameters(), critic.parameters())

    def sync() -> None:
        mx.synchronize()

    # --- inference ---
    for b in batches:
        x = mx.random.normal((b, OBS_DIM))
        mx.eval(x)

        def fwd() -> None:
            mx.eval(actor(x))

        dt = _bench_loop(fwd, sync)
        results.append(
            NetResult(
                backend="mlx",
                mode="inference",
                batch=b,
                per_call_ms=dt * 1e3,
                calls_per_sec=1.0 / dt,
                samples_per_sec=b / dt,
                notes="actor forward",
            )
        )

    # --- inference including the numpy handoff ---
    for b in batches:
        host = np.random.randn(b, OBS_DIM).astype(np.float32)

        def roundtrip() -> None:
            t = mx.array(host)
            out = actor(t)
            mx.eval(out)
            np.asarray(out)

        dt = _bench_loop(roundtrip, sync)
        results.append(
            NetResult(
                backend="mlx",
                mode="inference+transfer",
                batch=b,
                per_call_ms=dt * 1e3,
                calls_per_sec=1.0 / dt,
                samples_per_sec=b / dt,
                notes="numpy -> mx.array -> forward -> numpy",
            )
        )

    # --- training step ---
    for b in train_batches:
        x = mx.random.normal((b, OBS_DIM))
        adv = mx.random.normal((b,))
        ret = mx.random.normal((b,))
        old_lp = mx.random.normal((b,))
        mx.eval(x, adv, ret, old_lp)

        opt = optim.AdamW(learning_rate=3e-4)

        def loss_fn(model_a, model_c):
            mean = model_a(x)
            value = model_c(x).squeeze(-1)
            logp = -(mean**2).sum(-1)
            ratio = mx.exp(logp - old_lp)
            pg = -mx.minimum(ratio * adv, mx.clip(ratio, 0.8, 1.2) * adv).mean()
            vf = ((value - ret) ** 2).mean()
            return pg + 0.5 * vf

        grad_fn = nn.value_and_grad(actor, lambda m: loss_fn(m, critic))

        def train_step() -> None:
            _, grads = grad_fn(actor)
            opt.update(actor, grads)
            mx.eval(actor.parameters(), opt.state)

        dt = _bench_loop(train_step, sync, warmup=10, iters=50)
        results.append(
            NetResult(
                backend="mlx",
                mode="train",
                batch=b,
                per_call_ms=dt * 1e3,
                calls_per_sec=1.0 / dt,
                samples_per_sec=b / dt,
                notes="PPO-shaped fwd+bwd+AdamW (actor only)",
            )
        )

    return results


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--train-batches", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    ap.add_argument("--json", type=Path, default=REPO_ROOT / "bench_results" / "networks.json")
    args = ap.parse_args()

    hw = hardware.detect()
    import torch

    print("=" * 92)
    print("NEURAL NETWORK BACKEND BENCHMARK".center(92))
    print("=" * 92)
    print(f"  {hw.chip}: {hw.performance_cores}P + {hw.efficiency_cores}E cores, "
          f"{hw.gpu_cores} GPU cores, {hw.memory_gb:.0f} GB unified memory")
    print(f"  torch {torch.__version__}   mps_available={torch.backends.mps.is_available()}")
    print(f"  network: {OBS_DIM} -> {' -> '.join(str(h) for h in HIDDEN)} -> {ACT_DIM} (ELU)")
    print(f"  python {platform.python_version()}")
    print()

    all_results: list[NetResult] = []
    for name, fn in [
        ("torch-cpu", lambda: bench_torch("cpu", args.batches, args.train_batches)),
        ("torch-mps", lambda: bench_torch("mps", args.batches, args.train_batches)),
        ("mlx", lambda: bench_mlx(args.batches, args.train_batches)),
    ]:
        try:
            all_results += fn()
            print(f"  [ok]   {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")

    backends = sorted({r.backend for r in all_results})

    for mode in ("inference", "inference+transfer", "train"):
        rows = [r for r in all_results if r.mode == mode]
        if not rows:
            continue
        print()
        print(f"--- {mode} (milliseconds per call, lower is better) ---")
        header = f"  {'batch':>7s}" + "".join(f"{b:>16s}" for b in backends) + "   winner"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for b in sorted({r.batch for r in rows}):
            cells = []
            best_backend, best_ms = "", float("inf")
            for backend in backends:
                match = [r for r in rows if r.batch == b and r.backend == backend]
                if match:
                    ms = match[0].per_call_ms
                    cells.append(f"{ms:>16.3f}")
                    if ms < best_ms:
                        best_ms, best_backend = ms, backend
                else:
                    cells.append(f"{'-':>16s}")
            print(f"  {b:>7d}" + "".join(cells) + f"   {best_backend}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "hardware": {
                    "chip": hw.chip,
                    "gpu_cores": hw.gpu_cores,
                    "p_cores": hw.performance_cores,
                },
                "network": {"obs_dim": OBS_DIM, "hidden": list(HIDDEN), "act_dim": ACT_DIM},
                "torch_version": torch.__version__,
                "results": [asdict(r) for r in all_results],
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
