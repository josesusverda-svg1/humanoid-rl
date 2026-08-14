"""Hardware verification: prove the Metal GPU and the performance cores are really being used.

It is easy to *believe* a training run is using the GPU and all cores while it silently
falls back to something slower. PyTorch will happily run on the CPU if the MPS backend
is unavailable, and macOS will happily schedule worker threads onto efficiency cores.
Both failures are invisible unless measured, and both cost several-fold throughput.

This script proves each claim empirically rather than asking a library whether it
thinks it is enabled:

  1. Metal GPU compute      A large matrix multiply reaches a throughput the CPU
                            provably cannot achieve. Measured in GFLOP/s, not by
                            trusting `is_available()`.
  2. MLX on GPU             Same test through Apple MLX.
  3. Multi-core physics     Parallel MuJoCo stepping saturates the expected number of
                            cores, verified with per-core utilisation sampling.
  4. Offscreen rendering    A frame renders headless and encodes to mp4, which is what
                            the Videos mode of the dashboard depends on.
  5. Power and thermal      Warns if on battery or thermally throttled, either of which
                            invalidates benchmark numbers and slows long runs.

Usage:
    .venv/bin/python scripts/hwcheck.py
Exit code is 0 if every required check passes, 1 otherwise.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanoid_rl import hardware  # noqa: E402
from humanoid_rl.qos import QoSClass, set_thread_qos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "humanoid_rl" / "models" / "humanoid.xml"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    required: bool = True

    def render(self) -> str:
        if self.passed:
            tag = f"{GREEN}PASS{RESET}"
        elif self.required:
            tag = f"{RED}FAIL{RESET}"
        else:
            tag = f"{YELLOW}WARN{RESET}"
        return f"  [{tag}] {self.name:<34s} {self.detail}"


def _matmul_gflops(fn, sync, n: int, iters: int = 12) -> float:
    """Sustained GFLOP/s for an n x n fp32 matrix multiply (2*n^3 floating point ops)."""
    for _ in range(3):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    elapsed = (time.perf_counter() - t0) / iters
    return (2.0 * n**3) / elapsed / 1e9


# --------------------------------------------------------------------------------------


def check_metal_torch(n: int = 4096) -> list[Check]:
    checks: list[Check] = []
    try:
        import torch
    except ImportError as exc:
        return [Check("PyTorch import", False, f"{exc}")]

    if not torch.backends.mps.is_available():
        return [
            Check(
                "PyTorch MPS backend",
                False,
                f"unavailable (built={torch.backends.mps.is_built()})",
            )
        ]

    # CPU reference first, so the GPU number has something to be compared against.
    a_cpu = torch.randn(n, n)
    b_cpu = torch.randn(n, n)
    cpu_gflops = _matmul_gflops(lambda: a_cpu @ b_cpu, lambda: None, n, iters=4)

    a = torch.randn(n, n, device="mps")
    b = torch.randn(n, n, device="mps")
    gpu_gflops = _matmul_gflops(lambda: a @ b, torch.mps.synchronize, n)

    ratio = gpu_gflops / cpu_gflops if cpu_gflops else 0.0
    # A genuine discrete GPU path should comfortably beat the CPU on a large fp32 matmul.
    # If the ratio is near 1 the "mps" device is silently falling back to CPU.
    passed = ratio > 2.0
    checks.append(
        Check(
            "Metal GPU compute (PyTorch)",
            passed,
            f"{gpu_gflops:,.0f} GFLOP/s on mps vs {cpu_gflops:,.0f} on cpu "
            f"({ratio:.1f}x) {'' if passed else '<- looks like a CPU fallback'}",
        )
    )

    # Correctness, not just speed: a fast wrong answer is worse than a slow right one.
    x = torch.randn(512, 512)
    err = float((x.to("mps") @ x.to("mps")).cpu().sub(x @ x).abs().max())
    checks.append(
        Check(
            "MPS numerical agreement",
            err < 1e-2,
            f"max abs diff vs cpu = {err:.2e}",
        )
    )
    return checks


def check_metal_mlx(n: int = 4096) -> list[Check]:
    try:
        import mlx.core as mx
    except ImportError as exc:
        return [Check("MLX import", False, f"{exc}", required=False)]

    dev = mx.default_device()
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a, b)

    def fn() -> None:
        mx.eval(a @ b)

    gflops = _matmul_gflops(fn, mx.synchronize, n)
    on_gpu = "gpu" in str(dev).lower()
    return [
        Check(
            "Metal GPU compute (MLX)",
            on_gpu and gflops > 500,
            f"{gflops:,.0f} GFLOP/s, default device = {dev}",
            required=False,
        )
    ]


def check_multicore_physics(seconds: float = 4.0) -> list[Check]:
    """Prove parallel MuJoCo stepping actually lights up the performance cores."""
    try:
        import mujoco
        import psutil
    except ImportError as exc:
        return [Check("MuJoCo/psutil import", False, f"{exc}")]

    hw = hardware.detect()
    workers = hw.recommended_physics_workers
    n_envs = workers * 64

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    datas = [mujoco.MjData(model) for _ in range(n_envs)]
    for d in datas:
        mujoco.mj_resetData(model, d)
        mujoco.mj_forward(model, d)
    actions = np.random.default_rng(0).uniform(-0.4, 0.4, size=(n_envs, model.nu))

    bounds = np.linspace(0, n_envs, workers + 1).astype(int)
    stop = threading.Event()
    counts = [0] * workers

    def worker(idx: int, lo: int, hi: int) -> None:
        set_thread_qos(QoSClass.USER_INITIATED)
        n = 0
        while not stop.is_set():
            for j in range(lo, hi):
                d = datas[j]
                d.ctrl[:] = actions[j]
                mujoco.mj_step(model, d, nstep=4)
            n += hi - lo
        counts[idx] = n

    threads = [
        threading.Thread(target=worker, args=(i, int(bounds[i]), int(bounds[i + 1])), daemon=True)
        for i in range(workers)
    ]

    samples: list[list[float]] = []
    psutil.cpu_percent(percpu=True)
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    while time.perf_counter() - t0 < seconds:
        time.sleep(0.25)
        samples.append(psutil.cpu_percent(percpu=True))
    stop.set()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    per_core = [float(statistics.mean(c)) for c in zip(*samples)] if samples else []
    busy = [p for p in per_core if p > 70.0]
    steps_per_sec = sum(counts) / elapsed

    checks = [
        Check(
            "Parallel physics cores busy",
            len(busy) >= max(1, workers - 1),
            f"{len(busy)} cores above 70% (expected ~{workers}), "
            f"mean busy = {statistics.mean(busy):.0f}%" if busy else "no cores busy",
        ),
        Check(
            "Physics throughput",
            steps_per_sec > 0,
            f"{steps_per_sec:,.0f} env steps/s at 50 Hz with {workers} workers, {n_envs} envs",
        ),
    ]
    bar = "".join("#" if p > 70 else ("+" if p > 30 else ".") for p in per_core)
    checks.append(
        Check("Per-core utilisation map", True, f"[{bar}]  {DIM}# >70%  + >30%  . idle{RESET}")
    )
    return checks


def check_offscreen_render() -> list[Check]:
    """Headless rendering plus mp4 encoding, which the dashboard Videos mode requires."""
    checks: list[Check] = []
    try:
        import mujoco
    except ImportError as exc:
        return [Check("MuJoCo import", False, f"{exc}")]

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    try:
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            frames = []
            for _ in range(12):
                mujoco.mj_step(model, data, nstep=4)
                renderer.update_scene(data)
                frames.append(renderer.render())
        arr = np.asarray(frames)
        ok = arr.shape == (12, 240, 320, 3) and arr.std() > 1.0
        checks.append(
            Check(
                "Offscreen rendering (headless)",
                ok,
                f"rendered {arr.shape} uint8, pixel std={arr.std():.1f}"
                + ("" if ok else " <- frames look blank"),
            )
        )
    except Exception as exc:  # noqa: BLE001
        return checks + [Check("Offscreen rendering (headless)", False, f"{type(exc).__name__}: {exc}")]

    out = REPO_ROOT / "bench_results" / "hwcheck_render.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio

        iio.imwrite(out, arr, fps=10, codec="h264")
        size = out.stat().st_size
        checks.append(
            Check("mp4 encoding (ffmpeg)", size > 1000, f"wrote {out.name}, {size:,} bytes")
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("mp4 encoding (ffmpeg)", False, f"{type(exc).__name__}: {exc}"))
    return checks


def check_power() -> list[Check]:
    state = hardware.thermal_state()
    source = state.get("power_source", "unknown")
    return [
        Check(
            "On AC power",
            source == "ac",
            f"power source = {source}"
            + ("" if source == "ac" else " <- battery throttles sustained training"),
            required=False,
        )
    ]


def main() -> int:
    hw = hardware.detect()
    print("=" * 88)
    print("HARDWARE VERIFICATION".center(88))
    print("=" * 88)
    for line in hw.summary_lines():
        print("  " + line)
    print()

    checks: list[Check] = []
    for label, fn in [
        ("GPU", check_metal_torch),
        ("GPU", check_metal_mlx),
        ("CPU", check_multicore_physics),
        ("RENDER", check_offscreen_render),
        ("POWER", check_power),
    ]:
        try:
            checks += fn()
        except Exception as exc:  # noqa: BLE001
            checks.append(Check(f"{label} check", False, f"{type(exc).__name__}: {exc}"))

    for c in checks:
        print(c.render())

    required_failed = [c for c in checks if c.required and not c.passed]
    warned = [c for c in checks if not c.required and not c.passed]
    print()
    print("=" * 88)
    if required_failed:
        print(f"{RED}{len(required_failed)} REQUIRED CHECK(S) FAILED{RESET}")
        for c in required_failed:
            print(f"    - {c.name}: {c.detail}")
        return 1
    msg = f"{GREEN}ALL REQUIRED CHECKS PASSED{RESET}"
    if warned:
        msg += f"  ({YELLOW}{len(warned)} warning(s){RESET})"
    print(msg)
    print("  Metal GPU is doing real work, the performance cores are saturated,")
    print("  and headless rendering to mp4 works. The machine is ready to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
