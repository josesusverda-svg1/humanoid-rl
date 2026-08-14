"""Phase 0 benchmark: how fast can this machine simulate a humanoid, and how does it scale.

This benchmark exists to answer one architectural question with numbers instead of
opinion: should parallel environments run as **threads** (shared memory, no
inter-process copying) or as **processes** (true parallelism guaranteed, but every
observation must be serialised across a process boundary)?

The answer hinges on whether MuJoCo's Python bindings release the GIL.

    GIL (Global Interpreter Lock): CPython allows only one thread to execute Python
    bytecode at a time. A C extension can explicitly *release* the lock while it does
    pure C work, which lets other Python threads run concurrently. If mj_step releases
    the GIL, threads give real parallelism at zero copying cost. If it does not,
    threads are useless and we are forced into multiprocessing.

Measured configurations
-----------------------
1. serial            one thread, one environment. The baseline.
2. threads-unsync    N threads stepping independently. The raw parallel ceiling.
3. threads-sync      N threads with a barrier every control step. This is the realistic
                     vectorised-environment pattern, and it includes synchronisation cost.
4. threads-sync-qos  Same, but workers request USER_INITIATED QoS so macOS keeps them on
                     performance cores instead of parking some on efficiency cores.
5. rollout           mujoco.rollout.Rollout, MuJoCo's own native C++ thread pool. Upper
                     bound for physics throughput, though it cannot run a Python policy
                     in the loop, so it is a reference point rather than a candidate.
6. processes         N spawned processes with a shared-memory observation buffer and a
                     barrier per control step. The multiprocessing alternative.

Metrics
-------
env_steps_per_sec      policy-rate steps across all environments. This is the number
                       that matters for RL, because sample budgets are quoted in it.
physics_steps_per_sec  env_steps_per_sec * control_decimation. Reported for context.

Usage:
    .venv/bin/python scripts/bench_physics.py --quick
    .venv/bin/python scripts/bench_physics.py --json bench_results/physics.json
"""

from __future__ import annotations

# Keep numeric libraries single-threaded. Without this, numpy/BLAS spawns its own
# thread pool inside every worker and the machine oversubscribes badly, which would
# make the scaling numbers meaningless.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
import psutil

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_rl import hardware  # noqa: E402
from humanoid_rl.qos import QoSClass, set_thread_qos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "humanoid_rl" / "models" / "humanoid.xml"

# Control decimation: the policy acts at 1/DECIMATION of the physics rate.
# humanoid.xml uses a 0.005 s timestep (200 Hz), so decimation 4 gives a 50 Hz policy,
# which is the standard rate for locomotion control.
DECIMATION = 4


# --------------------------------------------------------------------------------------
# A minimal but realistic environment core
# --------------------------------------------------------------------------------------


class HumanoidCore:
    """One environment: a single MjData plus a representative observation function.

    Deliberately includes the observation computation. Benchmarking a bare mj_step loop
    overstates achievable throughput, because in a real training loop every step also
    builds an observation vector and evaluates a reward.
    """

    def __init__(self, model: mujoco.MjModel, seed: int) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.rng = np.random.default_rng(seed)
        self.prev_action = np.zeros(model.nu, dtype=np.float64)
        # Goal vector in the humanoid's local frame, present from day one so the
        # observation size matches what the goal-conditioned policy will actually use.
        self.goal = self.rng.normal(size=3)
        mujoco.mj_resetData(model, self.data)
        mujoco.mj_forward(model, self.data)

    def observe(self) -> np.ndarray:
        """Build the observation vector.

        Layout mirrors the design we will train with: joint state, root state expressed
        in a heading-invariant way, the previous action, and the local-frame goal.
        """
        d = self.data
        return np.concatenate(
            [
                d.qpos[7:],  # joint angles, excluding the 7-dof free root
                d.qvel[6:],  # joint velocities, excluding the 6-dof root velocity
                d.qpos[3:7],  # root orientation quaternion
                d.qvel[:6],  # root linear and angular velocity
                self.prev_action,  # previous action, for action-rate penalties
                self.goal,  # target direction in the local frame
            ]
        )

    def step(self, action: np.ndarray, decimation: int = DECIMATION) -> np.ndarray:
        """Apply one policy action, advance `decimation` physics steps, return the obs."""
        self.data.ctrl[:] = action
        for _ in range(decimation):
            mujoco.mj_step(self.model, self.data)
        self.prev_action[:] = action

        # Reset on fall, exactly as the real environment will. Keeps the benchmark
        # honest, since a fallen humanoid has different contact costs than a standing one.
        if self.data.qpos[2] < 0.8:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        return self.observe()


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    workers: int
    total_envs: int
    env_steps_per_sec: float
    physics_steps_per_sec: float
    elapsed_s: float
    speedup_vs_serial: float = 0.0
    parallel_efficiency: float = 0.0
    mean_busy_core_pct: float = 0.0
    notes: str = ""
    per_core_pct: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# CPU utilisation sampling
# --------------------------------------------------------------------------------------


class CoreMonitor:
    """Samples per-core CPU utilisation in the background during a benchmark run."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._samples: list[list[float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "CoreMonitor":
        psutil.cpu_percent(percpu=True)  # prime the counters
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._samples.append(psutil.cpu_percent(percpu=True))

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def per_core_mean(self) -> list[float]:
        if not self._samples:
            return []
        return [float(statistics.mean(core)) for core in zip(*self._samples)]


# --------------------------------------------------------------------------------------
# Benchmark 1: serial baseline
# --------------------------------------------------------------------------------------


def bench_serial(model: mujoco.MjModel, control_steps: int) -> BenchResult:
    env = HumanoidCore(model, seed=0)
    actions = np.random.default_rng(1).uniform(-0.4, 0.4, size=(control_steps, model.nu))

    with CoreMonitor() as mon:
        t0 = time.perf_counter()
        for i in range(control_steps):
            env.step(actions[i])
        elapsed = time.perf_counter() - t0

    return BenchResult(
        name="serial",
        workers=1,
        total_envs=1,
        env_steps_per_sec=control_steps / elapsed,
        physics_steps_per_sec=control_steps * DECIMATION / elapsed,
        elapsed_s=elapsed,
        per_core_pct=mon.per_core_mean(),
        notes="single thread, single environment",
    )


# --------------------------------------------------------------------------------------
# Benchmark 2 and 3: thread scaling
# --------------------------------------------------------------------------------------


def bench_threads(
    model: mujoco.MjModel,
    workers: int,
    total_envs: int,
    control_steps: int,
    synchronized: bool,
    qos: QoSClass | None,
) -> BenchResult:
    """N worker threads, each owning a contiguous slice of the environment pool.

    When `synchronized` is True a barrier runs every control step, reproducing the
    vectorised-environment pattern where the policy needs all observations before it
    can produce the next batch of actions. That barrier is the real cost of the design,
    so measuring without it would flatter the result.
    """
    envs = [HumanoidCore(model, seed=i) for i in range(total_envs)]
    rng = np.random.default_rng(2)
    actions = rng.uniform(-0.4, 0.4, size=(control_steps, total_envs, model.nu))
    obs_buf = np.zeros((total_envs, len(envs[0].observe())), dtype=np.float64)

    # Split environments across workers as evenly as possible.
    bounds = np.linspace(0, total_envs, workers + 1).astype(int)
    slices = [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]

    barrier = threading.Barrier(workers) if synchronized else None
    errors: list[BaseException] = []

    def worker(lo: int, hi: int) -> None:
        try:
            if qos is not None:
                set_thread_qos(qos)
            for t in range(control_steps):
                for j in range(lo, hi):
                    obs_buf[j] = envs[j].step(actions[t, j])
                if barrier is not None:
                    barrier.wait()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller below
            errors.append(exc)
            if barrier is not None:
                barrier.abort()

    threads = [threading.Thread(target=worker, args=s, daemon=True) for s in slices]

    with CoreMonitor() as mon:
        t0 = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        elapsed = time.perf_counter() - t0

    if errors:
        raise errors[0]

    total_env_steps = control_steps * total_envs
    label = "threads-sync" if synchronized else "threads-unsync"
    if qos is not None:
        label += "-qos"

    return BenchResult(
        name=label,
        workers=workers,
        total_envs=total_envs,
        env_steps_per_sec=total_env_steps / elapsed,
        physics_steps_per_sec=total_env_steps * DECIMATION / elapsed,
        elapsed_s=elapsed,
        per_core_pct=mon.per_core_mean(),
        notes=f"barrier={'yes' if synchronized else 'no'} qos={qos.name if qos else 'default'}",
    )


# --------------------------------------------------------------------------------------
# Benchmark 4: MuJoCo's native rollout thread pool
# --------------------------------------------------------------------------------------


def bench_rollout(
    model: mujoco.MjModel, workers: int, total_envs: int, control_steps: int
) -> BenchResult:
    """MuJoCo's own C++ thread pool.

    Reference point only. `rollout` executes a pre-computed open-loop control sequence,
    so a Python policy cannot run inside the loop. It shows the physics-only ceiling
    with no Python overhead at all.
    """
    from mujoco import rollout as mj_rollout

    nstep = control_steps * DECIMATION
    models = [model] * total_envs
    datas = [mujoco.MjData(model) for _ in range(min(workers, total_envs))]

    initial_state = np.zeros((total_envs, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)))
    for i in range(total_envs):
        d = mujoco.MjData(model)
        mujoco.mj_resetData(model, d)
        mujoco.mj_getState(model, d, initial_state[i], mujoco.mjtState.mjSTATE_FULLPHYSICS)

    control = np.random.default_rng(3).uniform(-0.4, 0.4, size=(total_envs, nstep, model.nu))

    with CoreMonitor() as mon:
        roll = mj_rollout.Rollout(nthread=workers)
        t0 = time.perf_counter()
        roll.rollout(models, datas, initial_state, control)
        elapsed = time.perf_counter() - t0
        roll.close()

    total_env_steps = control_steps * total_envs
    return BenchResult(
        name="rollout",
        workers=workers,
        total_envs=total_envs,
        env_steps_per_sec=total_env_steps / elapsed,
        physics_steps_per_sec=total_env_steps * DECIMATION / elapsed,
        elapsed_s=elapsed,
        per_core_pct=mon.per_core_mean(),
        notes="native C++ thread pool, open-loop control, no Python policy in the loop",
    )


# --------------------------------------------------------------------------------------
# Benchmark 5: process scaling
# --------------------------------------------------------------------------------------


def _process_worker(
    model_path: str,
    lo: int,
    hi: int,
    control_steps: int,
    obs_dim: int,
    shm_name: str,
    total_envs: int,
    barrier: "mp.Barrier",  # type: ignore[type-arg]
    ready: "mp.Barrier",  # type: ignore[type-arg]
) -> None:
    """Child process body: own the model, step a slice, publish obs to shared memory."""
    from multiprocessing import shared_memory

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    model = mujoco.MjModel.from_xml_path(model_path)
    envs = [HumanoidCore(model, seed=i) for i in range(lo, hi)]
    rng = np.random.default_rng(lo + 100)
    actions = rng.uniform(-0.4, 0.4, size=(control_steps, hi - lo, model.nu))

    shm = shared_memory.SharedMemory(name=shm_name)
    obs = np.ndarray((total_envs, obs_dim), dtype=np.float64, buffer=shm.buf)

    ready.wait()  # do not start the clock until every process has finished importing
    for t in range(control_steps):
        for k, env in enumerate(envs):
            obs[lo + k] = env.step(actions[t, k])
        barrier.wait()
    shm.close()


def bench_processes(
    model_path: Path, model: mujoco.MjModel, workers: int, total_envs: int, control_steps: int
) -> BenchResult:
    from multiprocessing import shared_memory

    obs_dim = len(HumanoidCore(model, seed=0).observe())
    nbytes = total_envs * obs_dim * 8
    shm = shared_memory.SharedMemory(create=True, size=nbytes)

    ctx = mp.get_context("spawn")  # macOS default and the only safe option with MuJoCo
    bounds = np.linspace(0, total_envs, workers + 1).astype(int)
    # The parent participates in `ready` but not in the per-step barrier.
    barrier = ctx.Barrier(workers)
    ready = ctx.Barrier(workers + 1)

    procs = [
        ctx.Process(
            target=_process_worker,
            args=(
                str(model_path),
                int(bounds[i]),
                int(bounds[i + 1]),
                control_steps,
                obs_dim,
                shm.name,
                total_envs,
                barrier,
                ready,
            ),
        )
        for i in range(workers)
    ]
    try:
        for p in procs:
            p.start()
        ready.wait()  # all children have imported mujoco and built their envs
        with CoreMonitor() as mon:
            t0 = time.perf_counter()
            for p in procs:
                p.join()
            elapsed = time.perf_counter() - t0
        per_core = mon.per_core_mean()
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        shm.close()
        shm.unlink()

    total_env_steps = control_steps * total_envs
    return BenchResult(
        name="processes",
        workers=workers,
        total_envs=total_envs,
        env_steps_per_sec=total_env_steps / elapsed,
        physics_steps_per_sec=total_env_steps * DECIMATION / elapsed,
        elapsed_s=elapsed,
        per_core_pct=per_core,
        notes="spawn context, shared-memory obs buffer, barrier per control step",
    )


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _finalise(results: list[BenchResult], serial_sps: float) -> None:
    for r in results:
        r.speedup_vs_serial = r.env_steps_per_sec / serial_sps if serial_sps else 0.0
        r.parallel_efficiency = r.speedup_vs_serial / r.workers if r.workers else 0.0
        if r.per_core_pct:
            busy = [p for p in r.per_core_pct if p > 20.0]
            r.mean_busy_core_pct = float(statistics.mean(busy)) if busy else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--control-steps", type=int, default=600)
    parser.add_argument("--total-envs", type=int, default=256)
    parser.add_argument("--quick", action="store_true", help="fewer steps, fewer worker counts")
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "bench_results" / "physics.json")
    parser.add_argument("--skip-processes", action="store_true")
    args = parser.parse_args()

    hw = hardware.detect()
    print("=" * 78)
    print("PHYSICS BENCHMARK".center(78))
    print("=" * 78)
    for line in hw.summary_lines():
        print("  " + line)
    print(f"  mujoco version         {mujoco.__version__}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    print(
        f"  model                  {args.model.name} "
        f"(nq={model.nq} nv={model.nv} nu={model.nu} timestep={model.opt.timestep})"
    )
    print(f"  control rate           {1.0 / (model.opt.timestep * DECIMATION):.0f} Hz "
          f"(decimation {DECIMATION})")

    control_steps = 200 if args.quick else args.control_steps
    total_envs = 64 if args.quick else args.total_envs

    p = hw.performance_cores or 8
    if args.quick:
        worker_counts = sorted({1, p // 2, p})
    else:
        worker_counts = sorted({1, 2, 4, p // 2, p, p + 2, hw.logical_cores})
    worker_counts = [w for w in worker_counts if 1 <= w <= total_envs]

    results: list[BenchResult] = []

    print("\n--- serial baseline ---")
    serial = bench_serial(model, control_steps)
    results.append(serial)
    print(f"  {serial.env_steps_per_sec:>12,.0f} env steps/s  "
          f"({serial.physics_steps_per_sec:,.0f} physics steps/s)")

    print("\n--- threads, unsynchronised (raw parallel ceiling) ---")
    for w in worker_counts:
        r = bench_threads(model, w, total_envs, control_steps, synchronized=False, qos=None)
        results.append(r)
        print(f"  workers={w:<3d} {r.env_steps_per_sec:>12,.0f} env steps/s  "
              f"speedup x{r.env_steps_per_sec / serial.env_steps_per_sec:5.2f}")

    print("\n--- threads, synchronised (realistic vectorised env) ---")
    for w in worker_counts:
        r = bench_threads(model, w, total_envs, control_steps, synchronized=True, qos=None)
        results.append(r)
        print(f"  workers={w:<3d} {r.env_steps_per_sec:>12,.0f} env steps/s  "
              f"speedup x{r.env_steps_per_sec / serial.env_steps_per_sec:5.2f}")

    print("\n--- threads, synchronised + USER_INITIATED QoS (prefer performance cores) ---")
    for w in worker_counts:
        r = bench_threads(
            model, w, total_envs, control_steps, synchronized=True, qos=QoSClass.USER_INITIATED
        )
        results.append(r)
        print(f"  workers={w:<3d} {r.env_steps_per_sec:>12,.0f} env steps/s  "
              f"speedup x{r.env_steps_per_sec / serial.env_steps_per_sec:5.2f}")

    print("\n--- mujoco.rollout native thread pool (physics-only ceiling) ---")
    for w in worker_counts:
        try:
            r = bench_rollout(model, w, total_envs, control_steps)
            results.append(r)
            print(f"  workers={w:<3d} {r.env_steps_per_sec:>12,.0f} env steps/s  "
                  f"speedup x{r.env_steps_per_sec / serial.env_steps_per_sec:5.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  workers={w:<3d} FAILED: {type(exc).__name__}: {exc}")

    if not args.skip_processes:
        print("\n--- processes (spawn + shared memory) ---")
        for w in worker_counts:
            try:
                r = bench_processes(args.model, model, w, total_envs, control_steps)
                results.append(r)
                print(f"  workers={w:<3d} {r.env_steps_per_sec:>12,.0f} env steps/s  "
                      f"speedup x{r.env_steps_per_sec / serial.env_steps_per_sec:5.2f}")
            except Exception as exc:  # noqa: BLE001
                print(f"  workers={w:<3d} FAILED: {type(exc).__name__}: {exc}")

    _finalise(results, serial.env_steps_per_sec)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hardware": {
            "chip": hw.chip,
            "performance_cores": hw.performance_cores,
            "efficiency_cores": hw.efficiency_cores,
            "gpu_cores": hw.gpu_cores,
            "memory_gb": round(hw.memory_gb, 1),
            "os": f"{hw.os_name} {hw.os_version}",
        },
        "config": {
            "model": args.model.name,
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "timestep": float(model.opt.timestep),
            "decimation": DECIMATION,
            "control_steps": control_steps,
            "total_envs": total_envs,
            "mujoco_version": mujoco.__version__,
        },
        "results": [asdict(r) for r in results],
    }
    args.json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.json}")

    # The single most important derived conclusion.
    best_thread = max(
        (r for r in results if r.name.startswith("threads-sync")),
        key=lambda r: r.env_steps_per_sec,
        default=None,
    )
    best_proc = max(
        (r for r in results if r.name == "processes"),
        key=lambda r: r.env_steps_per_sec,
        default=None,
    )
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if best_thread:
        print(f"  best threaded : {best_thread.env_steps_per_sec:>12,.0f} env steps/s "
              f"({best_thread.workers} workers, x{best_thread.speedup_vs_serial:.2f} vs serial, "
              f"{best_thread.parallel_efficiency * 100:.0f}% parallel efficiency)")
        if best_thread.speedup_vs_serial > 2.0:
            print("  -> MuJoCo releases the GIL. Threads are viable, and avoid all IPC cost.")
        else:
            print("  -> Threads did NOT scale. The GIL is held during mj_step; use processes.")
    if best_proc:
        print(f"  best processes: {best_proc.env_steps_per_sec:>12,.0f} env steps/s "
              f"({best_proc.workers} workers, x{best_proc.speedup_vs_serial:.2f} vs serial)")


if __name__ == "__main__":
    main()
