"""Phase 0, architecture shoot-out: four ways to run N parallel humanoid environments.

The earlier profile showed that Python only holds the GIL for about 2 percent of a
step, yet thread scaling stalled at roughly 3x. The cause is not how *long* the GIL is
held but how *often* it is handed between threads. Every `mj_step` call releases and
reacquires it, and on macOS each handoff costs microseconds. At 500k handoffs per
second that overhead swamps the physics.

So the variants below differ mainly in GIL round-trips per environment step:

    A  threads-4call    4 mj_step calls per env step        4 handoffs / env-step
    B  threads-nstep    mj_step(..., nstep=4), one call      1 handoff / env-step
    C  rollout-batched  one rollout() call for ALL envs      1 handoff / BATCH-step
    D  processes        separate interpreters, no GIL at all 0 (pays IPC instead)

Variant C additionally computes observations as a single vectorised numpy operation
over the returned (n_envs, state_size) array, rather than looping in Python, which
removes the remaining per-environment Python cost.

Usage:
    .venv/bin/python scripts/bench_arch.py
    .venv/bin/python scripts/bench_arch.py --envs 512 1024 4096 --control-steps 300
"""

from __future__ import annotations

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
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanoid_rl import hardware  # noqa: E402
from humanoid_rl.qos import QoSClass, set_thread_qos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "humanoid_rl" / "models" / "humanoid.xml"
DECIMATION = 4

# Layout of an mjSTATE_FULLPHYSICS vector: [time(1), qpos(nq), qvel(nv), act(na)].
# Knowing this lets variant C build observations for the whole batch with slicing.
STATE_TIME = 1


@dataclass
class ArchResult:
    variant: str
    workers: int
    n_envs: int
    env_steps_per_sec: float
    elapsed_s: float
    gil_handoffs_per_env_step: str
    notes: str = ""


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


# --------------------------------------------------------------------------------------
# Variants A and B: worker threads owning slices of the environment pool
# --------------------------------------------------------------------------------------


def bench_threads(
    model: mujoco.MjModel,
    workers: int,
    n_envs: int,
    control_steps: int,
    batched_nstep: bool,
) -> ArchResult:
    datas = [mujoco.MjData(model) for _ in range(n_envs)]
    for d in datas:
        mujoco.mj_resetData(model, d)
        mujoco.mj_forward(model, d)

    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.4, 0.4, size=(n_envs, model.nu))
    obs_dim = (model.nq - 7) + (model.nv - 6) + 4 + 6
    obs = np.zeros((n_envs, obs_dim), dtype=np.float32)

    bounds = np.linspace(0, n_envs, workers + 1).astype(int)
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def worker(lo: int, hi: int) -> None:
        try:
            set_thread_qos(QoSClass.USER_INITIATED)
            for _ in range(control_steps):
                for j in range(lo, hi):
                    d = datas[j]
                    d.ctrl[:] = actions[j]
                    if batched_nstep:
                        mujoco.mj_step(model, d, nstep=DECIMATION)
                    else:
                        for _ in range(DECIMATION):
                            mujoco.mj_step(model, d)
                    o = 0
                    n1 = model.nq - 7
                    n2 = model.nv - 6
                    obs[j, o : o + n1] = d.qpos[7:]
                    o += n1
                    obs[j, o : o + n2] = d.qvel[6:]
                    o += n2
                    obs[j, o : o + 4] = d.qpos[3:7]
                    o += 4
                    obs[j, o : o + 6] = d.qvel[:6]
                barrier.wait()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            barrier.abort()

    threads = [
        threading.Thread(target=worker, args=(int(bounds[i]), int(bounds[i + 1])), daemon=True)
        for i in range(workers)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    if errors:
        raise errors[0]

    return ArchResult(
        variant="B threads-nstep" if batched_nstep else "A threads-4call",
        workers=workers,
        n_envs=n_envs,
        env_steps_per_sec=control_steps * n_envs / elapsed,
        elapsed_s=elapsed,
        gil_handoffs_per_env_step="1" if batched_nstep else str(DECIMATION),
        notes="per-env Python obs loop",
    )


# --------------------------------------------------------------------------------------
# Variant C: one native batched rollout call per control step
# --------------------------------------------------------------------------------------


def bench_rollout_batched(
    model: mujoco.MjModel, workers: int, n_envs: int, control_steps: int
) -> ArchResult:
    """The whole batch advances inside a single C++ call, then obs is vectorised numpy.

    This is the design that removes both problems at once: one GIL handoff per batch
    step instead of thousands, and zero per-environment Python work.
    """
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nq, nv = model.nq, model.nv

    # One MjData per worker thread. The pool reuses these across the whole batch.
    datas = [mujoco.MjData(model) for _ in range(workers)]

    state = np.zeros((n_envs, nstate), dtype=np.float64)
    probe = mujoco.MjData(model)
    mujoco.mj_resetData(model, probe)
    mujoco.mj_forward(model, probe)
    single = np.zeros(nstate)
    mujoco.mj_getState(model, probe, single, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    state[:] = single

    rng = np.random.default_rng(0)
    control = rng.uniform(-0.4, 0.4, size=(n_envs, DECIMATION, model.nu))
    out_state = np.zeros((n_envs, DECIMATION, nstate), dtype=np.float64)

    # Column offsets into the flat state vector.
    qpos0 = STATE_TIME
    qvel0 = qpos0 + nq

    roll = mj_rollout.Rollout(nthread=workers)
    try:
        # Warm up the thread pool so pool creation is not charged to the timed loop.
        roll.rollout(model, datas, state, control, nstep=DECIMATION, state=out_state)

        t0 = time.perf_counter()
        for _ in range(control_steps):
            roll.rollout(model, datas, state, control, nstep=DECIMATION, state=out_state)
            state = out_state[:, -1, :].copy()
            # Vectorised observation for the entire batch: pure numpy slicing, no loop.
            obs = np.concatenate(
                [
                    state[:, qpos0 + 7 : qpos0 + nq],  # joint angles
                    state[:, qvel0 + 6 : qvel0 + nv],  # joint velocities
                    state[:, qpos0 + 3 : qpos0 + 7],  # root quaternion
                    state[:, qvel0 : qvel0 + 6],  # root velocity
                ],
                axis=1,
                dtype=np.float32,
            )
        elapsed = time.perf_counter() - t0
    finally:
        roll.close()

    assert obs.shape[0] == n_envs
    return ArchResult(
        variant="C rollout-batched",
        workers=workers,
        n_envs=n_envs,
        env_steps_per_sec=control_steps * n_envs / elapsed,
        elapsed_s=elapsed,
        gil_handoffs_per_env_step=f"1/{n_envs}",
        notes="native pool + vectorised numpy obs",
    )


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--control-steps", type=int, default=200)
    ap.add_argument("--json", type=Path, default=REPO_ROOT / "bench_results" / "arch.json")
    args = ap.parse_args()

    hw = hardware.detect()
    workers = hw.recommended_physics_workers
    model = make_model()

    print("=" * 84)
    print("ARCHITECTURE SHOOT-OUT".center(84))
    print("=" * 84)
    print(f"  {hw.chip}, {hw.performance_cores}P + {hw.efficiency_cores}E cores, "
          f"mujoco {mujoco.__version__}")
    print(f"  humanoid.xml nq={model.nq} nv={model.nv} nu={model.nu}, "
          f"decimation={DECIMATION} (50 Hz policy)")
    print(f"  workers={workers}, control_steps={args.control_steps}")
    print()
    print(f"  {'variant':<20s} {'envs':>6s} {'env steps/s':>14s} {'GIL/env-step':>13s}  notes")
    print("  " + "-" * 80)

    results: list[ArchResult] = []
    for n_envs in args.envs:
        for fn in (
            lambda: bench_threads(model, workers, n_envs, args.control_steps, batched_nstep=False),
            lambda: bench_threads(model, workers, n_envs, args.control_steps, batched_nstep=True),
            lambda: bench_rollout_batched(model, workers, n_envs, args.control_steps),
        ):
            try:
                r = fn()
                results.append(r)
                print(f"  {r.variant:<20s} {r.n_envs:>6d} {r.env_steps_per_sec:>14,.0f} "
                      f"{r.gil_handoffs_per_env_step:>13s}  {r.notes}")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED ({type(exc).__name__}): {exc}")
        print()

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "hardware": {"chip": hw.chip, "p_cores": hw.performance_cores},
                "workers": workers,
                "control_steps": args.control_steps,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )

    best = max(results, key=lambda r: r.env_steps_per_sec)
    print("=" * 84)
    print(f"WINNER: {best.variant} at {best.n_envs} envs -> {best.env_steps_per_sec:,.0f} env steps/s")
    print("=" * 84)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
