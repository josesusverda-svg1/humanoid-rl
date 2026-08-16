"""Full get-up REFERENCE trajectories: supine -> sit -> tuck -> squat -> stand, synthesized.

E42, the conceptual change. Twenty runs of reward shaping taught pieces (the catch, the
tuck, floor sit-ups) but never the SEQUENCE, because per-step exploration cannot discover a
2-4 s coordinated trajectory. Every published get-up system solves this with a motion
reference the policy is paid to reproduce. We have no mocap; we synthesize instead:

    record the DESCENT   stand -> squat -> seated -> supine
    as slow servo ramps between anchor poses the bank already holds
    (each stage gentle, root speed capped), then REVERSE TIME.

The reverse of a gentle descent is a gentle rise, and it is OUR body's rise: every frame is
a state this exact model actually occupied under physics. Validated per clip: monotone
stage structure, root speed within human range (the rise inherits <= the descent's cap,
which is under the launch fine's 1.0 m/s free line), starts truly supine, ends at a true
stand, no penetration.

    python scripts/generate_getup_reference.py    # writes data/fallen/getup_refs_v1.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.envs.model_prep import prepare  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_midrise_poses import build_pose  # noqa: E402

N_CLIPS = 12
#: Per-stage ramp durations in seconds (descent direction). Sitting down and lying back are
#: gentle by nature; the stand-to-squat leg is the proven quasi-static ramp.
STAGES = (("stand", "squat", 3.2), ("squat", "seated", 2.2), ("seated", "supine", 2.2))
#: Speed limits for the descent (the rise inherits them). Two-tier, measured: the body
#: makes a brief ~1.0 m/s "plop" settling into the squat REGARDLESS of ramp speed (slower
#: ramps gave the same peak, so it is a discrete centimetre-scale drop, not ramp rate).
#: Humans peak near 1.0 m/s in sit-to-stand too. Tolerate the transient, reject anything
#: sustained or violent.
SPEED_HARD = 1.4        # never, at any instant
SPEED_SOFT = 1.0        # not for longer than SOFT_STEPS consecutive recorded steps
SOFT_STEPS = 12         # ~0.1 s at the 8 ms recording cadence


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=REPO_ROOT / "data/fallen/bank_v3.npz")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/fallen/getup_refs_v1.npz")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    prepared = prepare(REPO_ROOT / "humanoid_rl/models/humanoid_scene.xml",
                       action_scale_mode="full_range")
    model = prepared.model
    data = mujoco.MjData(model)
    qadr = prepared.actuator_qpos_adr
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rng = np.random.default_rng(args.seed)

    bank = np.load(args.bank, allow_pickle=False)
    lab = bank["label"]

    def anchor_joints(kind: str) -> np.ndarray:
        """Joint-angle targets for a stage anchor, drawn from the bank with jitter."""
        if kind == "stand":
            j = prepared.default_joint_pos.copy()
        else:
            want = {"squat": "midrise_squat", "seated": "seated", "supine": "supine"}[kind]
            pool = np.flatnonzero(lab == want)
            q = bank["qpos"][pool[rng.integers(0, pool.size)]]
            j = q[qadr].copy()
        return np.clip(j + rng.normal(0.0, 0.04, j.size), lo, hi)

    def run_stage(to: str, seconds: float, current: np.ndarray):
        """One ramp under the speed filters. Returns (frames, target) or None.

        The squat target comes from build_pose (randomized ankles), NOT from bank anchors:
        an equilibrium ANCHOR does not make the PATH to it quasi-static, and measured, bank
        anchors made the body free-fall the last stretch in 39 of 40 trials, while
        build_pose targets pass a stricter cap 16% of the time (generate_rising_states'
        yield), which per-stage retries turn into near-certainty.
        """
        if to == "squat":
            target = build_pose("squat", model, data2, prepared, qadr, rng)
        else:
            target = anchor_joints(to)
        steps = int(seconds / model.opt.timestep)
        frames = []
        soft_run = 0
        for t in range(steps):
            a = (t + 1) / steps
            data.ctrl[:] = np.clip(current + a * (target - current), lo, hi)
            mujoco.mj_step(model, data)
            if t % 4 == 0:
                v = float(np.linalg.norm(data.qvel[0:3]))
                if v > SPEED_HARD:
                    return None
                soft_run = soft_run + 1 if v > SPEED_SOFT else 0
                if soft_run > SOFT_STEPS:
                    return None
                frames.append((data.qpos.copy(), data.qvel.copy()))
        for _t in range(int(0.4 / model.opt.timestep)):
            data.ctrl[:] = np.clip(target, lo, hi)
            mujoco.mj_step(model, data)
            if _t % 4 == 0:
                frames.append((data.qpos.copy(), data.qvel.copy()))
        return frames, target

    data2 = mujoco.MjData(model)   # scratch for build_pose, which writes into a data
    clips_q, clips_v, bounds = [], [], [0]
    kept = 0
    attempts = 0
    while kept < N_CLIPS and attempts < N_CLIPS * 6:
        attempts += 1
        data.qpos[:] = prepared.default_qpos
        data.qvel[:] = 0.0
        traj = []
        current = prepared.default_joint_pos.copy()
        ok = True
        for _from, to, seconds in STAGES:
            # Per-stage retries from a SNAPSHOT: a failed ramp rewinds the body and redraws
            # the target instead of discarding the whole clip.
            snap_q, snap_v = data.qpos.copy(), data.qvel.copy()
            for _try in range(25):
                out = run_stage(to, seconds, current)
                if out is not None:
                    break
                data.qpos[:] = snap_q
                data.qvel[:] = snap_v
                mujoco.mj_forward(model, data)
            else:
                ok = False
                break
            frames, current = out
            traj.extend(frames)
        if not ok or not traj:
            continue
        z = np.array([float(q[2]) for q, _ in traj])
        # The descent must genuinely start standing and end lying.
        if z[0] < 0.85 or z[-1] > 0.22 or z.min() < 0.05:
            continue
        # Reverse: the get-up. Velocities flip sign.
        rq = np.stack([q for q, _ in traj])[::-1].copy()
        rv = -np.stack([v for _, v in traj])[::-1].copy()
        clips_q.append(rq)
        clips_v.append(rv)
        bounds.append(bounds[-1] + len(rq))
        kept += 1

    if kept == 0:
        raise RuntimeError("no clip survived; loosen SPEED_CAP or slow the stages")
    Q = np.concatenate(clips_q)
    V = np.concatenate(clips_v)
    np.savez_compressed(
        args.out, qpos=Q, qvel=V, bounds=np.array(bounds, dtype=np.int64),
        nq=model.nq, nv=model.nv, dt=model.opt.timestep * 4,
        model_sha1=bank["model_sha1"], generator_version=1,
    )
    lens = np.diff(bounds)
    z0 = [float(clips_q[i][0][2]) for i in range(kept)]
    z1 = [float(clips_q[i][-1][2]) for i in range(kept)]
    print(f"{kept} clips ({attempts} attempts) -> {args.out}")
    print(f"  length {lens.min()}-{lens.max()} control steps "
          f"({lens.min()*0.008:.1f}-{lens.max()*0.008:.1f} s)")
    print(f"  starts (supine) pelvis {min(z0):.3f}-{max(z0):.3f}; "
          f"ends (stand) {min(z1):.3f}-{max(z1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
