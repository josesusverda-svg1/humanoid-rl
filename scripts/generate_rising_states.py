"""Mid-rise states WITH upward momentum, for bank_v3: the discovery bridge.

E39's checkpoint measured the wall exactly: even at the easiest exam (knee 1.30, hold 0.5 s)
rung starts stand 0.01% of steps, because the rise-and-catch transition never gets
discovered. Per-step Gaussian exploration at 125 Hz through an 8 Hz action filter is dither;
the coordinated 2 s push it would need to stumble on has effectively zero probability.

The bridge (DeepMimic's reference state initialization, applied without mocap): record the
generator's physically honest DESCENTS (stand -> squat/crouch, ramped servo targets, mass
balanced the whole way), reverse them in time, and sample states ALONG the reversed rise
with their reversed velocities. An episode that starts 0.3-1.5 s from standing, already
moving upward, puts the standing salary inside dither range, and the policy learns the catch
directly. Quasi-static trajectories stay dynamically feasible under time reversal, and every
sampled state additionally proves itself by COMPLETING: held at the standing target by the
plain servos, it must actually arrive at a stand.

Builds bank_v3.npz = bank_v2 (untouched) + the rising pool, generator == "rising".

    python scripts/generate_rising_states.py           # writes data/fallen/bank_v3.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from humanoid_rl.envs.model_prep import prepare  # noqa: E402

from generate_midrise_poses import build_pose  # noqa: E402

#: Descent families to reverse. squat and crouch are the two rungs whose rise is the
#: measured missing link; both are reached from a stand by build_pose targets.
KINDS = ("squat", "crouch")
PER_KIND = 80
#: Sample the rise at these phases (fraction of the way UP from the deepest point).
#: Weighted toward the top: the catch near standing is the skill that never got learned.
PHASES = (0.35, 0.5, 0.65, 0.8, 0.9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=REPO_ROOT / "data/fallen/bank_v2.npz")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/fallen/bank_v3.npz")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    prepared = prepare(REPO_ROOT / "humanoid_rl/models/humanoid_scene.xml",
                       action_scale_mode="full_range")
    model = prepared.model
    data = mujoco.MjData(model)
    qadr = prepared.actuator_qpos_adr
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rng = np.random.default_rng(args.seed)

    kept_q, kept_v, kept_kind, kept_phase = [], [], [], []
    stats: dict[str, list[float]] = {k: [] for k in KINDS}
    tries = {k: 0 for k in KINDS}

    def record_descent(target: np.ndarray, seconds: float):
        """Ramp from the stand to `target`, recording every control step.

        Returns None unless the descent stays QUASI-STATIC the whole way: root speed under
        0.6 m/s at every recorded step. The first version skipped this and 135 of 160 kept
        states came from descents that simply COLLAPSED (root speed to 3 m/s, pelvis to
        0.04); their time reversal is a ballistic launch, not a rise, and the completes()
        gate cannot tell the difference because momentum alone finishes a launch.
        """
        data.qpos[:] = prepared.default_qpos
        data.qvel[:] = 0.0
        start = prepared.default_joint_pos.copy()
        steps = int(seconds / model.opt.timestep)
        traj = []
        for t in range(steps):
            a = (t + 1) / steps
            data.ctrl[:] = np.clip(start + a * (target - start), lo, hi)
            mujoco.mj_step(model, data)
            if t % 4 == 0:                      # control-step resolution (decimation 4)
                if float(np.linalg.norm(data.qvel[0:3])) > 0.6:
                    return None
                traj.append((data.qpos.copy(), data.qvel.copy()))
        return traj

    def completes(qpos: np.ndarray, qvel: np.ndarray) -> bool:
        """From this state, do plain servos holding the STAND target finish the rise?

        The state proves it is genuinely on a completable rise. 1.2 s is generous: the
        deepest sampled phase is ~0.9 s of quasi-static travel from standing.
        """
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        if data.ncon and float(-data.contact.dist.min()) > 2e-3:
            return False
        for _ in range(int(1.2 / model.opt.timestep)):
            data.ctrl[:] = np.clip(prepared.default_joint_pos, lo, hi)
            mujoco.mj_step(model, data)
        return float(data.qpos[2]) >= 0.70

    for kind in KINDS:
        while len(stats[kind]) < PER_KIND:
            tries[kind] += 1
            if tries[kind] > 2000:
                break
            target = build_pose(kind, model, data, prepared, qadr, rng)
            seconds = float(rng.uniform(2.0, 3.2))    # slow: quasi-static or rejected
            traj = record_descent(target, seconds)
            if traj is None or len(traj) < 20:
                continue
            # Depth sanity: reached genuine depth WITHOUT collapsing (the speed filter above
            # already rejected falls, so a deep endpoint now means a controlled descent).
            z_bottom, z_top = float(traj[-1][0][2]), float(traj[0][0][2])
            if z_bottom > 0.60 or z_top < 0.80:
                continue
            # Phase by HEIGHT, not by time: a servo ramp loses almost no height early, so a
            # time index at "phase 0.9" sat at 98.5% of standing height and 73 of 160 kept
            # states passed the full standing predicate AT RESET (placed stands, the exact
            # promotion-gaming exploit fixed once already). Interpolate the height profile
            # instead, and cap kept height at 0.73, BELOW the standing predicate height threshold (0.745), so no rising state can pass the exam at spawn.
            zs = np.array([float(q[2]) for q, _v in traj])
            for ph in PHASES:
                if len(stats[kind]) >= PER_KIND:
                    break
                z_target = z_bottom + ph * (z_top - z_bottom)
                if z_target > 0.72:
                    z_target = 0.72
                idx = int(np.argmin(np.abs(zs - z_target)))
                q, v = traj[idx]
                rq, rv = q.copy(), -v.copy()     # time reversal flips every velocity
                if float(rq[2]) > 0.73:
                    continue
                if not completes(rq, rv):
                    continue
                kept_q.append(rq)
                kept_v.append(rv)
                kept_kind.append(kind)
                kept_phase.append(ph)
                stats[kind].append(float(rq[2]))

    base = np.load(args.base, allow_pickle=False)
    assert int(base["nq"]) == model.nq
    n_new = len(kept_q)
    out = {k: base[k] for k in base.files}
    out["qpos"] = np.concatenate([base["qpos"], np.stack(kept_q)])
    out["qvel"] = np.concatenate([base["qvel"], np.stack(kept_v)])
    out["label"] = np.concatenate([base["label"],
                                   np.array([f"rising_{k}" for k in kept_kind])])
    out["generator"] = np.concatenate([base["generator"], np.array(["rising"] * n_new)])
    out["split"] = np.concatenate([base["split"], np.array(["train"] * n_new)])
    out["generator_version"] = 3
    np.savez_compressed(args.out, **out)

    print(f"bank_v3: {len(base['qpos'])} base + {n_new} rising -> {args.out}")
    for k in KINDS:
        p = np.array(stats[k])
        if p.size:
            print(f"  rising_{k:<7} kept {len(p):>3}  tries {tries[k]:>4}  "
                  f"pelvis {p.min():.3f}-{p.max():.3f}")
        else:
            print(f"  rising_{k:<7} kept   0  tries {tries[k]:>4}  NONE COMPLETED")
    ph = np.array(kept_phase)
    if ph.size:
        print("  phase mix:", {p: int((ph == p).sum()) for p in PHASES})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
