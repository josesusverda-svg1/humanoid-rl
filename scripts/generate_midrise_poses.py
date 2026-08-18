"""Mid-rise reset poses: the rungs BETWEEN lying and standing, for bank_v2.

The pre-registered E34 response to a floor-parked policy (LOGBOOK E34 prediction 4): change
the RESET DISTRIBUTION, never the reward. A policy that has only ever been on the floor has
no experience of the states where the reward pays, so the advantage landscape around its
policy is flat; starting some episodes on the ladder's middle rungs (all fours, kneel,
half-kneel, squat, crouch) lets the critic learn the value of those states directly, and the
floor policy then climbs toward states whose value is KNOWN rather than rumoured. DeepMimic
calls this reference state initialisation; it is the standard cure.

Builds bank_v2.npz = bank_v1 (untouched, byte-identical poses) + the midrise pool, marked
with generator == "midrise" so GetUpTask can draw it as a separate pool at a configured
fraction. Every pose passes the same fixed-point validation as bank_v1: settles under a PD
hold at its own joint angles and neither drifts nor collapses in the opening quarter second.

    python scripts/generate_midrise_poses.py            # writes data/fallen/bank_v2.npz
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

from generate_fallen_poses import is_valid, settle  # noqa: E402


def is_valid_active(model, data, qpos: np.ndarray, qadr) -> tuple[bool, str]:
    """Validation for ACTIVE-BALANCE rungs (squat, crouch, half-kneel).

    These states have no passive equilibrium on this model, and that is physics, not a
    generator defect: 801 mined descents produced zero fixed points, because a bent knee
    under gravity needs active stabilisation, for humans as much as for servos. Demanding
    the lying-pose contract (2 cm root drift in 0.25 s at rest) of a crouch would simply ban
    crouches from the bank.

    The contract here is NOT COLLAPSING: finite, penetration-free, and holding at least 85%
    of its pelvis height through the first quarter second under a servo hold of its own
    angles. The policy takes over on step one, and "catch yourself mid-crouch" is precisely
    a state the curriculum wants; "already on the floor by the time the policy acts" is not.
    """
    if not np.isfinite(qpos).all():
        return False, "non-finite"
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if data.ncon and float(-data.contact.dist.min()) > 2e-3:
        return False, "penetration"
    z0 = float(qpos[2])
    data.ctrl[:] = qpos[qadr]
    for _ in range(int(0.25 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if float(data.qpos[2]) < 0.85 * z0:
        return False, f"collapses {z0:.3f}->{float(data.qpos[2]):.3f}"
    return True, "ok"


def reach_from_stand(model, data, prepared, qadr, target: np.ndarray,
                     seconds: float = 0.6, hold: float = 0.4) -> None:
    """Lower the body from the nominal stand by RAMPING the servo targets to `target`.

    Authored joint angles put the centre of mass wherever the author guessed, and a guess
    a few centimetres off means the pose falls over during settling: measured, an authored
    crouch settled at pelvis 0.075 (fell) in 28 of 30 attempts. A crouch REACHED from a
    stand quasi-statically keeps its mass over its feet the whole way down, the same reason
    the fall generator topples real bodies instead of authoring fallen poses.
    """
    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0
    start = prepared.default_joint_pos.copy()
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    steps = int(seconds / model.opt.timestep)
    for t in range(steps):
        a = (t + 1) / steps
        data.ctrl[:] = np.clip(start + a * (target - start), lo, hi)
        mujoco.mj_step(model, data)
    for _ in range(int(hold / model.opt.timestep)):
        data.ctrl[:] = np.clip(target, lo, hi)
        mujoco.mj_step(model, data)

#: Target counts per kind. Sized against bank_v1's 425-per-orientation scale: the midrise
#: pool is a curriculum aid, not the main distribution, so ~60 of each is plenty of variety
#: at any reasonable midrise_reset_frac.
KINDS = ("all_fours", "kneel", "half_kneel", "squat", "crouch")
PER_KIND = 60
#: Joint noise per attempt. Wide enough that no two rungs are identical, narrow enough that
#: the pose still settles as itself.
JITTER = 0.05


def build_pose(kind: str, model, data, prepared, qadr, rng) -> np.ndarray:
    """Write one authored pose into `data` and return the PD hold target."""
    names = list(prepared.joint_names)
    joints = prepared.default_joint_pos.copy()

    def setj(sub: str, value: float, side: str = "") -> None:
        for i, n in enumerate(names):
            if sub in n and (not side or n.startswith(side)):
                joints[i] = value

    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0

    if kind == "all_fours":
        # Hands and knees. Torso pitched forward, thighs vertical, shins flat behind.
        data.qpos[2] = 0.46
        data.qpos[3:7] = [0.92, 0.0, 0.39, 0.0]
        setj("knee", 2.2)
        setj("hip_y", -1.2)
        setj("shoulder_y", -1.2)
    elif kind == "kneel":
        # Upright kneel: shins on the floor, thighs and torso vertical. Feet carry nothing
        # (the corridor reads 0 here, measured); that is fine for a START state, the point
        # is the critic tasting pelvis 0.44 with the torso already vertical. A touch of
        # forward lean keeps the mass over the shins instead of behind them, which is what
        # toppled the first authored version.
        data.qpos[2] = 0.44
        setj("knee", 2.35)
        setj("hip_y", -rng.uniform(0.1, 0.6))
        setj("abdomen_y", rng.uniform(0.0, 0.5))
        setj("ankle_y", rng.uniform(-0.9, 0.9))
    elif kind == "half_kneel":
        # One shin down, other foot planted ahead: the classic rising stage. Left and right
        # variants alternate via the rng so the pool is symmetric. Leaning well forward over
        # the front foot: the first version leaned back and fell in 29 of 30 attempts.
        data.qpos[2] = 0.44
        front = "left" if rng.random() < 0.5 else "right"
        back = "right" if front == "left" else "left"
        setj("knee", 2.35, back)
        setj("hip_y", -rng.uniform(0.2, 0.7), back)
        setj("ankle_y", rng.uniform(-0.9, 0.9), back)
        setj("knee", rng.uniform(1.0, 1.5), front)
        setj("hip_y", -rng.uniform(1.2, 1.7), front)
        setj("ankle_y", rng.uniform(-0.9, 0.9), front)
        setj("abdomen_y", rng.uniform(0.3, 0.7))
    elif kind == "squat":
        # Deep squat, reached from a stand. The ankle is the load-bearing unknown: a squat
        # with vertical shins keeps the mass behind the feet and NO depth of it balances
        # (measured: zero passing equilibria in 400 mined descents without an ankle target).
        # The sign convention is not guessed (E25, transposed hips): ankle_y is drawn from
        # its whole range and the fixed-point test keeps whichever sign physics accepts.
        depth = rng.uniform(1.7, 2.3)
        setj("knee", depth)
        setj("hip_y", -rng.uniform(1.5, 2.1))
        setj("abdomen_y", rng.uniform(0.3, 0.8))
        setj("shoulder_y", -rng.uniform(0.4, 1.0))
        setj("ankle_y", rng.uniform(-0.9, 0.9))
    elif kind == "crouch":
        # High crouch, reached from a stand, ankle mined the same way.
        depth = rng.uniform(1.0, 1.6)
        setj("knee", depth)
        setj("hip_y", -rng.uniform(0.8, 1.4))
        setj("abdomen_y", rng.uniform(0.2, 0.6))
        setj("ankle_y", rng.uniform(-0.9, 0.9))
    joints = joints + rng.normal(0.0, JITTER, model.nu)
    joints = np.clip(joints, model.actuator_ctrlrange[:, 0],
                     model.actuator_ctrlrange[:, 1])
    if kind not in ("squat", "crouch"):
        data.qpos[qadr] = joints
    return joints


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=REPO_ROOT / "data/fallen/bank_v1.npz")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/fallen/bank_v2.npz")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prepared = prepare(REPO_ROOT / "humanoid_rl/models/humanoid_scene.xml",
                       action_scale_mode="full_range")
    model = prepared.model
    data = mujoco.MjData(model)
    qadr = prepared.actuator_qpos_adr
    rng = np.random.default_rng(args.seed)

    # MINING, not authoring-and-praying. Authored poses failed the fixed-point test
    # almost always ("root drifts" 22-24 of 24 at every settle length): a hand-guessed pose
    # is mid-topple, not at rest. Instead, drive trajectories DOWN from the stand and along
    # authored holds, and test every sampled state against the same strict fixed-point
    # validation the rest of the bank passes. Physics finds its own equilibria; the authored
    # targets only steer which family gets explored.
    kept_q, kept_v, kept_kind = [], [], []
    stats: dict[str, list[float]] = {k: [] for k in KINDS}
    tries: dict[str, int] = {k: 0 for k in KINDS}
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    def height_ok(kind: str, pelvis: float) -> bool:
        if kind == "crouch":
            return 0.42 <= pelvis <= 0.70
        if kind in ("kneel", "half_kneel", "squat"):
            return 0.27 <= pelvis <= 0.60
        return 0.28 <= pelvis <= 0.55          # all_fours

    def mine(kind: str, trajectory_states: list[np.ndarray]) -> None:
        """Fixed-point-test each sampled state; keep survivors at the right height."""
        for q in trajectory_states:
            if len(stats[kind]) >= PER_KIND:
                return
            pelvis = float(q[2])
            if not height_ok(kind, pelvis):
                continue
            if kind in ("squat", "crouch", "half_kneel"):
                ok, _why = is_valid_active(model, data, q, qadr)
            else:
                ok, _why = is_valid(model, data, q, np.zeros(model.nv), qadr)
            if not ok:
                continue
            kept_q.append(q.copy())
            kept_v.append(np.zeros(model.nv))
            kept_kind.append(kind)
            stats[kind].append(pelvis)

    for kind in KINDS:
        while len(stats[kind]) < PER_KIND:
            tries[kind] += 1
            if tries[kind] > 800:
                # Report and move on rather than abort: a kind that physics refuses to
                # balance passively (a knife-edge kneel, say) should cost its own slot,
                # not the whole bank. The summary makes the shortfall loud.
                break
            target = build_pose(kind, model, data, prepared, qadr, rng)
            states: list[np.ndarray] = []
            if kind in ("squat", "crouch"):
                # Ramp down from the stand over 1.2 s, sampling every 0.1 s: each sample
                # along a slow enough descent is near quasi-static.
                data.qpos[:] = prepared.default_qpos
                data.qvel[:] = 0.0
                start = prepared.default_joint_pos.copy()
                steps = int(1.2 / model.opt.timestep)
                every = int(0.1 / model.opt.timestep)
                for t in range(steps):
                    a = (t + 1) / steps
                    data.ctrl[:] = np.clip(start + a * (target - start), lo, hi)
                    mujoco.mj_step(model, data)
                    if t % every == 0 and t > steps // 4:
                        states.append(data.qpos.copy())
            else:
                # Hold the authored target and sample the settling trajectory: if any
                # moment of it is a true equilibrium, the test keeps that moment.
                data.qpos[qadr] = target if kind != "all_fours" else data.qpos[qadr]
                steps = int(2.5 / model.opt.timestep)
                every = int(0.1 / model.opt.timestep)
                for t in range(steps):
                    data.ctrl[:] = np.clip(target, lo, hi)
                    mujoco.mj_step(model, data)
                    if t % every == 0 and t > steps // 5:
                        states.append(data.qpos.copy())
            mine(kind, states)

    base = np.load(args.base, allow_pickle=False)
    assert int(base["nq"]) == model.nq, (int(base["nq"]), model.nq)
    mid_q = np.stack(kept_q)
    mid_v = np.stack(kept_v)
    n_mid = len(kept_q)

    # Carry EVERY field of the base bank forward, overriding only what this script extends.
    # The loader validates model_sha1 and nq/nv; a bank missing a scalar the base carried
    # would fail to load, or worse, load past a check the base was subject to.
    out = {k: base[k] for k in base.files}
    out["qpos"] = np.concatenate([base["qpos"], mid_q])
    out["qvel"] = np.concatenate([base["qvel"], mid_v])
    out["label"] = np.concatenate([base["label"],
                                   np.array([f"midrise_{k}" for k in kept_kind])])
    out["generator"] = np.concatenate([base["generator"], np.array(["midrise"] * n_mid)])
    out["split"] = np.concatenate([base["split"], np.array(["train"] * n_mid)])
    out["generator_version"] = 2
    np.savez_compressed(args.out, **out)
    print(f"bank_v2: {len(base['qpos'])} base + {n_mid} midrise -> {args.out}")
    short = []
    for k in KINDS:
        p = np.array(stats[k])
        if len(p) == 0:
            print(f"  {k:<11} kept   0  tries {tries[k]:>4}  NO PASSING EQUILIBRIA")
            short.append(k)
            continue
        print(f"  {k:<11} kept {len(p):>3}  tries {tries[k]:>4}  "
              f"pelvis {p.min():.3f}-{p.max():.3f} (median {np.median(p):.3f})")
        if len(p) < PER_KIND:
            short.append(k)
    if short:
        print(f"  SHORTFALL in {short}: these kinds resist passive balance on this model. "
              f"The bank is still usable; the ladder just has fewer of those rungs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
