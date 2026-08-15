"""Part 2: torso frame offset, and the hackability floor UNDER THE WIDENED ACTION BAND.

The published hackability numbers were measured with action_scale_fraction = 0.6.
Widening the band to cover the full joint range re-opens the "flick one joint" question,
so it has to be re-measured against the exact standing predicate the spec proposes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path("/Users/BrickLayer/Desktop/HumonoidAI")
sys.path.insert(0, str(REPO))
from humanoid_rl.envs.model_prep import prepare  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

P = prepare(REPO / "humanoid_rl/models/humanoid_scene.xml")
M = P.model
STAND_H = P.standing_height
STAND_HEAD = P.standing_head_height
DT_PHYS = M.opt.timestep
DEC = 4
DT = DEC * DT_PHYS                      # 0.008 s, 125 Hz

qadr = {mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j): int(M.jnt_qposadr[j])
        for j in range(M.njnt)}
ACT = P.joint_names
ACT2Q = np.array([qadr[n] for n in ACT], dtype=int)
LO, HI = M.actuator_ctrlrange[:, 0], M.actuator_ctrlrange[:, 1]
NOM = P.default_joint_pos
SCALE_OLD = P.action_scale
SCALE_NEW = np.maximum(NOM - LO, HI - NOM)

TOUCH = P.foot_touch_adr
KEY = {n: a for n, a in zip(P.key_body_names, P.key_body_adr)}
TZ, HEAD = P.torso_zaxis_adr, P.head_pos_adr
FORCE_TH = 0.02 * M.body_mass.sum() * 9.81


def qri(q, v):
    w, u = q[0], q[1:4]
    return v * (2 * w * w - 1) - 2 * w * np.cross(u, v) + 2 * u * np.dot(u, v)


# ------------------------------------------------------------------ torso frame offset
d = mujoco.MjData(M)
d.qpos[:] = P.default_qpos
mujoco.mj_forward(M, d)
tb = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "torso")
xmat = d.xmat[tb].reshape(3, 3)
ximat = d.ximat[tb].reshape(3, 3)
print("torso body-frame z-axis  (xmat[:,2]) :", xmat[:, 2])
print("torso inertial z-axis   (ximat[:,2]) :", ximat[:, 2])
print("sensor torso_zaxis                   :", d.sensordata[TZ:TZ + 3])
print("=> mjOBJ_BODY frame sensors report the INERTIAL frame (xipos/ximat), not the body frame")
tilt = np.degrees(np.arccos(np.clip(ximat[:, 2] @ xmat[:, 2], -1, 1)))
print(f"   constant offset between them, standing: {tilt:.2f} deg  (cos = {ximat[:,2]@xmat[:,2]:.4f})")
fb = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
print(f"left_foot  xpos {d.xpos[fb]}  xipos {d.xipos[fb]}  sensor {d.sensordata[KEY['left_foot']:KEY['left_foot']+3]}")
hb = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "head")
print(f"head       xpos {d.xpos[hb]}  xipos {d.xipos[hb]}  sensor {d.sensordata[HEAD:HEAD+3]}")


# ------------------------------------------------------------------ signals
def read(d):
    q = d.qpos
    quat = q[3:7]
    g = qri(quat, np.array([0.0, 0.0, -1.0]))
    f = d.sensordata[TOUCH]
    return dict(
        root_h=float(q[2]),
        g=g,
        tup=float(d.sensordata[TZ + 2]),
        head_ratio=float(d.sensordata[HEAD + 2]) / STAND_HEAD,
        foot_z=np.array([d.sensordata[KEY["left_foot"] + 2], d.sensordata[KEY["right_foot"] + 2]]),
        hand_z=np.array([d.sensordata[KEY["left_hand"] + 2], d.sensordata[KEY["right_hand"] + 2]]),
        contact=f > FORCE_TH,
        force=f.copy(),
        lin=np.linalg.norm(q[:2] * 0 + d.qvel[:2]),
        vz=float(d.qvel[2]),
        w=np.linalg.norm(d.qvel[3:6]),
    )


def standing(s):
    """The spec's STANDING predicate, all conjuncts."""
    return (
        s["root_h"] >= 0.82 * STAND_H
        and s["head_ratio"] >= 0.88
        and s["g"][2] <= -0.90
        and s["tup"] >= 0.90
        and bool(s["contact"].all())
        and float(s["foot_z"].max()) <= 0.08
        and float(s["hand_z"].min()) >= 0.45
        and abs(s["vz"]) <= 0.35
        and s["w"] <= 1.5
    )


# ------------------------------------------------------------------ fallen pose bank
def topple(n, seed=0):
    r = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        d = mujoco.MjData(M)
        mujoco.mj_resetData(M, d)
        d.qpos[:] = P.default_qpos
        d.qpos[7:] += r.normal(0, 0.05, M.nq - 7)
        # a random constant PD target inside the OLD commandable band, per the research
        tgt = np.clip(NOM + r.uniform(-1, 1, M.nu) * SCALE_OLD, LO, HI)
        d.ctrl[:] = tgt
        ang = r.uniform(0, 2 * np.pi)
        mag = r.uniform(1.5, 5.0)
        d.qvel[0] = mag * np.cos(ang)
        d.qvel[1] = mag * np.sin(ang)
        mujoco.mj_step(M, d, nstep=int(2.5 / DT_PHYS))
        d.qvel[:] = 0.0
        mujoco.mj_forward(M, d)
        s = read(d)
        if s["root_h"] < 0.45 and s["g"][2] > -0.7:
            out.append((d.qpos.copy(), d.qvel.copy(), s))
    return out


BANK = topple(16, seed=3)
lab = []
for _, _, s in BANK:
    g = s["g"]
    lab.append("supine" if g[0] < -0.6 else "prone" if g[0] > 0.6 else
               "side" if abs(g[1]) > 0.6 else "other")
print(f"\nfallen bank: {len(BANK)} poses, labels {dict((l, lab.count(l)) for l in set(lab))}")
print("  root heights:", np.round([s['root_h'] for _, _, s in BANK], 3))


# ------------------------------------------------------------------ rollout
def rollout(qpos, qvel, action, scale, seconds=5.0, hold_needed=2.0):
    d = mujoco.MjData(M)
    mujoco.mj_resetData(M, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(M, d)
    ctrl = np.clip(NOM + action * scale, LO, HI)
    # the fix from research 3.2: filter starts at the pose's own angles
    filt = d.qpos[ACT2Q].copy()
    beta = 1.0 - np.exp(-2 * np.pi * 8.0 * DT)
    steps = int(seconds / DT)
    need = int(hold_needed / DT)
    run = 0
    best_run = 0
    peak = 0.0
    ever = False
    for _ in range(steps):
        filt += beta * (ctrl - filt)
        d.ctrl[:] = filt
        mujoco.mj_step(M, d, nstep=DEC)
        if not np.isfinite(d.qpos).all():
            break
        s = read(d)
        peak = max(peak, s["root_h"])
        if standing(s):
            ever = True
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return dict(peak=peak, ever=ever, held=best_run >= need, best_run_s=best_run * DT)


def sweep(name, actions, scale):
    stood = held = 0
    peak = 0.0
    n = 0
    for qpos, qvel, _ in BANK:
        for a in actions:
            r = rollout(qpos, qvel, a, scale)
            n += 1
            stood += r["ever"]
            held += r["held"]
            peak = max(peak, r["peak"])
    print(f"{name:<44} n={n:<5} momentary {stood/n:6.1%}  HELD>=2s {held/n:6.1%}  peak_h {peak:.3f}")
    return stood / n, held / n, peak


print("\n############ HACKABILITY under the WIDENED band (scale = full half-range)")
zero = [np.zeros(M.nu)]
sweep("zero action (hold nominal standing target)", zero, SCALE_NEW)

flicks = []
for j in range(M.nu):
    for v in (-1.0, 1.0):
        a = np.zeros(M.nu)
        a[j] = v
        flicks.append(a)
sweep("single-joint flick, all 28 joints x +-1", flicks, SCALE_NEW)

pairs = []
for names, v in (
    (["left_knee", "right_knee"], 1.0), (["left_knee", "right_knee"], -1.0),
    (["abdomen_y"], 1.0), (["abdomen_y"], -1.0),
    (["left_hip_y", "right_hip_y"], -1.0), (["left_hip_y", "right_hip_y"], 1.0),
    (["left_elbow", "right_elbow"], 1.0),
    (["left_shoulder_x", "right_shoulder_x"], -1.0),
):
    a = np.zeros(M.nu)
    for nm in names:
        a[ACT.index(nm)] = v * (-1 if nm.startswith("left") and "elbow" in nm else 1)
    pairs.append(a)
sweep("symmetric 2-joint flicks (knees/abdomen/hips/etc)", pairs, SCALE_NEW)

rand = [rng.uniform(-1, 1, M.nu) for _ in range(120)]
sweep("120 random CONSTANT full-body targets", rand, SCALE_NEW)
sweep("120 random CONSTANT targets, OLD band (control)", rand, SCALE_OLD)
