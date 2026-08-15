"""Part 3: strict conjunctive predicate vs the naive one, on identical rollouts,
plus a settled-standing sanity check and the stage ladder evaluated on canonical poses."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path("/Users/BrickLayer/Desktop/HumonoidAI")
sys.path.insert(0, str(REPO))
from humanoid_rl.envs.model_prep import prepare  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(1)
P = prepare(REPO / "humanoid_rl/models/humanoid_scene.xml")
M = P.model
H, HEADH = P.standing_height, P.standing_head_height
DT_PHYS = M.opt.timestep
DEC, DT = 4, 4 * M.opt.timestep
qadr = {mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j): int(M.jnt_qposadr[j])
        for j in range(M.njnt)}
ACT = P.joint_names
ACT2Q = np.array([qadr[n] for n in ACT], dtype=int)
LO, HI, NOM = M.actuator_ctrlrange[:, 0], M.actuator_ctrlrange[:, 1], P.default_joint_pos
SCALE_OLD, SCALE_NEW = P.action_scale, np.maximum(NOM - LO, HI - NOM)
TOUCH, TZ, HEAD = P.foot_touch_adr, P.torso_zaxis_adr, P.head_pos_adr
K = {n: a for n, a in zip(P.key_body_names, P.key_body_adr)}
BW = float(M.body_mass.sum() * 9.81)
KNEE_Q = [qadr["left_knee"], qadr["right_knee"]]


def qri(q, v):
    w, u = q[0], q[1:4]
    return v * (2 * w * w - 1) - 2 * w * np.cross(u, v) + 2 * u * np.dot(u, v)


def sig(d):
    q, s = d.qpos, d.sensordata
    return dict(
        root_h=float(q[2]), g=qri(q[3:7], np.array([0.0, 0.0, -1.0])),
        tup=float(s[TZ + 2]), head=float(s[HEAD + 2]) / HEADH,
        fz=np.array([s[K["left_foot"] + 2], s[K["right_foot"] + 2]]),
        hz=np.array([s[K["left_hand"] + 2], s[K["right_hand"] + 2]]),
        ff=s[TOUCH].copy(), knee=float(np.mean(np.abs(q[KNEE_Q]))),
        vz=float(d.qvel[2]), w=float(np.linalg.norm(d.qvel[3:6])),
    )


def STRICT(s):
    return (s["root_h"] >= 0.88 * H and s["head"] >= 0.90 and s["g"][2] <= -0.93
            and s["tup"] >= 0.90 and bool((s["ff"] > 0.02 * BW).all())
            and float(s["fz"].max()) <= 0.08 and float(s["hz"].min()) >= 0.45
            and s["knee"] <= 0.50 and abs(s["vz"]) <= 0.30 and s["w"] <= 1.5)


def NAIVE(s):   # the test this project would have written: height + torso_upright
    return s["root_h"] > 0.62 * H and s["tup"] > 0.8


def stage(s):
    S1 = (abs(s["g"][0]) <= 0.5 and abs(s["g"][1]) >= 0.5) or s["head"] >= 0.25
    S2 = ((float(s["hz"].min()) <= 0.10 and s["root_h"] >= 0.35 and s["g"][0] >= 0.5
           and s["knee"] >= 1.0)
          or (s["g"][2] <= -0.75 and s["head"] >= 0.45))
    loaded = (s["ff"] >= 0.25 * BW) & (s["fz"] <= 0.08)
    S3 = bool(loaded.any()) and (s["root_h"] - float(s["fz"][np.argmax(s['ff'])]) >= 0.30) \
        and s["head"] >= 0.55
    S4 = STRICT(s)
    k = 0
    for i, p in enumerate([S1, S2, S3, S4], start=1):
        if p:
            k = i
    return k, (S1, S2, S3, S4)


# ---------------- settled standing check (real contact forces)
d = mujoco.MjData(M)
d.qpos[:] = P.default_qpos
d.ctrl[:] = P.default_qpos[ACT2Q]
mujoco.mj_step(M, d, nstep=int(0.3 / DT_PHYS))
s = sig(d)
print("settled standing:", {k_: (np.round(v, 3) if isinstance(v, np.ndarray) else round(v, 3))
                            for k_, v in s.items()})
print("  STRICT", STRICT(s), " NAIVE", NAIVE(s), " stage", stage(s))

# ---------------- canonical poses through the ladder
def make(joints, quat=(1, 0, 0, 0), settle=0.6):
    q = P.default_qpos.copy()
    q[3:7] = np.asarray(quat, float) / np.linalg.norm(quat)
    for n, v in joints.items():
        q[qadr[n]] = v
    dd = mujoco.MjData(M)
    dd.qpos[:] = q
    dd.qpos[2] = 1.5
    mujoco.mj_forward(M, dd)
    lowest = min(
        (float(dd.geom_xpos[g, 2]) - (
            (abs(dd.geom_xmat[g].reshape(3, 3)[2, 0]) * M.geom_size[g, 0]
             + abs(dd.geom_xmat[g].reshape(3, 3)[2, 1]) * M.geom_size[g, 1]
             + abs(dd.geom_xmat[g].reshape(3, 3)[2, 2]) * M.geom_size[g, 2])
            if M.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else
            (M.geom_size[g, 0] if M.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else
             abs(dd.geom_xmat[g].reshape(3, 3)[2, 2]) * M.geom_size[g, 1] + M.geom_size[g, 0])))
        for g in range(M.ngeom) if g != P.floor_geom_id)
    dd.qpos[2] = 1.5 - lowest + 0.002
    mujoco.mj_forward(M, dd)
    if settle:
        dd.ctrl[:] = dd.qpos[ACT2Q]
        mujoco.mj_step(M, dd, nstep=int(settle / DT_PHYS))
    return dd


def rq(ax, a):
    ax = np.asarray(ax, float)
    return np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * ax / np.linalg.norm(ax)])


POSES = {
    "standing": ({}, (1, 0, 0, 0)),
    "supine": ({}, rq([0, 1, 0], -np.pi / 2)),
    "prone": ({}, rq([0, 1, 0], np.pi / 2)),
    "side": ({}, rq([1, 0, 0], -np.pi / 2)),
    "seated_legs_out": ({"left_hip_y": -1.5, "right_hip_y": -1.5, "left_knee": .1,
                         "right_knee": .1, "left_shoulder_x": -1.2, "right_shoulder_x": 1.2},
                        (1, 0, 0, 0)),
    "all_fours": ({"left_hip_y": -1.6, "right_hip_y": -1.6, "left_knee": 1.6,
                   "right_knee": 1.6, "left_shoulder_x": -0.2, "right_shoulder_x": 0.2,
                   "left_shoulder_y": 0.0, "right_shoulder_y": 0.0},
                  rq([0, 1, 0], np.pi / 2)),
    "kneel": ({"left_knee": 2.6, "right_knee": 2.6, "left_hip_y": -.15,
               "right_hip_y": -.15}, (1, 0, 0, 0)),
    "half_kneel": ({"left_knee": 2.6, "left_hip_y": -.15, "right_knee": 1.5,
                    "right_hip_y": -1.4, "right_ankle_y": .3}, (1, 0, 0, 0)),
    "deep_squat": ({"left_hip_y": -1.8, "right_hip_y": -1.8, "left_knee": 2.3,
                    "right_knee": 2.3, "left_ankle_y": .7, "right_ankle_y": .7,
                    "abdomen_y": .6}, (1, 0, 0, 0)),
    "half_crouch": ({"left_hip_y": -.9, "right_hip_y": -.9, "left_knee": 1.2,
                     "right_knee": 1.2, "left_ankle_y": .4, "right_ankle_y": .4,
                     "abdomen_y": .3}, (1, 0, 0, 0)),
    "fold_fwd": ({"abdomen_y": 1.4}, (1, 0, 0, 0)),
    "fold_back": ({"abdomen_y": -0.9}, (1, 0, 0, 0)),
}
print(f"\n{'pose':<18}{'root/H':>7}{'head':>7}{'g_x':>7}{'g_z':>7}{'tup':>7}{'fzmax':>7}"
      f"{'hzmin':>7}{'knee':>6}{'Ffoot/bw':>10}   S1 S2 S3 S4  stage  STRICT NAIVE")
for name, (j, q) in POSES.items():
    s = sig(make(j, q))
    k, ps = stage(s)
    print(f"{name:<18}{s['root_h']/H:7.2f}{s['head']:7.3f}{s['g'][0]:7.2f}{s['g'][2]:7.2f}"
          f"{s['tup']:7.3f}{s['fz'].max():7.3f}{s['hz'].min():7.3f}{s['knee']:6.2f}"
          f"{s['ff'].sum()/BW:10.2f}    {int(ps[0])}  {int(ps[1])}  {int(ps[2])}  {int(ps[3])}"
          f"   S{k}    {int(STRICT(s))}     {int(NAIVE(s))}")

# ---------------- strict vs naive on identical rollouts
def topple(n, seed=3):
    r = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        dd = mujoco.MjData(M)
        dd.qpos[:] = P.default_qpos
        dd.qpos[7:] += r.normal(0, .05, M.nq - 7)
        dd.ctrl[:] = np.clip(NOM + r.uniform(-1, 1, M.nu) * SCALE_OLD, LO, HI)
        a, mag = r.uniform(0, 2 * np.pi), r.uniform(1.5, 5.0)
        dd.qvel[0], dd.qvel[1] = mag * np.cos(a), mag * np.sin(a)
        mujoco.mj_step(M, dd, nstep=int(2.5 / DT_PHYS))
        dd.qvel[:] = 0
        mujoco.mj_forward(M, dd)
        s = sig(dd)
        if s["root_h"] < .45 and s["g"][2] > -.7:
            out.append((dd.qpos.copy(), dd.qvel.copy(), s))
    return out


BANK = topple(16)
print("\nreset-stage of the 16 toppled bank poses:", [stage(s)[0] for _, _, s in BANK])

beta = 1.0 - np.exp(-2 * np.pi * 8.0 * DT)
res = {"strict_ever": 0, "strict_held": 0, "naive_ever": 0, "naive_held": 0, "n": 0,
       "peak": 0.0, "maxstage": 0}
acts = [rng.uniform(-1, 1, M.nu) for _ in range(40)]
need = int(2.0 / DT)
for qpos, qvel, _ in BANK:
    for a in acts:
        dd = mujoco.MjData(M)
        dd.qpos[:], dd.qvel[:] = qpos, qvel
        mujoco.mj_forward(M, dd)
        ctrl = np.clip(NOM + a * SCALE_NEW, LO, HI)
        filt = dd.qpos[ACT2Q].copy()
        rs = rn = bs = bn = 0
        es = en = False
        for _ in range(int(5.0 / DT)):
            filt += beta * (ctrl - filt)
            dd.ctrl[:] = filt
            mujoco.mj_step(M, dd, nstep=DEC)
            if not np.isfinite(dd.qpos).all():
                break
            s = sig(dd)
            res["peak"] = max(res["peak"], s["root_h"])
            res["maxstage"] = max(res["maxstage"], stage(s)[0])
            if STRICT(s):
                es = True
                rs += 1
                bs = max(bs, rs)
            else:
                rs = 0
            if NAIVE(s):
                en = True
                rn += 1
                bn = max(bn, rn)
            else:
                rn = 0
        res["n"] += 1
        res["strict_ever"] += es
        res["strict_held"] += bs >= need
        res["naive_ever"] += en
        res["naive_held"] += bn >= need
n = res["n"]
print(f"\n640-rollout constant-target oracle, widened band, 5 s each:")
print(f"  NAIVE  (root>0.62H and torso_upright>0.8):  momentary {res['naive_ever']/n:.1%}"
      f"   held>=2s {res['naive_held']/n:.1%}")
print(f"  STRICT (9-clause conjunction):              momentary {res['strict_ever']/n:.1%}"
      f"   held>=2s {res['strict_held']/n:.1%}")
print(f"  peak root height reached {res['peak']:.3f} m ({res['peak']/H:.2f}x standing);"
      f" highest stage reached S{res['maxstage']}")
