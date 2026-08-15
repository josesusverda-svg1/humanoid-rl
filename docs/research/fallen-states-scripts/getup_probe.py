"""Measure the signals a staged get-up task would key on, on THIS model.

Nothing here is assumed: every threshold in the spec comes out of this script.
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

P = prepare(REPO / "humanoid_rl/models/humanoid_scene.xml")
M = P.model
STAND_H = P.standing_height
STAND_HEAD = P.standing_head_height

# ---------------------------------------------------------------- index maps
qadr = {}
for j in range(M.njnt):
    name = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j)
    qadr[name] = int(M.jnt_qposadr[j])

ACT = P.joint_names                      # actuator order, length nu
# actuator index -> qpos index
ACT2Q = np.array([qadr[n] for n in ACT], dtype=int)
# qpos hinge slot (7..) -> actuator index
Q2ACT = np.zeros(M.nu, dtype=int)
for a, q in enumerate(ACT2Q):
    Q2ACT[q - 7] = a
print("ACT2Q:", ACT2Q.tolist())
print("Q2ACT:", Q2ACT.tolist())
print("identity?", bool(np.all(ACT2Q == np.arange(7, 7 + M.nu))))

# ---------------------------------------------------------------- force limits
print("\njnt_actfrclimited:", M.jnt_actfrclimited[:6], "...")
print("jnt_actfrcrange[1:6]:\n", M.jnt_actfrcrange[1:6])
print("actuator_forcelimited:", M.actuator_forcelimited[:6])
print("actuator_forcerange[:6]:\n", M.actuator_forcerange[:6])
print("actuator_ctrlrange[:4]:\n", M.actuator_ctrlrange[:4])

# ---------------------------------------------------------------- sensors
tz = P.torso_zaxis_adr
head = P.head_pos_adr
key = P.key_body_names
print("\nkey bodies:", key, "torso_zaxis_adr", tz, "head_adr", head)

data = mujoco.MjData(M)


def lowest_z(d):
    floor = P.floor_geom_id
    return min(
        _geom_low(g, d) for g in range(M.ngeom) if g != floor
    )


def _geom_low(g, d):
    pos_z = float(d.geom_xpos[g, 2])
    size = M.geom_size[g]
    t = M.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        r = d.geom_xmat[g].reshape(3, 3)
        return pos_z - (abs(r[2, 0]) * size[0] + abs(r[2, 1]) * size[1] + abs(r[2, 2]) * size[2])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos_z - float(size[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r = d.geom_xmat[g].reshape(3, 3)
        return pos_z - (abs(r[2, 2]) * float(size[1]) + float(size[0]))
    return pos_z - float(M.geom_rbound[g])


def quat_rotate_inverse(q, v):
    w = q[0]
    u = q[1:4]
    return v * (2 * w * w - 1) - 2 * w * np.cross(u, v) + 2 * u * np.dot(u, v)


def axis_angle_quat(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def make(joints: dict, quat=(1, 0, 0, 0), settle=0.0, drop=0.002):
    q = P.default_qpos.copy()
    q[3:7] = np.asarray(quat, float) / np.linalg.norm(quat)
    for name, val in joints.items():
        q[qadr[name]] = val
    d = mujoco.MjData(M)
    mujoco.mj_resetData(M, d)
    d.qpos[:] = q
    d.qpos[2] = 1.5
    mujoco.mj_forward(M, d)
    d.qpos[2] = 1.5 - lowest_z(d) + drop
    mujoco.mj_forward(M, d)
    if settle > 0:
        d.ctrl[:] = d.qpos[ACT2Q]
        mujoco.mj_step(M, d, nstep=int(settle / M.opt.timestep))
        mujoco.mj_forward(M, d)
    return d


def signals(d, label):
    q = d.qpos
    quat = q[3:7]
    g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
    tup = float(d.sensordata[tz + 2])
    hh = float(d.sensordata[head + 2])
    kb = {n: d.sensordata[a:a + 3].copy() for n, a in zip(key, P.key_body_adr)}
    # torso gravity via abdomen chain, from qpos only
    ax = q[qadr["abdomen_x"]]
    ay = q[qadr["abdomen_y"]]
    az = q[qadr["abdomen_z"]]
    qa = qmul(qmul(axis_angle_quat([1, 0, 0], ax), axis_angle_quat([0, 1, 0], ay)),
              axis_angle_quat([0, 0, 1], az))
    g_torso = quat_rotate_inverse(qa, g)
    # contacts
    foot_touch = d.sensordata[P.foot_touch_adr]
    root = q[:3]
    heading = np.arctan2(2 * (quat[1] * quat[2] + quat[0] * quat[3]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
    c, s = np.cos(-heading), np.sin(-heading)

    def local(p):
        r = p - root
        return np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])

    lf, rf = local(kb["left_foot"]), local(kb["right_foot"])
    lh, rh = local(kb["left_hand"]), local(kb["right_hand"])
    print(f"\n=== {label}")
    print(f"  root_h {root[2]:.3f} ({root[2]/STAND_H:.2f}x)   head {hh:.3f} ratio {hh/STAND_HEAD:.3f}")
    print(f"  gravity_body  {g}   torso_upright(sensor) {tup:+.3f}")
    print(f"  g_torso(qpos) {g_torso}   -g_torso_z {-g_torso[2]:+.3f}  ERR {abs(-g_torso[2]-tup):.2e}")
    print(f"  foot z  L {kb['left_foot'][2]:.3f} R {kb['right_foot'][2]:.3f}"
          f"   hand z  L {kb['left_hand'][2]:.3f} R {kb['right_hand'][2]:.3f}")
    print(f"  foot local (fwd,lat,dz vs root)  L {lf}  R {rf}")
    print(f"  hand local                       L {lh}  R {rh}")
    print(f"  touch N  L {foot_touch[0]:7.1f}  R {foot_touch[1]:7.1f}")
    print(f"  knee L {q[qadr['left_knee']]:+.2f} R {q[qadr['right_knee']]:+.2f}"
          f"  hip_y L {q[qadr['left_hip_y']]:+.2f} R {q[qadr['right_hip_y']]:+.2f}"
          f"  abd_y {ay:+.2f}")
    return dict(root_h=root[2], g=g, tup=tup, g_torso=g_torso, head_ratio=hh / STAND_HEAD,
                lf=lf, rf=rf, lh=lh, rh=rh, touch=foot_touch.copy())


print("\n\n############ CANONICAL POSES (mj_forward, resting on floor)")
print(f"standing_height {STAND_H:.4f}  standing_head {STAND_HEAD:.4f}  "
      f"mass {M.body_mass.sum():.1f} kg  weight {M.body_mass.sum()*9.81:.0f} N")
print(f"contact_force_threshold (2% bw) = {0.02*M.body_mass.sum()*9.81:.2f} N")

R = {}
R["standing"] = signals(make({}), "standing (nominal)")
R["supine"] = signals(make({}, quat=axis_angle_quat([0, 1, 0], -np.pi / 2)), "supine (rot -90 about y)")
R["prone"] = signals(make({}, quat=axis_angle_quat([0, 1, 0], +np.pi / 2)), "prone (rot +90 about y)")
R["side_L"] = signals(make({}, quat=axis_angle_quat([1, 0, 0], -np.pi / 2)), "side, body-left down")
R["side_R"] = signals(make({}, quat=axis_angle_quat([1, 0, 0], +np.pi / 2)), "side, body-right down")

R["seated"] = signals(make({
    "left_hip_y": -1.5, "right_hip_y": -1.5, "left_knee": 0.1, "right_knee": 0.1,
    "left_shoulder_x": -1.2, "right_shoulder_x": 1.2,
}), "seated legs out (on backside)")

R["seated_bent"] = signals(make({
    "left_hip_y": -1.9, "right_hip_y": -1.9, "left_knee": 1.6, "right_knee": 1.6,
}), "seated knees bent (crash sit)")

R["allfours"] = signals(make({
    "left_hip_y": -1.6, "right_hip_y": -1.6, "left_knee": 1.6, "right_knee": 1.6,
    "abdomen_y": 0.0, "left_shoulder_x": -1.57, "right_shoulder_x": 1.57,
}, quat=axis_angle_quat([0, 1, 0], np.pi / 2)), "all fours (pelvis pitched 90)")

R["kneel"] = signals(make({
    "left_knee": 2.6, "right_knee": 2.6, "left_hip_y": -0.15, "right_hip_y": -0.15,
}), "kneeling upright, both knees")

R["halfkneel"] = signals(make({
    "left_knee": 2.6, "left_hip_y": -0.15,
    "right_knee": 1.5, "right_hip_y": -1.4, "right_ankle_y": 0.3,
}), "half kneel: L knee down, R foot planted")

R["squat"] = signals(make({
    "left_hip_y": -1.8, "right_hip_y": -1.8, "left_knee": 2.3, "right_knee": 2.3,
    "left_ankle_y": 0.7, "right_ankle_y": 0.7, "abdomen_y": 0.6,
}), "deep squat")

R["crouch"] = signals(make({
    "left_hip_y": -0.9, "right_hip_y": -0.9, "left_knee": 1.2, "right_knee": 1.2,
    "left_ankle_y": 0.4, "right_ankle_y": 0.4, "abdomen_y": 0.3,
}), "half crouch")

R["fold"] = signals(make({"abdomen_y": 1.4}), "standing but folded at waist 80deg")
R["backfold"] = signals(make({"abdomen_y": -0.9}), "standing but folded BACKWARD 51deg")

# ---------------------------------------------------------------- action band
print("\n\n############ ACTION BAND")
lo = M.actuator_ctrlrange[:, 0]
hi = M.actuator_ctrlrange[:, 1]
nom = P.default_joint_pos
cur = P.action_scale
proposed = np.maximum(nom - lo, hi - nom)
rows = ["knee", "elbow", "shoulder_x", "hip_y", "abdomen_y", "ankle_y", "hip_x"]
print(f"{'joint':<20}{'lo':>7}{'hi':>7}{'nom':>7}{'scale_now':>10}{'band_now':>18}"
      f"{'scale_new':>10}{'band_new':>18}")
for i, n in enumerate(ACT):
    if not any(r in n for r in rows):
        continue
    bn = (max(lo[i], nom[i] - cur[i]), min(hi[i], nom[i] + cur[i]))
    bp = (max(lo[i], nom[i] - proposed[i]), min(hi[i], nom[i] + proposed[i]))
    print(f"{n:<20}{lo[i]:7.2f}{hi[i]:7.2f}{nom[i]:7.2f}{cur[i]:10.2f}"
          f"   [{bn[0]:6.2f},{bn[1]:6.2f}] {proposed[i]:10.2f}   [{bp[0]:6.2f},{bp[1]:6.2f}]")
frac_now = np.mean((np.minimum(hi, nom + cur) - np.maximum(lo, nom - cur)) / (hi - lo))
frac_new = np.mean((np.minimum(hi, nom + proposed) - np.maximum(lo, nom - proposed)) / (hi - lo))
print(f"mean fraction of joint range commandable:  now {frac_now:.3f}   proposed {frac_new:.3f}")
print("proposed action_scale (actuator order):", np.round(proposed, 3).tolist())
