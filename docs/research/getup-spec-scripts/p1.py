"""Probe 1: model constants, index maps, canonical pose table."""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare

P = prepare("/Users/BrickLayer/Desktop/HumonoidAI/humanoid_rl/models/humanoid_scene.xml")
m = P.model
np.set_printoptions(precision=3, suppress=True, linewidth=200)

mass = float(m.body_mass.sum())
print(f"nq={m.nq} nv={m.nv} nu={m.nu}")
print(f"standing_height = {P.standing_height:.4f}")
print(f"standing_head   = {P.standing_head_height:.4f}")
print(f"mass = {mass:.3f} kg   weight = {mass*9.81:.2f} N")
print(f"timestep = {m.opt.timestep}  dt(dec=4) = {4*m.opt.timestep}")
print(f"contact_force_threshold = {0.02*mass*9.81:.3f} N")
print(f"key_body_names = {P.key_body_names}")
print(f"actuator_forcelimited = {m.actuator_forcelimited}")
print(f"jnt_actfrclimited any = {bool(m.jnt_actfrclimited.any())}")

# actuator -> qpos address map
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
print("\nact_i name qposadr  7+i  match  range           action_scale")
mismatch = 0
for i in range(m.nu):
    nm = P.joint_names[i]
    jid = m.actuator_trnid[i, 0]
    lo, hi = m.jnt_range[jid]
    ok = (qadr[i] == 7 + i)
    mismatch += (not ok)
    print(f"{i:3d} {nm:20s} {qadr[i]:3d} {7+i:4d} {'OK ' if ok else 'BAD'} "
          f"[{lo:+.3f},{hi:+.3f}]  {P.action_scale[i]:.3f}  nom={P.default_joint_pos[i]:+.3f}")
print(f"qpos-order mismatches: {mismatch}")

QA = {nm: qadr[i] for i, nm in enumerate(P.joint_names)}
AI = {nm: i for i, nm in enumerate(P.joint_names)}

# sensor addresses
tz = P.torso_zaxis_adr
hp = P.head_pos_adr
touch = P.foot_touch_adr
key = P.key_body_adr
print(f"\ntorso_zaxis_adr={tz} head_pos_adr={hp} touch={touch} key={key}")


def settle(qpos, ctrl=None, seconds=1.5, keep_vel=False):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    d.ctrl[:] = qpos[qadr] if ctrl is None else ctrl
    mujoco.mj_step(m, d, nstep=int(seconds / m.opt.timestep))
    return d


def report(d, label):
    q = d.qpos
    quat = q[3:7][None, :]
    w = quat[:, 0:1]; u = quat[:, 1:4]
    down = np.array([[0.0, 0.0, -1.0]])
    g = (down * (2*w*w - 1.0) - 2*w*np.cross(u, down) + 2*u*(u*down).sum(1, keepdims=True))[0]
    s = d.sensordata
    tu = s[tz + 2]
    hr = s[hp + 2] / P.standing_head_height
    kb = s[key[0]:key[0]+12].reshape(4, 3)  # lf, rf, lh, rh
    ff = s[touch]
    lat_rel = kb[0, :2] - kb[1, :2]
    heading = np.arctan2(2*(q[4]*q[5] + q[3]*q[6]), 1 - 2*(q[5]**2 + q[6]**2))
    lat = abs(-np.sin(heading)*lat_rel[0] + np.cos(heading)*lat_rel[1])
    planar = np.linalg.norm(lat_rel)
    knees = [q[QA["right_knee"]], q[QA["left_knee"]]]
    print(f"{label:26s} h={q[2]:.3f}({q[2]/P.standing_height:.2f}H) hr={hr:.3f} "
          f"g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}) tu={tu:+.3f} "
          f"F=({ff[0]:6.1f},{ff[1]:6.1f}) fz=({kb[0,2]:.3f},{kb[1,2]:.3f}) "
          f"hz=({kb[2,2]:.3f},{kb[3,2]:.3f}) lat={lat:.3f} pl={planar:.3f} "
          f"kn=({knees[0]:.2f},{knees[1]:.2f})")
    return dict(h=q[2], hr=hr, g=g, tu=tu, F=ff, kb=kb, lat=lat, planar=planar)


def base(**joints):
    q = P.default_qpos.copy()
    for k, v in joints.items():
        q[QA[k]] = v
    return q


print("\n--- canonical poses (settled 1.5 s, PD holding own angles) ---")
report(settle(P.default_qpos.copy()), "standing nominal")

# supine / prone / side: rotate root
def rot(axis, ang, q=None):
    q = P.default_qpos.copy() if q is None else q.copy()
    ax = np.array(axis, float); ax /= np.linalg.norm(ax)
    quat = np.array([np.cos(ang/2), *(np.sin(ang/2)*ax)])
    q[3:7] = quat
    q[2] = 0.35
    return q

for lbl, ax, ang in [("supine (pitch -90)", (0,1,0), -np.pi/2),
                     ("prone  (pitch +90)", (0,1,0), +np.pi/2),
                     ("side L (roll -90)", (1,0,0), -np.pi/2),
                     ("side R (roll +90)", (1,0,0), +np.pi/2)]:
    report(settle(rot(ax, ang), seconds=2.5), lbl)

# seated legs out: hips flexed 90, knees straight, torso upright
seat = base(right_hip_y=-1.5, left_hip_y=-1.5, right_knee=0.05, left_knee=0.05,
            right_ankle_y=0.0, left_ankle_y=0.0)
seat[2] = 0.30
report(settle(seat, seconds=2.5), "seated legs out")

# kneeling
kneel = base(right_knee=2.4, left_knee=2.4, right_hip_y=-0.2, left_hip_y=-0.2,
             right_ankle_y=-0.7, left_ankle_y=-0.7)
kneel[2] = 0.55
report(settle(kneel, seconds=2.0), "kneeling")

# crouch sweep
for kn in (0.4, 0.6, 0.8, 1.0, 1.15, 1.3):
    q = base(right_knee=kn, left_knee=kn, right_hip_y=-kn*0.85, left_hip_y=-kn*0.85,
             right_ankle_y=-kn*0.5, left_ankle_y=-kn*0.5)
    q[2] = P.standing_height - 0.22*kn
    report(settle(q, seconds=1.2), f"crouch knee={kn:.2f}")

# splay sweep (lateral brace)
for hx in (0.0, 0.15, 0.3, 0.4, 0.5, 0.6):
    q = base(right_hip_x=-hx, left_hip_x=+hx, right_ankle_x=+hx, left_ankle_x=-hx)
    report(settle(q, seconds=1.2), f"splay hip_x={hx:.2f}")

# sagittal split sweep
for hy in (0.3, 0.5, 0.7, 0.9):
    q = base(right_hip_y=-0.06-hy, left_hip_y=-0.06+hy)
    report(settle(q, seconds=1.2), f"split hip_y=+-{hy:.2f}")

# waist fold sweep, forward (+) and backward (-)
for ay in (-0.9, -0.7, -0.55, -0.4, -0.2, 0.0, 0.2, 0.4, 0.55, 0.7, 0.9):
    q = base(abdomen_y=ay)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    s = d.sensordata
    print(f"fold abdomen_y={ay:+.2f}  torso_upright={s[tz+2]:+.4f}  "
          f"tz=({s[tz]:+.3f},{s[tz+1]:+.3f},{s[tz+2]:+.3f})  hr={s[hp+2]/P.standing_head_height:.4f}")
