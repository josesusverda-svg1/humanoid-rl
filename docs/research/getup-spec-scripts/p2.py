"""Probe 2: proper crouch/kneel heights, passive stand duration, foot-clack, reset torque."""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare, _geom_lowest_z

P = prepare("/Users/BrickLayer/Desktop/HumonoidAI/humanoid_rl/models/humanoid_scene.xml")
m = P.model
H = P.standing_height
np.set_printoptions(precision=3, suppress=True, linewidth=200)
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
QA = {nm: qadr[i] for i, nm in enumerate(P.joint_names)}
AI = {nm: i for i, nm in enumerate(P.joint_names)}
tz, hp, touch, key = P.torso_zaxis_adr, P.head_pos_adr, P.foot_touch_adr, P.key_body_adr
floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
DT = 4 * m.opt.timestep
BW = float(m.body_mass.sum()) * 9.81


def ground(q):
    """Place qpos so the lowest geom rests on z=0 (+2 mm)."""
    q = q.copy()
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    q[2] = 1.0; d.qpos[:] = q; mujoco.mj_forward(m, d)
    low = min(_geom_lowest_z(m, d, g) for g in range(m.ngeom) if g != floor_id)
    q[2] = 1.0 - low + 0.002
    return q


def base(**j):
    q = P.default_qpos.copy()
    for k, v in j.items():
        q[QA[k]] = v
    return ground(q)


def feats(d):
    q = d.qpos; s = d.sensordata
    w = q[3]; u = q[4:7]; down = np.array([0.0, 0.0, -1.0])
    g = down*(2*w*w-1) - 2*w*np.cross(u, down) + 2*u*np.dot(u, down)
    kb = s[key[0]:key[0]+12].reshape(4, 3)
    ff = s[touch]
    heading = np.arctan2(2*(q[4]*q[5]+q[3]*q[6]), 1-2*(q[5]**2+q[6]**2))
    rel = kb[0, :2] - kb[1, :2]
    c, sn = np.cos(-heading), np.sin(-heading)
    lat = sn*rel[0] + c*rel[1]           # signed lateral (left minus right)
    planar = np.linalg.norm(rel)
    tzv = s[tz:tz+3]
    fore = c*tzv[0] - sn*tzv[1]          # torso z-axis fore component in heading frame
    return dict(h=q[2], hr=s[hp+2]/P.standing_head_height, g=g, tu=s[tz+2],
                F=ff.copy(), fz=kb[:2, 2].copy(), hz=kb[2:, 2].copy(),
                lat=lat, planar=planar, fore=fore, tzv=tzv.copy(),
                knee=np.array([q[QA['right_knee']], q[QA['left_knee']]]),
                vroot=np.linalg.norm(d.qvel[0:3]), wroot=np.linalg.norm(d.qvel[3:6]))


def show(f, lbl):
    print(f"{lbl:28s} h={f['h']:.3f}({f['h']/H:.2f}H) hr={f['hr']:.3f} "
          f"gz={f['g'][2]:+.2f} gx={f['g'][0]:+.2f} tu={f['tu']:+.3f} fore={f['fore']:+.3f} "
          f"F=({f['F'][0]:6.1f},{f['F'][1]:6.1f})={f['F'].sum()/BW:.2f}BW "
          f"fz={f['fz'].max():.3f} hz={f['hz'].min():.3f} lat={f['lat']:+.3f} pl={f['planar']:.3f} "
          f"kn={f['knee'].max():.2f}")


def settle(q, ctrl=None, sec=1.5):
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    d.ctrl[:] = q[qadr] if ctrl is None else ctrl
    mujoco.mj_step(m, d, nstep=int(sec/m.opt.timestep))
    return d


print("=== crouch / kneel, placed on the ground then settled 1.2 s ===")
for kn in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6):
    q = base(right_knee=kn, left_knee=kn, right_hip_y=-0.105-kn*0.9, left_hip_y=-0.105-kn*0.9,
             right_ankle_y=-0.105-kn*0.45, left_ankle_y=-0.105-kn*0.45)
    show(feats(settle(q, sec=1.2)), f"crouch knee={kn:.2f}")

for kn in (2.0, 2.4, 2.7):
    q = base(right_knee=kn, left_knee=kn, right_hip_y=-0.4, left_hip_y=-0.4,
             right_ankle_y=-0.9, left_ankle_y=-0.9)
    show(feats(settle(q, sec=1.2)), f"kneel knee={kn:.2f}")

# all fours
q = base(abdomen_y=1.4, right_knee=2.4, left_knee=2.4, right_hip_y=-1.4, left_hip_y=-1.4,
         right_shoulder_x=0.2, left_shoulder_x=-0.2, right_elbow=0.1, left_elbow=-0.1)
show(feats(settle(q, sec=1.5)), "all fours-ish")

print("\n=== PASSIVE STAND DURATION: a=0, plant = 125 Hz + 8 Hz action filter ===")


def naive_pred(f):
    return (f['h'] > 0.80*H and f['hr'] > 0.85 and f['g'][2] < -0.90 and f['tu'] > 0.90
            and (f['F'] > 9.82).all())


def strict_pred(f):
    return (f['h'] >= 0.85*H and f['hr'] >= 0.90 and f['g'][2] <= -0.93
            and abs(f['fore']) <= 0.30 and abs(f['tzv'][1]) <= 0.30
            and f['F'].sum() >= 0.7*BW and f['F'].min() >= 0.2*BW
            and f['fz'].max() <= 0.10 and f['hz'].min() >= 0.45
            and f['knee'].max() <= 0.60 and 0.08 <= f['planar'] <= 0.35
            and f['vroot'] <= 0.4 and f['wroot'] <= 1.5)


def rollout(q0, ctrl_target, steps=1500, filt_hz=8.0, push=None, push_at=None):
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
    beta = 1.0 if filt_hz <= 0 else 1.0 - np.exp(-2*np.pi*filt_hz*DT)
    filt = ctrl_target.copy()
    best_naive = best_strict = 0
    run_n = run_s = 0
    for t in range(steps):
        filt += beta*(ctrl_target - filt)
        d.ctrl[:] = filt
        mujoco.mj_step(m, d, nstep=4)
        if push is not None and push_at is not None and t == push_at:
            d.qvel[0:3] += push
        f = feats(d)
        run_n = run_n+1 if naive_pred(f) else 0
        run_s = run_s+1 if strict_pred(f) else 0
        best_naive = max(best_naive, run_n); best_strict = max(best_strict, run_s)
    return best_naive, best_strict


nom_ctrl = P.default_qpos[qadr].copy()
q_stand = P.default_qpos.copy()
bn, bs = rollout(q_stand, nom_ctrl, steps=1500)
print(f"nominal stand, a=0        : naive run {bn} steps ({bn*DT:.2f}s)  strict run {bs} ({bs*DT:.2f}s)")

for hx in (0.15, 0.3, 0.4, 0.5):
    q = base(right_hip_x=-hx, left_hip_x=+hx, right_ankle_x=+hx, left_ankle_x=-hx)
    bn, bs = rollout(q, q[qadr], steps=1500)
    print(f"splay hip_x={hx:.2f}, a=const : naive run {bn} ({bn*DT:.2f}s)  strict run {bs} ({bs*DT:.2f}s)")

for hy in (0.3, 0.5, 0.7):
    q = base(right_hip_y=-0.105-hy, left_hip_y=-0.105+hy)
    bn, bs = rollout(q, q[qadr], steps=1500)
    print(f"split hip_y={hy:.2f}, a=const : naive run {bn} ({bn*DT:.2f}s)  strict run {bs} ({bs*DT:.2f}s)")

print("\n--- with one scripted lateral shove during the hold ---")
for mag in (0.2, 0.3, 0.4, 0.5, 0.7):
    bn, bs = rollout(q_stand, nom_ctrl, steps=1500, push=np.array([0.0, mag, 0.0]), push_at=60)
    q = base(right_hip_x=-0.4, left_hip_x=+0.4, right_ankle_x=+0.4, left_ankle_x=-0.4)
    bn2, bs2 = rollout(q, q[qadr], steps=1500, push=np.array([0.0, mag, 0.0]), push_at=60)
    print(f"push {mag:.2f} m/s: nominal naive {bn:4d} strict {bs:4d} | splay0.4 naive {bn2:4d} strict {bs2:4d}")

print("\n=== FOOT-CLACK: airborne, feet touching each other, do touch sensors fire? ===")
for hz_ in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0):
    q = P.default_qpos.copy(); q[2] = 1.6
    q[QA['right_hip_z']] = +hz_; q[QA['left_hip_z']] = -hz_
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    s = d.sensordata
    kb = s[key[0]:key[0]+12].reshape(4, 3)
    print(f"airborne hip_z=+-{hz_:.2f}: F=({s[touch[0]]:8.1f},{s[touch[1]]:8.1f})  "
          f"foot_z=({kb[0,2]:.2f},{kb[1,2]:.2f})  contact={(s[touch]>9.82)}")
# knees together / crossed legs on the ground
for hx_ in (0.0, -0.3, -0.5):
    q = base(right_hip_x=hx_, left_hip_x=-hx_)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    s = d.sensordata
    print(f"grounded crossed hip_x={hx_:+.2f}: F=({s[touch[0]]:7.1f},{s[touch[1]]:7.1f}) sum={s[touch].sum()/BW:.2f}BW")

print("\n=== RESET SPRING-LOAD: first-step torque with ctrl=nominal vs ctrl=own pose ===")
rng = np.random.default_rng(0)
poses = []
for _ in range(24):
    q = P.default_qpos.copy()
    q[7:] += rng.normal(0, 0.6, m.nq-7)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = rng.uniform(0, np.pi)
    q[3:7] = [np.cos(ang/2), *(np.sin(ang/2)*ax)]
    q[2] = 0.6
    d = settle(q, ctrl=q[qadr], sec=2.5)
    poses.append(d.qpos.copy())
for lbl, ctrl_fn in [("ctrl=nominal (current)", lambda q: P.default_qpos[qadr]),
                     ("ctrl=own joints (fix) ", lambda q: q[qadr])]:
    mx, mn, sat = [], [], []
    for q in poses:
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
        d.ctrl[:] = ctrl_fn(q)
        mujoco.mj_step(m, d, nstep=4)
        tau = np.abs(d.actuator_force)
        mx.append(tau.max()); mn.append(tau.mean())
        lim = m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1]
        sat.append((tau >= lim*0.999).sum())
    print(f"{lbl}: mean|tau| {np.mean(mn):8.2f}  max|tau| {np.max(mx):9.2f}  "
          f"saturated joints/step {np.mean(sat):.1f} of {m.nu}")
print("jnt_actfrcrange (upper) per actuator:",
      m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1])
