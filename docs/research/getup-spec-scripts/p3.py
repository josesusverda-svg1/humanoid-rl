"""Probe 3: threshold calibration, ballistic arrival coast, airborne touch-sensor search."""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare, _geom_lowest_z

P = prepare("/Users/BrickLayer/Desktop/HumonoidAI/humanoid_rl/models/humanoid_scene.xml")
m = P.model; H = P.standing_height; DT = 4*m.opt.timestep
BW = float(m.body_mass.sum())*9.81
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
QA = {nm: qadr[i] for i, nm in enumerate(P.joint_names)}
tz, hp, touch, key = P.torso_zaxis_adr, P.head_pos_adr, P.foot_touch_adr, P.key_body_adr
floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
np.set_printoptions(precision=3, suppress=True, linewidth=200)


def ground(q):
    q = q.copy(); d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
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
    kb = s[key[0]:key[0]+12].reshape(4, 3); ff = s[touch]
    heading = np.arctan2(2*(q[4]*q[5]+q[3]*q[6]), 1-2*(q[5]**2+q[6]**2))
    rel = kb[0, :2]-kb[1, :2]; c, sn = np.cos(-heading), np.sin(-heading)
    tzv = s[tz:tz+3]
    return dict(h=q[2], hr=s[hp+2]/P.standing_head_height, g=g, tu=s[tz+2], F=ff.copy(),
                fz=kb[:2, 2].copy(), hz=kb[2:, 2].copy(),
                lat=sn*rel[0]+c*rel[1], planar=np.linalg.norm(rel),
                fore=c*tzv[0]-sn*tzv[1], side=sn*tzv[0]+c*tzv[1], tzv=tzv.copy(),
                knee=np.array([q[QA['right_knee']], q[QA['left_knee']]]),
                v=np.linalg.norm(d.qvel[0:3]), w=np.linalg.norm(d.qvel[3:6]))


print("=== kinematic: root height / head ratio / hand height vs knee flexion (grounded) ===")
for kn in (0.209, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1):
    q = base(right_knee=kn, left_knee=kn, right_hip_y=-0.105-kn*0.9, left_hip_y=-0.105-kn*0.9,
             right_ankle_y=-0.105-kn*0.45, left_ankle_y=-0.105-kn*0.45)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    f = feats(d)
    print(f"knee={kn:.3f}: h={f['h']:.3f}({f['h']/H:.3f}H) hr={f['hr']:.3f} hz={f['hz'].min():.3f}")

print("\n=== hand height, standing vs propping ===")
for lbl, q in [("standing nominal", P.default_qpos.copy()),
               ("arms overhead", base(right_shoulder_x=2.4, left_shoulder_x=-2.4)),
               ("arms down/back", base(right_shoulder_x=0.0, left_shoulder_x=0.0)),
               ("reaching floor", base(abdomen_y=1.5, right_shoulder_x=0.3, left_shoulder_x=-0.3,
                                       right_elbow=0.0, left_elbow=0.0))]:
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = ground(q); mujoco.mj_forward(m, d)
    f = feats(d)
    print(f"{lbl:18s} h={f['h']:.3f} hr={f['hr']:.3f} hz_min={f['hz'].min():.3f} tu={f['tu']:+.3f} fore={f['fore']:+.3f}")

print("\n=== separation vs hip_x (grounded, kinematic) ===")
for hx in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
    q = base(right_hip_x=-hx, left_hip_x=+hx, right_ankle_x=+hx, left_ankle_x=-hx)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    f = feats(d)
    print(f"hip_x={hx:.2f}: lat={f['lat']:+.3f} planar={f['planar']:.3f} h={f['h']/H:.3f}H hr={f['hr']:.3f}")
print("--- sagittal split ---")
for hy in (0.0, 0.1, 0.2, 0.3, 0.4):
    q = base(right_hip_y=-0.105-hy, left_hip_y=-0.105+hy)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    f = feats(d)
    print(f"hip_y=+-{hy:.2f}: lat={f['lat']:+.3f} planar={f['planar']:.3f} h={f['h']/H:.3f}H hr={f['hr']:.3f}")

print("\n=== E23-style waist fold: signed fore component vs torso_upright ===")
for ay in (-0.8, -0.6, -0.5, -0.4, -0.3, -0.2, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
    q = base(abdomen_y=ay)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    f = feats(d)
    tilt = np.degrees(np.arccos(np.clip(f['tu'], -1, 1)))
    print(f"abd_y={ay:+.2f}: tu={f['tu']:+.4f} ({tilt:5.1f}deg) fore={f['fore']:+.4f} "
          f"hr={f['hr']:.4f} h={f['h']/H:.3f}H")

print("\n=== PASSIVE COAST after a BALLISTIC arrival (a=0 held at nominal) ===")


def strict(f):
    return (f['h'] >= 0.85*H and f['hr'] >= 0.90 and f['g'][2] <= -0.93
            and abs(f['fore']-(-0.102)) <= 0.25 and abs(f['side']) <= 0.25
            and f['F'].sum() >= 0.7*BW and f['F'].min() >= 0.2*BW
            and f['fz'].max() <= 0.10 and f['hz'].min() >= 0.45
            and f['knee'].max() <= 0.60 and 0.08 <= f['planar'] <= 0.35
            and f['v'] <= 0.4 and f['w'] <= 1.5)


def coast(q0, qvel0, steps=900):
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    d.qpos[:] = q0; d.qvel[:] = qvel0; mujoco.mj_forward(m, d)
    beta = 1.0-np.exp(-2*np.pi*8.0*DT); filt = P.default_qpos[qadr].copy()
    tgt = P.default_qpos[qadr].copy(); best = run = 0
    for t in range(steps):
        filt += beta*(tgt-filt); d.ctrl[:] = filt
        mujoco.mj_step(m, d, nstep=4)
        run = run+1 if strict(feats(d)) else 0
        best = max(best, run)
    return best


rng = np.random.default_rng(1)
q0 = P.default_qpos.copy()
print(f"perfect arrival, v=0        : strict run {coast(q0, np.zeros(m.nv))} steps")
for vz in (-0.5, -1.0, -2.0, -3.0):
    v = np.zeros(m.nv); v[2] = vz
    print(f"arrival v_z={vz:+.1f}          : strict run {coast(q0, v)} steps")
for vx in (0.2, 0.4, 0.6, 1.0):
    v = np.zeros(m.nv); v[0] = vx
    print(f"arrival v_x={vx:+.1f}          : strict run {coast(q0, v)} steps")
for wr in (0.5, 1.0, 2.0):
    v = np.zeros(m.nv); v[4] = wr
    print(f"arrival pitch rate {wr:.1f}    : strict run {coast(q0, v)} steps")
for jn in (0.05, 0.10, 0.20):
    runs = []
    for k in range(4):
        q = P.default_qpos.copy(); q[7:] += rng.normal(0, jn, m.nq-7)
        runs.append(coast(ground(q), np.zeros(m.nv)))
    print(f"arrival joint noise sd {jn:.2f}: strict runs {runs}")

print("\n=== AIRBORNE SELF-CONTACT SEARCH: can touch sensors fire with no floor? ===")
best = 0.0
rng = np.random.default_rng(3)
lo = m.jnt_range[m.actuator_trnid[:, 0], 0]; hi = m.jnt_range[m.actuator_trnid[:, 0], 1]
for k in range(4000):
    q = P.default_qpos.copy()
    q[qadr] = rng.uniform(lo, hi)
    q[2] = 2.0
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
    f = d.sensordata[touch].max()
    best = max(best, f)
print(f"4000 random airborne full-range poses: max foot touch force = {best:.4f} N "
      f"(threshold {0.02*float(m.body_mass.sum())*9.81:.2f} N)")

print("\n=== torque to HOLD the nominal stand (a=0, 0.5 s) ===")
d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = P.default_qpos; mujoco.mj_forward(m, d)
d.ctrl[:] = P.default_qpos[qadr]
mx = 0.0; peaks = np.zeros(m.nu)
for t in range(62):
    mujoco.mj_step(m, d, nstep=4)
    peaks = np.maximum(peaks, np.abs(d.actuator_force))
print(f"peak |actuator_force| holding the stand: max {peaks.max():.1f} N*m, mean {peaks.mean():.1f}")
lim = m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1]
print(f"peak/limit ratio max = {(peaks/lim).max():.3f}")
