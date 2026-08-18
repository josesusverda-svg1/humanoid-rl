"""Probe 5: dumb-controller floor under the WIDENED (full-range) action band."""
import sys, time, numpy as np, mujoco
sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare

P = prepare("/Users/BrickLayer/Desktop/HumonoidAI/humanoid_rl/models/humanoid_scene.xml")
m = P.model; H = P.standing_height; DT = 4*m.opt.timestep
BW = float(m.body_mass.sum())*9.81
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
tz, hp, touch, key = P.torso_zaxis_adr, P.head_pos_adr, P.foot_touch_adr, P.key_body_adr
lo = m.jnt_range[m.actuator_trnid[:, 0], 0]; hi = m.jnt_range[m.actuator_trnid[:, 0], 1]
nom = P.default_joint_pos.copy()
SCALE_FULL = np.maximum(nom-lo, hi-nom)          # proposed full_range mode
SCALE_06 = P.action_scale.copy()                  # shipped
BIAS = -0.1023
BETA = 1.0-np.exp(-2*np.pi*8.0*DT)


def strict_from(d):
    q = d.qpos; s = d.sensordata
    if q[2] < 0.85*H:
        return False
    w = q[3]; u = q[4:7]; down = np.array([0.0, 0.0, -1.0])
    g = down*(2*w*w-1) - 2*w*np.cross(u, down) + 2*u*np.dot(u, down)
    if g[2] > -0.93:
        return False
    if s[hp+2]/P.standing_head_height < 0.90:
        return False
    ff = s[touch]
    if ff.sum() < 0.70*BW or ff.min() < 0.20*BW:
        return False
    kb = s[key[0]:key[0]+12].reshape(4, 3)
    if kb[:2, 2].max() > 0.10 or kb[2:, 2].min() < 0.40:
        return False
    if max(q[qadr[17]], q[qadr[24]]) > 0.60:
        return False
    heading = np.arctan2(2*(q[4]*q[5]+q[3]*q[6]), 1-2*(q[5]**2+q[6]**2))
    rel = kb[0, :2]-kb[1, :2]; c, sn = np.cos(-heading), np.sin(-heading)
    pl = np.linalg.norm(rel)
    if not (0.08 <= pl <= 0.35):
        return False
    tzv = s[tz:tz+3]
    if s[tz+2] < 0.85:
        return False
    if abs((c*tzv[0]-sn*tzv[1])-BIAS) > 0.35 or abs(sn*tzv[0]+c*tzv[1]) > 0.35:
        return False
    if np.linalg.norm(d.qvel[0:3]) > 0.40 or np.linalg.norm(d.qvel[3:6]) > 1.5:
        return False
    return True


def make_fallen(n, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = P.default_qpos.copy()
        q[qadr] = np.clip(nom + rng.normal(0, 0.5, m.nu), lo, hi)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = rng.uniform(0.6*np.pi, np.pi)
        q[3:7] = [np.cos(ang/2), *(np.sin(ang/2)*ax)]
        q[2] = 0.55
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q; mujoco.mj_forward(m, d)
        d.ctrl[:] = q[qadr]
        mujoco.mj_step(m, d, nstep=int(2.5/m.opt.timestep))
        if np.isfinite(d.qpos).all() and d.qpos[2] < 0.45*H:
            out.append((d.qpos.copy(), d.qvel.copy()*0.0))
    return out


def run(q0, v0, action, scale, steps=500, check_every=2):
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
    tgt = np.clip(nom + action*scale, lo, hi)
    filt = q0[qadr].copy()          # WITH the proposed reset fix
    best = run_ = 0; peak_h = q0[2]
    for t in range(steps):
        filt += BETA*(tgt-filt); d.ctrl[:] = filt
        mujoco.mj_step(m, d, nstep=4)
        if not np.isfinite(d.qpos).all():
            break
        peak_h = max(peak_h, d.qpos[2])
        if t % check_every == 0:
            run_ = run_+2 if strict_from(d) else 0
            best = max(best, run_)
    return best, peak_h


poses = make_fallen(8)
print(f"generated {len(poses)} settled fallen poses; root heights "
      f"{[f'{p[0][2]:.3f}' for p in poses]}")
rng = np.random.default_rng(5)
for label, scale in (("shipped fraction=0.6", SCALE_06), ("proposed full_range", SCALE_FULL)):
    t0 = time.time()
    stats = {"zero": [], "flick": [], "random": []}
    peaks = {"zero": [], "flick": [], "random": []}
    for (q0, v0) in poses:
        b, ph = run(q0, v0, np.zeros(m.nu), scale)
        stats["zero"].append(b); peaks["zero"].append(ph)
        for j in range(m.nu):
            for sgn in (-1.0, 1.0):
                a = np.zeros(m.nu); a[j] = sgn
                b, ph = run(q0, v0, a, scale)
                stats["flick"].append(b); peaks["flick"].append(ph)
        for k in range(120):
            a = rng.uniform(-1, 1, m.nu)
            b, ph = run(q0, v0, a, scale)
            stats["random"].append(b); peaks["random"].append(ph)
    print(f"\n--- {label} ({time.time()-t0:.0f}s) ---")
    for k in stats:
        s = np.array(stats[k]); pk = np.array(peaks[k])
        print(f"  {k:7s} n={len(s):5d}  ever-satisfied {int((s>0).sum())}  "
              f"held>=250 {int((s>=250).sum())}  max run {s.max()} steps  peak root {pk.max():.3f} m "
              f"({pk.max()/H:.2f}H)")
