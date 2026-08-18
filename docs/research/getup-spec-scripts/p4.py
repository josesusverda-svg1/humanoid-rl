"""Probe 4: passive coast across the randomised model pool; action-scale coverage."""
import sys, numpy as np, mujoco
sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare
from humanoid_rl.envs.domain_rand import DomainRandConfig, build_model_pool

P = prepare("/Users/BrickLayer/Desktop/HumonoidAI/humanoid_rl/models/humanoid_scene.xml")
m = P.model; H = P.standing_height; DT = 4*m.opt.timestep
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
tz, hp, touch, key = P.torso_zaxis_adr, P.head_pos_adr, P.foot_touch_adr, P.key_body_adr
cfg = DomainRandConfig()
print("DomainRandConfig:", cfg)
rng = np.random.default_rng(0)
pool = build_model_pool(m, cfg, rng, P.floor_geom_id)
print(f"pool size = {len(pool)}")


def feats(d, mm):
    q = d.qpos; s = d.sensordata
    w = q[3]; u = q[4:7]; down = np.array([0.0, 0.0, -1.0])
    g = down*(2*w*w-1) - 2*w*np.cross(u, down) + 2*u*np.dot(u, down)
    kb = s[key[0]:key[0]+12].reshape(4, 3); ff = s[touch]
    heading = np.arctan2(2*(q[4]*q[5]+q[3]*q[6]), 1-2*(q[5]**2+q[6]**2))
    rel = kb[0, :2]-kb[1, :2]; c, sn = np.cos(-heading), np.sin(-heading)
    tzv = s[tz:tz+3]
    bw = float(mm.body_mass.sum())*9.81
    return dict(h=q[2], hr=s[hp+2]/P.standing_head_height, gz=g[2], tu=s[tz+2],
                Fs=ff.sum()/bw, Fm=ff.min()/bw, fzmax=kb[:2, 2].max(), hzmin=kb[2:, 2].min(),
                planar=np.linalg.norm(rel), fore=c*tzv[0]-sn*tzv[1], side=sn*tzv[0]+c*tzv[1],
                knee=max(q[qadr[17]], q[qadr[24]]),
                v=np.linalg.norm(d.qvel[0:3]), w=np.linalg.norm(d.qvel[3:6]))


BIAS = -0.1023


def strict(f):
    return (f['h'] >= 0.85*H and f['hr'] >= 0.90 and f['gz'] <= -0.93 and f['tu'] >= 0.85
            and abs(f['fore']-BIAS) <= 0.35 and abs(f['side']) <= 0.35
            and f['Fs'] >= 0.70 and f['Fm'] >= 0.20
            and f['fzmax'] <= 0.10 and f['hzmin'] >= 0.40
            and f['knee'] <= 0.60 and 0.08 <= f['planar'] <= 0.35
            and f['v'] <= 0.40 and f['w'] <= 1.5)


def coast(mm, steps=900, push=None, push_at=None, seed=0):
    d = mujoco.MjData(mm); mujoco.mj_resetData(mm, d)
    d.qpos[:] = P.default_qpos; mujoco.mj_forward(mm, d)
    beta = 1.0-np.exp(-2*np.pi*8.0*DT); tgt = P.default_qpos[qadr].copy(); filt = tgt.copy()
    best = run = 0
    for t in range(steps):
        filt += beta*(tgt-filt); d.ctrl[:] = filt
        mujoco.mj_step(mm, d, nstep=4)
        if push is not None and t == push_at:
            d.qvel[0:3] += push
        run = run+1 if strict(feats(d, mm)) else 0
        best = max(best, run)
    return best


runs = np.array([coast(pool[i]) for i in range(len(pool))])
print(f"\nPASSIVE COAST over the model pool (a=0, no push), strict predicate:")
print(f"  n={len(runs)}  min={runs.min()} p50={int(np.median(runs))} p95={int(np.percentile(runs,95))} max={runs.max()}")
print(f"  seconds: min={runs.min()*DT:.2f} p50={np.median(runs)*DT:.2f} p95={np.percentile(runs,95)*DT:.2f} max={runs.max()*DT:.2f}")
print(f"  frac >= 250 steps (2.0s): {(runs>=250).mean():.2f}")
print(f"  frac >= 375 steps (3.0s): {(runs>=375).mean():.2f}")
print(f"  frac >= 500 steps (4.0s): {(runs>=500).mean():.2f}")

rng2 = np.random.default_rng(7)
for mag in (0.3, 0.5, 0.7):
    r = []
    for i in range(0, len(pool), 2):
        ang = rng2.uniform(0, 2*np.pi)
        r.append(coast(pool[i], push=np.array([mag*np.cos(ang), mag*np.sin(ang), 0.0]), push_at=90))
    r = np.array(r)
    print(f"  with one {mag:.1f} m/s push at step 90: p50={int(np.median(r))} p95={int(np.percentile(r,95))} "
          f"frac>=250 {(r>=250).mean():.2f} frac>=375 {(r>=375).mean():.2f}")

print("\n=== ACTION SCALE COVERAGE ===")
lo = m.jnt_range[m.actuator_trnid[:, 0], 0]; hi = m.jnt_range[m.actuator_trnid[:, 0], 1]
nom = P.default_joint_pos
cur = P.action_scale
full = np.maximum(nom-lo, hi-nom)
cov_cur = (np.minimum(nom+cur, hi)-np.maximum(nom-cur, lo))/(hi-lo)
cov_full = (np.minimum(nom+full, hi)-np.maximum(nom-full, lo))/(hi-lo)
for i, nm in enumerate(P.joint_names):
    if nm in ("right_knee", "left_knee", "right_hip_y", "abdomen_y", "right_elbow", "right_ankle_y", "right_hip_x"):
        print(f"{nm:18s} range[{lo[i]:+.2f},{hi[i]:+.2f}] nom={nom[i]:+.3f} "
              f"scale0.6={cur[i]:.3f} band=[{max(nom[i]-cur[i],lo[i]):+.2f},{min(nom[i]+cur[i],hi[i]):+.2f}] cov={cov_cur[i]:.1%} "
              f"| full={full[i]:.3f} cov={cov_full[i]:.1%}")
print(f"mean coverage: fraction=0.6 -> {cov_cur.mean():.1%};  full_range -> {cov_full.mean():.1%}")
print(f"full-range scale keeps a=0 at nominal: yes by construction; max |offset| = {full.max():.3f} rad")
