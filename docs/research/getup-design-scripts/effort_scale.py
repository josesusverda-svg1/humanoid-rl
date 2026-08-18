"""Bounded effort term: mean_i clip(tau_i/tau_lim_i, -1, 1)^2, measured."""
import sys
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path("/Users/BrickLayer/Desktop/HumonoidAI")
sys.path.insert(0, str(ROOT))
from humanoid_rl.envs.model_prep import prepare

rng = np.random.default_rng(3)
p = prepare(ROOT / "humanoid_rl/models/humanoid_scene.xml")
m = p.model
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
tlim = np.abs(m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1])
print("torque limits per actuator:", sorted(set(tlim.tolist())))

def effort(tau):
    return float(np.mean(np.clip(tau / tlim, -1, 1) ** 2))

# standing
d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = p.default_qpos
mujoco.mj_forward(m, d); d.ctrl[:] = p.default_joint_pos
mujoco.mj_step(m, d, nstep=150)
print("standing effort:", effort(d.actuator_force), " applied |qfrc_actuator| max:",
      float(np.abs(d.qfrc_actuator[6:]).max()))

poses = []
while len(poses) < 12:
    q = p.default_qpos.copy()
    u = rng.normal(size=4); q[3:7] = u/np.linalg.norm(u)
    for i in range(m.nu):
        q[qadr[i]] = np.clip(p.default_joint_pos[i] + rng.normal(0, 0.6)*p.action_scale[i], lo[i], hi[i])
    q[2] = rng.uniform(0.35, 0.8)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[:] = q
    mujoco.mj_forward(m, d); d.ctrl[:] = q[qadr]; mujoco.mj_step(m, d, nstep=1250)
    if np.isfinite(d.qpos).all():
        poses.append((d.qpos.copy(), d.qvel.copy()))

vals = []
for q0, v0 in poses:
    for _ in range(6):
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
        d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d); d.ctrl[:] = q0[qadr]
        for k in range(400):
            if k % 40 == 0:
                d.ctrl[:] = rng.uniform(lo, hi)
            mujoco.mj_step(m, d, nstep=4)
            vals.append(effort(d.actuator_force))
vals = np.array(vals)
print("vigorous effort: median %.3f p90 %.3f p99 %.3f max %.3f" %
      (np.median(vals), np.percentile(vals, 90), np.percentile(vals, 99), vals.max()))
for w in (0.5, 0.25, 0.1):
    print(f"  at w=-{w}: median {-w*np.median(vals):.3f} p90 {-w*np.percentile(vals,90):.3f} worst {-w:.3f}/step")
