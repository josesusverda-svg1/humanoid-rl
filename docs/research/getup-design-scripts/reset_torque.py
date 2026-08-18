"""Torque on the first control step after a fallen reset: nominal target vs own pose."""
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path("/Users/BrickLayer/Desktop/HumonoidAI")
sys.path.insert(0, str(ROOT))
from humanoid_rl.envs.model_prep import prepare  # noqa: E402

rng = np.random.default_rng(1)
p = prepare(ROOT / "humanoid_rl/models/humanoid_scene.xml")
m = p.model
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
lim = np.abs(m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1])

poses = []
while len(poses) < 32:
    q = p.default_qpos.copy()
    u = rng.normal(size=4)
    q[3:7] = u / np.linalg.norm(u)
    for i in range(m.nu):
        q[qadr[i]] = np.clip(p.default_joint_pos[i] + rng.normal(0, 0.6) * p.action_scale[i], lo[i], hi[i])
    q[2] = rng.uniform(0.35, 0.8)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q
    d.qvel[:3] = rng.normal(0, 0.5, 3)
    d.qvel[3:6] = rng.normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    d.ctrl[:] = q[qadr]
    mujoco.mj_step(m, d, nstep=int(2.5 / m.opt.timestep))
    if np.isfinite(d.qpos).all():
        poses.append(d.qpos.copy())

for label, use_own in (("ctrl = nominal standing (engine today)", False),
                       ("ctrl = the reset pose's own angles", True)):
    means, maxs, sats, sq = [], [], [], []
    for q in poses:
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        d.ctrl[:] = q[qadr] if use_own else p.default_joint_pos
        mujoco.mj_step(m, d, nstep=4)
        tau = np.abs(d.actuator_force)
        means.append(tau.mean())
        maxs.append(tau.max())
        sats.append(int((tau >= 0.99 * lim).sum()))
        sq.append(float(np.sum(np.square(tau))))
    print(f"{label:40s} mean|tau| {np.mean(means):6.1f}  max {np.max(maxs):6.1f}  "
          f"saturated {np.mean(sats):5.1f}/{m.nu}  sum tau^2 {np.mean(sq):.3e}")

# joint-angle deviation from nominal in these poses, in units of the PD saturation error
dev = np.array([np.abs(q[qadr] - p.default_joint_pos) for q in poses])
kp = m.actuator_gainprm[:, 0]
print("mean |q - nominal| (rad):", float(dev.mean()), " saturation error lim/kp (rad) mean:",
      float(np.mean(lim / kp)))
