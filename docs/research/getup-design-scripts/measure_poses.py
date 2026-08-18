"""Pose signatures + torque scale + force limits, on this model."""
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path("/Users/BrickLayer/Desktop/HumonoidAI")
sys.path.insert(0, str(ROOT))
from humanoid_rl.envs.model_prep import prepare  # noqa: E402
from humanoid_rl.tasks.base import quat_rotate_inverse  # noqa: E402

p = prepare(ROOT / "humanoid_rl/models/humanoid_scene.xml")
m = p.model
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]

print("jnt_actfrclimited:", np.unique(m.jnt_actfrclimited))
print("jnt_actfrcrange sample:", m.jnt_actfrcrange[1:6].tolist())
print("actuator_forcelimited:", np.unique(m.actuator_forcelimited))

name_to_qadr = {}
for j in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    name_to_qadr[nm] = m.jnt_qposadr[j]


def setj(q, name, val):
    q[name_to_qadr[name]] = val


def report(label, d):
    g = quat_rotate_inverse(d.qpos[3:7][None, :], np.array([[0.0, 0.0, -1.0]]))[0]
    head = d.sensordata[p.head_pos_adr + 2]
    tz = d.sensordata[p.torso_zaxis_adr + 2]
    keys = {n: d.sensordata[a:a + 3].copy() for n, a in zip(p.key_body_names, p.key_body_adr)}
    f = d.sensordata[p.foot_touch_adr]
    print(f"{label:34s} root {d.qpos[2]:.3f} g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f}) "
          f"tz {tz:+.2f} head {head/p.standing_head_height:.2f} "
          f"feetz {keys['left_foot'][2]:.2f}/{keys['right_foot'][2]:.2f} "
          f"handz {keys['left_hand'][2]:.2f}/{keys['right_hand'][2]:.2f} "
          f"F {f[0]:5.0f}/{f[1]:5.0f}")


def run(label, qpos, seconds, ctrl_pose=None):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    d.ctrl[:] = (qpos if ctrl_pose is None else ctrl_pose)[qadr]
    if seconds:
        mujoco.mj_step(m, d, nstep=int(seconds / m.opt.timestep))
    report(label, d)
    return d


base = p.default_qpos.copy()
run("STANDING (holds own pose 0.3s)", base, 0.3)
print("  sum tau^2 standing:", float(np.sum(np.square(mujoco.MjData(m).actuator_force))))

d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
d.qpos[:] = base
d.ctrl[:] = base[qadr]
mujoco.mj_step(m, d, nstep=150)
print("  standing sum tau^2 =", float(np.sum(np.square(d.actuator_force))),
      " mean|tau| =", float(np.abs(d.actuator_force).mean()))

seated = base.copy()
for s in ("left", "right"):
    setj(seated, f"{s}_hip_y", -1.55)
    setj(seated, f"{s}_knee", 0.05)
seated[2] = 0.10
run("SEATED legs out (1.5s)", seated, 1.5)

kneel = base.copy()
for s in ("left", "right"):
    setj(kneel, f"{s}_knee", 2.6)
    setj(kneel, f"{s}_hip_y", 0.0)
    setj(kneel, f"{s}_ankle_y", -0.5)
kneel[2] = 0.55
run("KNEEL (0.05s, before topple)", kneel, 0.05)
run("KNEEL (0.5s)", kneel, 0.5)

squat = base.copy()
for s in ("left", "right"):
    setj(squat, f"{s}_knee", 2.2)
    setj(squat, f"{s}_hip_y", -1.6)
    setj(squat, f"{s}_ankle_y", 0.8)
squat[2] = 0.45
run("DEEP SQUAT (0.05s)", squat, 0.05)

allfours = base.copy()
for s in ("left", "right"):
    setj(allfours, f"{s}_knee", 2.4)
    setj(allfours, f"{s}_hip_y", -1.4)
    setj(allfours, f"{s}_shoulder_x", np.radians(83.0) * (1 if s == "right" else -1))
    setj(allfours, f"{s}_shoulder_y", -1.2 if s == "right" else -1.2)
allfours[3:7] = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]  # pitch +90 -> face down
allfours[2] = 0.45
run("ALL FOURS-ish (1.0s)", allfours, 1.0)

# torque on the first step from a fallen pose, ctrl = standing default (the current engine)
supine = base.copy()
supine[3:7] = [np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0]
supine[2] = 0.20
d = mujoco.MjData(m)
mujoco.mj_resetData(m, d)
d.qpos[:] = supine
mujoco.mj_forward(m, d)
d.ctrl[:] = supine[qadr]
mujoco.mj_step(m, d, nstep=int(1.5 / m.opt.timestep))
settled = d.qpos.copy()
report("SUPINE settled", d)

for label, ctrl in (("ctrl=nominal standing (engine today)", base[qadr]),
                    ("ctrl=own settled pose (proposed fix)", settled[qadr])):
    d2 = mujoco.MjData(m)
    mujoco.mj_resetData(m, d2)
    d2.qpos[:] = settled
    mujoco.mj_forward(m, d2)
    d2.ctrl[:] = ctrl
    mujoco.mj_step(m, d2, nstep=4)  # one control step
    tau = d2.actuator_force
    lim = m.jnt_actfrcrange[m.actuator_trnid[:, 0]]
    sat = np.sum(np.abs(tau) >= 0.99 * np.abs(lim[:, 1]))
    print(f"  reset step1 {label}: mean|tau| {np.abs(tau).mean():6.1f} max {np.abs(tau).max():6.1f} "
          f"sum tau^2 {np.sum(np.square(tau)):.3e} saturated {sat}/{m.nu}")
