"""Part 4: is the getting-up pose space inside the ACTION space, and what does a
fallen initial state cost in immediate PD torque?"""
from __future__ import annotations

import sys

import mujoco
import numpy as np

sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare  # noqa: E402

REPO = "/Users/BrickLayer/Desktop/HumonoidAI"
SCRATCH = ("/private/tmp/claude-502/-Users-BrickLayer-Desktop-HumonoidAI/"
           "2ea9e069-53fd-4656-90b0-68f777666b8d/scratchpad")
prep = prepare(f"{REPO}/humanoid_rl/models/humanoid_scene.xml")
m = prep.model
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
jid = np.array([m.actuator_trnid[i, 0] for i in range(m.nu)])
jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jid]
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
nom = prep.default_joint_pos
scale = prep.action_scale
kp = m.actuator_gainprm[:, 0]
frc = m.jnt_actfrcrange[jid, 1]

cmd_lo = np.maximum(nom - scale, lo)
cmd_hi = np.minimum(nom + scale, hi)
frac = (cmd_hi - cmd_lo) / (hi - lo)

print("COMMANDABLE PD-TARGET BAND (action in [-1,1] around the nominal STANDING pose)")
print(f"{'joint':<18}{'jnt range':>18}{'commandable':>20}{'% of range':>12}")
interesting = ["knee", "hip_y", "hip_x", "abdomen_y", "shoulder_x", "shoulder_y", "elbow",
               "ankle_y"]
for i in range(m.nu):
    if any(k in jname[i] for k in interesting):
        print(f"{jname[i]:<18}[{lo[i]:6.2f},{hi[i]:6.2f}]   "
              f"[{cmd_lo[i]:6.2f},{cmd_hi[i]:6.2f}]      {frac[i] * 100:5.1f}%")
print(f"mean commandable fraction over all 28 joints: {frac.mean() * 100:.1f}%")

# poses a getting-up motion actually needs
NEEDS = {
    "kneel (knee 2.6)": {"right_knee": 2.6, "left_knee": 2.6},
    "deep squat (hip -1.8, knee 2.4)": {"right_hip_y": -1.8, "left_hip_y": -1.8,
                                        "right_knee": 2.4, "left_knee": 2.4},
    "sit legs-out (hip -1.5)": {"right_hip_y": -1.5, "left_hip_y": -1.5},
    "hands under shoulders (shoulder_x 0.0, elbow 1.5)": {"right_shoulder_x": 0.0,
                                                          "left_shoulder_x": 0.0,
                                                          "right_elbow": 1.5,
                                                          "left_elbow": -1.5},
    "forward fold (abdomen_y 1.4)": {"abdomen_y": 1.4},
}
print("\nCan the policy COMMAND the poses a getting-up motion is made of?")
for label, joints in NEEDS.items():
    bad = []
    for jn, v in joints.items():
        i = jname.index(jn)
        if not (cmd_lo[i] - 1e-9 <= v <= cmd_hi[i] + 1e-9):
            need = abs(v - np.clip(v, cmd_lo[i], cmd_hi[i]))
            bad.append(f"{jn} short by {need:.2f} rad ({np.degrees(need):.0f} deg)")
    print(f"  {label:<50} {'REACHABLE' if not bad else 'NOT REACHABLE: ' + '; '.join(bad)}")

# ---------------------------------------------------------- spring-loaded resets
print("\nSPRING-LOADED RESET: torque applied on step 1 if the episode STARTS in a fallen pose")
for name, path in (("policy falls", f"{SCRATCH}/policy_falls.npy"),):
    q = np.load(path)
    j = q[:, qadr]
    out_lo = (j < cmd_lo).sum(axis=1)
    out_hi = (j > cmd_hi).sum(axis=1)
    err = np.clip(nom - j, None, None)          # action = 0 -> target is the nominal pose
    tau = np.clip(kp * err, -frc, frc)
    sat = np.abs(kp * err) > frc
    print(f"  [{name}] n={len(q)}")
    print(f"    joints outside the commandable band at reset: mean {np.mean(out_lo + out_hi):.1f} of 28, "
          f"max {np.max(out_lo + out_hi)}")
    print(f"    |PD torque| on step 1 with action=0: mean {np.abs(tau).mean():.1f} Nm, "
          f"max {np.abs(tau).max():.1f} Nm")
    print(f"    joints SATURATING their torque limit on step 1: mean {sat.sum(axis=1).mean():.1f} of 28, "
          f"max {sat.sum(axis=1).max()}")
    worst = np.argsort(-np.abs(tau).mean(axis=0))[:6]
    print("    worst joints: " + ", ".join(
        f"{jname[i]} {np.abs(tau)[:, i].mean():.0f}Nm" for i in worst))
