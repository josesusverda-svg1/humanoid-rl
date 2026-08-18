"""Hackability floor: can a dumb controller satisfy the proposed predicates from fallen states?

Builds a small drop-and-settle bank, then runs:
  - zero action (hold nominal standing target, i.e. the engine's reset behaviour)
  - single-joint flicks (each of 28 joints driven to each limit)
  - K random CONSTANT PD targets over the FULL joint range (the strongest dumb oracle)
and scores three predicates:
  naive   : root_h > 0.62*H and torso_upright > 0.8          (what a careless spec would use)
  strict  : the full conjunctive stand predicate proposed
  held    : strict satisfied continuously for HOLD seconds
"""
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path("/Users/BrickLayer/Desktop/HumonoidAI")
sys.path.insert(0, str(ROOT))
from humanoid_rl.envs.model_prep import prepare  # noqa: E402
from humanoid_rl.tasks.base import quat_rotate_inverse  # noqa: E402

rng = np.random.default_rng(0)
p = prepare(ROOT / "humanoid_rl/models/humanoid_scene.xml")
m = p.model
H = p.standing_height
HEAD = p.standing_head_height
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
lo = m.actuator_ctrlrange[:, 0].copy()
hi = m.actuator_ctrlrange[:, 1].copy()
CTRL_DT = 4 * m.opt.timestep
HOLD_S = 1.5
FTHRESH = 0.02 * m.body_mass.sum() * 9.81


def signals(d):
    q = d.qpos
    g = quat_rotate_inverse(q[3:7][None, :], np.array([[0.0, 0.0, -1.0]]))[0]
    head = d.sensordata[p.head_pos_adr + 2] / HEAD
    tz = d.sensordata[p.torso_zaxis_adr + 2]
    kb = np.array([d.sensordata[a:a + 3] for a in p.key_body_adr])
    f = d.sensordata[p.foot_touch_adr]
    return q[2], g, head, tz, kb, f, d.qvel[:3], d.qvel[3:6]


def naive_ok(s):
    root, g, head, tz, kb, f, v, w = s
    return root > 0.62 * H and tz > 0.8


def strict_ok(s):
    root, g, head, tz, kb, f, v, w = s
    feet_z = kb[:2, 2]
    hands_z = kb[2:, 2]
    stance = np.linalg.norm(kb[0, :2] - kb[1, :2])
    return bool(
        root > 0.80 * H
        and head > 0.85
        and -g[2] > 0.90
        and tz > 0.90
        and (f > FTHRESH).all()
        and feet_z.max() < 0.12
        and hands_z.min() > 0.35
        and 0.05 < stance < 0.50
        and np.linalg.norm(v) < 0.5
        and np.linalg.norm(w) < 1.5
    )


def make_bank(n):
    """Drop from height with randomised orientation and joints; PD holds the spawn pose."""
    poses = []
    while len(poses) < n:
        q = p.default_qpos.copy()
        u = rng.normal(size=4)
        q[3:7] = u / np.linalg.norm(u)
        for i in range(m.nu):
            a = p.default_joint_pos[i] + rng.normal(0, 0.6) * p.action_scale[i]
            q[qadr[i]] = np.clip(a, lo[i], hi[i])
        q[2] = rng.uniform(0.35, 0.8)
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q
        d.qvel[:3] = rng.normal(0, 0.5, 3)
        d.qvel[3:6] = rng.normal(0, 1.5, 3)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = q[qadr]
        mujoco.mj_step(m, d, nstep=int(2.5 / m.opt.timestep))
        if not np.isfinite(d.qpos).all():
            continue
        poses.append((d.qpos.copy(), d.qvel.copy()))
    return poses


def rollout(pose, ctrl, seconds=4.0):
    q0, v0 = pose
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q0
    d.qvel[:] = v0
    mujoco.mj_forward(m, d)
    d.ctrl[:] = ctrl
    n = int(seconds / CTRL_DT)
    naive = strict = 0
    run = 0
    best_run = 0
    peak = 0.0
    for _ in range(n):
        mujoco.mj_step(m, d, nstep=4)
        s = signals(d)
        peak = max(peak, s[0])
        if naive_ok(s):
            naive = 1
        if strict_ok(s):
            strict = 1
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return naive, strict, best_run * CTRL_DT >= HOLD_S, peak


bank = make_bank(32)
labels = []
for q, v in bank:
    g = quat_rotate_inverse(q[3:7][None, :], np.array([[0.0, 0.0, -1.0]]))[0]
    lab = ("prone" if g[0] > 0.7 else "supine" if g[0] < -0.7 else
           "side" if abs(g[1]) > 0.7 else "sit/other")
    labels.append(lab)
print("bank labels:", {l: labels.count(l) for l in set(labels)})
print("bank root heights: mean %.3f min %.3f max %.3f" %
      (np.mean([q[2] for q, _ in bank]), min(q[2] for q, _ in bank), max(q[2] for q, _ in bank)))

trials = {"zero_action(nominal target)": [p.default_joint_pos.copy()]}
flicks = []
for i in range(m.nu):
    for lim in (lo[i], hi[i]):
        c = p.default_joint_pos.copy()
        c[i] = lim
        flicks.append(c)
trials["single_joint_flick(56)"] = flicks
rand = [rng.uniform(lo, hi) for _ in range(20)]
trials["random_constant_fullrange(20)"] = rand

for name, ctrls in trials.items():
    nv = st = hd = 0
    peak = 0.0
    total = 0
    for pose in bank:
        for c in ctrls:
            a, b, h, pk = rollout(pose, c)
            nv += a
            st += b
            hd += h
            peak = max(peak, pk)
            total += 1
    print(f"{name:32s} trials {total:5d}  naive-momentary {100*nv/total:5.1f}%  "
          f"strict-momentary {100*st/total:5.1f}%  held{HOLD_S}s {100*hd/total:5.1f}%  peak root {peak:.3f}")

# best-of-K oracle per pose (the number that matters: an oracle picking the best target)
for name, ctrls in trials.items():
    nv = st = hd = 0
    for pose in bank:
        a = b = h = 0
        for c in ctrls:
            x, y, z, _ = rollout(pose, c)
            a, b, h = max(a, x), max(b, y), max(h, z)
        nv += a
        st += b
        hd += h
    print(f"BEST-OF per pose {name:28s} naive {100*nv/len(bank):5.1f}%  "
          f"strict {100*st/len(bank):5.1f}%  held {100*hd/len(bank):5.1f}%")
