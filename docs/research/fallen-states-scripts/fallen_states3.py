"""Part 3: hackability of a fallen start, and the cost of a scripted push-to-fall."""
from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare, _geom_lowest_z  # noqa: E402

REPO = "/Users/BrickLayer/Desktop/HumonoidAI"
MODEL = f"{REPO}/humanoid_rl/models/humanoid_scene.xml"
SCRATCH = ("/private/tmp/claude-502/-Users-BrickLayer-Desktop-HumonoidAI/"
           "2ea9e069-53fd-4656-90b0-68f777666b8d/scratchpad")

prep = prepare(MODEL)
m = prep.model
floor = prep.floor_geom_id
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
rng = np.random.default_rng(7)
DT = 0.008  # control step
print(f"standing_height={prep.standing_height:.4f}  standing_head={prep.standing_head_height:.4f}")
print("actuator force limits (Nm):",
      {n: float(m.actuator_forcerange[i, 1]) for i, n in
       enumerate(["abd_x", "abd_y", "abd_z"])}, "... knee",
      float(m.actuator_forcerange[[i for i in range(m.nu) if "knee" in
            (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) or "")][0], 1]))


def rollout(qpos, ctrl_fn, seconds=4.0):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    n = int(seconds / DT)
    peak_h, peak_tu, peak_head = d.qpos[2], -1.0, 0.0
    for k in range(n):
        d.ctrl[:] = np.clip(ctrl_fn(k * DT, d), lo, hi)
        mujoco.mj_step(m, d, nstep=4)
        if not np.isfinite(d.qpos).all():
            return dict(h=np.nan, tu=np.nan, head=np.nan, peak_h=np.nan,
                        peak_tu=np.nan, peak_head=np.nan)
        peak_h = max(peak_h, float(d.qpos[2]))
        peak_tu = max(peak_tu, float(d.sensordata[prep.torso_zaxis_adr + 2]))
        peak_head = max(peak_head, float(d.sensordata[prep.head_pos_adr + 2]
                                         / prep.standing_head_height))
    return dict(h=float(d.qpos[2]), tu=float(d.sensordata[prep.torso_zaxis_adr + 2]),
                head=float(d.sensordata[prep.head_pos_adr + 2] / prep.standing_head_height),
                peak_h=peak_h, peak_tu=peak_tu, peak_head=peak_head)


def summarise(name, rows):
    h = np.array([r["h"] for r in rows]); tu = np.array([r["tu"] for r in rows])
    ph = np.array([r["peak_h"] for r in rows]); ptu = np.array([r["peak_tu"] for r in rows])
    head = np.array([r["head"] for r in rows])
    stood = (h > 0.62 * prep.standing_height) & (tu > 0.8)
    touched = (ph > 0.62 * prep.standing_height) & (ptu > 0.8)
    print(f"  {name:46s} final h {np.nanmean(h):.3f}  tu {np.nanmean(tu):+.2f}  "
          f"head {np.nanmean(head):.2f} | STOOD {stood.mean() * 100:5.1f}%  "
          f"momentarily-up {touched.mean() * 100:5.1f}%  peak h {np.nanmax(ph):.3f}")


# ------------------------------------------------------------------ fallen sets
pol = np.load(f"{SCRATCH}/policy_falls.npy")


def drop_set(n=40, hold="current"):
    d = mujoco.MjData(m)
    out = []
    for _ in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = prep.default_qpos
        d.qpos[7:] += rng.normal(0, 0.3, m.nq - 7)
        np.clip(d.qpos[qadr], lo, hi, out=d.qpos[qadr])
        u = rng.random(3)
        q = np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                      np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                      np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                      np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
        d.qpos[3:7] = np.array([q[3], q[0], q[1], q[2]])
        mujoco.mj_forward(m, d)
        d.qpos[2] += 0.5 - min(_geom_lowest_z(m, d, g) for g in range(m.ngeom) if g != floor)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = prep.default_joint_pos if hold == "default" else d.qpos[qadr]
        mujoco.mj_step(m, d, nstep=1500)
        out.append(d.qpos.copy())
    return np.array(out)


drop = drop_set(40)

print("\n=== HACKABILITY: what does a trivial controller achieve from a fallen state? ===")
nom = prep.default_joint_pos


def c_zero(t, d):        # action == 0 forever: command the nominal standing pose
    return nom


def c_ramp(t, d):        # ramp from the pose it is in to the nominal pose over 1.5 s
    a = min(1.0, t / 1.5)
    return (1 - a) * d.qpos[qadr] + a * nom


def c_knee_flick(t, d):  # "flick one joint": slam both knees straight, hold the rest
    c = d.qpos[qadr].copy()
    for i in range(m.nu):
        n_ = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) or ""
        if "knee" in n_:
            c[i] = 0.0
    return c


def c_abdomen_flick(t, d):  # slam the waist to full flexion, hold the rest
    c = d.qpos[qadr].copy()
    for i in range(m.nu):
        n_ = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) or ""
        if n_ == "abdomen_y":
            c[i] = hi[i]
    return c


def c_random(seed):
    r = np.random.default_rng(seed)
    tgt = lo + r.random(m.nu) * (hi - lo)
    return lambda t, d: tgt


for label, src in (("policy falls", pol), ("dropped", drop)):
    print(f" -- from {label} (n={len(src)}) --")
    summarise("zero action (hold nominal standing pose)",
              [rollout(q, c_zero) for q in src])
    summarise("ramp to nominal pose over 1.5 s",
              [rollout(q, c_ramp) for q in src])
    summarise("flick both knees straight",
              [rollout(q, c_knee_flick) for q in src])
    summarise("flick abdomen_y to its limit",
              [rollout(q, c_abdomen_flick) for q in src])
    summarise("best of 40 random constant poses",
              [max((rollout(q, c_random(s)) for s in range(40)),
                   key=lambda r: r["peak_h"]) for q in src[:8]])

# ------------------------------------------------------------------ push to fall
print("\n=== COST of a scripted push-to-fall from standing ===")
for impulse in (1.5, 3.0, 5.0):
    times, ok = [], 0
    d = mujoco.MjData(m)
    t0 = time.perf_counter()
    N = 40
    for _ in range(N):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = prep.default_qpos
        d.qpos[7:] += rng.normal(0, 0.02, m.nq - 7)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = nom
        ang = rng.uniform(0, 2 * np.pi)
        d.qvel[0] = impulse * np.cos(ang)
        d.qvel[1] = impulse * np.sin(ang)
        d.qvel[5] = rng.normal(0, 2.0)
        t_fall = -1
        for k in range(int(6.0 / DT)):
            mujoco.mj_step(m, d, nstep=4)
            if t_fall < 0 and d.qpos[2] < 0.62 * prep.standing_height:
                t_fall = k * DT
            if t_fall > 0 and (k * DT - t_fall) > 1.5:
                break
        if t_fall > 0:
            ok += 1
            times.append(t_fall)
    wall = time.perf_counter() - t0
    print(f"  root impulse {impulse} m/s: fell {ok}/{N}, "
          f"time-to-floor median {np.median(times) if times else float('nan'):.2f} s "
          f"({np.median(times) / DT if times else 0:.0f} control steps), "
          f"then +1.5 s settle -> {(np.median(times) + 1.5) / DT if times else 0:.0f} steps total; "
          f"{wall / N * 1000:.0f} ms wall per fall single-thread")
