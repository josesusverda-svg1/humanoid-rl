"""Measurements for the fallen-initial-state research question.

Run from the repo root with the project venv.
"""
from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.envs.model_prep import prepare, _geom_lowest_z  # noqa: E402

MODEL = "/Users/BrickLayer/Desktop/HumonoidAI/humanoid_rl/models/humanoid_scene.xml"

prep = prepare(MODEL)
m = prep.model
print(f"nq={m.nq} nv={m.nv} nu={m.nu} timestep={m.opt.timestep} "
      f"standing_height={prep.standing_height:.4f} "
      f"terminate_height={0.62 * prep.standing_height:.4f} "
      f"mass={m.body_mass.sum():.1f} kg")
floor = prep.floor_geom_id
print(f"floor geom id={floor}, ngeom={m.ngeom}")

# joint limits in actuator order
jnt_lo = m.actuator_ctrlrange[:, 0].copy()
jnt_hi = m.actuator_ctrlrange[:, 1].copy()
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])

rng = np.random.default_rng(0)


def lowest_point(d):
    return min(_geom_lowest_z(m, d, g) for g in range(m.ngeom) if g != floor)


def rand_quat(rng, n=1):
    u = rng.random((n, 3))
    q = np.stack([
        np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1]),
        np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1]),
        np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2]),
        np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2]),
    ], axis=1)
    # scipy order (x,y,z,w) -> mujoco (w,x,y,z)
    return np.concatenate([q[:, 3:4], q[:, 0:3]], axis=1)


def contact_report(d):
    """Return (n_contacts, min_dist, n_self_contacts, max_penetration)."""
    if d.ncon == 0:
        return 0, 0.0, 0, 0.0
    dists = d.contact.dist[: d.ncon]
    g1 = d.contact.geom1[: d.ncon]
    g2 = d.contact.geom2[: d.ncon]
    is_self = (g1 != floor) & (g2 != floor)
    return int(d.ncon), float(dists.min()), int(is_self.sum()), float(max(0.0, -dists.min()))


def torso_upright_of(d):
    adr = prep.torso_zaxis_adr
    return float(d.sensordata[adr + 2])


def head_ratio_of(d):
    adr = prep.head_pos_adr
    return float(d.sensordata[adr + 2] / prep.standing_head_height)


def gravity_z(d):
    q = d.qpos[3:7]
    w, x, y, z = q
    # world -z rotated into body frame, z-component
    return -(1 - 2 * (x * x + y * y))


# ---------------------------------------------------------------- Experiment A
def experiment_A(n=64, joint_sigma=0.3, drop_h=(0.35, 0.9), seconds=4.0,
                 hold="default", label=""):
    """Drop with random orientation and let it settle under the PD servo."""
    d = mujoco.MjData(m)
    nsub = int(seconds / m.opt.timestep)
    settle_times, results = [], []
    t0 = time.perf_counter()
    explode = 0
    for i in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = prep.default_qpos
        d.qpos[7:] = prep.default_qpos[7:] + rng.normal(0, joint_sigma, m.nq - 7)
        np.clip(d.qpos[qadr], jnt_lo, jnt_hi, out=d.qpos[qadr])
        d.qpos[3:7] = rand_quat(rng)[0]
        mujoco.mj_forward(m, d)
        # place so the lowest point is at the requested drop height
        low = lowest_point(d)
        d.qpos[2] += rng.uniform(*drop_h) - low
        d.qvel[0:3] = rng.normal(0, 0.5, 3)
        d.qvel[3:6] = rng.normal(0, 1.5, 3)
        mujoco.mj_forward(m, d)
        if hold == "default":
            d.ctrl[:] = prep.default_joint_pos
        elif hold == "current":
            d.ctrl[:] = d.qpos[qadr]
        elif hold == "limp":
            d.ctrl[:] = d.qpos[qadr]
        settle_at = -1
        quiet = 0
        for k in range(nsub):
            mujoco.mj_step(m, d)
            if k % 25 == 0:
                v = np.abs(d.qvel).max()
                if v < 0.5:
                    quiet += 1
                    if quiet >= 4 and settle_at < 0:
                        settle_at = k * m.opt.timestep
                else:
                    quiet = 0
        if not np.isfinite(d.qpos).all():
            explode += 1
            continue
        ncon, mind, nself, pen = contact_report(d)
        results.append(dict(
            root_h=float(d.qpos[2]), gz=gravity_z(d), tu=torso_upright_of(d),
            head=head_ratio_of(d), vmax=float(np.abs(d.qvel).max()),
            settle=settle_at, ncon=ncon, nself=nself, pen=pen,
            qpos=d.qpos.copy(),
        ))
        settle_times.append(settle_at)
    wall = time.perf_counter() - t0
    arr = lambda k: np.array([r[k] for r in results])  # noqa: E731
    print(f"\n--- A[{label}] n={n} sigma={joint_sigma} hold={hold} {seconds}s ---")
    print(f"  wall {wall:.2f}s total, {wall / max(1, n) * 1000:.1f} ms per settle, "
          f"{n / wall:.1f} settles/s single-thread; diverged={explode}")
    if not results:
        return results
    st = np.array(settle_times)
    print(f"  settled (|qvel|max<0.5 held 100ms) within {seconds}s: "
          f"{(st >= 0).mean() * 100:.0f}%  median t={np.median(st[st >= 0]):.2f}s")
    print(f"  final |qvel|max: median {np.median(arr('vmax')):.2f} p90 {np.percentile(arr('vmax'), 90):.2f}")
    print(f"  root height: mean {arr('root_h').mean():.3f} p5 {np.percentile(arr('root_h'), 5):.3f} "
          f"p95 {np.percentile(arr('root_h'), 95):.3f}")
    print(f"  gravity_body_z: mean {arr('gz').mean():.2f} (-1 upright, 0 on its side/face)")
    print(f"  torso_upright: mean {arr('tu').mean():.2f} min {arr('tu').min():.2f} max {arr('tu').max():.2f}")
    print(f"  head ratio: mean {arr('head').mean():.2f}")
    print(f"  contacts at rest: median {np.median(arr('ncon')):.0f}, self-contacts median {np.median(arr('nself')):.0f}")
    print(f"  residual penetration at rest: max {arr('pen').max() * 1000:.2f} mm")
    # pose taxonomy by torso up-axis fore/lat/vert in world
    return results


# ---------------------------------------------------------------- Experiment C
def experiment_C(n=200):
    """Naive joint sampling + geometric placement, then mj_forward only."""
    d = mujoco.MjData(m)
    pen, nself, first_step_dv, ncon_l = [], [], [], []
    for _ in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = prep.default_qpos
        u = rng.random(m.nu)
        d.qpos[qadr] = jnt_lo + u * (jnt_hi - jnt_lo)
        d.qpos[3:7] = rand_quat(rng)[0]
        mujoco.mj_forward(m, d)
        d.qpos[2] += 0.001 - lowest_point(d)
        mujoco.mj_forward(m, d)
        ncon, mind, ns, p = contact_report(d)
        pen.append(p)
        nself.append(ns)
        ncon_l.append(ncon)
        d.ctrl[:] = d.qpos[qadr]
        v0 = d.qvel.copy()
        mujoco.mj_step(m, d, nstep=4)  # one control step
        first_step_dv.append(float(np.abs(d.qvel - v0).max()))
    pen = np.array(pen); nself = np.array(nself); dv = np.array(first_step_dv)
    print(f"\n--- C: uniform joint sampling + geometric root placement (n={n}) ---")
    print(f"  fraction with ANY self-collision at spawn: {(nself > 0).mean() * 100:.0f}% "
          f"(median count {np.median(nself):.0f}, max {nself.max()})")
    print(f"  penetration depth at spawn: median {np.median(pen) * 1000:.2f} mm, "
          f"p95 {np.percentile(pen, 95) * 1000:.2f} mm, max {pen.max() * 1000:.2f} mm")
    print(f"  |dqvel| after ONE 8 ms control step: median {np.median(dv):.2f}, "
          f"p95 {np.percentile(dv, 95):.2f}, max {dv.max():.2f} rad/s")
    print(f"  contacts at spawn: median {np.median(ncon_l):.0f}")


# ---------------------------------------------------------------- Experiment D
def experiment_D():
    """Is a settled state a fixed point? Re-apply and step with zero action."""
    res = experiment_A(n=24, joint_sigma=0.3, seconds=4.0, label="for-replay")
    d = mujoco.MjData(m)
    drifts_h, drifts_q, pens = [], [], []
    for r in res:
        mujoco.mj_resetData(m, d)
        d.qpos[:] = r["qpos"]
        mujoco.mj_forward(m, d)
        _, _, _, p = contact_report(d)
        pens.append(p)
        d.ctrl[:] = prep.default_joint_pos
        h0 = float(d.qpos[2]); q0 = d.qpos[7:].copy()
        mujoco.mj_step(m, d, nstep=int(0.5 / m.opt.timestep))
        drifts_h.append(abs(float(d.qpos[2]) - h0))
        drifts_q.append(float(np.abs(d.qpos[7:] - q0).max()))
    print("\n--- D: replaying a settled qpos through mj_forward only ---")
    print(f"  penetration when the state is re-applied: max {max(pens) * 1000:.3f} mm")
    print(f"  after 0.5 s of holding the NOMINAL pose target: "
          f"root height drift median {np.median(drifts_h) * 100:.1f} cm, "
          f"max joint drift median {np.median(drifts_q):.2f} rad")


if __name__ == "__main__":
    experiment_A(n=48, joint_sigma=0.10, label="tight-joints")
    experiment_A(n=48, joint_sigma=0.30, label="loose-joints")
    experiment_A(n=48, joint_sigma=0.30, hold="current", label="hold-spawn-pose")
    experiment_C(n=200)
    experiment_D()
