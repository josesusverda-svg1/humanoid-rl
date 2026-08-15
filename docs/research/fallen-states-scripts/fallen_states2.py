"""Part 2: policy-fall replay vs drop-and-settle, taxonomy, seated poses, roll stability."""
from __future__ import annotations

import sys
import time

import mujoco
import numpy as np
import torch

sys.path.insert(0, "/Users/BrickLayer/Desktop/HumonoidAI")
from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.envs.model_prep import prepare, _geom_lowest_z  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402

REPO = "/Users/BrickLayer/Desktop/HumonoidAI"
MODEL = f"{REPO}/humanoid_rl/models/humanoid_scene.xml"
RUN = f"{REPO}/runs/envelope-20260814-100402"

prep = prepare(MODEL)
m = prep.model
floor = prep.floor_geom_id
qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
jnt_lo = m.actuator_ctrlrange[:, 0].copy()
jnt_hi = m.actuator_ctrlrange[:, 1].copy()
jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
         for i in range(m.nu)]
rng = np.random.default_rng(1)


def taxonomy(qpos):
    """Classify a batch of qpos rows by body-frame gravity. Returns dict of fractions."""
    q = qpos[:, 3:7]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # world -z rotated into body frame
    gx = -(2 * (x * z - w * y))
    gy = -(2 * (y * z + w * x))
    gz = -(1 - 2 * (x * x + y * y))
    h = qpos[:, 2]
    lbl = np.full(len(qpos), "other", dtype=object)
    lbl[gx < -0.6] = "supine(face up)"
    lbl[gx > 0.6] = "prone(face down)"
    lbl[np.abs(gy) > 0.6] = "side"
    lbl[(gz < -0.6) & (h > 0.25)] = "torso-up (seated/kneel)"
    out = {}
    for k in ["supine(face up)", "prone(face down)", "side", "torso-up (seated/kneel)", "other"]:
        out[k] = float((lbl == k).mean())
    return out, (gx, gy, gz)


def describe(name, qpos):
    tax, (gx, gy, gz) = taxonomy(qpos)
    j = qpos[:, 7:]
    span = (jnt_hi - jnt_lo)
    jstd = j.std(axis=0)
    print(f"\n[{name}] n={len(qpos)}")
    print("  taxonomy: " + ", ".join(f"{k}={v * 100:.0f}%" for k, v in tax.items() if v > 0))
    print(f"  root height: mean {qpos[:, 2].mean():.3f} sd {qpos[:, 2].std():.3f} "
          f"[{qpos[:, 2].min():.3f}, {qpos[:, 2].max():.3f}]")
    print(f"  joint-angle sd, mean over joints: {jstd.mean():.3f} rad "
          f"({(jstd[np.argsort(qadr - 7)] / span[np.argsort(qadr - 7)]).mean() * 100:.1f}% of range)")
    name_of = {int(a - 7): jname[i] for i, a in enumerate(qadr)}
    order = np.argsort(-jstd)
    print("  most-varied joints: " + ", ".join(
        f"{name_of.get(int(o), o)}={jstd[o]:.2f}" for o in order[:4]))
    return tax


# --------------------------------------------------------------- policy fall replay
def policy_falls(n_envs=64, seconds_after_fall=2.0, max_steps=2200):
    cfg = Config.load(f"{RUN}/config.yaml")
    task = LocomotionTask(cfg.task)
    # never terminate: we want to watch it hit the floor and settle
    task.terminated_batch = lambda state: np.zeros(state.num_envs, dtype=bool)
    from humanoid_rl.envs.domain_rand import DomainRandConfig
    env = ThreadedVecEnv(MODEL, task, n_envs, decimation=4, max_episode_steps=10 ** 9,
                         seed=3, domain_rand=DomainRandConfig(enabled=False),
                         action_filter_hz=cfg.env.action_filter_hz)
    device = torch.device("cpu")
    policy = ActorCritic(env.obs_dim, env.nu,
                         actor_hidden=tuple(cfg.network.actor_hidden),
                         critic_hidden=tuple(cfg.network.critic_hidden),
                         activation=cfg.network.activation,
                         init_noise_std=cfg.network.init_noise_std).to(device)
    st = torch.load(f"{RUN}/checkpoints/best.pt", map_location=device, weights_only=False)
    policy.load_state_dict(st["policy"])
    policy.eval()

    obs = env.reset()
    fall_step = np.full(n_envs, -1)
    captured = {}
    t0 = time.perf_counter()
    hold = int(seconds_after_fall / env.dt)
    with torch.no_grad():
        for k in range(max_steps):
            a = policy.act_deterministic(torch.as_tensor(obs, device=device)).cpu().numpy()
            res = env.step(a)
            obs = res.obs
            s = env.state
            newly = (fall_step < 0) & (s.root_height < 0.62 * prep.standing_height)
            fall_step[newly] = k
            ready = (fall_step >= 0) & (k - fall_step == hold)
            for i in np.flatnonzero(ready):
                if i not in captured:
                    captured[i] = (s.qpos[i].copy(), s.qvel[i].copy(), fall_step[i])
            if len(captured) >= n_envs or (k > 900 and len(captured) == (fall_step >= 0).sum()
                                           and (fall_step >= 0).all()):
                break
    wall = time.perf_counter() - t0
    print(f"\n=== policy fall replay: {len(captured)}/{n_envs} envs fell within {k} steps "
          f"({k * env.dt:.0f} s), wall {wall:.0f}s ===")
    fs = fall_step[fall_step >= 0]
    if fs.size:
        print(f"  steps until first fall: median {np.median(fs):.0f} ({np.median(fs) * env.dt:.1f} s), "
              f"p10 {np.percentile(fs, 10):.0f}, p90 {np.percentile(fs, 90):.0f}")
    qp = np.array([v[0] for v in captured.values()])
    qv = np.array([v[1] for v in captured.values()])
    print(f"  |qvel|max {seconds_after_fall}s after the fall: median "
          f"{np.median(np.abs(qv).max(axis=1)):.2f}, p90 {np.percentile(np.abs(qv).max(axis=1), 90):.2f}"
          "  (policy still acting)")
    env.close()
    return qp, qv


# --------------------------------------------------------------- drop and settle
def drop_settle(n=200, joint_sigma=0.3, hold="default", seconds=3.0, tilt_only=False):
    d = mujoco.MjData(m)
    out = []
    for _ in range(n):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = prep.default_qpos
        d.qpos[7:] += rng.normal(0, joint_sigma, m.nq - 7)
        np.clip(d.qpos[qadr], jnt_lo, jnt_hi, out=d.qpos[qadr])
        if tilt_only:
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            ang = rng.uniform(0.6, 1.6)
            d.qpos[3:7] = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])
        else:
            u = rng.random(3)
            q = np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
                          np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                          np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
                          np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
            d.qpos[3:7] = np.array([q[3], q[0], q[1], q[2]])
        mujoco.mj_forward(m, d)
        low = min(_geom_lowest_z(m, d, g) for g in range(m.ngeom) if g != floor)
        d.qpos[2] += rng.uniform(0.3, 0.8) - low
        d.qvel[0:3] = rng.normal(0, 0.5, 3)
        d.qvel[3:6] = rng.normal(0, 1.5, 3)
        mujoco.mj_forward(m, d)
        d.ctrl[:] = prep.default_joint_pos if hold == "default" else d.qpos[qadr]
        mujoco.mj_step(m, d, nstep=int(seconds / m.opt.timestep))
        if np.isfinite(d.qpos).all():
            out.append(d.qpos.copy())
    return np.array(out)


# --------------------------------------------------------------- constructed poses
def constructed(name, joints: dict, settle=2.0, perturb=0.0, quat=None):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = prep.default_qpos
    if quat is not None:
        d.qpos[3:7] = quat
    for jn, val in joints.items():
        i = jname.index(jn)
        d.qpos[qadr[i]] = val
    np.clip(d.qpos[qadr], jnt_lo, jnt_hi, out=d.qpos[qadr])
    mujoco.mj_forward(m, d)
    low = min(_geom_lowest_z(m, d, g) for g in range(m.ngeom) if g != floor)
    d.qpos[2] += 0.002 - low
    mujoco.mj_forward(m, d)
    ncon0 = d.ncon
    pen0 = -min(d.contact.dist[: d.ncon]) if d.ncon else 0.0
    h0 = float(d.qpos[2])
    d.ctrl[:] = d.qpos[qadr]
    if perturb:
        d.qvel[3] = perturb  # roll rate about world x
    mujoco.mj_step(m, d, nstep=int(settle / m.opt.timestep))
    q = d.qpos.copy()
    tax, (gx, gy, gz) = taxonomy(q[None])
    lab = [k for k, v in tax.items() if v > 0][0]
    print(f"  {name:34s} spawn h={h0:.3f} pen={pen0 * 1000:.1f}mm ncon={ncon0} -> "
          f"after {settle}s h={q[2]:.3f} g=({gx[0]:+.2f},{gy[0]:+.2f},{gz[0]:+.2f}) {lab} "
          f"|qvel|max={np.abs(d.qvel).max():.2f}")
    return q


if __name__ == "__main__":
    print("=" * 78)
    qp_pol, qv_pol = policy_falls()
    np.save("/private/tmp/claude-502/-Users-BrickLayer-Desktop-HumonoidAI/"
            "2ea9e069-53fd-4656-90b0-68f777666b8d/scratchpad/policy_falls.npy", qp_pol)
    describe("policy falls, settled 2 s", qp_pol)

    qp_drop = drop_settle(n=200, joint_sigma=0.3, hold="default")
    describe("drop, random orientation, PD holds NOMINAL pose", qp_drop)

    qp_drop2 = drop_settle(n=200, joint_sigma=0.3, hold="current")
    describe("drop, random orientation, PD holds SPAWN pose", qp_drop2)

    qp_tilt = drop_settle(n=200, joint_sigma=0.3, hold="current", tilt_only=True)
    describe("drop from a topple (tilt 35-92 deg), PD holds spawn", qp_tilt)

    print("\n=== constructed poses: can they be built and are they stable? ===")
    constructed("seated, legs out front", {
        "right_hip_y": -1.5, "left_hip_y": -1.5, "right_knee": 0.2, "left_knee": 0.2,
        "right_shoulder_x": 1.2, "left_shoulder_x": -1.2})
    constructed("seated, knees bent (crash sit)", {
        "right_hip_y": -1.6, "left_hip_y": -1.6, "right_knee": 1.6, "left_knee": 1.6,
        "right_shoulder_x": 1.0, "left_shoulder_x": -1.0})
    constructed("kneeling", {
        "right_hip_y": -0.2, "left_hip_y": -0.2, "right_knee": 2.6, "left_knee": 2.6,
        "right_ankle_y": -0.9, "left_ankle_y": -0.9})
    constructed("all-fours (hands+knees)", {
        "right_hip_y": -1.5, "left_hip_y": -1.5, "right_knee": 2.4, "left_knee": 2.4,
        "abdomen_y": 1.4, "right_shoulder_x": 0.1, "left_shoulder_x": -0.1,
        "right_shoulder_y": 1.4, "left_shoulder_y": 1.4})
    constructed("deep squat", {
        "right_hip_y": -1.8, "left_hip_y": -1.8, "right_knee": 2.4, "left_knee": 2.4,
        "right_ankle_y": -0.9, "left_ankle_y": -0.9, "abdomen_y": 0.5})

    print("\n=== roll stability of a supine pose (trunk is 3 spheres) ===")
    SUPINE = np.array([0.70711, 0.0, -0.70711, 0.0])
    PRONE = np.array([0.70711, 0.0, 0.70711, 0.0])
    for w in (0.0, 0.5, 1.0, 2.0, 3.0):
        constructed(f"supine + roll rate {w} rad/s", {
            "right_shoulder_x": 1.3, "left_shoulder_x": -1.3,
        }, settle=2.0, perturb=w, quat=SUPINE)
    for w in (0.0, 1.0, 2.0):
        constructed(f"prone  + roll rate {w} rad/s", {
            "right_shoulder_x": 1.3, "left_shoulder_x": -1.3,
        }, settle=2.0, perturb=w, quat=PRONE)
