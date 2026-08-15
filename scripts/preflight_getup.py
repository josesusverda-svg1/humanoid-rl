"""Everything that must be true before a get-up run starts, checked rather than assumed.

Written after a run of faults that all shared one shape: a thing was built, looked right, and
was never actually connected. The full-range action mode existed for two runs before anything
called it. The video overlay printed a command the policy never received. A skeleton flip-book
silently rendered nothing for 2000 iterations. An Oracle check lost its decorator in a merge
and never ran.

Every check here FAILS LOUDLY rather than warning, and every one compares two things that were
specified independently. A check that reads a config value and prints it back has verified
nothing.

    python scripts/preflight_getup.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.getup import GetUpTask  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'XX'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    cfg = Config.load(REPO_ROOT / "configs" / "getup.yaml")
    task = GetUpTask(cfg.getup)
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=256, num_workers=4,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=0,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode,
    )
    prep = env.prepared
    model = prep.model

    # --- the action space can express the pose the task requires
    knee = [i for i, n in enumerate(prep.joint_names) if "knee" in n][0]
    j = model.actuator_trnid[knee, 0]
    lo, hi = model.jnt_range[j]
    reach = min(hi, prep.default_joint_pos[knee] + prep.action_scale[knee])
    check("knee can reach a kneel", reach >= 2.4,
          f"commandable to {reach:.2f} rad, a kneel needs 2.40")

    # --- and that mode is really the one training will use, not just the one in the yaml
    check("action_scale_mode reached the env", cfg.env.action_scale_mode == "full_range"
          and reach >= 2.4, f"config says {cfg.env.action_scale_mode!r}")

    # --- hold converts to a whole number of steps, and the episode has room for it
    dt = model.opt.timestep * cfg.env.decimation
    steps = cfg.getup.hold_seconds / dt
    check("hold is a whole number of control steps", abs(steps - round(steps)) < 1e-6,
          f"{cfg.getup.hold_seconds}s / {dt*1000:.0f}ms = {steps:.1f}")
    check("episode leaves room for the hold", cfg.env.max_episode_steps >= 3 * round(steps),
          f"{cfg.env.max_episode_steps} steps vs {round(steps)} needed, "
          f"{cfg.env.max_episode_steps*dt:.0f}s episode")

    # --- reward plumbing: as many terms as names, and none of them NaN
    obs = env.reset()
    rewards = []
    for _ in range(200):
        r = env.step(np.zeros((256, env.nu), dtype=np.float32))
        rewards.append(r.reward)
    check("reward term count matches names",
          r.reward_terms.shape[1] == len(task.reward_term_names),
          f"{r.reward_terms.shape[1]} columns, {len(task.reward_term_names)} names")
    check("rewards are finite", bool(np.isfinite(rewards).all()))

    # --- a do-nothing policy must score zero success, or the task is already gamed
    m = task.eval_metrics(env.state)
    check("idle policy earns no success", m["held_ever_frac"] == 0.0,
          f"held_ever {m['held_ever_frac']:.3f}")

    # --- every metric the watchers read actually exists
    need = ["standing_frac", "held_ever_frac", "time_to_fall_frac", "hold_progress",
            "head_height_ratio", "root_height", "pelvis_upright", "foot_load_bw",
            "hand_height_gap", "hands_down_frac", "knee_max", "spin_deg_s", "ball_hits"]
    missing = [k for k in need if k not in m]
    check("all watched metrics exist", not missing, f"missing {missing}" if missing else "")

    # --- the ball, if it is on at all. Switched off at the user's request: the pose bank now
    # covers every side explicitly, so knocking him down to generate variety is redundant.
    if cfg.getup.ball_enabled:
        class StandingReset(GetUpTask):
            def reset_pose(self, state, idx, rng):
                q = np.tile(prep.default_qpos, (idx.size, 1))
                state.task_state["from_standing"][idx] = False
                return q, np.zeros((idx.size, state.qvel.shape[1]))

        up_task = StandingReset(cfg.getup)
        up_env = ThreadedVecEnv(
            REPO_ROOT / cfg.env.model_path, up_task, num_envs=128, num_workers=4,
            decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
            action_filter_hz=cfg.env.action_filter_hz, seed=3,
            domain_rand=replace(cfg.domain_rand, enabled=False),
            action_scale_mode=cfg.env.action_scale_mode,
        )
        up_env.reset()
        for _ in range(400):
            up_env.step(np.zeros((128, up_env.nu), dtype=np.float32))
        balls = up_env.state.task_state["ball_hits"]
        check("every upright humanoid gets exactly one ball",
              float(balls.min()) == 1.0 and float(balls.max()) == 1.0,
              f"min {balls.min():.0f}, max {balls.max():.0f}")
        up_env.close()
    else:
        check("ball is off, as configured", True, "pose bank covers the sides instead")

    # --- impulse magnitude, computed from the config rather than sampled by luck
    rng = np.random.default_rng(0)
    dv = ((1.0 + cfg.getup.ball_restitution)
          * rng.uniform(*cfg.getup.ball_mass_range, 20000)
          * rng.uniform(*cfg.getup.ball_speed_range, 20000)
          / float(model.body_mass.sum()))
    check("ball topples without launching", dv.max() <= 1.5 and dv.min() >= 0.25,
          f"pelvis dv {dv.min():.2f}-{dv.max():.2f} m/s "
          f"(0.5 topples 100%, 2.0+ starts to throw him)")

    # --- pose bank matches this model and is mostly on the floor and balanced
    bank = np.load(REPO_ROOT / cfg.getup.bank_path, allow_pickle=False)
    check("pose bank matches the model", int(bank["nq"]) == model.nq,
          f"bank nq {int(bank['nq'])}, model nq {model.nq}")
    lab = bank["label"]
    floor = float((bank["generator"] != "standing").mean())
    check("bank is mostly on the floor", floor > 0.5, f"{floor:.0%}")
    # All four ways of lying, not just left against right. The request was explicitly "equally
    # on every side", and an earlier bank that passed a left/right check was still 44% prone
    # against 3% supine.
    four = {k: int((lab == k).sum()) for k in
            ("prone", "supine", "side_left", "side_right")}
    lo, hi = min(four.values()), max(four.values())
    check("all four lying orientations are balanced", hi <= 1.25 * max(lo, 1),
          ", ".join(f"{k} {v}" for k, v in four.items()))
    check("seated poses present", int((lab == "seated").sum()) > 0,
          f"{int((lab == 'seated').sum())} (only reachable by hand: a fall never lands there)")

    # --- THE CHECK THIS FILE MISSED THE FIRST TIME.
    #
    # The training, evaluation and render environments share ONE Task instance at different
    # widths (4096 / 64 / 1). Anything the task keeps per-environment on ITSELF is sized to
    # whichever env called init_state last and index-errors against the others. That crashed
    # the first evaluation of every run, and this file passed clean beforehand because it only
    # ever built one environment. Build a second, narrower one against the same task and step
    # both, which is exactly what the trainer does.
    narrow = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=8, num_workers=2,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=1,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode,
    )
    shared_ok, why = True, ""
    try:
        narrow.reset()
        for _ in range(30):
            narrow.step(np.zeros((8, narrow.nu), dtype=np.float32))
        env.reset()
        for _ in range(30):
            env.step(np.zeros((256, env.nu), dtype=np.float32))
        narrow.step(np.zeros((8, narrow.nu), dtype=np.float32))
    except Exception as exc:  # noqa: BLE001
        shared_ok, why = False, f"{type(exc).__name__}: {exc}"
    narrow.close()
    check("one task drives envs of different widths", shared_ok, why)

    env.close()

    # --- the walking task must be untouched by all of this
    walk_env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, LocomotionTask(Config.load(
            REPO_ROOT / "configs" / "default.yaml").task),
        num_envs=64, num_workers=4, decimation=4, max_episode_steps=2500,
        action_filter_hz=8.0, seed=0,
        domain_rand=replace(cfg.domain_rand, enabled=True),
    )
    walk_env.reset()
    wr = []
    for _ in range(200):
        res = walk_env.step(np.random.uniform(-0.3, 0.3, (64, walk_env.nu)).astype(np.float32))
        wr.append(res.reward.mean())
    check("walking task still runs", bool(np.isfinite(wr).all()), f"mean {np.mean(wr):.3f}")
    check("walking keeps the fraction action scale",
          abs(float(walk_env.prepared.action_scale[knee]) - 0.838) < 1e-3,
          f"{float(walk_env.prepared.action_scale[knee]):.3f}")
    walk_env.close()

    print(f"\n{'ALL CLEAR' if not failures else f'{len(failures)} FAILED: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
