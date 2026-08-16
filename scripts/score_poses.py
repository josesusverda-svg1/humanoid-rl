"""What does each pose actually pay? One table, every reward term, every pose that matters.

This project has shipped ten reward exploits. Every one of them was a pose that paid well and
should not have, and every one was found late: by a person watching a video, or by summing
reward terms on a policy that had already trained for hours. Not one was found by reading the
reward.

They would ALL have been visible in this table before the run:

* the headstand (E29) pays as much as standing under a pelvis-height potential
* the one-armed prop pays full `rise` with one hand on the floor
* the splay-and-hop pays full foot load with all the weight on one leg
* the curled supine ball (E32) pays 99% of the signal at a pelvis height of 0.193 m

The poses come from three places, so none of them is my invention:

1. THE POSE BANK, the actual reset states the task uses (supine, prone, side, seated).
2. THE MODEL'S OWN nominal standing pose.
3. REAL ROLLOUTS from past checkpoints, sampled at the moment of interest. This is the
   important one: the exploit poses in this table are the exact configurations trained
   policies converged to, not poses I imagined an exploit might look like.

    python scripts/score_poses.py                       # current reward
    python scripts/score_poses.py --run runs/getup-...  # that run's config
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.getup import GetUpTask  # noqa: E402

#: Poses captured from a real policy rollout, as (label, checkpoint glob, seconds into episode).
#: Each is a behaviour a trained policy actually converged to, named for what it is.
ROLLOUT_POSES = [
    ("E32 curled ball (frozen)", "runs/getup-20260815-121544", 8.0),
    ("E32 launch (airborne)", "runs/getup-20260815-121544", 1.2),
    ("E31 launch (airborne)", "runs/getup-20260815-104811", 1.2),
    ("E31 mid-episode", "runs/getup-20260815-104811", 8.0),
]


class InjectPose(GetUpTask):
    """A task that resets into whatever pose is handed to it.

    Going through `reset_pose` rather than writing `env.state.qpos` directly is not a style
    choice. Writing qpos does NOT move the physics: the fields the reward actually reads
    (`foot_force`, `gravity_body`, `key_body_pos`) are derived by MuJoCo, so a direct write
    scores the old contacts against the new joint angles. That mistake produced two invalid
    verifications in this project already.
    """

    qpos_target: np.ndarray | None = None
    qvel_target: np.ndarray | None = None

    def reset_pose(self, state, idx, rng):  # noqa: ANN001
        if self.qpos_target is None:
            return super().reset_pose(state, idx, rng)
        state.task_state["from_standing"][idx] = False
        return (np.tile(self.qpos_target, (idx.size, 1)),
                np.tile(self.qvel_target, (idx.size, 1)))


def score(task: GetUpTask, env: ThreadedVecEnv, names: list[str]) -> tuple[np.ndarray, float]:
    """Per-term reward for whatever state the env is currently in, env 0."""
    state = env.state
    terms = np.zeros((state.qpos.shape[0], len(names)))
    total = task.reward_batch(state, terms)
    return terms[0].copy(), float(total[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None,
                    help="score under this run's config instead of configs/getup.yaml")
    ap.add_argument("--settle", type=int, default=25,
                    help="steps holding the pose so MuJoCo resolves contacts")
    args = ap.parse_args()

    cfg = Config.load(args.run / "config.yaml" if args.run
                      else REPO_ROOT / "configs" / "getup.yaml")
    device = torch.device(resolve_device(cfg.run.device))
    task = InjectPose(cfg.getup)
    names = list(task.reward_term_names)

    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, task, num_envs=1, num_workers=1,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=0,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode,
    )
    prepared = env.prepared
    settle = args.settle
    rows: list[tuple[str, np.ndarray, float, float, float, float]] = []

    def record(label: str, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Reset the physics INTO this pose, hold it, and score it.

        TWO traps here, both of which produced a wrong table on the first attempt.

        Scoring at the reset state with no physics step gives ZERO contact force, because
        MuJoCo has not resolved contacts yet. The first version of this table reported the
        model's own standing pose as carrying 0.00 body weight through the feet, which made
        `rise` and all three U-gated terms read 0.000 for a perfect stand. A pose scorer that
        scores standing at zero is worse than no scorer.

        But settling with a zero action does not fix it either: in `full_range` mode action 0
        commands the NOMINAL STANDING joint angles, so a "settled" supine pose is a supine
        body being hauled upright by the servos, and what gets scored is the controller. So
        hold the pose explicitly: the action that reproduces these very joint angles is
        (q_joints - default) / action_scale, which is 0 for standing and the right thing
        everywhere else.
        """
        task.qpos_target, task.qvel_target = qpos, qvel
        env.reset()
        hold = np.clip(
            (qpos[prepared.actuator_qpos_adr] - prepared.default_joint_pos)
            / prepared.action_scale, -1.0, 1.0).astype(np.float32)[None, :]
        for _ in range(settle):
            env.step(hold)
        terms, total = score(task, env, names)
        s = env.state
        rows.append((label, terms, total, float(s.root_height[0]),
                     float(s.head_height_ratio[0]),
                     float(s.foot_force[0, :2].sum()) / task._body_weight))  # noqa: SLF001

    # 1. The nominal standing pose, from the model itself.
    record("STANDING (model nominal)", prepared.default_qpos,
           np.zeros(prepared.model.nv))

    # 2. The reset states the task actually uses, one of each label.
    bank = np.load(REPO_ROOT / cfg.getup.bank_path, allow_pickle=False)
    labels, qpos_bank = bank["label"], bank["qpos"]
    for want in ("supine", "prone", "side_left", "side_right", "seated"):
        idx = np.flatnonzero(labels == want)
        if idx.size:
            record(f"bank: {want}", qpos_bank[idx[0]], np.zeros(prepared.model.nv))

    # 3. Poses real policies converged to. These are the exploits, in their own words.
    #
    # Replayed in a SECOND env whose observations have the historical 95-dim width (E34
    # added 3 task observations, so old checkpoints do not fit the new env). The reward is
    # unaffected: reward_batch never reads the observation, so the new pricing scores the
    # old behaviours exactly.
    class LegacyObs(GetUpTask):
        @property
        def task_obs_dim(self) -> int:
            return 0

        def observe_batch(self, state, out):  # noqa: ANN001
            return

    legacy_task = LegacyObs(cfg.getup)
    legacy_env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, legacy_task, num_envs=1, num_workers=1,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=0,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode,
    )
    for label, run_dir, t in ROLLOUT_POSES:
        run = REPO_ROOT / run_dir
        ckpts = sorted((run / "checkpoints").glob("iter_*.pt"))
        if not ckpts:
            continue
        # Only the NETWORK section of the old run is needed; read it raw rather than through
        # Config.load, which rightly rejects configs containing since-deleted getup keys.
        import yaml
        net = yaml.safe_load((run / "config.yaml").read_text())["network"]
        pol = ActorCritic(
            legacy_env.obs_dim, legacy_env.nu, actor_hidden=tuple(net["actor_hidden"]),
            critic_hidden=tuple(net["critic_hidden"]),
            activation=net["activation"],
            init_noise_std=net["init_noise_std"]).to(device)
        pol.load_state_dict(torch.load(ckpts[-1], map_location=device,
                                       weights_only=False)["policy"])
        pol.eval()
        obs = legacy_env.reset()
        want_step = int(t / legacy_env.dt)
        with torch.no_grad():
            for _ in range(want_step):
                a = pol.act_deterministic(
                    torch.as_tensor(obs, dtype=torch.float32, device=device))
                obs = legacy_env.step(a.cpu().numpy()).obs
        terms, total = score(legacy_task, legacy_env, names)
        s = legacy_env.state
        rows.append((f"{label} @{t}s", terms, total, float(s.root_height[0]),
                     float(s.head_height_ratio[0]),
                     float(s.foot_force[0, :2].sum()) / legacy_task._body_weight))  # noqa: SLF001
    env.close()
    legacy_env.close()

    width = max(len(r[0]) for r in rows) + 2
    head = f"{'pose':<{width}}{'pelvis':>8}{'head':>7}{'feet':>7}"
    head += "".join(f"{n:>10}" for n in names) + f"{'TOTAL':>10}"
    print(f"\nreward per step, {'run ' + args.run.name if args.run else 'configs/getup.yaml'}\n")
    print(head)
    print("-" * len(head))
    standing_total = rows[0][2]
    for label, terms, total, pelvis, head_r, feet in rows:
        line = f"{label:<{width}}{pelvis:>8.3f}{head_r:>7.2f}{feet:>7.2f}"
        line += "".join(f"{v:>10.3f}" for v in terms) + f"{total:>10.3f}"
        print(line)
    print("-" * len(head))
    print("\nAs a fraction of what STANDING pays. Anything near or above 1.00 that is not "
          "standing is an exploit:\n")
    for label, _, total, *_ in rows:
        share = total / standing_total if abs(standing_total) > 1e-9 else float("nan")
        bar = "#" * max(0, min(40, int(share * 20)))
        print(f"  {label:<{width}}{share:>7.2f}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
