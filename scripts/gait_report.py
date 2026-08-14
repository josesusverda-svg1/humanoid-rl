"""Measure whether a policy is actually stepping, and compare it to real human mocap.

Motivated by a concrete failure. The `gait_symmetry` metric in LocomotionTask is

    min(left_contact_fraction, right_contact_fraction) / max(same)

which is 1.0 for a perfectly even gait, and *also* 1.0 for a humanoid standing with both
feet planted, since both fractions are then 1.0. A policy under a strong symmetry loss found
exactly that second solution: a wide two-footed brace, dragged forward by ground slip,
scoring 0.91 symmetry while taking no steps at all. The number said the gait defect was
fixed. A video said otherwise.

The quantities below cannot be satisfied by standing still, and each has a known value for
real human walking, so a policy can be scored against the retargeted mocap rather than
against a threshold someone guessed:

* **double support** - fraction of time both feet are loaded. Real walking is about 0.20 to
  0.25 at normal speed. Standing with both feet down is 1.0.
* **step rate** - foot strikes per second. Real walking is roughly 1.6 to 2.0. Standing is 0.
* **stance width** - lateral distance between the feet, in the humanoid's own heading frame
  so that turning does not read as a widening stance. Real walking is about 0.10 to 0.15 m;
  a bracing posture splays far wider.
* **stride length** - distance travelled per foot strike. Near zero for a policy that slides
  in place rather than stepping.

Usage:
    python scripts/gait_report.py --run runs/<run> --checkpoint best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.render import build_render_env  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionTask  # noqa: E402


def contact_stats(contact: np.ndarray, dt: float) -> dict[str, float]:
    """Gait statistics from a (T, 2) foot-contact trace.

    A step is counted as a rising edge of contact: the foot was in the air and is now
    loaded. That is a real foot strike, and unlike a contact *fraction* it cannot be
    produced by a humanoid that never lifts a foot.
    """
    left = contact[:, 0] > 0.5
    right = contact[:, 1] > 0.5
    strikes = int(np.sum(left[1:] & ~left[:-1])) + int(np.sum(right[1:] & ~right[:-1]))
    duration = max(len(contact) * dt, 1e-9)
    return {
        "double_support": float((left & right).mean()),
        "flight": float((~left & ~right).mean()),
        "step_rate": strikes / duration,
        "strikes": float(strikes),
    }


@torch.no_grad()
def rollout(run: Path, checkpoint: str, steps: int, seed: int, command: float) -> dict[str, float]:
    """Drive the policy at a constant forward command and measure how it moves."""
    config = Config.load(run / "config.yaml")
    device = torch.device(resolve_device(config.run.device))
    env = build_render_env(REPO_ROOT / config.env.model_path, LocomotionTask(config.task), seed=seed)

    policy = ActorCritic(
        env.obs_dim,
        env.nu,
        actor_hidden=tuple(config.network.actor_hidden),
        critic_hidden=tuple(config.network.critic_hidden),
        activation=config.network.activation,
        init_noise_std=config.network.init_noise_std,
    ).to(device)
    ckpt = torch.load(run / "checkpoints" / checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs = env.reset()
    contacts: list[np.ndarray] = []
    widths: list[float] = []
    positions: list[np.ndarray] = []

    for _ in range(steps):
        # Hold a constant "walk forward" command so the measurement is not confounded by a
        # command schedule that spends part of its time asking the policy to stand still.
        env.state.task_state["command"][0] = np.array([command, 0.0, 0.0])
        action = policy.act_deterministic(torch.as_tensor(obs, dtype=torch.float32, device=device))
        result = env.step(action.cpu().numpy())
        obs = result.obs

        st = env.state
        contacts.append(st.foot_contact[0].copy())
        # Feet are key bodies 0 (left) and 1 (right). Rotate their separation into the
        # heading frame and take the lateral component.
        rel = st.key_body_pos[0, 0, :2] - st.key_body_pos[0, 1, :2]
        yaw = float(st.heading[0])
        widths.append(abs(-np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1]))
        positions.append(st.root_pos[0, :2].copy())

        if result.terminated[0] or result.truncated[0]:
            break

    env.close()

    contact = np.stack(contacts)
    pos = np.stack(positions)
    dt = env.dt
    stats = contact_stats(contact, dt)
    # Path length, not straight-line displacement, so a turning policy is not undercounted.
    travel = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    elapsed = len(contact) * dt

    return {
        **stats,
        "stance_width": float(np.mean(widths)),
        "stride_length": travel / stats["strikes"] if stats["strikes"] > 0 else 0.0,
        "speed": travel / elapsed,
        "seconds": elapsed,
    }


def reference_stats(config: Config) -> dict[str, float]:
    """The same statistics on the retargeted mocap, as ground truth for this body."""
    from humanoid_rl.envs.model_prep import prepare
    from humanoid_rl.motion.library import MotionLibrary

    prepared = prepare(REPO_ROOT / config.env.model_path)
    lib = MotionLibrary.build(
        REPO_ROOT / config.clip_dir, prepared.model, include=list(config.clip_include) or None
    )
    # The reference has no simulated contacts, so foot height is the contact proxy: a foot
    # within 3 cm of its lowest observed height is taken to be loaded.
    feet = lib.key_local[:, :2, 2]
    contact = (feet < feet.min(axis=0, keepdims=True) + 0.03).astype(np.float32)
    dt = 1.0 / float(lib.fps)
    out = contact_stats(contact, dt)
    rel = lib.key_local[:, 0, :2] - lib.key_local[:, 1, :2]
    out["stance_width"] = float(np.abs(rel[:, 1]).mean())
    root = lib.qpos[:, :2]
    travel = float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum())
    out["stride_length"] = travel / out["strikes"] if out["strikes"] > 0 else 0.0
    out["speed"] = travel / (len(contact) * dt)
    return out


ROWS = [
    ("double_support", "0.20 - 0.25   (1.0 means never lifting a foot)"),
    ("flight", "0.00          (above zero means running)"),
    ("step_rate", "1.6 - 2.0     foot strikes per second"),
    ("stance_width", "0.10 - 0.15 m"),
    ("stride_length", "0.6 - 0.8 m   per strike"),
    ("speed", "1.2 - 1.4 m/s"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--command", type=float, default=1.0, help="forward command, m/s")
    args = ap.parse_args()

    policy = rollout(args.run, args.checkpoint, args.steps, args.seed, args.command)
    config = Config.load(args.run / "config.yaml")
    try:
        ref = reference_stats(config)
    except Exception as exc:  # noqa: BLE001 - the reference is a nicety, not a requirement
        print(f"(reference unavailable: {exc})")
        ref = {}

    print(f"\n{args.run.name} / {args.checkpoint}   "
          f"{policy['seconds']:.1f}s, command {args.command:.1f} m/s\n")
    print(f"{'':<16}{'policy':>9}{'mocap':>9}   real human walking")
    for key, expect in ROWS:
        p, r = policy.get(key), ref.get(key)
        ps = f"{p:>9.2f}" if p is not None else f"{'-':>9}"
        rs = f"{r:>9.2f}" if r is not None else f"{'-':>9}"
        print(f"{key:<16}{ps}{rs}   {expect}")
    # The same scoring the dashboard shows, so a terminal check and the UI cannot disagree.
    try:
        import json as _json

        from humanoid_rl.gait_score import human_summary, score_row

        evaluations = [
            _json.loads(line)
            for line in (args.run / "metrics.jsonl").read_text().splitlines()
            if "eval/episode_return" in line
        ]
        if evaluations:
            print()
            print(human_summary(score_row(evaluations[-1])))
    except (FileNotFoundError, ValueError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
