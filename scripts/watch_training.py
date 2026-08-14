"""Watchdog for a training run. Emits one line per noteworthy event, nothing otherwise.

Designed to be driven by the Monitor tool, where every stdout line becomes a notification.
That imposes two hard constraints:

1. **Be quiet.** A line per iteration would be thousands of notifications and the monitor
   would be shut off. Only genuine state *changes* are emitted.
2. **Never be silently quiet.** Silence must mean "healthy and progressing", never "died
   twenty minutes ago". So this watches for stalls, crashes and process death explicitly,
   rather than only watching for good news. A watchdog that only reports success cannot
   distinguish a converged run from a crashed one.

Usage:
    .venv/bin/python scripts/watch_training.py                    # newest run
    .venv/bin/python scripts/watch_training.py --run runs/... --poll 60
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def emit(kind: str, message: str) -> None:
    """One event. Flushed immediately: a buffered watchdog is a broken watchdog."""
    print(f"[{kind}] {message}", flush=True)


def read_last_rows(path: Path, count: int = 40) -> list[dict]:
    """Read the last `count` JSONL rows, tolerating a torn final line."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        block = min(400_000, size)
        handle.seek(size - block)
        text = handle.read(block).decode("utf-8", errors="ignore")
    rows = []
    for line in text.strip().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-count:]


def training_alive() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "humanoid_rl.train"], capture_output=True, text=True, timeout=10
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        return True  # never claim death on a failure to check


def newest_run(runs_dir: Path) -> Path:
    candidates = [
        p for p in runs_dir.iterdir() if p.is_dir() and (p / "metrics.jsonl").exists()
        and not p.name.startswith("ABANDONED")
    ]
    if not candidates:
        raise FileNotFoundError(f"no runs with metrics.jsonl in {runs_dir}")
    return max(candidates, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between checks")
    ap.add_argument("--milestone-every", type=int, default=500, help="iterations per summary")
    ap.add_argument("--stall-minutes", type=float, default=6.0)
    args = ap.parse_args()

    run_dir = args.run or newest_run(REPO_ROOT / "runs")
    metrics_path = run_dir / "metrics.jsonl"
    emit("WATCHING", f"{run_dir.name}  poll={args.poll:.0f}s  milestone={args.milestone_every}")

    # Thresholds. torso_upright is the specific regression being guarded against: the
    # fold-forward-at-the-waist gait that reward shaping found at 24.6M steps.
    TORSO_FLOOR = 0.70
    WALKING_FALL_RATE = 0.5
    # Foot slip as a fraction of travel speed. Above this, most forward motion is skidding
    # rather than stepping, which is the failure the slip penalty is supposed to prevent.
    # Judged as a ratio, not an absolute: 0.5 m/s of slip is catastrophic at 0.6 m/s of
    # travel and unremarkable at 3 m/s. Requires two consecutive evaluations so a single
    # noisy sample cannot trigger it.
    SLIP_RATIO_CEILING = 0.55
    slip_strikes = 0

    seen_milestone = 0
    fired: set[str] = set()
    baseline_sps: float | None = None
    last_iteration = -1
    last_progress_time = time.time()

    # Seed the best-so-far and the milestone counter from metrics already on disk. Without
    # this, restarting the watchdog mid-run re-fires BEST on the first evaluation it sees
    # even when that score is worse than one already achieved, which is exactly the kind of
    # false alarm that teaches you to ignore your own alerts.
    existing = read_last_rows(metrics_path, count=100_000)
    prior_evals = [r["eval/episode_return"] for r in existing if "eval/episode_return" in r]
    best_eval = max(prior_evals) if prior_evals else -float("inf")
    if existing:
        seen_milestone = int(existing[-1].get("iteration", 0)) // args.milestone_every

    while True:
        time.sleep(args.poll)
        # Read a window comfortably larger than the evaluation interval. At the default
        # 40 rows a milestone landing more than 40 iterations after the last evaluation
        # silently omitted fall rate and slip, which are the two most useful numbers in it.
        rows = read_last_rows(metrics_path, count=150)

        if not rows:
            if not training_alive():
                emit("CRASH", "training process is gone and no metrics were ever written")
                return 1
            continue

        last = rows[-1]
        iteration = int(last.get("iteration", 0))
        env_steps = int(last.get("env_steps", 0))

        # --- progress and stall detection ---
        if iteration > last_iteration:
            last_iteration = iteration
            last_progress_time = time.time()
        else:
            stalled_min = (time.time() - last_progress_time) / 60.0
            if stalled_min > args.stall_minutes:
                alive = training_alive()
                emit(
                    "CRASH" if not alive else "STALLED",
                    f"no new iterations for {stalled_min:.0f} min at iteration {iteration:,}"
                    f" (process {'gone' if not alive else 'still running'})",
                )
                return 1

        # --- completion ---
        if not training_alive():
            emit(
                "DONE",
                f"training process exited at iteration {iteration:,}, {env_steps / 1e6:.1f}M "
                f"env steps, last return {last.get('episode_return', 0):.1f}",
            )
            return 0

        # --- throughput regression, which on a laptop means thermal throttling ---
        sps_values = [r["env_steps_per_sec"] for r in rows if "env_steps_per_sec" in r]
        if sps_values:
            recent = sum(sps_values) / len(sps_values)
            if baseline_sps is None and iteration > 20:
                baseline_sps = recent
            elif baseline_sps and recent < 0.70 * baseline_sps and "slow" not in fired:
                fired.add("slow")
                emit(
                    "SLOWDOWN",
                    f"throughput {recent:,.0f} steps/s is {100 * recent / baseline_sps:.0f}% of "
                    f"the {baseline_sps:,.0f} baseline, likely thermal throttling",
                )

        # --- the regression this run exists to avoid ---
        torso = [r["eval/torso_upright"] for r in rows if "eval/torso_upright" in r]
        if torso and torso[-1] < TORSO_FLOOR and "torso" not in fired:
            fired.add("torso")
            emit(
                "REGRESSION",
                f"eval/torso_upright fell to {torso[-1]:.2f} (floor {TORSO_FLOOR}) at iteration "
                f"{iteration:,}: the fold-forward gait may be returning",
            )

        # --- skating instead of stepping ---
        slips = [r["eval/foot_slip_speed"] for r in rows if "eval/foot_slip_speed" in r]
        speeds = [r["eval/mean_speed"] for r in rows if "eval/mean_speed" in r]
        if slips and speeds and speeds[-1] > 0.2:
            ratio = slips[-1] / speeds[-1]
            if ratio > SLIP_RATIO_CEILING:
                slip_strikes += 1
                if slip_strikes >= 2 and "slip" not in fired:
                    fired.add("slip")
                    emit(
                        "SKATING",
                        f"foot slip {slips[-1]:.2f} m/s against {speeds[-1]:.2f} m/s travel "
                        f"(ratio {ratio:.0%}) for two evaluations at iteration {iteration:,}: "
                        f"it is sliding rather than stepping, raise task.w_feet_slip",
                    )
            else:
                slip_strikes = 0

        # --- action saturation ---
        # The failure this exists for: the policy drives its mean action far outside
        # [-1, 1], the environment clips it, and exploration noise supplies the fine
        # control. Training keeps improving while the deterministic evaluation policy
        # degenerates into bang-bang. It is invisible in return and episode length, which
        # is exactly why it needs its own alarm.
        oob = [r["action_oob_frac"] for r in rows if "action_oob_frac" in r]
        if oob and oob[-1] > 0.20 and "oob" not in fired:
            fired.add("oob")
            mags = [r.get("action_mean_abs", 0) for r in rows if "action_mean_abs" in r]
            emit(
                "SATURATION",
                f"{oob[-1]:.0%} of policy mean actions are outside [-1,1] "
                f"(mean |action| {mags[-1] if mags else 0:.2f}) at iteration {iteration:,}: "
                f"the bounds loss is losing, eval will collapse before training does",
            )

        # --- motion tracking (Phase 3 Stage 1) ---
        # A different objective needs different alarms. Locomotion cares about falls and
        # slip; tracking cares whether the pose error is closing and whether episodes
        # survive the clip. Detected by the presence of the tracking eval metric rather
        # than by a flag, so one watchdog serves both runs.
        pose = [r["eval/pose_error_rad"] for r in rows if "eval/pose_error_rad" in r]
        if pose:
            lens = [r["eval/episode_length"] for r in rows if "eval/episode_length" in r]
            if lens and lens[-1] > 210 and "tracking_ok" not in fired:
                fired.add("tracking_ok")
                emit(
                    "TRACKING",
                    f"episodes now survive {lens[-1]:.0f} steps ({lens[-1] / 50:.1f}s of a 6s "
                    f"clip) with pose error {np.degrees(pose[-1]):.1f} deg at iteration "
                    f"{iteration:,}: tracking is working, worth a video",
                )
            # Pose error going the wrong way over a long stretch means the reward is
            # fighting itself, which is the failure Stage 1 exists to catch early.
            if len(pose) >= 6 and pose[-1] > 1.15 * min(pose[:-3]) and "pose_up" not in fired:
                fired.add("pose_up")
                emit(
                    "TRACKING_REGRESSION",
                    f"pose error rose to {np.degrees(pose[-1]):.1f} deg from a best of "
                    f"{np.degrees(min(pose[:-3])):.1f} deg at iteration {iteration:,}",
                )

        # --- adversarial motion prior health ---
        # Alarms on the CONSEQUENCE, not the proxy. An earlier version fired on
        # discriminator accuracy above 0.95, and fired three times on a run that video
        # confirmed was working: arms tucked in, posture upright, foot slip at zero, all at
        # accuracy 0.97-0.98. High accuracy is not failure. It often just means the policy
        # genuinely does not move like a person yet, which is true and useful information.
        #
        # What actually matters is whether the style reward still carries gradient. Below
        # about 0.10 the bounded LSGAN reward is saturated near zero and the prior has
        # stopped teaching anything. That is the condition worth waking someone for.
        styles = [r["amp/style_reward"] for r in rows if "amp/style_reward" in r]
        acc = [r["amp/accuracy"] for r in rows if "amp/accuracy" in r]
        if styles and iteration > 200 and styles[-1] < 0.10 and "disc" not in fired:
            fired.add("disc")
            emit(
                "DISCRIMINATOR",
                f"style reward collapsed to {styles[-1]:.3f} at iteration {iteration:,} "
                f"(accuracy {acc[-1] if acc else 0:.2f}): the motion prior has stopped "
                f"providing gradient. Lower amp.learning_rate or amp.n_epochs.",
            )

        # --- gait asymmetry ---
        # One leg driving while the other acts as a passive strut. Invisible to every other
        # metric here, all of which aggregate over both legs, and first caught by a human
        # watching a video at a left/right stance ratio of 1.90 (symmetry 0.52).
        sym = [r["eval/gait_symmetry"] for r in rows if "eval/gait_symmetry" in r]
        if sym and iteration > 200 and sym[-1] < 0.60 and "asym" not in fired:
            fired.add("asym")
            emit(
                "ASYMMETRY",
                f"gait symmetry {sym[-1]:.2f} at iteration {iteration:,} (1.0 is even): one "
                f"leg is doing most of the work. Enable ppo.symmetry_augment.",
            )

        # --- the brace ---
        # The failure mode that a symmetry metric alone cannot see. A humanoid that plants
        # both feet wide and never lifts either one is perfectly symmetric by any left/right
        # comparison, and a previous run scored 0.91 doing exactly that while travelling
        # 6 cm per foot strike. Real human walking sits at 0.20 to 0.25 double support and a
        # stance 0.10 to 0.15 m wide, so both of these catch it directly, and neither can be
        # satisfied by refusing to step.
        ds = [r["eval/double_support"] for r in rows if "eval/double_support" in r]
        sw = [r["eval/stance_width"] for r in rows if "eval/stance_width" in r]
        if len(ds) >= 2 and iteration > 100 and min(ds[-2:]) > 0.55 and "brace" not in fired:
            fired.add("brace")
            width = f", stance {sw[-1]:.2f} m wide" if sw else ""
            emit(
                "BRACE",
                f"double support {ds[-1]:.0%} for two evaluations at iteration {iteration:,}"
                f"{width}: both feet are planted almost always, so it is bracing and "
                f"sliding rather than stepping (real walking is 20-25%). Check a video "
                f"before trusting any symmetry number.",
            )
        if len(sw) >= 2 and iteration > 100 and min(sw[-2:]) > 0.35 and "splay" not in fired:
            fired.add("splay")
            emit(
                "SPLAY",
                f"stance {sw[-1]:.2f} m wide at iteration {iteration:,}: human walking is "
                f"0.10 to 0.15 m. It is standing splay-legged for stability.",
            )

        # --- rocking instead of walking ---
        # The failure a human caught on video and no metric here could see: a fixed split
        # stance, one foot permanently in front, both feet pattering while the body rocks
        # over them. Measured at 0.42 lead swaps per second against 7.4 foot strikes. In a
        # real walk every step trades the lead, so those two rates are equal.
        swaps = [r["eval/lead_swaps_per_sec"] for r in rows if "eval/lead_swaps_per_sec" in r]
        if len(swaps) >= 2 and iteration > 200 and max(swaps[-2:]) < 1.0 and "rocking" not in fired:
            fired.add("rocking")
            emit(
                "ROCKING",
                f"leading foot changed only {swaps[-1]:.2f} times/s for two evaluations at "
                f"iteration {iteration:,} (human walking: 1.6-2.0, one swap per step). It is "
                f"holding a split stance and rocking rather than alternating its legs. "
                f"Check task.w_gait_phase and gait_frequency.",
            )

        # --- pogoing ---
        bounce = [r["eval/vertical_bounce"] for r in rows if "eval/vertical_bounce" in r]
        if len(bounce) >= 2 and iteration > 200 and min(bounce[-2:]) > 0.08 and "bounce" not in fired:
            fired.add("bounce")
            emit(
                "BOUNCING",
                f"pelvis bouncing {bounce[-1] * 100:.0f} cm peak-to-peak at iteration "
                f"{iteration:,} (human walking: 4-5 cm). Raise the magnitude of "
                f"task.w_vertical_vel.",
            )

        # --- wandering ---
        head = [r["eval/heading_error_deg"] for r in rows if "eval/heading_error_deg" in r]
        if len(head) >= 2 and iteration > 200 and min(head[-2:]) > 20.0 and "wander" not in fired:
            fired.add("wander")
            emit(
                "WANDERING",
                f"heading error {head[-1]:.0f} deg for two evaluations at iteration "
                f"{iteration:,}: told to walk forward it is not holding a direction "
                f"(a person holds within a few degrees). Check task.w_heading.",
            )

        # --- the milestone that actually matters ---
        falls = [r["eval/fall_rate"] for r in rows if "eval/fall_rate" in r]
        if falls and falls[-1] < WALKING_FALL_RATE and "walking" not in fired:
            fired.add("walking")
            emit(
                "WALKING",
                f"eval fall rate dropped to {falls[-1]:.0%} at iteration {iteration:,} "
                f"({env_steps / 1e6:.0f}M steps): it is staying upright, worth a video",
            )

        # --- new best evaluation ---
        evals = [r["eval/episode_return"] for r in rows if "eval/episode_return" in r]
        if evals and evals[-1] > best_eval * 1.25 and evals[-1] > 100:
            best_eval = evals[-1]
            emit(
                "BEST",
                f"eval return {evals[-1]:.0f} at iteration {iteration:,} "
                f"({env_steps / 1e6:.0f}M steps)",
            )

        # --- periodic summary, so long quiet stretches still confirm liveness ---
        milestone = iteration // args.milestone_every
        if milestone > seen_milestone:
            seen_milestone = milestone
            parts = [
                f"iter {iteration:,}",
                f"{env_steps / 1e6:.0f}M steps",
                f"ret {last.get('episode_return', 0):.0f}",
                f"len {last.get('episode_length', 0):.0f}",
                f"{last.get('env_steps_per_sec', 0):,.0f} sps",
            ]
            if torso:
                parts.append(f"torso {torso[-1]:.2f}")
            if falls:
                parts.append(f"falls {falls[-1]:.0%}")
            if slips and speeds and speeds[-1] > 0.05:
                parts.append(f"slip {slips[-1] / speeds[-1]:.0%}")
            if pose:
                parts.append(f"pose {np.degrees(pose[-1]):.1f}d")
            if acc:
                parts.append(f"disc {acc[-1]:.2f}")
            if sym:
                parts.append(f"sym {sym[-1]:.2f}")
            styles = [r.get("amp/style_reward") for r in rows if "amp/style_reward" in r]
            if styles:
                parts.append(f"style {styles[-1]:.2f}")
            roots = [r["eval/root_error_m"] for r in rows if "eval/root_error_m" in r]
            if roots:
                parts.append(f"root {roots[-1] * 100:.0f}cm")
            emit("MILESTONE", "  ".join(parts))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        emit("STOPPED", "watchdog interrupted")
        sys.exit(0)
