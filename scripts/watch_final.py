"""Watch the final run against E50's pre-registered kill criteria, and nothing else.

    python scripts/watch_final.py --run runs/final-s0-...

One line per event, silence while healthy. The thresholds are E50's, written before launch,
and this script does not soften them: it prints the rule, the measured value, and whether it
fired. It never kills anything itself, because "the rule fired" and "I decided to stop" are
different acts and the logbook needs to be able to tell them apart (E30 recorded a rule
firing for a reason that turned out to be wrong, and that was only legible because the rule
and the reason were separate).

Silence has to mean healthy, never "died twenty minutes ago", so the process check runs on
every poll and the crash line is emitted even when no metric moved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

STEPS_PER_ITER = 245_760

# E50's ladder. (label, env-step trigger, predicate over the rows so far, description)
K1_STEPS = 100_000_000
K2_STEPS = 500_000_000
K3_STEPS = 1_000_000_000


def rows(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially flushed final line is normal while the trainer is writing.
                continue
    return out


def evals(run: Path) -> list[dict]:
    """Evaluation events, which carry the metrics the predictions are stated in."""
    out = []
    for r in rows(run / "events.jsonl"):
        if r.get("event") == "eval" or "eval/torso_upright" in r:
            out.append(r)
    return out


def med(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=120.0)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    metrics = run / "metrics.jsonl"
    fired: set[str] = set()
    last_iter = -1

    print(f"WATCHING {run.name} against E50's pre-registered gates", flush=True)

    while True:
        m = rows(metrics)
        if not m:
            time.sleep(args.poll)
            continue
        cur = m[-1]
        it = int(cur.get("iteration", 0))
        steps = int(cur.get("env_steps", 0))

        # --- the instrument check, every poll, because it invalidates everything else -----
        std = float(cur.get("action_std", 0.0))
        if std > 0.501 and "instrument" not in fired:
            fired.add("instrument")
            print(f"KILL instrument: action_std {std:.3f} > 0.501 at iter {it}. The clamp is "
                  f"not running or the config did not load (E05). Stop; do not debug live.",
                  flush=True)

        # --- K1, cold-start calibrated -------------------------------------------------
        if steps >= K1_STEPS and "K1" not in fired:
            fired.add("K1")
            eplen = float(cur.get("episode_length", 0.0))
            verdict = "FIRED" if eplen < 300 else "passed"
            print(f"K1 {verdict}: episode_length {eplen:.0f} vs 300 at {steps/1e6:.0f}M steps "
                  f"(cold runs read 471.1 and 377.7 here)", flush=True)

        # --- K2 --------------------------------------------------------------------------
        if steps >= K2_STEPS and "K2" not in fired:
            fired.add("K2")
            ev = evals(run)
            ret = med([float(e.get("eval/episode_return", e.get("episode_return", 0.0)))
                       for e in ev[-5:]]) if ev else 0.0
            hit = std < 0.10 and ret < 1500
            print(f"K2 {'FIRED' if hit else 'passed'}: action_std {std:.3f} (<0.10?) and "
                  f"median eval return {ret:.0f} (<1500?) at {steps/1e6:.0f}M. "
                  f"{'Relaunch this seed with entropy_coef 0.005.' if hit else ''}",
                  flush=True)

        # --- K3, conjunctive on purpose ---------------------------------------------------
        if steps >= K3_STEPS and "K3" not in fired:
            fired.add("K3")
            ev = evals(run)[-10:]
            falls = med([float(e.get("eval/fall_rate", 0.0)) for e in ev]) if ev else 0.0
            up = med([float(e.get("eval/torso_upright", 1.0)) for e in ev]) if ev else 1.0
            hit = falls > 0.60 and up < 0.60
            print(f"K3 {'FIRED' if hit else 'passed'}: fall_rate {falls:.3f} (>0.60?) AND "
                  f"torso_upright {up:.3f} (<0.60?) at {steps/1e9:.1f}B. Both required; "
                  f"fall rate alone fires on the baseline's own oscillation.", flush=True)

        # --- the primary prediction, reported the first time it is met -------------------
        if "P3" not in fired:
            for e in evals(run):
                up = float(e.get("eval/torso_upright", 0.0))
                fr = float(e.get("eval/fall_rate", 1.0))
                sp = float(e.get("eval/mean_speed", 0.0))
                if up >= 0.85 and fr <= 0.15 and sp >= 0.75:
                    fired.add("P3")
                    print(f"PREDICTION 3 MET at iter {e.get('iteration')}: torso_upright "
                          f"{up:.3f}, falls {fr:.3f}, speed {sp:.3f}. Never met before in "
                          f"this project.", flush=True)
                    break

        # --- liveness. Silence must not be able to mean "dead" ---------------------------
        alive = subprocess.run(["pgrep", "-f", "humanoid_rl.train"],
                               capture_output=True).returncode == 0
        if not alive:
            done = it >= 11_393
            print(f"{'FINISHED' if done else 'PROCESS GONE'} at iter {it}, "
                  f"{steps/1e9:.2f}B steps", flush=True)
            return 0
        if it > last_iter:
            last_iter = it
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
