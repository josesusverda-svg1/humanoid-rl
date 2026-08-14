"""Wait for the heading error to come down, and say so once it does.

Narrow on purpose. `watch_training.py` fires when something goes WRONG; this exits when one
specific thing goes RIGHT, so it can be waited on rather than polled by hand.

Heading error is the number that decides whether "walk forward" means walking forward. The
policy that prompted the fix drifted 47 degrees in 11 seconds and finished 2.6 m off its
line; a person holds within a few degrees. The run being watched starts around 85 degrees
because it is relearning against a far larger command distribution with the heading term and
its observation newly added.

Exits 0 when the milestone is reached, 1 if the run ends or stalls first, so the caller can
tell "it worked" from "it stopped".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval/heading_error_deg" in row:
            rows.append(row)
    return rows


def alive() -> bool:
    out = subprocess.run(["pgrep", "-f", "humanoid_rl.train"], capture_output=True, text=True)
    return bool(out.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--target", type=float, default=30.0,
                    help="degrees; report once two consecutive evaluations are below this")
    ap.add_argument("--poll", type=float, default=120.0)
    args = ap.parse_args()

    run_dir = args.run
    if run_dir is None:
        runs = [p for p in (REPO_ROOT / "runs").iterdir()
                if p.is_dir() and (p / "metrics.jsonl").exists()
                and not p.name.startswith("ABANDONED")]
        run_dir = max(runs, key=lambda p: (p / "metrics.jsonl").stat().st_mtime)

    metrics = run_dir / "metrics.jsonl"
    print(f"watching {run_dir.name} for heading error below {args.target:.0f} deg", flush=True)
    seen = 0
    best = float("inf")

    while True:
        rows = evaluations(metrics)
        if len(rows) > seen:
            for row in rows[seen:]:
                deg = row["eval/heading_error_deg"]
                best = min(best, deg)
                print(f"  iter {row['iteration']:>5}  heading {deg:5.1f} deg   "
                      f"falls {row.get('eval/fall_rate', 0):.0%}   "
                      f"lead {row.get('eval/lead_swaps_per_sec', 0):.2f}/s   "
                      f"return {row.get('eval/episode_return', 0):.0f}", flush=True)
            seen = len(rows)

        # Two consecutive, so a single lucky evaluation does not call it.
        recent = [r["eval/heading_error_deg"] for r in rows[-2:]]
        if len(recent) == 2 and max(recent) < args.target:
            print(f"\nHEADING CAME DOWN: {recent[0]:.1f} then {recent[1]:.1f} deg, "
                  f"both under {args.target:.0f}. Best so far {best:.1f}.", flush=True)
            return 0

        if not alive():
            print(f"\nrun ended before the heading milestone. Best was {best:.1f} deg.",
                  flush=True)
            return 1
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
