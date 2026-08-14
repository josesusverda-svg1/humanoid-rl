"""The supervisor. Checks a setup before it burns compute, and a run while it burns it.

    python scripts/oracle.py --config configs/amp.yaml        # before launching
    python scripts/oracle.py --run runs/<run>                 # a live or finished run
    python scripts/oracle.py --run runs/<run> --watch         # keep checking

Why this exists. Every serious defect in this project was found by a human watching a video
and asking a blunt question ("is he actually walking forward?", "why does he bounce?"),
hours after the run containing it began. The metrics all looked fine, because the metrics
were derived from the same assumptions as the reward. A checker that only reads our own
numbers would have agreed with them.

So the checks here are built on things specified INDEPENDENTLY of the reward: measured human
walking, the reference mocap, and internal contradictions between settings that were written
in different places by different reasoning. That independence is the entire value. A check
that compares the config to itself is a rubber stamp.

Exit code is 1 if anything CONTRADICTION or UNREACHABLE was found, so this can gate a launch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.oracle import invariants  # noqa: E402
from humanoid_rl.oracle.invariants import Severity  # noqa: E402

COLOUR = {
    Severity.CONTRADICTION: "\033[91m",
    Severity.UNREACHABLE: "\033[93m",
    Severity.SUSPECT: "\033[96m",
    Severity.OK: "\033[92m",
}
RESET = "\033[0m"


def report(findings, *, verbose: bool) -> int:
    counts = {s: 0 for s in Severity}
    for f in findings:
        counts[f.severity] += 1

    print()
    for f in findings:
        if f.severity is Severity.OK and not verbose:
            continue
        print(f"{COLOUR[f.severity]}{f.line()}{RESET}")
        if f.remedy:
            print(f"     fix: {f.remedy}")
        if f.caught_before:
            print(f"     this check exists because: {f.caught_before}")
        print()

    blocking = counts[Severity.CONTRADICTION] + counts[Severity.UNREACHABLE]
    summary = (f"{counts[Severity.CONTRADICTION]} contradictions, "
               f"{counts[Severity.UNREACHABLE]} unreachable, "
               f"{counts[Severity.SUSPECT]} suspect, {counts[Severity.OK]} ok")
    print(summary)
    if blocking:
        print("\nThese are setup problems. Training cannot fix them, and every hour spent "
              "running is spent on the wrong objective.")
    elif counts[Severity.SUSPECT]:
        print("\nNothing contradictory. The suspect items are worth a glance before launch.")
    else:
        print("\nNo contradictions found. That is not proof the objective is right, only "
              "that its parts agree with each other.")
    return 1 if blocking else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="check a config before launching")
    ap.add_argument("--run", type=Path, help="check a run (uses its snapshotted config)")
    ap.add_argument("--watch", action="store_true", help="re-check periodically")
    ap.add_argument("--poll", type=float, default=300.0)
    ap.add_argument("--verbose", action="store_true", help="show passing checks too")
    args = ap.parse_args()

    if not args.config and not args.run:
        ap.error("pass --config or --run")

    path = args.config or (args.run / "config.yaml")
    if not path.exists():
        print(f"no config at {path}")
        return 2

    while True:
        config = Config.load(path)
        label = args.run.name if args.run else path.name
        print(f"\n=== oracle: {label} ===")
        # A warm-started run inherits weights that the config cannot describe, so the
        # checkpoint is checked too whenever there is one.
        ckpt = (args.run / "checkpoints" / "best.pt") if args.run else None
        code = report(invariants.run_all(config, ckpt), verbose=args.verbose)
        if not args.watch:
            return code
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
