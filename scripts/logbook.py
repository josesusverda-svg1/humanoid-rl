"""Read and write docs/LOGBOOK.md without the tedium that makes logbooks get skipped.

A logbook only pays off if it is cheap to consult and cheap to append to. Two commands:

    python scripts/logbook.py --about swing_height   # has this been tried? what happened?
    python scripts/logbook.py --settled              # everything already ruled out
    python scripts/logbook.py --run runs/<dir>       # entry skeleton, numbers pre-filled

The `--run` mode matters most. Filling a metrics table by hand is exactly the friction that
turns a logbook into an empty file, and hand-copied numbers are how a table ends up
disagreeing with the run it describes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGBOOK = REPO_ROOT / "docs" / "LOGBOOK.md"

#: The fixed metric block every entry carries, so entries are comparable by scanning a
#: column rather than by reading prose. Human reference values are measured, not chosen:
#: see docs/research/ and humanoid_rl/gait_score.py for where each comes from.
HEADLINE = [
    ("eval/fall_rate", "falls", "%", "0", "lower is better"),
    ("eval/episode_length", "episode", "steps", "", "vs the limit in config"),
    ("eval/mean_speed", "speed", "m/s", "1.2-1.4", "human free walking"),
    ("eval/commanded_speed", "commanded", "m/s", "", "what it was asked for"),
    ("eval/torso_upright", "torso upright", "", "0.95-1.00", "leaning shows up here"),
    ("eval/stance_width", "stance width", "m", "0.10-0.15", ""),
    ("eval/lead_swaps_per_sec", "lead swaps", "/s", "1.6-2.0", "equals step rate in a real walk"),
    ("eval/double_support", "double support", "", "0.20-0.25", "1.0 means never lifting a foot"),
    ("eval/difficulty_median", "curriculum level", "", "", "1.0 is the full envelope"),
]

DIRECTIONS = [("fwd", "forward"), ("back", "backward"), ("side", "sideways"), ("turn", "turning")]


def evaluations(run: Path) -> list[dict]:
    rows = []
    path = run / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"no metrics.jsonl in {run}")
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval/episode_return" in row:
            rows.append(row)
    return rows


def skeleton(run: Path) -> str:
    rows = evaluations(run)
    if not rows:
        raise SystemExit(f"{run.name} has no evaluations yet")
    best = max(rows, key=lambda r: r["eval/episode_return"])
    last = rows[-1]

    out = [f"### E??  <date>  <what changed in one line>",
           f"- **Change**: ",
           f"- **Why**: ",
           f"- **Prediction** (written before the run): ",
           f"- **Run**: `{run.name}`, {len(rows)} evaluations, "
           f"best at iteration {best['iteration']:.0f}",
           f"- **Verdict**: WORKED | NO EFFECT | WORSE | INVALID | MIXED",
           f"- **Learned**: ",
           "",
           "| metric | best eval | final eval | reference |",
           "|---|---|---|---|"]

    for key, label, unit, ref, note in HEADLINE:
        if key not in best and key not in last:
            continue
        fmt = (lambda v: f"{v:.0%}") if key == "eval/fall_rate" else (lambda v: f"{v:.2f}")
        b = fmt(best[key]) if key in best else "--"
        l = fmt(last[key]) if key in last else "--"
        suffix = f" {unit}" if unit and unit != "%" else ""
        out.append(f"| {label} | {b}{suffix} | {l}{suffix} | {ref or '--'}{' (' + note + ')' if note else ''} |")

    # Per-direction tracking, present only on runs from 2026-08-14 onward.
    have_dir = any(f"eval/cmd_{k}_share" in best for k, _ in DIRECTIONS)
    if have_dir:
        out += ["", "| direction | asked | delivered | follows | practised |", "|---|---|---|---|---|"]
        for key, label in DIRECTIONS:
            share = best.get(f"eval/cmd_{key}_share", 0.0)
            target = best.get(f"eval/cmd_{key}_target", 0.0)
            actual = best.get(f"eval/cmd_{key}_actual", 0.0)
            if share <= 0:
                out.append(f"| {label} | not commanded | -- | -- | 0% |")
                continue
            out.append(f"| {label} | {target / share:.2f} | {actual / share:.2f} | "
                       f"{actual / target:.0%} | {share:.0%} |")
    else:
        out += ["", "_Per-direction metrics unavailable: this run predates them._"]

    out += ["", f"- **Also run**: `python scripts/gait_report.py --run {run} --checkpoint best.pt`",
            f"- **AMP gate**: `python scripts/amp_readiness.py --run {run} --checkpoint best.pt`"]
    return "\n".join(out)


def search(term: str) -> str:
    """Every entry, settled row and bug row mentioning `term`.

    Substring, case-insensitive, deliberately dumb. A clever search that misses one entry is
    worse than a dumb one that shows three irrelevant ones, because the whole point is not
    to repeat something already tried.
    """
    if not LOGBOOK.exists():
        return "no logbook yet"
    text = LOGBOOK.read_text()
    needle = term.lower()

    hits: list[str] = []
    # Table rows (settled / bugs) are single lines.
    for line in text.splitlines():
        if line.startswith("|") and needle in line.lower() and "---" not in line:
            hits.append(line.strip())
    # Entries are blocks starting with "### E".
    for block in re.split(r"\n(?=### E)", text):
        if block.startswith("### E") and needle in block.lower():
            hits.append("\n" + block.strip())

    if not hits:
        return (f"nothing about {term!r} in the logbook.\n"
                f"That means it is untried, NOT that it is a good idea.")
    return f"{len(hits)} mention(s) of {term!r}:\n\n" + "\n".join(hits)


def settled() -> str:
    if not LOGBOOK.exists():
        return "no logbook yet"
    text = LOGBOOK.read_text()
    start = text.find("## Settled")
    end = text.find("## Instrumentation bugs")
    if start < 0 or end < 0:
        return "logbook is missing its Settled section"
    return text[start:end].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, help="generate an entry skeleton from a run dir")
    ap.add_argument("--about", type=str, help="search the logbook before changing something")
    ap.add_argument("--settled", action="store_true", help="print everything already ruled out")
    args = ap.parse_args()

    if args.about:
        print(search(args.about))
    elif args.settled:
        print(settled())
    elif args.run:
        print(skeleton(args.run))
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
