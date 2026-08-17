"""Score a final-run segment against E50's pre-registered predictions. No judgement calls.

    python scripts/score_final_run.py --run runs/final-s0-...

Every threshold here is copied from LOGBOOK E50, which was written before the run started.
The point of this script is that the verdict is computed, not composed: the project's rule is
that a verdict is mandatory and one of five, and the failure mode it guards against is
reading the numbers in the light of what happened.

Where a prediction turns out to be badly SPECIFIED rather than merely failed, the script says
so in its own column instead of quietly reinterpreting it -- E49 is the entry where a
pre-registered response was overruled by arithmetic, and the lesson recorded there is that
pre-registration protects against fitting the story to the result but not against a wrong
model. Naming a bad bar is allowed. Moving it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
STEPS_PER_ITER = 245_760

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


def evals(run: Path) -> list[dict]:
    out = []
    for line in (run / "events.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval/torso_upright" in e:
            e["_steps"] = e["iteration"] * STEPS_PER_ITER
            out.append(e)
    return out


def metrics(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]


def verdict(ok: bool | None) -> str:
    if ok is None:
        return f"{YELLOW}N/A{RESET}"
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else REPO_ROOT / args.run

    ev, mt = evals(run), metrics(run)
    if not ev:
        print("no evaluations in this run")
        return 1
    last_iter = int(mt[-1]["iteration"])
    print(f"\n=== {run.name}: {last_iter:,} iterations, "
          f"{mt[-1]['env_steps'] / 1e9:.2f}B env steps, "
          f"{mt[-1].get('total_wall_hours', 0):.1f} h ===\n")
    print(f"{DIM}Thresholds are E50's, written before launch. Baselines are the envelope run "
          f"that produced best.pt.{RESET}\n")

    rows: list[tuple[str, bool | None, str]] = []

    # P1 -- a GUARD, and E50 says so in advance: the ceiling was not expected to bind.
    std = np.array([m["action_std"] for m in mt])
    rows.append((
        "P1 exploration guard: action_std never exceeds 0.501",
        bool(std.max() <= 0.501),
        f"max {std.max():.4f}, final {std[-1]:.4f} "
        f"(a breach would mean the clamp is not running -- E05)",
    ))

    # P2 -- PRIMARY. Slope of torso_upright from 200M env steps to the end.
    w = [e for e in ev if e["_steps"] >= 200e6]
    x = np.array([e["_steps"] for e in w])
    y = np.array([e["eval/torso_upright"] for e in w])
    slope = float(np.polyfit(x, y, 1)[0] * 1e9)
    # BOTH clauses, reported separately, because they can disagree and this one does. The
    # slope clause asks "did the decay stop"; the endpoint clause asks "did it end at least
    # where it started". A run that trades posture for speed early and then holds flat passes
    # the first and fails the second, and quoting only the slope would be reporting the half
    # of my own prediction that happened to be met.
    rows.append((
        "P2 PRIMARY: torso_upright slope >= -0.10 per 1B, AND last >= first",
        bool(slope >= -0.10 and y[-1] >= y[0]),
        f"slope {slope:+.3f}/1B [{'pass' if slope >= -0.10 else 'FAIL'}] over {len(w)} evals; "
        f"endpoints {y[0]:.3f} -> {y[-1]:.3f} "
        f"[{'pass' if y[-1] >= y[0] else 'FAIL'}] (baseline -0.659, 0.961 -> 0.543)",
    ))

    # P3 -- the deliverable, a simultaneous conjunction.
    hits = [e for e in ev
            if e["eval/torso_upright"] >= 0.85
            and e["eval/fall_rate"] <= 0.15
            and e["eval/mean_speed"] >= 0.75]
    best_up = max((e for e in ev if e["eval/fall_rate"] <= 0.15),
                  key=lambda e: e["eval/torso_upright"], default=None)
    fast = max(ev, key=lambda e: e["eval/mean_speed"])
    rows.append((
        "P3 DELIVERABLE: some eval with upright>=0.85 AND falls<=0.15 AND speed>=0.75",
        bool(hits),
        (f"{len(hits)} such evals" if hits else
         f"none. Best upright at falls<=0.15: {best_up['eval/torso_upright']:.3f} "
         f"(speed {best_up['eval/mean_speed']:.3f}); fastest eval {fast['eval/mean_speed']:.3f} m/s"),
    ))

    # P4 -- speed and tracking at the end.
    tail = ev[-5:]
    sp = float(np.mean([e["eval/mean_speed"] for e in tail]))
    ratio = float(np.mean([e["eval/mean_speed"] / max(e.get("eval/commanded_speed", 1e-9), 1e-9)
                           for e in tail]))
    rows.append((
        "P4: mean_speed >= 0.85 m/s and speed/commanded >= 0.80",
        bool(sp >= 0.85 and ratio >= 0.80),
        f"last-5 mean speed {sp:.3f} m/s, tracking ratio {ratio:.3f} "
        f"(baseline 0.589 m/s at ratio 0.737)",
    ))

    # P5 -- human-likeness WITH the speed floor E50 added after the vacuity was caught.
    try:
        from humanoid_rl.gait_score import score_row

        scored = []
        for e in ev:
            try:
                s = score_row(e)
                scored.append((getattr(s, "overall", float("nan")), e))
            except Exception:  # noqa: BLE001
                pass
        elig = [(s, e) for s, e in scored
                if e["eval/fall_rate"] <= 0.30 and e["eval/mean_speed"] >= 0.60]
        best = max(elig, default=None, key=lambda t: t[0])
        rows.append((
            "P5: gait_score >= 0.40 at falls<=0.30 AND speed>=0.60",
            bool(best and best[0] >= 0.40),
            (f"best qualifying {best[0]:.3f} at speed {best[1]['eval/mean_speed']:.3f}, "
             f"falls {best[1]['eval/fall_rate']:.3f}" if best else
             "no eval met the speed+falls floor")
            + f" (best.pt scores 0.21; the speed floor exists because 0.4418 was already "
              f"reached by a 0.319 m/s crawl)",
        ))
    except Exception as exc:  # noqa: BLE001
        rows.append(("P5: gait_score >= 0.40 at falls<=0.30 AND speed>=0.60", None,
                     f"could not score: {exc}"))

    # P6 / P7 need separate tools; report them as pending rather than silently omitting.
    rows.append(("P6: amp_readiness gates 1-3", None,
                 "run scripts/amp_readiness.py on the selected checkpoint"))
    rows.append(("P7: torso tilt < 30 deg and backward share < 90%", None,
                 "run scripts/posture_score.py --command 1.0 (baseline 48.87 deg, 100% back)"))

    # P8 -- needs both segments.
    rows.append(("P8 seed hedge: best/worst episode_length ratio < 3.0 across seeds", None,
                 "needs segment 2; E22b measured 5.9x on a byte-identical pair"))

    for name, ok, detail in rows:
        print(f"  [{verdict(ok)}] {name}")
        print(f"         {DIM}{detail}{RESET}")

    scored_now = [r for r in rows if r[1] is not None]
    passed = sum(1 for r in scored_now if r[1])
    print(f"\n  {passed} of {len(scored_now)} scorable predictions passed "
          f"({len(rows) - len(scored_now)} pending).")
    print(f"\n{DIM}A verdict is mandatory and one of five: WORKED, NO EFFECT, WORSE, "
          f"INVALID, MIXED. Write it into docs/LOGBOOK.md with the numbers above.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
