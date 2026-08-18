"""Watch the terrain run against E55's pre-registered gates, and nothing else.

    python scripts/watch_terrain.py --run runs/terrain-warm-...

E55's gates differ from E50's, which is why this is a separate script rather than a flag on
scripts/watch_final.py: a warm start onto changed ground fails in different ways than a cold
start on flat ground, and a watcher whose thresholds were written for the other run would go
quiet through exactly the failure it was pointed at.

Silence must mean healthy, never "died an hour ago", so liveness is checked every poll and the
process-gone line fires even when no metric moved.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZERO_SHOT_ROUGH = 0.312      # E57: re-measured on the 14 cm field (was 0.078 at 5.25 cm)
ZERO_SHOT_FLAT = 0.000


def rows(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=180.0)
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else REPO_ROOT / a.run
    fired: set[str] = set()
    print(f"WATCHING {run.name} against E55's gates "
          f"(zero-shot baseline: rough {ZERO_SHOT_ROUGH:.3f}, flat {ZERO_SHOT_FLAT:.3f})",
          flush=True)

    while True:
        m = rows(run / "metrics.jsonl")
        ev = [e for e in rows(run / "events.jsonl") if "eval/fall_rate" in e]
        if m:
            cur = m[-1]
            steps = int(cur.get("env_steps", 0))

            # K2 -- the critic-warmup check. E38 measured approx_kl 24.45 on a warm start.
            kl = max((float(r.get("approx_kl", 0.0)) for r in m), default=0.0)
            if kl > 1.0 and "K2" not in fired:
                fired.add("K2")
                print(f"KILL K2: approx_kl reached {kl:.2f} (>1.0). The warm start is being "
                      f"destroyed, not adapted -- E38's failure. Stop.", flush=True)

            if steps >= 100e6 and "K1" not in fired and ev:
                fired.add("K1")
                fr = ev[-1]["eval/fall_rate"]
                print(f"K1 {'FIRED' if fr > 0.75 else 'passed'}: rough fall_rate {fr:.3f} "
                      f"vs 0.75 at {steps/1e6:.0f}M (started at {ZERO_SHOT_ROUGH:.3f})",
                      flush=True)
                print(f"     also: approx_kl max so far {kl:.4f} -- K2 wants it under 1.0",
                      flush=True)

            if steps >= 500e6 and "K3" not in fired and len(ev) >= 4:
                fired.add("K3")
                rough = med([e["eval/fall_rate"] for e in ev[-4:]])
                flat = med([e.get("eval_flat/fall_rate", 0.0) for e in ev[-4:]])
                hit = flat > 0.30 and rough < ZERO_SHOT_ROUGH
                print(f"K3 {'FIRED' if hit else 'passed'}: flat fall_rate {flat:.3f} (>0.30?) "
                      f"while rough {rough:.3f} improves. That trade is prediction 2 failing.",
                      flush=True)

            # Prediction 1, reported the first time it is met on a rough eval.
            if "P1" not in fired:
                for e in ev:
                    if e["eval/fall_rate"] <= 0.12:
                        fired.add("P1")
                        print(f"PREDICTION 1 MET at iter {e.get('iteration')}: rough "
                              f"fall_rate {e['eval/fall_rate']:.3f} <= 0.120, from a "
                              f"zero-shot {ZERO_SHOT_ROUGH:.3f}", flush=True)
                        break

        alive = subprocess.run(["pgrep", "-f", "humanoid_rl.train"],
                               capture_output=True).returncode == 0
        if not alive:
            it = int(m[-1]["iteration"]) if m else 0
            print(f"{'FINISHED' if it >= 11393 else 'PROCESS GONE'} at iter {it}", flush=True)
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
