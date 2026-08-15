#!/bin/zsh
# One full look at the get-up run: progress, every metric, the 13 standing conjuncts, and a
# rendered strip to actually look at.
#
# Written because in this project the metrics have never been enough on their own. The
# headstand showed as an odd pair of numbers and was obvious in a picture. The one-armed prop
# was described by eye before any metric moved. So this prints the numbers AND renders the
# frames, and the frames are the point.
#
#   scripts/getup_watch.sh            # newest get-up run
#   scripts/getup_watch.sh <run-dir>
#
# Exits non-zero if no run is training, so a caller can tell "finished" from "still going".

cd /Users/BrickLayer/Desktop/HumonoidAI || exit 2
RUN=${1:-$(ls -td runs/getup-* 2>/dev/null | head -1)}
[ -z "$RUN" ] && { echo "no get-up run found"; exit 2; }

echo "=============================================================="
echo "  $RUN    $(date +%H:%M:%S)"
echo "=============================================================="
tail -1 /tmp/getup*.log 2>/dev/null | tail -1

.venv/bin/python - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
rows = []
for line in open(f"{run}/metrics.jsonl", errors="ignore"):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if "eval/episode_return" in r:
        rows.append(r)
if not rows:
    print("no evaluations yet"); raise SystemExit(0)

g = lambda r, k: r.get(k, 0.0) or 0.0
print(f"\n{'iter':>6}{'knee':>7}{'pelvis':>8}{'head':>7}{'feet':>7}{'hands_dn':>10}"
      f"{'spin':>7}{'stand%':>8}{'held%':>7}{'return':>8}")
for r in rows[-10:]:
    print(f"{r['iteration']:>6.0f}{g(r,'eval/knee_max'):>7.2f}{g(r,'eval/root_height'):>8.3f}"
          f"{g(r,'eval/head_height_ratio'):>7.2f}{g(r,'eval/foot_load_bw'):>7.2f}"
          f"{g(r,'eval/hands_down_frac'):>9.0%}{g(r,'eval/spin_deg_s'):>7.0f}"
          f"{g(r,'eval/standing_frac'):>7.1%}{g(r,'eval/held_ever_frac'):>7.1%}"
          f"{g(r,'eval/episode_return'):>8.0f}")

first, last = rows[0], rows[-1]
print(f"\nsince the first eval:  return {g(first,'eval/episode_return'):.0f} -> "
      f"{g(last,'eval/episode_return'):.0f}   pelvis {g(first,'eval/root_height'):.3f} -> "
      f"{g(last,'eval/root_height'):.3f}   feet {g(first,'eval/foot_load_bw'):.2f} -> "
      f"{g(last,'eval/foot_load_bw'):.2f}")
PY

# The expensive half (conjuncts + rendering) runs at most every RENDER_EVERY seconds, so a
# frequent liveness ping stays cheap. Rendering takes ~40 s and pinning it to every call would
# make a one-minute cadence impossible.
RENDER_EVERY=${RENDER_EVERY:-540}
STAMP=/tmp/getup_watch_last
NOW=$(date +%s)
LAST=$(cat $STAMP 2>/dev/null || echo 0)
if [ $((NOW - LAST)) -lt $RENDER_EVERY ]; then
  echo "\n(next frames in $(( (RENDER_EVERY - NOW + LAST) / 60 ))m; run with RENDER_EVERY=0 to force)"
  pgrep -f "humanoid_rl.train" >/dev/null || { echo "\nTRAINING HAS FINISHED"; exit 1; }
  exit 0
fi
echo $NOW > $STAMP

echo "\n--- the 13 standing conjuncts, worst first (this is where it is stuck) ---"
.venv/bin/python scripts/getup_conjuncts.py --run "$RUN" 2>/dev/null | tail -14

echo "\n--- rendering frames ---"
.venv/bin/python scripts/getup_snapshot.py --run "$RUN" --out /tmp/getup_watch.png 2>&1 | tail -8

pgrep -f "humanoid_rl.train" >/dev/null || { echo "\nTRAINING HAS FINISHED"; exit 1; }
exit 0
