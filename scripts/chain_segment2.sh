#!/bin/bash
# Wait for segment 1 to finish, then launch segment 2. Nothing else.
#
# Segment 2 is a SEED REPLICATE, not a retry: configs/final_s1.yaml differs from
# configs/final.yaml in run.name and run.seed and nothing else, verified by diff before
# launch. E22b measured two byte-identical configs, both seed 0, diverging at iteration 3
# into a 7.6x outcome gap, so one draw is a lottery ticket.
#
# The guard below is the reason this is a script and not a one-liner. If segment 1 died
# early rather than finishing, launching a second 15-hour run on the same code is the wrong
# response -- something systemic broke and it needs a person. Auto-chaining is for the case
# where segment 1 ran to completion.
set -u
cd "$(dirname "$0")/.."

PID=${1:?usage: chain_segment2.sh <pid-of-segment-1> <run-dir>}
RUN=${2:?usage: chain_segment2.sh <pid-of-segment-1> <run-dir>}
MIN_ITERS=11000          # of 11,393; anything less means it did not finish its budget

echo "chain: waiting for segment 1 (pid $PID)"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "chain: segment 1 process exited"

LAST=$(.venv/bin/python - "$RUN" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/metrics.jsonl") if l.strip()]
print(int(rows[-1]["iteration"]) if rows else 0)
PY
)
echo "chain: segment 1 reached iteration $LAST"

if [ "$LAST" -lt "$MIN_ITERS" ]; then
  echo "chain: REFUSING to launch segment 2. Segment 1 stopped at $LAST of 11,393"
  echo "chain: that is not a completed run, and a second seed would not tell you why."
  exit 1
fi

echo "chain: launching segment 2 (seed 1)"
PYTHONUNBUFFERED=1 nohup caffeinate -dimsu .venv/bin/python -u -m humanoid_rl.train \
  --config configs/final_s1.yaml > /tmp/final_s1.log 2>&1 &
echo "chain: segment 2 pid $!"
