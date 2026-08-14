# Overnight report

You slept at ~23:30 after twelve hours. Here is what happened, in the order it matters.

## The stance-width hypothesis was tested and is HALF confirmed

Added the missing maximum side of XBot's foot-distance band (we had ported only the
minimum), sized at -3.0 against a ~6 positive budget because at the ported -1.0 a 0.67 m
brace cost 3.7% of budget, which a policy trading it for stability pays forever.

| | before | after | human |
|---|---|---|---|
| stance width | 0.67 m | **0.34-0.40 m** | 0.10-0.15 |
| stride length | 0.11 m | **0.11 m** | 0.6-0.8 |
| step rate | 3.78/s | 2.91/s | 1.6-2.0 |
| best falls | 2% | 6% | 0 |

**The term worked. The shuffle did not go away.** So splaying was not the cause of the
shuffling, which is a real result: one hypothesis eliminated by measurement.

## Then arithmetic found the actual cause, and it is ours

Stride length is not a free variable. It equals speed / strike-rate. So:

    gait_frequency_range = 1.0-2.0 Hz  ->  2.0-4.0 foot strikes/s
    human walking                      ->  1.6-2.0 foot strikes/s
    measured policy                    ->  2.91 strikes/s  (it was OBEYING us)

At 0.32 m/s and 2.91 strikes/s, an 0.11 m stride is forced. **We were commanding the
shuffle we spent the day trying to tune away.**

The range came in with last night's conformance pass: Booster T1 samples 1.0-2.0 Hz, and I
adopted it without checking whether their "frequency" counts strides or steps on a robot
with a different leg length. The nominal anchor (0.9 Hz = 1.8 strikes/s) was always correct;
only the sampled band was wrong.

The Oracle missed it because it checked the anchor and never the range. **Fixed**: it now
flags any band whose 2*f falls outside 1.6-2.0, and it fires as a CONTRADICTION.

## What is running now

One run, one variable: `gait_frequency_range` 1.0-2.0 -> **0.7-1.1 Hz** (1.4-2.2 strikes/s).
Nothing else changed. Launched ~00:10, ETA ~2.6 h, Oracle clean, watchers armed.

This was the single further run I permitted myself. Whatever it produces, no more changes
were made while you slept.

## How to judge it when you wake

Ignore return and episode length. Look at `scripts/gait_report.py` on the final checkpoint:

    stride_length   must move off 0.11 m toward 0.6-0.8
    step_rate       should land near 1.6-2.0
    stance_width    should stay at or below ~0.40 (the fix from the previous run)

If stride length moves and the video shows real steps, the cadence was the last blocker.
If stride length is STILL 0.11 m, then cadence was not it either, and the next suspect is
the speed itself: at 0.32 m/s nothing produces a human stride, so the question becomes why
the policy will not go faster when commanded to.

Videos: /tmp/hviz/stance_fixed.mp4 (previous run, stance fixed, still shuffling).
