# Every eval number in this project was biased, and it chose the checkpoints

Date: 2026-08-14. Found while reading the first twelve evals of the envelope run, because
`eval/fall_rate` said 100% while a held-command probe on the same policy said 3%.

## The bug

`humanoid_rl/evaluate.py` ran the policy until `num_episodes` (32) episodes had completed,
counting whichever completed first, across `eval.num_envs` (64) environments with autoreset.

The first episodes to complete are the short ones. Short means fell. An environment that
survives to the 1000-step limit contributes nothing unless enough of its neighbours survive
too, because the loop hits its quota of 32 on falls alone and stops before a single survivor
is counted.

So the metric had two modes and nothing between them. Over 101 evals of the cadence run:

```
fall_rate 1.00, 32-34 episodes counted   ->  85 evals
fall_rate 0.31, 64-71 episodes counted   ->  16 evals
```

The second mode is the honest one, and it happens by accident: when fewer than 32
environments fall before step 1000, the remaining ones all truncate on the same step and
enter the sample together.

## How wrong

Re-scoring saved checkpoints of the envelope run with one episode counted per environment:

| checkpoint | logged | actual |
|---|---|---|
| iteration 700 | 100% falls, ep_len 489, return 1216 | 33% falls, ep_len 836, return 2224 |
| iteration 900 | 100% falls, ep_len 342, return 955 | 73% falls, ep_len 546, return 1622 |

Two checkpoints that differ by more than a factor of two in fall rate were logged as
identical. On a checkpoint that had landed in the lucky mode the old code agreed with the
new one (31% vs 30%), which is why this survived: it was not wrong everywhere, it was wrong
whenever the policy was struggling.

## The damage went past the log

`best.pt` is selected on `eval/episode_return`, which inherits the same bias. Returns read
~2,700 in the lucky mode against ~1,000 in the biased one, a gap far larger than any real
difference between neighbouring checkpoints. Checkpoint selection was therefore mostly
deciding which mode the eval happened to land in, not which policy was better.

Three conclusions we drew from these numbers now need re-reading:

* "deterministic falls 100%, the policy needs its own noise to stand up". It does not. The
  noise-crutch gate in `scripts/amp_readiness.py` measures this directly and passes.
* abort rule 4 firing on the cadence run at iteration 2032. It fired on a biased number.
* every claim of the form "this run collapsed and then recovered". The bimodality is the
  metric switching modes, not the policy.

Training itself was never affected. The reward, the rollout and the PPO update never read
these numbers. Only the log, the abort rules and checkpoint selection did.

## Fix

Count the FIRST episode from each environment and no more, then stop when every environment
has reported. Each environment contributes exactly one sample regardless of how long it
lasted, so falls and survivors enter at the rate they actually occur. Two warnings were added
for the ways this can silently regress: environments that never finish inside `max_steps`,
and a sample narrower than the requested `num_episodes`.

## Why nothing caught it

The Oracle checks the setup, not the instrumentation, and a metric cannot be checked against
the config because it is not derivable from it. The check that would have caught this is a
consistency one: `eval/num_episodes` should not correlate with `eval/fall_rate`. It did,
perfectly, in every metrics.jsonl this project has ever written.
