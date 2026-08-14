# The policy was never asked to walk

Date: 2026-08-14. Found while answering "is the base good enough for AMP yet?"

## Summary

The humanoid does not walk at human speed because **it is not being told to**. The median
moving velocity command produced by the sampler is **0.38 m/s**. Human free walking is
1.2-1.4 m/s. The retargeted mocap in this repo runs 0.52-1.24 m/s.

The policy's measured 0.20 m/s is not a failure to track its command. It is obedience.
Every gait metric downstream (stride length 0.20 m, speed 0.40 m/s, "human-likeness 15%")
has been reading that obedience as a defect and sending us to look at the gait.

## Evidence

Sampling 200k commands straight from `LocomotionTask._draw_command` at full difficulty:

```
commanded speed: mean 0.45  median 0.38  p90 0.77  max 1.50
   fraction <0.3 m/s      (crawl)                21.4%
   fraction 0.52-1.24 m/s (mocap clip range)     17.2%
   fraction >1.2 m/s      (human walking)         2.1%
```

And the run's own eval metrics agree: `eval/commanded_speed` sat at 0.36-0.45 across all
three of the last 500M-step runs, at `difficulty_median = 1.0`. There is no curriculum level
at which this policy is asked to walk at a human pace.

Held at a coherent constant command (`scripts/amp_readiness.py`), the checkpoint is stable
and simply does not accelerate:

```
 commanded  achieved   ratio   falls
      0.50      0.20    0.39      0%
      0.80      0.20    0.24      0%
      1.00      0.14    0.14      2%
      1.50     -0.08   -0.05     99%
```

Two things worth separating here. Below ~0.5 m/s it tracks passably. Above that it does not
respond at all, and at 1.5 m/s it goes backwards and falls, which is what an out-of-
distribution input looks like: 98% of its training commands were below 1.2 m/s.

## Cause

Two changes compounded, each defensible alone.

1. **Polar sampling over an elliptical envelope.** Introduced to fix a real defect (a box
   over `(vx, vy)` put 52.7% of commands in the forward sector and 3.7% lateral, and capped
   a 45 degree diagonal at 0.57 m/s). It fixed direction coverage. But the envelope's
   lateral semi-axis is `lin_vel_y_range[1] = 0.4`, so the ellipse is crushed everywhere
   except straight ahead: at a 30 degree heading the reach is only 0.73 m/s.
2. **`magnitude = sqrt(uniform(0.25, 1.0))`**, which averages 0.79 of that already-small
   reach.

Direction coverage and speed coverage are different questions. The Oracle checked the first
and passed. Nothing checked the second.

## Why this also explains the AMP failure

The earlier AMP run's discriminator went 0.53 -> 0.98 accuracy while the style reward FELL
0.54 -> 0.31. The reading at the time was "a from-scratch policy does not move like a
person". That is true but incomplete. The clips run 0.52-1.24 m/s and only **17% of commands
land in that band**, so for most of every episode the policy was moving slower than any
reference the discriminator had ever seen. It was correctly calling the policy fake, and no
reachable behaviour would have changed that, because the task was not asking for one.

## Fixes landed

* `humanoid_rl/oracle/invariants.py`: new `commands_ask_for_walking_speed` check, flagging
  CONTRADICTION when the median moving command is below 0.6 m/s, and UNREACHABLE on an AMP
  run when under half the commands fall in the clip range.
* Same file: `commands_cover_every_direction` had been raising `KeyError: 'difficulty'` ever
  since difficulty scaling was added, so it was caught by the harness and reported SUSPECT
  rather than actually checking anything. Both checks now share `_sample_commands`, which
  supplies everything the sampler reads.
* `scripts/amp_readiness.py`: new gate script. Note its `impose()`: pinning only
  `task_state["command"]` is not enough, because the gait clock frequency and heading target
  are drawn alongside the velocity and the policy observes them. `scripts/gait_report.py`
  still has that confound and should be corrected the same way.

## Not yet done

Raising the envelope. The binding constraint is `lin_vel_y_range`. Raising it uniformly is
wrong, because human lateral walking genuinely is slow. The likely shape is to keep polar
direction sampling but draw the radius as a fraction of each heading's own reach with the
mass centred high, so "as fast as this direction allows" is the common case rather than the
rare one. One variable, one run, and `eval/commanded_speed` is the number that says whether
it took.
