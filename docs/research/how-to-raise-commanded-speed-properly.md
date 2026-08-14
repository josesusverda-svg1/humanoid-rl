# How the field raises commanded speed, and what we did instead

Date: 2026-08-14. Researched while the envelope run oscillated between tracking its command
and staying upright, to find out whether that oscillation is a known thing.

It is. It has a name, a documented cause, and a standard fix, and we did the documented
wrong thing.

## What we did

Raised the command envelope in one step, from a median moving command of 0.38 m/s to 0.71,
with the difficulty curriculum switched off (`difficulty_init = difficulty_min = 1.0`), so
every environment trains at the full envelope from step zero.

The result, over 18 unbiased evals: speed ratio climbing to 0.97 while the fall rate went to
100% and episode length collapsed to 230, then both backing off, repeatedly. The policy is
oscillating along the speed/balance trade-off rather than converging on it.

## What the field does

**Every reference expands the range gradually, gated on measured tracking performance.**

`legged_gym`, the codebase all three of our reference implementations descend from, does it
in seven lines, called at episode boundaries and only when `commands.curriculum` is set:

```python
def update_command_curriculum(self, env_ids):
    if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
        self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
        self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
```

Expand by 0.5 m/s, and only once the tracking reward is above 80% of its own maximum. Not on
a schedule, not on iteration count: on demonstrated competence.

KSLC (Zhang et al., 2024), which reaches **3.5 m/s** on a humanoid, uses the same rule with
the same threshold (lambda = 0.8), the same 0.5 m/s increment, and adds two things we do not
have. ALMI (2025) does the same for its motion-tracking policy: "when the tracking error is
smaller than a threshold, the command sampling range will be extended".

The failure mode when you skip it is stated outright in KSLC: the robot "may struggle to earn
rewards if the early training is dominated by high-velocity commands, potentially leading to
failure in rapid locomotion tasks."

## The part we would have missed

KSLC's second contribution is the one that actually explains our falls, and it is not about
commands at all. It is about **reward targets that do not move when the command does**:

> Fixed rewards focus on a single motion pattern and cause robots to default to slower
> walking behaviours, resulting in overly conservative strategies at higher speeds.

Their velocity-dependent rewards scale three targets with the commanded velocity:

* **base height** (CoM drops as speed rises)
* **feet clearance** (swing height rises with speed)
* **joint position / stride length**

Ours are all fixed constants:

| target | ours | scales with speed? |
|---|---|---|
| `target_height` | 0.877 m | no |
| `swing_height_target` | 0.08 m, at weight **-20.0** | no |
| `body_height_range` | sampled at random | no, and uncorrelated with the command |

So at 1.5 m/s our reward is still demanding standing-height posture and 8 cm of foot
clearance, and penalising deviation at weight -20. The policy is being asked to sprint
without changing its posture. That is precisely the "overly conservative at higher speeds"
trap, and it is a better explanation of the falls than anything about balance.

KSLC's third piece is a **cycle-time curriculum**: shrink the gait period to 95% of its
previous value each time the same 80% gate is met, floor 0.48 s. We couple cadence to speed
by Inman's square-root law instead, which is defensible and probably fine, but ours is
clipped at `gait_frequency_range` 1.1 Hz and theirs is not.

## The number that reframes the whole project

XBot-L is the 1.65 m humanoid our conformance audit was written against, and the one with
proven zero-shot hardware transfer. Its shipped command config:

```
lin_vel_x  = [-0.3, 0.6]   m/s
lin_vel_y  = [-0.3, 0.3]   m/s
ang_vel_yaw = [-0.3, 0.3]  rad/s
cycle_time = 0.64 s
base_height_target = 0.89
```

**XBot-L trains at 0.6 m/s forward, maximum.** Our new envelope reaches 1.5, two and a half
times that. Human free walking is 1.2-1.4 m/s.

This corrects a claim in our own `locomotion.py`, which says the references "train the full
range from scratch" and uses that to justify disabling the curriculum. The claim is true but
misleading: they train their full range from scratch, and their full range is 0.6 m/s. At
0.6 m/s no curriculum is needed. At 1.5 m/s, KSLC needed one to get there and so will we.

So human walking speed is not a bug fix. It is a stretch goal beyond what our reference
achieves, and the paper that does reach it spends three separate mechanisms getting there.

## On AMP, the original question

ALMI trains "a basic locomotion policy without any upper body intervention" first and only
then begins adversarial iterations. That matches what our own AMP failure taught us and what
`scripts/amp_readiness.py` now gates on. The order is: stable walker, then style. Not
concurrently, and not before.

## What this implies for us

1. Re-enable the difficulty curriculum, gated on tracking reward rather than left off. Our
   machinery already promotes on competence and demotes on falls; it was switched off on
   reasoning taken from a config whose envelope was 0.6 m/s.
2. Make `target_height` and `swing_height_target` functions of commanded speed. This is the
   cheapest of the three and the one with the clearest mechanism behind our falls.
3. Treat 1.5 m/s as the top of a curriculum, not the starting envelope.

## Sources

* legged_gym, `legged_robot.py` (leggedrobotics) — `update_command_curriculum`
* Zhang et al., *Achieving Stable High-Speed Locomotion for Humanoid Robots with Deep
  Reinforcement Learning* (KSLC), arXiv:2409.16611
* *Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning* (ALMI),
  arXiv:2504.14305
* Humanoid-Gym / XBot-L, `humanoid_config.py` (roboterax)
* Gu et al., *Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot
  Sim2Real Transfer*, arXiv:2404.05695
