# Logbook

Every change, every result, every bug. Read before changing anything.

## How to use this

**Before you change something**, search this file for the knob you are about to touch:

```bash
python scripts/logbook.py --about swing_height
python scripts/logbook.py --settled          # everything already ruled out
```

**After a run ends**, generate the entry skeleton with real numbers already filled in:

```bash
python scripts/logbook.py --run runs/<run-dir>
```

Three rules that keep this honest:

1. **Write the prediction before the run, not after.** An entry whose "expected" column was
   filled in afterwards teaches nothing, because everything looks predictable in hindsight.
2. **A verdict is mandatory and one of five.** WORKED, NO EFFECT, WORSE, INVALID (the
   measurement was broken so the run says nothing), MIXED. "Promising" is not a verdict.
3. **When an instrumentation bug is found, go back and mark what it invalidated.** Four
   times this week a bug meant earlier conclusions were about the metric, not the policy.

---

## Settled: do not retry without new information

| Thing | Verdict | Entry | Why |
|---|---|---|---|
| `symmetry_loss_coef` as a reward penalty | WORSE | E03 | Zero for any policy ignoring its input. PPO found a 61 cm two-footed brace as a global optimum. Use mirror data augmentation instead. |
| AMP from a non-walking policy | WORSE | E08 | Discriminator 0.53 → 0.98 accuracy in 97 iterations, style reward FELL 0.54 → 0.31. It correctly calls a crawl fake. Gate on `scripts/amp_readiness.py` first. |
| Booster T1's `w_torque = -2e-4` | WORSE | E09 | Costs -9.4/step against a +6.4 positive budget. Use XBot's -1e-5. |
| Weakening gait terms so AMP owns gait shaping | WORSE | E08 | Foot slip 42-51% of travel speed vs 27% baseline, over 7 evals, never trended down. |
| Velocity-scaled swing-height target (KSLC) | NO EFFECT | E15 | Measured on our own mocap: mean swing clearance is `0.063 + 0.020*speed`. Worth ≤0.03/step across the whole envelope. Not our problem. |
| `difficulty_init = 0.45` | WORSE | E13 | On the OLD crushed envelope this meant 0.17 m/s commands, cheaper to ignore than follow. Retried at 0.7 on the fixed envelope: E17. |
| Abdomen exploration floor to fix the fold | INCONCLUSIVE | E24 | Underpowered by 2x: MDE 0.38 against a predicted 0.16. All 7 arms still folded backward 34-68 deg, so the fold is structural, not exploratory. |
| Steepening the torso posture reward | NO EFFECT expected | E23 | Measured counterfactual: straightening RAISES reward +0.13/step at equal speed. Reward already prefers upright; the problem is optimisation, not pricing. |
| Reward normalisation to rescue FastTD3 | NO EFFECT | E22c | Critic measured unsaturated: 4.7e-16 mass on the top atom, 65 of 401 atoms in use. Scale is not the problem. |
| Single-run A/B on training outcome | INVALID | E22b | Byte-identical configs gave 323 vs 2451 mean return. Any effect under ~7x is inside the noise. |
| Replay ratio as the FastTD3 fix | WORSE | E22 | Ratio 2 and 16 both flat, 100% falls on 96 of 96 evals. |
| `promote_gait_match = 0.80` | INVALID | E17 | Unreachable. Measured gait match runs 0.66-0.73. Standing alone scores 0.60, so the usable band is 0.60-1.0. |

## Instrumentation bugs found

A bug here means earlier conclusions were about the *measurement*, not the policy. Each row
lists what it invalidated.

| Bug | Found | Invalidated | Fixed in |
|---|---|---|---|
| **Actuator order != qpos order**: hip_y/hip_z transposed on both legs, 4 of 28 | E25 | The joint-limit penalty scored hip_y against hip_z's tighter range at w=-5.0, penalising deep hip flexion. `tracking.py` commanded swapped hip angles on every clip. | `PreparedModel.actuator_qpos_adr` |
| Videos played at 0.4x (125 Hz physics encoded at 50 fps) | E01 | Every visual gait judgement before it | `frame_skip`, `encode_fps` |
| `gait_symmetry` scored 1.0 for standing still (defined over stance time) | E02 | Every symmetry claim before it | Redefined over SWING time |
| `explained_variance` was tautological (`returns = advantages + values`) | E10 | All critic-quality claims | Renamed `advantage_share` |
| Ablation ran 6 byte-identical arms (`amp.yaml` has no `task:` section) | E11 | The entire first ablation | `TASK_SECTION` + Oracle check |
| **Eval counted the first 32 of 64 episodes to finish, which are the falls** | E16 | **Every fall rate, episode length and `best.pt` choice in project history** | One episode per env |
| Episodes were 8 s, not the 20 s every comment claimed (50 Hz assumed, we run 125 Hz) | E18 | All "survived the episode" numbers; `command_hold_range` never fired | `max_episode_steps` 1000 → 2500 |
| `log_std` sat above its clamp, which passes no gradient | E05 | 610 iterations of frozen exploration | In-place clamp after optimiser step |
| Normaliser `COUNT_MAX = 1e6` destroyed warm starts | E06 | Warm-started runs before it | Cap removed |
| Heading command was an integrated yaw rate, so the target spun away | E07 | All heading-error numbers before it | XBot heading command |

## Current state

Best policy: `runs/envelope-20260814-100402/checkpoints/best.pt`, **iteration 3100**.

READ THE CONDITION COLUMN. An earlier version of this table mixed three different
measurement conditions into one column and produced a policy that does not exist.

| Metric | Value | Condition | Human |
|---|---|---|---|
| return | 3034 | eval at iter 3100 | - |
| falls | 15.6% | eval at iter 3100 | - |
| episode length | 931 steps (7.4 s) | eval at iter 3100 | - |
| mean speed | 0.589 m/s | eval at iter 3100, mixed commands | 1.2-1.4 |
| speed | 0.78 m/s | held 1.0 m/s command, gait_report | 1.2-1.4 |
| torso_upright | **0.726** | eval at iter 3100 | 0.95-1.00 |
| torso_upright | 0.658 | held 1.0 m/s command | 0.95-1.00 |
| torso tilt DIRECTION | **48.5 deg BACKWARD and LEFT** | held 1.0 m/s command | upright |

The often-quoted `torso_upright 0.543` belongs to iteration **5050**, a later and WORSE
checkpoint (return 2328, falls 29.7%) that was never selected as best. Quoting it beside
iteration 3100's return described a policy that never existed.

AMP readiness: NOT ready. Passes "stays up", fails "obeys speed".

## Entries

Newest first. `E##  date  what changed`.

### E28  2026-08-15  The sitting trap: `upright` rewards a pose that is NOT on the path
- **Found before spending compute**, by computing the reward at each stage of a rise rather than watching another run fail.
- **The defect is in the definition, not in a number.** `upright = clip(-gravity_body[2], 0, 1)` measures PELVIS ORIENTATION, which is **not monotone** along the path from floor to stand: maximal sitting, low on all fours and kneeling, maximal again standing. So the route out of a sit runs DOWNHILL, and both previous runs parked in exactly that sit (`pelvis_upright` 0.93, head 0.54, standing 0%).
- Pelvis HEIGHT is monotone by geometry: an intermediate pose cannot lie outside the interval between lying (~0.15 m) and standing (0.877). It needs no verification, unlike orientation.
- **Fix**: potential-based shaping on pelvis height, `F = gamma*Phi(s') - Phi(s)`, `Phi = root_height / standing_height`.
- **Two sizing decisions, both from measurement rather than taste**:
  - **Weight 150, not 5.** Sitting pays ~0.50/step and the intermediate poses ~0.20, so the route out costs ~0.30/step. Lifting the pelvis 0.35 -> 0.60 m in a second moves Phi by 0.00228/step, so covering the dip needs ~150. At 5 it would have been 0.011/step, three percent of the trap, and would have changed nothing.
  - **gamma = 1.0, NOT ppo.gamma, and this breaks strict invariance deliberately.** At 0.99 and 125 Hz the `-(1-gamma)*Phi` drain dominates: measured, even rising at 0.3 m/s scored NEGATIVE at weight 5, and at a useful weight it costs 1.5/step just for being upright. At gamma = 1 the sum telescopes exactly, so the shaping over an episode is `weight * (Phi_end - Phi_start)` and nothing else: path length is irrelevant and pumping the pelvis up and down pays exactly zero.
- **Guard**: an Oracle check fails any config where `weight * (1 - gamma)` exceeds 0.10/step. Verified to bite at gamma 0.99 ("costs 1.50/step") and stay quiet on the real config.
- **Prediction on the record, before the run**: ~60% he sits and parks again, ~25% reaches occasional stands but cannot hold 2 s, ~15% real successes. The early tell is `knee_max` above 1.2, which the old action range made impossible (ceiling 1.05, peak 1.057).
- **Also this session**: the ball was built, measured, and then switched off at the user's request; the pose bank was rebuilt with QUOTAS BY OUTCOME (425 each of prone/supine/side-left/side-right) because commanding an orientation is not enough, a body laid on its side often rolls onto its front as it settles.

### E27  2026-08-15  The knee could not reach a kneel, and the fix was written but never connected
- **Found by a person watching**: "he bends his knees, leans on his hands and heels, but I do not see him trying to rock or rise. As if he likes hanging there."
- **Measured, and it is not a preference**:

```
knee joint range            [0.00, 2.79] rad
COMMANDABLE ceiling          1.05 rad      (action_scale_mode = "fraction")
a kneel needs               ~2.40 rad
peak observed in the run     1.057 rad     <- the ceiling, to a hundredth
```

- **The action space could not express the pose.** The agents' spec said this in section 0.1 and I built `action_scale_mode="full_range"`, verified it gives 100% joint coverage against 56.2%, committed it, and **never plumbed it through**. `vec_env` called `prepare(model_path)` with no mode, so every run since silently kept the old mapping. Two get-up runs, 600M steps total, spent on a body that physically could not fold a leg under itself.
- **Verdict**: WASTED. Not a slow-learning problem: no amount of training makes an inexpressible movement expressible.
- **Honest caveat**: mean knee angle was 0.38 against the 1.05 ceiling, so the limit was not binding on average and part of the parking is still the reward. The PEAK sitting exactly on the ceiling is what proves the limit bit.
- **Fix**: `env.action_scale_mode` plumbed through Config -> ThreadedVecEnv -> prepare(), and passed at all five construction sites. Knee now commandable to 2.79 rad; driving the action to +1.0 reaches 2.80. Walking stays on "fraction" and its runs are bit-identical.
- **Guard added, because this is the third time**: the trainer now ASSERTS that the render env's `action_scale` matches the training env's. A render env with a different action mapping is a different robot, so every video would show behaviour the policy never produced. Same family as the 8-second episodes, the transposed actuators and the overlay printing a command the policy never received.
- **Lesson**: building a capability and not wiring it is indistinguishable, from the outside, from not building it. The Oracle checks configs against each other; nothing checked that a config field reaches the code that consumes it.

### E26  2026-08-15  Get-up run 1: sits up on one arm, never uses its legs
- **Setup**: new GetUpTask, 300M steps, 1200-pose bank, 2 s hold, 13-conjunct standing predicate.
- **Result**: got up to sitting and stopped. Best eval at iteration 2700: head 0.537, root 0.322, pelvis_upright 0.933, **standing 0.0%**, return 690 (of a ~3500 ceiling).
- **Verdict**: WORSE than intended, and the failure was found by a person watching the video, not by any metric. Their description: "all the pressure on one hand, lifting his hip, raising one hand, drifts in circles, doesn't bend his knee."
- **Measured, and it matched every word**:

| | |
|---|---|
| left hand height | 0.469 m |
| right hand height | 0.064 m |
| at least one hand on the floor | 98% of the time |
| BOTH hands down | 1% |
| knee angle | 0.53 rad, max ever 1.06 (a kneel needs ~2.4) |
| yaw drift | 58.7 deg/s, a full turn every 6 s |

- **Root cause, and it is not the convexity I blamed earlier.** NOTHING in the reward required the legs to do anything. `upright` pays for pelvis verticality and `rise` for head height; a one-armed prop buys both without using a leg. The foot-force conjuncts (U7, U8) exist but gate only the STANDING terms, which pay zero for the entire approach, so the legs were irrelevant on the whole path from lying to standing. The cheapest way to raise the pelvis and head was to push with one arm, and the spin is that arm's reaction torque.
- **Fix**: `rise` is now multiplied by foot load, ramped to full at 0.30 BW. Not a new term. This is the same device the design already uses to stop height bought by DIVING from paying (`rise` is multiplied by pelvis uprightness); the arm-propping hole was simply left open. A one-armed prop with unloaded feet now earns zero rise; the same posture with the feet under the body earns it in full.
- **Also added**: `foot_load_bw`, `hand_height_gap`, `hands_down_frac`, `knee_max`, `spin_deg_s` to eval metrics. All five were invisible before, which is why this needed a human and a video. A defect that only a person can see is a missing metric.
- **Lesson, general**: gating a reward on a state the policy must EARN is stronger than penalising the alternative. Every anti-cheat in this task that has held is a gate; every one that leaked was an unguarded positive.

### E25  2026-08-14  Actuator order is NOT qpos order, and two live bugs came from assuming it is
- **Found while designing the get-up task**, by agents measuring the model rather than reading comments.
- **The fact**: on this humanoid, actuator `i` does NOT control `qpos[7 + i]`. On both legs `hip_y` and `hip_z` are transposed, so **4 of 28 actuators** disagree with that assumption:

```
actuator 15 right_hip_z -> qpos 23      qpos[7+15] is right_hip_y
actuator 16 right_hip_y -> qpos 22      qpos[7+16] is right_hip_z
actuator 22 left_hip_z  -> qpos 30      (same transposition)
actuator 23 left_hip_y  -> qpos 29
```

- **Live bug 1, in the walking task we have been training all week.** The soft joint-limit penalty built its bounds from `actuator_ctrlrange` (actuator order) and indexed them against `state.qpos[:, 7:7+nu]` (qpos order). So **`hip_y`'s true +-2.44 rad range was scored against `hip_z`'s +-1.05 rad one**, at weight -5.0. The penalty therefore fired on deep hip flexion, which is exactly the motion a long stride requires, and we have spent the week fighting short strides (0.11 -> 0.30 m against a human 0.6-0.8).
- **Honest size of the effect**: `reward/dof_pos_limits` measured -0.007/step, so it was not a large direct cost. Whether it acted as a barrier to the deeper flexion that never got tried is not established by that number, and I am not claiming it was.
- **Live bug 2**, `tasks/tracking.py:181`: returns `lib.qpos[idx, 7:]` as an `action_offset`, which the engine applies in actuator order. Every tracked clip commands each `hip_z` servo the reference's `hip_y` angle and vice versa. The tracking task is not currently in use, so nothing downstream is contaminated, but it would have been.
- **Fix**: `PreparedModel.actuator_qpos_adr` now carries the map, `vec_env` hands it to the task, and the limit penalty indexes through it. Verified: the map differs from the naive assumption on exactly 4 of 28 actuators, and the environment still runs with finite rewards.
- **Rule**: anything pairing a per-actuator quantity (control range, target, action offset) with a joint angle must go through `actuator_qpos_adr`. Never `qpos[7 + i]`.

### E24  2026-08-14  Abdomen exploration floor: INCONCLUSIVE, and I should have known before running
- **Hypothesis**: the 48 degree backward waist fold persists because `abdomen_y` has the lowest exploration of all 28 action dimensions (std 0.196 against a mean of 0.889), so PPO never samples its way out.
- **Design**: 2 arms (control, `explore_floor = -0.70` on abdomen dims 0/1/2) x 3 seeds x 350 iterations, warm-started from the envelope best.pt, all seven checkpoints scored under an identical held 1.0 m/s command.
- **Result**, held-command scoring:

| arm | torso_upright | | | mean | sd |
|---|---|---|---|---|---|
| control | 0.8275 | 0.7041 | 0.5589 | 0.697 | 0.134 |
| floored | 0.3730 | 0.5389 | 0.7390 | 0.550 | 0.183 |
| baseline (untrained warm start) | 0.6578 | | | | |

- Difference (floored - control) = **-0.147, SE 0.131, t = -1.12 on ~4 df, 95% CI -0.51 to +0.22.** Spans zero comfortably.
- **Verdict**: INCONCLUSIVE. The hypothesis is not supported, and it is not refuted either.
- **The process failure, which is the real lesson.** With the control sd of 0.134, three seeds per arm can only detect an effect of about **0.38**. The effect I was chasing, from the open-loop counterfactual, was about **0.16**. The experiment was underpowered by more than a factor of two BEFORE it ran, and computing that takes one line of arithmetic I did not do. E22b had already warned me the noise on this machine is enormous; I applied that warning to the choice of metric and not to the sample size. Detecting 0.16 here needs roughly **17 seeds per arm**, about 6 hours.
- **What DID come out of it, and it is worth more than the experiment.** All seven checkpoints, across both treatments and three seeds, fold **BACKWARD**, between 34 and 68 degrees, with a consistent leftward roll (lateral +0.31 to +0.46). Not one arm under any condition came out upright. So the backward-left fold is a **systematic property of this reward and this body**, not a random local optimum that a nudge to exploration could escape. That reframes the fix: it is structural, not exploratory.
- **Also note**: control seeds span 0.559 to 0.828 on posture from byte-identical configs. E22b's noise floor is confirmed to apply to `torso_upright`, not just to episode return.
- **Rule adopted**: compute the minimum detectable effect BEFORE launching any arm, and write it in the plan next to the predicted effect. If MDE > predicted effect, the experiment does not run.

### E23  2026-08-14  The lean: three of my claims were wrong, and it is not a reward problem
- **What I claimed**: the humanoid leans FORWARD at torso_upright 0.54, exploiting the termination boundary at 0.50, because the reward prices speed above posture roughly 2:1.
- **All three are false.** Verified independently, not taken from the review:

1. **Checkpoint mismatch.** `best.pt` is iteration 3100 with `torso_upright 0.726`. The 0.543 figure is iteration 5050, a later checkpoint with return 2328 and 29.7% falls that was never selected. I quoted its posture next to iteration 3100's return, speed and fall rate, describing a policy that does not exist.
2. **The termination boundary is dead code.** `locomotion.py:1111-1112` sets `stooped = np.zeros_like(fallen)` and `head_down = np.zeros_like(fallen)`; the conformance audit removed posture termination. `terminate_torso_upright = 0.5` is read only by `tracking.py`. There is no cliff at 0.50, so nothing is hugging it.
3. **The lean is BACKWARD, not forward.** Rolling out best.pt at a held 1.0 m/s and decomposing the torso z-axis in the heading frame: fore component **-0.640, forward in 0.0% of samples**; lateral +0.369, **left in 100%**; `abdomen_y = -0.755 rad`; tilt 48.5 degrees. It is a backward-and-left waist fold. `torso_upright = cos(tilt)` is SIGN-BLIND, so the metric cannot tell forward from backward and neither could I.

- **The actual finding, and it inverts the diagnosis.** Paired counterfactual, same seed, abdomen actuator outputs scaled by alpha:

| alpha | torso_upright | tilt | speed | **reward/step** |
|---|---|---|---|---|
| 1.00 | 0.658 | 48.9 deg | 0.694 | 3.377 |
| **0.75** | **0.816** | **35.3 deg** | 0.693 | **3.507** |
| 0.50 | 0.927 | 22.0 deg | 0.606 | 3.402 |
| 0.25 | 0.978 | 12.0 deg | 0.561 | 3.289 |

Straightening the trunk **raises total reward by +0.13/step at no speed cost**. The reward already prefers upright. The policy is sitting in a local optimum that its own reward function disprefers.

- **Verdict**: the planned fix (steepen the posture term) was aimed at the wrong thing. This is not mispricing, it is an optimisation failure.
- **Mechanism to test next**: PPO explores with i.i.d. per-step Gaussian noise, but the postural gain only materialises when an abdomen offset is HELD across a whole stride. Independent noise averages it away, so the improvement is never sampled coherently even though it is well inside the exploration range. If that is right, the fix is temporally correlated exploration on the abdomen dimensions, not a reward weight.
- **Learned, generally**: a cosine-based uprightness metric cannot distinguish the direction of a tilt, and we shipped one as a reward term, a termination condition and a headline dashboard number. Any angular metric needs its sign checked before it is trusted.

### E22  2026-08-14  Replay ratio 2 vs 16, and a finding that outranks it
- **Change**: FastTD3 at replay ratio 2 and 16, both async, 50M env steps each, same seed, evals aligned to the same env-step grid.
- **Result**: BOTH FLAT. 100% falls on all 96 evaluations across both arms.

| env steps | ratio 2 | falls | ratio 16 | falls |
|---|---|---|---|---|
| 1.0M | 252 | 100% | 134 | 100% |
| 13.3M | 316 | 100% | 563 | 100% |
| 25.6M | 304 | 100% | 556 | 100% |
| 44.0M | 398 | 100% | 483 | 100% |
| best | 426 @ 17.4M | | 734 @ 3.1M | |

- **Verdict**: WORSE. Replay ratio is not the deciding variable; the arms differ by less than the noise floor (see below) and both are far under PPO's 3034 at 7.4 s upright.
- **Useful negative**: ratio-16 async tracked the earlier SYNC run point for point, so the async collector changes speed and not learning. That part is confirmed sound.

### E22b  2026-08-14  **The noise floor: identical configs, 7.6x different outcomes**
- **Found while adversarially reviewing an experiment design.** `runs/arm-warm-20260813-180357` and `runs/arm-warm-20260813-181716` have byte-identical `config.yaml` (verified by `diff`), the same `run.seed: 0`, the same warm start and the same 350 iterations.

| | 180357 | 181716 |
|---|---|---|
| mean training return, iters 101-350 (n=245) | **323.2** | **2451.4** |
| mean episode length | 133.5 | 789.8 |

- One collapsed into the 100%-falls attractor; one held the walk. **Same settings, same seed.** Threaded physics across 10 workers and MPS kernels are both non-deterministic, so a seed does not pin a trajectory here.
- **Verdict**: this is the single most important measurement in the logbook, because it sets the bar every other entry must clear.
- **What it invalidates**: any conclusion drawn from comparing the TRAINING OUTCOME of two single runs where the effect was smaller than roughly 7x. That includes several claims in earlier entries.
- **What it does NOT invalidate**: mechanical facts measured directly rather than through training. E14's `commanded_speed 0.38 -> 0.70` is a property of the sampler, verified by drawing 200k commands. E16's eval bias was proven by re-scoring fixed checkpoints. E21's throughput numbers are wall-clock. Those stand.
- **How to apply**: an outcome comparison needs multiple seeds, or a within-run paired measurement, or an effect larger than 7x. A single-run A/B on final return cannot support a conclusion on this machine.

### E22c  2026-08-14  Reward scale was NOT killing the off-policy arm
- **My hypothesis, now disconfirmed.** I argued that 21 unnormalised reward terms spanning 0-3000 were overloading the TD3 critic, since PPO normalises advantages and is scale-invariant while TD3 is not.
- **Measured on the ratio-2 best checkpoint**: mass on the top atom **4.7e-16** (saturation would be near 1.0), mass on the bottom atom 6.6e-17, **65 of 401 atoms in use**, E[Q] = 210 against a realised discounted return of the same order.
- **Verdict**: the critic is fitting cleanly and is nowhere near saturated. Reward normalisation would not have fixed anything. Experiment dropped BEFORE spending compute on it.
- **Also corrected**: the `value resolution` sub-check I added to the Oracle was wrong and has been removed. It warned that coarse atoms mean "one step of improvement may not move the target". False: the categorical projection is exactly mean-preserving and the actor consumes only `E[Q] = sum(p*z)`, which is continuous in the probabilities at any atom spacing. Coarse atoms limit the representable SHAPE of the distribution, not the quantity the policy gradient uses. A wrong check is worse than no check.

### E21  2026-08-14  Async collector: overlap physics and gradients
- **Change**: `humanoid_rl/algos/async_collector.py`. Environment stepping moves to a background thread; the learner owns the replay buffer exclusively and the actor holds a policy snapshot. Enabled by `fasttd3.async_collection`.
- **Why**: Phase 0 measured that CPU physics and Metal updates barely interfere (CPU keeps 91.5%, GPU 100.3% concurrent), yet every trainer alternated them strictly. Only legal off-policy; PPO must stop the world for on-policy data.
- **Prediction** (before the run): overlap should give `max(physics, gradients)` instead of the sum, so up to 1.17x at replay ratio 16 and 1.75x at ratio 2.
- **Result**, four arms all at exactly 1.23M env steps:

| replay ratio | sync | async | gain | vs theory |
|---|---|---|---|---|
| 16 | 12,244 sps | 13,158 sps | +7% | 92% of the 1.17x available |
| 2 | 45,927 sps | 61,386 sps | +34% | 76% of the 1.75x available |

- **Verdict**: WORKED, and smaller than it first appeared. Overlap pays in proportion to how BALANCED the two sides are. At ratio 16 the GPU takes 295 ms against physics' 49, so there is only 14% to reclaim no matter how good the implementation is.
- **Bug caught in my own benchmark**: the first async arm reported 55,262 sps, a 4.5x "speedup". It was fake. `drain()` took the whole queued backlog, so the actor ran 8x ahead of the learner and the configured replay ratio of 16 silently became 1.9. The giveaway was the step counts not matching: 13.4M env steps against the sync arm's 1.6M for the same 400 iterations. Fixed by taking exactly the requested steps and shortening the queue to 4 for backpressure.
- **Learned**: throughput comparisons between RL configurations are meaningless unless the replay ratio is pinned and verified afterwards. Ratio is not a tuning detail, it is the axis the whole comparison sits on.
- **Open, and now the important question**: ratio 2 async runs at 61,386 sps, faster than PPO's 50,244 while still reusing every transition twice. Whether ratio 2 LEARNS as well per environment step as ratio 16 is untested here and the FastTD3 paper argues the opposite. That is the next experiment.

### E20  2026-08-14  Curriculum collapsed to its floor  **(live, and my own fault)**
- **What happened**: difficulty went 0.70 → 0.67 → 0.52 → **0.50 (floor) by iteration 150** and has been pinned there for 500 iterations. Commanded speed fell to 0.36-0.40 m/s, which is exactly the crawl E14 existed to fix.
- **Cause**: I shipped E17 (curriculum on) and E18 (episodes 8 s → 20 s) in the SAME run, against my own one-variable rule. Twenty-second episodes make "survive without falling" a 2.5x harder bar, so nearly every environment falls at some point. Demotion is -0.10 per fall and promotion +0.05 per clean segment, so at a 50-100% fall rate everything slides to the floor within 150 iterations and stays.
- **Verdict**: WORSE. The run is training a crawler. The curriculum is technically working as specified and the specification is wrong.
- **Learned, and this is the general lesson**: a curriculum floor is a promise about the *easiest command you are willing to train on*. `difficulty_min = 0.5` gives a median command of 0.37 m/s, and we have already proved at length that a policy asked to crawl learns to crawl. **The floor must never sit below a command worth practising.** At 0.7 the median is 0.49; at 0.85 it is 0.60.
- **Also wrong**: promote +0.05 against demote -0.10 needs two clean segments per fall just to hold station. That is unreachable when a fall is likely in any 20 s window.
- **Not yet fixed.** Candidate: `difficulty_min` 0.5 → 0.75 and symmetric promote/demote, or demote only after two consecutive failures. One at a time.

### E19  2026-08-14  FastTD3 groundwork (no run yet)
- **Change**: new off-policy algorithm alongside PPO, on branch `fasttd3`. Env, reward, obs, model untouched.
- **Why**: PPO discards every transition after a few gradient steps. Physics is 57.4% of our iteration time, so sample efficiency is the lever, not a faster chip.
- **Prediction**: not yet run. Head-to-head against PPO on the same task once the CPU frees.
- **Result**: 11 correctness tests pass. Oracle rejected the first value support before any compute (see below).
- **Verdict**: n/a, groundwork only.
- **Learned**: sizing a distributional critic's support from *measured* reward (3.27/step) was wrong; the reward *weights* allow 6.4/step, so reachable return is 640 not 327. A short support saturates every good state on the top atom and the critic goes blind while its loss looks perfect.

### E18  2026-08-14  Episode length 8 s → 20 s
- **Change**: `max_episode_steps` 1000 → 2500. Also `amp.yaml` 600 → 2500, `tracking.yaml` 300 → 750.
- **Why**: the comment said "20 s at 50 Hz". We run 125 Hz, so 1000 steps was 8.0 s. Copied from legged_gym without converting for a 2.5x faster control loop.
- **Second effect**: `command_hold_range` is 8-12 s, *longer than the whole episode*, so the command never changed mid-walk. About 0.8 commands per episode against legged_gym's 2.
- **Verdict**: BUG FIXED. Not comparable across the change: surviving is now a 2.5x harder bar.
- **Also rescaled**: abort rules 1/2/6 are raw step counts (250→625, 500→1250, 400→1000).

### E17  2026-08-14  Difficulty curriculum re-enabled
- **Change**: `difficulty_init` 1.0 → 0.7, `difficulty_min` 1.0 → 0.5, `promote_gait_match` 0.80 → 0.68.
- **Why**: E14 raised the envelope in one jump and the policy oscillated between tracking speed and falling. legged_gym, KSLC and ALMI all expand the range only when tracking reward exceeds 0.8 of its maximum. Our machinery existed and was switched off.
- **The justification for switching it off was wrong**: it read "the references train their full range from scratch". True, but XBot-L's full range is `[-0.3, 0.6]` m/s. Ours reaches 1.5.
- **Prediction**: difficulty median rises off 0.70; falls lower than the control at equal ratio.
- **Result**: running (`runs/curriculum-20260814-125526`).
- **Caught before launch**: `promote_gait_match = 0.80` was unreachable (measured 0.66-0.73), which would have pinned difficulty at 0.70 all run and looked like the curriculum simply not working.

### E16  2026-08-14  Eval selection bias  **(the big one)**
- **Found**: `evaluate()` stopped after the first 32 of 64 episodes finished. Under autoreset those are the SHORT ones, i.e. the falls. Survivors only counted when enough truncated together.
- **Signature**: sample size correlates with the reported fall rate. Over 101 evals: `fall_rate 1.00` with 32-34 episodes (85 evals), `fall_rate 0.31` with 64-71 (16 evals). Two modes, nothing between.
- **Re-scored the same checkpoints**: iteration 700 logged 100% falls / return 1216, actually **33% / 2224**. Iteration 900 logged 100% / 955, actually **73% / 1622**.
- **Verdict**: BUG FIXED, count one episode per environment.
- **Invalidated**: every fall rate and episode length in project history, and every `best.pt` choice (returns read ~2700 in the lucky mode vs ~1000 otherwise, a gap larger than any real quality difference).
- **Guard added**: abort rule 7 watches for sample size correlating with fall rate.

### E15  2026-08-14  Velocity-scaled reward targets — measured, then dropped
- **Change proposed**: scale `target_height` and `swing_height_target` with commanded speed, per KSLC.
- **Measured on our own mocap first**: pelvis height `0.902 + 0.012*speed` (moves 1.7 cm, and *upward*). Mean swing clearance `0.063 + 0.020*speed`, worth ≤0.03/step at weight -20.
- **Verdict**: NO EFFECT expected, not implemented. Only the constant was corrected, 0.08 → 0.09.
- **Trap recorded**: measured as *peak* swing height the slope is 4x larger (`0.090 + 0.085*speed`) and implies a 0.30/step penalty that looks like a smoking gun. The reward penalises every airborne step, not the apex, so the mean is the matching statistic.

### E14  2026-08-14  Command envelope  **(largest single win so far)**
- **Change**: `lin_vel_y_range` ±0.4 → ±0.6, `lin_vel_x_range` -0.5 → -0.8, new `forward_bias_prob 0.4` over a 30° cone.
- **Why**: median moving command was **0.38 m/s**. Human walking is 1.2-1.4, our mocap 0.52-1.24. The policy was tracking its command faithfully; the command was a crawl. The lateral value is the ellipse's semi-axis, so it crushed reach at *every* off-axis heading.
- **Prediction**: `eval/commanded_speed` rises well past the 0.36-0.45 all previous runs sat at.
- **Result**: median command 0.38 → 0.70; in-clip-range 17% → 70%; forward ≥1.0 m/s 4.6% → 20.2%. Policy speed 0.40 → **0.78 m/s**, stride 0.11 → 0.30 m.
- **Verdict**: WORKED, and it also explains E08: only 17% of commands landed in the clip range, so the discriminator was shown a crawl and correctly called it fake.
- **Cost**: torso upright fell 0.77 → 0.54. It got faster by leaning.

### E13  2026-08-14  Cadence range
- **Change**: `gait_frequency_range` (1.0, 2.0) → (0.7, 1.1) Hz.
- **Why**: 1.0-2.0 Hz commands 2-4 foot strikes/s against a human 1.6-2.0. The policy delivered 2.91, i.e. obeying. Stride is then forced arithmetic: speed / strike rate.
- **Result**: step rate 2.91 → 2.06, stride 0.11 → 0.20 m. Speed 0.32 → 0.40.
- **Verdict**: WORKED, partially. Stride still far from 0.6-0.8, which led to E14.

### E12  2026-08-13  Max stance width penalty
- **Change**: `feet_distance_max = 0.45`, penalty on exceeding it.
- **Why**: XBot rewards a band (min AND max); only the minimum was ported. The policy stood 0.67 m wide because nothing opposed splaying, and splaying is free stability.
- **Result**: stance width 0.67 → 0.34-0.40 m. Stride unchanged at 0.11.
- **Verdict**: WORKED for stance, NO EFFECT on stride.

### E11  2026-08-13  Ablation, first attempt
- **Verdict**: INVALID. Six of seven arms were byte-identical because `amp.yaml` has no `task:` section and an AMP run reads `amp_task:`. The giveaway was results being *identical* rather than merely similar.
- **Fixed**: `TASK_SECTION` retargeting in `scripts/ablate.py`, plus an Oracle check.

### E10  2026-08-13  Critic quality misdiagnosed
- **Verdict**: INVALID measurement. `explained_variance` reduced to `1 - Var(A)/Var(V+A)` because `returns = advantages + values`. Read +0.95 where the honest value was +0.72. Also measured against the deterministic policy when the critic fits the noisy one.

### E09  2026-08-13  Conformance audit against shipped code
- **Why**: prompted by "why can't you just go learn publicly available code that works".
- **Changes**: `w_torque` -1e-5 (XBot scale), `w_dof_vel` -1e-4, `w_dof_acc` -1e-7, `w_orientation` -1.0, `w_swing_height` -20.0, `w_feet_distance` -3.0, `w_dof_pos_limits` -5.0.
- **Rejected during the audit**: T1's `w_torque = -2e-4`, measured at -9.4/step against a +6.4 budget.
- **Deferred and still open**: asymmetric actor-critic. The actor is blind to base linear velocity, which the references give the critic as privileged information. Named the #1 sim2real divergence.

### E08  2026-08-13  AMP from a non-walking policy
- **Result**: discriminator accuracy 0.53 → 0.98 in 97 iterations, `acc_real` pinned at 1.00, gradient penalty collapsed 4.67 → 0.65, style reward FELL 0.54 → 0.31 while task return rose. Slowing the discriminator 4x only delayed it to iteration 233.
- **Feature-level diagnosis**: overall real-vs-policy separability only 0.28 with no dominant artifact, most separable feature the neck. The discriminator was correctly reporting that the policy does not move like a person.
- **Verdict**: WORSE. AMP polishes a walk into a human walk; it does not turn a shuffle into a walk.

### E07  2026-08-13  Heading term unlearnable
- **Found**: the command's third component is a turn RATE, so "0" never meant "keep facing that way". The integrated target spun away and the error was uniformly random (sin/cos variance 0.5000, i.e. exactly chance).
- **Fixed**: XBot heading command, target direction drawn with the command, yaw rate recomputed each step from the wrapped error.

### E06  2026-08-13  Observation normaliser froze
- **Found**: `RunningMeanStd.COUNT_MAX = 1e6` meant a warm-started policy's normaliser drifted 50x faster than the statistics it was meant to track.

### E05  2026-08-13  Exploration noise was frozen
- **Found**: all 28 `log_std` components sat above `log_std_max`. `torch.clamp` passes no gradient outside its range, so the parameter was a gradient sink for 610 iterations.
- **Fixed**: in-place clamp after the optimiser step.

### E04  2026-08-13  Gait rhythm clock
- **Change**: Siekmann periodic reward composition, `w_gait_phase 1.0`, frequency coupled to speed by Inman's square-root law.
- **Why**: a human watching a video said it "walks like a horse". Measured: it swapped which foot led 0.42 times a second against 7.4 foot strikes. In a real walk those are equal.
- **Verdict**: WORKED on rhythm.

### E03  2026-08-13  Symmetry loss as a reward penalty
- **Verdict**: WORSE. The term is zero for any policy whose output ignores its input, so "hold a symmetric pose" is a global optimum, and PPO found it: a 61 cm two-footed brace, both feet loaded 90% of the time, 6 cm of travel per foot strike.
- **Replaced with**: mirror data augmentation. Reflecting a transition of a symmetric body gives a genuine transition, so it is unbiased, and a constant symmetric policy earns no return on either copy.

### E02  2026-08-13  `gait_symmetry` was gameable
- **Found**: defined as `min(left,right)/max(left,right)` over *stance* fraction, which is 1.0 for a humanoid standing on both feet. A policy under symmetry pressure scored 0.91 while taking no steps.
- **Fixed**: defined over SWING time, so never lifting a foot scores 0.

### E01  2026-08-13  Every video was 0.4x slow motion
- **Found**: 125 Hz physics encoded at 50 fps. Every video in the project's history played at 40% speed.
- **Invalidated**: every visual gait judgement made before it.
