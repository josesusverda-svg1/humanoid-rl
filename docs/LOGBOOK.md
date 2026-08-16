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
| ~~Single-run A/B on training outcome~~ | ~~INVALID~~ **RETRACTED** | E22b, retracted in E31 | E31 ran two configs byte-identical through 300 iterations (same returns to 4 decimals, same std). Training IS seed-deterministic. The 7x "noise floor" was never real and has been used to dismiss effects since; anything dismissed by it needs rechecking. |
| `shaping_gamma = 1.0` against `ppo.gamma = 0.99` | WORSE | E31 | Breaks the telescoping in the DISCOUNTED sum, which is the one PPO maximises. Measured at 44.7 discounted vs 22.3 undiscounted, 58% of the whole reward signal, all of it earned by cycling up and down. Potential shaping must use the RL gamma. |
| Ungated `upright` | WORSE | E31 | 306 of 306 non-shaping reward, and it is paid in mid-air: pelvis orientation with no ground-contact gate while `rise`, `quiet` and `posture` are all gated. |
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
| **`gamma = 0.99` copied from a 50 Hz repo onto our 125 Hz loop: a 0.80 s horizon, not 2.0 s** | E31 | **Every get-up run.** `hold_seconds = 2.0` sits at 250 steps, discounted to 0.081; a full get-up-and-hold to 0.00053. The task's own success criterion was outside the agent's horizon the whole time, so no reward change could have reached it. Same root as E19. | not yet fixed |
| Potential shaping checked for telescoping in the UNDISCOUNTED sum | E31 | `getup.py:159`'s claim that "oscillating the pelvis up and down pays exactly zero". True undiscounted (22.3 net of 613.8 gross), false discounted (44.7). The proof only holds when the shaping gamma equals the RL gamma. | not yet fixed |
| **The hold shove (0.6 m/s into qvel) violated the 0.4 m/s success conjunct by arithmetic, every attempt, re-arming on every miss** | E33 | **Every "he cannot hold" conclusion in twelve runs.** `standing_frac` 0.0% and `held_ever` 0 measured a predicate that was unsatisfiable by construction, not the policy. 3.0 of the 4.5 positive reward budget was unreachable the whole time. E29-E32's verdicts about the hold are void; their pump/cap findings stand. | `hold_push_grace` (velocity conjunct only) + `push_vel_xy` 0.7 -> 0.3 + Oracle `external_impulses_cannot_void_the_hold` |
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

### E47  2026-08-16  The reference was unexecutable. Stop authoring, start searching.

**E42-E46 VERDICT: INVALID at the source.** All four runs imitated a reference film that no
controller can perform. The film was built by recording a scripted DESCENT (stand -> squat ->
seated -> supine) and reversing time, on the argument that a reversed feasible trajectory is
feasible. That argument is wrong, and the measurement is not close: commanding the reversed
film's own joint angles as servo targets, starting from its own first pose, the pelvis went
**0.094 -> 0.111 while the film went 0.095 -> 0.879**, falling 0.18 m behind by 23% of the
way through. Descending is gravity-assisted. Rising fights gravity with the same actuators.

This voids E46's headline number. Film distance median 0.99 was a real measurement of a
meaningless quantity: the policy rode 99% of a film's phase while staying ~15 cm below it,
which is the only thing available to a policy chasing an impossible target.

**Second attempt, also failed, and this one is the useful failure.** Authoring the film by
RISING instead (waypoint families tuck -> squat -> half -> stand, ramped servo targets from a
bank lying pose, accepted only if the pelvis ends >= 0.82 after a settle): **0 clips in 1200
attempts, best pelvis 0.181.** Hand-designed waypoints do not stand this body up.

**The change: `scripts/search_getup_trajectory.py`.** Stop authoring a reference; search for
one. Cross-entropy method over open-loop servo-target trajectories (5 waypoints x 28 joints,
each ramped over 0.75 s, in the env's own [-1,1] action units, through the env's own filter
and decimation so anything found is reproducible by a policy), then a 1.5 s hold at the
nominal stand where the body must stay up on its own. Either a feasible get-up exists and the
search finds it, or a serious search fails and THAT is a finding about the body, measured
rather than assumed after each new reward idea.

**What the first searches established, which nine months of reward engineering did not:**

| | |
|---|---|
| First random population, peak pelvis | 0.641 |
| After 22 generations, pelvis at the end of the hold | 0.722 |
| After 33 generations (run 2), pelvis at the end of the hold | **0.874** |
| Standing height of this body | 0.877 |
| Best the policy ever reached from the floor, 16 runs | 0.417 |

The body can be stood up. That was never in evidence before.

**Three defects found in the search itself, all fixed, two of them the project's own
recurring classes:**

1. **CEM converged its own sampler, not the problem.** Sigma fell 0.70 -> 0.17 by generation
   20 and the next 20 generations resampled one basin. Fixed with injected exploration noise
   decayed over the run.
2. **A plain elite mean cannot be moved by one outstanding sample.** The 0.874 candidate sat
   inside an elite of 76 whose mean final height was 0.127, so the sampler kept drawing
   around a posture its own champion had already beaten. Fixed with CMA-ES-style rank
   weights plus a reseat-on-the-champion after 8 stale generations.
3. **The score was gamed by a jump, caught live.** Generation 44 held a champion at pelvis
   0.874; generation 50 replaced it with one at 0.235 whose peak was **1.017, above the 0.877
   standing height, therefore airborne**. Mean pelvis height over the hold is earnable by
   flight. This is the same class as exploits 1-10 and it appeared within an hour of writing
   a fresh objective. Fixed three ways at once: the MINIMUM height over the hold carries the
   weight (a body that leaps and collapses has a low minimum however high its mean), the peak
   term is capped at standing height so exceeding a stand buys nothing, and flight time (both
   feet clear of the floor) is subtracted outright.

**Acceptance is the task's own 13-conjunct standing predicate at the hardest exam level, true
for >= 90% of the hold.** Not a height. Height thresholds in this project have been satisfied
by a headstand and by knees locked backwards; the predicate rejected both.

**Standing instruction that follows from E47**: a reference clip is never trained against
until it has been executed, frame by frame, by the same servos the policy will use, and
looked at by a person. The executability test costs seconds and its absence cost four runs.

### E34  2026-08-15  PRE-REGISTERED BEFORE RESULTS: the reward, rebuilt as one package

**Run**: `runs/getup-20260815-165218` (second attempt; the first,
`runs/getup-20260815-161332`, was killed at iteration ~400 by the pre-launch review's
follow-up, see the addendum below). Written before any result.

**ADDENDUM, same day.** The adversarial review of the implementation (5 angles, 0 blockers)
measured ONE important defect on the live first attempt and it forced a restart:
**synchronized episode boundaries starve the standing starts.** With no early termination
every env truncates on the same step, so the 30% standing resets arrived as a wall once per
~104 iterations; `reward/stand` was exactly 0.0 in 332 of 349 iterations, each wave spiked
KL to 0.09-0.18 against the 0.01 target (slashing the LR), and by the third wave the trained
policy destroyed a standing pose within one iteration. The mechanism the 30% exists for
never engaged. Fix: `stagger_initial_episodes` (random initial episode phase per env,
training env only; a staggered EVAL env would bias every episode metric). Verified: 256 envs
over 60 steps produce 6 single-env truncations (the exact expected trickle) instead of one
256-env wall, and the progress line's `ret` now updates live instead of freezing for 104
iterations. Review also fixed: `getup_conjuncts.py` used RAW foot force for clauses 7-8
(over-reported 405 vs 317 steps on the E32 launch policy; now height-masked like the
predicate), `getup_snapshot.py` printed unmasked force and a max over all 28 joints as
"knee" (now masked, and the actual knees), constructor guards against inverted/degenerate
corridor edges (smoothstep would NaN silently), preflight's watched-metrics list extended
with the E34 surface, and the watch table now prints `gate%` and `ovspd%`.
Review verdicts otherwise: independent pose-table rebuild CONFIRMS the ladder is monotone
(stand nets +5.39/step, kneeling pays at or below the floor, slow pumping nets 0.043/step
and loses to parking, parking loses to climbing, so the gradient points up everywhere);
no reward exploit found across seven attack angles; the latch is ornamental (~4% of the
carrot; w_stand+w_hold do the work); the floor gradient is carried by the shaping
(~2-3 sigma when sustained across a horizon), the lift term only engages above 0.35 BW.

**The package** (adversarially designed: 4 designs x 2 attackers x critic, plus the user's
force mechanism; every element traced to a measured exploit):

| Change | Kills |
|---|---|
| `upright` and `rise` DELETED; one dense `lift` term: height of min(pelvis, head) through a force corridor | E32 curl (orientation paid lying down), E29 headstand |
| corridor opens 0.35-0.75 BW, **closes 1.6-2.4 BW** | E33 feet-press-lying (1.0+ BW supine), jump take-offs (2.5-6.2 BW) |
| foot force HEIGHT-MASKED everywhere (feet above 0.10 m read zero) | 13 BW mid-air sensor ghost |
| `stand` 3.0/step behind U with soft quality [0.6, 1.0] | stillness-as-gate bugs (shipped twice) |
| `hold` seniority 0->2.0 linear over the 2 s | jump-flash standing instants (E33) |
| `latch` 40 once, on hold completion | visible only at gamma 0.9985 (E31's lesson) |
| penalties damped 10x on the floor | E32 corpse selection (struggling cost more than stillness) |
| `launch` fine: upward root velocity above 1.0 m/s, quadratic | 5 m/s ballistic get-up; 1.0 m/s = a 5 cm hop = the user's stated tolerance |
| shaping phi = lift, shaping_gamma = ppo.gamma, weight 150 -> 60 | E31 pump (58% of signal) |
| 3 task observations: per-foot force, pelvis height | the policy was graded on instruments it could not see |
| standing_reset_frac 0.15 -> 0.30 | the big salary was rumour: never once experienced in 13 runs |

**Verified before this entry**: pose table (STANDING 3.697/step; supine, prone, side, seated,
E32 curl, E32/E31 launches ALL at or below -0.006, the curl's pressed feet closed out by the
corridor); preflight 17/17; Oracle **zero contradictions** (first time); 67-iteration smoke,
all 7 terms logged, obs 95 -> 98 through the real trainer, mirror confirmed off for getup.

**Pre-registered predictions:**

1. **Floor reward is near zero and NEGATIVE early.** Expected, not a failure signal. The
   floor is deliberately silent; early eval return ~-50 (drain + penalties) is by design.
2. **`gate_frac` and `launch_overspeed_frac` are the watch metrics.** Jumping dying =
   overspeed fraction falling from E33's ~60% toward single digits.
3. **PRIMARY: `standing_frac` > 0 by iteration 2000** (now reachable AND priced: 5.0/step
   behind U against a floor of ~0). Secondary: `held_ever_frac` > 0 by 2500.
4. If the policy parks (crouch or kneel, never completing U), the pre-registered response is
   the reset distribution (mid-rise poses into the bank), NOT a new reward term. Adding
   terms is how exploits 1, 2, 4, 5, 7 and 10 shipped.
5. `action_std` at or below 1.00 throughout (E30 cap).

**Known accepted risks, stated up front**: the floor's dense dynamic range is ~0.003/step,
possibly too flat for PPO to find the rise without mid-rise resets (response pre-registered
above); a near-stand failing one conjunct collects lift ~0.59/step forever (watch `foot_sep`,
`knee_max`); the latch is 40 sigma-ish in advantage terms on the step it fires (GAE and value
clipping absorb spikes, and it fires at most once per env per episode).

### E35  2026-08-15  PRE-REGISTERED: the ladder deployed. E34's verdict on the way in.

**E34 verdict (run getup-20260815-165218, stopped at iteration ~1080): MIXED, leaning
worked-as-designed.** Scored against its pre-registration:

| # | Prediction | Outcome |
|---|---|---|
| 1 | floor near zero, negative early, by design | YES: returns -50 to +80, no exploit ever paid |
| 2 | ovspd% falls = jumping dies | **YES, immediately: 0.0-0.8% on every eval vs E33's ~60%. The launch fine plus closed corridor killed ballistics outright.** |
| 3 | standing_frac > 0 by 2000 | NO by 1050: 0.0% on all 21 evals |
| 4 | if parked: reset distribution, not a new term | **TRIGGERED at 1050** (parking stable from ~450, 600 iterations, pelvis 0.09-0.16, std quieting 0.75 -> 0.42) |
| 5 | std at or below 1.00 | YES: 0.42-0.75 throughout |

The reward no longer pays ANY cheat (jumping, curling, pressing, all dead: the best the
policy found on the floor was "lie still at ~0/step"), but the floor's honest slope alone
did not pull PPO up the ladder within 1000 iterations. That is exactly the accepted risk
E34 stated up front, with exactly the response it pre-registered.

**E35 change, two yaml lines**: `bank_path: bank_v2.npz`, `midrise_reset_frac: 0.25`.
Resets now: 30% standing, 25% ladder rungs (all-fours / kneel / half-kneel / squat /
crouch, mined as physical equilibria, LOGBOOK E34b), 45% floor. Success still counted ONLY
from genuine floor starts.

**Pre-registered predictions for E35:**
1. **Rung starts collect `lift` immediately** (crouch pays ~0.16/step through an open
   corridor), so `reward/lift` batch mean jumps 10-100x from E34's ~0.0005.
2. **standing_frac > 0 by iteration 1500**: a crouch start is 2 rungs from a stand, and the
   critic now tastes those states every batch.
3. The gradient chain runs downhill: crouch envs learn stand -> squat envs learn crouch ->
   floor envs learn to reach a squat. Watch `eval/root_height` masked on floor starts LAST;
   it moves only after the upper rungs are mastered.
4. Failure mode to watch: rung starts collapse instantly under the noisy policy (like the
   standing starts did early), delivering nothing. Check `reward/lift` in the first 100
   iterations; if it is NOT elevated vs E34, the rungs are dying before paying and the fix
   is a shorter settle horizon on the rung poses, not more of them.
5. std at or below 1.00 (E30 cap), unchanged.

### E44  2026-08-16  The imitation was assembled without two of its standard parts

**E43 verdict: WORKED as a fix, insufficient as a run.** Making the reference observable
did exactly what it should: `track` climbed monotonically 0.012 -> 0.088 across twelve
100-iteration windows with no reversals (E42, blind, sat at 0.005-0.009 and oscillated). But
the user called it: the outcome had not moved, and the arithmetic says why.

**What track 0.088 means physically.** The term is `2.0 * exp(-err_sq / 2.0)`, so:

| paid | joint error |
|---|---|
| 0.088 (E43's level) | 27.1 deg per joint |
| 1.000 | 12.7 deg |
| 1.800 | 5.0 deg |

and the slope where the policy actually lives:

| joint error | our kernel paid | at sigma^2 = 8 |
|---|---|---|
| 30 deg | **0.043** (2% of max) | 0.766 |
| 20 deg | 0.363 | 1.306 |
| 15 deg | 0.766 | 1.573 |

The policy sat at 27 deg, i.e. on a slope 2% of the term's height. It was climbing, on a
gradient too flat to finish inside any budget we have. **The kernel was mis-sized, not the
idea.**

**And the part that was simply missing: early termination on tracking failure.** An env
that lost the film in its first second kept running the remaining ~2400 steps collecting
nothing, so most experience was gathered far from the reference. Terminating on deviation
has been standard since DeepMimic for exactly this reason, and we had none: `terminated_batch`
was a NaN guard, by a deliberate decision made when falling was the only thing that could
end an episode.

**E44 changes, both standard, both measured before launch:**
- `track_sigma_sq` 2.0 -> 8.0.
- `track_fail_err_sq` 12.0 (|dq| ~ 37 deg/joint): film-riders terminate on drift; nothing
  else ever terminates. Verified: 26 film terminations and **0** non-film terminations over
  400 idle steps.
- **Second-order defect caught in that same verification**: with early termination a film
  episode lasts seconds while an ordinary one lasts 20 s, so returning lost riders to the
  ordinary 30% draw starved the tracking population (measured 69 -> 29 on-film envs in 400
  steps, still falling). Lost riders now respawn ON the film; population holds at ~15% of
  envs, and the residual decline is by design (clips that play to the end hand off to the
  standing salary).

**Run**: `runs/getup-20260816-141541`, warm start from E43 iter_00001300 (observation width
unchanged, so the warm start is valid). First 50 iterations read `track` 0.20, already 2.3x
E43's endpoint.

**Pre-committed decision point, not to be softened**: if `track` has not reached **0.8**
(14 deg/joint, recognisable imitation) within 2500 iterations, imitation is declared a dead
end for this project and the next move changes the problem statement, not another parameter.

### E43  2026-08-16  E42 VERDICT: INVALID (unobservable objective). The fix, and a near miss.

**E42 stopped at ~2100. Verdict INVALID, not "worse": the term it was built around could not
be learned by construction.** `reward/track` crept 0.005 -> 0.009 of a possible 2.0 over
2000 iterations. The diagnostic (all envs spawned on the film, deterministic policy):

| step | joint err² | \|dq\| rad | film pelvis | body pelvis | track |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.144 | 0.144 | 1.000 |
| 30 | 3.422 | 0.350 | 0.146 | 0.151 | 0.197 |
| 200 | 7.551 | 0.519 | 0.341 | 0.165 | 0.006 |
| 600 | 24.236 | 0.930 | 0.479 | 0.164 | 0.000 |

The spawn is exact, and 0.24 s later the body is already 0.35 rad/joint away. **The policy
never sees the reference.** Its 98 observations carry joints, velocities, gravity, foot
forces and pelvis height, and nothing about which film is playing or what pose is due.
Every imitation system since DeepMimic feeds phase and target pose; without them a tracking
term is a lottery, and no weight fixes that. This is E34's lesson repeated on a new term:
**a reward may only depend on what the policy can observe.**

**E43 fix**: `task_obs_dim` 3 -> 33 when a reference bank is loaded: an on-film flag, the
phase, and 28 target-minus-current joint deltas (a control error, not an absolute pose the
policy would have to difference itself). Off-film envs get an all-zero block, disambiguated
from perfect tracking by the flag.

**Caught before launch, and it would have wasted the whole run**: `ThreadedVecEnv` read
`task.task_obs_dim` BEFORE calling `configure_for_prepared`, so a task whose width depends
on data loaded during configuration reported its unconfigured width. The reference channels
would silently not have existed while `observe_batch` wrote into a too-narrow buffer.
Configuration now happens before the width is read. Verified end to end: obs_dim 95 -> 128,
flag exactly 1.0 on-film and 0.0 off, phase spans 0.002-0.987, deltas exactly 0 at spawn
(the spawn IS the reference) and grow to 0.247 mean after 60 idle steps: the gap the policy
is paid to close is now visible to it.

**Also fixed while verifying** (a dashboard defect with the same shape): the console's
evaluation panels were empty because the trainer writes eval results INTO the training row,
and the API split rows with `if/elif`, so every eval row was swallowed. Both the get-up
series and the cross-run compare endpoint were affected.

**Run**: `runs/getup-20260816-131918`, COLD start (the observation width changed, so no warm
start is possible). Predictions: (1) `reward/track` rises by orders of magnitude, not
percent: the first 50 iterations already start at 0.295 before the untrained actor spoils
it; (2) film-riders complete level-0 holds at the film's end, the ladder promotes; (3)
PRIMARY: `standing_frac_strict` > 0 and the first full floor-to-stand-to-hold; (4) failure
mode now genuinely testable: if track saturates near 2.0 while floor starts stay at zero,
the policy has learned to be a puppet on-film and nothing off it, and the response is to
lower `track_reset_frac` and lean on the salary.

### E42  2026-08-16  PRE-REGISTERED: full-reference imitation, the conceptual change

E41 (EMA promoter) was cut short at ~2300 by the user's verdict on the whole approach, and
the verdict was fair: skill fragments kept accumulating (E41@2100 probe: 17/80 deterministic
level-hold completions, rung hold streak 90, both records) but no full floor-to-stand ever
appeared, because no mechanism ever taught the SEQUENCE. Twenty runs of reward shaping
cannot substitute for the thing every published get-up system uses: a motion reference.

**The reference, synthesized without mocap** (`scripts/generate_getup_reference.py` ->
`data/fallen/getup_refs_v1.npz`): record a gentle scripted DESCENT stand -> squat -> seated
-> supine (servo ramps between bank anchors; per-stage retries from state snapshots; the
squat stage uses build_pose targets with randomized ankles because bank equilibrium anchors
made the path free-fall 39/40 times), speed-filtered (never >1.4 m/s, never >1.0 sustained
past 0.1 s: the plop into the squat is a discrete event that ramp speed does not remove),
then REVERSE TIME. Result: 8.8 s supine (0.095) -> press-up -> tuck -> squat -> stand
(0.879), visually human (frame strip shown to the user), every frame a state this body
actually occupied, rise speeds inherited under the launch fine's free line. 2 clips for
now; yield improvement deferred.

**The integration**: `track_reset_frac` 0.30 of episodes spawn ON the reference at a random
phase (early-weighted, phase^1.5) and an eighth reward term `track` = w_track *
exp(-joint_err^2/2) * exp(-dz^2/0.02) pays for staying near the film as its playhead
advances. Finite and monotone: it ends at the stand and hands off to the salary, so it
cannot be farmed by cycling. Verified before launch: 29.5% of resets on-film; track = 2.00
exactly at spawn and 0.0000 for every other env; a mannequin diverges to 0.25 within 120
steps, so the follow-the-film gradient is real and measured. Pools now: floor 0.30, track
0.30, standing 0.20, midrise 0.10, rising 0.10. Exam ladder + EMA promoter stay from E41.

**Run**: `runs/getup-20260816-120145`, warm start from E41 iter_00002331, budget 900M.
Predictions: (1) reward/track climbs from its 0.02 start as the policy learns to ride the
film (it starts at 2.0 and the film runs away; recovery of tracking = learning); (2) the
ladder FINALLY promotes: film-riders complete level-0 holds en masse at the film's end; (3)
PRIMARY: standing_frac_strict > 0 and the first FULL floor-to-stand-to-hold under the film's
guidance; (4) failure mode: the policy tracks the film loosely for the pay but bails before
the top; visible as track plateauing near ~1.0 with the ladder stuck, response: raise
w_track or slow-phase RSI, decided at the verdict.

### E41  2026-08-16  PRE-REGISTERED: the churn measured, the promoter smoothed

**The measurement the E40 verdict asked for, done first.** Across the last five checkpoints
(100 iterations apart): catch rate 72/80-80/80, STABLE; hold completions 0-21/80, churning.
Latch-rate over the whole run: waves 2-86% with 17 threshold crossings, median consecutive
latch streak 2 iterations, run-average 39%. KL correlates only weakly (-0.27). Verdict on
the two candidate reads: **the catch skill accumulates and does not decay (worst version of
(b) refuted); the hold-under-shove outcome is intrinsically noisy (stochastic shove timing
and direction), and the promotion criterion sat on top of that noise demanding 1200
CONSECUTIVE steps: a gauntlet the measured signal passes never, while its average clears
the bar by 39x.** The brittle promoter (a) is the disease; entropy/LR (b) stays untouched.

**E41 change, single**: promotion by EMA (timescale exam_promote_steps = 1200 env steps)
instead of a consecutive-step streak. Tested on the measured churn shape (alternating
3%/0%, 200-step windows): promotes in ~1000 steps where the old criterion promotes never;
a steady 0.5% (below bar) and placed stands still never promote. Warm start from E40 final
(`iter_00009155`), budget 900M. Also `keep_last_checkpoints` 5 -> 24, because E40's probe
history was destroyed by the 5-checkpoint window and the churn analysis had to be
reconstructed from metrics alone.

**Predictions**: (1) promotion 0 -> 1 within ~1000 iterations (the earned frac is already
oscillating around the bar); (2) each new level initially drops the latch rate, then
recovers: that is the curriculum working, not regression; (3) PRIMARY: level 2+ by run end
and the first nonzero standing_frac_strict; (4) risk: promotion into oscillation stalls at
some level with EMA hovering just under the bar; if the ladder sticks mid-level for 3000+
iterations, the next lever is per-level exam_promote_frac or hold-outcome variance reduction
(narrow the shove window), decided then.

### E40 FINAL, iteration 9155: verdict MIXED, and the most progress of any run

**Run** `runs/getup-20260816-020136` (rising bridge + exam ladder, warm start from E39,
900M steps). Against the four predictions:

| # | Prediction | Outcome |
|---|---|---|
| 1 | rising starts caught within ~500 iterations | **YES**: the latch (level-hold completion) fired from iteration ~460 onward, and in total in **3580 of 9155 iterations**, the first hold completions in project history |
| 2 | earned promotion 0 -> 1 | **NO**: the earned frac (>1% for 1200 consecutive steps) never sustained; level 0 the whole run |
| 3 | skills flow down the ladder | PARTIAL: floor probe went 0.00% -> 0.07% -> 0.50% -> 0.82% U-frac with max hold 131 (nonzero floor standing for the FIRST TIME in any probe, growing monotonically); midrise 0 -> 0.17%/32 |
| 4 | spoiling (catch rate collapses) | NO: catch rate oscillated 55 -> 92.5 -> 60 -> **97.5%** (78/80 at the final checkpoint), ending near-perfect |

**What the bridge bought, measured across the night's probes:**

| checkpoint | floor U / hold | standing U / hold | catch rate | det. hold completions |
|---|---|---|---|---|
| 2000 | 0.07% / 11 | 0.81% / 12 | 55% | 0/80 |
| 4000 | 0.50% / 85 | 9.82% / **238 of 250** | 92.5% | 11/80 |
| 6100 | 0.26% / 93 | 7.80% / 188 | 60% | 4/80 |
| 9155 | **0.82% / 131** | 1.42% / 19 | **97.5%** | 6/80 |

The catch skill is real and by the end near-perfect. Floor standing exists and grows. The
standing-pool numbers OSCILLATE wildly between checkpoints (238 -> 19), which is the
remaining story: the policy cycles through skill configurations instead of accumulating
them, and the promotion criterion (1% earned for 1200 CONSECUTIVE steps) never survives the
churn even though the average is near the bar. Level-0 holds completed in 39% of all
training iterations, yet never steadily enough.

**For the next session, two candidate reads, both recorded rather than decided at 6 a.m.:**
(a) the promotion criterion is too brittle for an oscillating learner (1200 consecutive
steps of a noisy 1% signal is a coin-flip gauntlet; an EMA-based criterion would promote on
sustained average instead); (b) the oscillation itself is the disease (KL-driven LR +
entropy churn destroys skills as fast as they form; candidate levers: lower entropy_coef,
LR schedule, or freezing exploration once latch frequency is high). Measure (b) before
touching (a): if skills genuinely decay between checkpoints, a smoother promoter would just
promote into a regressing policy.

### E40 PREPARED while E39 finishes: rising starts, the discovery bridge

`scripts/generate_rising_states.py` -> `data/fallen/bank_v3.npz` = bank_v2 + 160 mid-rise
states WITH upward momentum, built by time-reversing the generator's physically honest
descents (stand -> squat/crouch) and sampling phases 35-90% of the way up. EVERY kept state
proves itself by completing: plain servos holding the standing target must finish the rise
from it (pelvis >= 0.70 within 1.2 s). Pelvis span 0.22-0.88, upward velocities +0.05 to
+2.09 m/s (tails above 1 m/s briefly meet the launch fine, which teaches exactly the
deceleration a catch is made of).

Task support: `rising_reset_frac` (default 0.0), fourth reset pool. Rising starts share the
midrise flag, so they are EXCLUDED from the success denominator (the start was given) but
COUNT toward ladder promotion (catching a given rise into a stand is precisely the skill;
persisting in a given stand is not, and stays masked). Verified: pools draw at configured
fractions, momentum arrives through the reset path, success at level 0 stays zero, default
config byte-path unchanged.

**E40 REVIEW FOUND TWO BLOCKERS; both fixed and re-verified before deploy.**

1. **The "phase" was a TIME index into descents that were not quasi-static.** Most recorded
   descents collapsed (root speed to 3 m/s, pelvis to 0.04), and a servo ramp loses almost
   no height early, so time-phase 0.8-0.9 sat at 98.5% of standing height: 73 of 160
   "rising" states passed the FULL standing predicate AT RESET (placed stands, the exact
   promotion-gaming exploit fixed once already), and the collapse tail time-reversed into
   ballistic launches that the completes() gate cannot reject (momentum alone completes).
   Fixed: descents rejected unless root speed stays under 0.6 m/s at every step (quasi-static
   or nothing), phase parameterized by HEIGHT, kept states capped at pelvis 0.73, BELOW the
   predicate's 0.745 height threshold. Re-verified: 0 of 64 rising starts pass the predicate
   at reset; 64 of 64 still get caught by zero action within 300 steps. The pool is now 80
   states, 0.61-0.72, squat family (crouch descents never pass the quasi-static filter).
2. **`on_batch_end` ticks once per ENV STEP, not per PPO iteration** (vec_env calls it
   inside step()), so the promotion streak "50 iterations" was really 50 steps = 0.4 s, 24x
   faster than documented. It never bit E39 only because earned stands were zero. Fixed:
   `exam_promote_steps` = 1200 (50 iterations x horizon 24), field renamed so the unit is in
   the name, comment states the call-site fact. Additionally `was_down` bookkeeping: a stand
   counts toward promotion only if the env was NOT-standing earlier in the same episode, so
   any near-stand spawn must lose the predicate before its standing can count (belt to
   from_standing's suspenders; an adversarial flag-stripped test showed was_down alone is
   insufficient, both stay).

**E40 pre-registration** (deploys when E39 completes, warm start from its final policy):
`bank_path: bank_v3.npz`, `rising_reset_frac: 0.15`, `midrise_reset_frac: 0.25`, all else
E39. Predictions: (1) rising starts get caught into level-0 stands within ~500 iterations
(the completes() test proves a trivial controller can; the policy has 98 obs of context the
servos lack); (2) the earned-stand promotion fires, level 0 -> 1, FIRST LADDER PASS in
project history; (3) skills flow backwards down the phase ladder: catches at phase 0.9
teach catches at 0.65, then rung starts start converting; (4) failure mode: the policy
LEARNS to spoil given rises (dropping is locally cheaper than catching under the effort
fine); watch reward/latch and the rising-start standing frac in probes; if spoiling is
systematic, the effort penalty during the catch window is the suspect, not the bridge.

### E39 checkpoint, iteration 3000: prediction 4 TRIGGERED, recorded on schedule

Level 0 at iteration 3008, zero promotions under the earned-stand criterion. The 2000 probe
says it precisely: even at the EASIEST exam (knee 1.30, hold 0.5 s), rung starts stand 0.01%
of steps with a best streak of 3, and floor starts 0%. The gap is not the exam's strictness;
it is the rise-and-catch transition itself, which random per-step exploration does not find
(and cannot: iid Gaussian noise at 125 Hz through an 8 Hz action filter is dither, not a
strategy; the coordinated 2 s, 28-joint push it would need to stumble on has effectively
zero probability). The run continues to completion per the pre-commitment.

**Next lever, as pre-registered: reversed-descent imitation.** The midrise generator already
builds physically honest quasi-static DESCENT trajectories (stand -> squat, ramped servo
targets, mass balanced the whole way). Played backwards they are reference RISE trajectories
in this exact body, no mocap needed. Directed exploration instead of waiting for luck: the
policy is paid for reproducing the reference from matching rung starts, which is what every
published get-up system does. Build begins while E39 finishes overnight.

### E38 ABANDONED at ~1600 of 9155; E39 PRE-REGISTERED: the exam ladder

**E38** (warm start + 3x time): the 1600 probe showed standing max hold 31 of 250 against
E37's 195: the warm start did NOT carry the streak skill (prediction 2 failed; the KL-11.65
unfreeze spike at iteration 31 is the suspect). The user called the wider verdict, and they
are right: sixteen runs, and he has never once stood up from the floor. Stopped.

**E39, the pre-registered u_knee lever, widened into what the published record actually does.**
HumanUP (RSS 2025), the only real-robot get-up, does not demand a strict exam on day one; it
lets the robot stand ANY way, then tightens. Ours demanded mastery from the first minute of
run one. The exam ladder: levels (knee 1.30, hold 0.5 s) -> (1.00, 1.0) -> (0.80, 1.5) ->
(0.60, 2.0 = the real exam, constructor-enforced). ONLY knee and hold relax; all other 11
conjuncts stay strict at every level. Promotion is achievement-gated (train-batch standing
frac > 1% for 50 consecutive iterations), training env only (width-gated: the shared task
instance must not be advanced by evaluations). **Reporting never relaxes**: eval carries
`standing_frac_strict` beside the level predicate, and success counts only full-exam
completions at the final level. Verified before launch: floor mannequin passes nothing and
success stays 0 even where standing-start mannequins complete the level-0 hold; promotion
fires after exactly 50 good wide batches; a narrow batch cannot advance the level; a ladder
not ending at the real exam is rejected by the constructor. Preflight 17/17, Oracle 0.

**AMENDMENT, 15 minutes in.** The first promotion criterion counted the WHOLE batch's
standing fraction, and the 30% placed standing starts promoted the ladder 0 -> 2 within the
opening hundred iterations while the latch sat at zero: promotions without a single earned
stand. Caught by the "two PROMOTED lines but latch 0/384" contradiction in the first watch
cycle. Fixed: promotion now counts only `standing & ~from_standing` (a stand reached from
the floor or pushed up from a rung; a placed stand proves nothing by persisting). Verified:
60 batches of placed-only stands promote nothing; 55 batches of earned stands promote to
level 1. Relaunched as `runs/getup-20260815-233654`.

**Run**: `runs/getup-20260815-233654`, warm start from E37 final, budget 900M (~9155
iterations, overnight). Predictions: (1) level 0 is passed and promotion 0 -> 1 happens
within ~1500 iterations (the current policy already flashes near-stands with bent knees);
(2) the latch fires many times at level 0 (it now marks level-hold completions); (3)
PRIMARY: the ladder reaches level 2 or higher by run end, with `standing_frac_strict` > 0
appearing once level 2+ is active; (4) failure mode: parked forever at level 0 (never 1% for
50 straight), which would say the gap is below even the easiest exam and the next lever is
reversed-descent imitation (the physically-honest rise trajectories already exist in the
midrise generator, played backwards).

### E37 FINAL, iteration 3051: verdict MIXED. The tuck taught the posture, not yet the push.

**Run** `runs/getup-20260815-202213` (tuck axis, height-masked). Against the pre-registration:

| # | Prediction | Outcome |
|---|---|---|
| 1 | tucking within ~300 iterations | **YES**: knee 2.25 / feet 0.83 BW / gate 81% at eval 250, and the tuck recurs through the whole run (knee 1.8-2.45 in most late evals). The posture the user prescribed is now part of the behaviour. |
| 2 | floor probe beats 0.417 by 1400 | NO: 0.358 at the 1200 probe |
| 3 | standing_frac > 0 | **NO: 0.0% on all 61 evals** |
| 4 | hook-lying corpse | No frozen tuck; the final policy rests lying with feet lightly pressed, cycling through tuck episodes |

**Highlights**: standing-pool hold streak **195 of 250** at the 1200 probe, the project record
(78% of the exam, 200 iterations earlier than either baseline). The latch never fired; hold
averaged 0.00019 with peaks above the E35 plateau but no sustained break.

**The honest mechanics, observed live**: holding a tuck costs drain until the rise completes,
so tuck episodes pay only when followed through; the policy tucks, cannot yet push through,
and relaxes back flat. The axis built the user's posture into the repertoire; the missing
piece is now purely the push-to-stand and the last stretch of the hold. Four full runs on
the E34 reward, ZERO exploits: the reward is holding. What has not been given is TIME: every
run trains 1.7 h from scratch and the skill curves (hold streaks 127 -> 195) are still
climbing when the budget ends.

### E38  2026-08-15  PRE-REGISTERED: same objective, three times the time, no reset to zero

**The lever is optimisation time, not another mechanism.** E38 warm-starts from E37's final
policy (`--init-from .../iter_00003000.pt`) with `critic_warmup_updates: 30` (the recorded
guard against the measured init_from KL blow-up: curriculum-20260814 hit approx_kl 24.45 at
iteration 1 without it) and `total_env_steps` 900M (~9150 iterations, ~5 h, overnight).
Config otherwise identical to E37.

Predictions: (1) no KL blow-up in the first 50 iterations (warmup working); (2) hold streaks
resume near 195 rather than restarting near 0 (the warm start carries the skill); (3)
PRIMARY: the latch fires at least once (first completed 2 s hold in project history);
(4) standing_frac > 0 on some eval. If after tripled time the latch still never fires, the
next lever is the u_knee curriculum (0.6 -> 1.2 eased, annealed back), pre-registered as the
last candidate before a design rethink.

### E36 VERDICT at 1400 of 3051: NO EFFECT. Stopped; E37 deploys the user's sequencing axis.

**Run** `runs/getup-20260815-191757`, midrise_reset_frac 0.40 (vs E35's 0.25), stopped at
iteration ~1400 on its pre-registered criterion. All three signals negative:

| Signal | Outcome |
|---|---|
| (a) reward/hold past E35's ~0.0003 plateau | NO: mean 0.0001 over iters 800-1400 (peak 0.00135, transient) |
| (b) latch fires once | NO: never |
| (c) standing_frac > 0 | NO: 0.0% on all 28 evals |

Pool probe, base-to-base at iteration 1400: floor 0%/0 (best pelvis 0.238 vs E35's 0.417),
rungs 0.02%/7 (still sliding down), standing 2.35%/34 (vs 6.9%/127; both runs oscillate,
but nothing about 0.40 is better and the pre-registered read is clean). More rung exposure
did NOT build the rung-to-stand link. SETTLED: raising midrise_reset_frac beyond 0.25 buys
nothing by itself; do not retry without a mechanism change.

**E37, deployed immediately (the user's insight, watching the videos): a second potential
axis, "feet tucked under the pelvis".** The user named what the metrics could not: "он не
понимает, что он делает" lying down, and prescribed the sequence: raise the torso a little,
TUCK THE FEET UNDER, then push up from that crouch, balancing. The mechanism: the height
axis is nearly silent for a supine body (no small motion changes min(pelvis, head)), so the
floor gradient pointed nowhere; the tuck axis (horizontal pelvis-to-feet distance mapped to
[0,1]) is loud from the very first supine centimetre and its completion IS the squat the
rest of the reward already pays. Implemented STRICTLY through the potential
(phi = lift + 0.25*tuck, gammas matched), never as a term: the adversarially-reviewed
staged-bonus design died to boundary farming, and the potential provably cannot be farmed.
Verified before launch: Phi monotone along the user's sequence (supine 0.08, seated 0.10,
all-fours 0.12, squat 0.70, crouch 0.77, standing 1.23); foot in-out cycling telescopes to
zero; a curl collects its tuck value once on the way in and never again.

**E37 AMENDMENT, 20 minutes in.** The first tuck implementation measured HORIZONTAL
pelvis-to-feet distance unmasked, and the camera caught the consequence within 300
iterations: a shoulder-stand ("candle", legs straight up) puts the feet at zero horizontal
distance while touching nothing, so the axis paid a pose with the feet in the air. The E34
rule (every foot quantity is height-masked) had been skipped on the new axis. Fixed
per-foot: a foot above u_foot_height contributes zero tuck. Re-verified: the candle now
reads tuck 0.00 (was ~0.9), the ladder stays monotone (supine 0.08, squat 0.70, crouch
0.76, standing 1.21). Old run killed at ~300 iterations; relaunched as
`runs/getup-20260815-202213`. Cost of the miss: 10 minutes. Also noted: all-fours reads
tuck 0 under the mask (feet planted far behind), which is correct: hands-and-knees is not
"feet under you", and the height axis prices that path instead.

**E37 pre-registration**: single change vs E36 = `shaping_tuck_gain` 0.0 -> 0.25.
Predictions: (1) supine floor envs start tucking within 300 iterations, visible as
`eval/knee_max` rising with pelvis LOW (hook-lying has bent knees) and in frames as
knees-up-feet-planted; (2) the floor pool probe's best pelvis exceeds E35's 0.417 by 1400;
(3) PRIMARY, same as ever: standing_frac > 0, now expected via floor-to-squat-to-stand;
(4) failure mode to watch: tucked-and-frozen (a hook-lying corpse); the tuck potential pays
it once only, so it should not stick, but if it does the response is NOT a new term, it is
episode-mix rebalancing. Next candidates if E37 fails: warm start from the best policy with
critic warmup, then a measured u_knee curriculum.

### E35 FINAL, iteration 3051: verdict MIXED. The first clean run in project history.

**Against the pre-registration:**

| # | Prediction | Outcome |
|---|---|---|
| 1 | reward/lift elevated 10-100x | YES: ~7-9x sustained (0.0033-0.0045 vs 0.0005), rungs paid all run |
| 2 | standing_frac > 0 by 1500 | **NO: 0.0% on all 61 evaluations, to the very end** |
| 3 | chain runs downhill | PARTIAL, see the probe: top cemented, bottom learned to sit, the middle link never formed |
| 4 | rungs collapse before paying | did not happen (they pay, then slide) |
| 5 | ovspd stays near zero | YES except honest push-up bursts at 1950-2050 (9-15%), which the launch fine priced back to 0.1-0.4% |

**What this run is, despite the failed primary**: the first full run with ZERO exploits end
to end. No jumping, no curl, no press, no headstand, no freeze-for-profit: the reward held
under 300M steps of optimisation pressure. Everything the policy did was honest: it sat up
(pelvis 0.42 from the floor, vs 0.16 flat in E34), it held placed stands to 127 of 250 steps
(vs instant destruction), it found the side-plank arm-prop transition on camera, and its
late-run push-up bursts were exactly the right idea at exactly the wrong speed.

**The wall, precisely**: the rung-to-stand link. From crouch and squat starts the policy
slides DOWN to sitting within 2 s instead of pushing up the last two rungs. All three pools
measured separately at 1400 (see the checkpoint entry): floor 0/96 stands, rungs 0.01%
U-frac, standing 6.9% U-frac with max hold 127.

**Honesty notes**: action_std drifted 0.38 -> 0.79 in the last thousand iterations (entropy
bonus pushing against a plateaued return); the cap held it. The mid-run sit-up and
side-plank behaviours did not survive into the final policy, which quieted back toward
low-lying activity. Held_ever never fired once, so the latch never paid, and the hold
seniority topped out around half.

**E36, deployed on completion**: `midrise_reset_frac` 0.25 -> 0.40 (the lever chosen at the
1500 checkpoint from the pool probe), everything else identical. Floor share drops to 30%.
The bet: the rung-to-stand link needs more attempts and a critic that tastes rung states
more often; if E36's probe still shows rungs sliding down at its checkpoint, the next
candidate is a warm start from this run's final policy with critic warmup, and after that,
easing u_knee (the strictest conjunct) as a measured curriculum, not a permanent softening.

### E35 checkpoint, iteration 1500: prediction 2 FAILED, recorded on schedule

`standing_frac` 0.0% at 1500, as the pre-committed rule anticipated it might be. The run
continues to completion (it costs nothing and its trends are the best in project history).

**The per-pool probe that decides the lever** (iter_00001400, 96 envs x 8 s per pool,
deterministic):

| start pool | U-frac | max hold | pelvis@2s | best pelvis |
|---|---|---|---|---|
| floor | 0.00% | 0 | 0.333 | 0.417 |
| midrise rungs | 0.01% | 7 | 0.327 | 0.685 |
| standing | **6.87%** | **127 of 250** | 0.238 | 0.907 |

Three facts fall out. (1) The standing pool is HALF-WAY to a completed hold: max streak 127
of 250, against "destroyed within one iteration" at E34's start. The top of the ladder is
being cemented. (2) The floor policy now genuinely sits up on its own: best pelvis 0.417,
holding ~0.37 at episode end, against 0.16 flat in E34. (3) **The missing link is
rung-to-stand: crouch and squat starts slide DOWN to sitting within 2 s** (pelvis 0.68 ->
0.33) instead of pushing the last two rungs up. The rungs pay (lift is collected, prediction
1 holds), the policy just does not yet know that pushing UP from a rung is worth more than
sliding down.

**Lever decision, per the pre-registration**: HIGHER midrise fraction (0.25 -> 0.40), not
the shorter-settle variant. The shorter-settle lever targets "rungs collapse before paying",
which the probe rules out (they pay; they collapse under the POLICY's own actions seconds
later). More rung exposure attacks the actual gap: more attempts at the rung-to-stand push,
and a critic that tastes rung states 60% more often. Deploys as E36 when E35 finishes.

### E34b  2026-08-15  The response lever, built BEFORE its trigger: bank_v2 mid-rise rungs

Prepared while E34 runs, so that if the pre-registered trigger fires (floor-parking stable
500+ iterations, or standing_frac 0.0% at 2000) the response deploys as two yaml lines plus
a restart, instead of an hour of tooling under time pressure.

`scripts/generate_midrise_poses.py` -> `data/fallen/bank_v2.npz` = bank_v1 (byte-identical,
all fields carried) + 300 ladder rungs: 60 each of all-fours, kneel, half-kneel, squat,
crouch, spanning pelvis 0.27-0.70 against a 0.877 stand.

**What building it taught, the hard way:**
- **Authored poses are mid-topple, not at rest.** Hand-guessed joint angles failed the bank's
  fixed-point validation 22-24 times of 24 at every settle length (the body is slowly
  falling the whole time). Squat and crouch are now REACHED by ramping servo targets down
  from a stand, and every kind is MINED: sample states along physical trajectories, keep the
  ones that pass validation. Physics picks the equilibria; the authored targets only steer.
- **The ankle was the load-bearing unknown.** With default (vertical-shin) ankles, zero
  squat/crouch equilibria exist in 801 mined descents: the mass stays behind the feet at
  every depth. The ankle target is drawn from its whole range and mining keeps what works
  (E25's lesson: never guess a sign convention).
- **A crouch has NO passive equilibrium on this model, and that is physics, not a bug.**
  801 descents, zero fixed points under the strict lying-pose contract. Humans stabilise a
  crouch actively too. Active-balance rungs (squat, crouch, half-kneel) therefore carry
  their own documented contract, `is_valid_active`: finite, penetration-free, and holding
  85% of pelvis height through the first quarter second. "Catch yourself mid-crouch" is a
  state the curriculum wants; "already fallen by the time the policy acts" is not.

**Task support**: `midrise_reset_frac` (default 0.0, so nothing changes until deployed),
three-pool draw in `reset_pose`, and midrise starts excluded from the success denominator
exactly like standing starts. Verified end-to-end: pools draw at configured fractions
(28.9% / 24.6% / 46.5% at 0.30/0.25), success counts 0 standing rows, 0 midrise rows, and
with bank_v1 at frac 0.0 the draw path is unchanged (0 midrise rows).

**Deployment, when and only when the trigger fires**: in `configs/getup.yaml` set
`bank_path: data/fallen/bank_v2.npz` and `midrise_reset_frac: 0.25`, restart, new entry.

### E33  2026-08-15  THE HOLD WAS UNSATISFIABLE BY CONSTRUCTION. For twelve runs.

**The defect.** The task's own anti-cheat shove made the success predicate impossible to
satisfy, for any policy, ever:

1. `hold_push_vel = 0.6` m/s is written by the engine STRAIGHT INTO qvel. Not through the
   actuators. No policy action can prevent or resist it.
2. Conjunct 13 of `U` requires `|v| <= 0.4` m/s. Measured over 69-72 shove events: |v| on the
   step after a shove is **0.62**. So the shove violates `U` by arithmetic.
3. The shove fires at `hold_steps` in [40, 140), always before the 250 needed.
4. The miss resets `hold_steps` to 0, re-arms `pushed`, redraws `push_at` from the same
   window. The next attempt is shoved identically. The cycle has no exit.

Plus the second arm: `domain_rand.push_vel_xy = 0.7`, same mechanism, invisible to the task,
~3.2 times per 20 s episode.

**Proof, both directions.** The model's own nominal stand, held by a fixed-target servo (a
controller that cannot balance at all), 64 envs:

| | best hold reached | completions |
|---|---|---|
| shove 0.6 as shipped | 137 / 250 | 0 of 64 |
| shove 0.6, DR off | 186 / 250 | 0 of 64 |
| **shove OFF** | **330 / 250** | completes |

The hold was reachable the whole time. The mechanism built to TEST it was TERMINATING it.

**What this invalidates.** `standing_frac = 0.0%` and `held_ever = 0` across twelve runs were
read, every time, as evidence about the reward, the exploration, or the discount horizon
(E29-E32 all did this, including yesterday's E32 verdict). They were evidence of nothing
except this defect. `w_stand` (2.0), `w_quiet` (0.5) and `w_posture` (0.5), i.e. 3.0 of the
4.5 positive budget, were unreachable by construction, so every policy ever trained here was
optimising the remaining 1.5, which is exactly the ungated `upright` + `rise` world the E32
curl exploited. The E30/E31/E32 findings about the PUMP and the CAP remain valid (they were
measured on their own terms), but every claim of the form "he cannot hold" is void.

**How it survived twelve runs of scrutiny.** Every earlier check verified reachability of the
POSE (Oracle `getup_hold_and_thresholds_are_reachable` checks thresholds against a settled
stand; the preflight checks the hold fits the episode). Nothing ever simulated the hold
BOOKKEEPING end to end under the shove. The one test that would have caught it, "can a
perfect stand complete the hold at all", did not exist until today.

**The fix**, two arms of one principle (no impulse the setup itself injects may void the
predicate by arithmetic):

- `U` split into `U_geom` (conjuncts 1-12, pose) and `U_vel` (13, motion). After a shove THIS
  TASK fired, `hold_push_grace = 50` steps (0.40 s) forgive `U_vel` ONLY. Pose conjuncts are
  never forgiven for a single step: a body that topples still fails instantly. The shove now
  tests what it was built to test, staying ARRANGED like a stand while being pushed.
- `domain_rand.push_vel_xy` 0.7 -> 0.3, below the 0.4 cap, because the engine push is
  invisible to the task and no grace can cover it.

**Verified after the fix**: servo max hold 137 -> **643**, completions 0/64 -> 3/64 (the
three whose geometry survived the shove; the servo cannot balance, so 3 is the honest
number). Zero-action mannequin on the real task: `held_ever` **0 of 128** over 1000 steps,
the anti-cheat is not resurrected. Preflight 17/17. New Oracle check
`external_impulses_cannot_void_the_hold` fires on the old config and passes the new one.

**Pre-registered predictions for the next run (single change: this fix; gamma 0.9985 and the
E32 reward stay exactly as they are, pump and all):**

1. **`standing_frac` > 0 for the first time in project history**, by iteration 2000. This is
   the primary prediction. The E32 curl attractor still exists and still pays 0.50/step, so
   the bet is specifically that 2.0 of newly-reachable `stand` (+ up to 1.0 of quiet/posture)
   against the curl's 0.50 changes which basin wins.
2. `held_ever_frac` > 0 by iteration 2500 on at least one evaluation.
3. If `standing_frac` is still exactly 0.0% at iteration 2000, THE REWARD is the remaining
   suspect (the curl basin), and the response is the deferred redesign (gate `upright`,
   shaping_gamma match), NOT another bookkeeping hunt.
4. `action_std` stays at or below 1.00 (E30 cap).
5. Honesty note: prediction 1 can fail simply because PPO never explores into the far basin
   in 3000 iterations. That outcome does not falsify the fix (the servo proof stands); it
   says the slope to the basin is the problem, which is what the deferred redesign is for.

**Deep review before launch** (5 adversarial angles, all measured against the live code): 0
blockers. The bookkeeping hand-trace matched a numerical drive of the real class; the grace
opens the step before the impulse lands and forgives exactly 49 steps; observations carry
none of push_at/push_grace/hold_steps (`task_obs_dim` 0), so nothing can be pre-braced. An
independent re-implementation reproduced 3/64 completions exactly, and the grace-0 control
reproduced the defect (158/250, 0/64), proving the grace is the load-bearing change. One
real hardening found and applied: residual grace survived a mid-attempt reset, letting a
deliberate one-step geometry break stack two graces into one counted hold (99 of 250 steps
above the cap); `push_grace[reset_now] = 0` closes it, re-verified. Also confirmed: the DR
push at 0.3 cannot void conjunct 13 even at the boundary (quiet-stand sway 0.04-0.10 + 0.30
<= 0.40), task and DR impulses never stack (task overwrites), and eval envs run DR-off while
the task shove still fires there, so evaluation tests the shove.

**Known dead config, logged not fixed**: `configs/getup.yaml`'s `task:` section is stale
locomotion keys GetUpTask never reads, and `no_config_section_is_silently_ignored` does not
cover the getup task (returns [] silently). Harmless for this run, but an ablation that
patches `task:` on a getup run would silently no-op, the exact six-identical-arms failure
that check was written for. Extend the check's mapping before any getup ablation.

**Run**: `runs/getup-20260815-150447`, launched 2026-08-15 15:04, ~1.5 h.

---

#### VERDICT: **MIXED**. The fix is proven; the reward then funded a jumping machine.

Stopped at iteration 1820 of 3051 by the user, whose words are the verdict: "он будет просто
прыгать. Ему выгодно просто прыгать и получать награду. Он все награды получает, но не даёт
нам то, что нужно."

**Scored against the pre-registered predictions:**

| # | Prediction | Outcome |
|---|---|---|
| 1 | standing_frac > 0 by iteration 2000 | **NO.** 0.0% on all 35 evaluations, to 1750. |
| 2 | held_ever_frac > 0 by 2500 | NO (run stopped at 1820, was 0.0% throughout). |
| 3 | if 0.0% at 2000: the reward's basin is the suspect, go to the redesign | **TRIGGERED** (called at 1800 by the user; the trend was flat). |
| 4 | action_std at or below 1.00 | Held: 0.583-0.995, never above. |
| 5 | fix not falsified by a floor-stuck policy | Held, and more: the policy DID leave the floor. |

**What changed behaviourally, and it is real progress in the wrong currency.** At iteration
~1050 the policy abandoned the curl (pelvis eval 0.20 -> 0.65 sustained for 700 iterations,
knees unfolding, feet loaded ~0.5 BW). At 1300 the frames showed a genuine transient
standing pose, the second ever seen in this project. It then converged not to standing but to
**ballistic cycling**: jump, flash an upright instant, crash, repeat.

**Jump forensics, measured on iter_00001700** (32 envs, 14 s, no DR):

| | his policy | human getting up |
|---|---|---|
| airborne share of all steps | **63.8%** | ~0 |
| apex pelvis height | median 1.48 m, max 1.65 | 0.877 (standing) |
| take-off velocity | up to **5.0 m/s** | 0.5-1.0 m/s |
| foot force at take-off | median 1.5 BW, max **6.2 BW** | 1.1-1.3 BW |

A 5 cm hop is 0.99 m/s of take-off velocity, so "human-fast" and "a 5 cm hop" are the same
1.0 m/s line, and his 4.65-5.0 m/s sits five times above it. Clean separation on both axes.

**Why he jumps: it is PAID, twice.** The shaping pump (shaping_gamma 1.0 vs ppo.gamma 0.9985)
still nets a profit per up-down cycle, and ungated `upright` pays in mid-air. E31 measured
both; E33 kept them deliberately unchanged to isolate the hold fix. The hold fix is proven by
construction (servo completes) and now the reward is the last suspect standing, exactly as
prediction 3 pre-registered.

**Next: E34, the full reward redesign as one verified package** (adversarially designed
2026-08-15, four designs x two attackers x critic, plus the user's force mechanism):
delete `upright` and `rise`; one dense `lift` term (min(pelvis, head) height) behind a
force-CORRIDOR gate (opens 0.35-0.75 BW, closes again 1.6-2.4 BW: human rise forces pay,
jump take-offs do not), evenness and stance-width factors; `stand` + seniority `hold` +
once-per-episode `latch` behind U; penalties damped 10x on the floor; a one-sided quadratic
`launch` penalty above 1.0 m/s upward root velocity (free at a 5 cm hop, ruinous at 5 m/s);
shaping on phi = lift at shaping_gamma = ppo.gamma, weight re-derived 150 -> 60; foot force
height-masked (sensor reads 13 BW in mid-air self-contact); foot force and root height added
to the OBSERVATION (the reward depended on quantities the policy could not see);
standing_reset_frac 0.15 -> 0.30 so the big salary is experienced from iteration 1.

### E31  2026-08-15  Exploration cap WORKED. He now jumps, and two measurements say why.

- **Run**: `runs/getup-20260815-104811`. `log_std_max` 5.0 -> 0.0 (the E30 fix), everything else as E30.
- **Stopped at iteration 1200 by the pre-committed rule** ("torso not inverted" below 40%). It was **16.8%** at iteration 1195, **21.8%** re-measured at 1200. The rule fired and I did not soften it, but the reason it recorded ("the `2*min` both-feet gate is too harsh") is **WRONG**, and two measurements below say what is actually happening. The rule was aimed at the previous failure and kept pointing there after the failure changed.

**What the fix bought.** VERDICT: **WORKED**, cleanly.

| | E30 (log_std_max 5.0) | E31 (capped at 0.0) |
|---|---|---|
| action_std, iter 300 -> end | 0.79 -> **3.43** | 1.000 -> **1.000**, every single check |
| return, iter 300 -> end | 833 -> **19** | 833 -> **886 peak** |
| torso not inverted | **0.1%** | **21.8%** |
| pelvis high | 0.8% | **42.1%** |
| head high | 0.1% | **33.4%** |

`torch.clamp` behaved exactly as predicted in E30: the parameter sat ON the ceiling for 900 iterations and was still pulled back, never once above 1.001.

**Determinism, and E22b is wrong.** The two runs were **byte-identical through iteration 300** (returns 81.3314, 163.7646, 323.6682, 585.0565, 556.8055, 833.2446 in both; std 0.740/0.728/0.785 in both), diverging only near 450 when the cap began to bind. Training IS deterministic under a fixed seed. E22b concluded from one identical-config pair (323 vs 2451) that "a seed does not pin a trajectory" and set a 7x noise floor that has been used to dismiss effects ever since. That conclusion is now **INVALID**; the 323/2451 pair differed by something other than the seed.

**The new failure, seen before it was measured.** Frames at iteration 1200, from the newest checkpoint:

| t | pelvis | head | feet BW | what |
|---|---|---|---|---|
| 0.0 s | 0.120 | 0.16 | 0.31 | on his back |
| 1.2 s | **1.164** | 1.10 | **0.00** | airborne, hands at 1.47 m |
| 2.4 s | 1.046 | 1.09 | **0.00** | still airborne |
| 4.0 s | **1.276** | 0.99 | 0.43 | second launch |
| 8.0 s | 0.371 | 0.18 | 0.00 | crashed |

Standing pelvis is 0.877 and the standing threshold 0.745. He reaches **1.28 m with zero ground contact**. "not ballistic" ran 0.3-12.1% all run and was the worst conjunct in 3 of 4 measurements. He is not failing to get up. He is getting up **by jumping**, and landing on his head.

**Cause 1, measured: `shaping_gamma = 1.0` is a reward pump in the sum PPO actually maximises.** The comment at `getup.py:159` claims "oscillating the pelvis up and down pays exactly zero". Measured over 9.6 s of the final policy, 64 envs:

| | undiscounted | discounted at ppo.gamma |
|---|---|---|
| shaping alone | **22.3** | **44.7** |
| everything else | 297.8 | 31.8 |
| TOTAL | 320.0 | 76.5 |

The shaping paid out **613.8** and clawed back **-591.5**. Undiscounted it does telescope, net 22.3 of 613.8 gross, so the comment is true about the sum I checked. It is false about the sum that is optimised: **discounted, the shaping is 44.7, larger than its own undiscounted value and larger than every other reward term combined (58% of the whole signal)**. Ng-Harada-Russell requires the shaping gamma to EQUAL the RL gamma; then `sum_t g^t (g*Phi_{t+1} - Phi_t)` telescopes to `-Phi(s_0)` and depends on nothing else. With `shaping_gamma = 1.0` against `ppo.gamma = 0.99` the sum is `sum_t g^t (Phi_{t+1} - Phi_t)`, which does not telescope: the rise is discounted less than the matching fall, so **every up-and-down cycle nets a profit**. Cycling beats standing, and standing still earns the shaping exactly zero. The policy is optimising correctly; the reward is wrong.

**Cause 2, measured: `upright` is the largest term in the task and is paid in mid-air.** 306.0 of the 297.8 non-shaping total. It is `clip(-gravity_body_z)`, pelvis orientation alone, with no ground-contact gate, unlike `rise`, `quiet` and `posture` which are all gated. A level pelvis in free flight collects it in full.

**The gate the rule blamed is not the problem.** `rise` is gated on foot load and therefore pays **zero** during flight, and it contributed 16.6 undiscounted against `upright`'s 306.0. It is not what is buying the jump. The `2*min` gate remains untested for a third run.

**Cause 3, structural, found while checking cause 1: `gamma = 0.99` at 125 Hz is a 0.8 second horizon.**

```
gamma 0.99, dt 8 ms  ->  1/(1-gamma) = 100 steps = 0.80 s
hold_seconds 2.0     =   250 steps, discounted to 0.081  (12.3x smaller than immediate)
a 4 s get-up + 2 s hold  ->  0.00053
```

legged_gym runs 0.99 at 50 Hz, which is a 2.0 s horizon. **We copied the constant, not the horizon.** This is the same mistake as E19 (`max_episode_steps = 1000` copied from 50 Hz onto our 125 Hz loop) and has the same shape: a number lifted from a repo running at a different control rate. The task asks him to hold for 2 s. At this discount, succeeding at the hold is worth 8% of an immediate reward, and the whole get-up-and-hold is worth 0.05%. **The hold is outside the agent's horizon.** No reward shaping can fix a target the discount has erased. Every get-up run in this project has had this.

- **Next**, in this order, and one at a time: (1) `gamma` 0.99 -> **0.998**, see E32; (2) `shaping_gamma` -> match `ppo.gamma` exactly, and re-derive `shaping_weight` with it; (3) gate `upright` on foot contact, or move its weight into `rise`.
- **ARITHMETIC SLIP IN THIS ENTRY, corrected 2026-08-15.** It first read "the drain is 0.003/step at gamma 0.997, not 1.0/step". 0.003 is bare `(1-gamma)`, not the drain. The drain is `shaping_weight * (1-gamma) * Phi` = `150 * 0.003 * 0.7` = **0.315/step** at Phi 0.7, and 0.45/step at a settled stand. So step (2) does NOT come for free with a higher gamma: matching the gammas at weight 150 still costs a third of a point per step, against a standing reward near 4.3. Step (2) needs the weight re-derived, not just the gamma copied across. Caught by an adversarial audit of this entry, not by me.
- **New Oracle checks, written and verified to bite**: `discount_horizon_covers_the_task` (the horizon must exceed the longest thing the task asks for) and `shaping_gamma_matches_rl_gamma`. Both fire CONTRADICTION on the config as it stood.

### E31b  2026-08-15  The generator of the rate bugs, found and killed

An audit of E31 went looking for the SOURCE of "constants copied from a 50 Hz repo" and found it. Two places asserted the wrong control rate, and one of them is a comment on the function that loads the model:

- `humanoid_rl/envs/model_prep.py`: "MuJoCo then runs the PD at the full **200 Hz** physics rate while the policy sets targets at **50 Hz**". The scene's timestep is 0.002 (500 Hz) and decimation is 4 (125 Hz). Both numbers wrong, on the live code path.
- `docs/research/conformance-audit.md`: OURS policy rate recorded as `50 Hz (200 Hz phys, decim 4)`, verdict **"ALIGNED (G1 exactly)"**. That audit is the document that signed off `gamma`, `gae_lambda`, `horizon`, `entropy_coef` and the reward weights as matching hardware-proven references. It was comparing a 125 Hz loop against 50 Hz repos and calling the numbers equal.

Both corrected, and the audit now opens with a retraction listing every row it got wrong. **Rule recorded there: never compare a constant, compare the quantity it stands for.** A discount factor is a horizon in seconds, not a number. An episode limit is a duration, not a step count.

- **Also found, logged not fixed**: `ppo.horizon` 24 is 0.19 s of experience here against 0.48-0.60 s at every reference. No provenance either way (`default.yaml` documents a batch-size rationale), so it is not a confirmed rate copy. Not bundled with E32.
- **Also found, logged not fixed**: this repo does not multiply reward by `dt` while all three references do. Harmless on its own (advantages are normalised, and getup's weights are not sourced from the references), but it is why our value function is ~3x theirs.
- **Stale Oracle remedy removed**: the shaping-drain check's remedy read "Use shaping_gamma = 1.0 so the sum telescopes", which is precisely the reward pump E31 measured. An Oracle recommending the bug it is supposed to catch.
- **`fasttd3.gamma` is also 0.99** and carries the identical 0.80 s horizon, as does its value support `v_max: 800`. Not touched, because FastTD3 is not running and changing an idle config teaches nothing. Fix it before any FastTD3 rerun: E22's "ratio 2 and 16 both flat, 100% falls" was measured under the same erased horizon and may say less than it appears to.
- **`getup_watch.sh` read the wrong log by luck**: `tail -1 /tmp/getup*.log | tail -1` picked the current run only because `_` sorts after `8` in ASCII across nine log files. Now selects by mtime.

### E32  2026-08-15  PRE-REGISTERED BEFORE LAUNCH: gamma 0.99 -> 0.9985

**Change**: `ppo.gamma` 0.99 -> 0.9985 in `configs/getup.yaml`. ONE change. Nothing else moved.
0.998 was tried first and lands on exactly 4.00 s, failing `discount_horizon_covers_the_task`
by 4e-15 s of floating point; the check was left alone and the number given margin instead.

**Why this and nothing else.** The audit measured that raising gamma does two opposing things
to the two behaviours in competition, and the arithmetic is the whole prediction:

| | gamma 0.99 | gamma 0.9985 | change |
|---|---|---|---|
| a reward 2 s away (the hold) | 0.0811 | 0.6871 | **8.5x more valuable** |
| a reward 4 s away (rise then hold) | 0.00655 | 0.4721 | **72x** |
| pump profit per up-down cycle, `1 - g^k` at k=75 steps (0.6 s) | 0.529 | 0.1065 | **5.0x less profitable** |

Holding gets 8.5x better and jumping gets 5x worse, so the ratio between them moves by about
42x. If the jumping is a discount artefact, this is enough. If it is not, nothing about this
change will help and the reward pump (`shaping_gamma`, step 2) is the remaining suspect.

**Predictions, written before the run, falsifiable, in order of how much I believe them:**

1. **"not ballistic" clears 40% by iteration 1200.** It has run 0.3-12.1% and was the worst of
   the 13 conjuncts in 3 of 4 measurements. This is the primary prediction and the one I would
   bet on. Below 20% at 1200 means gamma was not the mechanism.
2. **`standing_frac` becomes non-zero for the first time in this project.** It has read exactly
   0.0% on every evaluation of every get-up run ever. Predict above 1% by iteration 2000.
3. `held_ever_frac` above 0 by iteration 2500. Weaker: it needs all 13 conjuncts at once for
   250 consecutive steps, and stance width (4.4-12.1%) and per-foot load (8.7-15.5%) are also
   low for reasons gamma does not touch.
4. Pelvis height does NOT need to improve and probably will not. It is already 0.60-0.73 against
   a 0.745 threshold. If the story is right, what changes is that he stops leaving.
5. `value_loss` rises 3-5x in the first iterations and recalibrates within about 25. Measured on
   this repo's own critic: 21 iterations to fit the wider target versus 14. **This is expected
   and is not a failure signal.** Do not stop the run for it.
6. `action_std` stays at 1.000. If it exceeds 1.01 the E30 cap is not binding and that is a bug.
7. Eval return is undiscounted, so it stays comparable: predict 600-1200, no jump from the
   gamma change itself.

**What I expect to still be broken afterwards**, so a partial success is not read as a full one:
the `upright` term is 306 of 306 non-shaping reward and is still collected in mid-air, and the
shaping is still a pump, 5x smaller but not zero. Steps 2 and 3 remain.

**KNOWN OPEN CONTRADICTION, shipped deliberately.** `scripts/oracle.py` still reports
`shaping gamma: shaping_gamma 1.0 != ppo.gamma 0.9985`. That is step 2 and bundling it would
make this run uninterpretable: two changes, one number. Recorded here rather than left for
someone to find in a red Oracle they have learned to skip. `train.py` does not gate on the
Oracle at all, which is its own finding: nothing in the pipeline forces anyone to read it.

**Pre-committed stopping rule, not to be softened:** if "not ballistic" is below 20% at
iteration 1200, stop and record that the discount horizon was not the mechanism.

---

#### VERDICT: **MIXED**. The change did exactly what it was designed to do, and the run got worse.

Stopped at iteration 600 of 3051. The stopping rule did NOT fire ("not ballistic" was 62.1%,
far above its 20% floor). Stopped for a different, measured reason, stated below.

**The mechanism worked, and this is the part to keep.** Same measurement script as E31, same
policy-rollout conditions, 9.6 s over 64 envs:

| | E31 (gamma 0.99) | E32 (gamma 0.9985) |
|---|---|---|
| shaping, discounted | 44.7 | **6.0** |
| shaping as a share of the whole signal | 58% | **3.5%** |
| shaping gross flow, paid / clawed back | 613.8 / -591.5 | 275.9 / -276.9 |

The pump fell **7.5x**. Prediction 3 in the table above said 5x. Raising the discount horizon
does defuse a gamma-mismatched potential shaping, quantitatively and about as hard as predicted.

**And it immediately exposed the next exploit, which was pre-registered on this very page as
"what I expect to still be broken afterwards".** With the pump gone, the reward has almost
nothing else in it:

| term | undiscounted | discounted |
|---|---|---|
| **upright** | **393.6** | **169.3** |
| effort | -12.4 | -6.7 |
| rise | 3.4 | 2.6 |
| stand, quiet, posture | 0.0 | 0.0 |

`upright` is **169.3 of 171.2, i.e. 99% of the entire signal**. It is `clip(-gravity_body_z)`,
pure pelvis ORIENTATION, with no height requirement and no ground-contact gate. Over the same
9.6 s: mean pelvis 0.336, **0.0% of steps above the 0.745 standing line**, any foot contact on
11.1% of steps. `stand`, `quiet` and `posture` are all gated on U and paid exactly zero, so the
task's own definition of success contributed nothing to the return at any point.

**The policy converged to lying still.** Final checkpoint, rendered:

| t | pelvis | head | feet BW | knee | hands |
|---|---|---|---|---|---|
| 1.2 s | 0.952 | 0.60 | 0.00 | 2.62 | 0.63 |
| 2.4 s | 0.209 | 0.12 | 0.00 | 2.78 | 0.04 |
| 4.0 s | 0.193 | 0.20 | 0.00 | 2.80 | 0.04 |
| 8.0 s | **0.193** | **0.19** | **0.00** | **2.80** | **0.04** |
| 15.0 s | **0.193** | **0.19** | **0.00** | **2.80** | **0.04** |

One launch, one crash, then a tight ball with the knees at their limit, motionless for eleven
seconds, identical to three significant figures. Return climbed 563 -> 915 -> 1057 across three
consecutive evaluations while pelvis height fell 0.482 -> 0.265 -> 0.253. **The best return in
this task's history was earned by a humanoid that does not move.** That is why it was stopped:
the gradient was actively deepening it and 2500 iterations remained.

**Scored against the predictions, honestly:**

| # | Prediction | Outcome |
|---|---|---|
| 1 | "not ballistic" clears 40% | **Met numerically (62.1%) and worthless.** See below. |
| 2 | `standing_frac` above 1% | **NO.** 0.0% on all 12 evaluations, as on every get-up run ever. |
| 3 | `held_ever_frac` above 0 | NO. |
| 4 | pelvis stays 0.60-0.75, no improvement needed | **WRONG, and wrong in direction.** It collapsed to 0.235. |
| 5 | value_loss recalibrates in ~25 iterations | Held. No instability. The audit's measurement was right. |
| 6 | `action_std` stays at or below 1.000 | Held: 0.583-0.654 throughout, never near the cap. |
| 7 | eval return 600-1200, comparable | Held: 563-1057. Comparable, and meaningless. |

**THE LESSON, and it is the third time: I pre-registered an ABSENCE predicate as the success
metric.** "not ballistic" is `speed < threshold`. A body lying motionless satisfies it
perfectly. So does a corpse. It went 5.1% -> 43.8% -> 62.1% while everything requiring an
actual stand went the other way (knees straight 2.6% -> 0.2% -> **0.0%**, feet on the floor
4.5% -> 0.7%, each foot 20% 2.7% -> 1.2%). The primary metric improved 12x by the policy
getting worse at the task.

This repo has now shipped that exact class of bug three times: `gait_symmetry` reading 1.0 for
standing still (E02), the stillness terms that `getup.py:447` warns "this repo has shipped that
exact bug twice", and now the metric I chose to judge the fix by. **Rule: a success criterion
must be a thing the humanoid DOES, never a thing it refrains from.** Absence predicates belong
in conjunctions as guards, never alone as a headline.

- **Next**: step 3, gate `upright` on ground contact. It must use the foot-force SUM, not
  `rise`'s `2*min(left, right)`, or a half-kneel (one foot planted, one knee down), a
  legitimate stage of a human get-up, scores zero.
- **Carried forward, still untested for a fourth run**: the `2*min` both-feet gate on `rise`.
- **Keep**: gamma 0.9985. It is not the cause of this failure and it fixed what it was aimed at.
  The run was stopped before it could test whether a longer horizon helps a non-degenerate
  reward, so that question is still open.

### E30  2026-08-15  Exploration ran away and drowned the policy; the both-feet gate is untested
- **Run**: get-up with `rise` gated on `2*min(left, right)` foot force instead of the sum.
- **Stopped at iteration 900 by the pre-committed rule** ("torso not inverted" below 40% by iteration 1000). It was at **0.1%**.
- **But the gate is not what failed.** Exploration std ran **0.79 -> 3.43** on an action range of [-1, 1], and eval return went **833 at iteration 300 -> 19 at iteration 900**. The policy drowned in its own noise before the gate could be judged either way.

| | iteration 300 | iteration 900 |
|---|---|---|
| head high | 19.1% | **0.1%** |
| torso not inverted | 5.7% | **0.1%** |
| pelvis high | 26.2% | **0.8%** |
| return | 833 | 19 |

- **Cause**: `log_std_max = 5.0`, which is std 148 and therefore no ceiling at all, plus a positive `entropy_coef` and nothing pulling back. This is the SECOND run lost to log_std after E05, and in the opposite direction: E05 froze it above its ceiling, this let it escape.
- **Fix**: `log_std_max` 5.0 -> 0.0, capping std at 1.0. Walking trained fine at 0.4-1.4. Verified safe: `torch.clamp` passes gradient 1.0 at exactly the boundary and 0.0 only strictly outside, and `clamp_log_std()` runs in place after every optimiser step, so the parameter can sit on the ceiling and still be pulled back down. That distinction is precisely what E05 got wrong.
- **New Oracle check** `exploration_has_a_ceiling`, covering BOTH directions: a ceiling above std 2 is not a ceiling, and `init_noise_std` above the ceiling freezes the parameter from step one. Verified to bite on both.
- **A defect in my own watching, worth as much as the run.** `getup_snapshot.py` rendered `best.pt`, which only moves on a new record. It froze at iteration 300 while the run was at 900, so for nine minutes I was looking at a 600-iteration-old policy and printing current metrics beside it. Switching to the newest checkpoint changed the picture instantly and for the worse. Watching the wrong object is worse than not watching.
- **Carried forward untested**: the `2*min` foot gate. It removed 87% of `rise` for the previous policy and may still be too harsh, but this run cannot say.

### E29  2026-08-15  The shaping was gamed in 200 iterations, by a headstand
- **Caught by looking**, after the metrics showed an impossible combination: pelvis at 0.804 m (92% of standing) with the head at 0.322 of standing height. The head was BELOW the pelvis.
- **The policy inverted.** Rendered frames: lying -> pike on hands and feet with the hips up -> balanced head-down with the legs in the air -> folded over. Measured on the resulting policy at 3.2 s: pelvis 0.71 m, head 0.14 m.
- **My error, and the design note stated the opposite.** E28's comment claims the shaping "cannot reward standing on your hands". That is true of `rise`, which is multiplied by pelvis uprightness. The SHAPING had no gate at all: it paid for bare pelvis height, and inverting is the cheapest way to raise a pelvis. I wrote the guarantee for one term and applied it in my head to another.
- **Fix**: `Phi = min(pelvis_height / standing_height, head_height_ratio)`. Both ends of the body must be off the floor, which is what upright means without touching orientation, and orientation is the thing that is not monotone along the path.
- **Scored against the poses that policy actually found**:

| pose | pelvis-only Phi | min(pelvis, head) |
|---|---|---|
| inverted | 0.81 | **0.09** |
| pike / downward dog | 0.82 | 0.50 |
| kneeling | 0.68 | 0.66 |
| standing | 1.00 | 1.00 |

The cheat collapses from nearly-standing to nearly-nothing; the honest poses barely move.
- **Worth keeping**: this run was still the best yet on the thing it was fixed for. `knee_max` reached 1.10 against a hard ceiling of 1.057 in every earlier run, so the action-range fix (E27) is confirmed working. The humanoid is now physically capable of the poses a get-up needs; it just found a faster way to be paid.
- **Lesson**: a guarantee proved for one term does not transfer to another term in the same function. Every positive needs its own gate, and I now have three instances of exactly this (E03 symmetry, E26 arm-prop, E29 shaping).

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
