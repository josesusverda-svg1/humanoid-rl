# E23 — Torso posture: root cause, and the experiment that can actually test it

**Status:** design, pre-registered. Written before any arm was run.
**Budget:** 3.5 h wall clock. Fits.
**Author's note:** every number below marked *(measured)* was re-derived in this session from
files in `runs/` or from a rollout of `best.pt`. Numbers I could not verify are marked
*(unverified)* and are not load-bearing.

---

## 0. Read this first: three premises in the brief are false

The brief asks us to fix a policy that "walks by LEANING FORWARD" at `torso_upright 0.54`.
All three parts of that sentence are wrong, and the experiment has to be built on what is
actually true or it will measure nothing.

**(1) `0.54` does not belong to `best.pt`.** *(measured)*
`best.pt` stores `iteration = 3100, best_return = 3033.66`. The eval row at iteration 3100 in
`runs/envelope-20260814-100402/events.jsonl` reads:

| field | value |
|---|---|
| `eval/episode_return` | 3033.66 |
| `eval/episode_length` | 930.56 |
| `eval/fall_rate` | 0.15625 |
| **`eval/torso_upright`** | **0.7256** |

`0.5431` is the **last** eval, iteration 5050, a checkpoint that returns 2327.6 at 29.7% falls
and was never selected. The brief's headline pairs one checkpoint's return, length and fall
rate with a different checkpoint's posture, 1950 iterations later. **Fix the LOGBOOK
"Current state" row to 0.726 before anything else.**

**(2) `terminate_torso_upright = 0.50` is dead code.** *(measured)*
`humanoid_rl/tasks/locomotion.py:1111-1112` hardcodes `stooped = np.zeros_like(fallen)` and
`head_down = np.zeros_like(fallen)`. `grep` over `humanoid_rl/` shows the config key's only
live reader is `humanoid_rl/tasks/tracking.py:256` — a different task. It is set with
explanatory comments in 8 config files and read by none of them in this task. There is no
cliff at 0.50, so "the policy has found the maximum lean that does not trigger termination"
is false. The comment three lines above the dead guard still describes this exact failure.

**(3) The lean is BACKWARD, not forward.** *(measured, this session, domain randomisation ON)*
Rolling out `best.pt` at a held 1.0 m/s command, 32 envs × 500 settled steps, and
reconstructing the torso's true body-frame z-axis through `mj_forward`:

| quantity | value |
|---|---|
| sensor `torso_upright` | 0.6844 |
| **true** body-frame uprightness | 0.7478 (41.6° tilt) |
| fore component in heading frame | **−0.5425**, forward in **0.4%** of samples |
| left component | +0.3500, left in **99.8%** of samples |
| `abdomen_x`, `abdomen_y` | −0.587, −0.748 rad |

The trunk is folded **backward ~42° and rolled left ~34°**, held by a near-DC action at
about −0.94 of ±1. That is a broken-symmetry postural contortion, not a gait strategy.
`torso_upright` is `cos(tilt)` and therefore **sign-blind**: a fix that merely flipped the
fold forward would score as a large win while reproducing the failure the torso sensor was
added to catch. The sign must be in the decision rule, not a footnote.

---

## 1. The root cause you should believe

> **The reward is not mispriced at the operating point. It already prefers the upright
> posture — total reward is measurably HIGHER with the trunk straightened. The policy is
> stuck at a posture its own reward disprefers, because reaching the better posture requires
> a *temporally sustained* abdomen offset, and i.i.d. Gaussian exploration does not produce
> one. The linear torso term's constant 0.6/unit gradient is too weak, relative to per-step
> noise, for PPO to accumulate credit for a change that only pays off when held.**

This is the paired open-loop counterfactual, same seed in every arm, domain randomisation
**ON**, held 1.0 m/s command, 32 envs × 500 settled steps, abdomen_x/y/z action scaled by
alpha *(measured, this session)*:

| alpha | true `torso_upright` | tilt | forward speed | **total reward/step** | terminations |
|---|---|---|---|---|---|
| 1.00 (unmodified) | 0.7478 | 41.6° | 0.718 m/s | 3.2865 | 3 |
| **0.75** | **0.8729** | **29.2°** | **0.690 m/s** | **3.3903** | 3 |
| 0.50 | 0.9592 | 16.4° | 0.626 m/s | 3.2816 | 6 |

Straightening the trunk from 0.748 to 0.873 **raises total reward by +0.104/step** and costs
3.8% of speed with no change in falls. Decomposing into the torso term and everything else:

| segment | d(non-torso reward)/d(uprightness) |
|---|---|
| u 0.748 → 0.873 | **+0.137** (standing up *improves* the other 20 terms) |
| u 0.873 → 0.959 | −2.012 (genuinely expensive past ~0.87) |

**This kills the "leaning buys speed at 2:1" story outright.** There is no trade at the
operating point. The policy is off its own reward frontier, in the free region.

### Why PPO does not climb a gradient that points the right way

The needed action change is +0.235 on `abdomen_y`. The policy's own exploration std on that
dimension is `exp(log_std[1]) = 0.196` *(measured from `best.pt`)* — so the better region sits
**1.2 sigma away** and is sampled constantly. It is not an exploration-magnitude problem.

It is an exploration-*correlation* problem. PPO's noise is independent per step and is then
low-passed by the 8 Hz action filter. A one-step +0.235 deviation barely moves a 22 kg trunk
and mostly registers as jitter; the postural benefit only appears when the offset is *held*
across a stride. So the advantage estimator sees "abdomen noise ≈ no benefit" and the mean
never migrates. My alpha sweep applies the offset coherently on every step, which is exactly
the perturbation on-policy noise cannot manufacture.

**This is why the reward fix is still the right lever, but for a different reason than the
brief gives.** The torso term rewards the *state* every step, so it does not need exploration
to discover that a sustained posture is better — it pays for the posture directly. Raising
and steepening it amplifies an already-correctly-signed state signal until it clears the
per-step noise. We are not re-pricing a profitable lean; we are strengthening a weak signal
on a direction that is already free.

### What is NOT the root cause

- **Not the quadratic-vs-linear asymmetry with the pelvis.** The pelvis is fine: `eval/upright`
  reads 0.951 at iteration 3100 *(measured)*. The fold is entirely at the abdomen, which the
  pelvis `orientation` term cannot see. The asymmetry is real as written but is not what is
  driving this.
- **Not the command envelope alone.** `runs/omni-conformed-20260813-203055` has identical
  `w_torso_upright 0.6` and `w_orientation -1.0` and stays upright *(unverified — I did not
  re-derive this run's numbers)*. The E14 envelope change is a real contributor
  *(measured: humanoid-20260814-022748 vs envelope, matched windows, separation 0.2015 at
  iters 2001-3000 and 0.2074 at 3001-4000)* but reverting it would cost the project's largest
  speed win.

### Confidence and the honest caveat

**Medium-high.** The counterfactual is a perturbation of a converged closed-loop policy at a
**single held 1.0 m/s command**. Training uses a wide envelope including backward, lateral,
turning and zero commands. The free region may not be free there. Arm B exists to test the
endpoint rather than trust the slope.

---

## 2. Code and config changes

### F0 — instrumentation. Applies to **every** arm including the control.

**F0 is not zero-effect and must not be described as "reward unchanged."** `terms[:, 10]`
consumes `state.torso_upright` directly, so this rotates that term's zero-error point by
5.87° toward true vertical. The control arm is labelled **"F0-only"**, not "unchanged", and a
small posture move in the control is a confirmation, not a surprise.

**F0a — fix the torso sensor frame.**
`humanoid_rl/envs/model_prep.py:216`, inside the `if UPRIGHT_BODY in body_names:` block:

```python
# OLD
s.objtype = mujoco.mjtObj.mjOBJ_BODY
# NEW
s.objtype = mujoco.mjtObj.mjOBJ_XBODY
```

`mjSENS_FRAMEZAXIS` with `mjOBJ_BODY` reads the body's **inertial** frame (`ximat`), not the
body frame (`xmat`). *(measured, at `default_qpos`)*: `xmat[torso][:,2] = [0, 0, 1]` exactly,
while `sensordata = [-0.10234, 0, 0.99475]` — the inertia principal axis sits **5.87° behind**
the torso's long axis. A perfectly upright torso can never read above 0.99475, and because
this policy folds *backward*, the two tilts add and the sensor **understates** uprightness by
0.063 at the operating point *(measured: sensor 0.6844 vs true 0.7478)*.

**F0b — DO NOT fix the other two sensors in this experiment.** The identical bug is on the
`FRAMEPOS` sensors at `model_prep.py:206-212` and `219-223` *(measured at `default_qpos`)*:

| sensor | sensordata z | body-frame `xpos` z | bias |
|---|---|---|---|
| `left_foot_pos` / `right_foot_pos` | 0.02950 | 0.05200 | **−2.25 cm** |
| `head_pos` | 1.51391 | 1.33891 | **+17.5 cm** |

The foot value feeds `terms[:, 18] = w_swing_height * Σ(foot_z − swing_height_target)²` at
**w = −20.0** against a 0.08 m target — a 28% bias on the target of the largest-magnitude
weight in the reward. Fixing it is a *large reward change*, not instrumentation, and bundling
it here would confound both arms identically but make the result untransferable. **Log both
as KNOWN-UNFIXED in the LOGBOOK bug table with the measured offsets above**, so nobody later
reads `swing_height` or `head_height_ratio` as calibrated. Fix them in their own experiment.

**F0c — log the metric, read BEFORE autoreset.**
This is the difference between a working experiment and a silently biased one.
*(measured from source)*: `humanoid_rl/envs/vec_env.py` calls `reward_batch` at line ~573 and
`_do_resets(done_idx, ...)` at line ~595; `_do_resets` calls `_compute_derived(done_idx)`
(line ~509), which **overwrites `state.torso_upright` for every env that just terminated**
with its fresh reset-pose value. `res.reward_terms` is snapshotted pre-reset; reading
`self.env.state.torso_upright` after `step()` returns is **post**-reset.

That estimator is biased upward in proportion to the termination rate — and every treatment
here is expected to move the fall rate, so the confound points the same way as the effect.

In `humanoid_rl/envs/vec_env.py`, immediately after the reward block and **before** the
`done_idx = np.flatnonzero(done)` / `self._do_resets(...)` lines:

```python
# Pre-reset postural snapshot. Must be taken here: _do_resets -> _compute_derived
# overwrites state.torso_upright for every terminating env with its reset pose.
self._torso_upright_mean = float(s.torso_upright.mean())
self._torso_fore_mean = float(np.einsum('ij,ij->i', self._torso_axis_xy(), self._heading_xy()).mean())
```

Add both to `StepResult` (`torso_upright: float`, `torso_fore: float`) and populate them in
the `return StepResult(...)` block. `_torso_axis_xy` / `_heading_xy` compute the torso z-axis
planar components rotated into the heading frame — the **sign** channel, which `torso_upright`
cannot represent. If that is too invasive, log `torso_fore` from the confirmation rollout only,
but `torso_upright` must move pre-reset.

Then in `humanoid_rl/train.py`, next to `reward_term_sums += res.reward_terms.mean(axis=0)`
(line ~295), accumulate `torso_sum += res.torso_upright` and `fore_sum += res.torso_fore`, and
emit next to the `reward/<name>` block (line ~307):

```python
timings["train/torso_upright"] = torso_sum / cfg.ppo.horizon
timings["train/torso_fore"] = fore_sum / cfg.ppo.horizon
timings["train/difficulty_median"] = float(np.median(self.env.state.task_state["difficulty"]))
```

`train/difficulty_median` is mandatory. `eval/difficulty_median` reads the **eval** env's own
independent `task_state["difficulty"]` array (`locomotion.py:502` runs per `ThreadedVecEnv`)
and cannot observe the training curriculum at all.

### F1 — the treatment: quadratic torso **bonus**

`humanoid_rl/tasks/locomotion.py:748`:

```python
# OLD
terms[:, 10] = cfg.w_torso_upright * np.clip(state.torso_upright, 0.0, 1.0)
# NEW
terms[:, 10] = cfg.w_torso_upright * np.square(np.clip(state.torso_upright, 0.0, 1.0))
```

with `w_torso_upright: 1.5`.

**Write it as a bonus, not as `−1.5 * (1 − u²)`.** The two have identical marginal rate
`dR/du = 3u`, but they differ by an additive constant of 1.5/step, and that constant is not
absorbed anywhere. *(measured)*: there is no reward normaliser in `humanoid_rl`, and
`humanoid_rl/algos/ppo.py` bootstraps 0 on `terminated`, so a uniform per-step offset `c`
changes the value of terminating by `c/(1−γ) = 100c` at γ = 0.99. The penalty form would make
falling ≈111 return units *cheaper* before it touched posture at all — a survival-incentive
confound sitting inside the treatment.

It also protects the only-positive clip at `locomotion.py:801`. *(measured, iters 3000-3200)*:
raw term sum is **2.7546**/step, so the clip is currently inert. The bonus form raises headroom
to ~3.14; the penalty form would cut it to ~1.64, switching the posture gradient **off**
precisely in the folded and failing states where it is most needed.

`dR/du` goes from a constant 0.6 to `3u` — 2.24 at the current u, 2.9 near upright — which is
the "stronger where it matters" property the linear term lacks. Given the measured cost curve
(free below u ≈ 0.87, −2.0/unit above), this equilibrates comfortably past 0.87.

### F2, F3, F4 — deliberately NOT run

- **F2 (`w_torso_upright: 2.5`, linear).** The proposal's own solve puts F2's equilibrium at
  0.903 and F1's at 0.909 — a predicted separation of **0.006**, far below any threshold this
  design can resolve. The head-to-head is unpowered by construction and the proposal
  pre-commits to shipping F1 regardless. Spending 25% of compute on it buys nothing.
- **F3 (new 22nd reward term).** Largest code surface of the menu — term array width,
  `reward_term_names`, eval plumbing, FastTD3 value support — in a project whose failure log is
  dominated by exactly that class of change. Not worth it before F1 is measured.
- **F4 (restore the dead termination).** **It censors its own metric.** `stooped =
  torso_upright < 0.5` is a hard termination on the exact scalar being population-averaged: it
  deletes the left tail of the distribution and replaces each deleted trajectory with a fresh
  episode opening near the reset value. Its predicted +0.03 to +0.08 is precisely the magnitude
  pure censoring would produce, and no other arm has this property, so it is not on the same
  measurement scale as the control. This is the same shape as the E16 eval bias. Restoring the
  guard is still a legitimate **bug fix** — file it separately, and score it only on a
  fixed-checkpoint rollout with the termination disabled.

### The config trap that would have silently ruined this

**Base every arm on `runs/envelope-20260814-100402/config.yaml`. Not `configs/default.yaml`,
and not `configs/.envelope.yaml`.** *(measured)*

`configs/.envelope.yaml` exists — it is the sparse launch config for the run — but it does
**not** set `difficulty_init` / `difficulty_min`. Those fell through to the dataclass, which at
the time was `1.0 / 1.0`. The current dataclass default is **`0.75 / 0.75`**
(`locomotion.py:242, 257`), and `promote_gait_match` moved 0.8 → 0.68 and `swing_height_target`
0.08 → 0.09. `configs/default.yaml` additionally sets `max_episode_steps: 2500` against the
run's 1000.

So the envelope run had the curriculum **pinned off**, and relaunching from either file today
turns it **on**. `locomotion.py:591` scales the entire command envelope by `difficulty`, and
`_maybe_promote` gates promotion on tracking error and gait match — precisely the behaviours a
posture change perturbs. An arm that falls more promotes less, sits nearer 0.75, receives
commands up to 25% weaker, and therefore **reads more upright with zero postural improvement**.
The confound is endogenous, so it is not shared with the control, and it points in exactly the
direction that makes the fix look like it worked. *(measured)*: `Config.load` reads the archived
run dump cleanly — `difficulty_init 1.0, max_episode_steps 1000, w_torso_upright 0.6`.

Every arm config must explicitly carry:

```yaml
env:
  max_episode_steps: 1000
task:
  difficulty_init: 1.0
  difficulty_min: 1.0
```

### Pre-flight assert — mandatory, prints before iteration 1

E11 ran six byte-identical arms. There is **no `--seed` and no `--name` CLI flag**
*(measured: `train.py` argparse has only `--config`, `--resume`, `--init-from`, `--iterations`,
`--num-envs`, `--eval-interval`)*, so arm identity lives entirely in the generated YAML and
nothing today would catch a copy-paste error.

Generate the 12 YAMLs programmatically from the run dump, and print at startup:

```
run.name, run.seed, w_torso_upright, term form (linear|quadratic),
realised terms[:,10] at a hand-set torso_upright = 0.8,
sensor objtype for torso_zaxis (must be 2 = mjOBJ_XBODY),
difficulty_init, difficulty_min, max_episode_steps
```

Expected `terms[:,10]` at u = 0.8: **control 0.480**, **treatment 0.960**. If an arm prints the
other one, it is mislabelled — abort. Also assert `sensordata[torso_adr:torso_adr+3] ==
xmat[torso][:,2]` to 1e-9 at `default_qpos`, which is a direct test that F0a took effect.

---

## 3. Arms and seeds

| arm | n seeds | config | change |
|---|---|---|---|
| **A — CONTROL (F0-only)** | 6 | run dump + F0 | sensor frame fixed, metric logged. Reward form and weight unchanged. |
| **B — TREAT (F0 + F1)** | 6 | run dump + F0 + F1 | `terms[:,10] = w * u²`, `w_torso_upright: 1.5` |

Seeds `0, 1, 2, 3, 4, 5` in both arms. 350 iterations each, all warm-started:

```
--init-from runs/envelope-20260814-100402/checkpoints/best.pt --iterations 350
```

**Two arms, not four.** With 6 seeds instead of 3, `SE(Δ)` improves by √2 and the design
yields a **5-df within-arm** and **10-df pooled** variance estimate that replaces the current
1-df guess for every future experiment in this repo. That estimate is a deliverable in its own
right and is worth more than F2 and F4 combined.

**Run order must be interleaved** — `A0, B0, A1, B1, …` — never all of one arm then the other.
Twelve sequential runs on one machine otherwise confound arm with thermal and scheduler state,
and thread-scheduling nondeterminism is the stated source of the E22b variance. **Never run two
arms concurrently**: contention changes throughput and the collector's timing.

Set `eval.video_on_best: false` and `eval.video_every_n_evals: 0` in all arms. Video costs
wall clock and adds a variable; `best_return` starts at `-inf` on a fresh Trainer, so the first
eval of every arm would otherwise render one.

---

## 4. The noise floor, honestly

The proposal's `0.0122` is the **minimum** of its window family, presented as a worst case.
*(measured, all windows of the E22b byte-identical twins `arm-warm-20260813-180357` vs
`arm-warm-20260813-181716`, on `reward/torso_upright / 0.6`)*:

| window length | n windows | min \|Δ\| | median | **max** |
|---|---|---|---|---|
| 150 iters | 201 | 0.0002 | 0.0133 | **0.0587** |
| **200 iters** | 151 | 0.0000 | 0.0147 | **0.0380** |
| 250 iters | 101 | 0.0122 | 0.0229 | **0.0234** |

The three windows the proposal reported were all anchored to iteration 350 — the one anchor
that produces the minimum. **Plan against the max for the chosen window length: 0.0380.**

Two further caveats, both of which the design must respect rather than argue away:

1. **1 degree of freedom.** One pair. `s = |Δ|/√2`; the one-sided 95% upper bound on σ is far
   larger than the effect we want to detect. No claim of the form "N noise floors" is
   defensible in either direction from this.
2. **Different regime.** *(measured)*: the twins carry **46** `task` config keys against the
   envelope run's **71** — no `w_orientation`, no `w_swing_height`, no `w_feet_distance`, no
   `w_dof_pos_limits`, no curriculum keys at all — plus `entropy_coef 0.0` vs `0.01`,
   `log_std_max −0.7` vs `5.0`, `init_noise_std 0.45` vs `0.8`. Exploration is hard-clamped
   there and unclamped here, which most likely makes 0.0380 an **under**estimate.

**Therefore 0.0380 is a planning figure only. The operative floor is the CONTROL arm's own
between-seed SD, measured in this experiment on 6 seeds, and it is computed before the
treatment arm is looked at.**

Non-overlapping 50-iteration blocks of the twins *(measured)*: 0.0073 / 0.0950 / 0.0624 /
0.0126 / 0.0380 / 0.0102 / 0.0389. The two large early values are why the window excludes the
warm-start transient — both twins decay from ~0.98 and at different rates.

---

## 5. Pre-committed decision rule

**Written before any arm was run. Do not amend after seeing results.**

**Primary statistic.** For each seed: the mean of `train/torso_upright` over
**iterations 151–350 inclusive** (200 iterations). Arm statistic: the mean over that arm's 6
seed means. Effect: `Δ = mean(B) − mean(A)`.

The window is fixed on principle — iterations 1–150 are the warm-start transient, where the
twins' matched-block divergence peaks at 0.095. It is **not** chosen for its floor: on the
twins the 151–350 window happens to give 0.0004, near the minimum of its family, and I am
planning against the family **max** of 0.0380 precisely so that coincidence cannot flatter the
result.

**Test.** Welch two-sample t on the 6 vs 6 seed means, two-sided, α = 0.05. Welch, not pooled:
F1 changes the per-step reward scale, so the arms are not assumed variance-comparable.

**SHIP the treatment only if all four hold:**

1. `Δ > 0` with Welch p < 0.05.
2. `Δ ≥ 0.05` in absolute units. Statistical significance alone is not enough — a 0.02 effect
   is not worth a reward change.
3. **Cost guards pass** (§6), all calibrated against arm A **measured in this experiment**.
4. **Sign guard passes** (§6): the fold does not flip forward.

**If Δ is significant but < 0.05:** record the effect and the measured floor; do not ship.
**If Δ is not significant:** report the 95% CI on Δ and the measured between-seed SD. That SD
is the experiment's most durable output regardless of outcome.

**Collapsed-seed rule, fixed in advance.** A seed whose mean `episode_length` over iterations
151–350 is **< 200** is declared collapsed. Collapsed seeds are **reported, never silently
dropped**. Each arm reports (a) the all-seeds mean and (b) the collapse count. If the two arms'
collapse counts differ by ≥ 2, the posture comparison is void and the finding is the stability
difference, not the posture difference. E22b's variance is a bimodal attractor collapse, not
Gaussian scatter, and averaging a collapsed seed into an arm mean makes the number meaningless.

**Predicted values, stated in advance so they can be wrong.**
Post-F0 the metric is on a new scale and every archived number shifts. *(measured)*: the
warm-start baseline is `reward/torso_upright/0.6 = 0.7452` at envelope iters 3000-3200 in
**old** units; the F0 offset at the operating point is **+0.063**. So arm A should open near
**0.808** and I predict its window mean at **0.78–0.84** (the envelope run drifted about −0.10
per 400 iterations at steady state, so a control that slides is expected). I predict arm B's
window mean at **0.88–0.93**, i.e. `Δ ≈ +0.09 ± 0.04`. **The control's iteration-1 value must
be measured, not assumed** — if it does not land near 0.808, F0 did not do what I think and the
experiment stops there.

---

## 6. Guards against this project's actual failure modes

**G1 — the ship guards must be calibrated in-experiment, because the obvious ones are broken.**
*(measured, envelope stationary tail, evals 3000–5050, n = 42)*:

| metric | mean | sd | note |
|---|---|---|---|
| `eval/fall_rate` | **0.4535** | 0.2228 | only **31%** of tail evals are below 0.30 |
| `eval/mean_speed` | 0.6284 | 0.0561 | |
| `eval/torso_upright` | 0.6427 | 0.0704 | ~5× noisier than the training metric |
| `eval/episode_return` | 2190.8 | 516.6 | 23.6% — unusable |
| `eval/lin_vel_error` | 0.5943 | 0.1587 | |
| `eval/episode_length` | 757.1 | 147.2 | |

A guard of "`eval/fall_rate` < 0.30" **cannot be passed by the unchanged control** — the 0.156
figure the proposal quotes is the single eval at iteration 3100, the 4.8th percentile of its own
tail. That is the same single-eval stitching error the proposal convicts the brief of. A guard of
"`eval/mean_speed` > 0.50" sits *below* the design's own predicted cost and is unfailable.

**Replace both with control-relative guards:**

- `eval/lin_vel_error(B) ≤ mean(A) + 1.0 × between-seed sd(A)` — command-normalised, so it
  separates "walks slower because commanded slower" from "tracks worse".
- `eval/episode_length(B) ≥ mean(A) − 1.0 × between-seed sd(A)`.
- Both averaged over each run's last 3 evals, never a single eval.

**G2 — the sign guard.** `torso_upright = cos(tilt)` is sign-blind, and this policy's fold is
backward in 99.6% of samples. An arm that converts a 42° backward fold into a 30° forward stoop
would post a large `Δ` while reproducing the original failure. **Requirement:**
`|train/torso_fore|` must decrease in arm B relative to arm A, and if `torso_fore` changes sign
the arm **fails** regardless of `Δ`.

**G3 — the metric must be the same estimator the floor was measured on.** Assert once, at
iteration 1 of every **control** seed, that
`train/torso_upright == reward/torso_upright / 0.6` to within 1e-6. This passes only if the
accumulation is pre-reset (F0c) and the control's reward form is genuinely unchanged. It is a
single equality that catches both the reset-ordering bias and a mislabelled arm. It cannot be
run on arm B, where the term is `w·u²` — which is exactly why B must never be the arm that
validates the plumbing.

**G4 — never recover `u` by dividing the reward term by the weight.** Under the control that
yields `u`; under the treatment it yields `u²`. **These agree at u = 1.0 and are within 0.02 of
each other for all u > 0.86** — squarely inside the region both arms will occupy. A mis-scaled
arm would look entirely plausible. This is the same shape as the parity coincidence already in
the logbook. Read the raw `train/torso_upright` scalar, always.

**G5 — the curriculum tell-tale.** `train/difficulty_median` must read exactly 1.0 in every arm
at every iteration. Any arm where it moves is **void**, not adjusted. This is the endogenous
confound from §2 and it is otherwise invisible.

**G6 — the eval loop is contaminated; do not build the decision on it.**
`humanoid_rl/evaluate.py` accumulates `speed_sum` and the task metrics (including
`torso_upright`) over **every env at every step** until `counted.all()`, so envs that already
fell keep contributing near-zero speed and near-upright post-reset poses. The measured effect is
small (≤ 0.005 on `torso_upright`) but the bias is structurally the E16 bug. Eval metrics are
**cost guards only**; the decision rests on the training-batch metric and the confirmation
rollout.

**G7 — confirmation rollout, mechanism check only.** After training, roll out each of the 12
final checkpoints deterministically at a held 1.0 m/s command, fixed seed and fixed DR seed,
identical across arms, and report `torso_upright`, `torso_fore`, speed and falls.
**Do not quote its paired SE (~0.006) as if it bounded the arm comparison** — that is rollout
noise on a frozen checkpoint, whereas the arm comparison's variance is training-trajectory
divergence, which pairing on rollout seeds does not touch.

**G8 — specificity.** *(unverified, from the critique)*: across the warm-started 350-iteration
arm family the end-of-run posture spans ~0.18, driven by exploration knobs that touch no posture
term. F1 changes the per-step reward scale, hence the advantage scale, hence the entropy
trajectory. **Log `action_std` and policy entropy alongside `train/torso_upright`** so the
exploration channel can be ruled out rather than assumed away. Given that §1 identifies the root
cause as an exploration-correlation failure, this is not a formality — if entropy explains the
move, that is the real finding.

---

## 7. Wall clock

*(measured)*: `iteration_time` mean is **1.918 s** over the envelope run and **1.836 s** over its
first 400 iterations (734.4 s). The 1.625 s figure in the proposal comes from dividing total wall
hours by iteration count, which is wrong because that run was resumed.

| item | cost |
|---|---|
| F0 implementation + pre-flight asserts | 25 min |
| Phase A verification (done in this session; re-run after F0) | 15 min |
| 12 runs × 350 iters × 1.836 s | 128 min |
| startup + evals, 12 × ~55 s | 11 min |
| G7 confirmation rollouts, 12 × ~90 s | 18 min |
| analysis, LOGBOOK write-up | 20 min |
| **total** | **≈ 3.6 h** |

Inside the 3–4 h budget with ~25 min of genuine slack. **Do not spend that slack on a third
arm** — it is there for the one run that crashes.

---

## 8. What this design cannot answer, stated plainly

The critics were right that the *original* four-arm design could not answer the question in
budget: its F2 arm was predicted to differ from F1 by 0.006, its F4 arm censored its own metric,
its fall guard was unpassable by the control, its metric was read post-reset, and its config
base would have silently switched on a curriculum whose feedback loop manufactures the desired
result. Two arms at 6 seeds, on a frozen config, with an in-experiment variance estimate, is the
cheapest design that answers a real question.

But it answers a **narrower** question than the brief asks:

- **It answers:** does steepening the torso term move a warm-started policy's sustained posture,
  by how much, and at what cost in tracking and episode length — with a measured, in-experiment
  noise floor rather than an imported 1-df guess.
- **It does not answer:** whether the posture holds to convergence. 350 warm-start iterations is
  ~7% of the envelope run. The envelope run's own posture drifted −0.10 per 400 iterations at
  steady state, so a 350-iteration win is **not** evidence of a stable endpoint.
- **It does not answer:** whether F1 beats F2, F3 or F4. Those comparisons are deliberately not
  attempted, because at this noise floor they are not resolvable in this budget.
- **It does not reach the mocap reference.** *(measured)*: the cost curve turns to −2.0/unit
  above u ≈ 0.87. The honest ceiling for a 350-iteration warm start is ~0.90, not the 0.95–1.00
  human range.
- **It does not test the root cause directly.** §1 attributes the fold to i.i.d. exploration
  being unable to produce a sustained offset. F1 *routes around* that by paying for the state
  every step; it does not prove the mechanism. If G8's entropy logging shows the move tracks
  `action_std` rather than the term change, the follow-up is temporally-correlated exploration
  (coloured noise on the abdomen dimensions), not a further reward tweak.

**One thing to do before any of this, at zero cost:** correct the LOGBOOK "Current state" row.
It currently stitches `best.pt`'s return, length and fall rate to a different checkpoint's
posture, and adds a held-command speed from a third measurement condition. Every number in this
document is downstream of getting that row right.
