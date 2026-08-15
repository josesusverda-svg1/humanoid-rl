# Get-up task: a minimal, conjunctive design

Design agent, 2026-08-14. Companion to `getup-task-literature.md` (what the field does) and
`generating-fallen-initial-states.md` (how to make the initial states). This file is the
implementable spec: exact predicates over real `BatchState` fields, four reward terms, and the
argument that it cannot be gamed.

Everything marked **measured** was run on this machine against
`humanoid_rl/models/humanoid_scene.xml` through `envs/model_prep.prepare()`. Scripts are in the
appendix at the end.

---

## 0. Measured constants this spec is built on

| Quantity | Value |
|---|---|
| `standing_height` (pelvis) | **0.8769 m** |
| `standing_head_height` | **1.5139 m** |
| body mass / weight | 50.05 kg / 491 N |
| control step `state.dt` | **0.008 s** (125 Hz, decimation 4 × 2 ms) |
| foot contact threshold | 9.82 N (2% body weight); standing reads **246 N per foot** |
| joint torque limits (`jnt_actfrcrange`) | 50 / 70 / 90 / 100 / 150 / 200 N·m |
| Σ τ² while standing still | **160** (mean \|τ\| 0.98 N·m) |

### Pose signatures, measured

`u = -gravity_body[:,2]` (pelvis uprightness, signed), `fx = gravity_body[:,0]`,
`h = head_height_ratio`, `r = root_height`, F = foot touch force.

| Pose | r | u | fx | `torso_upright` | h | feet z | F (N) |
|---|---|---|---|---|---|---|---|
| standing nominal | 0.877 | **1.00** | 0.00 | 0.99 | **1.00** | 0.03 | 246 / 246 |
| **seated, legs out** | **0.054** | **1.00** | −0.09 | **0.97** | **0.45** | 0.09 | **0 / 0** |
| kneeling (0.5 s) | 0.445 | 1.00 | −0.03 | 0.99 | 0.71 | 0.34 | **0 / 0** |
| deep squat | 0.446 | 1.00 | −0.03 | 0.99 | 0.71 | 0.09 | 167 / 167 |
| supine (settled) | 0.085 | 0.07 | **−1.00** | −0.03 | 0.08 | 0.09 | 46 / 46 |
| prone | 0.103 | 0.01 | **+1.00** | 0.13 | 0.07 | 0.09 | — |
| side-lying | 0.181 | 0.11 | 0.01 | 0.09 | 0.15 | 0.05–0.14 | — |

**The seated row is the whole argument for a conjunctive predicate.** A humanoid sitting on its
backside on the floor has `u = 1.00` and `torso_upright = 0.97`: it is *perfectly upright* by
every orientation metric this project owns. What separates it from standing is head height
(0.45 vs 1.00), root height (0.054 vs 0.877) and foot load (0 N vs 246 N). Any success test
built on uprightness — signed or not — passes a humanoid sitting on the floor.

### The sign question, answered on this body

`gravity_body` is a **3-vector in the pelvis frame** and is fully sign-aware:

* `fx = +1` → face-down (**prone**); `fx = −1` → face-up (**supine**)
* `fy = −1` → lying on the **right** side; `fy = +1` → **left** side
* `u = −g_z = +1` → pelvis vertical; `0` → horizontal; `−1` → inverted (handstand)

`torso_upright` is `cos(tilt)` of the trunk and is blind to tilt *direction* (E23). This spec
therefore uses it in exactly one place — as a **symmetric magnitude threshold** (`> 0.90`, i.e.
"trunk within 25.8° of vertical, in any direction"), which is the one use where sign-blindness is
harmless, because we do not care which way it must not lean. It is never used as a quantity to
maximise and never as a diagnostic. Its inversion sensitivity is intact (a handstand reads −1),
so the threshold still excludes inverted poses.

The reward uses **head height × pelvis `u`** instead of trunk cosine, and both are unambiguous:
head height *falls* for a fold in either direction, and `u` is a signed vector component.

---

## 1. Fallen / sitting detection

Pure function of published fields; no history, no new sensors.

```python
u  = -state.gravity_body[:, 2]      # pelvis up, signed
fx =  state.gravity_body[:, 0]      # +prone / -supine
fy =  state.gravity_body[:, 1]      # -right side / +left side
h  =  state.head_height_ratio
r  =  state.root_height / self.standing_height
loaded = state.foot_contact.all(axis=1)
```

**`is_fallen`** (the trigger a walking policy would hand over on):

```python
fallen = (u < 0.5) | (r < 0.50) | (h < 0.60)
```

`u < 0.5` is a 60° pelvis tilt. Lu et al. (2026) gate recovery on `|g_z + 1| > 0.6`, the same
threshold to within rounding. It fires **later** than locomotion's `terminate_height = 0.62 ×
standing`, which is deliberate: the two tasks must not both claim the same boundary. Recovery is
declared over only when the full stand predicate of §2 holds — a Schmitt trigger with a wide gap,
so a walk↔get-up handover cannot chatter (UniReLo reports exactly that chatter from a hard gate).

**Class label** (`np.select`, in this priority order — the classes overlap near the thresholds and
first-match wins):

| Class | Predicate | Measured anchor |
|---|---|---|
| `STANDING` | the stand predicate of §2 | — |
| `CROUCH` | `u > 0.7 and loaded and 0.35 < r < 0.80` | squat r 0.45, u 1.00, loaded |
| `KNEEL` | `u > 0.7 and not loaded and 0.30 < r < 0.80` | kneel r 0.45, u 1.00, unloaded |
| `SIT` | `u > 0.6 and r < 0.35 and h < 0.65` | seated r 0.054, u 1.00, h 0.45 |
| `PRONE` | `fx > 0.6` | fx +1.00 |
| `SUPINE` | `fx < -0.6` | fx −1.00 |
| `SIDE` | `abs(fy) > 0.6` | fy ∓0.99 |
| `TUMBLING` | anything else | — |

`u`, `fx`, `fy` are components of a unit vector, so thresholds at 0.6–0.7 partition cleanly and
`TUMBLING` collects only genuinely ambiguous diagonals. Kneel-vs-crouch is separated by **foot
load**, not by height: measured, both sit at r ≈ 0.45, and only the squat has its feet loaded.

---

## 2. Success: "upright in a normal human pose"

One predicate, seven conjuncts, evaluated every step. Thresholds are fractions of measured model
constants, never absolute metres, so swapping the humanoid cannot silently invalidate them
(`configure_for_model` supplies `standing_height`; `head_height_ratio` is already normalised).

```python
stance = |lateral separation of the two feet in the heading frame|   # locomotion._stance_width

stand = (
      (state.root_height        > 0.80 * standing_height)   # 1  pelvis high        (0.702 m)
    & (state.head_height_ratio  > 0.85)                     # 2  head high          (1.287 m)
    & (-state.gravity_body[:,2] > 0.90)                     # 3  pelvis vertical    (< 25.8°)
    & (state.torso_upright      > 0.90)                     # 4  trunk vertical     (< 25.8°)
    & state.foot_contact.all(axis=1)                        # 5  both feet loaded
    & (state.foot_force.sum(axis=1) > 0.5 * body_weight_n)  # 6  feet carry the body (>245 N)
    & (stance > 0.05) & (stance < 0.50)                     # 7a normal stance width
    & (|lin_vel_body| < 0.5) & (|ang_vel_body| < 1.5)       # 7b still, not passing through
)
```

Why each conjunct is not redundant, checked against the measured table:

1. **root > 0.80·H** kills the squat (r 0.45) and everything on the floor.
2. **head > 0.85** kills the kneel (0.71) and the squat (0.71) independently of (1), and kills
   "tall but stooped".
3. **`u` > 0.90** kills inverted and horizontal pelvises. Signed, so a handstand (`u = −1`) fails.
4. **`torso_upright` > 0.90** is *not* implied by (1)+(2): at full standing height, head ratio 0.85
   still admits `cos(fold) = (1.287 − 0.877)/0.637 = 0.64`, a **50° waist fold** — the exact
   failure that made this project add torso sensors in the first place. This conjunct closes it.
5. **both feet loaded** kills sitting (measured 0 N under both feet while `u = 1.00`,
   `torso_upright = 0.97`), kneeling (0 N), and any airborne/hopping frame.
6. **feet carry >50% of body weight** kills "one toe grazing the floor at 10 N while the knees and
   hands do the work". Conjunct 5 alone accepts 2% of body weight per foot.
7. **stance width band** kills the 61 cm two-footed brace this project has converged on twice
   (E03), and crossed feet. **Velocity bound** kills a ballistic pass-through. Both are evaluated
   only inside a conjunction that already requires uprightness, so the heading angle they use
   (ill-conditioned when the pelvis is face-down) is always well-defined where it is read.

**Deliberately excluded, with reasons.** A foot-height conjunct (`foot_z < 0.12`, HumanUP's) is
implied by conjunct 5 on flat ground and would be *wrong* on the Phase-4 heightfield, where foot z
is a world height and not a clearance. A hand-height conjunct is geometrically implied: with the
root at 0.70 m and the trunk within 26°, the shoulder is at ≈1.18 m and the arm is 0.534 m long,
so a hand cannot get below ≈0.65 m. A joint-pose ("natural stance") term is left out because
posture *style* is what this repo's AMP path exists for; spending a reward term on it here buys a
weight to tune and a term to interact.

**Success reported to the engine.** `success_batch` returns a latch: `held_ever`, true once a
continuous hold has completed. The engine ANDs it with `done`, and because every episode ends at
the same step (§4), the sample is unbiased by construction.

---

## 3. The hold

```python
hold_steps  = np.where(stand, hold_steps + 1, 0)     # resets to zero on any violation
held        = hold_steps >= round(hold_seconds / state.dt)     # 1.5 s -> 188 steps
held_ever  |= held
```

* `hold_seconds = 1.5` (inside the user's 1–3 s; Tao et al. use 100 steps at 40 Hz = 2.5 s).
  **Configured in seconds and converted with `state.dt` at runtime** — E18 was a step count
  copied across a rate change.
* Stored in `state.task_state`, allocated in `init_state`, zeroed in `reset_batch`:
  `hold_steps` (int32), `held_ever` (bool), `first_hold_step` (int32, −1 = never),
  `stand` and `held` (bool caches).
* Computed **once**, at the top of `reward_batch` (the first task callback in `step()`), and cached
  in `task_state` so `terminated_batch`, `success_batch`, `on_batch_end` and `eval_metrics` all
  read the same values instead of recomputing a seven-conjunct predicate four times per step.
* **Payment begins only when the counter reaches T** — there is no first-crossing bonus at all.

**Why jump-collect-fall is strictly dominated, arithmetically.** Standing pays `w_hold` per step
for as long as it lasts. A cycle costs the rise (≈2.5 s) plus the qualifying hold (1.5 s) plus the
fall (≈0.7 s) before payment resumes, so cycling earns `w_hold·H/(4.7 + H) < w_hold` per unit
time for any hold length H. Over an 8 s episode: rise-and-hold ≈ **2470 return**, sit-and-stop
≈ **650**, rise-hold-fall-repeat ≈ **940**. Staying up wins by 2.6× against the best cycling
schedule and there is no first-crossing payment to farm.

---

## 4. Reward: four bounded terms

```python
reward_term_names = ("upright", "height", "hold", "effort")

u      = clip(-state.gravity_body[:, 2], 0, 1)                    # signed, in [0,1]
hh     = clip(state.head_height_ratio, 0, 1)
shape  = (exp(kappa * hh) - 1) / (exp(kappa) - 1)                 # convex, in [0,1], kappa = 3
effort = mean_i( clip(state.torque[:, i] / torque_limit_i, -1, 1) ** 2 )     # in [0,1]

terms[:,0] = w_up     * u                    #  0.5
terms[:,1] = w_height * u * shape            #  1.0   <-- GATED by u
terms[:,2] = w_hold   * held                 #  2.5
terms[:,3] = w_effort * effort               # -0.25
return terms.sum(axis=1)                     # no clipping needed: total in [-0.25, 4.0]
```

| Term | Formula | Weight | What it does / prevents |
|---|---|---|---|
| `upright` | `clip(-g_body_z, 0, 1)` | **+0.5** | The only term with a gradient from a flat body. Rolling a supine pelvis toward vertical raises it from 0 immediately, so the roll-over phase — the part HumanUP had to give its own policy — is paid for. Signed: a handstand scores 0, a face-down body scores 0. |
| `height` | `u · (e^{3h}−1)/(e³−1)` | **+1.0** | The climb. **Multiplied by `u`**, so head height bought without a vertical pelvis (dive, handstand, throwing the head up while horizontal) pays nothing. Convex in h, so the marginal payoff *grows* toward standing — this is what prices the kneeling local optimum out, and κ is the one knob to raise if a kneel or sit is observed, instead of adding a term. |
| `hold` | `1[hold_steps ≥ T]` | **+2.5** | The task. Pays nothing on first crossing; pays every step once the pose has been held T seconds and the counter has not been broken. This is the anti-flash mechanism and the term that makes standing worth *staying in* rather than *reaching*. |
| `effort` | `mean(clip(τ/τ_lim, ±1)²)` | **−0.25** | Prices sustained actuator saturation and buzzing. **Bounded in [0,1] by construction**, so it can never dominate — unlike raw Σ τ², which is *unbounded* here (see below). Measured: standing 0.0003 (free), vigorous whole-body flailing median 0.112 / p90 0.284 / max 0.785. Worst-case cost 6% of the held budget. |

**Why `effort` is normalised and not Σ τ².** `state.torque` is `d.actuator_force`, which on this
model is **not** the applied torque: `actuator_forcelimited` is False everywhere and the physical
clamp lives in `jnt_actfrcrange`, applied to `qfrc_actuator`. Measured under vigorous effort,
Σ τ² has median 1.4e5, p90 2.0e6 and **max 1.35e7** — at locomotion's `w_torque = −1e-5` that is
−19/step at p90 and −135/step at the maximum, against a positive budget of 4. Porting the walking
task's torque penalty would forbid the push-off outright and the failure would look like "the
policy never gets up". The normalised form is bounded, is in interpretable units ("fraction of
actuator capacity in use"), and is free at the goal state.

**Reward ladder, computed from the measured pose table** (κ = 3):

| Pose | `u` | h | shape(h) | upright | height | hold | total |
|---|---|---|---|---|---|---|---|
| supine settled | 0.07 | 0.08 | 0.014 | 0.035 | 0.001 | 0 | **0.036** |
| side-lying | 0.11 | 0.15 | 0.030 | 0.055 | 0.003 | 0 | **0.058** |
| seated on backside | 1.00 | 0.45 | 0.150 | 0.500 | 0.150 | 0 | **0.650** |
| kneeling | 1.00 | 0.71 | 0.388 | 0.500 | 0.388 | 0 | **0.888** |
| deep squat | 1.00 | 0.71 | 0.388 | 0.500 | 0.388 | 0 | **0.888** |
| standing, not yet held | 1.00 | 1.00 | 1.000 | 0.500 | 1.000 | 0 | **1.500** |
| **standing, held** | 1.00 | 1.00 | 1.000 | 0.500 | 1.000 | 2.5 | **4.000** |

Every intermediate posture is on a monotone ladder to the goal, and the goal is 4.5× the best
intermediate. Nothing in the ladder is flat, so there is no plateau to park on for free.

### Termination

```python
terminated = ~np.isfinite(state.qpos).all(axis=1)     # divergence only
```

No fall termination, no success termination (VIGOR does the same, deliberately). Consequences:

* Every episode has **exactly** `max_episode_steps` steps, so the E16 eval selection bias —
  "the first episodes to finish are the failures" — is **structurally impossible**, not merely
  fixed. The research note warns E16 would otherwise come back *inverted* here, because with
  fallen starts the easy poses finish first.
* A fall mid-episode is recoverable and generates get-up data instead of ending the rollout.
* **`eval/fall_rate` becomes structurally 0.0 and must be ignored** for this task. It is the
  headline number on the dashboard, so this needs an Oracle warning (§7), or someone will read
  "falls 0%" as a result.

---

## 5. Initial states

Follows `generating-fallen-initial-states.md`; only the interface and the deltas are given here.

**`humanoid_rl/motion/fallen_bank.py` — `FallenPoseBank`**, flat and index-addressable exactly
like `MotionLibrary`:

```
qpos (K, nq)  qvel (K, nv)  label (K,) int8  difficulty (K,) float  split (K,) int8
meta: standing_height, standing_head_height, body_mass, model_sha1, nq, nv, generator_version
```

**`scripts/generate_fallen_poses.py` → `data/fallen/bank.npz`**, K = 8192 (6144 train / 2048
held-out eval). Measured cost 36 ms per pose single-threaded → ~5 minutes on one core.

Four generators:

1. **Policy falls, ~40%.** Roll out `runs/envelope-.../best.pt` with terminations disabled and
   harvest 1–3 s after `is_fallen` fires. The only on-distribution source. Mirror left/right to
   double it.
2. **Scripted topple, ~35%.** Standing + noise, root impulse 1.5–5 m/s, **PD held at a random
   constant target inside the commandable band** — *not* at the nominal pose. Measured: holding
   nominal while falling gives a joint-angle spread of 0.012 rad (0.5% of range), i.e. one rigid
   mannequin at different yaws; holding the spawn pose gives 0.278 rad (13.3%). Nothing else in
   the pipeline would flag a bank that is large and rank-1.
3. **Free fall from height, ~15%.** Random orientation, covers the supine/prone tails that
   toppling under-produces.
4. **Hand-authored keyframes, ~10%** (HiFAR's KSI). Seated-legs-out, all-fours, side-lying with a
   trapped arm, prone with hands under the shoulders. **The seated pose the user explicitly named
   appears in 0 of 600 drop samples** — it has to be authored. It is passively stable (measured:
   holds at r = 0.054, `u` = 1.00 under PD).

Plus **10% of resets from the standing pose** (UHG's `reset_final_p`), outside the curriculum, so
the hold reward is exercised from iteration 1 and the policy is never out-of-distribution when it
finally arrives at the goal.

Build-time validation, reject on failure: `mj_forward` penetration < 2 mm (achieved ≤ 1.02 mm);
finite qpos; fixed-point check (0.25 s holding its own angles: root drift < 2 cm, joint drift <
0.05 rad; achieved 0.0 / 0.00). `qvel` zeroed except a deliberate ~10% slice captured before
settling with momentum retained, so the policy also sees the still-tumbling case. Yaw and xy are
randomised **at build time** (poses stored already rotated and re-centred), not at reset time:
rotating a free joint's velocity at runtime needs `qvel[0:3]` (world frame) and `qvel[3:6]`
(body frame) handled differently, and that is a silent-wrong-transform waiting to happen.

**Distribution.** The measured policy-fall taxonomy is 76% side / 21% prone / 3% supine, which is
one checkpoint's signature, not a specification. Train on a deliberate stratified mix — side 35%,
supine 20%, prone 20%, sit 10%, crouch/kneel 5%, standing 10% — and **evaluate on a balanced,
held-out, deterministic deal** (§6). The tension is real and is recorded here so the choice is
visible rather than inherited.

`GetUpTask.reset_pose(state, indices, rng)` returns `bank.qpos[idx]`, `bank.qvel[idx]` verbatim.
Measured: a settled pose replayed through the engine's `mj_resetData → write qpos/qvel →
mj_forward` path is a true fixed point (0.0 cm root drift, ≤1.016 mm penetration). Do **not** add
`reset_noise` on top — it would break the settled contact state that made the pose valid.

### Prerequisite engine fixes (both measured, both block the task)

**(a) The action space cannot express a get-up.** Targets are `nominal ± 0.6 × half_range`, so the
knee is commandable over **[0, 1.05] rad of its [0, 2.79] range (37.5%)** and the elbow over
39.4%. Every intermediate pose is out of reach: deep squat needs knee 2.2, all-fours needs 2.4,
kneeling 2.6. "Fold into a pose, push off, rise" is currently *inexpressible*; the policy can only
be pushed there by contact, against a servo fighting back at 150–200 N·m.

Fix: `action_scale_mode = "full_range"` in `prepare()`, i.e.
`action_scale_i = max(nominal_i − lo_i, hi_i − nominal_i)`. This makes every joint angle
commandable **while keeping action = 0 at the standing pose**, which the alternative (mapping to
the range centre) does not — and the standing pose is this task's goal state, so it belongs at the
centre of the action space. Knee scale goes 0.838 → 2.58 (3.1×), so pair it with
`network.init_noise_std: 0.5`. Default stays `"fraction"`, so every existing run is byte-identical.

**(b) A fallen reset is spring-loaded.** `_do_resets` sets `ctrl` and `_ctrl_filtered` to
`_default_joint_pos` on every reset, so step 1 commands the *standing* pose from a body on the
floor. Measured over 32 settled fallen poses: mean \|τ\| **76.8 N·m**, peak **983.7 N·m**,
Σ τ² **6.2e5** — against **1.4 N·m / 10.5 / 1.25e2** when `ctrl` is initialised to the reset
pose's own joint angles. A 5000× impulse at t = 0 of every episode.

Fix: in `_do_resets`, after `reset_pose` returns, set both to the reset pose's own angles. **It
must use the actuator↔qpos map, not `qpos[7:]`** — see the permutation below.

### A real permutation bug found while checking this

`qposadr` per actuator is *not* `7 + i`: the XML declares the hip joints as x, y, z in the body
tree but lists the motors as x, **z**, **y**. Measured mismatches: actuator 15 `right_hip_z` →
qpos 23, actuator 16 `right_hip_y` → qpos 22, and the same swap on the left (22↔23 → 30↔29).

Consequence outside this task: `locomotion.reward_batch` compares `state.qpos[:, 7:7+n]` (joint
order) against `self._limit_lo/_limit_hi` built from `actuator_ctrlrange` (actuator order), so the
`dof_pos_limits` penalty scores **`right_hip_y` against `right_hip_z`'s ±1.0472 range** on all four
hip y/z joints. A hip_y at −1.5 rad (normal in a deep stride, mandatory in a get-up) reads as
0.56 rad past a limit it is nowhere near, at weight −5.0. Not fixed here; reported.

---

## 6. Curriculum

**One curriculum, over initial states only.** Per-environment level `d ∈ [0.25, 1.0]`, init 0.35.
At reset, sample uniformly from bank entries with `difficulty ≤ d`; at `d = 1.0` that is the whole
bank, so easy states never leave the data (the terrain-curriculum "graduate recycling" trick, free
here rather than bolted on).

Per-class difficulty: standing 0.0, crouch 0.2, sit 0.25, side 0.5, supine 0.7, prone 0.85,
tumbling (momentum retained) 1.0.

Promotion `+0.05` if the finished episode set `held_ever`, demotion `−0.05` otherwise —
**symmetric**. E20 is explicit about why: at +0.05/−0.10 the curriculum needs two clean episodes
per failure just to hold station, and every environment slid to the floor within 150 iterations
and stayed there for 500 more. The floor of 0.25 is a promise about the easiest state worth
training on: crouch and sit, never standing (standing starts are dealt separately at a fixed 10%).

**Rejected, with reasons:**

* **A hold-duration curriculum** (T growing 0.5 → 2.0 s). It re-opens the flash-and-fall hack
  during exactly the period when the policy is forming its strategy, and HumanUP reports that
  Stage-I behaviours discovered under loose constraints are often incompatible with the tighter
  Stage-II ones. The hold is the specification; it does not get to be easy first.
* **Assist force** (HoST's 200 N → 0, gated on a near-vertical trunk). Powerful, ablated by its
  authors as load-bearing, but it needs `xfrc_applied` in the worker loop and it is a knob that
  can silently stay on. Held as a **pre-registered contingency**: enable only if, after 500
  iterations, no environment in the hardest tercile has ever completed a hold.
* **A torque/authority curriculum** (Tao's β = 0.95 per stage). Its purpose is to stop a strong
  character brute-forcing an unrealistic motion. Measured here, the plant already refuses:
  0 of 1792 single-joint flicks and 0 of 640 random full-range constant targets ever stood
  (§7). Adding it now would be paying for a problem this body does not have.

---

## 7. Anti-cheat: what blocks what

| # | Measure | Hack it blocks |
|---|---|---|
| 1 | Hold counter resets to 0 on any violation; **no first-crossing bonus**; payment only after T = 1.5 s | "Jump up, collect, fall, repeat" — the exact hack HumanUP observed in Tao's MuJoCo humanoid. A 0.3 s flash earns literally zero. |
| 2 | Success is a **conjunction of 7 necessary conditions**, not a weighted sum | Trading one criterion off against another. A conjunct has no price and cannot be paid for out of another term's budget — which is precisely how `terminate_torso_upright` failed as a soft penalty in this repo. |
| 3 | Conjunct 5+6: both feet loaded **and** carrying >50% of body weight | Sitting on the backside (measured: `u` = 1.00, `torso_upright` = 0.97, **0 N under both feet**), kneeling (0 N), hopping, and "one toe grazing at 10 N". |
| 4 | Conjunct 1+2: root > 0.80·H **and** head > 0.85 | Deep squat and kneel, both of which pass every orientation test (measured `u` = 1.00, `torso_upright` = 0.99). |
| 5 | Conjunct 4 kept despite 1+2 | "Tall but folded at the waist": at full standing height, head ratio 0.85 still admits a 50° trunk fold — the failure that put torso sensors in this repo. |
| 6 | Conjunct 3 uses **signed** `gravity_body[:,2]`; conjunct 4 uses the cosine only as a symmetric magnitude bound | Handstands and inverted poses; and the E23 class of error, where a sign-blind cosine was used as a maximand and a diagnostic. Signed fore/lateral lean is *reported* in eval metrics so a directional lean is visible in numbers, not only on video. |
| 7 | Conjunct 7: stance-width band and velocity bound | The 61 cm two-footed brace (found twice, E03), crossed feet, and ballistic pass-through. |
| 8 | Height term **multiplied** by pelvis uprightness | Getting the head high by diving, handstanding, or flinging the head up from a horizontal pelvis. |
| 9 | Convex height shaping (κ = 3) + hold worth 2.5× the whole rise budget | Parking in the kneel — the local optimum Tao names as *the* failure of get-up learning. Measured ladder: kneel 0.888 vs held 4.000. |
| 10 | **Every term bounded**: upright ≤ 0.5, height ≤ 1.0, hold ≤ 2.5, effort ≥ −0.25 | "One term quietly dominated everything else" — the sentence in `tasks/base.py` describing nearly every debugging session here. With four bounded terms it is visible in the per-term log at a glance, and with the effort term normalised by torque limit it is impossible by construction (raw Σ τ² was measured at 1.35e7, which at the walking weight is −135/step). |
| 11 | **No early termination**; every episode is the same length | E16, inverted. With fallen starts the *easy* poses finish first, so a first-N-episodes eval would report the success rate of whichever class stands fastest. Same-length episodes make the bias impossible rather than fixed. |
| 12 | Eval deals **held-out** poses **deterministically by environment index**, balanced across classes, ignoring the curriculum | A curriculum that drifts toward easy states inflating its own score; and eval variance across checkpoints, which matters because the measured single-run noise floor here is **7.6×** (E22b). |
| 13 | Bank stores `standing_height`, `body_mass` and a model hash; the task asserts they match the runtime model | A stale bank silently generated against a different model. This repo already has a run directory named `ABANDONED-staleClips-tracking`. |
| 14 | Oracle invariants run before compute (§8) | A mis-specified predicate. The strongest one: **the stand predicate must be false for every pose in the bank at t = 0**, and a zero-action rollout must never complete a hold. |
| 15 | Reset `ctrl`/`_ctrl_filtered` to the reset pose's own angles | A 76.8 N·m mean / 983.7 N·m peak impulse on step 1 of every episode, which would make any early-motion analysis a study of the engine rather than the policy. |
| 16 | Plant-level force adequacy: `jnt_actfrcrange` (50–200 N·m) + the 8 Hz action filter, both already shipped | "Snap upright by flicking one joint". **Measured: 0 of 1792 single-joint flicks (28 joints × 2 limits × 32 fallen poses) and 0 of 640 random full-range constant targets ever satisfied even a *naive* `root > 0.62·H and torso_upright > 0.8` test, let alone the strict one, let alone held it.** The dumb-controller floor is 0.0%, so any nonzero success number is real behaviour. Expressing this as a plant property costs no reward term and cannot be traded off. |
| 17 | `eval/fall_rate` is structurally 0 and is flagged as meaningless for this task | Reading the dashboard's headline fall rate as a result. |

### The case for minimalism, in this project's own terms

* **21 terms is 210 pairwise interactions; 4 terms is 6.** Nearly every logbook entry (E02, E03,
  E04, E12, E13, E15, E23) is a term measuring the wrong thing or two terms pulling against each
  other. Reducing the count is the only intervention that attacks all of them at once.
* **The measured single-run noise floor is 7.6×** (E22b: byte-identical configs, 323 vs 2451 mean
  return). A 21-weight reward cannot be tuned empirically on this machine — no A/B under 7× means
  anything. A 4-term reward has **two free ratios** (`w_up : w_height : w_hold` up to scale, and
  `w_effort`), which is at the edge of what can actually be validated here.
* **The specification lives in the predicate, the gradient lives in the reward.** Conjuncts are
  unweighted and untradeable, so all the anti-hack burden sits where it cannot be bargained away;
  the reward only has to supply a monotone slope. That split is what lets the term count be small
  without the task being loose.
* **Get-up is a set-membership problem**, not a continuum of commanded behaviours. Locomotion
  genuinely needs shaping for rhythm, clearance and stance because "walk at 1.2 m/s" does not
  pin down a gait. "Be standing, and stay standing" pins down a set, and a conjunction is the
  natural expression of a set.
* **Style is AMP's job.** HumanUP reports unnatural hand-raising as a residual defect of exactly
  this kind of reward. This repo already has an adversarial motion prior; spending get-up reward
  terms on naturalness duplicates it and adds interactions.

---

## 8. Integration

### New files

| File | Contents |
|---|---|
| `humanoid_rl/tasks/getup.py` | `GetUpConfig` (dataclass, all numbers above) and `GetUpTask(Task)` |
| `humanoid_rl/motion/fallen_bank.py` | `FallenPoseBank` (load/validate/sample, `MotionLibrary`-shaped) |
| `scripts/generate_fallen_poses.py` | the four generators + build-time validation → `data/fallen/bank.npz` |
| `configs/getup.yaml` | the run config below |

### `GetUpTask` against the real interface

| Member | Implementation |
|---|---|
| `reward_term_names` | `("upright", "height", "hold", "effort")` |
| `task_obs_dim` | **2**: `[clip(hold_steps/T, 0, 1), stand]`. The hold threshold is a discontinuity in time-since-standing; without it in the observation the critic cannot represent the value function it is being asked to fit. Both slots are mirror-invariant. |
| `init_state` | allocates `hold_steps`, `held_ever`, `first_hold_step`, `stand`, `held`, `pose_class`, `bank_index`, `difficulty`, `eval_poses` (bool scalar) |
| `reset_batch` | promote/demote `difficulty` from `held_ever` of the finished episode, then choose `bank_index` (curriculum sample, or the fixed deterministic deal when `eval_poses`), zero the hold state |
| `reset_pose` | `return bank.qpos[idx].copy(), bank.qvel[idx].copy()` — takes precedence over `reset_noise`, which is not implemented |
| `observe_batch` | writes the two slots |
| `reward_batch` | computes `stand`, updates the hold counters, caches both in `task_state`, fills the four terms |
| `terminated_batch` | `~np.isfinite(state.qpos).all(axis=1)` only |
| `success_batch` | `held_ever` |
| `on_batch_end` | records `first_hold_step`; returns `{"stand_frac", "hold_frac", "difficulty_median"}` |
| `eval_metrics` | `stand_fraction`, `hold_fraction`, `head_ratio`, `pelvis_up`, `root_height`, `pelvis_lean_fore` (= `gravity_body[:,0]`, **signed**), `foot_force_frac`, `stand_at_end`, and a per-class block using locomotion's sums/`num_envs` pattern so a momentarily empty class cannot divide by zero |
| `mirror_task_obs` | `return task_obs.clone()` — both slots are mirror-invariant. Returning `None` would silently disable mirror augmentation for a task that is perfectly bilaterally symmetric. |
| `action_offset` | `None` (zero action = standing pose) |
| `configure_for_model(standing_height)` | derives every threshold; asserts the bank's `standing_height` matches to 1e-6 |
| `set_joint_limits(lo, hi)` | unused (no joint-limit term) |
| *new* `set_torque_limits(lim)` | called by `vec_env` via `hasattr`, exactly like `set_joint_limits`; falls back to `bank.meta["torque_limits"]` if the engine change is skipped |

**Instance state is forbidden.** `train.py` passes the *same* `Task` object to the training env,
the eval env and the render env, so anything per-environment must live in `state.task_state`.
(The precedent is already load-bearing: `LocomotionTask.init_state` rebinds `self._rng`, so the
render env's generator replaces the trainer's.)

### Changes to existing files

| File | Change |
|---|---|
| `humanoid_rl/config.py` | add `getup: GetUpConfig`; `EnvConfig.action_scale_mode: str = "fraction"`; `RunConfig.task` docstring gains `"getup"` |
| `humanoid_rl/train.py` | `_build_task`: `if kind == "getup": return GetUpTask(config.getup, FallenPoseBank.load(...))`. After building the eval and render envs, call `task.use_eval_poses(env.state)` so they deal the held-out balanced set. |
| `humanoid_rl/envs/model_prep.py` | `prepare(..., action_scale_mode="fraction")`; in `"full_range"` mode `action_scale[i] = max(nominal−lo, hi−nominal)` |
| `humanoid_rl/envs/vec_env.py` | (1) thread `action_scale_mode` through; (2) **the reset-ctrl fix**, using a precomputed actuator→qpos index map (`self._qadr = [m.jnt_qposadr[m.actuator_trnid[i,0]] for i in range(nu)]` — *not* `qpos[7:]`, see the permutation above); (3) `if hasattr(task, "set_torque_limits")`; (4) recommended: publish the full `torso_zaxis` (N,3) and keep `torso_upright` as its `[:,2]` view, so trunk lean has a *direction* — the missing 2 numbers that cost an analysis cycle in E23 |
| `humanoid_rl/render.py` | `render_episode` writes `env.state.task_state["command"][0]` **unconditionally** → `KeyError` for any task without commands. Guard it; add `GETUP_SCHEDULE = [CommandSegment(10.0, (0,0,0), "get up")]` |
| `scripts/play.py`, `render.py`, `gait_report.py`, `posture_score.py` | all construct `LocomotionTask` directly. Extract `_build_task` to a module-level `build_task(config)` and call it, or a get-up video renders the wrong task |
| `humanoid_rl/oracle/invariants.py` | the checks below |

### Oracle invariants (run before any compute)

1. **`hold_seconds / env.dt` is an integer step count and the episode is at least 3× it** — the
   E18 rate-conversion trap.
2. **`gamma` matches the control rate.** `gamma = 0.99` at 125 Hz is a **0.8 s** effective
   horizon, and the hold bonus arrives 2–4 s after the action that earns it: 0.99^375 = 0.023, so
   credit for the push-off is discounted 40×. The references run 50 Hz with 0.99; the equivalent
   here is `0.99^(50/125) = 0.996`. Flag `gamma < 0.995` for a get-up run as a CONTRADICTION.
3. **The bank's `standing_height` / `model_sha1` match the prepared model.**
4. **No bank pose satisfies the stand predicate at t = 0.** If any does, the predicate is wrong,
   or the bank contains standing poses in the fallen split.
5. **A zero-action rollout from 32 bank poses never completes a hold.** Measured floor: 0.0%.
6. **`action_scale_mode == "full_range"` for a get-up run**, else report UNREACHABLE with the
   measured knee coverage (37.5%) and the poses that are out of reach.
7. **`fall_rate` is meaningless when the task never terminates** — emit an OK-severity note so the
   dashboard reader is told, rather than discovering it.

### `configs/getup.yaml` (the parts that differ from `default.yaml`)

```yaml
run:  {task: getup, algo: ppo, name: getup}
env:  {max_episode_steps: 1000, action_scale_mode: full_range}   # 8.0 s at 125 Hz
ppo:  {gamma: 0.996, symmetry_augment: true}
network: {init_noise_std: 0.5}
eval: {num_envs: 64, num_episodes: 64}
domain_rand: {push_interval_s: 0.0}      # first run only; re-enable as a single-variable change
getup:
  hold_seconds: 1.5
  stand_root_frac: 0.80
  stand_head_ratio: 0.85
  stand_pelvis_up: 0.90
  stand_torso_up: 0.90
  stand_foot_load_frac: 0.50
  stand_stance_min: 0.05
  stand_stance_max: 0.50
  stand_lin_vel_max: 0.5
  stand_ang_vel_max: 1.5
  height_kappa: 3.0
  w_upright: 0.5
  w_height: 1.0
  w_hold: 2.5
  w_effort: -0.25
  standing_reset_prob: 0.10
  difficulty_init: 0.35
  difficulty_min: 0.25
  difficulty_step: 0.05          # symmetric promote/demote (E20)
```

`max_episode_steps: 1000` rather than 2500: a rise takes 2–4 s, so a 20 s episode spends 80% of
its samples on the already-solved standing phase. At 8 s the rise is ~40% of the data and the
episode still holds 4 s past a 1.5 s hold. Tao uses 6.25 s + 2.5 s; HoST uses 10 s.

`push_interval_s: 0` for the first run: a 0.7 m/s shove arriving on average every 5 s breaks the
hold counter for reasons outside the policy's control, which is a confound during learning and a
robustness feature afterwards. One variable at a time.

### Pre-registered failure diagnostics

Log these from day one, because each names the conjunct most likely to be wrong:

* `foot_force_frac` while `stand` is otherwise satisfied — if holds never complete, conjunct 6
  (50% of body weight) is the first suspect.
* `stand_at_end` vs `held_ever` — the gap is UniReLo's Time-to-Fall: got up, then fell.
* per-class `stand_fraction` — a headline success rate is the rate on whichever class is easiest.
* `sit_fraction` and `kneel_fraction` — the two named local optima, visible in numbers instead of
  only on video. If either rises while `stand_fraction` stalls, raise `height_kappa`. Do not add
  a term.

---

## Appendix: scripts

Reproduced under `docs/research/getup-design-scripts/`: `measure_poses.py` (pose signature table,
commandable band, actuator↔qpos permutation), `hackability_floor.py` (zero action / 56 single-joint
flicks / 20 random full-range constants × 32 fallen poses, scored against naive, strict and held
predicates), `reset_torque.py` (the spring-loaded reset), `effort_scale.py` (Σ τ² and the
normalised effort distribution).
