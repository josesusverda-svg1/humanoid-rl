# Staged get-up task: implementable spec

Written against the real `Task` interface in `humanoid_rl/tasks/base.py`. Every threshold
below was **measured on this model** through `envs/model_prep.prepare()`; nothing is carried
over from a paper's robot. Reproduction scripts are noted at the end.

Companion documents: [`getup-task-literature.md`](getup-task-literature.md) (what the field
does) and [`generating-fallen-initial-states.md`](generating-fallen-initial-states.md) (how to
build the pose bank, and the two blockers §3.1/§3.2 this spec resolves).

---

## 0. Measured constants

| Quantity | Value |
|---|---|
| `standing_height` (pelvis) | **0.8769 m** |
| `standing_head_height` | **1.5139 m** |
| mass / weight | 50.05 kg / **491 N** |
| `contact_force_threshold` (engine, 2 % bw) | **9.82 N** |
| control step `state.dt` | 0.008 s (125 Hz) |
| settled standing: root / head_ratio / `torso_upright` | 0.875 / 0.998 / **0.994** |
| settled standing: `gravity_body` | (−0.011, 0.000, −1.000) |
| settled standing: foot com z / hand com z | **0.027** / 0.834 |
| settled standing: foot force each | **245.6 N** (0.50 bw each) |
| settled standing: mean \|knee\| | 0.225 rad |

### Three facts about the published signals that change thresholds

1. **`key_body_pos`, `head_pos` and `torso_zaxis` are `mjOBJ_BODY` frame sensors, and in
   MuJoCo that means the *inertial* frame (`xipos`/`ximat`), not the body frame
   (`xbody` → `xpos`/`xmat`).** Measured: `left_foot` body origin is at (0.001, 0.085, 0.052)
   while the published `key_body_pos` is (0.046, 0.085, **0.029**) — the foot's centre of
   mass. A foot flat on the floor therefore reads **z = 0.027–0.029**, not 0.05. A hand
   (4 cm sphere, com = body origin) resting on the floor reads **z = 0.040**.
2. **`torso_upright` carries a constant 5.87° bias.** The torso body's inertial z-axis is
   (−0.102, 0, 0.995) at zero abdomen angles, because the torso body holds the torso sphere
   plus two asymmetric clavicle capsules. So a perfectly upright torso reads **0.9947, never
   1.0**. Any threshold on `torso_upright` must be read as "inertial axis within
   arccos(θ) of vertical, ± 5.9° depending on lean direction".
3. **Actuator order ≠ qpos order.** `ACT2Q = [7..21, 23, 22, 24..28, 30, 29, 31..34]`:
   `hip_y` and `hip_z` are swapped on **both** legs (actuator 15/16 and 22/23). Two shipped
   call sites index across this without a map — see §8.4.

---

## 1. Why staged

Three independent reasons, two of them measured here.

**(a) The correct motion makes negative progress on every scalar the obvious reward uses.**
HumanUP names this as the property that separates get-up from locomotion: to stand up you
must first *lower* your head and fold, to build a base of support. Measured on this body:
the kinematic head-height ratio of the poses a get-up passes through is
supine 0.074 → side 0.191 → all-fours 0.371 → seated 0.517 → kneel 0.714 → half-kneel 0.802 →
standing 1.000, but the *pelvis* height ordering is different and non-monotone
(all-fours 0.64 × H sits **above** kneeling 0.51 × H). A single monotone height reward
therefore either fights phase 1 or prefers the wrong intermediate. A ladder of milestones
does not have to be monotone in any one scalar.

**(b) The task is otherwise sparse to the point of being unlearnable.** Measured: an oracle
over 40 constant PD targets × 16 toppled poses × 5 s — 640 rollouts, with the action band
widened to the full joint range — reached a peak pelvis height of **0.915 m (1.04 × standing)**
and satisfied the standing predicate **0 % of the time, momentarily or held**, and never got
past stage S2. Random search does not find this by accident. Something dense has to lead.

**(c) The stage ladder is the only dense signal that is provably un-farmable.** Paid as a
potential over the *running maximum* stage (§4, term 1), the total shaping over an entire
episode telescopes to `Φ(k_max) − Φ(k_reset)` regardless of the path taken. There is no cycle
with positive payout, so "jump up, collect, fall, jump again" collects exactly once — which is
the user's stated requirement, discharged structurally rather than by tuning. HumanUP's own
progress term, `𝟙(h_t > h_{t−1})`, does **not** have this property: bouncing 1 cm a hundred
times pays a hundred times. That is the same class of bug as E02/E03 in this repo's logbook,
and it is avoided by keying on the running max instead of the previous step.

Stages are a **difficulty ordering, not a mandatory sequence.** A prone body pushing straight
to all-fours skips S1 legitimately and is neither rewarded for nor penalised for the skip,
because the potential pays `Φ(highest reached)`.

---

## 2. Fallen / sitting detection

Sign comes from **`gravity_body`, which is a signed 3-vector**, never from `torso_upright`,
which is a sign-blind cosine (E23). Measured canonical values (`mj_forward`, resting on floor):

| pose | root/H | head_ratio | g_x | g_y | g_z | `torso_upright` | foot z | hand z |
|---|---|---|---|---|---|---|---|---|
| standing | 1.00 | 1.000 | 0 | 0 | **−1.00** | 0.995 | 0.029 | 0.838 |
| supine (face up) | 0.13 | 0.074 | **−1.00** | 0 | 0.00 | −0.102 | 0.158 | 0.155 |
| prone (face down) | 0.16 | 0.090 | **+1.00** | 0 | 0.00 | +0.102 | 0.090 | 0.094 |
| side, left down | 0.33 | 0.191 | 0 | **+1.00** | 0.00 | 0.000 | 0.204/0.374 | 0.042 |
| side, right down | 0.33 | 0.191 | 0 | **−1.00** | 0.00 | 0.000 | 0.374/0.204 | 0.042 |
| seated, legs out | 0.17 | 0.517 | 0 | 0 | **−1.00** | 0.995 | 0.092 | 0.139 |
| kneeling upright | 0.51 | 0.714 | 0 | 0 | **−1.00** | **0.995** | 0.328 | 0.404 |
| half-kneel | 0.66 | 0.802 | 0 | 0 | −1.00 | **0.995** | 0.462/0.062 | 0.538 |
| half-crouch | 0.89 | 0.923 | 0 | 0 | −1.00 | 0.981 | 0.080 | 0.738 |

**A kneeling humanoid is identical to a standing one in `gravity_body` and `torso_upright`.**
Only height, foot contact and knee angle separate them. That single row is why the standing
predicate must be conjunctive and why height-plus-uprightness is not enough.

### Taxonomy (vectorised, all from published fields)

```
g   = state.gravity_body                    # (N,3), signed
h   = state.root_height / H                 # H = 0.8769
hr  = state.head_height_ratio
fz  = state.key_body_pos[:, 0:2, 2]         # 0 left_foot, 1 right_foot
hz  = state.key_body_pos[:, 2:4, 2]         # 2 left_hand, 3 right_hand
knee = np.abs(state.qpos[:, [31, 24]])      # left_knee, right_knee  (qpos slots)

PRONE  =  g[:,0] >= +0.60
SUPINE =  g[:,0] <= -0.60
SIDE   = (np.abs(g[:,1]) >= 0.60) & ~PRONE & ~SUPINE
SEATED = (g[:,2] <= -0.80) & (h < 0.45) & (hr >= 0.40) & (hr < 0.70)
LOW    = (g[:,2] <= -0.80) & (h >= 0.45) & (h < 0.88)     # kneel / squat / quadruped-ish
FALLEN = PRONE | SUPINE | SIDE | SEATED
```

`SEATED` is "sitting on its backside", the case the user named: pelvis upright (g_z ≈ −1),
pelvis low (0.17 × H), head at 0.52 of standing. **Root height cannot separate sitting from
lying on this body** — seated rests at 0.148 m and supine at 0.112 m. `g_z` separates them
cleanly (−1.00 vs 0.00). Prone/supine separate only on the **sign of `g_x`**; a cosine cannot
do it.

---

## 3. The stage ladder

`k ∈ {0,1,2,3,4}`. Each predicate is self-contained; the ratchet takes the max.

### S1 — off the flat / onto a side
```
S1 = (|g_x| <= 0.50 & |g_y| >= 0.50) | (hr >= 0.25)
```
Supine/prone have `|g_x| = 1.00` and `hr ≤ 0.09` → false. Side-lying has `|g_y| = 1.00`,
`|g_x| = 0.00` → true. A sit-up straight from supine reaches `hr ≥ 0.25` → also true.
Measured on 16 toppled bank poses: 6 of 16 start already at S1 (see §5, ratchet init).

### S2 — trunk supported: quadruped **or** seated-upright
```
S2a = (hz.min(1) <= 0.10) & (h_abs >= 0.35) & (g_x >= 0.50) & (knee.mean(1) >= 1.00)
S2b = (g_z <= -0.75) & (hr >= 0.45)
S2  = S2a | S2b
```
S2a is hands-and-knees: both hands within 6 cm of their resting height (0.040), pelvis
raised ≥ 0.35 m, pelvis pitched face-down, knees folded. S2b is any upright-pelvis prop:
seated-legs-out (`g_z −1.00`, `hr 0.517`) ✓, kneeling (`hr 0.714`) ✓, crash-sit (`hr 0.631`) ✓.
Both are genuine milestones and the policy may use either.

### S3 — one foot planted and loaded, pelvis above it
```
loaded = (state.foot_force >= 0.25 * 491) & (fz <= 0.08)          # 123 N, not 9.8 N
S3     = loaded.any(1) & (h_abs - fz[loaded_argmax] >= 0.30) & (hr >= 0.55)
```
The force clause is the anti-cheat: `foot_contact` fires at 2 % of body weight, which a
dangling toe supplies. A quarter of body weight is a foot that is actually carrying you
(standing carries 0.50 bw per foot). The `root − foot ≥ 0.30 m` clause is what forbids
"slam a foot down while lying next to it": supine gives −0.046, half-kneel 0.518, deep squat
0.314, standing 0.848.

### S4 — upright in a normal human pose (the standing predicate `U`)

Nine conjuncts. Measured margin on a settled stand in brackets.

| # | clause | threshold | settled stand |
|---|---|---|---|
| U1 | `root_height ≥ 0.88 · H` | 0.772 m | 0.875 ✓ |
| U2 | `head_height_ratio ≥ 0.90` | — | 0.998 ✓ |
| U3 | `gravity_body[2] ≤ −0.93` | pelvis tilt ≤ 21.6° | −1.000 ✓ |
| U4 | `torso_upright ≥ 0.90` | torso tilt ≤ 25.8° | 0.994 ✓ |
| U5 | `foot_contact.all()` | both loaded | ✓ |
| U6 | `max(foot z) ≤ 0.08` | both feet down | 0.027 ✓ |
| U7 | `min(hand z) ≥ 0.45` | neither hand propping | 0.834 ✓ |
| U8 | `mean(\|knee\|) ≤ 0.50` | not a crouch/kneel | 0.225 ✓ |
| U9 | `\|qvel[2]\| ≤ 0.30` and `‖qvel[3:6]‖ ≤ 1.5` | not mid-flight/tumble | 0.001 / 0.042 ✓ |

**The sign problem is handled by making U4 tight rather than by adding a signed sensor.** At
`torso_upright ≥ 0.90` the tilt is under 25.8° in *either* direction, so forward and backward
folds are both rejected and sign-blindness cannot matter. Verified: a forward waist fold
(abdomen_y +1.4) reads `torso_upright 0.270`, a backward fold (abdomen_y −0.9) reads 0.538 —
both fail. Every *loose* or directional test in this spec (S1, S2a, the taxonomy) uses
`gravity_body`, which is signed.

**Each clause earns its place** — verified against the canonical poses:

| pose | rejected by |
|---|---|
| kneeling | U1 (0.51 H), U6 (foot z 0.328), U8 (knee 2.60), U5 |
| half-kneel | U1 (0.66 H), U2 (0.802), U6 (0.462), U8 (2.17) |
| half-crouch | **U8 only** (knee 1.20) — passes U1 0.89 H, U2 0.923, U4 0.981 |
| forward waist fold | **U4 only** (0.270) — passes U1 1.00 H, U5, U6 |
| backward waist fold | **U4 only** (0.538) — passes U2 0.900 |
| seated | U1, U2 |
| hand-propped stand | U7 |

The half-crouch and half-kneel rows are the important ones: both satisfy the naive test this
project would otherwise have written (`root_height > 0.62·H and torso_upright > 0.8`), and
both are exactly the kneeling local optimum Tao et al. report as the failure mode of a
too-weak get-up objective.

`S4 ⇒ S3 ⇒ S2b`, so the ladder is consistent with a monotone potential.

---

## 4. Reward

14 terms (locomotion has 21). Total clipped at ≥ 0, as `LocomotionTask` does.

```python
reward_term_names = (
    "stage", "head_progress", "upright_hold", "stand_quality", "foot_under",
    "unsupported_rise", "vertical_speed",
    "torque", "dof_vel", "dof_acc", "action_rate", "action_smooth", "dof_pos_limits",
    "alive_floor",
)
```

| # | term | formula | weight | what it prevents |
|---|---|---|---|---|
| 1 | `stage` | `Φ(k_t) − Φ(k_{t−1})`, `k` = running max stage, `Φ = (0, 3, 9, 18, 30)` | 1.0 | Farming a milestone by oscillating across its boundary; re-collecting after a fall. Episode total telescopes to `Φ(k_max) − Φ(k_reset)`. |
| 2 | `head_progress` | `max(0, hr_t − hr_max)`, `hr_max` = running max | 60 | Bouncing the head to farm HumanUP's `𝟙(h_t > h_{t−1})`. Total bounded by `60·(1 − hr_reset) ≈ 55`. |
| 3 | `upright_hold` | `𝟙(U)` per step | 1.0 | The recurring prize. Falling costs every remaining step, so standing and staying beats standing and falling without any extra machinery. |
| 4 | `stand_quality` | `𝟙(U) · exp(−‖v‖²/0.10) · exp(−‖ω‖²/0.50) · exp(−Σ(q−q_nom)²/4)` | 1.0 | Technically-standing-but-wobbling, and standing in a contorted pose. Multiplicative (Tao's `R_balance`) so any factor at zero zeroes the term. |
| 5 | `foot_under` | `𝟙(any loaded foot) · exp(−d²/0.09)`, `d` = horizontal distance from the pelvis ground-projection to the nearest loaded foot | 0.3 | UniReLo's "reaches target height with feet poorly positioned"; feet splayed out front. |
| 6 | `unsupported_rise` | `−max(0, v_z)² · 𝟙(no foot loaded ∧ min(hand z) > 0.15)` | −5.0 | Ballistic launch. Gated so pushing up on the hands (a legitimate get-up phase) is free. |
| 7 | `vertical_speed` | `−max(0, \|v_z\| − 0.5)²` | −2.0 | HumanUP's "< 1 s unsafe ballistic get-up". Human pelvis rise is ≈ 0.4 m/s. |
| 8–13 | regularisers | `torque` −1e-5·Στ², `dof_vel` −1e-4, `dof_acc` −1e-7, `action_rate` −0.01, `action_smooth` −0.01·‖a_t − 2a_{t−1} + a_{t−2}‖² (HoST's L2C2-lite), `dof_pos_limits` −5.0 | audit weights, E09 | Tremor, oscillation ("HoST without smoothness: oscillations in all scenes"), joint-limit fighting. `w_torque` is XBot's −1e-5, **not** T1's −2e-4 (E09: costs −9.4/step). |
| 14 | `alive_floor` | constant 0.02 | 0.02 | Keeps the pre-clip total positive early so the ≥0 clip does not erase the gradient of terms 1–2 while every penalty is active. |

Budget: one-time ladder ≈ 85 (≈ 0.07/step over a 1250-step episode); standing pays ≈ 2.0/step,
so an 8 s hold pays ≈ 2000. Ratio ≈ 23:1 in favour of actually standing. That ordering is the
point: the ladder is scaffolding, the hold is the objective.

`action_smooth` needs `a_{t−2}`; `BatchState` publishes `prev_action` only, so keep
`task_state["prev_action2"]` and roll it in `on_batch_end`.

---

## 5. Initial states

Offline **pose bank**, flat and index-addressable, exactly like `motion/library.py`, indexed
from `GetUpTask.reset_pose()`. Full justification in
[`generating-fallen-initial-states.md`](generating-fallen-initial-states.md); the measured
essentials:

- drop-and-settle costs **36 ms per pose**, 100 % settle within 4 s, 0 divergences in 400;
- a settled pose is a **true fixed point** of the engine's reset (`mj_resetData` → write qpos →
  `mj_forward`): re-applying it drifts 0.0 cm root / 0.00 rad joints over 0.5 s;
- **the drop distribution is the wrong distribution**: real policy falls are 76 % side /
  21 % prone / 3 % supine at mean root height 0.258 m, while free drops are 4 % side /
  51 % prone / 46 % supine at 0.098 m — nearly inverted, and 2.6× flatter;
- the PD target held *while falling* silently controls all diversity: holding the nominal pose
  gives joint sd 0.012 rad (0.5 % of range, a rigid mannequin); holding the spawn pose gives
  0.278 rad (13.3 %).

Mix (from the research, unchanged): **40 % harvested policy falls, 35 % scripted topple,
15 % free fall from height, 10 % hand-authored keyframes** (seated-legs-out, all-fours,
side-lying with an arm trapped, prone with hands under the shoulders — drops produce the
seated case 0 % of 600 samples), plus **8 % of resets from the standing pose** so the hold
reward is exercised from iteration 1.

Bank record per pose: `qpos (nq)`, `qvel (nv)`, and **precomputed** `root_height`,
`head_ratio`, `gravity_body (3)`, `torso_upright`, `foot_z (2)`, `hand_z (2)`,
`foot_force (2)`, `knee_mean`, plus `label` and `tier`.

### The ratchet must be initialised from the reset pose, not from zero

`Task.reset_batch` runs **before** the physics reset, so `state.*` still holds the previous
episode's values for those rows — reading `state.gravity_body` there is a stale-data bug. The
bank's precomputed fields solve it exactly: sample the index in `reset_batch`, then set

```python
ts["stage"][idx]    = bank.stage[pick]        # from stored metadata
ts["best_hr"][idx]  = bank.head_ratio[pick]
ts["hold"][idx]     = 0
```

**Measured why this matters: 6 of 16 toppled bank poses already satisfy S1 at reset.** Without
this, 38 % of episodes would collect `Φ(1) = 3` for doing nothing — the same shape of bug as
"the seed that falls 91 % of the time has the highest `torso_upright`, because resets put it
back upright".

### Two blockers that must be fixed before the bank is built

**5.1 The get-up motion is currently outside the action space.** With
`action_scale_fraction = 0.6`, the commandable knee band is `[0.00, 1.05]` of a `[0, 2.79]`
joint — **37.5 %**. Kneeling is short by 1.55 rad, deep squat by 1.35, sit-legs-out by 0.35,
hands-under-shoulders by 0.40, forward fold by 0.61. *Every* sub-pose of a get-up is
uncommandable.

Fix, measured: set the get-up task's per-joint scale to
`s_j = max(nominal_j − lo_j, hi_j − nominal_j)`. Because `s_j` is ≥ both distances, `a = −1`
clips to `lo_j` and `a = +1` clips to `hi_j`, so **100.0 % of every joint's range becomes
commandable (up from a mean of 56.2 %) while `a = 0` still means the nominal standing pose**.
Knee band `[0.00, 2.79]`, elbow `[0.00, 2.79]`, hip_y `[−2.44, 1.05]`, abdomen_y
`[−1.05, 1.57]`. Cost: 3.1× coarser resolution on the knee, which is what terms 6, 7, 12 and
the hold requirement exist to police — and which §7 re-measures rather than assumes.

**5.2 A fallen reset is spring-loaded by the engine.** `vec_env._do_resets` sets `s.ctrl` and
`_ctrl_filtered` to `_default_joint_pos` on every reset, so step 1 of every get-up episode
commands the *standing* pose from a body on the floor: mean |PD torque| **100.2 N·m**, and
**15.4 of 28 joints saturate their torque limit** on the first step. Fix in §8.2.

---

## 6. Hold, success, and the metric

```python
U        = standing_predicate(state)                 # §3, S4
ts["hold"] = np.where(U, ts["hold"] + 1, 0)          # consecutive steps only
```

- **Reward** gates terms 3 and 4 on `U` per step — no bonus for *reaching* standing, only for
  *being* standing. This is the field's consensus mechanism (HoST's `r_post` group: six terms
  at weight 10 that pay only above `H_stage2`).
- **Success** = `ts["hold"] >= hold_required_steps`. The engine evaluates
  `success_batch(s) & done`, and because this task never terminates on failure, `done` is
  always truncation at `max_episode_steps`. So success ⇔ **`U` held continuously through the
  final `hold_required` steps of the episode** — HoST's "maintained for the remainder of the
  episode" and Tao's explicit 100-step balance phase, both obtained for free from the existing
  engine semantics with no new machinery.
- `hold_required_s = 2.0` → **250 steps at 125 Hz**, inside the user's 1–3 s and matching
  Tao's measured 2.5 s. `max_episode_steps = 1250` (10 s, HoST's horizon), leaving up to 8 s
  to get up — HumanUP's deployed get-up is 8 s.
- **`ever_held`** is tracked separately (a sticky boolean). `ever_held ∧ ¬success` is
  UniReLo's *Time-to-Fall* failure — got up, then fell — and it is the number that tells a
  jump-collect-fall policy from a real one. It must be logged; if it is not, the two are
  indistinguishable in the headline.

A jump-collect-fall policy scores `hold = 0` at truncation → success = 0, and earns strictly
less reward than staying up, because every step spent on the floor forgoes 2.0/step. The
incentive and the metric agree, and neither depends on the other.

---

## 7. Curriculum

**Ship A and B in run 1. Hold C for run 2** — E22b measured a 7.6× outcome spread between
byte-identical configs, so a multi-variable run cannot be read.

**A. Initial-state difficulty**, per environment, modelled on `LocomotionTask`'s radial
difficulty. Bank tiers by distance from standing: **tier 0** seated / all-fours keyframes
(start at S2), **tier 1** side, **tier 2** prone, **tier 3** flat supine. Level
`ℓ ∈ [ℓ_min, 1]` sets the tier mixture (`ℓ = ℓ_min` → tiers 0–1 only; `ℓ = 1` → the measured
policy-fall mix 76/21/3). Promote **+0.05** on success, demote **−0.05** on failure —
**symmetric**, because E20 recorded that +0.05/−0.10 collapses every environment to the floor
inside 150 iterations when failure is common. `ℓ_min = 0.30`, chosen so the easiest tier
trained on is still a real get-up (E20's lesson: the floor is a promise about the easiest
thing worth training on). Graduate recycling at the top, as locomotion does.

**B. Hold duration**, global: `hold_required_s` 0.5 → 2.0 s, +0.1 s when the batch success
rate exceeds 0.5, −0.1 s below 0.2, floor 0.5 s. The first rung must be passable (E17: a
`promote_gait_match` of 0.80 was unreachable and would have pinned the curriculum all run).
**The reported metric always uses the fixed 2.0 s**, never the curriculum value, or the
headline number improves whenever the curriculum eases.

**C. Strong-to-weak authority (run 2).** Tao's mechanism, and the only published one that
addresses "the force must be adequate": train at full torque, then scale `jnt_actfrcrange` by
`β^i` with `β = 0.95` per stage to a floor of 0.6, advancing on demonstrated success. Tao
measured the alternatives and both failed: *"adding an energy cost without the strong-to-weak
curriculum has minimal effect"*, and penalising joint velocities *"either has negligible
effects or leads to training instability"*. Note this model's limits live on
`jnt_actfrcrange` (200 N·m abdomen/hip, 150 knee, 100 shoulder, 90 ankle, 70 elbow, 50 neck);
`actuator_forcerange` is unset.

**Pre-registered decision rule for whether run 2 is needed** (write the prediction before the
run — logbook rule 1): run 2 is triggered iff the run-1 policy shows **median time from first
S2 to first S4 < 1.0 s** or **p95 |v_z| > 0.8 m/s**. Otherwise the motion is not ballistic and
C buys nothing.

**Known risk, not mitigated:** HoST's ablation reports **zero** success on every terrain
without their multi-critic, attributing it to reward groups spanning ~7 orders of magnitude
under one critic. This repo has one critic. Mitigations taken: 14 terms not 21, scales inside
~2 orders of magnitude, per-term logging already in place. E22c measured this repo's critic
fitting cleanly and unsaturated, so the risk is real but not evidenced here. Multi-critic is
the first fallback if run 1 stalls with a healthy ladder and no S4.

**Also known:** HoST found supine and prone training *"negatively impacted performance due to
interference between sampled rollouts"*, and HumanUP trains them separately. The stratified
eval (§9) measures this per class rather than assuming it; split the policy only if per-class
success diverges.

---

## 8. Integration

### 8.1 New files
- **`humanoid_rl/tasks/getup.py`** — `GetUpConfig` (dataclass, every number above) and
  `GetUpTask(Task)`.
- **`humanoid_rl/motion/fallen_bank.py`** — `FallenPoseBank`, flat arrays + metadata + tiered
  sampler, mirroring `motion/library.py`.
- **`scripts/generate_fallen_poses.py`** → `data/fallen/bank.npz`. Build-time validation,
  reject on failure: penetration < 2 mm (achievable ≤ 1.02), finite qpos, fixed-point check
  (0.25 s holding its own pose, root drift < 2 cm, joint drift < 0.05 rad), `qvel` zeroed
  except a deliberate ~10 % slice captured pre-settle with momentum kept.
- **`configs/getup.yaml`**.
- **`scripts/getup_report.py`** — stratified evaluation (§9).

### 8.2 `humanoid_rl/envs/vec_env.py` — three changes

1. **Reset the PD target to the reset pose's own angles** (fixes §5.2). In `_do_resets`, after
   `self._reset_qpos_abs, self._reset_qvel_abs = pose`:
   ```python
   joints = self._reset_qpos_abs[:, self.prepared.actuator_qpos_adr]   # actuator order
   s.ctrl[done_idx] = joints
   self._ctrl_filtered[done_idx] = joints
   ```
   The existing assignments at the top of `_do_resets` stay as the no-`reset_pose` default.
   **Must use the actuator↔qpos map**, not `[:, 7:]` — see 8.4.
2. **Plumb the action scale.** `__init__(..., action_scale_mode: str = "fraction",
   action_scale_beta: float = 1.0)` → `prepare(model_path, action_scale_mode=...)`, then
   `self._action_scale = self.prepared.action_scale * action_scale_beta`. In `model_prep.prepare`,
   `action_scale_mode == "full_range"` computes `np.maximum(angle - lo, hi - angle)` for
   limited joints (unlimited joints keep the current fallback). Default path byte-identical.
3. **Optional, recommended:** publish `torso_zaxis` as a `(N, 3)` `BatchState` field
   (`s.torso_zaxis[idx] = sens[:, tz:tz+3]`) and keep `torso_upright` as its z component.
   Not required by this spec — every predicate here gets its sign from `gravity_body` — but
   E23 cost an analysis cycle for want of exactly this, and it is 3 lines.

### 8.3 Small changes
- `humanoid_rl/config.py`: import `GetUpConfig`, add `getup: GetUpConfig = field(...)`, add
  `EnvConfig.action_scale_mode` and `action_scale_beta`. (`_from_dict` raises on unknown keys,
  so the YAML section will not silently no-op — the E11 failure mode.)
- `humanoid_rl/train.py::_build_task`: add the `kind == "getup"` branch (build the bank, pass
  it to `GetUpTask`) and extend the error message's valid-task list.
- **`humanoid_rl/render.py:202` and `humanoid_rl/viz/skeleton.py:184`** write
  `env.state.task_state["command"][0]`, which **raises `KeyError` for any task without a
  velocity command**. `train.py` catches every exception around video and skeleton capture, so
  a get-up run would silently produce **no videos and no skeletons for the entire run**. Guard
  both with `if "command" in env.state.task_state:`, and give get-up its own render schedule
  (a list of pose classes to start from) with `max_episode_steps` set to the task's own value
  rather than `build_render_env`'s 100 000, or the ratchet never resets inside a video.
- `humanoid_rl/oracle/invariants.py`: the command-coverage checks build a `LocomotionTask`
  probe unconditionally; make them skip when `run.task == "getup"` and add get-up checks —
  (i) `hold_required_steps < max_episode_steps`, (ii) no bank pose satisfies `U` at reset
  except the deliberate standing slice, (iii) the bank's class mix matches the policy-fall
  taxonomy within tolerance, (iv) `Φ` is strictly increasing, (v) `S4 ⇒ S3 ⇒ S2` on the
  canonical pose table.

### 8.4 Two latent bugs this work uncovers (independent of get-up)

`ACT2Q` is not the identity: `hip_y`/`hip_z` are swapped on both legs.

- **`tasks/tracking.py:181`** — `action_offset` returns `self.lib.qpos[idx, 7:]`, in **qpos
  order**, and the engine adds it to `s.ctrl`, which is in **actuator order**. The
  right/left `hip_z` servos are therefore commanded the reference's `hip_y` angle and vice
  versa. During a walk `hip_y` swings ±0.5 rad while `hip_z` stays near 0, so this feeds a
  ±0.5 rad yaw command to both hips throughout every tracked clip.
- **`tasks/locomotion.py:793-795`** — the `dof_pos_limits` penalty compares
  `state.qpos[:, 7:7+nu]` (qpos order) against `self._limit_lo/_hi`, derived from
  `actuator_ctrlrange` (actuator order). Four of 28 joints are compared against the wrong
  limits. Minor in effect (`hip_y` ±2.44/1.05 vs `hip_z` ±1.05), but wrong.

Fix once by exposing `PreparedModel.actuator_qpos_adr` and using it at all three sites.

### 8.5 `Task` interface conformance

| hook | GetUpTask |
|---|---|
| `reward_term_names` | 14 names, §4 |
| `task_obs_dim` | **13**, order frozen, append-only: `[0:5]` stage one-hot, `[5]` `best_hr`, `[6]` `hold/hold_required` clipped, `[7]` `episode_step/max`, `[8]` `head_height_ratio`, `[9]` `root_height/H`, `[10:12]` `foot_z − root_z` (L,R), `[12]` `min(hand_z) − root_z` |
| `init_state` | allocate `stage`, `best_hr`, `hold`, `ever_held`, `prev_action2`, `bank_idx`, `tier`, `difficulty`, `hold_required_steps` |
| `reset_batch` | promote/demote `difficulty` on the finished episode, pick the bank index, **initialise the ratchet from the bank's stored metadata** (§5) |
| `reset_pose` | return `(bank.qpos[pick], bank.qvel[pick])` — absolute, exactly as `TrackingTask` does |
| `observe_batch` | as above |
| `reward_batch` | §4 |
| `terminated_batch` | **NaN guard only** — `~np.isfinite(state.qpos).all(axis=1)`. Falling never ends the episode (VIGOR) |
| `success_batch` | `hold >= hold_required_steps` |
| `on_batch_end` | roll `prev_action2`, tick `hold`/`ever_held`, update the hold curriculum, return per-stage occupancy metrics |
| `mirror_task_obs` | supported: stage/hold/`best_hr`/time are mirror-invariant, slots 10 and 11 swap. Keeps mirror augmentation available — **not** a symmetry reward term (E03) |
| `action_offset` | `None`. The nominal standing pose stays the `a = 0` baseline; expressiveness comes from the scale (§5.1), which keeps standing at zero action |
| `eval_metrics` | §9 |

**Non-Markovian reward warning:** terms 1 and 2 depend on `stage` and `best_hr`, which are
history. They **must** be in the observation (slots 0–5) or the critic is fitting a reward it
cannot see. Slots 8 and 9 are privileged (absolute height is not measurable on hardware); they
are acceptable in simulation, and E09 already lists asymmetric actor-critic as the deferred
alternative.

---

## 9. Evaluation

`evaluate()` already counts one episode per environment (E16, fixed). This task strengthens
that structurally: **with no termination on failure, every episode is exactly
`max_episode_steps` long, so there is no length-dependent selection at all** — the
E16-inverted hazard the research flagged (easy poses finishing first) cannot occur.

- Headline: `success_rate` from `success_batch` — held ≥ 2.0 s through truncation.
- `eval_metrics` is averaged over *steps*, so it must return instantaneous batch means. Use
  the sums-over-envs / `num_envs` convention `LocomotionTask._per_direction_metrics` uses, so
  an empty class cannot divide by zero: per class ∈ {supine, prone, side, seated, standing},
  report `share`, `standing_time_fraction`, `mean_stage`. Averaged over steps these are
  unbiased and interpretable ("fraction of time a supine start spends standing").
- `scripts/getup_report.py` for the strict stratified table: per-class success, per-class
  `ever_held ∧ ¬success` (Time-to-Fall), time-to-first-S4, peak |v_z|, and the stage histogram.
- Calibration from the literature: HumanUP 78.3 % real / HoST ~99 % sim / Jiang 99.5 % sim.

---

## Appendix: reproduction

`scratchpad/getup_probe.py` (canonical pose signal table, action band), `getup_probe2.py`
(inertial-frame finding, hackability under the widened band), `getup_probe3.py` (strict vs
naive predicate on identical rollouts, ladder over canonical poses). All import
`humanoid_rl.envs.model_prep.prepare`, so they see the plant training sees.

Headline measurement: **4 880 rollouts** (zero action; 896 single-joint flicks at ±1; 128
symmetric two-joint flicks; 1 920 random constant full-body targets) from 16 toppled poses,
5 s each, under the **widened** action band → **0 % momentary, 0 % held ≥ 2 s**, highest stage
reached **S2**, peak pelvis height 0.915 m. The plant plus the conjunctive predicate already
satisfy the user's "must not snap upright by flicking one joint"; this is the gate to re-run
whenever the action band or the PD gains change.
