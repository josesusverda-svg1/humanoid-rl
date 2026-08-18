# Get-up task: implementable specification

Status: not implemented. This is the spec to build against.
Measurement scripts: `docs/research/getup-spec-scripts/p1..p5.py` (re-runnable, no checkpoint needed).
Background research: `docs/research/getup-task-literature.md`, `docs/research/generating-fallen-initial-states.md`.

Every number below was measured on `humanoid_rl/models/humanoid_scene.xml` through
`envs/model_prep.prepare()` during the writing of this spec, not copied from a paper and not
assumed. Where a claim is inherited from earlier research rather than re-measured here, it says so.

**Model constants, measured.** `standing_height H = 0.8769 m`, `standing_head = 1.5139 m`,
mass `50.050 kg`, weight `BW = 490.99 N`, `dt = 0.008 s` (125 Hz), `contact_force_threshold =
9.820 N`, `nq = 35`, `nv = 34`, `nu = 28`, `key_body_names = (left_foot, right_foot, left_hand,
right_hand)`. Torque ceilings live on `jnt_actfrcrange`: 200 N·m abdomen/hip, 150 knee, 100
shoulder, 90 ankle, 70 elbow, 50 neck. Holding the nominal stand needs a peak of **14.8 N·m**,
9.9 % of the tightest ceiling.

---

## 0. Two facts that change what is buildable, and must be fixed first

Both are pre-existing defects. Building the pose bank before they are fixed is an analysis
cycle spent on the wrong object.

### 0.1 The action space cannot express a get-up

`prepare(action_scale_fraction=0.6)` makes a full-scale action an offset of 0.6 × half-range
from the nominal pose, clipped to the joint limit. Measured coverage of each joint's full range:

| joint | range | nominal | commandable band at 0.6 | coverage |
|---|---|---|---|---|
| right_knee | [0.00, 2.79] | 0.209 | [0.00, 1.05] | **37.5 %** |
| right_elbow | [0.00, 2.79] | 0.262 | [0.00, 1.10] | 39.4 % |
| right_hip_y | [−2.44, 1.05] | −0.105 | [−1.15, 0.94] | 60.0 % |
| abdomen_y | [−1.05, 1.57] | 0.000 | [−0.79, 0.79] | 60.0 % |
| **mean over 28 joints** | | | | **56.2 %** |

A kneel needs knee ≈ 2.4–2.5 rad and all-fours ≈ 2.4. Neither is commandable. "Fold into a
pose, push off the legs and arms, and rise" is currently inexpressible; the policy can only be
pushed there by contact, against a servo pulling back at up to 200 N·m.

**Fix.** Add `action_scale_mode` to `prepare()`. In `"full_range"` mode,
`action_scale[i] = max(nominal_i − lo_i, hi_i − nominal_i)`. Measured: coverage goes to
**100.0 %** on every joint while `a = 0` still means the nominal standing pose, because the
scale is a half-width about the nominal, not a re-centring. Largest resulting offset 3.369 rad
(right_shoulder_z). Default stays `"fraction"`, so every existing run is byte-identical.

Cost: 3.1× coarser knee resolution. That is the trade this spec's dumb-controller floor (§6.3)
is measured against, and it must be re-measured whenever the band or the PD gains change.

### 0.2 Every fallen reset is spring-loaded by the engine

`vec_env._do_resets` sets `s.ctrl[done_idx]` and `_ctrl_filtered[done_idx]` to
`_default_joint_pos` (lines 447, 486). Step 1 of a get-up episode therefore commands the
**standing** pose from a body lying on the floor. Measured over 24 settled contorted fallen
poses, first control step:

| ctrl seeded from | mean \|τ\| | max \|τ\| | joints at their torque ceiling |
|---|---|---|---|
| `_default_joint_pos` (current) | 92.8 N·m | 1002.5 N·m | 6.5 of 28 |
| the reset pose's own joint angles | **1.47 N·m** | **11.1 N·m** | **0.0** |

A 63× reduction in mean demand. Without this, the first ~100 ms of every episode is a
full-torque convulsion no policy chose, and any early-motion analysis studies the engine.

**Fix**, in `_do_resets`, only when `self._has_reset_pose`:

```python
joints = self._reset_qpos_abs[:, self.prepared.actuator_qpos_adr]   # ACTUATOR order
s.ctrl[done_idx] = joints
self._ctrl_filtered[done_idx] = joints
```

It **must** use an actuator→qpos map, not `qpos[:, 7:]`. Measured: `qpos[7:]` is not in
actuator order. `hip_y` and `hip_z` are transposed on both legs — actuator 15 (`right_hip_z`)
reads qpos 23, actuator 16 (`right_hip_y`) reads qpos 22, and the same on the left. Four of 28
joints are swapped. Expose `PreparedModel.actuator_qpos_adr` and use it everywhere.

> Two live bugs found by this measurement, out of scope here but worth logging.
> `tasks/tracking.py:181` returns `self.lib.qpos[idx, 7:]` as an `action_offset`, which the
> engine adds to `ctrl` in actuator order — so both hip_z servos are commanded the reference's
> hip_y angle and vice versa on every tracked clip.
> `tasks/locomotion.py:791-794` compares `qpos[:, 7:7+nu]` against limits derived from
> `actuator_ctrlrange`, so `hip_y` (±2.44/1.05) is scored against `hip_z`'s (±1.047) at weight −5.0.

---

## 1. Predicates

All fields exist in `BatchState` today except `torso_zaxis`, which is three lines to publish
(§7.2). Everything is pure numpy over `(num_envs, …)`.

```python
H  = standing_height                       # 0.8769, from configure_for_model
BW = 490.99                                # nominal body weight, N
TZ_BIAS = -0.1023                          # measured torso z-axis fore offset at zero abdomen

g   = state.gravity_body                   # (N,3) SIGNED
h   = state.root_height
hr  = state.head_height_ratio
fz  = state.key_body_pos[:, 0:2, 2]        # foot world z (left, right)
hz  = state.key_body_pos[:, 2:4, 2]        # hand world z
F   = state.foot_force                     # (N,2) newtons
knee = state.qpos[:, [KNEE_R_ADR, KNEE_L_ADR]]      # 24, 31 — from actuator_qpos_adr
sep  = norm(key_body_pos[:,0,:2] - key_body_pos[:,1,:2], axis=1)   # PLANAR foot separation

c, s_ = cos(-heading), sin(-heading)
tzv   = state.torso_zaxis                  # (N,3) unit vector, NEW
lean_fore = (c*tzv[:,0] - s_*tzv[:,1]) - TZ_BIAS    # + is forward, − is backward
lean_side =  s_*tzv[:,0] + c*tzv[:,1]              # + is left
```

### 1.1 Fallen / sitting taxonomy — a TOTAL function

Used for the initial-state bank, the curriculum, and per-class evaluation. Every state gets
exactly one label; the classes are ordered and first match wins. `np.select` with a default.

```python
DOWN   = (hr < 0.60) | (h < 0.50*H)
PRONE  = DOWN & (g[:,0] >=  0.60)
SUPINE = DOWN & (g[:,0] <= -0.60)
SIDE   = DOWN & (abs(g[:,1]) >= 0.60) & ~PRONE & ~SUPINE
SEATED = DOWN & (g[:,2] <= -0.70) & (F.sum(1) < 0.20*BW)
LOW    = DOWN & ~(PRONE|SUPINE|SIDE|SEATED)     # kneel, all-fours, mid-tumble, diagonals
UP     = the standing predicate U, §1.2
MID    = ~DOWN & ~U                             # crouch, half-kneel, transit
```

`PRONE | SUPINE | SIDE | SEATED | LOW | MID | UP` is a partition by construction. The `LOW`
and `MID` buckets exist so that no state is unlabelled — an unlabelled state belongs to no
class in the stratified eval, and the per-class rates then silently omit a population.

**Measured anchors** (settled, PD holding each pose):

| pose | h/H | hr | g_x | g_y | g_z | torso_upright | F sum | label |
|---|---|---|---|---|---|---|---|---|
| standing nominal | 1.00 | 0.995 | −0.08 | 0.00 | −1.00 | **+0.984** | 1.00 BW | UP |
| supine | 0.10 | 0.078 | **−1.00** | 0.00 | −0.07 | −0.033 | 0.19 BW | SUPINE |
| prone | 0.12 | 0.069 | **+1.00** | 0.00 | −0.01 | +0.128 | 0.26 BW | PRONE |
| side, left down | 0.21 | 0.153 | +0.01 | **+0.99** | −0.11 | +0.092 | 0.24 BW | SIDE |
| side, right down | 0.21 | 0.153 | +0.01 | **−0.99** | −0.11 | +0.092 | 0.24 BW | SIDE |
| **seated, legs out** | 0.06 | 0.450 | −0.13 | 0.00 | **−0.99** | **+0.964** | **0.00 BW** | SEATED |
| all-fours | 0.51 | 0.062 | +0.99 | 0.00 | −0.11 | −0.949 | 0.00 BW | LOW |
| kneel, toppling | 0.46 | 0.532 | +0.78 | 0.00 | −0.63 | +0.709 | 0.00 BW | LOW |

**The row that decides the design is `seated`.** The user's named case reads
`torso_upright = +0.964` and `gravity_body[2] = −0.99` — perfectly upright by *both*
orientation signals this project owns — while the pelvis sits at 5.4 cm and **neither foot
carries a newton**. No orientation-only detector can see it. Root height cannot separate it
from lying either (seated 0.054 m, supine 0.085 m, prone 0.103 m). It takes `head_height_ratio`
(0.450 vs 0.069–0.078) plus foot load.

Prone and supine separate **only** on the sign of `g_x`; side-lying only on the sign of `g_y`.
Both are components of a signed 3-vector. No cosine can do this.

### 1.2 The standing predicate `U`

Thirteen conjuncts. A conjunct has no price and cannot be paid for out of another term's
budget, which is exactly how this repo's soft `terminate_torso_upright` failed. All of the
anti-hack burden sits here; the reward only supplies a slope.

| # | clause | threshold | settled stand | rejects, alone |
|---|---|---|---|---|
| U1 | `root_height ≥ 0.85·H` | 0.745 m | 0.873 (0.995 H) | crouch, kneel, everything on the floor |
| U2 | `head_height_ratio ≥ 0.90` | — | 0.995 | seated (0.450), stooping |
| U3 | `gravity_body[:,2] ≤ −0.93` | 21.6° pelvis tilt | −1.000 | inverted and horizontal pelvises |
| U4 | `torso_upright ≥ 0.85` | inversion guard only | +0.984 | inverted torso |
| U5 | `abs(lean_fore) ≤ 0.35` | ±0.35 rad waist fold | ≈ 0.00 | **forward** stoop |
| U6 | `abs(lean_side) ≤ 0.35` | | ≈ 0.00 | lateral fold |
| U7 | `F.sum() ≥ 0.60·BW` | 295 N | 491 N (1.00 BW) | hands/knees carrying the body |
| U8 | `F.min() ≥ 0.20·BW` | 98 N | 245.7 N (0.50 BW) | one-legged stand, grazing toe |
| U9 | `fz.max() ≤ 0.10 m` | | 0.027 m | airborne self-contact (§6.4), one foot up |
| U10 | `hz.min() ≥ 0.40 m` | | 0.835 m | tripod stand on a hand |
| U11 | `knee.max() ≤ 0.60 rad` | 34° | 0.24 rad | half-crouch, lunge, ski tuck |
| U12 | `0.08 ≤ sep ≤ 0.35 m` | | 0.170 m | **both** brace families (§6.1, §6.2) |
| U13 | `‖qvel[0:3]‖ ≤ 0.40` and `‖qvel[3:6]‖ ≤ 1.5` | | ≈ 0.00 | ballistic pass-through |

Notes that are load-bearing:

- **U7/U8 use foot FORCE, never `foot_contact`.** Measured: `foot_contact` fires at 9.82 N
  (2 % of body weight) and is **True for a supine corpse** (46.4 N per foot), for a prone one
  (65.0 N), and for side-lying (102.0 N). HumanUP's published "standing on feet" indicator
  `1((‖F‖>0) & (h_feet<0.2))` evaluates True for a corpse on this body.
- **U7/U8 thresholds are fractions of the NOMINAL weight**, while `domain_rand.mass_scale_range
  = (0.85, 1.15)` varies the real weight per pool entry. At the light end a settled stand reads
  0.85 BW total and 0.425 BW per foot, against thresholds of 0.60 and 0.20. The margin covers
  the randomisation; an Oracle check asserts `0.60 < 0.85 × 0.85`.
- **U12 is the PLANAR separation, not `locomotion._stance_width`.** That helper returns
  `abs(-sin(h)*rel_x + cos(h)*rel_y)`, the lateral component only. Measured: a fore-aft split
  at `hip_y = ±0.5` puts the feet **0.878 m apart** and `_stance_width` reports **0.142 m**.
- **U11 uses `max`, not `mean`.** A mean over two legs lets one knee at 1.0 rad be paid for by
  the other locked at 0.0.
- **`sep ≥ 0.08` blocks crossed feet only by magnitude.** A signed lateral clause
  (`lat_signed > 0.03`, left foot left of right) is a cheap addition and is recommended;
  `np.abs` in the existing helper is sign-blind and would score a 36 cm scissor as a normal stance.

---

## 2. Success, and the sign-blindness of `torso_upright`

### 2.1 The measurement

`torso_upright` is `cos(tilt)` and cannot tell a forward fold from a backward one. That is
already logged (E23). What was **not** logged, and what this spec measures, is that it is also
**biased and asymmetric on this body**. The torso body's inertial z-axis at zero abdomen angles
is `(−0.1023, 0, +0.9947)` — a built-in 5.87° backward lean from the torso sphere plus two
asymmetric clavicle capsules. Consequences, measured by sweeping `abdomen_y`:

| abdomen_y | torso_upright | apparent tilt | `lean_fore` (de-biased) | head_ratio |
|---|---|---|---|---|
| −0.60 (back) | 0.7632 | 40.3° | **−0.544** | 0.9540 |
| −0.40 | 0.8764 | 28.8° | −0.379 | 0.9792 |
| −0.20 | 0.9546 | 17.3° | −0.196 | 0.9947 |
| **0.00** | **0.9947** | **5.9°** | **0.000** | 1.0000 |
| +0.20 | **0.9953** | **5.6°** | +0.199 | 0.9947 |
| +0.40 | 0.9561 | 17.0° | +0.395 | 0.9792 |
| +0.60 (fwd) | 0.8788 | 28.5° | **+0.579** | 0.9540 |

`torso_upright` **peaks at abdomen_y = +0.20, not at zero.** A threshold of 0.85 on it accepts
a **+0.66 rad forward** fold and rejects a **−0.45 rad backward** one — a 1.47× directional
asymmetry, in a clause a reader would describe as "within 32° either way". `head_height_ratio`
by contrast is exactly symmetric (0.9540 at both ±0.60) — a height has no sign to get wrong —
but it is also nearly blind: a 28° fold costs it 4.6 %.

### 2.2 How this spec handles it

1. The direction comes from the **signed** heading-frame components of the full torso z-axis,
   `lean_fore` and `lean_side`, de-biased by the measured −0.1023 offset. Measured symmetry of
   the resulting clause `|lean_fore| ≤ 0.35`: it accepts up to +0.354 rad forward and −0.375
   rad backward. Symmetric to within 6 %.
2. `torso_upright` is used at **exactly one place**, U4, purely as an inversion guard. Given
   U5 and U6 the unit-vector identity forces `|tz_z| ≥ 0.869`, so U4 decides only the sign of
   an otherwise-determined quantity. That is the one use where sign-blindness cannot bite.
3. The **reward never uses the cosine at all** (§4).
4. `eval_metrics` may never report `torso_upright` without `lean_fore` and `lean_side`
   beside it. This is a reporting rule, not a suggestion: E23 cost an analysis cycle because a
   sign-blind number shipped as a reward term, a termination condition and a headline metric.

### 2.3 Success

```python
success = task_state["held_ever"] & task_state["standing"]
```

returned from `success_batch`. The engine ANDs it with `done` (`vec_env.py:587`), and because
`terminated_batch` is a NaN guard only, `done` is always truncation at `max_episode_steps`.
So success reads: **"held `U` continuously for the required duration at some point, AND is
standing when the clock runs out."**

Both halves are necessary and each blocks the other's hack:

- `held_ever` alone scores "got up, held 2 s at t = 3 s, then lay on the floor for 7 s" as a
  success. That is UniReLo's Time-to-Fall failure and it is a real published behaviour.
- standing-at-the-end alone scores a single upright frame at the buzzer.

The gap between them is reported separately as `time_to_fall_frac` = `held_ever & ~standing`.

---

## 3. The hold, its bookkeeping, and why duration alone does not work

### 3.1 The measurement that decides this section

A humanoid holding a **fixed** joint configuration is an inverted pendulum, so it topples. The
question is how long that takes relative to the hold. Measured with the real plant (125 Hz,
decimation 4, 8 Hz action filter), zero action, `ctrl` frozen at the nominal joint angles, over
the whole 64-entry randomised model pool, counting the longest run satisfying `U`:

| | steps | seconds |
|---|---|---|
| min | 137 | 1.10 |
| median | **287** | **2.30** |
| p95 | 898 | 7.18 |
| max | 898 | 7.18 |

| hold requirement | fraction of the pool that passes it with **zero policy involvement** |
|---|---|
| 250 steps (2.0 s) | **69 %** |
| 375 steps (3.0 s) | **33 %** |
| 500 steps (4.0 s) | **28 %** |

**No hold duration inside — or near — the user's 1–3 s window proves balance on this plant.**
Raising it to 4 s still leaves 28 % of the model pool passable by a mannequin. This refutes the
"hold length is the anti-ballistic mechanism" assumption that all three candidate designs made.

What does work is a disturbance inside the window. Same rollouts, one shove at step 90:

| shove | fraction still reaching 250 steps | fraction reaching 375 |
|---|---|---|
| 0.3 m/s | 28 % | 25 % |
| 0.5 m/s | **6 %** | 3 % |
| 0.7 m/s | **0 %** | 0 % |

For reference, `domain_rand.push_vel_xy` is already 0.7 m/s, so this magnitude is inside what
the plant sees today, and it is applied as `d.qvel[0:3] += push`, which sets the root velocity
directly — at 0.6 m/s the instantaneous root speed lands above U13's 0.40 bound for a few steps,
so a genuinely balanced policy re-enters the predicate rather than a mannequin passing through it.

### 3.2 The hold as specified

- `hold_seconds = 2.0`, **configured in seconds** and converted with `state.dt` at runtime.
  A hardcoded step count is a repeat of E18 (1000 steps assumed 50 Hz on a 125 Hz loop). Oracle
  asserts `hold_seconds/dt` is an integer and that the episode is at least 3× it.
  At 125 Hz that is `HOLD = 250` steps. Inside the user's 1–3 s window.
- **Consecutive**, hard-reset to 0 on any miss, never decayed. A cumulative counter is satisfied
  by 250 separate one-step flashes, which is the jump-collect-fall behaviour itself.
- **A scripted shove fires inside every hold attempt.** When `hold_steps` reaches a per-attempt
  trigger drawn uniformly from [40, 140] steps, the task requests a push of magnitude
  `hold_push_vel = 0.6` m/s in a **uniformly random direction**. The trigger is redrawn each
  time the counter resets, so a policy cannot restart the counter and coast. The direction is
  not observable in advance, so it cannot be pre-braced directionally.
- **`hold_steps` is NOT observed.** A policy that can see "112 steps to go" will schedule its
  collapse, and one that can see the trigger will pre-brace.

### 3.3 Bookkeeping and autoreset

Everything lives in `state.task_state`, allocated once in `init_state`, never reallocated. The
engine's call order inside `step()` is fixed and was read from the source, not assumed:

```
573  reward = task.reward_batch(s, terms)      <-- ALL hold bookkeeping happens here
580  task.on_batch_end(s, {})                  <-- push request written here
582  terminated = task.terminated_batch(s)
587  success  = task.success_batch(s) & done   <-- latch is READ here
595  _do_resets(done_idx, _select_pushes())
       483  s.episode_step[done_idx] = 0       <-- BEFORE reset_batch
       486  s.ctrl[done_idx] = default         <-- the §0.2 fix goes here
       489  task.reset_batch(s, done_idx, rng) <-- latch is CLEARED here
       493  task.reset_pose(...)
```

`reward_batch` is the only place `hold_steps` is mutated. It is called from exactly one site
(`vec_env.py:573`), once per step, and it runs before the termination and success callbacks, so
the counter, the reward and the success flag all read the same step's value.

```python
# top of reward_batch
U = self._standing(state)                                  # 13 conjuncts, computed ONCE
ts = state.task_state
ts["standing"][:] = U
ts["hold_steps"][:] = np.where(U, ts["hold_steps"] + 1, 0)
np.logical_or(ts["held_ever"], ts["hold_steps"] >= self.hold_steps, out=ts["held_ever"])
fresh = U & (ts["hold_steps"] == 1)                         # a hold attempt just began
ts["push_trigger"][fresh] = self._rng.integers(40, 141, fresh.sum())
fire = U & (ts["hold_steps"] == ts["push_trigger"])
ts["push_now"][:] = fire                                    # consumed by the engine at 595
# (direction drawn in on_batch_end into ts["push_request"])
```

`reset_batch` clears the episode-scoped state for `indices` **only**, and records the finished
episode's outcome **before** clearing it:

```python
self._record_outcome(state, indices)     # per-class success, curriculum promote/demote
for k in ("hold_steps", "held_ever", "standing", "push_now", "prev_prev_action"):
    ts[k][indices] = 0
ts["push_trigger"][indices] = 10**9
assert not ts["held_ever"][indices].any()
```

Three ways this breaks silently, all of which the assertion or an Oracle check catches:

1. Not clearing `held_ever` → every episode after the first reports success.
2. Clearing in `on_batch_end` (line 580) instead → the latch is wiped before `success_batch`
   reads it at 587, and success is always 0.
3. Advancing the counter in `on_batch_end` → the counter lags the reward by one step.

**Never read `state.episode_step` inside `reset_batch`.** Line 483 zeroes it before line 489.
`LocomotionTask.reset_batch:531` does exactly this and has been feeding `episode_len_ema` zeros
for the whole project; it is on the dashboard.

---

## 4. Reward

Seven terms. This project's failures come from term interactions, not from missing terms, so
every term is bounded, and the three that could be farmed while lying on the floor are gated on
`U`. Weights are absolute, not curriculum-scaled.

`reward_term_names = ("upright", "rise", "stand", "quiet", "posture", "effort", "smooth")`

| # | term | formula | weight | bound | the hack it blocks |
|---|---|---|---|---|---|
| 1 | `upright` | `clip(-g[:,2], 0, 1)` | **+0.5** | [0, 0.5] | Total sparsity. This is the only term with a gradient from flat on the floor: rolling a supine pelvis toward vertical raises it from 0 immediately, so the roll-over phase is paid for. Signed by construction — a handstand scores 0, a face-down body scores 0. |
| 2 | `rise` | `u · (exp(3h)−1)/(exp(3)−1)`, `u = clip(-g[:,2],0,1)`, `h = clip(head_ratio, 0, 1)` | **+1.0** | [0, 1.0] | **h is clipped at 1.0**, so throwing the head above standing height pays nothing — blocks the head-whip (measured: a 3 m/s launch reaches `head_ratio` 1.30, and tiptoe already reads 1.008). **Multiplied by pelvis uprightness**, so height bought by diving or handstanding pays nothing. **Convex in h** (HumanUP's `exp(h)−1`, renormalised), so the marginal payoff grows toward standing — blocks parking in the kneel, the local optimum Tao et al. name as *the* failure mode of get-up learning. This is the single knob to steepen if a kneel is observed; do not add a term. |
| 3 | `stand` | `1(U)` | **+2.0** | [0, 2.0] | Paid **per step, from the first step U holds** — there is no first-crossing bonus anywhere in this design, and no ramp. Makes standing worth *staying in* rather than *reaching*. Blocks cycling arithmetically: one fall-and-recover cycle costs ≈1.5 s of transit at ≈0.3/step instead of 2.5/step, i.e. ≈320 forgone reward, and gains nothing. A ramp was considered and rejected — it erases the gradient for the honest partial skill along with the cheat. |
| 4 | `quiet` | `1(U) · exp(−‖v‖²/0.25 − ‖ω‖²/4.0)` | **+0.5** | [0, 0.5] | The corpse hack. An **un**gated stillness term is maximised by a motionless body on the floor; this repo has shipped that exact bug twice (E02, E03). Gated on U it is worth zero to a corpse. Also blocks technically-standing-but-wobbling. |
| 5 | `posture` | `1(U) · exp(−Σ(q−q_nom)²/2.0)` over all 28 joints | **+0.5** | [0, 0.5] | Upright-by-the-thresholds but contorted. Same U gate for the same reason. |
| 6 | `effort` | `−mean_i clip(\|τ_i\|/τ_lim_i, 0, 2)² / 4` | **−0.25** | [−0.25, 0] | Sustained actuator saturation. Normalised by each joint's `jnt_actfrcrange` so the term is dimensionless and **cannot dominate**: measured, holding the stand peaks at ratio 0.099, costing 0.0025/step, 0.1 % of the standing reward. The clip at 2 (not 1) keeps it informative to 200 % of limit. |
| 7 | `smooth` | `−mean((a_t − 2a_{t−1} + a_{t−2})²)/16` | **−0.10** | [−0.10, 0] | Oscillation. HoST's ablation without their L2C2 smoothness term: "motion oscillations are observed in all scenes… often leading to standing-up failures." `a_{t−2}` is not in `BatchState`; keep `prev_prev_action` in task_state and roll it at the **end** of `reward_batch`. |

**Total is in [−0.35, +3.5] and is returned WITHOUT the `np.maximum(…, 0)` clip** that
`LocomotionTask` uses. Locomotion can afford that clip because its positive budget is ≈6/step;
here a prone policy's positive budget is ≈0/step, so a clip at zero would bind on most steps and
erase the entire penalty gradient — measured elsewhere in this project's attack phase at 45–100 %
of steps. The bounds make the clip unnecessary: the worst possible penalty is 0.35/step against
a rise gain of up to 1.5/step, a 4× margin, and E09's sizing rule is satisfied with room to spare.

**Terms deliberately NOT included, each with its reason.**

- *A "tuck the feet under the pelvis" term.* Attacked and broken: gated on being down, it is
  maximised by a fetal curl on the floor with a foot pressed under the hip, paying up to 0.5/step
  forever for never leaving the ground. There is no gating that fixes it without making it
  redundant with `rise`.
- *A vertical-speed or unsupported-rise penalty.* Tao et al. measured that penalties alone do
  not suppress ballistic motion ("adding an energy cost without the strong-to-weak curriculum
  has minimal effect"; joint-velocity penalties "either have negligible effects or lead to
  training instability"). The anti-ballistic guarantee here comes from the plant's torque
  ceilings, the shove inside the hold, and the measured dumb-controller floor (§6.3) — not from
  a weight.
- *An alive bonus.* With no early termination it is a constant, hence a no-op.
- *A `dof_pos_limits` penalty.* A get-up genuinely needs joints near their limits, and the
  shipped implementation is index-misaligned (§0.2).
- *A symmetry reward term.* E03: zero for any policy ignoring its input; PPO found a 61 cm
  two-footed brace scoring 0.91. Use mirror augmentation, which is preserved (§7.1).
- *A joint-pose/naturalness term beyond `posture`.* That is what the AMP path exists for.

---

## 5. Fallen initial states, concretely

An offline pose bank, flat and index-addressable exactly like `motion/library.py`, sampled from
`GetUpTask.reset_pose()`. Runtime cost zero. The alternative — letting the walker fall inside the
episode — was measured at a median of 1107 control steps (8.9 s) to first fall, and a scripted
shove plus settle costs 232–268 steps, i.e. 9–11 % of an episode, all of which would have to be
masked out of the reward.

### 5.1 The two traps

**The distribution is nearly inverted between the obvious generator and reality.** Measured
previously on this model (`docs/research/generating-fallen-initial-states.md`, `best.pt` iter 3100):

| source | supine | prone | **side** | mean root height |
|---|---|---|---|---|
| actual policy falls (n = 29) | 3 % | 21 % | **76 %** | 0.258 m |
| free drop, random orientation | 46 % | 51 % | **4 %** | 0.098 m |

Real falls land sideways three times out of four and rest half-propped; drops land flat.
Training on the drop distribution trains mostly on the case that does not happen.

**The PD target held while falling silently controls all diversity.** Holding the *nominal*
pose during the fall gives a joint-angle spread of 0.012 rad (0.5 % of range) — one rigid
mannequin at different yaws, a rank-1 bank that nothing downstream would flag. Holding the
*spawn* pose gives 0.278 rad (13.3 %).

### 5.2 The four generators

`scripts/generate_fallen_poses.py` → `data/fallen/bank_v1.npz`, K = 8192 (6144 train, 2048
held-out eval — HumanUP splits initial states train/test and so should we). Measured cost:
36 ms per pose single-threaded, so the whole bank is ~5 minutes on one core.

1. **Policy falls, 40 %.** Roll out `runs/envelope-20260814-100402/checkpoints/best.pt` with
   terminations disabled; harvest 1–3 s after `DOWN` first fires. The only on-distribution
   source. Mirror left/right to double it.
2. **Scripted topple, 35 %.** Standing + joint noise, root impulse 1.5–5 m/s, settle 1.5 s,
   **with the PD held at a random constant target inside the commandable band** — not at nominal
   (see the trap above).
3. **Free fall from height, 15 %.** Random orientation, spawn-pose hold. Covers the supine/prone
   tails that toppling under-produces.
4. **Hand-authored keyframes, 10 %.** Seated-legs-out, all-fours, side-lying with a trapped arm,
   prone with hands under the shoulders. **The seated case the user explicitly named appears in
   0 of 600 drop samples and can only get in by hand.** Measured here: it is passively stable
   under PD, settling at root 0.054 m, `g_z = −0.99`, `head_ratio = 0.450`.

Plus **8 % of resets from the standing pose** with joint noise, so the `stand`, `quiet` and
`posture` terms are exercised from iteration 1 rather than only after the policy can already
stand. These episodes are **excluded from the success denominator and from curriculum
promotion** (§6.7) — otherwise a do-nothing policy books a free 8 % success rate.

### 5.3 Build-time validation, reject on failure

- `np.isfinite(qpos).all()`.
- `mj_forward` penetration < 2 mm (achievable ≤ 1.02 mm).
- **Fixed-point check**: 0.25 s holding the pose's own joint angles, root drift < 2 cm and max
  joint drift < 0.05 rad (achievable 0.0 cm / 0.00 rad). This is what makes an offline bank
  valid at all: the engine's reset path is `mj_resetData → write qpos/qvel → mj_forward` and
  never settles, so a settled pose is a true fixed point of it.
- `qvel` zeroed for 90 % of poses; a deliberate ~10 % slice is captured **before** settling with
  momentum retained, so the still-tumbling case is represented.
- Yaw and xy randomised at **build** time, poses stored already rotated and re-centred. Doing it
  at reset means transforming `qvel[0:3]` (world frame — confirmed: `_compute_derived` rotates it
  into the body frame) and `qvel[3:6]` differently, which is a silent-wrong-transform waiting to happen.
- Store per pose: `qpos, qvel, label, difficulty, split`, and meta `{standing_height,
  standing_head_height, body_mass, model_sha1, nq, nv, generator_version}`. The task asserts the
  meta matches the prepared model in `configure_for_model` — this repo already has a run
  directory named `ABANDONED-staleClips-tracking`.

`randomize_model()` touches friction, mass/inertia, com, kp/kd and armature but **never
geometry**, so one bank is valid across the whole 64-entry pool.

### 5.4 The engine constraint a naive implementation gets wrong

`vec_env._has_reset_pose` is a **single global bool** (lines 331, 496). If `reset_pose` returns
anything non-None, *every* resetting environment reads `_reset_qpos_abs[row]`. The 8 % standing
mix therefore cannot be expressed by returning None for some rows — those rows must be filled
with the nominal pose plus noise, taken from `state.task_state["_nominal_qpos"]`, which the
engine publishes at `vec_env.py:207`. Return a fresh copy; the workers read the array during
`_run_phase(RESET)`.

`reset_noise` returns None. Additive noise on top of a settled pose breaks the contact state
that made it valid.

---

## 6. Surviving hacks from the Attack phase

Each was re-measured here unless marked. Countermeasures are already folded into §1–§5; this
section is the audit trail and the list of what to re-run when anything changes.

### 6.1 The splayed lateral brace — BLOCKED by U12

Measured, settled, PD holding a constant target:

| `hip_x` | foot separation | root | head_ratio | torso_upright | foot force | longest run under a **naive** predicate |
|---|---|---|---|---|---|---|
| 0.00 | 0.170 m | 1.00 H | 0.996 | 0.986 | 245.5 N each | 354 steps |
| 0.30 | **0.679 m** | 0.95 H | 0.971 | 0.991 | 245.5 N each | **1498 (the whole rollout)** |
| 0.40 | **0.841 m** | 0.92 H | 0.951 | 0.993 | 245.5 N each | **1498** |
| 0.50 | **0.998 m** | 0.87 H | 0.925 | 0.995 | 245.5 N each | **1498** |

A 1.0 m straddle on a 0.88 m-tall robot passes every orientation, height and load test, and is
*more* passively stable than a normal stance by a factor of 4. It is E03's 61 cm brace and E12's
0.67 m splay, both already in the settled-do-not-retry table, arriving through a third door.
Reachable **today** without the widened band (`hip_x` scale is 0.471 rad).
**With U12 the longest run is 0 steps.**

### 6.2 The sagittal split brace — BLOCKED by U12, invisible to `_stance_width`

| `hip_y` | planar separation | **`_stance_width` reports** | root | head_ratio | torso_upright |
|---|---|---|---|---|---|
| ±0.30 | 0.556 m | 0.148 m | 0.98 H | 0.984 | 0.996 |
| ±0.50 | **0.878 m** | **0.142 m** | 0.91 H | 0.945 | 0.995 |
| ±0.70 | **1.194 m** | 0.146 m | 0.78 H | 0.873 | 0.995 |

Every candidate design reused `locomotion._stance_width` as its anti-brace clause. It measures
the lateral component only and reports a metre-wide lunge as a 14 cm stance.
**With the planar bound, the longest run is 0 steps.**

### 6.3 The ballistic snap — plant-blocked, and re-measured under the widened band

The user's requirement "it must not be able to snap upright by flicking one joint" is enforced
by the **plant**, not by a reward term: `jnt_actfrcrange` clamps at 50–200 N·m and the shipped
8 Hz action filter applies to every consumer (training, eval, play, render). Neither can be
traded off against a reward.

Measured floor, 8 settled fallen poses × 5 s each, **with the §0.2 reset fix in place**:

| controller | n | ever satisfied U | held ≥ 250 | peak root height reached |
|---|---|---|---|---|
| zero action, shipped band | 8 | **0** | 0 | 0.24 m |
| single-joint flick (28 × ±1), shipped band | 448 | **0** | 0 | 0.32 m |
| random constant full-body target, shipped band | 960 | **0** | 0 | 0.56 m |
| zero action, **full_range** band | 8 | **0** | 0 | 0.24 m |
| single-joint flick, **full_range** band | 448 | **0** | 0 | 0.68 m (0.77 H) |
| random constant target, **full_range** band | 960 | **0** | 0 | **0.81 m (0.93 H)** |

2832 rollouts, zero satisfied `U` for even one step. Note the last row: a constant target can
vault the pelvis to 93 % of standing height and still fail every predicate. **So the dumb-
controller floor is 0.0 % and any non-zero success number is real behaviour.** This is the gate
to re-run whenever `action_scale_mode`, the PD gains, the action filter or U changes.

A strong-to-weak torque curriculum (Tao et al.) is therefore **not** adopted: it pays for a
problem this plant does not have, and shrinking the action band directly undoes §0.1.

### 6.4 Faking foot load in mid-air — BLOCKED by U9

The foot touch sensors are `mjSENS_TOUCH` on a box site enclosing the foot geom with a 1 cm
margin, so they report **any** contact inside that volume, including self-contact. Measured:
4000 random full-range joint configurations at root z = 2.0 m, entirely airborne, gave a peak
foot touch reading of **6481 N** — 13 body weights, with no floor anywhere. Any reward term or
predicate clause gated on `foot_force` or `foot_contact` without a height gate is exploitable.

U9 (`fz.max() ≤ 0.10 m`; a foot flat on the floor reads 0.027 m) closes it, and U13 plus the
hold duration close it again — an airborne body cannot keep `‖qvel[0:3]‖ ≤ 0.40` for 250 steps.

**Honest limitation:** U9 is a world height and is therefore correct on flat ground only. On the
Phase-4 heightfield it must become a clearance (`fz − terrain_height`), or the touch sensors must
be replaced by a `data.contact` filter on `floor_geom_id`. This is stated rather than solved.

### 6.5 The mannequin hold — the countermeasure requires an engine change

See §3.1 for the numbers. **There is no hold duration that fixes this**, and this is the single
strongest finding in the spec. The countermeasure is the scripted shove inside the hold window,
which needs ~10 lines in `vec_env._select_pushes` (§7.2). If that change is rejected, the
honest position is that the hold measures duration and not balance, and it must be reported
that way. Do not substitute a longer hold and claim the property.

### 6.6 Timing the stand to the truncation buzzer — PARTIALLY mitigated

`done` is always truncation at a fixed step, so the scored window is a learnable constant. A
policy that can hold 2 s but not 8 s can rise at step 1000 and score a success.

Mitigations: `episode_step` is deliberately **not** in the task observation; success requires
`held_ever` **and** standing at the end; and the per-step `stand` reward makes a late rise cost
≈2000 return on a ≈2500 episode, so it is not the return-maximising behaviour and `best.pt` is
selected on return. `time_to_first_stand` is logged as a distribution.

**Not fully closed.** Fully closing it means randomising `max_episode_steps` per environment,
which is currently a scalar on the engine. Recorded as a known residual, not waved away.

### 6.7 Free successes inflating the metric and the curriculum — BLOCKED by masking

Standing-start resets (8 %) satisfy `U` at t = 0. A zero-action policy books them as successes
and, if promotion is driven by success, ratchets the curriculum upward for doing nothing —
structurally the same defect as "the seed that falls 91 % of the time has the highest
torso_upright because resets put it back upright".

Countermeasure: the reset class is stored per environment; `success_batch` returns
`held_ever & standing & (start_class != STANDING)`, and the curriculum promotes only on episodes
whose start came from the curriculum. The Oracle invariant is therefore **"no pose in the
curriculum-eligible split satisfies U at t = 0"**, not "no pose in the bank" — the latter is
contradicted by the standing slice by construction and would fail at build time as written in
two of the three candidate designs.

### 6.8 Metric plumbing — BLOCKED, but only by discipline

- `train.py` passes the **same Task instance** to the training, eval and render environments, so
  nothing per-environment may live on `self`. (`LocomotionTask.init_state` rebinds `self._rng`,
  so the render env's generator already replaces the trainer's.) All state goes in `task_state`;
  the task keeps a generator only for `on_batch_end`, and that is documented as shared.
- `success_batch` receives no argument telling it which environment is calling, so it must
  **always** use the fixed `hold_seconds`. Nothing about success may ever be curriculum-scaled.
- `evaluate()` averages `eval_metrics` over **every step** (`evaluate.py:121-122`). A latched
  per-episode quantity put through it becomes a time fraction: "held the last 250 of 1250 steps"
  reports 0.20, not 1.0. `eval_metrics` returns **instantaneous batch means only**; per-episode
  outcomes go through `success_batch`, which the evaluator reads once per environment.
- `eval/fall_rate` is **structurally 0.0** for this task (nothing terminates) and
  `eval/episode_length` is a constant. Abort rules 1, 2, 4, 6 and 7 in
  `scripts/watch_abort_rules.py` therefore cannot fire for a get-up run. New rules must be keyed
  on quantities that vary: `stand_fraction`, `max_hold_seconds`, per-class success, `sep`.

### 6.9 Hacks with NO good countermeasure, stated plainly

1. **The noise floor.** E22b measured 7.6× outcome spread between byte-identical configs on this
   machine. A single run cannot validate this design, and no amount of spec quality changes that.
   Any claim that get-up "works" needs multiple seeds or a within-run paired measurement.
   Compute the minimum detectable effect before launching anything (E24's adopted rule).
2. **U9 on non-flat terrain** (§6.4). Stated, not solved.
3. **A policy that succeeds only on the passively-stable third of the model pool.** The shove
   removes the free pass (0.7 m/s → 0 % coast), but nothing logs success against the pool entry.
   Logging `success` broken down by pool index would fix it and is not specified here because
   `_env_model` is engine-private.
4. **Whether the hand-authored seated keyframes are realistic.** They are the only source of the
   case the user named, and there is no validation for them other than a human watching a video.
5. **Buzzer timing** (§6.6), partially mitigated only.

---

## 7. Files, changes, and Oracle invariants

### 7.1 New files

| path | contents |
|---|---|
| `humanoid_rl/tasks/getup.py` | `GetUpConfig` (every number in this document) and `GetUpTask(Task)` |
| `humanoid_rl/motion/fallen_bank.py` | `FallenPoseBank`: load / validate / tiered sample, shaped like `motion/library.py` |
| `scripts/generate_fallen_poses.py` | the four generators of §5.2 plus the validation of §5.3 → `data/fallen/bank_v1.npz` |
| `configs/getup.yaml` | see below |
| `scripts/getup_report.py` | stratified per-class evaluation |

`GetUpTask` against the real ABC in `tasks/base.py`:

- `reward_term_names` — the 7 names of §4.
- `task_obs_dim = 1` — a single slot, `float(U)`. It is the regime indicator that gates three
  reward terms, so making it observable lets the critic represent the discontinuity exactly
  instead of inferring a 13-conjunct boundary from raw state. It carries privileged absolute
  height (the same class as E09's still-open asymmetric-actor-critic item; acceptable in sim,
  noted for hardware). `hold_steps` and `episode_step` are deliberately **excluded** (§3.2, §6.6).
- `mirror_task_obs` — `return task_obs.clone()`. The slot is mirror-invariant. Returning None
  would silently disable mirror augmentation for a perfectly bilaterally symmetric task.
- `init_state` — allocate `hold_steps, held_ever, standing, push_trigger, push_now,
  push_request (N,3), prev_prev_action (N,nu), start_class, bank_index, difficulty`.
- `reset_batch` — record the outcome, promote/demote, clear the episode-scoped arrays, choose
  the next bank index. Never read `episode_step`.
- `reset_pose` — `bank.qpos[idx].copy(), bank.qvel[idx].copy()`, every row filled (§5.4).
- `reset_noise` — None.
- `reward_batch` — compute `U` once, update the counters, fill 7 terms, return the sum unclipped,
  roll `prev_prev_action` last.
- `terminated_batch` — `~np.isfinite(state.qpos).all(axis=1)` **only**. Falling never ends an
  episode: VIGOR does the same deliberately, and it makes E16's selection bias structurally
  impossible because every episode is exactly `max_episode_steps` long. (E16 inverted also
  matters here: with fallen starts the *easy* poses would otherwise finish first.)
- `success_batch` — §2.3, masked by start class.
- `on_batch_end` — draw the shove direction into `push_request` for rows where `push_now`, update
  the curriculum, return per-class occupancy.
- `action_offset` — None. Zero action stays the nominal standing pose; expressiveness comes from
  the scale (§0.1), which keeps standing reachable at `a = 0`.
- `configure_for_model(standing_height)` / `set_joint_limits(lo, hi)` — both already called by
  the engine via `hasattr`. Assert the bank's meta matches.
- `eval_metrics` — instantaneous batch means only: `stand_fraction`, `head_ratio`, `pelvis_up`,
  `root_height`, `foot_load_frac`, `foot_load_ratio`, `sep_planar`, `knee_max`,
  **`lean_fore` and `lean_side` (signed)**, `torso_upright` (never alone), `tau_ratio_max`
  (unclipped), and per-class occupancy shares using `LocomotionTask._per_direction_metrics`'
  sums-over-envs / num_envs convention so a momentarily empty class cannot divide by zero.

### 7.2 Changes to existing files

- **`envs/model_prep.py`** — `prepare(..., action_scale_mode="fraction")`; `"full_range"` computes
  `max(nominal−lo, hi−nominal)` for limited joints. Add `PreparedModel.actuator_qpos_adr`.
- **`envs/vec_env.py`**, three changes:
  1. The reset-ctrl fix of §0.2, guarded on `_has_reset_pose` so it is a no-op for locomotion.
  2. Publish `BatchState.torso_zaxis` as (N, 3): `s.torso_zaxis[idx] = sens[:, tz:tz+3]`, and
     keep `torso_upright` as its `[:, 2]` view. Three lines. Not optional — it is the only signed
     measurement of trunk lean, and its absence is what cost the E23 analysis cycle.
  3. Task-requested shoves in `_select_pushes()`:
     ```python
     req = self.state.task_state.get("push_request")
     if req is not None:
         ridx = np.flatnonzero(self.state.task_state["push_now"])
         if ridx.size:
             self._push_vel[ridx] = req[ridx]
             idx = np.union1d(idx, ridx)      # union1d returns SORTED, which _do_resets needs
     ```
     `on_batch_end` (line 580) runs before `_select_pushes()` (line 595), so the ordering is correct.
- **`config.py`** — add `getup: GetUpConfig`; add `EnvConfig.action_scale_mode` (currently a
  `prepare()` kwarg with no config route) and `EnvConfig.action_scale_fraction`. `_from_dict`
  raises on unknown keys, so a mistyped section cannot silently no-op (the E11 failure mode).
- **`train.py::_build_task`** — add the `kind == "getup"` branch (call `prepare`, load the bank,
  construct the task) and extend the "unknown run.task" message. Extract `_build_task` to a
  module-level `build_task(config)`.
- **`render.py:202`, `viz/skeleton.py:184`, `scripts/{play,gait_report,posture_score,render,
  capture_skeletons,smoke_env}.py`** — all write `task_state["command"][0]` or construct
  `LocomotionTask` directly. `train.py` catches every exception around video and skeleton
  capture, so a get-up run would silently produce **no videos and no skeletons for the entire
  run**. Guard with `if "command" in env.state.task_state:` and call `build_task(config)`.
- **`oracle/invariants.py`** — `_sample_commands` builds a `LocomotionTask` probe
  unconditionally; skip the command checks for `run.task == "getup"` and add §7.4.

### 7.3 `configs/getup.yaml`

```yaml
run:   {task: getup, algo: ppo, name: getup}
env:   {max_episode_steps: 1250, action_scale_mode: full_range}   # 10.0 s at 125 Hz
ppo:   {gamma: 0.996, symmetry_augment: true}
network: {init_noise_std: 0.5}
eval:  {num_envs: 64, num_episodes: 64}
getup:
  hold_seconds: 2.0
  hold_push_vel: 0.6            # m/s, random direction, inside every hold attempt
  hold_push_window: [40, 140]   # steps after the counter starts
  stand_root_frac: 0.85
  stand_head_ratio: 0.90
  stand_pelvis_up: 0.93
  stand_torso_up: 0.85
  stand_lean_max: 0.35
  stand_load_sum_frac: 0.60
  stand_load_min_frac: 0.20
  stand_foot_z_max: 0.10
  stand_hand_z_min: 0.40
  stand_knee_max: 0.60
  stand_sep_min: 0.08
  stand_sep_max: 0.35
  stand_lin_vel_max: 0.40
  stand_ang_vel_max: 1.50
  w_upright: 0.5
  w_rise: 1.0
  rise_kappa: 3.0
  w_stand: 2.0
  w_quiet: 0.5
  w_posture: 0.5
  w_effort: -0.25
  w_smooth: -0.10
  standing_reset_prob: 0.08
  difficulty_init: 0.35
  difficulty_min: 0.30
  difficulty_step: 0.05         # SYMMETRIC promote/demote (E20)
```

`max_episode_steps: 1250`, not 2500: a rise takes 2–4 s, so a 20 s episode spends most of its
samples on the already-solved standing phase. 10 s is HoST's horizon and leaves 8 s of rise
budget against a 2 s hold. `gamma: 0.996` because 0.99 at 125 Hz is a 0.8 s effective horizon
while the standing income arrives 2–4 s after the push-off that earns it; 0.99 at the references'
50 Hz is `0.99^(50/125) = 0.996` here. This is the same rate-conversion trap as E18.

**Curriculum: one, over initial states only. Nothing in the reward or the success test ever
changes.** Per-environment level in [0.30, 1.0], init 0.35, sampling from bank entries whose
`difficulty ≤ level`, so easy states never leave the distribution. Class difficulties: standing
0.0, seated 0.25, low 0.35, side 0.5, supine 0.7, prone 0.85, tumbling 1.0 — ordered by measured
distance from standing, **not** by root height (seated rests at 0.054 m and is easy; supine rests
at 0.085 m and is hard). Promotion ±0.05, **symmetric**, on the strict success only, excluding
standing starts. E20: at +0.05/−0.10 every environment slid to the floor within 150 iterations
and stayed there for 500 more, and the curriculum "was working exactly as specified".
Log the **histogram** of levels, not the median — a median hides a collapse.

Explicitly rejected, with reasons recorded so they are not retried: a hold-duration curriculum
(it re-opens the flash hack during the exact period when the policy forms its strategy, and
HumanUP reports Stage-I behaviours discovered under loose constraints are often incompatible
with tighter ones); a torque or action-bound curriculum (§6.3); HoST's 200 N assist force
(powerful, but it is a knob that can silently stay on — held as a **pre-registered contingency**,
enabled only if after 500 iterations no environment in the hardest difficulty tercile has ever
completed a hold, and that trigger must be stated in **reset counts**, not iterations, so
"never dealt" is distinguishable from "dealt and failed").

### 7.4 Oracle invariants, before any compute

| # | check | severity if it fires |
|---|---|---|
| 1 | `hold_seconds / dt` is an integer and `max_episode_steps ≥ 3 × hold_steps` | CONTRADICTION (E18) |
| 2 | `action_scale_mode == "full_range"`, quoting the measured 37.5 % knee coverage otherwise | UNREACHABLE |
| 3 | The `_do_resets` ctrl fix is present and uses `actuator_qpos_adr` — assert `mean\|τ\| < 5 N·m` on the first step from 8 bank poses | CONTRADICTION |
| 4 | `gamma ≥ 0.995` at 125 Hz, quoting `0.99^375 = 0.023` | CONTRADICTION |
| 5 | The bank's meta matches the prepared model (standing height, mass, model hash) | CONTRADICTION |
| 6 | No pose in the **curriculum-eligible** split satisfies `U` at t = 0 | CONTRADICTION |
| 7 | Class labels partition the bank: shares sum to 1.000, no unlabelled pose | CONTRADICTION |
| 8 | The bank's class mix matches the policy-fall taxonomy within tolerance | SUSPECT |
| 9 | Dumb-controller floor: zero-action and single-joint-flick rollouts from 32 bank poses never satisfy `U` (§6.3 measured 0 of 2832) | CONTRADICTION |
| 10 | Passive-coast check: a mannequin rollout with the shove enabled never completes a hold (§3.1 measured 0 % at 0.7 m/s) | CONTRADICTION |
| 11 | The settled nominal stand satisfies all 13 conjuncts with ≥ 10 % margin on each | CONTRADICTION |
| 12 | The pose table of §1.1 is re-derived and every canonical pose gets its documented label | CONTRADICTION |
| 13 | `U7` threshold survives mass randomisation: `0.60 < 0.85 × mass_scale_min` | CONTRADICTION |
| 14 | `torso_upright` never appears in `eval_metrics` without `lean_fore`/`lean_side` | SUSPECT |
| 15 | Note, always: `eval/fall_rate` and `eval/episode_length` are structurally constant for this task | OK |

---

## 8. Honest assessment

**What is solid.** The predicates and their thresholds are measured, not guessed; every conjunct
has a named pose it rejects and a margin against the settled stand. The dumb-controller floor is
**0 of 2832**, so any non-zero success number will be real behaviour, which is more than could be
said of most metrics in this project's history. The reward is 7 bounded terms instead of 21
unnormalised ones, the three that could be farmed on the floor are gated on the same predicate,
and the worst-case penalty is 0.35/step against a 1.5/step positive gradient — E09's sizing rule
with a 4× margin. The two engine defects (§0.1, §0.2) are real, cheap to fix and benefit the
tracking task too.

**What is genuinely hard, in order.**

1. **Sparsity across the dead band.** Every intermediate pose of a real get-up — kneel, all-fours,
   deep squat — sits between `DOWN` and `U`. In that band `stand`, `quiet` and `posture` are all
   zero and only `upright` and `rise` pay. `rise` is convex, which is deliberate (it is what stops
   the policy parking in a kneel), but convexity also means the payoff for the *first* half of the
   climb is small. This is the field's stated core difficulty: HumanUP names reward sparsity as
   the property separating get-up from locomotion, and notes that a correct get-up must *lower*
   the head to fold before raising it. If training stalls, the first thing to change is
   `rise_kappa`, and the second is the class curriculum's starting mix — not a new term.
2. **The hold requires standing balance, which nothing in this project has yet demonstrated.**
   The best walker falls 15.6 % of the time in 7.4 s *while walking*, with 0.7 m/s shoves. A get-up
   policy must stand still through a 0.6 m/s shove for 2 s. If that turns out to be the binding
   constraint rather than the rise, `hold_push_vel` is the knob, and the diagnostic that
   distinguishes the two cases is `stand_fraction` from **standing-start** episodes: a policy that
   cannot hold from a standing reset cannot possibly hold after a rise.
3. **Measurement.** E22b's 7.6× noise floor means one run proves nothing. Budget seeds.

**What would make this fail, ranked by probability.**

- *The dead band is never crossed* and the policy converges to lying still with `upright ≈ 0.5`.
  Signature: `stand_fraction ≈ 0` with a healthy `head_ratio` distribution capped around 0.5–0.7.
- *The shove makes the hold unreachable*, so `held_ever` is 0 for the whole run and the task looks
  unlearnable when only the last 2 s of it are. Signature: `stand_fraction` rises steadily while
  `max_hold_seconds` sits below 0.5 s. Mitigation is a magnitude ramp on `hold_push_vel`, which is
  a curriculum on the success criterion and therefore carries the exact hazard §7.3 rejects
  elsewhere — if it is used, the reported metric must stay pinned at 0.6 m/s.
- *Supine/prone interference.* HoST measured that training both together "negatively impacted
  performance due to interference between sampled rollouts", and HumanUP trains them separately
  with a distinct roll-over policy. Do not pre-emptively split: the stratified per-class eval
  measures it directly. Split only if the per-class numbers diverge.
- *Single-critic overload.* HoST reports **zero** success on every terrain without their
  multi-critic, attributed to reward groups spanning orders of magnitude. This repo has one critic.
  Mitigation taken: 7 terms, all inside one order of magnitude, per-term logging already in place.
  E22c measured this repo's critic fitting cleanly and unsaturated, so the risk is real but not
  evidenced here. Multi-critic is the first fallback if training stalls with a healthy `rise` and
  no `stand`.
- *An unnoticed term interaction.* This is the project's historical failure mode and the reason
  the design is 7 terms rather than 14. The residual exposure is the `U` gate shared by terms 3,
  4 and 5: anything wrong with `U` is wrong with 3.0 of the 3.5 available reward at once. That is
  a deliberate trade — a single, heavily-tested, conjunctive boundary beats three independently
  tunable ones — but it means `U` is the object to attack first if behaviour looks strange.

**Before launch, write the prediction down** (logbook rule 1) and compute the minimum detectable
effect (E24's rule). The pre-registered predictions for run 1: `stand_fraction > 0.05` by
iteration 800 from side-lying starts; per-class success ordered seated > low > side > prone >
supine; `sep_planar` p95 below 0.35 by construction; `time_to_fall_frac` below 0.30 by iteration
1500. If `stand_fraction` is exactly 0.0 at iteration 800, the dead band is the problem and the
run should be stopped rather than extended.
