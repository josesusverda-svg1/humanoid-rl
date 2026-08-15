# Get-up task: implementable spec, built around the HOLD requirement

Everything numeric below was **measured on this model** through `envs/model_prep.prepare()`
during the writing of this document, not taken from the literature or from earlier docs.
Where an earlier doc's number failed to reproduce, that is called out.

Model constants (measured): `standing_height 0.87686 m`, `standing_head_height 1.51391 m`
(the head **FRAMEPOS sensor on a BODY returns `xipos`, the body's centre of mass**, not the
joint frame — this is why the sensor reads 1.514 while `data.xpos[head]` reads 1.339),
mass 50.05 kg → body weight 491.0 N, `contact_force_threshold 9.82 N`, control step
`dt = 0.008 s` (125 Hz), `terminate_height = 0.62 * H = 0.5437 m`.

---

## 0. Five measured facts that shape every decision here

### 0.1 `gravity_body` sign convention, verified

`gravity_body = R_pelvis^T · (0,0,-1)`.

| pelvis attitude | gravity_body | meaning |
|---|---|---|
| upright | `( 0.00,  0.00, -1.00)` | standing |
| pitch **+90°** about local y | `(+1.00, 0, 0)` | **PRONE, face down** |
| pitch **−90°** | `(−1.00, 0, 0)` | **SUPINE, face up** |
| roll **−90°** about local x | `( 0, +1.00, 0)` | lying on its **LEFT** side |
| roll **+90°** | `( 0, −1.00, 0)` | lying on its **RIGHT** side |

`left_foot` sits at `+y` (`+0.085`), `right_foot` at `−y`, confirming local `+y` is left.
So `gravity_body[0]` separates supine from prone and `gravity_body[1]` separates left from
right. **`gravity_body` is a signed 3-vector and carries every sign `torso_upright` lacks.**

### 0.2 The sitting case defeats *both* uprightness signals

Settled poses, PD holding each pose, all measured:

| pose | root_h | g_x | g_y | g_z | torso_upright | head_ratio | foot COM z | foot force (L,R) N |
|---|---|---|---|---|---|---|---|---|
| standing nominal | 0.874 | −0.06 | 0.00 | −1.00 | **+0.986** | **0.996** | 0.027 | 245.5, 245.5 |
| standing tiptoe | 0.848 | +0.28 | 0.00 | −0.96 | +0.986 | 0.964 | 0.048 | 206.7, 206.7 |
| **seated, legs out** | **0.053** | −0.15 | 0.00 | **−0.99** | **+0.957** | **0.447** | 0.092 | 0.0, 0.0 |
| kneeling | 0.286 | −1.00 | 0.00 | −0.03 | −0.078 | 0.197 | 0.092 | **234.6, 234.6** |
| supine | 0.085 | −1.00 | 0.00 | −0.07 | −0.033 | 0.078 | 0.089 | 46.4, 46.4 |
| prone | 0.103 | +1.00 | 0.00 | −0.01 | +0.128 | 0.069 | 0.089 | 65.0, 65.0 |
| side (left) | 0.181 | +0.01 | +0.99 | −0.11 | +0.092 | 0.153 | — | 101.8, 13.8 |
| supine bridge | 0.482 | −0.95 | 0.00 | **+0.33** | −0.854 | 0.062 | 0.027 | 111.6, 111.6 |
| all fours | 0.459 | +1.00 | 0.00 | +0.08 | −0.785 | 0.062 | 0.128 | 0.0, 0.0 |

**Sitting on its backside reads `torso_upright = 0.957` and `gravity_body[2] = −0.99`.**
Both orientation signals say "upright". Only `root_height` (0.053) and
`head_height_ratio` (0.447) say otherwise. Any success predicate built on orientation is
wrong on this body; the predicate must be conjunctive and must include a **height**.

### 0.3 `foot_contact` is TRUE for a corpse

The threshold is 9.82 N (2% of body weight). Supine reads 46.4 N per foot, prone 65.0 N,
side 101.8 N, kneeling **234.6 N** — 96% of body weight through the feet while kneeling.
HumanUP's published "standing on feet" indicator `1((‖F_feet‖ > 0) & (h_feet < 0.2))`
evaluates **TRUE for supine, prone, side-lying and kneeling** on this humanoid. It must be
replaced by a **load fraction** `ΣF / 491 N` with a real threshold, and even that does not
exclude kneeling (0.96) — only head height does (0.197).

### 0.4 `head_height_ratio` is the right primary signal, precisely because it has no sign

Feet flat on the floor, waist folded by `abdomen_y`, settled:

| abdomen_y (rad) | torso_upright | head_ratio |
|---|---|---|
| 0.000 | +0.992 | 0.997 |
| −0.500 | +0.753 | 0.945 |
| **−0.755** (the E23 lean) | **+0.531** | **0.888** |
| −1.000 | +0.282 | 0.823 |

Kinematic sweep of both signs (offset-corrected): `abdomen_y = ±0.755` both give head_ratio
**0.927**, while `torso_upright` gives 0.654 backward vs 0.795 forward — asymmetric, because
the torso z-axis already carries `x = −0.102` in the nominal pose. A height is sign-free by
construction: **the sign problem cannot recur in a quantity that has no sign.**

Crouch depth (offset-corrected, feet flat): knee 0.9 → head_ratio 0.946, root 0.798;
knee 1.2 → 0.905 / 0.735; knee 1.6 → 0.833 / 0.627; knee 2.0 → 0.747 / 0.497.

### 0.5 The humanoid has **no torque limit at all**, and one joint can deliver 785 N·m

`actuator_forcelimited` is 0 for all 28 actuators and `actuator_forcerange` is `(0,0)` —
MuJoCo therefore applies **no ceiling**. Measured peak `|actuator_force|`:

| condition | peak | mean |
|---|---|---|
| holding the nominal standing pose | **64.9 N·m** (knee) | 9.7 N·m |
| supine, single-joint slam `abdomen_y` at full commandable offset (0.79 rad, kp 1000) | **785 N·m** | — |
| supine, single-joint slam `right_hip_y` (1.05 rad, kp 500) | **528 N·m** | — |
| supine, single-joint slam `right_knee` (0.84 rad, kp 500) | **430 N·m** | — |

One joint can produce **12× the torque the whole body needs to stand**. The only reason a
single-joint flick does not currently stand this humanoid is *geometry* (the reachable pose
is not a standing pose), not authority. §5 widens the reachable pose band, which removes
that accidental protection. **Widening the action band without adding a force limit
manufactures the exact hack the user asked to prevent.**

---

## 1. Fallen / sitting detection, from keypoints

All fields verified present in `BatchState`. Derived once per step, vectorised:

```python
gz  = state.gravity_body[:, 2]        # signed: -1 upright, 0 horizontal, +1 inverted
gx  = state.gravity_body[:, 0]        # signed: >0 face-down, <0 face-up
gy  = state.gravity_body[:, 1]        # signed: >0 on LEFT side, <0 on RIGHT side
hr  = state.head_height_ratio         # head COM z / 1.51391, sign-free
h   = state.root_height
load = state.foot_force.sum(1) / 491.0            # body-weight fractions, NOT foot_contact
foot_z = state.key_body_pos[:, :2, 2]             # [left_foot, right_foot] COM z
hand_z = state.key_body_pos[:, 2:4, 2]            # [left_hand, right_hand] COM z
foot_mid = 0.5 * (state.key_body_pos[:, 0, :2] + state.key_body_pos[:, 1, :2])
base_offset = np.linalg.norm(state.root_pos[:, :2] - foot_mid, axis=1)
```

```python
DOWN    = (hr < 0.55) | (h < 0.45)                      # head OR pelvis below any stance
FLAT    = DOWN & (gz > -0.5)                            # trunk axis near horizontal
SUPINE  = FLAT & (gx < -0.6)
PRONE   = FLAT & (gx > +0.6)
SIDE    = FLAT & (np.abs(gy) > 0.6)
SITTING = DOWN & (gz <= -0.7) & (np.abs(gx) < 0.6) & (np.abs(gy) < 0.6)
OTHER   = DOWN & ~(SUPINE | PRONE | SIDE | SITTING)     # kneel, all-fours, bridge, tumbling
```

Verified against every row of the §0.2 table. `SITTING` fires on the seated pose
(hr 0.447, gz −0.99) which is exactly the user's "sitting on its backside", and is the one
case that no orientation-only detector can see. `DOWN` is False for the E23 lean
(hr 0.823, h 0.871), so a bad posture is correctly not a fall. A deep squat
(knee 2.0, hr 0.747) is also not DOWN, which matters because a squat is a *stage of the
get-up*, not a failure.

`OTHER` is deliberately large. Kneeling, all-fours and mid-tumble are all "not fallen and
not standing", and both the reward and the class curriculum need that third state.

---

## 2. Success: upright, in a normal human pose, and HELD

Hysteretic (Schmitt-trigger) conjunction. Every conjunct is necessary — §7 names the pose
that defeats each one alone.

| conjunct | ENTER | HOLD | what it blocks alone |
|---|---|---|---|
| `head_height_ratio` ≥ | **0.90** | 0.86 | sitting (0.447), kneeling (0.197), waist fold in **either** direction |
| `root_height` ≥ | **0.78 m** (0.89·H) | 0.74 | sitting (0.053), bridge (0.482), all-fours (0.459) |
| `gravity_body[2]` ≤ | **−0.90** | −0.85 | bridge (+0.33), all-fours (+0.08), any inversion |
| `abs(gravity_body[0])` ≤ | **0.35** | 0.42 | signed pitch, forward and backward checked separately |
| `abs(gravity_body[1])` ≤ | **0.35** | 0.42 | signed roll, left and right checked separately |
| `torso_upright` ≥ | 0.85 | 0.80 | redundant conjunct; **never used alone, never reported alone** |
| `load = ΣF/491` ≥ | **0.80** | 0.65 | airborne, hanging, propped |
| `min(F_left, F_right)/491` ≥ | **0.20** | 0.12 | one-legged stand, weight all on one side |
| `max(foot_z)` ≤ | **0.10 m** | 0.13 | a foot in the air (standing 0.027, all-fours 0.128) |
| `base_offset` ≤ | **0.25 m** | 0.32 | feet not under the body (standing 0.088, fallen 0.85) |
| `min(hand_z)` ≥ | **0.35 m** | 0.30 | leaning on a hand (standing 0.83, prone 0.04) |
| `‖lin_vel_body‖` ≤ | **0.40 m/s** | 0.55 | passing through upright mid-flight |
| `‖ang_vel_body‖` ≤ | **1.50 rad/s** | 2.00 | tumbling through vertical |
| `abs(qpos[8])` (abdomen_y) ≤ | **0.60** | 0.75 | the E23 fold; **signed**, both directions |
| `abs(qpos[7])` (abdomen_x) ≤ | **0.50** | 0.62 | lateral fold; signed |
| `mean(qpos[24], qpos[31])` (knees) ≤ | **1.00 rad** | 1.20 | a permanent crouch |
| `mean(abs(qpos[22]), abs(qpos[29]))` (hip_y) ≤ | **0.80 rad** | 0.95 | jack-knifed hips |

**The sign problem is handled three ways, not one.**
1. The primary posture signal is `head_height_ratio`, **a height**, which has no sign and
   which drops identically for a forward and a backward fold (measured: 0.927 at
   `abdomen_y = ±0.755`).
2. The two angular pelvis conjuncts are the **individual signed components** `gravity_body[0]`
   and `gravity_body[1]`, bounded separately by absolute value, so a forward fold and a
   backward fold are separate failures and the direction is recoverable.
3. `torso_upright` appears only as a redundant conjunct and is never reported without the
   signed `gravity_body[0]` and `qpos[8]` (`abdomen_y`) beside it in `eval_metrics`.

**`qpos[7:]` is NOT in actuator order.** Measured: actuator order is
`… right_hip_x, right_hip_z, right_hip_y, right_knee …` but the qpos addresses are
`right_hip_x → 21, right_hip_y → 22, right_hip_z → 23`. Four joints (hip_y/hip_z on both
legs) are transposed. Indexing `qpos[:, 7:7+nu]` with actuator indices — which
`locomotion.py:793-796` does for `dof_pos_limits`, and `tracking.py` does for pose error —
silently applies each leg's hip_z limits to hip_y and vice versa. `GetUpTask` must build an
explicit address array from `prepared.joint_names`, never assume the offset.

---

## 3. The hold: mechanics, bookkeeping, and how it survives autoreset

### 3.1 Exact call order in `vec_env.step()` (read from the source)

```
573  reward   = task.reward_batch(s, terms)        <-- ALL hold bookkeeping happens here
580  metrics  = task.on_batch_end(s, {})
582  terminated = task.terminated_batch(s)
584  truncated  = ~terminated & (episode_step >= max_episode_steps)
587  success    = task.success_batch(s) & done     <-- masked by done
590  final_obs[done], ep_return_out[done], ep_length_out[done]
595  _do_resets(done_idx):
483        s.episode_step[done_idx] = 0            <-- BEFORE reset_batch
486        s.ctrl[done_idx] = _default_joint_pos
447        _ctrl_filtered[done_idx] = _default_joint_pos
489        task.reset_batch(s, done_idx, rng)      <-- clear the latch here
493        task.reset_pose(s, done_idx, rng)
507        _run_phase(RESET)                       <-- workers write qpos/qvel, mj_forward
```

### 3.2 Bookkeeping goes in `reward_batch`, and nowhere else

`reward_batch` is called from exactly one place, `vec_env.py:573`, once per step
(verified by grep across the whole repo — no script or trainer calls it directly). It runs
**before** `on_batch_end`, `terminated_batch` and `success_batch`, so putting the counter
advance there is the only placement in which the reward ramp, the termination check, the
success latch and `eval_metrics` all read the *same* step's value.

- In `on_batch_end` instead → the reward ramp lags the counter by one step.
- In `success_batch` instead → it only runs where `done`, so the counter never advances.

```python
def _advance_hold(self, state) -> np.ndarray:
    ts = state.task_state
    enter = self._standing(state, strict=True)     # ENTER column of §2
    keep  = self._standing(state, strict=False)    # HOLD column of §2
    up = np.where(ts["up_prev"], keep, enter)      # Schmitt trigger
    ts["hold_steps"][:] = np.where(up, ts["hold_steps"] + 1, 0)   # HARD reset, no decay
    ts["up_prev"][:] = up
    newly = up & (ts["hold_steps"] >= ts["hold_required"]) & ~ts["succeeded"]
    ts["succeeded"][newly] = True
    ts["success_step"][newly] = state.episode_step[newly]
    ts["fell_after_success"] |= ts["succeeded"] & ~up
    return up
```

`success_batch` is then a pure read: `return state.task_state["succeeded"]`. The engine
ANDs it with `done`, so the latch is reported exactly once, on the episode's final step,
whatever step the hold actually completed on.

### 3.3 Surviving autoreset

Everything in `task_state` is allocated once in `init_state` and **persists across episodes
by default** — the arrays are never reallocated. The only thing that makes the hold
episode-scoped is `reset_batch` clearing it, for exactly the resetting rows:

```python
def reset_batch(self, state, indices, rng):
    if indices.size == 0: return
    ts = state.task_state
    self._record_outcomes(state, indices)   # curriculum: read the latches BEFORE clearing
    ts["hold_steps"][indices]        = 0
    ts["up_prev"][indices]           = False
    ts["succeeded"][indices]         = False     # forgetting this = success forever after
    ts["success_step"][indices]      = -1
    ts["fell_after_success"][indices]= False
    ts["head_best"][indices]         = 0.0
    ts["base_best"][indices]         = 0.0
    ts["prev_prev_action"][indices]  = 0.0
    ts["hold_required"][indices] = rng.integers(cfg.hold_min_steps, cfg.hold_max_steps + 1,
                                                indices.size)
```

Three ways this breaks silently, all worth an assertion:
1. Not clearing `succeeded` → every episode after the first reports success. Assert
   `not ts["succeeded"][indices].any()` at the end of `reset_batch`.
2. Clearing in `on_batch_end` → runs at line 580, **before** `success_batch` at 587, so the
   latch is wiped before it is read and success is always 0.
3. Advancing `hold_steps` in `on_batch_end` → one-step lag against the ramp.

**Do not read `state.episode_step` inside `reset_batch`.** Line 483 zeroes it *before*
`reset_batch` is called at 489. `LocomotionTask.reset_batch:531` does exactly this
(`lengths = state.episode_step[indices]`), so `episode_len_ema` has been fed zeros for the
whole project and the number on the dashboard is meaningless. It currently gates only the
penalty-leniency curriculum, which is clamped off (`penalty_scale_init = penalty_scale_min
= 1.0`), so nothing downstream is affected — but it is reported in `eval_metrics`. The
get-up curriculum must use `success_step` (latched during `reward_batch`) instead.

### 3.4 The hold requirement itself

`hold_required` is drawn **per episode** uniformly from `[125, 375]` steps = **1.0 to 3.0 s**
at 125 Hz, the user's stated range. It is **not observed by the policy** — a policy that can
see "112 steps to go" will schedule its collapse. The policy observes only its own elapsed
hold, saturating, and never `succeeded`; and because the reward stream (§4) continues
unchanged after the requirement is met, there is no event to schedule in the first place.

Do not ramp `hold_required` over training: it is the denominator of the reward ramp, so
changing it moves the reward scale under the critic mid-run. The per-episode draw already
covers the range.

---

## 4. Reward: the hold *is* the reward, not a bonus on top

Phase gate `up = _advance_hold(state)`, `down = ~up`, `ramp = clip(hold_steps / hold_required, 0, 1)`.

There is **no completion bonus**. A bonus is an event, and an event can be timed, farmed and
re-collected. Instead the standing income is *ramped by the hold*:

| # | term | formula | weight |
|---|---|---|---|
| 0 | `stand` | `up * ramp` | **+4.0** |
| 1 | `quiet` | `up * ramp * exp(-‖v‖²/0.25 - ‖ω‖²/4.0)` | +1.0 |
| 2 | `posture` | `up * ramp * exp(-Σ(q - q_nom)²/4.0)` over all 28 joints | +1.0 |
| 3 | `head_ratchet` | `max(0, hr - head_best)`, then `head_best = max(head_best, hr)` | +30.0 |
| 4 | `base_ratchet` | `max(0, h/H - base_best)`, then update | +15.0 |
| 5 | `feet_under` | `down * exp(-base_offset²/0.25)` | +0.5 |
| 6 | `scoot` | `down * ‖root_lin_vel_xy‖²` | −1.0 |
| 7 | `rise_rate` | `clip(qvel[:,2] - 0.6, 0, None)²` | −4.0 |
| 8 | `torque` | `Σ τ²` | −1e-5 |
| 9 | `dof_vel` | `Σ q̇²` | −1e-4 |
| 10 | `dof_acc` | `Σ ((q̇ - q̇_prev)/dt)²` | −1e-7 |
| 11 | `action_rate` | `Σ (a - a_prev)²` | −0.01 |
| 12 | `action_smooth` | `Σ (a - 2a_prev + a_prev2)²` | −0.01 |
| 13 | `dof_pos_limits` | overflow past 90% of range, **correct qpos addresses** | −5.0 |
| 14 | `hand_prop` | `(h > 0.6) * (min(hand_z) < 0.15)` | −1.0 |

Total = `sum(terms)`, **not** clipped at zero. `LocomotionTask` clips because its positives
dominate once walking; here the whole first phase is legitimately near-zero positive and a
clip at zero would erase the entire penalty gradient during the get-up.

### Why the ramp makes transients worth nothing — the arithmetic

Episode 1250 steps (10 s). Take `hold_required K = 250` (2 s).

- **Real get-up**: stands at t = 4 s, holds for the remaining 750 steps.
  `stand` = `4.0 · (Σ_{k=1..250} k/250 + 500)` = `4.0 · 625.5` = **2502**.
  Plus `quiet + posture` ≈ 1000. Episode total ≈ **3500**.
- **Both ratchets over an entire rise** (head 0.07 → 0.95, base 0.06 → 0.90):
  `30·0.88 + 15·0.84` = **39.8**, i.e. **1.1%** of the stand income. A ballistic dive that
  flashes maximum height and collapses collects at most this.
- **Pop up and fall, five times, 30 steps each**:
  `5 · 4.0 · Σ_{k=1..30} k/250` = **7.4**. Plus the ratchets pay **once** (best-so-far), so
  repeats add nothing there.

**A real hold outearns five pop-and-drops by roughly 470×, and a ballistic flash by 88×.**
The integral over a partial hold of length k is `w·k(k+1)/(2K)` — **quadratic in duration**,
so halving the hold quarters the payout. That is what "transient poses are worth nothing"
means mechanically.

After `k ≥ K` the ramp saturates at 1 and the income continues unchanged for as long as it
stands: no cliff, no discontinuity for the critic, nothing to time. Falling costs the full
rate immediately **and** resets the ramp to 0, so a second get-up in the same episode must
re-climb it and earns a quarter of what continuing to stand would have.

### Terms 3 and 4 are ratchets, not Δ-indicators

HumanUP's `1(h_t > h_{t-1})` pays on every rising half-cycle, so a 2 Hz bounce collects it
half the time for free. Paying only for exceeding the **best so far this episode** is
monotone by construction: total payout equals `w · (final best − initial)` no matter what
path was taken, so oscillation earns exactly zero. `head_best`/`base_best` are cleared in
`reset_batch`.

### Terms 1 and 2 are gated on `up * ramp`, and that gating is load-bearing

An un-gated stillness term is maximised by a **motionless body on the floor** (`v = 0`,
`ω = 0`). This repo has shipped that exact class of bug twice: `gait_symmetry` scored 1.0
for standing still (E02) and the symmetry reward scored 0.91 for a policy ignoring its input
(E03). Both terms here are zero unless `up` and scaled by `ramp`, so a corpse earns nothing.

---

## 5. Reachability and authority — two changes that must ship together

### 5.1 The get-up motion is currently outside the action space

`action_scale_fraction = 0.6`, band = `clip(nominal ± 0.6·half_range, joint_range)`.
Measured commandable fractions: **knee 37.5%** (`[0, 1.05]` of `[0, 2.79]`), elbow 39.4%,
shoulder_x 52.8%, hip_y 60.0%, abdomen_y 60.0%. Folding into a squat, a kneel or a
hands-under-shoulders push is not merely hard, it is **not commandable**.

At `action_scale_fraction = 1.0` (measured): knee `[0, 1.604]` (92° of flexion), hip_y
`[−1.85, 1.05]` (full flexion side), abdomen_y `[−1.05, 1.31]` (full range), elbow
`[0, 1.657]`, ankle_y `[−0.96, 0.855]`, shoulder_x `[−0.73, 2.44]`. That is enough for a
deep squat, a sit-up, and a hands-under-shoulders push. **A full kneel (knee 2.6) remains
uncommandable**; if training stalls in a kneel-adjacent optimum the next lever is a constant
`action_offset` at knee +0.7, which moves the band to `[0, 2.79]`.

### 5.2 …which is exactly why the force limit is now mandatory

Raising the fraction to 1.0 raises the single-joint impulse in proportion: `abdomen_y` goes
from 785 N·m to **1308 N·m**, against 65 N·m to hold the standing pose. With
`actuator_forcelimited = 0` today there is nothing to stop it.

Set, in `model_prep.prepare()`, `actuator_forcelimited = 1` and
`actuator_forcerange = ±β · τ_max` with τ_max grounded in the measurement above (standing
needs ≤ 65 N·m peak, 9.7 N·m mean):

| group | τ_max (N·m) |
|---|---|
| abdomen x/y/z | 200 |
| hip x/y/z | 200 |
| knee | 200 |
| ankle x/y/z | 120 |
| shoulder x/y/z | 80 |
| elbow | 60 |
| neck x/y/z | 20 |

`β` is the strong-to-weak curriculum variable of §6.

---

## 6. Curriculum — three, in priority order

**(a) Strong-to-weak actuation (Tao et al. 2022).** This is the field's measured answer to
"it must not be able to snap upright by flicking one joint", and Tao states directly that
*"adding an energy cost without the strong-to-weak curriculum has minimal effect"* and that
velocity penalties alone are negligible or destabilising. β starts at **2.0** (permissive
enough that a solution is found at all), decays **×0.95 per stage** to a floor of **1.0**,
promoting when the batch success EMA ≥ 0.6 and the stage has run ≥ 100 iterations.
Implemented by a new `ThreadedVecEnv.set_actuator_force_scale(beta)` that writes
`actuator_forcerange` across every entry of `model_pool`; called from `on_batch_end`, which
runs on the main thread between worker barriers, so it is safe against the physics threads.

**(b) Initial-state class curriculum (UniReLo-style, on top of a fixed bank).** Per-class
success EMAs (`seated`, `side`, `supine`, `prone`, `other`, `standing_start`) drive sampling
weights `w_c ∝ (1 − succ_c) + 0.10`, floored at 0.05 so no class ever leaves the
distribution — the repo's own "graduate recycling" pattern from `_maybe_promote`. Start
weighted toward `seated` and `side` (highest initial head height, so the ratchet has a
gradient from step one) and let supine/prone rise on their own.

**(c) `standing_reset_prob = 0.08`.** Eight percent of resets start from the nominal
standing pose with joint noise. Without this, `r_stand` is never sampled early, `ramp` never
leaves 0, and terms 0–2 supply no gradient whatsoever for the first several hundred
iterations. This is the single cheapest thing that makes the hold learnable.

**Explicitly not used:** HoST's 200 N vertical assist force. It requires per-step external
force on the base and its own ablation shows it matters most on platforms and slopes, which
this scene does not have. Hold it as the fallback if (a) stalls.

**Do not curriculum the hold length** (§3.4).

---

## 7. Anti-cheat: each measure and the specific hack it blocks

1. **Ramped stand income** → "jump up, collect, fall, jump again". Measured margin: a real
   hold outearns five pop-and-drops **470×**.
2. **`hold_steps` hard-reset to 0, never decayed** → accumulating credit across flickers.
3. **Sticky once-per-episode `succeeded` latch** → re-collecting success within one episode.
4. **Success never terminates the episode** → ending at the peak; also keeps the value
   bootstrap correct (terminating on a *good* state would zero its bootstrap, a classic
   PPO bug the repo's own `base.py` docstring warns about).
5. **Schmitt-trigger ENTER/HOLD bands** → the inverse failure, where boundary chatter under
   domain-randomised contact makes any hold unachievable and the gradient becomes a lottery.
6. **Height in the conjunction, not only orientation** → the sitting hack. Measured: seated
   reads `torso_upright 0.957` and `gravity_body[2] −0.99` at root height **0.053 m**.
7. **Foot *load fraction*, not `foot_contact`** → "feet are touching, therefore standing".
   Measured: `foot_contact` is TRUE in supine (46 N), prone (65 N), side (102 N) and
   kneeling (235 N). HumanUP's published indicator scores TRUE for a corpse on this body.
8. **`min(F_left, F_right) ≥ 0.20 BW`** → all weight on one leg with the other dangling.
9. **`min(hand_z) ≥ 0.35 m`** → propping up on a hand. Standing reads 0.83, prone 0.04.
10. **`base_offset ≤ 0.25 m`** → standing with the feet somewhere else (a split or lunge).
    Standing 0.088, every fallen pose 0.85.
11. **Ratchet on best-so-far, not a Δ-indicator** → earning height reward by bouncing.
12. **`quiet` and `posture` gated on `up * ramp`** → the corpse hack; a motionless body on
    the floor maximises any un-gated stillness term. Direct descendant of E02/E03.
13. **`scoot` penalty gated on `down`** → dragging along the floor to a favourable spot.
    The repo has already shipped a policy that "dragged itself forward by ground slip".
14. **`rise_rate` penalty + the §5/§6 torque cap** → the ballistic snap. Measured: one joint
    delivers 785 N·m today (1308 at fraction 1.0) against 65 N·m to hold a stand.
15. **Signed joint conjuncts on `abdomen_x`/`abdomen_y`** → the E23 pathology, "upright by
    the numbers, folded in half". Measured: `abdomen_y = −0.755` gives head_ratio 0.888 and
    fails the 0.90 gate; `−0.5` gives 0.945 but `torso_upright 0.753` and fails there.
16. **`head_height_ratio` chosen as the primary posture signal because it is a height** →
    the sign-blindness class of bug. A height has no sign to get wrong. Measured identical
    (0.927) for `abdomen_y = ±0.755` where `torso_upright` gives 0.654 vs 0.795.
17. **No early termination** → E16's "the first episodes to finish are the falls". With
    divergence as the only terminator, every episode is exactly `max_episode_steps` long,
    all 64 eval environments truncate on the same step, and the selection bias is
    *structurally impossible* rather than merely fixed.
18. **Stratified per-class eval** → E16 inverted (the research doc's hazard: easy poses
    finish first). Uniform episode length already removes the ordering effect; per-class
    success is still mandatory or a headline 45% hides "seated 90%, supine 0%".
19. **`eval/fall_rate` will read ≈0 by construction** and must not be read as competence.
    Report `fell_after_success_frac` (UniReLo's Time-to-Fall) and `standing_at_horizon`
    instead, and say so in the config comment.
20. **Explicit qpos address array** → the measured `hip_y`/`hip_z` transposition, which
    silently mis-applies joint limits in the existing tasks.

---

## 8. Integration

### New files
- `humanoid_rl/tasks/getup.py` — `GetUpConfig` (dataclass, every number above) and
  `GetUpTask(Task)`.
- `humanoid_rl/motion/fallen_bank.py` — `FallenPoseBank`, flat and index-addressable exactly
  like `motion/library.py`: `qpos (M,35) f32`, `qvel (M,34) f32`, `label (M,) i8`,
  `root_height (M,)`, `gravity_body (M,3)`, `head_ratio (M,)`, `source (M,) i8`.
- `scripts/generate_fallen_poses.py` → `data/fallen/bank_v1.npz`.
- `configs/getup.yaml` — `run.task: getup`, `env.max_episode_steps: 1250` (10 s, HoST's
  budget), `env.action_scale_fraction: 1.0`.

### `Task` interface mapping
| hook | GetUpTask |
|---|---|
| `task_obs_dim` | **4**: `[hold_progress, down_flag, head_ratio, force_scale]` |
| `init_state` | allocate all §3.3 arrays; stash `self._rng = rng` |
| `reset_batch` | §3.3 verbatim; record outcomes first |
| `reset_pose` | sample the bank; **must return a full `(len(indices), nq)` array for every row** |
| `reset_noise` | unused (returns None; `reset_pose` takes precedence anyway) |
| `observe_batch` | 4 slots above |
| `reward_batch` | `_advance_hold` first, then the 15 terms of §4 |
| `terminated_batch` | `~np.isfinite(state.qpos).all(axis=1)` **only** |
| `success_batch` | `return state.task_state["succeeded"]` |
| `on_batch_end` | curriculum EMAs, force-scale stage, class weights |
| `eval_metrics` | §8.3 |
| `mirror_task_obs` | identity on all 4 slots (none is chiral) — returns a clone unchanged |
| `action_offset` | `None` initially; the knee-+0.7 constant is the §5.1 fallback |
| `configure_for_model` | store `standing_height`, derive `root_height` thresholds as fractions |
| `set_joint_limits` | store, but build the **qpos address array** from `prepared.joint_names` |

### `reset_pose` — the constraint a naive implementation gets wrong

`vec_env._has_reset_pose` is a **single global bool** (line 234/496). If `reset_pose` returns
anything non-None, *every* resetting environment reads `_reset_qpos_abs[row]` where
`row = _reset_row[i] = position within done_idx`. So the "8% start standing" mix cannot be
expressed by returning `None` for some rows — those rows must be filled with the nominal
pose (`state.task_state["_nominal_qpos"]`, already published by the engine at line 207) plus
joint noise. Return a **fresh copy**; the workers read the array during `_run_phase(RESET)`.

Per-row randomisation applied to every sampled bank pose: random yaw about world z applied
to `qpos[3:7]` **and** to `qvel[0:3]` and `qvel[3:6]` (root qvel is world-frame — confirmed
by `_compute_derived` rotating it into the body frame), and `qpos[0:2]` zeroed. Zeroing XY
is safe here, unlike in `TrackingTask` where it broke the world-frame reference comparison,
because nothing in this task compares against a world position.

### Required `vec_env.py` changes (small, and one is a general bug fix)

1. **Seed the control filter from the reset pose.** Today `_do_resets` sets both
   `s.ctrl[done_idx]` and `_ctrl_filtered[done_idx]` to `_default_joint_pos` — the *standing*
   pose — so step 1 after a fallen reset commands standing from a body on the floor.
   Measured over 40 contorted settled fallen poses, first control step:
   **mean |τ| 164.4 N·m, p95 479, max 753, 11.7 of 28 joints over 150 N·m**, against
   **mean 0.1 N·m** when the target is the pose's own joint angles — a **2059×** reduction.
   (Note: on a *non*-contorted supine pose, i.e. nominal joints with only the root rotated,
   the effect is only 2.3 N·m mean. The severity is entirely a function of how far the joints
   are from nominal, which is why this must be measured on real fallen poses.)
   Fix, guarded so it is a no-op for tasks that supply no pose:
   ```python
   if self._has_reset_pose:
       joints = self._reset_qpos_abs[self._reset_row[done_idx]][:, self._qpos_adr]
       s.ctrl[done_idx] = joints
       self._ctrl_filtered[done_idx] = joints
   ```
   `TrackingTask` has the identical latent bug and this fixes it too.
2. **`set_actuator_force_scale(beta)`** — writes `actuator_forcelimited = 1` and
   `actuator_forcerange = ±beta * tau_max` across `self.model_pool`. Four lines.
3. *(Optional, recommended)* publish the full torso z-axis as `BatchState.torso_zaxis (N,3)`
   — `_compute_derived` already reads `sens[:, self._torso_adr + 2]` and the other two
   components are adjacent. This would give the torso lean a **sign**, closing the E23 gap
   at its source. The spec above does not depend on it.

### `model_prep.py` changes
- Add `joint_qpos_adr: np.ndarray` to `PreparedModel` (the measured transposition makes this
  necessary for any correct joint-indexed term).
- Add the nominal `actuator_forcerange` table of §5.2 and set `actuator_forcelimited = 1`.
- Expose `action_scale_fraction` through `EnvConfig` so `configs/getup.yaml` can set 1.0
  without editing code.

### `eval_metrics`
`success_rate_live`, `standing_now`, `hold_seconds`, `head_ratio`, `root_height`,
`torso_upright`, **`pelvis_pitch_signed` (`gravity_body[0]`)**, **`pelvis_roll_signed`
(`gravity_body[1]`)**, **`abdomen_y_signed` (`qpos[8]`)**, `knee_flexion`, `foot_load_frac`,
`base_offset_m`, `frac_supine/prone/side/sitting/down`, `time_to_stand_s`,
`fell_after_success_frac`, and the six per-class success EMAs. Per-class numbers are held as
EMAs in `task_state` and echoed each step, so `evaluate()`'s per-step averaging is
idempotent rather than noisy. **No uprightness cosine is ever reported without its signed
companions beside it.**

---

## 9. Sequencing

The two blockers in §5 must be settled **before** the pose bank is generated: both change
what "reachable" means, and a bank built against an action space that cannot express a
squat is an analysis cycle spent on the wrong object (the E23 failure mode). Order:

1. `model_prep` force limits + `joint_qpos_adr` + `action_scale_fraction` exposure.
2. `vec_env` filter-seeding fix + `set_actuator_force_scale`.
3. `scripts/generate_fallen_poses.py`, then validate the bank.
4. `GetUpTask`, with a unit test asserting `succeeded` is False for every row immediately
   after `reset_batch`, and that `hold_steps` never exceeds `episode_step`.
5. Smoke run at 256 envs for 50 iterations, checking that `terms[:, 0]` (`stand`) is
   non-zero — if it is identically zero, `standing_reset_prob` is not wired and nothing
   downstream can work.
