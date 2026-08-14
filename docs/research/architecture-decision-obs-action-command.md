# Architecture decision: observation, action and command space

Lead-engineer reconciliation of four parallel research reports (hierarchy vs end-to-end,
terrain perception, multi-skill conditioning, omnidirectional commands), plus verification
against this repo and the run in progress.

Written 2026-08-13. Run under measurement: `runs/amp-20260813-134908`.

---

## 0. What was verified in this repo before deciding

| Claim | Verdict |
|---|---|
| `train.py:593-615` zero-pads a widened first layer on warm start | TRUE — columns only, not rows |
| `RunningMeanStd` has an unbounded `count` and divides by `sqrt(var+1e-8)` | TRUE (`networks.py:32-71`) |
| Commands are drawn only in `reset_batch`, never mid-episode | TRUE (`locomotion.py:235-258`) |
| `gait_frequency=0.9`, `stance_fraction=0.6`, `0.5` foot offset are hardcoded constants invisible to the policy | TRUE (`locomotion.py:103,107,388`) |
| `air_time_target=0.25` is a constant decoupled from the clock | TRUE — and the clock implies `(1-0.6)/0.9 = 0.44 s`. **The two terms contradict each other today.** |
| `mirror_clips: false` in the AMP config (claimed by the omnidirectional researcher) | **FALSE.** It is `true` in `configs/amp.yaml` and in the running config. That researcher read a stale file. |
| `clip_include` excludes the fast clips | TRUE. Measured mean speeds: FW 0.65, BW 0.60, SW 0.52, TR1 0.85 (included); **FR 1.24, BR 0.89, SR 0.97 (excluded)**. Commands ask for up to 1.5 m/s. |
| Command box gives 52.7% forward / 3.7% lateral | TRUE, reproduced over 2M samples. 45° diagonals are structurally capped at 0.57 m/s — not rare, *unrepresentable*. |
| Proposed sampler + clock cost | **Measured here: 175 µs per policy step at 4096 envs = 0.20% of the 86 ms budget.** |

Run state at iteration 871/4069 (79M/400M steps, 47,630 sps, **a full run is 2.3 hours**):

```
gait_phase  0.435 -> 0.53 by it.500 -> flat/declining (max 1.0; ~0.5 is chance)
feet_air_time  -0.020  (net penalty: mean swing < 0.25 s, vs the 0.44 s the clock asks)
lin_vel  0.29 / 1.0        episode_length  415 / 600 (falls at ~8.3 s)
amp/style_reward 0.336 -> 0.151      amp/accuracy 0.43 -> 0.998
```

The clock term is at chance. The discriminator has won. This run is a control, not a
foundation.

---

## A. The observation layout to adopt today

**102 → 108. Thirteen task dimensions. Zero reserved-for-later slots.**

```
[  0 :  95]  proprioception                                  USED NOW   unchanged
               joint pos 28, joint vel 28, gravity 3,
               lin vel 3, ang vel 3, prev action 28, foot contact 2
[ 95 :  98]  velocity command (vx, vy, yaw_rate)             USED NOW   unchanged
[ 98 : 100]  gait clock (sin, cos) of 2*pi*phase             USED NOW   masked to (0,0) when f_cmd == 0
[100 : 102]  heading error (sin, cos)                        USED NOW   unchanged
[102]        f_cmd, commanded stride frequency in Hz         USED NOW   NEW
[103]        s_cmd, commanded stance fraction in (0,1)       USED NOW   NEW
[104]        d_cmd, inter-foot phase offset in [0,1)         USED NOW   NEW
[105]        alpha, clock authority in [0,1]                 USED NOW   NEW
[106]        h_cmd, body height as a ratio of standing       USED NOW   NEW
[107]        command age, min(t_since_resample, 2.0)/2.0     USED NOW   NEW
```

Action: **28 PD position targets, unchanged.**

### Why zero reserved slots — this is the ruthless part

The hierarchy researcher wants eight zeros appended at `[102:110]`. Reject.

1. **They buy nothing this repo does not already have.** `train.py:593-615` already grows
   the first layer and zero-fills, verified identical to 0.000e+00. Reserving a zero today
   and populating it in six months is *the same operation* as appending it in six months,
   performed six months early. The mechanism is the mechanism; the width is not.
2. **A constant-zero channel is actively dangerous here.** `RunningMeanStd.update` merges
   batches with an uncapped `count`. A channel that is exactly 0.0 for a run collapses to
   `var ≈ 1e-9` (init `var=1, count=1e-4`, first 98,304-sample batch drives it to ~1e-9).
   `normalize` then divides by `sqrt(1e-9 + 1e-8) ≈ 1.05e-4`, a gain of ~9,500. The day the
   channel carries 0.3, it normalises to ~2,850 and clamps at the ±10 bound — and every
   reserved channel does it *simultaneously*. Worse, after a 400M-step run `count ≈ 4e8`, so
   a fine-tune batch of 98,304 samples moves the statistics by 2.4e-4 of the way: tens of
   thousands of iterations to recover. That is a de-facto fresh start caused purely by the
   normaliser. Reserving zeros is not free; it is a loaded gun.
3. **Semantic drift.** A slot with no meaning and no reward invites a future definition that
   conflicts with the warm-started weights. A slot with a value and a reward cannot drift.

**Rule: reserve by exercising, or do not reserve.** Every one of the six new slots is
sampled with real variance from step one and is read by at least one reward term. Nothing is
carried dead.

### Justification, slot by slot

| Slot | Future capability | Why it must be live *today* rather than appended later | Cost |
|---|---|---|---|
| `[102] f_cmd` | run, slow-walk, crouch-walk, cadence change near a drop, terrain | The clock is currently pinned at 0.9 Hz — right at ~1.25 m/s and wrong everywhere else. A policy that has seen exactly one cadence will refuse another. This is the slot that turns "run" from a retrain into a command value. Precedent: Walk These Ways `gait_frequency_cmd_range [2.0,4.0]`, Booster T1 `[1.0,2.0]` randomised per segment. | 1 float |
| `[103] s_cmd` | run (s<0.5 ⇒ flight phase), longer double support for balancing near a drop | Siekmann et al. (ICRA 2021, Cassie) showed stand/walk/run/hop/skip are all one reward with different (f, s, d). `s` is the walk↔run parameter. Also removes the `expected[standing]=1.0` special case. | 1 float |
| `[104] d_cmd` | hop, two-footed jump takeoff, skip | The only route to symmetric two-footed skills *through* the clock rather than around it. Narrow today ([0.45,0.55]); widening the range later is a sampling change, not a shape change. | 1 float |
| `[105] alpha` | jump, land, climb, ledge balance — anything aperiodic | The answer to "crouch/climb fighting the clock". `alpha` multiplies `w_gait_phase` per environment *and* is observed, so the policy is told when the schedule is void rather than being punished by a term it cannot see. Reward-routing, as in arXiv 2505.20619, done as a continuous observed scalar instead of a one-hot. | 1 float |
| `[106] h_cmd` | **squat / crouch to go under things** — explicitly on the roadmap | WTW got crouching on real Go1 hardware with no retraining purely because `body_height_cmd` was a training input. Costs nothing here: the height reward already exists, it just reads `h_cmd * standing_height` instead of a constant. | 1 float |
| `[107] command age` | the future 10 Hz nav layer; jump takeoff countdown | Lets the policy distinguish "command just changed, expect a transient" from steady state — the regime a high-level layer keeps it in permanently. Becomes ANYmal Parkour's *time-to-target* for free. | 1 float |

### Explicitly rejected slots

- **A one-hot skill vector.** Every skill on this roadmap is either a point in the
  continuous (f, s, d, h) space (stand, run, hop, crouch, duck) or is selected by geometry
  through a perception encoder (jump, climb, ledge). Humanoid Parkour Learning (CoRL 2024)
  runs a real 19-DoF humanoid that jumps 0.8 m gaps and climbs 0.42 m platforms with **no
  skill command at all** — the terrain embedding is the selector. A one-hot is a slot that
  must be extended every time a skill is added, which is the failure mode we are avoiding.
- **A target-relative vector (dx, dy) in the low-level observation.** The heading error at
  `[100:102]` already *is* the target-direction signal in the only form the low level can
  act on. Adding a position target teaches the low level to derive its own velocity, which
  silently moves arbitration into the frozen layer; the symptom appears months later as the
  nav layer being unable to override the walker. Rudin et al. (IROS 2022) and Skill-Nav do
  not *add* a position command, they *replace* the velocity command — a coherent but
  different architecture, and the wrong trade here.
- **HOVER-style mask bits.** A mask bit is only needed when a channel's value can be
  undefined. Every slot above has an always-defined value (`alpha` is itself the continuous
  "off switch", and it is always defined). Two slots saved.
- **The 231-dim height scan, today.** See §E.
- **Foot swing clearance and torso pitch commands.** Both need new reward machinery
  (per-foot swing apex tracking) that does not exist. They fail the "cheap to exercise
  today" test, and appending them at the terrain phase is a warm-started fine-tune. Dropped.

### Mirror rules — mandatory, and one of them is currently wrong

`ppo.py:419` routes everything past the proprioception width to `mirror_task_obs`, so new
slots pass through *unchanged* by default. That is correct for `f, s, alpha, h, age`
(mirror-invariant) but silently wrong for `d_cmd`.

Correct derivation: the observed clock drives the **left** foot; the right foot is at
`phase + d`. Under reflection the new left foot is the old right, so `phase' = phase + d`
and `d' = 1 - d`. The current code negates `sin` and `cos`, which is a rotation by exactly
half a cycle — correct **only when d = 0.5**. With `d ∈ [0.45, 0.55]` it is off by up to 18°
of phase on every augmented sample.

Fix (4 lines): rotate the (sin, cos) pair by `phi = 2*pi*d` exactly —
`sin' = sin*cos(phi) + cos*sin(phi)`, `cos' = cos*cos(phi) - sin*sin(phi)` — and map
`d -> 1 - d`. At `d = 0.5` this reduces to the current negation, so the change is a no-op
for the present behaviour.

### Compute cost of the whole layout

- Observation 102 → 108: `+6 × 512 = 3,072` weights on the actor first layer and the same on
  the critic, ~0.6% of that layer, ~0.3% of the ~320k-parameter actor. Observation batch
  1.67 MB → 1.77 MB at 4096 envs.
- Sampler + parameterised clock, **measured on this machine**: 175 µs per policy step at
  4096 envs against an 86 ms budget at the run's actual 47,630 sps. **0.20%.**

---

## B. The action space

**Nothing must change. 28 PD position targets, today and for jumping, crouching and
climbing.**

Every surveyed system agrees, on real hardware: ANYmal Parkour's five skills (walk, jump,
climb up, climb down, crouch) all output the same 12 joint targets as its walk; Humanoid
Parkour Learning's jump/leap/climb all output the same 19; HOVER's 15 real-world control
configurations all output the same 19. Not one paper adds an action dimension for a skill.

### On the proposed 29th output (a policy-chosen phase offset)

The multi-skill researcher wants `phi_{t+1} = phi_t + f*dt + clip(a_dphi, ±0.06)` (arXiv
2504.13619), on the grounds that the action space is the one thing that is expensive to
retrofit. Reject, for three reasons:

1. **`alpha` subsumes it, and is strictly more general.** That paper needed a policy-chosen
   phase offset because their clock reward is *always binding*. With `alpha`, the schedule
   can be released entirely — commanded from above, where a future high-level layer can
   decide "no rhythm here", rather than negotiated by the low level.
2. **It breaks the mirror machinery.** `mirror.py` builds `action_perm` and `action_sign`
   from the model's actuators, length `nu = 28`. A 29th non-actuator output needs a
   special case in `MirrorSpec.mirror_action`, and `symmetry_augment: true` is on.
3. **The premise is false in this repo.** Growing the action is *also* warm-startable —
   the loader just does not do it yet. `train.py:600` handles `dim == 2` growth along
   columns only. Adding the symmetric row-growth branch (pad new output rows with zeros,
   pad `log_std`) makes an appended action output a provable no-op at introduction, exactly
   as an appended observation slot is.

**Do the loader patch now** (30 minutes, zero behavioural change). It converts "the action
space is irreversible" from true to false, which is worth far more than the 29th output.

**What genuinely is irreversible in the action space:** changing the number of actuated
joints, changing joint/actuator order, or changing action scaling / the default-pose offset.
Those redefine what an existing output *means*, and no amount of padding survives that.

---

## C. The gait clock

The clock fixed a real defect (the rocking split-stance gait: right foot locked 36 cm
forward, 7.4 foot strikes/s, 0.42 lead swaps/s). Keep it. It is not wrong because it is a
clock; it is wrong because its three parameters are constants the network cannot see.

### The parameterisation (Siekmann et al., ICRA 2021 — one reward spans stand/walk/run/hop/skip)

Per environment, sampled at every command resample and **observed**:

```
f       stride frequency, Hz            f = 0  <=>  standing (both feet expected in stance)
s       stance fraction in (0,1)        s < 0.5 => a flight phase => running
d       inter-foot phase offset         0.5 = alternating (walk/run), 0.0 = hopping / two-footed
alpha   clock authority in [0,1]        1 = schedule binding, 0 = schedule void
```

Expected stance per foot, wrap-safe and smooth (replaces the hard boolean at
`locomotion.py:390`, which is a step discontinuity at both boundaries of every cycle):

```python
u_L = phase % 1.0
u_R = (phase + d) % 1.0
# centre each stance window on 0, so the wrap is handled by the modulo
def stance(u, s, w=0.05):                 # w = transition width in cycles
    up = (u - s / 2 + 0.5) % 1.0 - 0.5    # in [-0.5, 0.5)
    return np.clip((s / 2 - np.abs(up)) / w + 0.5, 0.0, 1.0)
expected = np.stack([stance(u_L, s), stance(u_R, s)], axis=1)
expected[f <= 0] = 1.0                    # standing: both feet down; no special case on the command
phase = (phase + f * dt) % 1.0            # f is now a vector, not cfg.gait_frequency
```

`f = 0` is the *only* standing condition. It replaces the current
`expected[standing] = 1.0` command-magnitude test, and it also masks the clock observation
to `(0, 0)` — exactly Booster T1's protocol. Standing becomes a point in the gait space
instead of an `if` statement.

Gait coverage without any shape change: walk `(f≈0.9, s=0.6, d=0.5)`, run `(f≈1.6, s=0.4,
d=0.5)`, hop `(f, s, d=0.0)`, stand `(f=0)`, crouch-walk `(f≈0.7, s=0.65, h_cmd=0.7)`.

### Gating — three reward terms are hardcoded to walking, not one

This is the part that matters for "a future crouch or climb is not fighting it". `alpha` on
its own is not enough, because two other terms encode walking just as hard:

1. **`gait_phase`** — multiply by `alpha` per environment. Trivial.
2. **`flight` (`w_flight = -0.3`)** — this term *directly forbids running and jumping*. It
   currently penalises any instant with neither foot down, unconditionally. Change it to
   penalise only *unscheduled* flight:
   ```python
   expected_flight = np.clip(1.0 - expected.sum(axis=1), 0.0, 1.0)   # 1 when both feet swing
   terms[:, 8] = cfg.w_flight * np.maximum(0.0, flight - expected_flight)
   ```
   With `s ≥ 0.5, d = 0.5` this is identical to today's behaviour. With `s < 0.5` the
   schedule licenses the flight it asks for. Without this fix, `[103]` is a slot the reward
   refuses to let the policy use.
3. **`air_time_target` (`0.25`)** — must become `(1 - s) / f`, the swing duration the clock
   actually implies. **This is already broken today, before any of these changes:** the
   current clock implies `(1 - 0.6)/0.9 = 0.44 s`, the constant says `0.25 s`, and the
   measured `reward/feet_air_time` is `-0.020` — a net penalty, meaning the policy is
   stepping faster than either target. The clock term is stuck at 0.53 (chance is ~0.5).
   The two terms are pulling in different directions and the clock is losing.

Do **not** gate `feet_air_time`, `feet_slip` or the AMP style reward by `alpha`. On
`alpha = 0` environments those are the only things standing between the policy and a
rhythmless shuffle — the exact pathology the clock was introduced to kill. (Note the AMP
style reward cannot be relied on for this: it is at 0.151 and falling, with discriminator
accuracy 0.998.) Gating AMP off belongs to the jump phase specifically, per the
selective-AMP result, and should then be a separate per-environment weight, not `alpha`.

### Frequency command law

```
f_nom = clip(0.9 * (max(|v_xy|, 0.3) / 1.25) ** 0.5, 0.60, 1.35)   # Hz, Inman's sqrt law
f_cmd = f_nom * U(0.85, 1.15)                                       # jitter: the part that buys the future
f_cmd = 0                                                           # when the command is zero
```

The anchor `(0.9 Hz at 1.25 m/s)` is the repo's own current constant placed at a defensible
speed. The **jitter is load-bearing**: without it the slot is a dead input the network learns
to ignore, and it will be as unusable in six months as if it had never existed. Verify after
a few hundred iterations that `obs_rms.var[102:107]` is non-trivial.

The exponent and the clip bounds are judgement, not a measured result for this humanoid.
Inman's law is fitted around preferred speed and over-predicts the cadence drop at the low
end; the clip floor at 0.60 Hz exists for that reason.

---

## D. Command sampling

### Resample mid-episode. Yes. This is the single largest live defect.

Commands are drawn only in `reset_batch`. At 1000-step episodes (`default.yaml`) that is
**one command held for 20 seconds**; at 600 steps (`amp.yaml`) 12 seconds. Every reference
implementation resamples mid-episode: legged_gym 10 s, Isaac Lab `(10.0, 10.0)`, Walk These
Ways 10 s, Booster Gym `[8, 12]` s. The policy in training today has never experienced a
command transient, and a future navigation layer changes the command every 5-10 low-level
steps.

```
resample interval    U(2.5, 6.0) s, independent per environment
                     -> 4-8 command segments per 20 s episode, 2-5 per 12 s AMP episode
on resample          redraw command, f, s, d, alpha, h_cmd; reset command age
do NOT reset         gait phase, pose, or desired_heading
```

**Bootstrap the value at the change.** Copy Booster Gym's one line in spirit: flag the
resample step as a truncation so GAE bootstraps `V(s)` there without resetting the episode.
Without it the critic must predict a return across a reward-scale discontinuity it cannot
see coming; value loss spikes and advantages are corrupted across the whole preceding
segment. Mid-episode resampling *without* this is worse than not resampling at all.

I reject the proposal to re-anchor `desired_heading` from the current facing on each
resample. The heading term exists precisely to punish accumulated drift; re-anchoring
forgives exactly what it is there to catch. Anchor at episode reset only, as today.

**Do** make `desired_heading` externally settable (a per-environment `integrate | hold`
mode). It is already in `task_state` and therefore already writable; the change is a mode
flag and a test. This is the cheapest possible hook for camera-based target seeking and it
changes no observation.

I also reject the proposed 0-3 step command-latency ring buffer. Randomised hold lengths
already expose the policy to step changes, and AOW drove a 10 Hz high level from a low level
trained with a 0.005/step jolt — what matters is robustness to a step change, not its rate.

### Ranges: polar under an elliptical envelope, not a box

The box is measurably incapable of the roadmap's first line. Over 2M samples of
`vx ~ U(-0.5,1.5), vy ~ U(-0.4,0.4)`, binned into 45° sectors (moving commands only):

```
             share   mean |v|   p95 |v|          max commandable in that direction
forward      52.7%     0.98      1.47            1.50
fwd-diag      9.9%     0.49      0.86            0.57
lateral       3.7%     0.30      0.40            0.40
back-diag     6.9%     0.39      0.57            0.57
backward      6.2%     0.37      0.50            0.50
```

Forward is practised 14× more often than pure strafing, at 3.3× the speed. And the diagonal
cap is **geometric, not statistical**: "walk diagonally at 1 m/s" cannot be expressed in this
command space at all. No amount of training fixes that.

Replace with polar sampling under a half-ellipse pair:

```python
theta = rng.uniform(-pi, pi, k)
m     = sqrt(rng.random(k))                       # uniform over the envelope's area
a     = where(cos(theta) >= 0, FWD, BACK)
R     = 1.0 / sqrt((cos(theta)/a)**2 + (sin(theta)/LAT)**2)
speed = m * R
vx, vy = speed*cos(theta), speed*sin(theta)
yaw    = rng.uniform(-1.0, 1.0, k)                # independent
FWD, BACK, LAT = 1.5, 0.8, 0.6                    # m/s
```

Measured result (same 2M-sample protocol): **12.2-12.9% in every one of the eight sectors**,
with per-direction ceilings forward 1.50 / diagonal 0.79 / lateral 0.60 / backward 0.80.
Direction coverage becomes uniform while the *speed* envelope stays human-shaped.

The envelope is anchored on human data — Laufer 2005 (forward 138.3±16.5 vs backward
82.6±18.2 cm/s), Handford & Srinivasan 2014 (preferred sideways 0.575±0.123 m/s, >3× the
cost of transport of forward walking; above ~0.97 m/s humans stop walking sideways
altogether) — and independently corroborated by this project's own mocap, measured today:
FW 0.65 / FR 1.24, BW 0.60 / BR 0.89, SW 0.52 / SR 0.97 m/s. The current config's
asymmetry has the right *shape* and the wrong *magnitudes*: forward is above human preferred
while backward (0.5) and lateral (0.4) are below it — the two directions the roadmap says
must be as good as forward are the two that are undertrained.

Do **not** copy Margolis's "lateral is a small separate uniform" — that is a quadruped fact
(a quadruped strafes by yawing and driving forward; a humanoid needs crossover/side-step
footfalls that share almost no structure with forward gait).

### Rest of the sampler

```
deadband          |v_xy| < 0.15 m/s snaps to exactly 0   (kills the march-in-place band,
                  and makes the existing moving/standing gates at 0.1 exactly consistent)
zero_command_prob 0.05 -> 0.10                            (Booster still_proportion: 0.1)
yaw               U(-1.0, 1.0) rad/s, independent of the linear draw
s_cmd             U(0.55, 0.65)      d_cmd  U(0.45, 0.55)
alpha             0.0 on 10% of segments, 1.0 otherwise
h_cmd             U(0.94, 1.02)
```

### No command curriculum

The literature is genuinely split — Margolis et al. report outright training failure without
one ("a robot jittering in place"), but that is 3.9 m/s on a Mini Cheetah. Isaac Lab ships
no command curriculum at all; Booster Gym implements a full grid curriculum and ships it
`curriculum: false`. At 1.5 m/s on a humanoid this is the Isaac Lab / Booster regime.
There is also a project-specific reason to refuse: Booster's version couples lateral range to
the forward level, a forward-first schedule that would bake in exactly the asymmetry being
removed. If PPO stalls for >30M steps, the escape hatch is a **uniform** scale on `R(theta)`
from 0.4× to 1.0× — never per-axis. Spend the curriculum machinery on terrain in Phase 4,
where the evidence is unambiguous.

### Superset discipline for the future nav layer

Train on the envelope above; clamp any future high-level output to ~0.7× of it
(`FWD 1.05, BACK 0.55, LAT 0.42`, yaw ±0.7) with a bounded output distribution (AOW used a
Beta). The low level must always have been trained outside what the high level can ask for.

### Two reward terms must change with the sampler, in the same commit

- Split `lin_vel` into per-axis `track_vx` (w 1.0, σ 0.15) and `track_vy` (w 1.0, σ 0.15),
  reusing the existing term slots 0 and 15 so the reward matrix keeps 17 columns. The
  current `sigma_lateral_vel = 0.05` kernel was the right fix for uncommanded drift, but
  under polar sampling most segments *command* lateral motion, and a 0.05 kernel on a
  commanded 0.6 m/s strafe is punitive (a 0.2 m/s error scores 0.45 instead of 0.85). The
  command distribution, not an asymmetric kernel, is now what teaches lateral control.
  Watch the straight-forward eval case specifically, since that is where the drift bug
  originally appeared.
- **Fix the AMP reference set in the same commit or it will veto everything above.** The
  included clips top out at 0.65 m/s mean (measured) while commands ask for 1.5, so the
  discriminator has no human example of fast walking and the style reward actively opposes
  the task reward at the top of the range. `Neutral_FR` (1.24 m/s), `Neutral_BR` (0.89) and
  `Neutral_SR` (0.97) are already on disk. Add them to `clip_include`. (`mirror_clips` is
  already `true` — the claim that it was `false` was wrong.) Then balance the
  discriminator's real batch by direction rather than by clip length, so eight equal-length
  clips do not silently become a forward-heavy prior.

### Evaluation, so "equally well" is measurable

Replace random-command eval with a fixed 8-direction protocol: `eval.num_envs 64 -> 128`,
16 environments pinned per compass direction, commanded at `0.6 * R(theta)` with zero yaw.
Report per-direction tracking error, foot slip, gait symmetry and step rate — and make the
headline number the **worst direction, not the mean**. A mean is exactly what lets a
52.7%-forward policy look fine. Keep total episode count flat so eval cost does not move.
Add a second protocol that steps the command through forward → strafe → diagonal → stop on
the resample interval and reports settling time, since that is what a nav layer produces.

---

## E. The migration path

### Free — bit-identical continuation, no capability loss

1. **Appending observation slots at the END.** `train.py:593-615`, verified 0.000e+00.
2. **Appending action outputs at the END** — after the 30-minute row-growth loader patch.
3. **Adding an encoder module whose output is concatenated at the END, with a zero-initialised
   final layer.** This is the important one and it is why the terrain block does not need to
   exist today. `load_state_dict(strict=False)` leaves a brand-new module at its
   initialisation; if that module's last `Linear` has zero weight *and* zero bias, its latent
   is identically zero on the first forward pass, so the widened trunk columns see exactly
   the zeros the loader would have padded. The augmented policy is bit-identical to the
   checkpoint. This is RMA's mechanism — a fixed-width latent slot with a swappable producer
   — and it is the actual anti-retrain guarantee, not observation width.
4. **Anything in the critic**, once the critic's observation is split from the actor's. The
   critic is discarded at deployment. Until that split exists, however, anything you want the
   critic to see must also go into the actor's observation — which is a reason not to shove
   privileged information in now.

### Needs a fine-tune (warm start, hours, no capability loss if rehearsed)

Reward weights, sampling distributions, termination thresholds, domain randomisation, new
reward terms, widened command envelopes, new terrain, a new skill, training a newly-attached
encoder. **A full run here is 2.3 hours.** Fine-tuning is not expensive on this machine; the
scarce resource is the number of sequential experiments, not compute.

The one rule that makes skill addition safe: **do not fine-tune on the new skill alone.**
Parkour in the Wild's measured recipe — keep every old terrain in the sampler and let the new
skill occupy ~3% of environments — took a new terrain from 11% to 92.4% success with old
terrains unaffected, and beat new-terrain-only fine-tuning *even though only 3% of samples
were on the new terrain*. At 4096 environments that is an indexing change: 128 new, 3968 old.
Reinforce with a KL penalty against the frozen pre-skill policy on old-distribution states
(one extra forward pass of a 320k MLP — negligible). Keep the critic across the warm start:
`ActorCritic` bundles it in `state["policy"]`, so it already travels. Do not "clean that up"
— a freshly initialised critic is the classic cause of early-fine-tune collapse.

### Genuinely requires a fresh start

- Changing the **number of actuated joints**, or their **order** (breaks action semantics and
  the mirror permutations, which are derived from the model).
- Changing **action scaling, PD gain semantics, or the default-pose offset** — these redefine
  what an existing output means.
- **Inserting or reordering** observation slots, or **redefining the meaning** of an existing
  slot. This is strictly worse than appending, because it corrupts a warm start silently.
  Reinterpreting the shared clock `(sin, cos)` as per-foot would be exactly this mistake.
- Changing the model file or the physics engine.
- (Near-fresh) Changing `decimation` / control frequency: `prev_action` semantics and the
  effective dynamics both move. Recoverable, but expect to pay most of a run.

**Write this as a comment above `task_obs_dim` and add a unit test pinning the index of every
existing slot.** The test is what stops a future self from inserting.

### What will *not* be an extension of this policy — say it now

Parkour-grade footstep precision (landing a foot on a specific 15 cm ledge, cliffside
climbing) is above the ceiling of a velocity-commanded interface. Rudin et al. (IROS 2022)
demonstrated velocity-command decomposition cannot cross terrain a position-and-time
commanded policy can; WoCoCo does cliffside climbing with **no height map at all**, using
explicit base-frame contact-goal geometry. The honest budget is that **this walker is the
first of a small family of low-level policies sharing one observation and action shape**,
selected by a high-level layer — exactly ANYmal Parkour's structure. Everything in the first
three roadmap milestones sits comfortably below that ceiling.

### Decisions frozen today that cost nothing to freeze

- **Vision is a separate layer above a frozen velocity-commanded walker.** Convergent across
  ANYmal Parkour, the ANYmal-on-wheels city navigation system, ViNL, NaVILA and Fu et al.,
  on quadrupeds, wheeled-legged robots and humanoids, all on real hardware. ViNL is the
  decisive one: a navigation policy trained in Habitat and a locomotion policy trained in
  Isaac Gym composed **zero-shot** through a 3-number velocity command. The velocity command
  is a load-bearing API boundary, not a convenient fiction.
- **The Apple Silicon constraint makes this the only option, not a preference.** Madrona-MJX,
  the batch renderer used for vision RL with MuJoCo, is a CUDA ray tracer and does not run on
  Metal. End-to-end vision at 4096 envs × 50 Hz needs ~204,800 depth frames/s against roughly
  1e3/s from MuJoCo's CPU renderer. Any future advice proposing pixels inside the 4096-env
  loop should be rejected on this ground alone. A high level at 10 Hz over 64 envs needs ~640
  frames/s and fits comfortably.
- **Perception enters through an encoder to a 32-dim latent concatenated at the END, never
  into the flat vector.** 32 is Humanoid Parkour Learning's width for both its 11×19 scandot
  MLP and its 48×64 depth CNN, which is what lets the depth encoder later replace or join the
  height-scan encoder without reshaping the trunk. **Nothing is appended after perception.**
- **Terrain, when it lands (Phase 4):** symmetric 21×11 grid at 0.1 m = 231 points, base-
  centred, yaw-only rotation, `clip(root_z - H_nom - terrain_z, -2, +1)` × a **fixed 5.0**,
  **excluded from `RunningMeanStd`**, computed as an analytic heightfield gather on MPS
  (measured 3.73 ms/batch = 4.6% of budget) and never by ray casting (measured 629-768 ms per
  batched step, 8-9× the entire budget; `mj_multiRay` takes a single shared origin and cannot
  express a parallel downward grid). One shared heightfield at **0.10 m cells** (measured
  3.2× plane collision cost, vs 6.5× at 0.05 m) with environments at different xy offsets —
  per-environment terrain is impossible because `hfield_data` lives in `MjModel` and
  `vec_env.py:153-159` already documents that 4096 copies would be 21 GB. The grid must be
  y-symmetric or `mirror.py` cannot reflect it.

### Why the terrain block is not being added today

The terrain researcher makes a good case for a live "virtual" height scan over flat physics
during Phase 3 (the BeamDojo trick). I am deferring it, because the zero-init encoder makes
its later introduction *provably* bit-identical, so the only thing training it now buys is a
warm encoder and warm statistics — worth roughly a 20-50M-step fine-tune, i.e. under an hour
on this machine — against building terrain generation (`humanoid_rl/terrain/` is an empty
package), an MPS gather path in an observation pipeline that is currently numpy on the main
thread, and a multi-encoder network, all during a phase whose gait is not yet working.
Deferring is the cheaper trade by a wide margin. What must **not** be deferred is fixing
`RunningMeanStd`, because that is the thing that would turn the later addition from a
fine-tune into a fresh start.

---

## F. What to do right now, in order

The run at `runs/amp-20260813-134908` is at iteration 871/4069, 79M/400M steps, 47,630 sps —
**1.9 hours to completion, 2.3 hours for a full run.** Restarting is cheap. Let it finish: it
is the control, and its `best.pt` is the warm-start source. Nothing below requires killing it.

| # | Change | Effort | Compute | Risk |
|---|---|---|---|---|
| 1 | **Loader + normaliser hardening.** Row-growth branch in `init_policy_from` (action growth) + `log_std` padding. Cap `RunningMeanStd.count` at ~1e6 so the normaliser stays adaptive across phases, and add a per-channel variance floor. Unit test pinning every observation index. | 45 min | none | none — no behavioural change |
| 2 | **AMP reference set.** `clip_include: [FW, FR, BW, BR, SW, SR, TR1, ID]`; balance the discriminator's real batch by direction, not clip length. | 15 min | none | low |
| 3 | **Command sampler.** Polar elliptical envelope, `U(2.5,6) s` mid-episode resample, deadband 0.15, `zero_command_prob 0.10`, truncation flag on resample for the value bootstrap. | 2-3 h | none | medium — see below |
| 4 | **Clock parameterisation.** `f, s, d, alpha` as per-environment arrays; six new observation slots; soft stance indicator; `flight` and `air_time_target` derived from the schedule; height reward reads `h_cmd`; per-axis velocity kernels; the `d`-aware mirror rotation. | 3-4 h | none | medium |
| 5 | **Eval protocol.** 8 pinned directions × 16 envs, headline = worst direction; a command-transition settling-time protocol. | 1 h | ~none | none |
| 6 | **Pre-flight probe.** 10-minute run commanding pure lateral at 0.6 m/s from the current `best.pt`, to check this 28-DoF model can physically strafe at the proposed envelope at the current PD gains. | 20 min | 10 min | none — this is the check |
| 7 | **Launch the fine-tune**, `--init-from` the current run's `best.pt`, observation 102 → 108 (zero-padded, verified identical). | — | 2.5 h | — |

Roughly one working day of implementation and three hours of compute.

### Things to watch on that first run

- `obs_rms.var[102:108]` must be non-trivial after a few hundred iterations. If any of those
  channels has near-zero variance, the corresponding parameter is not actually being
  randomised and the slot is dead.
- `reward/gait_phase` should climb above 0.53. It has been flat at chance since iteration 500
  with a schedule the policy is not following.
- **Forward tracking will get transiently worse.** Forward's share of commands drops from
  52.7% to 12.9% by construction, and mean commanded speed drops from 0.69 to 0.53 m/s. Judge
  on the worst-direction metric; the mean and any metric that implicitly rewards high
  commanded speed will fall for reasons unrelated to policy quality.
- `alpha = 0` environments: watch `gait_symmetry` and `lead_swaps_per_sec` on that subset
  separately. If a rhythmless shuffle appears, drop the fraction from 10% to 5% before
  weakening anything else.
- **Separate live defect, outside this brief but blocking Phase 3:** the discriminator is at
  0.998 accuracy with the style reward decayed 0.336 → 0.151. The motion prior is
  contributing almost nothing, which is the failure the warm-start was supposed to prevent.
  Adding the fast clips (#2) will help slightly by making the real distribution broader, but
  this needs its own investigation.

---

## Where the researchers conflicted, and the calls

| Conflict | Call | Why |
|---|---|---|
| Reserve 8 zero slots (hierarchy) vs exercise-only (multi-skill, terrain) | **Exercise-only, zero reserved slots** | The repo already zero-pads on widening, so reserved zeros buy nothing a later append does not; and `RunningMeanStd` turns a constant channel into a ~9,500× gain with an uncapped count. |
| 231-dim height scan today (terrain) vs later | **Later, through a zero-init encoder to a 32-dim latent** | The zero-init encoder makes the later addition bit-identical using the *existing* loader. Training it now buys warm statistics for under an hour of fine-tune, against days of plumbing. |
| Action 28 → 29 (multi-skill) vs 28 (hierarchy, terrain) | **28** | `alpha` subsumes the phase-offset action and is commanded from above rather than chosen by the low level; a 29th output needs a special case in `mirror.py`; and action growth is warm-startable once the loader patch lands, so it is not now-or-never. |
| Command curriculum (Margolis) vs none (Isaac Lab, Booster) | **None** | Margolis's regime is 3.9 m/s on a quadruped. Booster implements one and ships it disabled. Escape hatch is a uniform envelope scale, never per-axis. |
| Re-anchor `desired_heading` on resample (omnidirectional) | **Reject** | It forgives exactly the drift the heading term exists to punish. Anchor at episode reset only. |
| 0-3 step command latency buffer (hierarchy) | **Reject** | Redundant with randomised hold lengths; its only real content is an obs/reward disagreement, better modelled as command observation noise. |
| HOVER mask bits (multi-skill) | **Reject** | Every proposed slot has an always-defined value. `alpha` is the only off-switch and it is continuous. |

## Asserted without evidence — flagged

- **`mirror_clips: false`** (omnidirectional researcher) is **wrong**; it is `true` in
  `configs/amp.yaml` and in the running config. That researcher was reading a stale file.
  Their `clip_include` finding is correct and was independently confirmed by measuring the
  clips.
- The terrain researcher's per-machine timings (height-scan gather 3.73 ms on MPS,
  heightfield collision 3.2× plane at 0.10 m cells, ray casting 629-768 ms/batch) were **not
  re-verified here**. They are internally consistent and the raycast-vs-gather conclusion
  matches legged_gym's own design choice, so I am acting on them — but the 0.10 m cell-size
  decision should be re-measured before Phase 4 commits to it.
- **No published parameter table exists** for Siekmann's per-gait `(theta, r, cycle time)` or
  the von Mises `kappa`. The specific ranges for `f, s, d` are judgement scaled from this
  repo's own constants.
- The **envelope numbers (1.5 / 0.8 / 0.6 m/s)** are human data and are untested on this
  28-DoF model at these PD gains. Hence step 6, the pre-flight probe. If the model cannot
  strafe at 0.6 m/s, the envelope must come down or the reward will spend the whole run
  paying for something unreachable.
- Three 2026 citations (`arXiv:2601.07718` "Hiking in the Wild", `arXiv:2604.19102`
  selective-AMP multi-gait, `arXiv:2604.19102`'s per-gait AMP coefficients) **could not be
  independently verified here**. They support the "even end-to-end systems keep a velocity
  command" and "gate AMP off for jumping" claims. Both conclusions are corroborated by
  sources I could verify, so the plan does not depend on them.
- The hierarchy researcher self-reported correcting an early misreading of
  `arXiv:2601.07718` (hierarchical → actually single-stage end-to-end), noting the erroneous
  reading would have supported their conclusion more conveniently. That is the right
  behaviour and increases my confidence in the rest of that report.
