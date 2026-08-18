# Generating fallen initial states for a get-up task

Research agent, 2026-08-14. Scope: how to produce a realistic, diverse distribution of
on-the-floor poses to start get-up episodes from, in MuJoCo, on **this** model.

Everything below marked "measured" was run against `humanoid_rl/models/humanoid_scene.xml`
through `envs/model_prep.prepare()` on this machine. Scripts are reproduced in the appendix.

Model constants used throughout: `standing_height = 0.8769 m`, `standing_head = 1.5139 m`,
mass 50.0 kg, `terminate_height = 0.62 * standing = 0.5437 m`, physics 500 Hz, control 125 Hz
(one control step = 0.008 s).

---

## 1. What the field actually does

| Work | Sim | How fallen states are made | Notes |
|---|---|---|---|
| [HumanUP](https://arxiv.org/abs/2502.12152) (RSS 2025, G1) | Isaac Gym | **One** canonical supine pose and **one** prone pose, hard-coded root state + dof vector, plus ±0.3 m xy jitter | Diversity is NOT in the initial state. Their curriculum is over *control constraints* (slow-motion discovery policy → deployable tracking policy). They also ship a **custom collision-mesh URDF** because contact fidelity dominates this task. |
| [HoST](https://arxiv.org/abs/2502.08378) (RSS 2025 best-systems finalist, G1) | Isaac Gym | Posture diversity comes from **terrain**, not pose sampling: ground, platform (trunk supported, 20–92 cm), wall (14–84°), slope (1–14°) | Success = base height > 0.7 m (0.6 on slope) **maintained for the remainder of the episode**. Multi-critic over 4 reward groups; action rescaler β decays 1.0 → 0.25 to bound motion speed and torque. |
| [FRASA](https://arxiv.org/html/2410.08655) (Sigmaban) | **MuJoCo** | Random joint angles within limits + random trunk pitch → released above ground → **step the sim using the current joint configuration as PD targets until stability is reached**, then the episode begins | This is the canonical MuJoCo recipe and the one I reproduced and measured. |
| [Unified Humanoid Get-Up](https://github.com/utra-robosoccer/unified-humanoid-getup) (RoboCup 2025, 7 morphologies) | **MuJoCo** | Same recipe, but **pre-generated offline** by `standup_generate_initial.py` into a pickled bank of "a few thousand" settled poses; `stabilization_time = 2.0 s`; `reset_final_p = 0.1` starts 10% of episodes from the final *standing* state | The state-caching answer, shipped. |
| [HiFAR](https://arxiv.org/html/2502.20061) | Isaac | **Key State Initialization**: 6 hand-authored keyframes (3 prone, 3 supine) taken from handcrafted recovery motions | Explicitly chosen *to avoid unreachable and unrealistic poses* that random sampling produces. |
| [UniReLo](https://arxiv.org/html/2606.08922v3) (2026) | Isaac | Terrain-pose *plasticity-aware* initialization: prioritised sampling from an existing pose bank by measured variance in recovery success | A curriculum **on top of** a bank, not a way to build one. |

**Consensus**: drop-and-settle from randomised joints while the PD holds the spawn pose,
cached offline into a bank, then supplemented — with terrain (HoST) or hand-authored
keyframes (HiFAR) — to cover the postures dropping cannot reach. Nobody samples joint
configurations and projects them; nobody uses mocap of falls.

Mocap is unavailable to us anyway: `data/clips` is 100STYLE Neutral locomotion only, AMASS
registration is pending, LAFAN1 and LocoMuJoCo are ruled out by a NoDerivatives clause. Even
with AMASS, "getting up from the floor" clips are rare and retargeting a contact-rich floor
motion is far harder than retargeting a walk (contacts must be preserved, not just joints).
**Rule mocap out for this task.**

---

## 2. Measurements on this model

### 2.1 Drop-and-settle works and is nearly free

Random full orientation, joints = nominal ± N(0, σ), placed 0.3–0.8 m above contact, random
root linear (σ 0.5 m/s) and angular (σ 1.5 rad/s) velocity, 4 s of settling:

- **100 %** settle (max |qvel| < 0.5 held 100 ms) within 4 s; median quiet time **1.65 s**.
- **36 ms wall per pose**, single-threaded → 27.5 poses/s. 10 000 poses ≈ 6 min on one core,
  ≈ 40 s across the 10 physics workers.
- 0 divergences in 400 trials.
- Residual penetration at rest: **≤ 1.02 mm**.

### 2.2 Cached settled poses replay exactly through `reset_pose()`

The engine's reset path is `mj_resetData` → write `qpos/qvel` → `mj_forward`. It does **not**
settle and does **not** resolve penetration. Re-applying a settled `qpos` that way:

- penetration **≤ 1.016 mm**,
- after 0.5 s of holding the nominal target, root-height drift **0.0 cm**, max joint drift
  **0.00 rad**.

So a settled pose is a genuine fixed point of this engine and an offline bank is exactly
compatible with `Task.reset_pose()`. Precedent already exists: `tasks/tracking.py` +
`motion/library.py`.

Also verified: `envs/domain_rand.randomize_model()` touches only floor friction, body
mass/inertia, com offset, kp/kd and armature — **never geometry**. A bank settled on the base
model stays geometrically valid on every model-pool member.

### 2.3 Sampling joint configurations and placing them geometrically is unusable

Uniform joint angles over the full range, random orientation, root translated so the lowest
geom point sits at z = 0.001, `mj_forward` only (n = 200):

- **63 %** have at least one self-collision at spawn (median 1, max 12 contacts),
- penetration median **26.0 mm**, p95 **104.7 mm**, max **126.8 mm**,
- |Δqvel| after **one** 8 ms control step: median 1.32, p95 **12.1**, max **24.4 rad/s**.

That is an explosion at t = 0. Without a projection/relaxation step this option is dead, and
the projection step is just settling by another name — so settle instead.

### 2.4 The PD target during settling is the diversity knob, and it is silent

Same drop procedure, only the `ctrl` held during the fall differs:

| ctrl held while falling | joint-angle sd across the bank | as % of joint range |
|---|---|---|
| nominal standing pose | **0.012 rad** | **0.5 %** |
| the spawn pose (FRASA's recipe) | **0.278 rad** | **13.3 %** |

Holding the nominal pose produces *one* fallen pose repeated with different yaw — a rigid
mannequin lying down. The gains are stiff (kp 400–1000, saturating at 0.2–0.4 rad of error),
so the body simply cannot deform on the way down. Nothing in the pipeline would tell you this
had happened; the bank would look large and be rank-1.

### 2.5 Distribution mismatch with real falls is the largest error, and it is measurable

`best.pt` (iteration 3100), 64 envs, terminations monkey-patched off, deterministic actions,
state captured 2 s after root height first crosses 0.5437 m. 29 of 64 fell within 18 s.

| Source | supine (face up) | prone (face down) | side | root height |
|---|---|---|---|---|
| **Actual policy falls** (n = 29) | **3 %** | 21 % | **76 %** | mean 0.258, sd 0.109, range 0.124–0.482 |
| Drop, uniform orientation, nominal-hold | 46 % | 51 % | 4 % | mean 0.098, sd 0.021 |
| Drop, uniform orientation, spawn-hold | 42 % | 39 % | 19 % | mean 0.133, sd 0.062 |
| Topple only (tilt 35–92°), spawn-hold | 40 % | 28 % | 31 % | mean 0.132, sd 0.044 |

Two separate mismatches:

1. **Orientation is nearly inverted.** A walking humanoid falls sideways three quarters of the
   time; free-fall from a uniformly random orientation almost never lands on its side (4 %).
   Training on the drop distribution trains overwhelmingly on the case that does not happen.
2. **Real falls are not flat.** Policy falls rest at 0.26 m mean root height with a long tail
   to 0.48 m — half-collapsed, one hip under, partly propped. Dropped bodies are pancakes at
   0.098 ± 0.021 m. The *hard* part of getting up (organising limbs from flat) is
   over-represented; the common real case (partly propped, already half-way there) is absent.

Caveat: n = 29, one checkpoint, and the policy keeps acting on the fallen body during the 2 s
window (median residual |qvel| 0.68). Treat the direction as solid and the exact percentages
as indicative.

### 2.6 Natural falls are too slow to be a reset mechanism; scripted shoves are not

- Time to first fall under the walking policy: median **1107 control steps (8.9 s)**,
  p10 313, p90 1695. Unusable as an in-episode collapse phase.
- Scripted shove from standing (root impulse in a random direction, PD holding nominal):

| impulse | fell | time to floor | + 1.5 s settle | wall per fall |
|---|---|---|---|---|
| 1.5 m/s | 40/40 | 0.65 s (81 steps) | 268 steps | 20 ms |
| 3.0 m/s | 40/40 | 0.39 s (49 steps) | 236 steps | 17 ms |
| 5.0 m/s | 40/40 | 0.36 s (45 steps) | 232 steps | 15 ms |

An in-episode collapse phase therefore costs ~235–270 control steps, **9–11 % of a 2500-step
episode**, and those transitions must be masked out of the reward/termination or they pollute
the buffer. An offline bank costs zero at runtime. Use the bank; use the shove only to *build*
the bank.

### 2.7 Which constructed poses are actually stable

Pose set by joint angles, root placed on contact, PD holding that pose, 2 s of settling:

| constructed pose | outcome |
|---|---|
| seated, legs out front | **stable**: root h 0.051 m, pelvis upright (g_z −0.98), |qvel| → 0 |
| all-fours (hands + knees) | **stable**: root h 0.447 m, g_x +1.00 |
| seated, knees bent ("crash sit") | topples to supine, h 0.078 |
| kneeling | topples backward to supine, h 0.200 |
| deep squat | collapses to supine, h 0.100 |

So the seated pose the user explicitly named **is** constructible and stable, but it must be
hand-authored: the drop generators produced it **0 %** of the time in 600 samples. Kneeling
and squatting are *not* passively stable on this body — they can only appear as transients,
which means they belong in the reward/terminal logic, not in the initial-state bank.

Supine and prone are strong attractors: a supine body given roll rates up to **3 rad/s** about
its long axis does not roll over (splayed arms act as outriggers, and the CoM must rise ~5 cm,
≈ 25 J, against ≈ 13 J of rotational energy at 3 rad/s).

Contact-geometry note: the trunk is three spheres (pelvis r 0.09, upper_waist r 0.07, torso
r 0.11), the hands are 4 cm spheres with no wrist, the feet are 17.7 × 9 × 5.5 cm boxes. There
is no flat back. Body geoms are `condim="1"`, the floor is `condim="3"`; MuJoCo takes the max,
so **body-vs-floor contacts are frictional (μ = 1.0) but body-vs-body self-contacts are
frictionless**. Pushing off the floor with a hand works; a hand sliding along a thigh does not
grip.

A consequence for detection: a **seated** pose rests at root height ≈ 0.05 m and a **lying**
pose at ≈ 0.09–0.10 m. Root height cannot separate sitting from lying on this body — the
separating signal is `gravity_body[:, 2]` (≈ −1 seated, ≈ 0 lying) together with head height.

---

## 3. Two findings that block the task regardless of how the bank is built

### 3.1 The get-up motion is outside the action space

Actions are offsets from the **nominal standing pose**, scaled by
`action_scale_fraction = 0.6` of each joint's half-range, then clipped to the joint limit. The
commandable PD-target band is therefore `clip(nominal ± 0.6·half_range, joint_range)`:

| joint | joint range | commandable band | % of range |
|---|---|---|---|
| knee (both) | [0.00, 2.79] | **[0.00, 1.05]** | **37.5 %** |
| elbow (both) | [0.00, 2.79] | [0.00, 1.10] | 39.4 % |
| shoulder_x | [−1.92, 2.44] | [0.14, 2.44] | 52.8 % |
| hip_y | [−2.44, 1.05] | [−1.15, 0.94] | 60.0 % |
| abdomen_y | [−1.05, 1.57] | [−0.79, 0.79] | 60.0 % |

Mean over all 28 joints: **56.2 %**. Checking the poses a get-up is made of:

| sub-pose | verdict |
|---|---|
| kneel (knee 2.6) | **NOT reachable**, knee short by 1.55 rad (89°) |
| deep squat (hip −1.8, knee 2.4) | **NOT reachable**, knee short 1.35 rad, hip short 0.65 rad |
| sit legs-out (hip −1.5) | **NOT reachable**, hip short 0.35 rad (20°) |
| hands under shoulders (elbow 1.5) | **NOT reachable**, elbow short 0.40 rad, shoulder short 0.14 rad |
| forward fold (abdomen_y 1.4) | **NOT reachable**, short 0.61 rad (35°) |

Every one. The policy can be *pushed* into these configurations by contact, but it can never
*command* them, and the servo fights back at up to 150–200 N·m. "Fold into a pose, push off
the legs and arms, and rise from there" is not expressible in the current action space.

Fix options, in order of intrusiveness: give the get-up task its own `action_scale` covering
the full half-range and centred at joint mid-range instead of the standing pose; or raise
`action_scale_fraction` to 1.0 for this task only; or override `Task.action_offset()` with a
get-up-appropriate baseline. All three change what a warm-started weight means, so this is a
fresh-start decision, not a config tweak.

### 3.2 A fallen reset is spring-loaded by the engine's own reset code

`envs/vec_env.py` sets, on every reset, `s.ctrl[done_idx] = self._default_joint_pos` and
`self._ctrl_filtered[done_idx] = self._default_joint_pos`. With a fallen `reset_pose` that
means step 1 commands the **standing** pose from a body lying on the floor. Measured over the
29 policy-fall poses with action = 0:

- mean |PD torque| on step 1 = **100.2 N·m**, max **200 N·m**,
- **15.4 of 28 joints saturate their torque limit** on step 1 (max 19),
- 2.4 joints (max 7) start outside the commandable band entirely.

Saturation thresholds are `frc_limit / kp`: abdomen 0.2 rad, hip 0.4, knee 0.3, shoulder 0.25,
elbow 0.23, ankle 0.225 — so any fallen pose saturates most of the body. The first ~100 ms of
every get-up episode is a full-torque convulsion that no policy chose.

Fix: after `reset_pose` is applied, initialise both `s.ctrl` and `_ctrl_filtered` for those
rows to the **reset pose's own joint angles**, not `_default_joint_pos`. There is no Task hook
for this today; it is a ~3-line change in `_do_resets`.

---

## 4. Hackability baseline (measured, for the reward designer)

Trivial constant controllers from the fallen pose sets, 4 s each, "stood" =
root height > 0.5437 **and** torso_upright > 0.8 at the end:

| controller | from policy falls (n=29) | from dropped (n=40) |
|---|---|---|
| zero action (hold nominal standing pose) | 0 % stood, 0 % momentary | 0 % / 0 % |
| ramp to nominal pose over 1.5 s | 0 % / 0 % | 0 % / 0 % |
| flick both knees straight | 0 % / 0 % | 0 % / 0 % |
| flick abdomen_y to its limit | 0 % / 0 % | 0 % / 0 % |
| best of 40 random constant PD targets | 0 % stood, 0 % momentary, peak h 0.731 | 0 % stood, **12.5 % momentarily satisfied**, peak h **0.819** |

Good news: no single-joint flick stands this body up, so the user's "must not snap upright by
flicking one joint" concern is satisfied by the plant itself. Bad news: an oracle search over
just 40 *constant* pose targets already makes 12.5 % of dropped states momentarily satisfy a
naive "upright" test, reaching 0.819 m root height. PPO searches far harder than 40 samples.
**The hold-duration requirement is load-bearing, not belt-and-braces**, and 12.5 % is the
measured floor a "momentarily upright" metric would report for a controller that does nothing.

---

## 5. Recommendation

Build an **offline pose bank**, flat and index-addressable, exactly like `motion/library.py`,
and index it from `GetUpTask.reset_pose()`. Mix four generators:

1. **Policy falls, ~40 %.** Run the existing walker with terminations disabled and harvest
   states 1–3 s after the fall threshold. This is the only on-distribution source and it is
   cheap: with 1024 envs, ~1100 control steps yields several hundred falls, roughly one
   training iteration of physics. Keep the yaw and mirror left/right to double the sample.
2. **Scripted topple, ~35 %.** Standing + joint noise, random root impulse 1.5–5 m/s, PD held
   at a **random constant target drawn inside the commandable band** (not the nominal pose —
   see §2.4), settle 1.5 s. Realistic dynamics, real diversity, and every resulting pose is
   one the policy could have produced.
3. **Free fall from height, ~15 %.** Random full orientation, spawn-pose hold. Covers the
   supine/prone tails that toppling under-produces.
4. **Hand-authored keyframes, ~10 %** (HiFAR's KSI). Seated-legs-out, all-fours, side-lying
   with an arm trapped under, prone with hands under the shoulders. This is the only way to
   get "sitting on its backside" into the bank at all (drops produce it 0 % of the time), and
   it is what the user explicitly asked to be handled.

Plus **5–10 % of resets from the standing pose** (UHG's `reset_final_p`), so the hold-duration
reward is exercised from iteration 1 rather than only after the policy first stands.

Validate every candidate at build time and reject on failure:

- `mj_forward` → max penetration < 2 mm (measured achievable: ≤ 1.02 mm),
- finite `qpos`,
- fixed-point check: `mj_step` 0.25 s holding its own joint pose, require root-height drift
  < 2 cm and max joint drift < 0.05 rad (measured achievable: 0.0 cm / 0.00 rad),
- `qvel` zeroed, except for a deliberate ~10 % slice captured *before* settling with its
  momentum kept, so the policy also sees the still-tumbling case.

Store alongside each pose: taxonomy label, root height, `gravity_body`, `torso_upright`, head
ratio. Then the sampler can balance the bank to the *policy-fall* taxonomy (≈ 76/21/3
side/prone/supine) rather than the drop taxonomy, and a later curriculum (UniReLo-style) can
prioritise by measured recovery difficulty without regenerating anything.

Suggested shape: `humanoid_rl/motion/fallen_bank.py` (`FallenPoseBank` with
`qpos (n, nq)`, `qvel (n, nv)`, `label`, `root_height`, `gravity_body`, `torso_upright`),
`scripts/generate_fallen_poses.py` writing `data/fallen/*.npz`.

### Sequencing

§3.1 (action space) and §3.2 (spring-loaded reset) must be settled **before** the bank is
built, because both change what a "reachable" pose means. Building a beautiful bank against an
action space that cannot express kneeling would be a full analysis cycle spent on the wrong
object — the same failure mode as E23.

### One eval hazard, from this project's own history

E16: eval counted the first 32 of 64 episodes to finish, which are systematically the falls.
The identical trap reappears here inverted: with fallen initial states, the *easy* poses finish
first (they stand and truncate; hard ones flail for the full episode). Any get-up evaluation
must be one episode per environment **and** stratified by pose class, reporting success per
class, or the headline number is the success rate on whichever class happens to terminate
fastest.

---

## Appendix: reproduction

Scripts are in [`fallen-states-scripts/`](fallen-states-scripts/), run with the project venv
from the repo root:

- `fallen_states.py` — settling cost, penetration, replay fidelity (§2.1, §2.2, §2.3)
- `fallen_states2.py` — policy fall replay, taxonomy, constructed poses, roll stability (§2.5, §2.7)
- `fallen_states3.py` — hackability baselines, push-to-fall cost (§2.6, §4)
- `fallen_states4.py` — action-space reachability, reset torque (§3.1, §3.2)

`fallen_states3.py` consumes `policy_falls.npy` written by `fallen_states2.py`; both write to
the scratchpad path hard-coded at the top of each file, so change `SCRATCH` before rerunning.
All four import `humanoid_rl.envs.model_prep.prepare`, so they see exactly the plant training
sees (PD servo conversion, derived standing pose, sensors).
