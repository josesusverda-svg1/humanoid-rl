# Conformance Audit — OURS vs XBot-L / Unitree G1 / Booster T1

Date: 2026-08-13. References: humanoid-gym (XBot-L, main), unitree_rl_gym (G1, main, SHA-verified), booster_gym (T1, main @ da396a0). All three are hardware-proven. Default verdict is CONFORM; a divergence survives only with a cited measurement.

Verdicts:
- **ALIGNED** — ours matches shipped practice (at least one reference, no contradiction from the others).
- **JUSTIFIED-DIVERGENCE** — differs, but a recorded measurement backs it.
- **CONFORM** — differs with no measurement; adopt the reference value.
- **MISSING-FROM-OURS** — reference mechanism absent in ours; adopt.

---

## RETRACTION, added 2026-08-15 (E31). READ THIS BEFORE USING ANY ROW BELOW.

**The OURS column's control rate was wrong, and it is the single source of a family of bugs.**

This audit recorded ours as `50 Hz (200 Hz phys, decim 4)`. Measured from the shipped scene:
`humanoid_scene.xml` has `timestep 0.002` (**500 Hz** physics) and `decimation 4`, so the real
control rate is **125 Hz** and one control step is **8 ms**. The same wrong rate was written
into a comment on `model_prep.py`, the function that loads that very model, so nothing in the
repo contradicted it.

Every verdict below that compares a per-step or per-second quantity against a 50 Hz reference
therefore compared the wrong thing, and "ALIGNED (G1 exactly)" was the most wrong of all: G1
runs at 50 Hz, we run 2.5x faster. Rows signed off by this audit that are now known bad:

| Row | Said | Actually |
|---|---|---|
| Policy rate | 50 Hz, ALIGNED with G1 exactly | **125 Hz**, 2.5x faster than every reference in the table |
| Episode length | 20 s (1000 steps) | 1000 steps was **8 s** at 125 Hz. Fixed to 2500 steps (E19). |
| `gamma` 0.99 | ALIGNED | 0.99 buys the references a 1.67-3.33 s horizon and buys us **0.80 s** (E31) |
| `horizon` 24 | ALIGNED | 0.48-0.60 s at the references, **0.19 s** here. Not yet changed; logged. |
| Per-step reward weights | ALIGNED | Accumulate 2.5x faster per second here. The references multiply reward by `dt`; this repo does not. |

**Rule going forward: never compare a constant, compare the quantity it stands for.** A
discount factor is not a number, it is a horizon in seconds. An episode limit is not a step
count, it is a duration. Before adopting any value from another repo, convert it through that
repo's control rate and ours.

---

## A) Row-by-row table

### Control / actuation

| Parameter | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| Policy rate | **125 Hz (500 Hz phys, decim 4)** | 100 Hz (1000 Hz, decim 10) | 50 Hz (200 Hz, decim 4) | 50 Hz (500 Hz, decim 10) | ~~ALIGNED (G1 exactly)~~ **RETRACTED E31**: faster than all three. |
| Episode length | 20 s (2500 steps) | 24 s | 20 s | 30 s | **ALIGNED** (G1), but only after E19 fixed 1000 steps, which was 8 s here. |
| Action semantics | [-1,1] × per-joint scale + default pose, position servos | 0.25·a + default, PD torque | 0.25·a + default, PD torque | 1.0·a + default, clip 1.0, PD torque | **ALIGNED** (T1-style unit actions). Structural note: refs compute explicit PD torque and clip to torque limits; ours relies on MuJoCo position actuators + ctrlrange clamp (see E) |
| Action low-pass filter (8 Hz one-pole in plant) | yes | no (random delay-blend + mult. noise instead) | no | no (random 0–18 ms delay instead) | **JUSTIFIED-DIVERGENCE** — measured noise-crutch: falls 2% with 62 Hz noise vs 100% deterministic. But see "actuator latency DR" row: refs model *random delay*, which we lack |
| Actuator latency DR | none | per-step random delay-blend delay∈[0,0.5] | none | per-env delay 0–18 ms, resampled each reset | **CONFORM** (adopt T1 per-env delay; 2/3 refs model latency; our fixed 8 Hz filter is deterministic and learnable — a random delay is what breaks delay-exploitation) |
| Action clip | env clips to [-1,1] | 18 (pre-scale) | 100 | 1.0 | **ALIGNED** (T1) |

### Rewards — tracking

| Term | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| tracking_lin_vel | 1.0, exp(-err/0.25), body-frame xy | 1.2, exp(-err·5) | 1.0, exp(-err/0.25) | 1.0 per axis, exp(-err/0.25), EMA-filtered vel | **ALIGNED** (G1/T1 form and value) |
| tracking_ang_vel | 0.5, exp(-err/0.25) | 1.1, exp(-err·5) | 0.5, exp(-err/0.25) | 0.5, exp(-err/0.25) | **ALIGNED** |
| lateral_vel (extra narrow kernel) | 0.4, sigma 0.05 | — | — | (T1 splits x/y but with same sigma 0.25, weight 1.0) | **CONFORM** — no reference has a second, narrower lateral kernel. Either delete (lin_vel already covers vy) or adopt T1's symmetric per-axis split at sigma 0.25 |
| heading (extra reward term) | 0.5, exp(-err²/0.15), pinned desired heading | no reward — heading **command**: yaw_cmd = clip(0.5·wrap(hdg−ψ), ±1) every step | same heading-command mechanism | neither | **CONFORM** — the 47°-drift measurement proves the *problem*, not this mechanism. XBot and G1 solve identical drift with a heading command driving the existing yaw-tracking term: simpler, no new reward, hardware-proven. Adopt heading_command; delete the reward term |

### Rewards — posture / base

| Term | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| orientation | upright +0.3 clip(−g_z) AND torso_upright +0.6 | +1.0 dual exp kernel | −1.0 · Σ g_xy² | −5.0 · Σ g_xy² | **CONFORM** — replace the two positive clip terms with a single projected-gravity penalty, −1.0·Σsquare(gravity_body[:, :2]) (G1). Positive clip form saturates (zero gradient once roughly upright); squared penalty keeps a gradient |
| head_height | +0.3 · clip(head ratio) | — | — | — | **CONFORM (delete)** — ours-only, no measurement; redundant with height + orientation terms |
| base_height | +0.3 exp kernel sigma 0.02, target 0.877·height_cmd | +0.2 exp·100, target 0.89 | −10.0 squared, target 0.78 | −20.0 squared, target 0.68 | **ALIGNED** (XBot-style positive kernel; magnitude in family). The body-height *command* input is ours-only — see D |
| lin_vel_z / vertical_vel | −0.8 × p_scale · vz² | positive exp kernel (0.5) | **−2.0** · vz² | **−2.0** · vz² | **CONFORM** — −2.0, no p_scale (G1+T1 agree exactly) |
| ang_vel_xy | −0.05 × p_scale | positive exp kernel | −0.05 | −0.2 | **ALIGNED on value (G1)** but **CONFORM: remove p_scale gating** (no ref attenuates penalties) |
| alive/survival | 0.5 | — (only_positive total instead) | 0.15 | 0.25 | **CONFORM** — 0.25 (T1). At 0.5, standing-and-breathing pays 2–3.3× reference practice, which is exactly the standing-profitable failure mode already measured at low difficulty |
| base/root acc | — | +0.2 exp | — | −1e-4 | ALIGNED (G1 also lacks it; 1/3+1/3 split — no action) |

### Rewards — effort / smoothness

| Term | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| torque penalty | −0.005 × p_scale on **ctrl² (position targets!)** | −1e-5 on torque² | −1e-5 on torque² | −2e-4 on torque² (+ torque_tiredness −1e-2, power −2e-3) | **MISSING-FROM-OURS** — all three penalize *actual torque squared*; ours penalizes squared position commands, a quantity no reference uses (and which punishes large joint targets, not effort). Adopt: read MuJoCo `actuator_force`, weight ≈ −2e-4 (T1, closest robot class) |
| dof_vel penalty | **none** | −5e-4 | −1e-3 | −1e-4 | **MISSING-FROM-OURS** — unanimous. Adopt −1e-4 (T1) to start |
| dof_acc penalty | **none** | −1e-7 | −2.5e-7 | −1e-7 | **MISSING-FROM-OURS** — unanimous. Adopt −1e-7 (XBot=T1). This is the canonical anti-jitter term for sim2real |
| action_rate | −0.01 × p_scale | −0.002 (incl. 2nd deriv + magnitude) | −0.01 | −1.0 (unit-action robot) | **ALIGNED** (G1 value) — but remove p_scale |
| dof_pos_limits | none (ctrlrange clamp only) | none (safety clip 0.85 torque instead) | −5.0 at soft limit 0.9 | −1.0 at hard limits | **CONFORM** (adopt G1: −5.0, soft_dof_pos_limit 0.9; 2/3 refs; clamping the command does not stop momentum carrying the joint into the stop) |

### Rewards — feet / gait

| Term | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| gait clock ↔ contact match | +1.0 soft window, parameterised (freq, stance frac, offset observed) | +1.2 (+1/−0.3 discrete) | +0.18 bool match | +3.0 swing-window only | **JUSTIFIED-DIVERGENCE** — parameterised clock (Siekmann lineage) is the recorded design; all four use a clock-contact term, ours is a superset |
| feet_air_time | +1.0 · (air − target)·first_contact, signed error vs gait-derived target | +1.0 · clamp(air,0,0.5)·first_contact | **0.0 — explicitly disabled** (phase contact term replaces it) | none | **CONFORM** — no ref pays a *signed* error against a derived target. Either disable (G1 precedent: clock reward makes it redundant) or use XBot's clamped-positive form. Recommend disable |
| flight (all-feet-airborne) penalty | −0.3 unscheduled flight | — | — | — | **CONFORM (delete)** — ours-only, no measurement; clock match + (new) swing-height term cover it |
| feet_slip | −0.2 · Σ v_xy² of loaded feet | −0.05 sqrt form | −0.2 · Σ v² (contact_no_vel) | −0.1 · Σ v² | **ALIGNED** (G1 value, G1/T1 form) |
| swing foot height target | **none** | +1.0 clearance @0.06 m ±0.01 | −20.0 · (z−0.08)²·~contact | none (swing window only) | **CONFORM** (adopt G1: penalize (foot_z − 0.08)² on non-contact feet; 2/3 refs; directly buys real-terrain clearance) |
| feet_distance / crossing guard | **none** | +0.2 (min 0.2 m, max 0.5) + knee 0.2 | none | −1.0 clip(0.2 − lateral_sep, 0, 0.1) | **CONFORM** (adopt T1 form, ref 0.2 m; 2/3 refs; leg-crossing is the classic unpenalized failure) |
| contact force cap | none | −0.01 (>700 N) | none | none | ALIGNED (1/3 — no action) |
| foot yaw/roll alignment | none | none | none | −1.0 yaw_diff, −1.0 yaw_mean, −0.1 roll | ALIGNED (T1-only — no action) |
| collision penalty | none (termination covers) | −1.0 on base | 0.0 (disabled) | −1.0 on many bodies | ALIGNED (split practice; our tilt/height terminations play XBot's base-contact role) |
| only_positive total clip | yes | yes | yes | yes | **ALIGNED** (unanimous) |
| reward scales × dt | **unverified in our code** | yes (×0.01) | yes (×0.02) | yes (×0.02) | **See E** — unanimous in refs; must verify ours |
| penalty leniency schedule (p_scale 0.25→1.0, EMA-gated) | yes, on 4 terms | — | — | — | **CONFORM (delete)** — ours-only, no measurement. All refs ship fixed penalty weights from step 0 |

### PPO

| Parameter | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| gamma / lambda | 0.99 / 0.95 | 0.994 / 0.9 | 0.99 / 0.95 | 0.995 / 0.95 | **ALIGNED** (G1) |
| lr, schedule | 1e-3, adaptive KL 0.01, bounds [1e-5, 1e-2] | 1e-5 adaptive | 1e-3 adaptive | 1e-5 adaptive (same 1.5× rule, same bounds) | **ALIGNED** (G1; T1 shares the exact adapt rule) |
| horizon / envs / batch | 24 / 4096 / 98,304 | 60 / 4096 | 24 / 4096 / 98,304 | 24 / 4096 / 98,304 | **ALIGNED** (G1=T1 exactly) |
| epochs / minibatches | 5 / 4 | 2 / 4 | 5 / 4 | 20 full-batch, recomputed GAE | **ALIGNED** (G1 exactly) |
| clip, value clip, grad norm | 0.2 / clipped / 1.0 | same | same | 0.2 / no value clip / 1.0 | **ALIGNED** |
| entropy_coef | **0.0** | 0.001 | 0.01 | +0.01 (sign quirk → bonus) | **CONFORM** — all three ship a positive entropy bonus. Our zero exists only to prop up the ours-only std ceiling (next row); fix both together |
| exploration std | init 0.45, **hard ceiling 0.5** (log_std_max −0.70), floor exp(−5), in-place clamp | init 1.0, free | init 0.8, free | init 0.135 (logstd −2), free, no decay | **CONFORM** — no reference caps std. Adopt the T1 recipe as a package: init_noise_std ≈ 0.135–0.2, entropy_coef +0.01, bounds_loss_coef 1.0, **no ceiling**. The "entropy drove log_std into ceiling" observation is an artifact of the ceiling+0.45-init combination, not a measurement against any reference recipe |
| bounds_loss_coef | 10.0 (IsaacGymEnvs value) | — | — | **1.0** | **CONFORM** — 1.0; T1 is the only reference with this loss and ships 1.0 |
| advantage norm | whole-rollout | rsl_rl std | rsl_rl std | whole-batch, per mini-epoch | **ALIGNED** |
| optimizer | AdamW | Adam | Adam | Adam | **CONFORM** (Adam; trivial but zero refs use AdamW) |
| exploit lanes (25% envs @ 0.1× noise) | yes | — | — | — | **JUSTIFIED-DIVERGENCE** — measured noise-crutch persistence below cutoff |
| symmetry loss / augment | 0.0 / off (dead config) | — | — | dead yaml key only | **ALIGNED** (dead in T1 too) — delete the dead keys |
| critic warmup, obs-stat freeze | 0 / iteration-frozen stats | n/a (no obs norm) | n/a | n/a | see obs-normalisation row below |

### Observations / networks

| Parameter | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| **base lin vel in ACTOR obs** | **YES (lin_vel_body in proprio)** | NO — privileged critic only | NO — privileged only | NO — privileged only | **MISSING-FROM-OURS (asymmetric actor-critic)** — unanimous and the single biggest sim2real divergence: base linear velocity is not directly measurable on hardware; every reference trains the actor blind to it |
| foot contact in ACTOR obs | YES (2%-bodyweight threshold) | NO (privileged stance/contact masks) | NO | NO | **CONFORM** — remove from actor (unanimous); move to privileged critic obs |
| privileged critic obs | **none** (actor and critic see same obs) | 219-dim (push, friction, mass, contact, ref-diff) | 50 (=obs+lin_vel) | 61 (=obs+14: mass/com, lin vel, height, push) | **MISSING-FROM-OURS** — unanimous. Minimum viable: critic obs = actor obs + true lin_vel + push state + mass/friction scalars |
| obs normalisation | RunningMeanStd (Chan), checkpointed, freeze-per-iter | fixed hand scales + clip 18 | fixed scales + clip 100 | fixed scales (dof_vel×0.1) + noise | **See D** — zero refs use running norm; all use fixed per-component scales. Ours also already ate one warm-start failure (COUNT_MAX cap) from this machinery |
| obs noise | Gaussian on proprio block | Gaussian ×0.6 level | uniform ×1.0 level | Gaussian | **ALIGNED** |
| history | single frame | 15-frame stack | LSTM | single frame | **ALIGNED** (T1) |
| networks | actor+critic [512,512] ELU separate | 512-256-128 ELU | LSTM64+[32] | 256-128-128 / 256-256-128 ELU | **ALIGNED** (in family; T1 nearest) |
| clock obs | sin/cos + f, stance, offset, authority | sin/cos | sin/cos | sin/cos, gated to 0 when standing | **JUSTIFIED-DIVERGENCE** (parameterised clock, measured lineage); standing gating ALIGNED with T1 |

### Commands / curriculum

| Parameter | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| sampling geometry | polar elliptical, area-uniform, mag ≥ 50% | box | box | box | **JUSTIFIED-DIVERGENCE** — measured: box gave 52.7% forward bias, diagonals capped 0.57 m/s |
| ranges | vx [−0.5, 1.5], vy ±0.4, yaw ±1.0 | vx [−0.3,0.6], vy ±0.3, yaw ±0.3 | ±1.0 / ±1.0 / ±1.0 | ±1.0 / ±1.0 / ±1.0 | **ALIGNED** (G1/T1 class) |
| deadband | 0.15 m/s | 0.2 | 0.2 | still_proportion 0.1 | **CONFORM** (0.2 — XBot=G1; trivial) |
| zero-command prob | 0.05 | deadband only | deadband only | 0.10 | **ALIGNED** (between practices) |
| resample cadence | hold U(2.5, 6.0) s | 8 s | 10 s | U(8, 12) s | **CONFORM** — lengthen to U(8,12) s (T1). All refs hold commands ≥8 s; 2.5 s holds never let steady-state gait errors accumulate into the reward |
| truncate-at-redraw bootstrap | yes (command_changed) | n/a | n/a | yes (time_out at resample, no reset) | **ALIGNED** (T1 does exactly this) |
| difficulty curriculum | **enabled**: radial per-env 0.70→1.0, promote/demote, gait-match gates, graduate recycling | curriculum=False | curriculum=False (code dead) | curriculum: false (shipped off) | **CONFORM** — all three ship with curriculum **disabled** and train full ranges from step 0. Fix difficulty ≡ 1.0; the measured 0.45-floor failure is evidence the curriculum *created* a standing-profitable regime, and the ≥50%-magnitude floor survives in the polar sampler anyway |
| gait frequency | clip(0.9·sqrt(v/1.25), 0.60, 1.35)·U(0.85,1.15); 0 standing | fixed 1.5625 Hz (0.64 s) | fixed 1.25 Hz (0.8 s) | U(1.0, 2.0) Hz sampled | **CONFORM-ADJUST** — the parameterised clock is justified, but our band 0.60–1.35 Hz sits almost entirely *below* every reference (1.25, 1.56, 1.0–2.0). Shift to ≈ U(1.0, 2.0) Hz (T1) or re-anchor the Inman law to 1.25 Hz @ 1.25 m/s |
| stance fraction / offset jitter | U(0.55,0.65) / U(0.45,0.55) | fixed (sinusoid) | fixed 0.55 / 0.5 | fixed 0.2-swing / 0.5 | **JUSTIFIED-DIVERGENCE** (parameterised clock package; jitter centers match G1's 0.55/0.5) |
| body-height command | U(0.94, 1.02)·standing | — | — | — | **CONFORM (delete/freeze at 1.0)** — no reference commands height; ours-only, no measurement |

### Terminations / resets / DR

| Parameter | OURS | XBOT | G1 | T1 | Verdict |
|---|---|---|---|---|---|
| terminations | height <0.62·stand; tilt g_z>−0.7; stooped torso<0.5; head<0.65; NaN | base contact only | pelvis contact; |pitch|>1.0, |roll|>0.8 | vel² >50; height <0.45 m (0.66·target) | height+tilt **ALIGNED** (T1 height ratio 0.66≈ours 0.62; G1 tilt ~same class). stooped/head_down: **CONFORM (delete)** — ours-only, redundant with tilt+height. NaN guard: harmless keep |
| timeout bootstrap | truncation bootstraps V(s) | yes | yes | yes | **ALIGNED** (unanimous) |
| reset noise | joints N(0, 0.02), root untouched | dof ±0.1, root vel 0 | dof ×U(0.5,1.5), root vel U(±0.5) | dof N(0,0.05), yaw U(0,2π), lin vel N(0,0.1) | **CONFORM-lean** — ours is the most timid; no ref leaves root velocity unperturbed and only T1-style ±1 m xy is matched. Adopt T1: dof std 0.05, random yaw, root lin vel noise. Cheap robustness |
| friction DR | U(0.5, 1.25) | U(0.1, 2.0) | U(0.1, 1.25) | U(0.1, 2.0) | **CONFORM** — lower bound 0.1 (unanimous); low-friction is the case hardware actually meets |
| mass/CoM DR | per-body ×U(0.85,1.15), CoM N(0,0.015) | base ±5 kg | base +[−1,3] kg | base ×U(0.8,1.2), CoM ±0.1 m | **ALIGNED** (T1-style; our CoM range is narrower than T1's ±0.1 m base — optional widen) |
| kp/kd DR | ×U(0.85, 1.15) | none | none | ×U(0.95, 1.05) | **ALIGNED-ish** (1/3 refs; ours wider — acceptable, watch for over-randomisation) |
| pushes | ~5 s, vel impulse U(0,0.7) | 4 s, ±0.2 lin ±0.4 ang | 5 s, ±1.5 xy | 5 s force + 2 s kicks | **ALIGNED** (mid-pack) |

---

## B) CONFORM CHANGE SET (ranked by expected impact)

1. **Actor obs: remove `lin_vel_body`; add privileged critic** (unanimous — XBot/G1/T1 all train the actor without base linear velocity and feed it to the critic only). Also remove `foot_contact` from actor obs (unanimous), move both into a critic-only privileged block (+ push state, mass/friction scalars per T1's 14-dim block). Keys: obs layout in `locomotion.py`/`vec_env.py`, critic input dim in `networks.py`. *This is the #1 sim2real divergence.*
2. **Add `w_dof_acc = -1e-7`** (XBot = T1; G1 −2.5e-7) — unanimous, absent in ours.
3. **Add `w_dof_vel = -1e-4`** (T1; XBot −5e-4, G1 −1e-3) — unanimous, absent in ours.
4. **Replace ctrl² penalty with a true torque penalty**: `w_torque = -2e-4` on `actuator_force²` (T1). No reference penalizes squared position commands.
5. **Exploration package (adopt T1 wholesale)**: `entropy_coef 0.0 → 0.01`, delete `log_std_max` ceiling (no ref caps std), `init_noise_std 0.45 → ~0.15`, `bounds_loss_coef 10.0 → 1.0`. Change as one unit — the pieces are coupled.
6. **Delete penalty-leniency schedule** (`p_scale` → constant 1.0). No reference anneals penalties; keys: `penalty_scale` init/floor/ceiling/EMA machinery in `locomotion.py`.
7. **Disable difficulty curriculum**: difficulty ≡ 1.0, remove promote/demote/graduate-recycling (all three ship `curriculum=False`). Keep polar envelope + magnitude ≥ 50% floor (both measured).
8. **`w_vertical_vel -0.8 → -2.0`**, un-gated (G1 and T1 agree on −2.0 exactly).
9. **Replace heading reward with heading command** (XBot/G1): sample heading target, every step `cmd_yaw = clip(0.5·wrap(heading − yaw), −1, 1)`; delete `w_heading` term and desired-heading pinning logic.
10. **Command hold 2.5–6 s → U(8, 12) s** (T1; XBot 8, G1 10).
11. **Gait frequency band 0.60–1.35 Hz → ~U(1.0, 2.0) Hz** (T1; fixed refs at 1.25/1.56 Hz). Ours currently walks slower-cadence than every hardware-proven policy.
12. **Add swing-height penalty**: −20.0 · (foot_z − 0.08)² on non-contact feet (G1; XBot equivalent clearance term) — 2/3 refs.
13. **Add feet_distance guard**: −1.0 · clip(0.2 − lateral_sep, 0, 0.1) (T1; XBot equivalent) — 2/3 refs.
14. **Add dof_pos_limits penalty**: −5.0 at soft limit 0.9 (G1) — 2/3 refs.
15. **`w_alive 0.5 → 0.25`** (T1; G1 0.15).
16. **Friction DR lower bound 0.5 → 0.1** (unanimous).
17. **Delete**: `w_lateral_vel` (or fold into T1-style per-axis sigma 0.25), `w_flight`, `w_head_height`; merge `upright`+`torso_upright` into single −1.0 · Σ gravity_xy² (G1). Delete stooped/head_down terminations. Freeze body-height command at 1.0.
18. **Feet_air_time: disable** (G1 precedent — clock-contact reward supersedes it) or convert to XBot's clamped-positive form; delete the signed-error-vs-derived-target form.
19. **Add per-env action delay DR** 0–20 ms resampled at reset (T1; XBot has per-step variant).
20. **Reset noise**: add root yaw U(0,2π) and root lin vel noise N(0,0.1), dof std 0.02→0.05 (T1).
21. Minor: deadband 0.15 → 0.2; optimizer AdamW → Adam; delete dead symmetry config keys.

## C) What ALL THREE do that we do not (highest signal)

1. **Actor trained without base linear velocity; privileged/asymmetric critic.** We feed `lin_vel_body` (and `foot_contact`) straight into the actor. Not deployable-faithful; unanimous in all three hardware stacks.
2. **`dof_acc` squared penalty** (−1e-7…−2.5e-7). Absent in ours. The canonical anti-vibration term — directly relevant to the noise-crutch problems we instead patched with the 8 Hz filter.
3. **`dof_vel` squared penalty** (−1e-4…−1e-3). Absent in ours.
4. **True torque² penalty** — all three penalize applied torque; we penalize position-target magnitude, a different physical quantity.
5. **Positive entropy bonus** (0.001–0.01) with an *uncapped* learned std.
6. **Curriculum disabled** — every reference trains its full command range from step 0.
7. **Fixed per-component obs scaling** (no running normalisation) — see D.
8. **Reward scales multiplied by dt** — unanimous in refs; unverified in ours (see E).

## D) What WE do that NONE of them do, with no measurement (delete-candidates)

| Mechanism | Recommendation |
|---|---|
| Penalty-leniency schedule (p_scale 0.25–1.0, episode-EMA gated) | **Delete** — set 1.0. Confounds every penalty weight comparison above |
| Difficulty curriculum (promote/demote/graduate recycling) | **Delete** — fix 1.0. The one related measurement (0.45-floor failure) indicts the curriculum, and the 50%-magnitude floor lives in the sampler |
| Exploration std ceiling (log_std_max = −0.70) + entropy 0 | **Delete ceiling**, restore entropy 0.01 (coupled change, B#5) |
| `lateral_vel` narrow kernel (sigma 0.05) | **Delete** or conform to T1 sigma 0.25 |
| Heading-hold reward term + desired-heading pinning | **Replace** with XBot/G1 heading command |
| `flight` penalty | **Delete** |
| `head_height` reward + head_down/stooped terminations | **Delete** (redundant with orientation + height + tilt) |
| `torso_upright` as separate positive term | **Merge** into single squared projected-gravity penalty |
| Body-height command U(0.94, 1.02) | **Freeze** at 1.0 |
| ctrl² penalty as effort proxy | **Replace** with torque² |
| Signed feet_air_time vs derived target | **Disable** (G1) or XBot clamped form |
| AdamW | Adam |
| RunningMeanStd obs normalisation (+ freeze/checkpoint machinery, critic warmup) | **Flag, not immediate delete**: zero references use running norm — all use fixed scales + clip — and this machinery has already produced one failure (COUNT_MAX warm-start destruction). Invasive to swap mid-project; if the current run struggles, conforming to fixed scales (dof_vel ×0.05–0.1, ang_vel ×0.25, clip) is the shipped-practice fallback |

**Survivors (measured, keep):** 8 Hz action filter (2% vs 100% falls) — but add random-delay DR alongside; polar elliptical commands (52.7% forward bias measurement); parameterised clock f/s/d observations (Siekmann lineage); exploit lanes (noise-crutch persistence); only-positive total clip (legged_gym-native, unanimous anyway); magnitude ≥ 50% floor (0.45 standing-profitable measurement).

## E) Honest limits

1. **Reward × dt in ours unverified.** All three refs multiply reward scales by dt (0.01/0.02). The OURS extraction shows raw `terms.sum` with no dt mention; whether our weights are per-step-absolute or per-second matters for every weight comparison in this table (a ref weight 1.0 is 0.02/step effective). Verify in `locomotion.py`/`ppo.py` before copying any absolute weight.
2. **YAML plumbing not verified.** YAML vs dataclass disagree on entropy_coef, log_std_max, init_noise_std, command ranges, zero_command_prob, free_gait_prob; YAML treated as effective but the config loader was not read.
3. **Per-joint action_scale values not extracted** — effective action authority vs refs' 0.25 rad unknown; the [-1,1]+scale comparison to T1 is structural, not numeric.
4. **Actuator model equivalence assumed.** Ours uses MuJoCo position servos converted from torque motors in model_prep; refs compute explicit PD torque clipped to effort limits at 500–1000 Hz. Whether our servos enforce torque limits (forcerange) was not verified.
5. **Ours' critic assumed non-privileged** — inferred from networks.py (same obs to both trunks); if a privileged path exists that the extraction missed, row C#1 downgrades to the actor-obs items only.
6. **G1 hip_pos DOF indices** [1,2,7,8] asserted as hip roll/yaw from URDF ordering; URDF not fetched. G1's first `legged_robot.py` fetch was contaminated (wrong repo content) — discarded, SHA-verified re-fetch used; any earlier note citing cfg.safety/256 buckets/terrain curriculum for G1 is bogus.
7. **Dead config in refs** — T1 `symmetric_coef` unused; G1 terrain fields dead. Weights above are the *live* code paths only.
8. Booster T1 repo untagged; values reflect main @ da396a0 on 2026-08-13.
