# Clock / periodic-reward gaits trained FROM SCRATCH — what the record says, with actual numbers

Researched 2026-08-13 for the final-task curriculum design. Sources: papers + public repo configs
(repo config treated as truth where they disagree).

## 1. Siekmann et al. (ICRA 2021, arXiv:2011.01387) — the canonical recipe

**All policies trained from scratch.** No warm starts, no reference trajectories.

- Clock mechanics: cycle time φ ∈ [0,1); per-foot swing/stance intervals defined by ratio r
  (swing lasts r, stance 1−r) and cycle offsets θ_left, θ_right. Phase indicators are
  **probabilistic von Mises intervals** — I(φ) = P(A<φ<B) with A~Φ(2πa,κ), B~Φ(2πb,κ). RL is run
  on the **expectation** of the reward, so the indicators are smooth ramps, not binary gates. The
  paper explicitly credits the κ-smoothing at phase boundaries with "more stable and consistent
  learning". **The numeric κ is not published in the paper** (it lives in OSU's unreleased configs).
- Coefficients: c_swing_frc = −1, c_swing_spd = 0; c_stance_spd = −1, c_stance_frc = 0. i.e. swing
  penalizes foot force, stance penalizes foot velocity; nothing else in the clock.
- Measurement kernels (all bounded [0,1]): q_frc = 1−exp(−ω·||F||²/100), q_spd = 1−exp(−2ω·||v||²),
  q_ẋ = 1−exp(−2ω·|ẋ_des−ẋ|), q_orientation = 1−exp(−3(1−(q̂ᵀq_des)²)),
  q_action_diff = 1−exp(−5||a_t−a_{t−1}||), q_torque = 1−exp(−0.05||τ||),
  q_pelvis_acc = 1−exp(−0.10(||ω_pelvis||+||a_pelvis||)).
- **Multi-gait weights (the only published composition):**
  `0.400·R_bipedal + 0.300·R_cmd + 0.100·R_smooth + 0.100·(ω−1)·q_standing + 0.100·(−1)·q_hop_sym + 1`
  Clock : velocity-tracking : smoothness = 4 : 3 : 1, plus a **constant +1 alive bias per step** —
  the total reward is positive nearly everywhere, so penalties shape but never dominate, and early
  termination is never attractive.
- Standing is a region of gait-parameter space (r_swing→0); a sigmoid ω = (1+exp(−50(r_swing−0.15)))⁻¹
  gates velocity/force terms off and a standing cost (action-diff + foot symmetry) on.
- Training: PPO + LSTM (2×128), recurrent critic, mirror loss, fixed exploration std = e⁻¹,
  lr 1e-4, batch 32 trajectories × ≤300 steps (40 Hz ⇒ 7.5 s), buffer 50k, 4 epochs/iter.
  **150M samples, 24–36 h** per policy on a CPU cluster, cassie-mujoco-sim. Time-to-first-walking
  is not broken out; 150M samples is the full budget for a sim-to-real-robust policy.
- **Curriculum: none.** Single-gait: r, θ constant all of training. Multi-gait: ratios sampled over
  a range (range unpublished), offsets fixed; both varying = failure ("asymmetric walking instead of
  hopping"), fixed by always-on *transition penalties* (hop-symmetry, standing cost), not by staging.
  Commands: ẋ_des, ẏ_des "randomized during training" (ranges unpublished). Dynamics randomization
  from step one (damping 0.3–4.0×, mass 0.5–1.5×, friction 0.35–1.1, slope ±0.03 rad, encoder ±0.05 rad).
- **No ramp of any reward weight anywhere. Full strength from the first sample.**

## 2. Successor systems (repo-config truth)

### rohanpsingh/LearningHumanoidWalking (JVRC-1 / HRP-5P / H1, MuJoCo CPU — closest analog to this project)
- Clock: OSU-style but **PCHIP monotone spline** ramps instead of von Mises, built over one cycle
  with `strict_relaxer = 0.1` (each phase boundary relaxed by 10% of that phase's length —
  effectively the κ analog). Stance mode "grounded". swing_duration 0.75 s?? — config says
  `swing_duration: 0.75, stance_duration: 0.35, total_duration: 1.1 s`, control_dt 0.025 (40 Hz).
- Weights (sum = 1.0): foot_frc_score 0.225, foot_vel_score 0.225 (**clock = 45%**),
  com_vel_error 0.150 + yaw_vel_error 0.150 (**cmd = 30%**), root_accel 0.05, height_error 0.05,
  upper_body 0.05, posture 0.05, torque 0.025, action 0.025. Same 4:3 clock:cmd ratio as Siekmann.
- Standing mode: clock overridden to constants (r_frc = 1, r_vel = −1 ⇒ reward ground contact,
  penalize foot motion). Termination: pelvis z < 0.6 or > 1.4 m, or self-collision. From scratch, PPO.

### walk-these-ways (Margolis & Agrawal, Go1 — quadruped but the definitive public clock config)
- Soft indicators: desired contact states smoothed by **Normal(0, κ).cdf with κ = kappa_gait_probs = 0.07**
  (transition width ≈ 7% of cycle). Policy observes clock inputs (`observe_clock_inputs = True`).
- Reward (train.py overrides, not the base-config zeros):
  `tracking_contacts_shaped_force = 4.0` (penalty of (1−C_des)·(1−exp(−F²/σ)), σ_force = 100),
  `tracking_contacts_shaped_vel = 4.0` (C_des·(1−exp(−v²/σ)), σ_vel = 10),
  tracking_lin_vel 1.0, tracking_ang_vel 0.5, `feet_air_time = 0.0` (**air-time reward removed once
  the clock is present**). Clock terms are 4× the velocity term — full strength from step one.
- Gait commands sampled: frequency **[2.0, 4.0] Hz**, phase/offset [0,1], duration (stance fraction)
  fixed 0.5, footswing height [0.03, 0.35] m.
- `only_positive_rewards_ji22_style = True`: total = rew_pos · exp(rew_neg/σ) (Ji et al. 2022) — negative
  terms can only scale down positive reward, never make the sum negative.
- **Curriculum on commands, not on rewards**: velocity/gait command box grows only when tracking
  performance crosses thresholds — lin_vel 0.8, ang_vel 0.7, **contact-force AND contact-vel
  tracking ≥ 0.90** — i.e. the gait must be clean before commands get harder.

### humanoid-gym / roboterax (XBot-L, from scratch, Isaac Gym→MuJoCo transfer)
- `cycle_time = 0.64 s`. Phase from episode time; stance mask from **sin(2πφ)**: left stance when
  sin ≥ 0, right when sin < 0, **both in stance when |sin| < 0.1** (explicit double-support band ≈
  3.2% of cycle each crossing). Hard mask, but the ±0.1 sin band is the softening.
- Weights: joint_pos 1.6 (phase-indexed reference knee/hip/ankle targets, scale 0.17/0.34 rad,
  kernel exp(−2‖Δq‖)−0.2‖Δq‖), feet_contact_number **+1.2** (per-foot +1.0 if contact==mask else −0.3),
  feet_clearance 1.0 (target_feet_height 0.06 m, ±1 cm tolerance during swing), feet_air_time 1.0
  (**air time clamped to 0.5 s**, paid at first contact), tracking_lin_vel 1.2, tracking_ang_vel 1.1,
  orientation 1.0, base_height 0.2 (0.89 m), foot_slip −0.05, action_smoothness −0.002,
  torques −1e-5, dof_vel −5e-4, dof_acc −1e-7, collision −1.
- `only_positive_rewards = True` (legged_gym clip: total clipped at ≥ 0). **No curriculum**
  (command curriculum disabled; commands x [−0.3,0.6], y ±0.3, yaw ±0.3, resample 8 s). All terms
  full strength from iteration 0.

### unitree_rl_gym (G1, the config Unitree ships)
- Fixed clock: `period = 0.8 s`, right leg offset 0.5; obs gets sin(2πφ), cos(2πφ).
- Contact reward is **positive and hard**: is_stance = leg_phase < 0.55 (stance = 55% of half-cycle);
  reward += ~(contact XOR is_stance), weight **contact = 0.18**, plus `alive = 0.15`/step.
- feet_swing_height −20.0 on (z − **0.08 m**)² during swing, `feet_air_time = 0.0` (clock replaces it),
  base_height −10, orientation −1, hip_pos −1, contact_no_vel −0.2, action_rate −0.01,
  dof_acc −2.5e-7, tracking_lin_vel 1.0, tracking_ang_vel 0.5. LSTM-64 policy. No reward ramps.

### Booster Gym (T1 humanoid, from scratch, sim-to-real validated)
- Height-based swing reward instead of forces: `feet_swing = 3.0`, `swing_period: 0.2 s`,
  **gait_frequency sampled [1.0, 2.0] Hz**; survival 0.25/step; tracking_lin_vel_x 1.0, _y 1.0,
  tracking_ang_vel 0.5 (σ=0.25); base_height −20, orientation −5, action_rate −1, feet_slip −0.1,
  feet_distance −1 (ref 0.2 m), dof_acc −1e-7. `only_positive_rewards: true`, `curriculum: false`,
  resample commands every 8–12 s. Full-strength clock from step one.

### Berkeley Humanoid (HybridRobotics/isaac_berkeley_humanoid, RSL-RL) — the no-clock counterexample
- **No gait clock at all.** Bipedal gait shaped by `feet_air_time` weight **2.0** with
  `threshold_min = 0.2, threshold_max = 0.5` s + feet_slide −0.25, tracking 1.0/0.5 (σ²=0.25).
  Curriculum on terrain, push force, and command velocity — never on reward weights.

### Isaac Lab official G1/H1 velocity task
- No clock; `feet_air_time_positive_biped`, weight 0.25, **threshold 0.4 s** (rewards min(air, contact)
  time in single stance, so double-stance camping earns nothing), feet_slide −0.1,
  termination penalty −200, tracking 1.0/2.0 with σ=0.5.

### ASAP / HumanoidVerse-lineage loco config (LeCAR, G1)
- No clock in the base loco reward: feet_air_time 1.0, feet_height_target 0.12 m,
  penalty_close_feet_xy −10, base_height −10, slippage −1, `only_positive_rewards: False`.

## 3. Does anyone ramp the clock weight? **No.**
Across Siekmann, LearningHumanoidWalking, walk-these-ways, humanoid-gym, unitree G1, Booster T1:
clock/periodic terms are **full strength from the first environment step**. What *is* staged:
- Command difficulty (WTW: box grows gated on tracking ≥ 0.7–0.9; Berkeley: modify_command_velocity
  curriculum; blind-stairs Cassie: terrain).
- Gait-conditioned multi-phase curriculum (arXiv 2505.20619, G1): stages **tasks** (Phase 1 walk
  0–2 m/s → Phase 2 stand + walk↔stand transitions → Phase 3 run to 4 m/s) and uses
  **gait-conditioned reward masking** ("route only the relevant reward components"), i.e. on/off
  routing per commanded mode — still not weight ramps.
The bootstrap-friendliness lever is instead: (a) bounded [0,1] kernels + a constant alive bias
(Siekmann β=1; unitree 0.15; Booster 0.25), and/or (b) total-reward clipping —
`only_positive_rewards` (legged_gym/humanoid-gym/Booster) or Ji22 multiplicative
(WTW) — so penalty terms cannot go net-negative and suppress exploration.

## 4. Clock vs early flailing — what the evidence says
- The clock **helps** the stand→step transition; it is the densest anti-standing signal available.
  During the commanded swing window, a planted foot bleeds reward every step (force penalty active
  ~half the cycle per foot), so "stand still" is a reward valley, not a local optimum. Siekmann calls
  the auxiliary costs constraints that guide "viable exploration towards stable locomotion" — the
  clock is the exploration guide, not a suppressor. Nobody in this lineage reports the clock
  *hindering* gait emergence; the reported failure modes are of the *multi-gait* kind (gait
  blending/asymmetric hopping), fixed with transition penalties.
- The flailing-suppression fear is real but it attaches to the **smoothness/torque penalties**, not
  the clock — and every from-scratch system neutralizes it structurally rather than by ramping:
  Siekmann's smoothness block is only 0.100 of a ~1.9-max reward with kernels bounded [0,1];
  legged_gym-family sets `only_positive_rewards` so a flailing recovery step can never earn
  negative total reward; WTW's Ji22 form multiplies rather than subtracts. Your phase2 success
  already followed this shape.
- Two standing-related gotchas the record does flag: (i) with a clock and zero command you must
  define standing explicitly (Siekmann's ω-gate + standing cost; LearningHumanoidWalking's constant
  clock override; 2505.20619's dedicated Phase 2) or the policy marches in place; (ii) pure
  air-time rewards without a clock have the both-feet-down camping optimum — Isaac Lab's
  `feet_air_time_positive_biped` and command-gating (reward only when ‖cmd‖ > 0.1) exist precisely
  to patch that.
- Smooth indicator edges matter for early learning: Siekmann's κ smoothing "usefully encourage[s]
  more stable and consistent learning"; WTW uses κ = 0.07 cdf smoothing; LearningHumanoidWalking
  relaxes each boundary by 10%; humanoid-gym keeps a |sin| < 0.1 double-support band. Nobody ships
  razor-sharp binary gates on force penalties; unitree's hard XOR gate works because it is a small
  *positive* match bonus (0.18), not a penalty.

## 5. Air-time / swing-duration targets
- legged_gym baseline: rew = Σ(t_air − **0.5 s**)·first_contact, weight 1.0, **gated ‖cmd‖ > 0.1**;
  paid only at touchdown. Anymal-class default, inherited everywhere.
- humanoid-gym: same first-contact form, air time **clamped to 0.5 s**, weight 1.0 — coexists with the
  contact-mask clock at cycle_time 0.64 s (so a nominal step ≈ 0.32 s; the 0.5 cap is a ceiling not a target).
- Isaac Lab G1/H1: threshold **0.4 s**, weight 0.25, positive-biped form. Berkeley Humanoid:
  threshold_min **0.2**, threshold_max **0.5**, weight 2.0 (their main gait-shaping term, no clock).
- Never phased in anywhere — active from step one, but always command-gated and/or capped.
- Systems with a real clock either **zero the air-time reward** (WTW, unitree G1) or keep it as a
  redundant small bonus (humanoid-gym): the clock's swing window already dictates swing duration
  (stance fraction × period), so a separate air-time target is mostly a tuning liability.

## Implications for the final-task curriculum (this repo)
1. Train the clock at full weight from iteration 0; do not ramp it. Ramping is unprecedented in
   every shipped system; the from-scratch failure risk lives elsewhere (net-negative reward, sharp
   gates, overwide command box at t=0).
2. Copy the Siekmann proportions the project's phase2 implicitly matched: clock ≈ 40%, velocity
   ≈ 30%, smoothness ≈ 10% of max-attainable, plus a constant alive bias; keep every kernel bounded
   [0,1]; clip total reward at ≥ 0 (or Ji22 multiplicative) so recovery flailing is never net-punished.
3. Soften indicator edges: von Mises/cdf width ≈ 0.07 of cycle, or 10% boundary relaxation —
   equivalently a smooth double-support band ≈ 0.1 of cycle. Penalize force-in-swing and
   velocity-in-stance only; make any hard-gated term a positive match bonus.
4. Put the curriculum on the **command distribution** (velocity box, resample rate, gait-frequency
   range), gated on gait-tracking quality (WTW gates expansion on contact-tracking ≥ 0.9), and on
   pushes/terrain — never on reward weights. Start near phase2's command box, expand toward the
   full polar set.
5. Fix the clock parameters early in training (single f ≈ 1.25 Hz — matches unitree G1's 0.8 s
   period; stance fraction 0.5–0.55) and widen the sampled range (0.6–1.35 Hz) only after
   breakthrough, WTW-style. Sampling the full clock-parameter range from step 0 is what WTW does at
   4096 envs on a quadruped, but Siekmann's multi-gait experience says humanoid gait blending is the
   risk — stage the range, not the weight.
6. Define standing/zero-command behavior explicitly from the start (ω-gate or constant-clock
   override), or exclude zero commands until locomotion is stable.
