# Reward ramps, termination curricula, and early-training leniency in shipped legged-RL configs

Research date: 2026-08-13. Sources: public repo configs (treated as ground truth over papers) plus
papers where noted. Scope: what shipped systems actually do about (1) penalty ramps, (2) termination
leniency, (3) reference-state initialisation for velocity tasks, (4) alive-bonus vs penalty balance,
(5) push randomisation phasing.

## 1. Penalty ramps over training

**legged_gym (leggedrobotics, master)** does NOT ramp any penalty. All scales are constant
(action_rate -0.01, dof_acc -2.5e-7, lin_vel_z -2.0, ang_vel_xy -0.05, torques -1e-5). The
early-training leniency mechanism is instead the total-reward clip:

```python
# legged_robot_config.py
only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
# legged_robot.py, compute_reward()
if self.cfg.rewards.only_positive_rewards:
    self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
```

This is a *self-annealing* penalty schedule: early on, summed penalties exceed the tracking reward,
the clip floors the total at 0, and the agent never sees a net-negative step (so falling fast is
never better than trying). As tracking rewards grow, totals go positive and the penalties act at
full strength. No explicit schedule, no gating.

**walk-these-ways (Improbable-AI, train.py overrides)** replaces the hard clip with the Ji et al.
2022 multiplicative form ("ji22 style"):

```python
Cfg.rewards.only_positive_rewards = False
Cfg.rewards.only_positive_rewards_ji22_style = True
Cfg.rewards.sigma_rew_neg = 0.02
# legged_robot.py
self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
```

Total = (sum of positive terms) x exp(sum of negative terms / 0.02). Reward is always >= 0, so
early-termination-seeking is impossible by construction; penalties modulate rather than dominate.

**RSL-RL itself has no reward-curriculum utilities** — it is just the PPO runner. The curriculum
tooling lives in Isaac Lab's curriculum manager:

```python
# isaaclab/envs/mdp/curriculums.py — modify_reward_weight
if env.common_step_counter > num_steps:
    self._term_cfg.weight = weight   # STEP change, not a linear ramp
```

Docs example: `CurriculumTermCfg(func=mdp.modify_reward_weight, params={"term_name": ..., "weight": 0.5, "num_steps": 100_000})`.
Note: the shipped H1/G1 velocity configs do NOT use it — their only curriculum term is
`terrain_levels_vel`. `modify_env_param` / `modify_term_cfg` exist for arbitrary scheduled
parameter changes (also step-wise via a user `modify_fn`).

**HumanoidVerse / ASAP (LeCAR-Lab) — the one true shipped penalty ramp.** Performance-gated,
multiplicative, applied to a named list of penalty terms:

```python
# legged_robot_base.py
if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
    self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
    self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)
self.reward_penalty_scale = np.clip(self.reward_penalty_scale, min_scale, max_scale)
# applied as: if name in reward_penalty_reward_names: rew *= self.reward_penalty_scale
```

`average_episode_length` is an EMA over the last `num_compute_average_epl` resets. Config values
(reward_h1_locomotion.yaml): level_down_threshold=400 steps, level_up_threshold=700 steps,
degree=1e-5 per reset-update, initial scale 1.0 (README uses 0.5 for Genesis runs; ASAP motion
tracking uses initial 0.1 per paper), min 0.0, max 1.0. The ramped list covers regularisers only:
torques, dof_acc, dof_vel, action_rate, feet_contact_forces, stumble, slippage, feet_ori,
in_the_air, dof_pos/vel/torque limits, termination, feet_air_time, feet_max_height. Tracking terms
are never scaled. ASAP training commands ship `rewards.reward_penalty_curriculum=True
rewards.reward_penalty_degree=0.00001`.

**Booster Gym (T1.yaml)**: no ramps; constant scales with a survival bonus (see section 4). Command
curriculum exists but ships `curriculum: false` (update_rate 0.1, 10 lin_vel and 10 ang_vel levels
when on).

**Takeaway for us**: nobody linearly ramps action-rate/energy penalties on a wall-clock schedule.
The two shipped mechanisms are (a) reward-level clipping/multiplicative composition (legged_gym /
walk-these-ways) which is always-on and self-annealing, and (b) HumanoidVerse's episode-length-gated
multiplicative penalty scale. Isaac Lab's step-change `modify_reward_weight` exists but is unused in
the flagship locomotion configs.

## 2. Termination leniency curricula

- **legged_gym**: no leniency curriculum. Termination = contact force > 1.0 N on
  `terminate_after_contacts_on` bodies, plus timeout. Constant throughout.
- **Isaac Lab H1/G1 velocity**: constant terminations (illegal contact on torso_link, threshold
  1.0 N; timeout). No relaxation schedule.
- **walk-these-ways**: adds terminal_body_height=0.05 m and terminal roll/pitch
  (terminal_body_ori=1.6 rad) — constant, not scheduled.
- **Booster Gym T1**: terminate_height 0.45 m, terminate_vel 50 — constant.
- **ASAP motion tracking — the shipped termination curriculum** (tracking-error based, not
  tilt/height): threshold *loosens* when episodes are short and *tightens* when long:
  ```python
  if self.average_episode_length < level_down_threshold:   # 40 steps
      self.terminate_when_motion_far_threshold *= (1 + degree)
  elif self.average_episode_length > level_up_threshold:   # 42 steps
      self.terminate_when_motion_far_threshold *= (1 - degree)
  np.clip(threshold, min, max)
  ```
  Config: initial 1.5 m, max 2.0 m, min 0.25 m (README command overrides min to 0.3,
  degree 2.5e-5; env yaml default degree 2.5e-6), level_down=40 / level_up=42 steps.
  ASAP's plain locomotion env uses fixed gravity-projection termination (x/y > 0.8) and no
  curriculum.

**Takeaway**: for velocity tracking, no shipped system relaxes tilt/height termination early. The
only production termination curriculum (ASAP) is for motion-tracking error, but its recipe
(multiplicative +/- degree per reset, gated on EMA episode length with a tight hysteresis band, hard
min/max clip) is directly transplantable to a tilt threshold.

## 3. Reference-state initialisation for velocity tasks

- **legged_gym**: standing start with heavy joint noise: `dof_pos = default_dof_pos *
  U(0.5, 1.5)` per joint, dof_vel = 0, base lin+ang vel `U(-0.5, 0.5)` (all 6 components).
- **Isaac Lab velocity envs**: same idea — reset_robot_joints position_range (0.5, 1.5) x default,
  velocity_range (0.0, 0.0); reset_base pose x/y +/-0.5 m, yaw +/-3.14, all base velocity components
  U(-0.5, 0.5).
- **Booster Gym T1**: much tamer — dof_pos additive gaussian sigma 0.05, base lin_vel_xy gaussian
  sigma 0.1, base xy +/-1 m.
- **humanoid-gym (XBot-L)**: fixed standing pose (all default joint angles 0), no reset
  randomisation documented.
- **Mid-gait initialisation appears only in the Cassie/OSU line**: "Dynamic Bipedal Maneuvers
  through Sim-to-Real RL" (arXiv 2207.07835) initialises from a bank of poses saved from a
  pre-trained running policy across commanded speeds and gait phases. For clocked policies the
  cheap, standard equivalent is randomising the *clock phase* at reset, not the body state.
- No shipped velocity-tracking config uses mocap-style RSI, and standing starts demonstrably do not
  block learning (every legged_gym-descendant success is a standing start). The load-bearing parts
  are the joint-scale noise (0.5-1.5x) and base-velocity noise, which force the value function to
  generalise off the nominal pose, plus random clock phase when a gait clock exists.

## 4. Alive bonus vs penalty balance / early-termination-seeking

The failure mode is acknowledged in legged_gym's own comment: "avoids early termination problems".
Three shipped defences, usually combined:

1. **Clip total reward at zero** (legged_gym `only_positive_rewards=True`; also humanoid-gym
   XBot with its mostly-positive exponential terms and `only_positive_rewards=True`). Falling early
   can never beat surviving at reward 0.
2. **Multiplicative composition** (walk-these-ways ji22 style, sigma_rew_neg=0.02): reward >= 0
   always; penalties scale the positive term instead of subtracting.
3. **Explicit survival/termination terms**:
   - unitree_rl_gym G1: `alive = 0.15` per step (vs tracking_lin_vel 1.0) on top of the base-class
     zero clip. Small — it only needs to break ties near zero, because the clip already floors the
     total.
   - Booster Gym T1: `survival: 0.25` with NO zero-clip; penalties there are sized so the standing
     policy nets positive (action_rate -1.0 is on the squared normalised delta, etc.).
   - Isaac Lab G1/H1: the inverse construction — no alive bonus, instead
     `termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)`: one -200 hit on
     termination dwarfs any per-step penalty stream a fall could avoid (episode ~20 s at 50 Hz,
     summed shaping penalties are O(10) per episode).
4. **HumanoidVerse penalty curriculum** (section 1) attacks the same failure from the other side:
   if mean episode length < 400 steps, penalties shrink until survival recovers.

Sizing rule of thumb extracted from the configs: either make total reward non-negative by
construction, or make the one-shot termination cost ~1-2x the *episode-summed* shaping penalties
(Isaac Lab's -200), or keep an alive bonus that exceeds the typical per-step summed penalty at the
untrained policy's operating point (G1's 0.15 with penalties clipped, T1's 0.25 unclipped).

## 5. Push randomisation during bootstrap

Nobody phases pushes in over training within a single run; the knob is binary and set per run:

- legged_gym: on from step one — `push_interval_s = 15`, `max_push_vel_xy = 1.0` (sets base xy
  velocity to U(-1,1) every 15 s).
- unitree_rl_gym G1: on from step one, harder — push_interval_s = 5, max_push_vel_xy = 1.5.
- humanoid-gym XBot-L: on from step one, gentle — push_interval_s = 4, max_push_vel_xy = 0.2.
- Isaac Lab velocity: interval event mode, interval_range_s = (10, 15), velocity_range +/-0.5 —
  active from the first step.
- Booster Gym T1: gaussian force pushes (sigma 10 N, torque sigma 2 Nm) every 5 s lasting 1 s —
  from step one.
- walk-these-ways: `push_robots = False` entirely (they rely on friction 0.1-3.0 and motor-strength
  0.9-1.1 randomisation instead).
- ASAP delta-finetune stage: `domain_rand.push_robots=False` — pushes are *removed* for the
  fine-tune, the opposite of a phase-in.

**Takeaway**: on-from-step-one is universal when pushes are used, but magnitudes for humanoids that
must learn from scratch are chosen gentle (0.2 m/s XBot) to moderate (0.5 m/s Isaac Lab); the 1.5
m/s G1 value coexists with the zero-clip and alive bonus. Phasing in is unnecessary if the reward
floor prevents termination-seeking; if we want leniency anyway, the shipped pattern would be the
HumanoidVerse gate (enable/strengthen pushes once EMA episode length crosses a threshold), not a
wall-clock ramp.

## Direct recommendations for our from-scratch final-task run

1. Keep `only_positive_rewards`-style clipping (or the ji22 multiplicative form given our many
   exponential terms) from step one — this is the mechanism that let phase2 through and is the
   field's standard answer to penalties-overwhelm-alive.
2. Add a HumanoidVerse-style penalty scale on regularisers only (action_rate, dof_acc, energy,
   vertical velocity, roll/pitch rate), gated on EMA episode length with our numbers: at 50 Hz
   control and ~10-12 s episodes, thresholds ~level_down 300 / level_up 550 steps, degree 1e-4 per
   reset-update (we have far fewer total steps than IsaacGym runs; 1e-5 would never finish
   annealing in 500M steps at 4096 envs), clip [0.1, 1.0], start at 0.25-0.5.
3. Do not schedule termination leniency; keep fixed tilt/contact termination. If bootstrap stalls,
   port ASAP's multiplicative hysteresis to the tilt threshold rather than inventing a linear ramp.
4. Standing starts with legged_gym-strength noise (dof 0.5-1.5x default, base vel +/-0.5) plus
   uniform random gait-clock phase at reset. No mid-gait RSI needed.
5. Pushes on from step one but at humanoid-gentle magnitude (<= 0.5 m/s equivalent), interval
   ~8-15 s; rely on the reward floor, not a push phase-in.

## Sources

- https://github.com/leggedrobotics/legged_gym — `legged_gym/envs/base/legged_robot_config.py`, `legged_robot.py`
- https://github.com/unitreerobotics/unitree_rl_gym — `legged_gym/envs/g1/g1_config.py`
- https://github.com/isaac-sim/IsaacLab — `manager_based/locomotion/velocity/velocity_env_cfg.py`, `config/g1/rough_env_cfg.py`, `isaaclab/envs/mdp/curriculums.py`
- https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html
- https://github.com/Improbable-AI/walk-these-ways — `scripts/train.py`, `go1_gym/envs/base/legged_robot_config.py`, `legged_robot.py`
- https://github.com/BoosterRobotics/booster_gym — `envs/T1.yaml`
- https://github.com/LeCAR-Lab/HumanoidVerse — `humanoidverse/envs/legged_base_task/legged_robot_base.py`, `config/rewards/loco/reward_h1_locomotion.yaml`
- https://github.com/LeCAR-Lab/ASAP — `humanoidverse/config/env/motion_tracking.yaml`, `env/locomotion.yaml`, README training commands
- https://github.com/roboterax/humanoid-gym — `humanoid/envs/custom/humanoid_config.py`
- Ji et al. 2022, "Concurrent Training of a Control Policy and a State Estimator" (the "ji22" multiplicative reward)
- Siekmann et al. 2021, arXiv:2011.01387 (periodic reward composition); arXiv:2207.07835 (pose-bank initialisation)
