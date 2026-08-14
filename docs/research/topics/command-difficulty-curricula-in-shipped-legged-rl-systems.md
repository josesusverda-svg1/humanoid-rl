# Command/difficulty curricula in shipped legged-RL systems — exact rules as implemented

Status: `good` · Confidence: `high` (all rules read directly from repo source, 2026-08-13)

Scope: the update rules that shipped systems actually run — legged_gym/RSL-RL, Isaac Lab,
walk-these-ways (+ its rapid-locomotion precursor), Booster Gym, humanoid-gym (XBot-L),
HumanoidVerse, Unitree RL gym. Repo config is treated as ground truth over papers.

---

## 1. legged_gym / RSL-RL command curriculum (leggedrobotics/legged_gym, master)

`legged_robot.py::update_command_curriculum` — verbatim:

```python
def update_command_curriculum(self, env_ids):
    """ Implements a curriculum of increasing commands """
    if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length \
            > 0.8 * self.reward_scales["tracking_lin_vel"]:
        self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                      -self.cfg.commands.max_curriculum, 0.)
        self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5,
                                                      0., self.cfg.commands.max_curriculum)
```

Call site (`reset_idx`):

```python
if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
    self.update_command_curriculum(env_ids)
```

Mechanics, decoded:
- **Gate metric**: mean *per-step* `tracking_lin_vel` reward of the envs being reset,
  computed as `episode_sums / max_episode_length`. Note `reward_scales` have already been
  multiplied by dt in `_prepare_reward_function`, so the condition is literally
  "average tracking reward ≥ 80% of the per-step maximum achievable"
  (tracking reward is `exp(-err²/0.25)`, so 0.8 corresponds to vel error ≈ 0.24 m/s).
- **Update**: only `lin_vel_x`; both ends grow **simultaneously and symmetrically** by
  0.5 m/s, clipped to ±`max_curriculum`. `lin_vel_y` and `ang_vel_yaw` never grow.
- **Cadence**: evaluated only when `common_step_counter % max_episode_length == 0`
  (~once per 20 s episode window), on the envs that happen to reset then.
- **No demotion.** Ranges only expand.
- Defaults: `curriculum = False`, `max_curriculum = 1.0`, ranges x/y/yaw all [-1, 1],
  `resampling_time = 10.`, `heading_command = True`, `tracking_sigma = 0.25`,
  scales: tracking_lin_vel 1.0, tracking_ang_vel 0.5.
- **No shipped config in the repo actually enables it** (anymal_c_rough, cassie etc. don't
  override `commands.curriculum`). The famous curriculum in this codebase is the terrain
  one (§5). RSL-RL itself (the PPO library) contains no curriculum logic.

## 2. Isaac Lab (isaac-sim/IsaacLab, main)

Velocity tasks (`manager_based/locomotion/velocity/`):
- `CommandsCfg`: `UniformVelocityCommandCfg`, ranges lin_vel_x (-1,1), lin_vel_y (-1,1),
  ang_vel_z (-1,1), heading (-π,π), `resampling_time_range = (10., 10.)`,
  `rel_standing_envs = 0.02` (2% of resamples command standstill), `rel_heading_envs = 1.0`
  (yaw always derived from heading error). Rewards: track_lin_vel_xy_exp weight 1.0,
  track_ang_vel_z_exp weight 0.5, std = sqrt(0.25).
- `CurriculumCfg` contains **exactly one term**: `terrain_levels = mdp.terrain_levels_vel`
  (same promotion/demotion rule as legged_gym, §5). **No command curriculum ships for any
  velocity task.**
- **Humanoids are forward-biased permanently, no curriculum**: both
  `config/h1/rough_env_cfg.py` and `config/g1/rough_env_cfg.py` override
  `lin_vel_x = (0.0, 1.0)`, `lin_vel_y = (0.0, 0.0)` [G1: (-0.0, 0.0)],
  `ang_vel_z = (-1.0, 1.0)`. Forward-only training, zero lateral; they never fixed the
  asymmetry because the demo never asks for backward walking.
- Generic hooks that exist for building your own: `isaaclab/envs/mdp/curriculums.py` has
  `modify_reward_weight(term_name, weight, num_steps)` (fires once
  `common_step_counter > num_steps` — pure step schedule, no performance gate) and
  `modify_term_cfg` / `modify_env_param` (dotted-path runtime mutation, e.g. address
  `"commands.base_velocity.ranges.lin_vel_x"`, with a user `modify_fn`). So Isaac Lab's
  official pattern for command curricula is *step-scheduled range widening*, not
  performance-gated.

## 3. walk-these-ways (Improbable-AI, master) — grid-adaptive curriculum

The only shipped system with a genuinely *directional* command curriculum. 15-dim command
space; the velocity dims are discretized into a bin grid:

Config (`scripts/train.py`, Go1):
- initial region: lin_vel_x [-1, 1], lin_vel_y [-0.6, 0.6], ang_vel_yaw [-1, 1]
- limits (final envelope): limit_vel_x [-5, 5], limit_vel_y [-0.6, 0.6], limit_vel_yaw [-5, 5]
- bins: x 21, y 1, yaw 21 → bin width ≈ 0.476 m/s (x), 0.476 rad/s (yaw)
- thresholds (fraction of each reward's per-step max):
  tracking_lin_vel 0.8, tracking_ang_vel 0.7,
  tracking_contacts_shaped_vel 0.90, tracking_contacts_shaped_force 0.90
- `gaitwise_curricula = True` (a separate curriculum instance per gait category)

Update rule (`curriculum.py::RewardThresholdCurriculum.update`), verbatim core:

```python
is_success = 1.
for task_reward, success_threshold in zip(task_rewards, success_thresholds):
    is_success = is_success * (task_reward > success_threshold)   # AND over all 4 metrics
self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
adjacents = self.get_local_bins(bin_inds[is_success], ranges=local_range)
for adjacent in adjacents:
    self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)
```

- **Gate metric**: per-episode `command_sums[key] / ep_len` (average per-step reward for
  the command the env just executed) compared against
  `curriculum_thresholds[key] * reward_scales[key]`. ALL four tracking metrics must pass.
- **Update**: the succeeded bin gets weight +0.2 (cap 1.0), and every bin within
  `local_range` (0.55 in command units for x/y/yaw — i.e. immediate neighbors) also gets
  +0.2. Weights start at 1.0 inside the initial region, 0 outside; **never decay**.
- **Sampling**: bins drawn with probability ∝ weights (normalized), then a uniform sample
  inside the chosen bin's cell. Update happens at command resampling time (on reset),
  using the previous episode's performance of that env.
- Effect: the practiced region is a growing *frontier* — expansion happens only in
  directions where the policy actually succeeds, per bin. This is the "per-sector
  curriculum" pattern. Precursor repo rapid-locomotion-rl (Margolis, RSS'22): same idea,
  `num_lin_vel_bins = 20`, `lin_vel_step = 0.3`, limit_vel_x [-10, 10] (limit_vel_y ±0.6),
  forward_curriculum_threshold 0.8, yaw_curriculum_threshold 0.5.

## 4. Booster Gym (BoosterRobotics/booster_gym, main) — signed 2-D level grid, humanoid T1

Shipped default is `curriculum: false` (T1 walks sim2real with the *static* ranges below),
but the implemented mechanism is the cleanest template for symmetric growth:

Config (`envs/T1.yaml` commands section, verbatim):

```yaml
num_commands: 3
still_proportion: 0.1
lin_vel_x: [-1.0, 1.0]
lin_vel_y: [-1.0, 1.0]
ang_vel_yaw: [-1, 1]
resampling_time_s: [8., 12.]
gait_frequency: [1.0, 2.0]
curriculum: false
update_rate: 0.1
lin_vel_levels: 10
ang_vel_levels: 10
lin_vel_x_resolution: 0.2
lin_vel_y_resolution: 0.1
ang_vel_resolution: 0.2
episode_length_toler: 0.1
lin_vel_x_toler: 0.4
lin_vel_y_toler: 0.2
ang_vel_yaw_toler: 0.2
```

Mechanism (`envs/t1.py`):
- Probability grid of shape `(1+2*lin_vel_levels, 1+2*ang_vel_levels)` = 21×21, **signed
  levels −10..+10 on each axis, only the center cell (0,0) starts at prob 1.0**.
- **Success gate** (per env, at resample): episode survived
  `episode_length > max_episode_length * (1 - 0.1)` AND filtered |v_x − cmd_x| < 0.4 AND
  |v_y − cmd_y| < 0.2 AND |ω_z − cmd_yaw| < 0.2.
- **Update**: on success, current cell **and its 4 neighbors** get `prob += 0.1`
  (clamped to 1.0). No decay/demotion.
- **Sampling**: `torch.multinomial` over the flattened grid;
  `cmd_x = (lin_vel_level + U(-0.5, 0.5)) * 0.2` (so level 10 → 2.0 m/s + jitter, later
  clipped by the range), `cmd_yaw = (ang_vel_level + U(-0.5, 0.5)) * 0.2`,
  and crucially `cmd_y = |lin_vel_level| * U(-1, 1) * 0.1` — **lateral magnitude scales
  with the same radial level, sign uniform**, so forward/backward/lateral difficulty grows
  in lock-step from the center outward. 10% of resampled envs are zeroed (stand still).

## 5. Terrain curriculum — the promotion/demotion template (legged_gym & Isaac Lab, identical rule)

`legged_robot.py::_update_terrain_curriculum` / Isaac Lab `mdp.terrain_levels_vel`:

```python
distance = torch.norm(root_states[env_ids, :2] - env_origins[env_ids, :2], dim=1)
move_up   = distance > terrain.env_length / 2
move_down = (distance < torch.norm(commands[env_ids, :2], dim=1) * max_episode_length_s * 0.5) * ~move_up
terrain_levels[env_ids] += 1 * move_up - 1 * move_down
# at max level: resample a random level (uniform 0..max) instead of clipping
terrain_levels[env_ids] = torch.where(terrain_levels >= max_terrain_level,
                                      torch.randint_like(terrain_levels, max_terrain_level),
                                      torch.clip(terrain_levels, 0))
```

Template properties worth stealing (Rudin et al. CoRL'21 "game-inspired curriculum"):
- Evaluated **per env at its own reset** — no global gate, thousands of independent
  promotions/demotions per iteration.
- **Promote** on achievement (walked > half the tile), **demote** on clear failure
  (covered < 50% of what the commanded velocity implied over the full episode) — the
  `~move_up` guard makes the two exclusive.
- **Graduation recycling**: envs that clear the top level are re-dealt a *random* level,
  keeping easy data in the mix forever (prevents forgetting + keeps the level distribution
  from collapsing onto max difficulty).
- Skipped during initialization (`if not self.init_done: return`).

## 6. Systems that ship NO command curriculum at all (static ranges, real-robot results)

- **humanoid-gym / XBot-L (roboterax)**: lin_vel_x [-0.3, 0.6], lin_vel_y [-0.3, 0.3],
  ang_vel_yaw [-0.3, 0.3], heading ±π, resampling_time 8 s, episode 24 s,
  terrain curriculum False. Mildly forward-biased static box; walks on the real robot.
- **HumanoidVerse (LeCAR)**: x/y/yaw all [-1, 1], heading ±π, resample 10 s. No curriculum
  code in `envs/locomotion/locomotion.py` — uniform static sampling.
- **Unitree unitree_rl_gym (G1/H1)**: no commands override at all → inherits legged_gym
  defaults ([-1,1] everywhere, curriculum False).
- **Isaac Lab H1/G1**: see §2 — static AND forward-only.

Takeaway: for modest envelopes (≤1 m/s-ish) shipped humanoids skip command curricula
entirely; curricula appear when the envelope is aggressive (rapid-locomotion 10 m/s grid,
Booster's 2 m/s grid) or the command space is huge (walk-these-ways 15-dim).

## 7. Direction coverage under a curriculum — what the evidence supports

The measured failure mode here (52.7% forward practice from a forward-biased envelope)
maps onto the shipped designs like this:

- **legged_gym** avoids asymmetry *within x* trivially: both ends of lin_vel_x grow by the
  same 0.5 at the same time off one aggregate gate. But the gate is the *mean* tracking
  reward over all commands — the easy (slow/forward) commands subsidize the gate, so range
  growth does not certify backward competence; it just stays symmetric by construction.
- **walk-these-ways** is per-bin: probability mass only appears where success already
  happened. Mid-training the practiced set IS asymmetric (frontier grows faster in easy
  directions), but weights never exceed 1.0 and never decay, so mastered easy bins stop
  gaining relative mass as the frontier unlocks; the asymmetry is transient, not baked in.
- **Booster Gym** is the "uniform scale" pattern: a single radial level index expands an
  ellipse-shaped shell (x: level*0.2, y: |level|*0.1, yaw independent axis) symmetrically
  from zero. Direction mix is constant at every stage of the curriculum by construction.
- **Isaac Lab humanoids** simply accept the bias (forward-only ranges) — the anti-pattern
  for this project's equal-8-sector requirement.

**Ready-to-implement recommendation for this project** (polar commands, envelope
fwd 1.5 / back 0.8 / lat 0.6):
1. Sample direction θ uniform over sectors exactly as designed (fixes the practice mix at
   12.5%/sector at *every* curriculum stage), then set magnitude
   `v = s · r_env(θ)` with `r_env` the direction-dependent envelope (the ellipse-ish hull
   through 1.5/0.6/0.8) and `s` the single curriculum scalar — this is Booster's radial
   level generalized to the polar envelope, and it can never re-introduce the 52.7% skew.
2. Drive `s` with the terrain-curriculum template, not legged_gym's mean-reward gate:
   per-env at reset, promote `s_env += Δ` if the env survived ≥ 90% of max episode length
   AND per-axis filtered tracking error is inside Booster-style tolerances
   (0.4 / 0.2 / 0.2 scaled to this robot); demote on early fall. Keep per-sector scalars
   `s[k]` (8 of them, walk-these-ways-style) only if measured sector competence diverges;
   gate each sector's growth on ITS OWN tracking, never the mean.
3. Start s covering ≈ 0.3–0.5 m/s (all shipped cold starts begin at or below ~1 m/s
   envelopes; Booster starts at literal 0 ± half a cell), step Δ ≈ 0.05–0.1 (Booster's one
   cell = 0.2 m/s; legged_gym 0.5 m/s is coarse), cap at 1.0.
4. Recycle graduates: once `s = 1`, resample that env's scale from U(0.3, 1.0) (terrain
   curriculum's randint trick) so slow/easy commands stay in the data mix.
5. Keep gait-clock ranges (f 0.6–1.35 Hz etc.) OUT of the difficulty scalar initially —
   walk-these-ways is the only system that curricularizes gait parameters, and it does so
   with the same bin machinery, not by shrinking ranges; simplest port: start f range
   narrow around the comfortable frequency and widen it on the same `s` schedule.

## Sources (all read 2026-08-13)

- https://github.com/leggedrobotics/legged_gym — envs/base/legged_robot.py, legged_robot_config.py
- https://github.com/isaac-sim/IsaacLab — velocity/mdp/curriculums.py, velocity_env_cfg.py, config/h1/rough_env_cfg.py, config/g1/rough_env_cfg.py, isaaclab/envs/mdp/curriculums.py
- https://github.com/Improbable-AI/walk-these-ways — go1_gym/envs/base/curriculum.py, legged_robot.py, scripts/train.py
- https://github.com/Improbable-AI/rapid-locomotion-rl — mini_gym/envs/base/legged_robot_config.py
- https://github.com/BoosterRobotics/booster_gym — envs/T1.yaml, envs/t1.py
- https://github.com/roboterax/humanoid-gym — humanoid/envs/custom/humanoid_config.py
- https://github.com/LeCAR-Lab/HumanoidVerse — config/env/locomotion.yaml, envs/locomotion/locomotion.py
- https://github.com/unitreerobotics/unitree_rl_gym — legged_gym/envs/g1/g1_config.py
