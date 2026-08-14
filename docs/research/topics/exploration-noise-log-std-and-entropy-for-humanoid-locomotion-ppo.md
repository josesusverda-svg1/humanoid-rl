# Exploration noise, log_std handling and entropy bonus for humanoid locomotion PPO

status: `defect confirmed`, confidence: `high`
date: 2026-08-13
scope: what `init_noise_std`, `log_std` clamping and `entropy_coef` should be for this
project's AMP + velocity-command humanoid, measured against what legged_gym / RSL-RL /
Isaac Lab / IsaacGymEnvs-AMP / ASE / MimicKit actually configure.

---

## 1. Verdict

`std = 1.0`, frozen, is a defect — but not for the reason the ceiling comment in
`networks.py` gives. Two independent things are wrong:

1. **The clamp is a one-way trapdoor.** `torch.clamp` passes gradient *at* the boundary and
   zero *above* it. `log_std` is initialised at exactly `log(1.0) = 0.0 = log_std_max`, and
   the entropy bonus pushes it up. Verified with autograd:

   | `log_std` | `d(clamp(log_std,-5,0))/d(log_std)` |
   |---|---|
   | `-1e-9`   | 1.0 |
   | `+0.0`    | 1.0 |
   | `+1e-9`   | **0.0** |
   | `+0.0035` (this run) | **0.0** |

   One optimizer step past zero and the parameter is dead forever. There is no recovery
   path: not the entropy bonus, not the return signal, not the KL-adaptive LR. Setting
   `log_std_max` in the YAML has no effect on such a parameter either, so the knob looks
   live and is not.

2. **1.0 is the wrong operating point for this action space**, because unlike every
   reference implementation this repo *hard-clips the sampled action to [-1, 1]*
   (`humanoid_rl/envs/vec_env.py:521`, `np.clip(actions, -1.0, 1.0, out=s.action)`).

## 2. What the reference implementations actually configure

### 2a. The legged-locomotion line (large learned std, no ceiling, decays)

| system | `init_noise_std` | parameterisation | clamp | `entropy_coef` | action clip | action scale |
|---|---|---|---|---|---|---|
| legged_gym `LeggedRobotCfgPPO` | 1.0 | `nn.Parameter(std)` | **none** | 0.01 | `clip_actions = 100.` | 0.5 rad |
| Isaac Lab ANYmal-C rough | 1.0 | as above | **none** | 0.005 | `ActionTermCfg.clip = None` | 0.5 rad |
| Isaac Lab **G1** rough (humanoid) | 1.0 | as above | **none** | 0.008 | none | 0.5 rad |
| Isaac Lab **H1** rough (humanoid) | 1.0 | as above | **none** | 0.01 | none | 0.5 rad |

`rsl_rl/modules/actor_critic.py` (v2.3.1):

```python
if self.noise_std_type == "scalar":
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
elif self.noise_std_type == "log":
    self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
...
std = self.std.expand_as(mean)          # no clamp anywhere
```

So `init_noise_std = 1.0` **is** standard — as an *initial* value of an *unbounded* learned
parameter that is expected to fall. Nvidia's own Isaac Lab H1 tutorial prints
`Mean action noise std: 1.04` at iteration 15 — it goes **above** 1.0 early, exactly the
motion this repo's ceiling forbids, before decaying (RSL-RL logs of converged runs sit
around 0.2). The entropy bonus at 0.005–0.01 is a *floor* against premature collapse in
those setups, never a ceiling, because nothing bounds the parameter from above.

### 2b. The motion-imitation line (small **fixed** std, no entropy bonus)

This is the lineage this project is actually in.

| system | std | learned? | `entropy_coef` | bounds/`action_bound_weight` |
|---|---|---|---|---|
| AMP paper (Peng et al. 2021), §6.2 | manually specified | **no** | — | — |
| IsaacGymEnvs `HumanoidAMPPPO.yaml` | `sigma_init const -2.9` → **0.055** | `learn_sigma: False` | **0.0** | **10** |
| ASE `amp_humanoid.yaml` | `const -2.9` → **0.055** | `learn_sigma: False` | **0.0** | **10** |
| MimicKit (Peng, 2025) `amp_humanoid_agent.yaml` | `action_std: 0.05` | `actor_std_type: FIXED` | **0.0** | **10.0** |
| MimicKit `amp_g1_agent.yaml` (real Unitree G1) | `action_std: 0.05` | `FIXED` | **0.0** | **10.0** |
| MimicKit `deepmimic_g1_ppo_agent.yaml` | `action_std: 0.05` | `FIXED` | **0.0** | **10.0** |
| MimicKit `amp_go2_agent.yaml` (quadruped) | `action_std: 0.1` | `FIXED` | **0.0** | **10.0** |

AMP paper, §6.2, verbatim: the covariance values "are manually-specified and kept fixed
over the course of training."

IsaacGymEnvs AMP network builder, when `learn_sigma: False`:

```python
self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False,
                                      dtype=torch.float32), requires_grad=False)
```

MimicKit `StdType.FIXED` is likewise `requires_grad=False`.

**This is where `bounds_loss_coef: 10.0` in `configs/amp.yaml` came from.** It is
MimicKit/ASE/IsaacGymEnvs `action_bound_weight: 10`, on a normalised `[-1, 1]` action space,
with the mean penalised quadratically outside `±1` — identical to `ppo.py`. In every one of
those systems the accompanying std is **0.05** and the entropy weight is **0.0**. This repo
imported the bound weight and paired it with std 1.0 and `entropy_coef 0.003`.

### 2c. Nobody hard-clips the sampled action at ±1

- legged_gym: `normalization.clip_actions = 100.` — effectively unbounded.
- Isaac Lab: `ActionTermCfg.clip: dict | None = None`, and the G1/H1/ANYmal velocity envs
  do not set it.
- IsaacGymEnvs `VecTask`: `self.clip_actions = config["env"].get("clipActions", np.Inf)`,
  and `HumanoidAMP.yaml` has **no** `clipActions` key → `np.Inf`.
- rl_games' bound loss uses `soft_bound = 1.1`, not 1.0 — deliberately outside the range.

This repo clips at exactly ±1 *and* runs σ = 1.0. That combination is unique to it.

## 3. What σ = 1.0 costs on this action space, quantified

`action_scale = 0.6 × joint half-range`, per joint (`model_prep.prepare`, computed on the
actual model): min 0.314 rad, median 0.681 rad, max 1.309 rad.

| σ | median joint-target noise | hip_y | shoulder_x | dims clipped per step (of 28, μ=0) | pure-noise `action_rate` tax |
|---|---|---|---|---|---|
| **1.0 (current)** | **39.0°** | 60.0° | 75.0° | **8.88** | **−0.289 / step** |
| 0.5 | 19.5° | 30.0° | 37.5° | 1.27 | −0.129 |
| 0.4 | 15.6° | 24.0° | 30.0° | 0.35 | −0.088 |
| 0.25 | 9.8° | 15.0° | 18.8° | 0.002 | −0.035 |
| legged_gym σ=1.0 @ scale 0.25–0.5 | 14.3–28.6° | — | — | 0 (no clip) | — |
| MimicKit σ=0.05 @ scale 1.4× half-range | ~4.2° at a 60° joint | — | — | 0 (no clip) | — |

Radian-equivalent of MimicKit's humanoid/G1 setting in *this* repo's units:
`0.05 × 1.4 / 0.6 ≈ 0.12`. The current run is **8.6×** that.

Three separate consequences, all measurable in the live run's `metrics.jsonl`:

- **The `action_rate` penalty is 100% noise.** Logged `reward/action_rate = −0.2777`;
  the analytic floor from iid N(0, 1) noise through the ±1 clip is **−0.289**. The policy
  cannot reduce it by any means except shrinking σ, because
  `w_action_rate × Σ_d (a_t − a_{t−1})²` is evaluated on the *clipped sampled* action.
  That is ~8.4% of a ~3.3/step reward burned on nothing.
- **Control authority is attenuated ~34%.** `E[clip(μ+ε, −1, 1)]` with σ = 1: commanded
  0.25 → executed 0.169, 0.5 → 0.331, 0.75 → 0.480, 1.0 → 0.610. Logged
  `action_mean_abs = 0.387`, so the executed action averages ~0.26 — the environment
  systematically under-executes what the policy commands.
- **The PPO gradient is biased on ~32% of action dims.** The stored `log_prob` is the
  unclipped Gaussian density (`ActorCritic.act` docstring says so explicitly), but 8.9 of
  28 dims per step are executed at a boundary atom. This is precisely the setting of
  Fujita & Maeda, *Clipped Action Policy Gradient*, ICML 2018 (arXiv:1802.07564): the
  naive estimator that ignores clipping is higher-variance than the one that accounts for
  it. At σ ≤ 0.25 the issue disappears on its own.

## 4. `bounds_loss_coef: 10.0` is **not** the culprit

Measured in the live run: `bounds_loss = 0.0003`, `action_oob_frac = 0.0030`,
`action_mean_abs = 0.387`. The policy *mean* essentially never leaves `[-1, 1]`, so the
bound penalty contributes ~0.003 to the loss and is not shaping anything. The 2.24 vs 1.15
per-step reward gap between the deterministic and the noisy policy is dominated by the
direct dynamical cost of the noise (`ang_vel_xy`, `action_rate`, `lin_vel`, `gait_phase`,
`feet_slip`, `flight`), not by the bound penalty. **Leave `bounds_loss_coef` at 10.0** — it
is the correct AMP-lineage value and it is what keeps the mean inside the range so that the
±1 clip stays a rare event once σ is lowered.

## 5. Low explained variance is a symptom of the noise, not a critic defect

The identity: for any value function, `EV ≤ 1 − E_s[Var(G|s)] / Var(G)`. `Var(G|s)` is
irreducible — it is the spread of returns the *same* state produces under different noise
draws, and it grows with σ. A noisier behaviour policy therefore lowers the achievable EV
even for a perfect critic.

Measured on this checkpoint (208,896 transitions, `scratchpad/r2.npz`):

- Actual `EV(V, Monte-Carlo return) = +0.7152`.
- The **best EV obtainable by any recalibration of V** — i.e. replacing V with
  `E[G | V]`, the optimal function of the critic's own output — is **+0.7268**
  (100 quantile bins; +0.7212 / +0.7254 / +0.7264 / +0.7268 at 20/50/100/200 bins).

The critic is within **1.2 percentage points** of the ceiling reachable without giving it
new information. 96% of the residual is noise, not miscalibration. Meanwhile the
termination fraction is 0.662 noisy vs 0.319 deterministic — the "does this rollout fall"
coin that dominates return spread is being flipped by the exploration noise itself.

**Do not enlarge the critic, add return normalisation, raise `value_loss_coef`, or lengthen
the horizon.** Fix σ and the EV will rise on its own.

Separately: `ppo.py`'s new `explained_variance` is `1 − Var(A)/Var(V+A)` by construction
(`compute_returns` sets `returns = advantages + values`, and `update()` then computes
`residual = returns − values`, which is identically the raw advantage). It reads +0.949 on
data whose honest Monte-Carlo EV is +0.718. It is not a critic health check.

## 6. Is a max clamp equal to the init value a mistake?

Yes, on three counts.

1. **No reference implementation clamps std at all.** rsl_rl: bare `nn.Parameter`.
   rl_games: bare `nn.Parameter` (or frozen). MimicKit: `FIXED` / `CONSTANT` / `VARIABLE`,
   no clamp in any branch.
2. **A ceiling at the init value makes the parameter monotone-decreasing at best**, and the
   published behaviour of these runs is to go *up* first (Isaac Lab H1: 1.04 at iter 15).
3. **A ceiling reached from above is permanent** (§1). The same trapdoor exists on the
   `-5.0` floor in reverse.

Correct patterns, in order of preference for this project:

- **(a) Fix it.** `requires_grad=False`, no clamp, no entropy bonus — the AMP/ASE/MimicKit
  answer, used for the real Unitree G1.
- **(b) Learn it with no ceiling**, `entropy_coef` 0.005–0.01 as a *floor* — the
  legged_gym/Isaac Lab answer.
- **(c) If a bound is genuinely wanted**, project the parameter *in place after*
  `optimizer.step()` (`log_std.data.clamp_(lo, hi)`) so it sits **at** the bound with a live
  gradient below it, instead of clamping inside the forward pass.

What must never happen is (c) implemented as a forward-pass clamp with the init sitting on
the bound.

## 7. Recommendation

Ordered; 1 is a prerequisite for everything else.

1. **`humanoid_rl/algos/networks.py` — remove the forward-pass clamp** and project after
   the step instead, plus clamp on load so `--init-from` cannot smuggle an out-of-range
   value in (this run inherited `log_std = +0.003632` from
   `runs/phase2-walking-121M/checkpoints/best.pt`, which is why `init_noise_std: 0.8` in the
   YAML was silently ignored for the entire run).
2. **Set the operating point to σ = 0.25 and freeze it** for the AMP phase:
   `network.init_noise_std: 0.25`, `network.learn_noise_std: false`. Measured on this exact
   checkpoint: reward/step 1.148 (σ=1.0) → 1.802 (0.5) → **2.087 (0.25)** → 2.239
   (deterministic). σ = 0.25 recovers **93%** of the deterministic reward, drives expected
   clipped dims from 8.9 to ~0, and cuts the `action_rate` noise tax from −0.289 to −0.035.
   AMP-lineage parity would be ~0.12; 0.25 is the conservative first step from a policy
   trained at 1.0.
3. **`ppo.entropy_coef: 0.003 → 0.0`.** Every AMP/ASE/MimicKit config uses 0.0. It is
   currently inert (no gradient path) but becomes live the instant fix 1 lands, and with 28
   dims it is what pushed σ to the ceiling in the first place.
4. **Leave `bounds_loss_coef` at 10.0** (§4), leave the critic alone (§5).
5. Optional, second-order: widen the env's action clip from ±1 to ±3 in `vec_env.py`, or
   drop it (the subsequent `np.clip(s.ctrl, ctrl_lo, ctrl_hi)` is the real physical bound).
   No reference implementation clips at ±1. At σ = 0.25 this barely matters; at σ = 1.0 it
   was doing real damage.

### Expected effect

Per-step task reward ~1.15 → ~2.0+; training-time termination fraction 0.66 → ~0.35;
`episode_return` ~530 → ~1000–1400 within a few hundred iterations, with the train/eval gap
(530 noisy vs ~1150 deterministic at the same checkpoint) largely closing. `value_loss`
will jump once as the return scale rises and then settle — that is not a regression. If the
gait plateaus or the AMP style reward stops improving, σ = 0.25 is the first thing to
revisit (raise to 0.35), not the last.

---

## Sources

- legged_gym `legged_robot_config.py` — https://github.com/leggedrobotics/legged_gym
- rsl_rl `modules/actor_critic.py` (v2.3.1) — https://github.com/leggedrobotics/rsl_rl
- Isaac Lab G1/H1/ANYmal-C `rsl_rl_ppo_cfg.py`, `velocity_env_cfg.py`,
  `manager_term_cfg.py` — https://github.com/isaac-sim/IsaacLab
- IsaacGymEnvs `HumanoidAMPPPO.yaml`, `HumanoidPPO.yaml`, `HumanoidAMP.yaml`,
  `base/vec_task.py` — https://github.com/isaac-sim/IsaacGymEnvs
- ASE `amp_humanoid.yaml` — https://github.com/nv-tlabs/ASE
- MimicKit `data/agents/*.yaml`, `learning/distribution_gaussian_diag.py`,
  `learning/base_agent.py` — https://github.com/xbpeng/MimicKit
- rl_games `a2c_continuous.py`, `network_builder.py` — https://github.com/Denys88/rl_games
- Peng et al., *AMP: Adversarial Motion Priors*, SIGGRAPH 2021 — https://arxiv.org/abs/2104.02180
- Fujita & Maeda, *Clipped Action Policy Gradient*, ICML 2018 — https://arxiv.org/abs/1802.07564
- Isaac Lab H1 training log excerpt —
  https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_isaac_robotics/4_isaac_rfl/
