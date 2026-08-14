# Motion imitation for a human-like gait (AMP and successors)

Research date 2026-08-12. Produced as a standalone run after the workflow's imitation
agent died on an API error. Recorded here because it is the design basis for Phase 3.

> **Three claims in this document were tested on the machine and do not hold.** They are
> marked inline as CORRECTED. Believe the corrections, not the report.

---

## 1. Recommendation

**Implement AMP (Peng et al. 2021), using the paper's own hyperparameters, not the
Isaac Gym ones.**

The non-obvious reason: the AMP paper was trained on **16 CPU cores**, with 4096 samples
per update and a discriminator batch of 256. Its Table 4 *is* the small-batch CPU config.
The widely-copied `HumanoidAMPPPO.yaml` (minibatch 32768, AMP minibatch 4096, 4096 GPU
envs) is a later GPU retune that does not transfer. Most people porting AMP copy the wrong
table.

**Feasibility.** The paper reports 100M to 300M samples, 30 to 140 hours on 16 CPU cores in
2021. DeepMimic converges on a single walk clip in about 7.4M samples. At a measured 80k
env steps/s, 100M samples is about 21 minutes of pure physics, so 1 to 6 hours with the
learner in the loop. **Compute is not the constraint here. Implementation correctness is.**

### Ranked

1. **AMP** — composes with arbitrary task rewards (waypoints) and generalises off the
   reference manifold (terrain), which is exactly what phase-based tracking is bad at.
2. **DeepMimic phase tracking** — as a *mandatory bring-up step*, not an alternative. See
   the build order. Also a viable fallback with adaptive tolerance plus a symmetry loss.
3. **ADD** (Adversarial Differential Discriminators, SIGGRAPH Asia 2025) — ~150 lines in
   MimicKit, removes tracking-reward weight tuning entirely. An upgrade, not a starting point.
4. **SMP / score-matching priors** — where the field is going (no adversarial equilibrium to
   maintain), but a diffusion forward pass per step would dominate a CPU simulator. Stretch.

**Do not build on:** ASE (deprecated by its own authors in favour of MimicKit), CALM
(abandoned 2023), PULSE and PHC (need a trained teacher, ~10B samples, murky licences),
MaskedMimic (16,384 envs on 4x A100 for 14 days), HumanoidVerse (motion tracking still
TODO), ASAP and H2O/OmniH2O (sim-to-real and teleop, we have no robot).

---

## 2. What is disqualified on this machine

| Runtime | Status | Evidence |
|---|---|---|
| Isaac Gym Preview 4 | Dead. Linux x86_64 + CUDA only | NVIDIA: "legacy software... no longer supported" |
| Isaac Lab / Isaac Sim | CUDA only, Ubuntu or Windows | official install docs |
| MJX via `jax-metal` | `jax-metal` 0.1.1 last released **2024-10-08**, abandoned | PyPI, jax discussion #34648 |
| MuJoCo Warp | NVIDIA GPU for speed, CPU only for debugging | repo README |
| mjlab | "macOS is supported for evaluation only" | repo README |
| Newton | macOS CPU only | repo README |

**None of the algorithms need CUDA.** DeepMimic, AMP, ASE, PHC, PULSE, ADD are all PPO,
MLPs, GANs and transformers in stock PyTorch. What is CUDA-locked is Isaac Gym's
GPU-resident physics API, and the tacit assumption of 10^9 to 10^10 samples.

---

## 3. Implementation spec

### 3.1 Body model

Use MimicKit's `humanoid.xml` (Apache-2.0). **Feet are boxes, not capsules**, which is the
deciding property: a capsule foot is effectively a line contact and cannot produce a
visible heel-to-toe roll. 28 actuated DoF plus a free root.

### 3.2 Discriminator architecture

Unanimous across the paper and every implementation:

```
Linear(obs, 1024) -> ReLU -> Linear(1024, 512) -> ReLU -> Linear(512, 1)
```

Single scalar logit, **no output activation**. Output layer initialised
`uniform(-1, 1)`, bias zero.

### 3.3 Discriminator observations (the most-botched detail)

**The discriminator sees a state TRANSITION, `D(phi(s), phi(s'))`, not a state.**

105 dims per frame, 2 frames, so 210 total:

| Component | Dims | Frame |
|---|---|---|
| root height | 1 | world z (see the terrain warning) |
| root rotation, tangent-normal | 6 | heading-removed when `localRootObs=True` |
| root linear velocity | 3 | heading-local |
| root angular velocity | 3 | heading-local |
| joint rotations (`dof_obs`) | 52 | local joint frames |
| joint velocities | 28 | local |
| key body positions | 12 | root-relative, heading-rotated (hands, feet) |

**Highest-value single-line fix: set `localRootObs = True`.** The shipped Isaac config has
it `False`, which makes root rotation global and teaches the discriminator a
heading-*dependent* style. Fatal for a policy that must walk toward arbitrary waypoints.

**Frames of history: start with 2.** The paper, IsaacGymEnvs and IsaacLab all use 2.
MimicKit's current configs use 10, which is a later change, not the published method. More
frames give a richer style signal but also more room for the discriminator to latch onto
simulation-specific temporal signatures such as contact chatter.

### 3.4 Style reward: pick the bounded form

**Use the paper's LSGAN form:**

```
r_style = max(0, 1 - 0.25 * (D(s, s') - 1)^2)
```

Bounded in [0, 1]. Discriminator regresses to +1 for data, -1 for policy.

Everything NVIDIA-lineage (IsaacGymEnvs, ASE, skrl, MimicKit) instead uses BCE:
`-log(max(1 - sigmoid(logit), 1e-4)) * 2`, which is **unbounded above** (caps near 18.4).
An unbounded style reward is far easier to blow up against a fixed-scale task reward, and
we have small batches and no ability to brute-force through instability.

### 3.5 Discriminator loss

```
argmin_D  E_data[(D - 1)^2] + E_policy[(D + 1)^2] + (w_gp / 2) * E_data[||grad_phi D||^2]
```

with `w_gp = 10`. The apparent "5 versus 10" conflict with NVIDIA configs is not real: the
paper's term carries a factor of `w_gp / 2`, so the effective multiplier on
`mean(sum(grad^2))` is **5** either way.

- Gradient penalty on **real data only**, taken with respect to the **normalised**
  observation (after the running mean/std). Getting the order wrong scales the penalty by
  the observation variance.
- Logit weight regularisation on the final layer only, coefficient **0.01**.
- Weight decay 1e-4, preferably via the optimiser rather than an explicit summed L2 term.

### 3.6 Reward combination

```
r = w_task * r_task + w_style * r_style        with w_task = w_style = 0.5
```

**Everyone adds; nobody multiplies.** This weight is the primary tuning knob and it is
genuinely sensitive. Disney's "Learning to Walk in Costume" (2025-09) swept it: style
coefficient **0.4 scored best overall**; 0.9 gave the best style match but degraded safety.

### 3.7 Reference State Initialization

Essential, and inherited from DeepMimic whose ablations call it and early termination
"crucial". Initialise the character to states sampled randomly from all motion clips.

**The bug you will otherwise ship:** after an RSI reset the AMP observation *history* must
also be seeded from the motion clip, at negative time offsets, not from zeros. Miss this
and the discriminator sees a garbage first transition on every reset, which at small
environment counts is a large fraction of the negatives.

Early termination on any non-foot body touching the ground. Without it the character falls
and mimes the motion on the ground, a classic local optimum.

### 3.8 Normalisation and buffers

- **Separate** running normalisers for policy observations and discriminator observations.
  Checkpoint them separately. The AMP normaliser is updated with both real and policy data.
- Demo buffer of reference transitions: 200,000.
- Policy replay buffer: **1e5** (the paper's value, matching our scale), with
  `keep_prob = 0.01` on insert once full.
- The negative batch is current-rollout concatenated with replay samples, so roughly 2x
  the positive batch.

### 3.9 Hyperparameters (AMP paper Table 4)

| Parameter | Value |
|---|---|
| task / style reward weights | 0.5 / 0.5 |
| gradient penalty `w_gp` | 10 (effective multiplier 5) |
| samples per update iteration | 4096 |
| batch size / discriminator batch | 256 / 256 |
| policy stepsize (tasks) | 4e-6 |
| value stepsize (tasks) | 2e-5 |
| discriminator stepsize | 1e-5 |
| replay buffer | 1e5 |
| discount gamma (tasks) | 0.99 |
| GAE lambda | 0.95 |
| SGD momentum | 0.9 |
| PPO clip | **0.02** (not 0.2) |

Optimiser is SGD with momentum, three separate stepsizes. Use a **separate discriminator
optimiser**, not a joint loss, so the discriminator learning rate is independently tunable.

---

## 4. Failure modes and stabilisers

| Stabiliser | Status |
|---|---|
| Gradient penalty on real data | **Essential.** The paper's ablation calls it most vital |
| Policy replay buffer | **Essential.** Prevents overfitting to the most recent batch |
| Discriminator observation normalisation | **Essential.** Never seen turned off |
| RSI + early termination | **Essential** |
| Bounded (LSGAN) style reward | Strongly recommended at our scale |
| Logit regularisation + weight decay | Standard, cheap |
| Spectral norm, label smoothing, logit clamping | **No evidence in the AMP literature.** Skip |

**Discriminator collapse.** Once it separates policy from data perfectly, the style reward
saturates and gradients vanish. NEAR (arXiv:2501.14856) documents an AMP policy on stylised
walking *degrading* over training. Instrument discriminator accuracy on held-out real and
fake batches from day one; above roughly 90 percent means trouble.

**Small-batch warning (low confidence, extrapolated).** Nobody has published AMP tuning for
CPU-scale training. Our negatives per update will be far fewer than the 4096-env configs
these numbers were tuned on, which argues for a larger effective replay mix, a stronger
gradient penalty, and fewer discriminator updates per PPO update.

**Mode collapse** is real but largely not our problem: it bites when the dataset is
diverse. With walking-only clips the opposite risk applies, an over-constraining prior on
terrain the clips never contained.

**The discriminator detecting simulation artifacts rather than style.** The least
documented failure mode, but provable from code. AMP_for_hardware feeds **raw world z** into
the discriminator while correctly using terrain-relative height for the *policy*
observation. On flat ground world z carries real style information. On slopes and steps it
becomes a proxy for "where on the terrain am I", which separates policy from mocap with
zero style content. Our reference clips are all at flat-ground height and our policy will
spend most of its life at other heights.

---

## 5. Flat mocap versus terrain the mocap never contained

**Existence proof:** Wu et al., IEEE RA-L 8(8):4975-4982, 2023, report zero-shot
generalisation from a **flat-terrain motion dataset** to challenging real-world rough
terrain on a blind robot. Flat mocap plus rough terrain does work.

In order of cost:

1. **Restrict what the discriminator can see. Do this first, it is free.** Drop `root_h`
   from the AMP observation, or replace world z with height above measured terrain.
   Everything else in the AMP observation is already frame-local and terrain-agnostic.
2. **Terrain curriculum.** Start flat, ramp with a success-gated level system.
3. **Tune the style weight, do not guess it.**
4. **Residual policy on a frozen flat-terrain prior** (Wu et al.; Motion Priors Reimagined).
5. **Adaptive style weighting via CMDP** (Wen et al., CoRL 2025): maximise style imitation
   *subject to* task performance staying near-optimal.
6. **Condition the discriminator on terrain** (T-GMP) — but this needs terrain-conditioned
   reference data, which we do not have.
7. **Augment the dataset with physically-corrected terrain motion** (PARC). Most principled,
   most expensive.

### Waypoints

The AMP paper defines both of our task types directly, both at 0.5 / 0.5 weights:
- **Target heading**: move at target speed along a direction.
- **Target location**: move to a point, with the goal expressed **in the character's local
  frame**.

Two details worth copying verbatim: the goal is *always* local-frame, so global position
never leaks into the policy; and the target **resamples every 100 to 200 steps** rather than
persisting for a whole episode, which is a cheap and effective form of multi-waypoint
training.

---

## 6. Mac-specific gotchas

1. ~~The gradient penalty cannot run on MPS (PyTorch #98498, `aten::linear_backward`
   double-backward unimplemented). Keep the discriminator on CPU.~~
   **CORRECTED 2026-08-13.** Tested directly on torch 2.13.0: `torch.autograd.grad(...,
   create_graph=True)` through `nn.Linear` works on MPS and agrees with CPU
   (0.0093 vs 0.0094). The issue has been fixed. **The discriminator stays on the GPU.**
2. ~~MPS may be slower than CPU at these network sizes; benchmark before committing.~~
   **CORRECTED 2026-08-13.** Measured on the `[1024, 512]` discriminator: MPS is **3x to
   4x faster than CPU at every batch size tested**, including 256 (0.374 ms vs 1.557 ms).
3. ~~`amp_humanoid.xml` may not load in MuJoCo 3.11.~~ **CORRECTED 2026-08-13.** MimicKit's
   `humanoid.xml` loads unmodified (nq=35, nv=34, nu=28, stable). ASE's version fails on a
   missing material `grid`, and is NVIDIA non-commercial anyway.
4. Batch policy inference across all environments in one process. Many worker processes
   each doing batch-1 forwards would be dominated by dispatch overhead.
5. Avoid `.cpu().numpy()` in the inner loop; write torch-native running mean/std.
6. Skip mixed precision. MimicKit itself sets `use_mixed_precision: false` for AMP.

---

## 7. Data

| Dataset | Licence | Registration | Note |
|---|---|---|---|
| **100STYLE** | **CC BY 4.0** | **none** | A locomotion-*style* dataset. May cover Phase 3 outright |
| PHUMA | permissive | none | Physics-filtered AMASS: no floating, penetration or foot skating |
| AMASS | non-commercial research/education/artistic | **yes** | Highest fidelity, SMPL-based |
| LAFAN1 | CC BY-NC-**ND** | none | The **no-derivatives** clause makes retargeting legally awkward |
| ASE clips (`amp_humanoid_walk.npy`) | NVIDIA non-commercial | none | Matched to the AMP humanoid, fastest path to a first clip |

**Compute reference velocities carefully.** Mocap velocities come from finite differencing,
which amplifies measurement noise; simulated velocities come from the integrator. If the two
are computed differently, the discriminator learns to detect *that difference* instead of
style. Filter the mocap and use the same stencil and timestep on both sides.

Retargeting: **GMR** (MIT, CPU, 35 to 70 FPS, accepts BVH with no SMPL registration) is the
least painful maintained option. `poselib` also ships a ready-made
`retarget_cmu_to_amp.json`.

---

## 8. Build order

| Stage | What | Estimate |
|---|---|---|
| 0 | Load the humanoid in MuJoCo 3.11, tune PD gains and contacts, play a mocap clip kinematically and eyeball it | 2-3 days |
| 1 | **DeepMimic single-clip tracking with RSI and early termination. Do not proceed until the gait looks right.** ~7.4M samples | 3-5 days |
| 2 | AMP on flat ground, no task reward, paper hyperparameters. Instrument discriminator accuracy | 4-6 days |
| 3 | Add the heading task at 0.5 / 0.5 | 2 days |
| 4 | Procedural terrain plus curriculum. **Remove world-z from the AMP observation first** | 3-5 days |
| 5 | Multi-waypoint courses, local-frame goal | 2 days |
| 6 | Optional: ADD instead of the tracking reward, or a symmetry loss for cleaner arm swing | 2-4 days |

Stage 1 is not optional. It validates the MJCF, PD gains, contacts, retargeting and RSI
*before* an adversary can paper over their defects. If the humanoid cannot track one walk
clip, AMP will never produce a clean gait.

**Evaluation.** LocoMuJoCo ships DTW and discrete Frechet gait-similarity metrics (MIT),
which are the right off-the-shelf instruments for "does this match the reference gait".
