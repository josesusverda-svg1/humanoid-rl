# Get-up / fall-recovery policies: what the literature actually does

Research sweep, 2026-08-14. Sources are papers, not this repo's measurements. Where a claim
here conflicts with a local measurement, **the measurement wins**.

Scope: how published get-up work defines success, generates fallen initial states, structures
reward, whether it requires a *hold*, what curriculum it uses, what it says is hard, and what
degenerate solutions it reports. Numbers and equations are quoted from the sources.

---

## The five papers that matter, ranked by transferability to this repo

| # | Paper | Sim / robot | Why it matters here |
|---|---|---|---|
| 1 | **Tao et al. 2022, "Learning to Get Up"** ([arXiv:2205.00307](https://arxiv.org/abs/2205.00307), SIGGRAPH 2022) | **MuJoCo**, 1.5 m / 38.3 kg / 21 DoF humanoid, SAC | Closest match to our setup. Only paper with an **explicit hold requirement in timesteps**. |
| 2 | **He, Dong, Chen, Gupta 2025, "HumanUP"** ([arXiv:2502.12152](https://arxiv.org/abs/2502.12152), RSS 2025) | Isaac Gym, Unitree G1 | The paper the task brief names. Full reward tables with weights. |
| 3 | **Huang et al. 2025, "HoST"** ([arXiv:2502.08378](https://arxiv.org/abs/2502.08378), ICLR 2025) | Isaac Gym, Unitree G1 | The **hold-until-end-of-episode** success definition, force curriculum, multi-critic. |
| 4 | **Azulay, Xu, Scheffer, Yu 2026, "VIGOR"** ([arXiv:2602.16511](https://arxiv.org/abs/2602.16511)) | Isaac, G1 | 2026 follow-up. Explicit "sustained duration" success, and **no early termination on failure**. |
| 5 | **Xu et al. 2026, "UniReLo/CG-MuTra"** ([arXiv:2606.08922](https://arxiv.org/abs/2606.08922)) | Isaac, humanoid | 2026 follow-up. Unifies recovery + locomotion; introduces **Time-to-Fall** as a metric, i.e. it explicitly measures the fall-after-standing hack. |

Also read: Spraggett 2025, cross-morphology zero-shot get-up ([arXiv:2512.12230](https://arxiv.org/abs/2512.12230));
Lu et al. 2026, state-dependent AMP for walk/run/recover ([arXiv:2605.18611](https://arxiv.org/abs/2605.18611));
Lei et al. 2026, KungFuAthlete fall-resilient tracking ([arXiv:2602.13656](https://arxiv.org/abs/2602.13656));
Jiang et al. 2025, two-stage sit/stand ([PMC12650239](https://pmc.ncbi.nlm.nih.gov/articles/PMC12650239/));
Xu, Li, Lin, Yu 2025, unified fall-safety from few demos ([arXiv:2511.07407](https://arxiv.org/abs/2511.07407)).

---

## 1. How success is defined

Nobody defines success as "reached a height once". Every paper either evaluates the criterion
**at the end of the episode** or requires it **sustained**.

**HumanUP** (§ evaluation):
> "the robot's head height must be ≥ 1.1 m at termination, thus, the robot needs to continue
> to stand for success."

The hold is implicit: the check is at the *last* step of a fixed horizon, so a jump-and-fall
scores zero. Rolling-over success is separate: cosine between base/knee/torso orientation and
the target supine orientation ≥ 0.9.

**HoST** (§ metrics) — the cleanest statement:
> "The episode is considered successful if the robot's base height, h_base, exceeds a target
> height h_targ **and is maintained for the remainder of the episode**, indicating stable
> standing."

h_targ = 0.70 m on flat/platform/wall, 0.60 m on slope. Episode = 500 steps at 50 Hz = **10 s**.
So the effective hold is "from first success to 10 s", which is a *variable* hold, longest for
policies that get up fast.

**Tao et al. 2022** — the only explicit fixed-duration hold, and it is inside the user's
requested 1–3 s window:
> "the head height h_head is above 1.2 m" ends the get-up trajectory, after which the character
> must "maintain balance for another **100 timesteps** unless early termination is triggered".

At 40 Hz control that is **2.5 seconds**. Early termination: "The early termination will be met
if the center of mass height is below 0.5 m." Character standing height 1.5 m, so the up-threshold
is 0.80 of standing head height and the give-up threshold is 0.33 of standing CoM height.

**VIGOR** (2026):
> success demands the robot "stably reach[es] a reference standing configuration, with all
> tracked links and the relative head height within a fixed tolerance **for a sustained
> duration**", with a 7.5 s budget. They report two rates: Succ (89.5% stand-up / 90.5% fall-recovery)
> and Succ_safe (86.7% / 89.3%), the latter additionally requiring the head never came within
> 5 cm of the terrain.

**UniReLo** (2026) reports two protocols and, critically, two *times*:
- **Time-to-Stand (TTS)**: activation → first stable-upright instant, over successful trials only.
- **Time-to-Fall (TTF)**: first stable-upright instant → secondary fall.

TTF exists precisely to catch the "got up, then fell over" policy that a first-crossing success
metric would score as a win. Their continuous-command protocol requires the robot to
"recover, establish sustained command tracking, and **avoid a secondary fall** within the
evaluation horizon".

**Reported success rates for calibration**: HumanUP 78.3% real-world get-up (vs 41.7% for the
G1 factory controller) and 98.3% rolling-over across six terrains. HoST ~99% in sim.
Jiang et al. 99.5% sim / 85–90% real. Cross-morphology zero-shot 86 ± 7%.

---

## 2. How fallen initial states are generated

Three approaches. All of them **simulate the fall** rather than hand-authoring a pose.

**HumanUP — offline pose bank (the method to copy).** A dataset 𝒫 of **20K supine + 20K prone**
poses, generated by:
> "randomizing initial DoFs from canonical lying poses, dropping the humanoid from **0.5 m**,
> and simulating for **10 s** to resolve self-collisions."

10K poses per set used for training, the other 10K held out for evaluation. Note the held-out
split: they treat initial states as a train/test axis, which is a discipline this repo does not
currently apply anywhere.

**KungFuAthlete 2026 — "Gravity-based Randomized State Initialization" (D_GRSI).** The robot is
> "initialized in a zero-torque mode and released under gravity with randomized contact friction",
augmented by "randomly re-combining the rotational components (r, θ)". Plus **Low Kinetic Energy
Sampling**: episodes may only start at low-energy anchor timesteps, because
> "failures often occur at aerial or high-kinetic-energy phases, where the robot state is
> physically unrecoverable within a single episode."

That last point is a direct warning for us: if we hand fallen states straight from our walking
policy's falls, some will be mid-tumble and *unrecoverable*, and the policy will be punished for
states no controller could save.

**Tao et al. 2022 — ragdoll drop.** Character starts from
> "a rag-doll fall at 1.5 m above the ground with a randomized pose",
with actions sampled a ~ 𝒩(0, 0.1) for **80 control timesteps** (2 s at 40 Hz) until ground
collision settles.

**HoST — terrain-induced diversity instead of pose sampling.** Four terrains produce the variety:
ground (flat), platform (trunk supported, heights 20–92 cm), wall (trunk supported, inclination
14–84°), slope (1–14°). Note their finding under § 7: training supine and prone *together* hurt.

**UniReLo 2026 — prioritized initial-state sampling.** Sampling priority combines a normalized
moving-average return term (1 − V̂_{j,ξ}), a terrain-plasticity index λ_i I_j where
I_j = Var_{ξ∈K}(p̂_{j,ξ}) is the variance of recovery success across terrains, and an
under-exploration bonus λ_u / √(N_{j,ξ}+1). This "emphasizes challenging, terrain-sensitive, and
underexplored initial states".

---

## 3. Reward structures, with weights

### HumanUP Stage I (discovery), Table II

Penalties:

| Term | Expression | Weight |
|---|---|---|
| Torque limits | 𝟙(τ_t ∉ [τ_min, τ_max]) | −0.1 |
| DoF position limits | 𝟙(d_t ∉ [q_min, q_max]) | −5 |
| Energy | ‖τ ⊙ q̇‖ | −1e−4 |
| Termination | 𝟙_termination | **−500** |

Regularization: DoF acceleration ‖d̈_t‖² −1e−7 · DoF velocity ‖ḋ_t‖²₂ −1e−4 · action rate ‖a_t‖²₂ −0.1 ·
torque ‖τ_t‖ −6e−7 · angular velocity ‖ω²‖ −0.1 · base velocity ‖v²‖ −0.1 ·
foot slip 𝟙(F_z^feet > 5.0)·‖v_z^feet‖ −1.

**Task terms (this is the interesting block):**

| Term | Expression | Weight |
|---|---|---|
| Base height, exponential | exp(h_base) − 1 | **+5** |
| Head height, exponential | exp(h_head) − 1 | **+5** |
| Δ base height | 𝟙(h_t^base > h_{t−1}^base) | +1 |
| Δ feet contact force | 𝟙(‖F_t^feet‖ > ‖F_{t−1}^feet‖) | +1 |
| Standing on feet | 𝟙((‖F^feet‖ > 0) & (h^feet < 0.2)) | **+2.5** |
| Body upright | exp(−g_z^base) | +0.25 |
| Soft body symmetry | ‖a_left − a_right‖ | −1.0 |
| Soft waist symmetry | ‖a_waist‖ | −1.0 |

Three design choices worth stealing:

1. **Height is exponential, not linear or tolerance-banded.** exp(h) − 1 has a *growing* gradient,
   so the marginal payoff for the last 10 cm exceeds the payoff for the first 10 cm. This is
   the opposite shaping from a tolerance band and it is what stops the policy settling into a kneel.
2. **The two indicator terms are pure progress signals** (𝟙 height increased, 𝟙 foot force
   increased). They pay +1 for *making progress this step*, which densifies an otherwise sparse task.
3. **"Standing on feet" is conjunctive**: foot loaded AND foot below 0.2 m. It cannot be earned by
   a hand or a knee, and it cannot be earned by a foot in the air.

Rolling-over task (separate policy): base/torso/knee gravity errors 1 − cos θ, each weighted −2.

### HumanUP Stage II (deployment), Table III

Same skeleton, plus **ankle-specific and upper-body-specific limit penalties** (−0.01 torque,
−5 DoF position), and regularization tightened by orders of magnitude: DoF velocity −1e−4 → **−1e−3**,
torque −6e−7 → **−0.003** (a 5000× increase). The task terms are replaced by one tracking term:

> Tracking DoF position: exp(−(d_t − d_t^target)² / 4), weight **+8**

where the target is the Stage-I trajectory **slowed 8×** by interpolation (get-up 8 s, roll-over 4 s).

### HoST — four reward *groups* with separate critics

r_t = w^task·r^task + w^style·r^style + w^regu·r^regu + w^post·r^post,
with w^task = 2.5, w^style = 1, w^regu = 0.1, w^post = 1.

- **Task** (active when h_base < H_stage1): head height f_tol(h_head, [1, ∞), 1, 0.1) ×1;
  base orientation f_tol(−θ_base^z, [0.99, ∞), 1, 0.05) ×1.
- **Style**: waist yaw −10 if |q_waist| > 1.4; hip roll/yaw −10; knee −0.25 (ground) / −10 (platform,
  slope, wall); shoulder roll −2.5; foot displacement +2.5; **ankle parallel +20**; foot distance −10
  if ‖q_feet^l − q_feet^r‖² > 0.9; feet stumble 0 (ground) / −25 (other); shank orientation +10;
  base angular velocity +1.
- **Regularization**: joint acc −2.5e−7‖p̈‖²; action rate −1e−2‖a_t − a_{t−1}‖²; **smoothness
  −1e−2‖a_t − 2a_{t−1} + a_{t−2}‖²** (second difference, i.e. jerk); torque −2.5e−6‖τ‖²;
  joint power −2.5e−5|τ|·|ṗ|ᵀ; joint velocity −1e−4‖ṗ‖²₂; joint tracking error −2.5e−1;
  position-limit violation −1e2; velocity-limit violation −1.
- **Post-task** (active only when h_base > H_stage2 = 0.65 m) — **this is the hold reward**:
  base angular velocity ×10, base linear velocity ×10, base orientation ×10, base height ×10,
  upper-body posture ×10, feet parallel ×2.5.

The post-task group is the mechanism that makes standing *worth staying in*: six terms at weight
10 that only pay while the robot is already up and quiet. Without it the height reward alone is a
first-crossing bonus.

**Multi-critic is load-bearing, not a nicety.** Their ablation HoST-w/o-MuC achieves **zero success
rate on all terrains**. One critic could not balance four groups spanning ~7 orders of magnitude
of scale. (Relevant to this repo's 21 unnormalised terms.)

### VIGOR 2026

RB position tracking +1.25 · RB rotation tracking +0.50 · joint position tracking +0.50 ·
torque −1e−6 · torque-limit violation −0.1 · joint position limit −10.0 · joint velocity limit −5.0 ·
**head height (post-recovery only) +0.25** · base linear velocity penalty −1.0.
Note the head-height term is gated to post-recovery and weighted *low* — the work is done by
tracking a reference, and head height only shapes the final hold.

### KungFuAthlete 2026, recovery-specific terms

Motion body position +4.0 · orientation +2.0 · angular velocity +1.0 · CoM +2.0 ·
**close feet −1000** · feet slip −2.0 · root orientation −1.0 · action rate knee −3 / **ankle −20** ·
soft DoF limit −100 · undesired contacts −0.5 · **XY root movement before stand −1.0** ·
**action rate before stand −2.0**.

The last two are get-up-specific: horizontal root translation and action rate are penalized *only*
while the shoulder height is still far from the reference, i.e. while the robot is still down. This
directly discourages "scoot along the floor" solutions.

### Tao et al. 2022 — multiplicative, not additive

This is the structural difference from everyone else. Rewards **multiply**, each component
normalized to [0,1] by a Gaussian tolerance function f(i, b, m, v):

- Discovery / weaker stage: **R_weak = r_h · r_straight · r_v^xy · r_feet**
  - r_h = f(h_head, [1.55, ∞), 0.37, 0.1)
  - r_straight = f(z_torso^up, [0.9, ∞), 1.9, 0.0) if h_com > 0.5 m, else 1.0
  - r_v^xy = ½ Σ f(v', [−0.3, 0.3], 1.2, 0.1)  ← penalizes horizontal CoM velocity
  - r_feet = f(d_feet, [0, 0.9], 0.38, 0)
- Slow stage: **R_slow = r_com · r_ori · (r_hip + 2)/3**
- Balance/hold stage: **R_balance = r_v^xy · r_straight · r_com · r_pose**,
  with r_pose = exp[−¼ Σ‖q_j − q̂_j‖²] against a **manually designed standing pose**.

A product means **any single factor at zero zeroes the whole reward**. Height alone buys nothing
if the torso is not straight, and neither buys anything if the CoM is skating sideways. This is
structurally immune to the "one term quietly dominates" failure logged repeatedly in this repo's
LOGBOOK, at the cost of much weaker gradients early.

Note `r_straight` is gated on `h_com > 0.5 m` — uprightness is not asked for while still on the
floor, because on the floor it is unachievable and would just be a constant negative.

---

## 4. Is there a hold requirement? Yes, in four different forms

| Paper | Mechanism | Duration |
|---|---|---|
| Tao et al. 2022 | Explicit balance phase after h_head > 1.2 m, tracking a standing pose | **100 steps @ 40 Hz = 2.5 s** |
| HoST | Success evaluated as "maintained for the remainder of the episode"; post-task reward group active only above H_stage2 | variable, up to 10 s |
| HumanUP | Success measured at **termination** of the episode, not at first crossing | whole remaining horizon |
| VIGOR | Tolerance on tracked links + relative head height "for a sustained duration" | unstated, within a 7.5 s budget |
| UniReLo | Separate **Time-to-Fall** metric from first-upright to secondary fall | reported, not thresholded |

**Consensus**: the standard way to prevent jump-collect-fall is *not* a bonus for reaching height,
it is (a) evaluating the criterion at the end of a fixed horizon and (b) paying a dense
low-velocity/upright/pose reward that only accrues while already standing. HoST's post-task group
and Tao's R_balance are the same idea implemented two ways.

---

## 5. Curriculum

Three distinct curricula appear, and they are solving three different problems.

**(a) Exploration assist, then withdraw it.**
HoST applies a vertical pulling force ℱ on the base, **initially 200 N**, that
> "takes effect only when the robot's trunk achieves a near-vertical orientation."

On reaching the target head height the force **decreases by 20 N** per curriculum stage, floor **0 N**.
Ablation: "Without the proposed force curriculum, the robot fails to stand up on all terrains
except the platform." Jiang et al. use the same idea ("vertical upward force gradually reduced").

The conditional gating matters: the force only helps once the robot has already done the hard part
(righting the trunk), so it does not simply lift a ragdoll.

**(b) Strong-to-weak actuation, to escape the kneel local optimum.**
Tao et al. train first with human-calibrated torque limits 𝒯, then set limits to **β^i × 𝒯** at
curriculum stage i with **β = 0.95**, sampling the multiplier per episode as 𝒩(β^i, ε), ε = 0.04.
Advance when accumulated test reward exceeds ω = 60; stop when gradient steps exceed
ℳ_i = clip(1.5 × N_{i−1}, 3e5, 8e5). Final limits typically **40–60% of default 𝒯**.

Their justification is the important part:
> "Discovering get-up motions from scratch using DRL is particularly challenging in that the
> exploration process can readily become trapped in local minima, which results in variants of a
> kneeling motion."

and, on why the obvious alternative fails:
> "adding an energy cost without the strong-to-weak curriculum has minimal effect" and
> "penalizing high joint velocities either has negligible effects or leads to training instability."

**This is the direct answer to "it must not be able to snap upright by flicking one joint."**
The field's answer is not a reward penalty. It is an *action-bound / torque-limit curriculum*
that starts permissive (so a solution is found at all) and tightens (so the found solution has to
be re-solved with realistic authority). A penalty alone was measured not to work.

HoST implements the same idea as an **action rescaler**: p_t^d = p_t + β·a_t with a_t ∈ [−1,1]^n,
β **initially 1**, decreasing **0.02** per stage, floor **0.25**. Same trigger as the force curriculum.

**(c) Two-stage discovery → deployment (HumanUP, Jiang et al.).**
Stage I: simplified collision mesh, canonical pose, weak regularization → discovers a **< 1 s**
motion. Stage II: full URDF mesh, 10K randomized poses, terrain and dynamics randomization
(base CoM offset, control delay), regularization raised 5000× on torque, and the objective replaced
by DoF-position tracking of the **Stage-I trajectory slowed 8×**. Control 50 Hz, sim 1000 Hz.

Stated risk of this design, from the authors:
> "Motions discovered in Stage I could be incompatible with stronger control regularization used
> in Stage II."

Jiang et al. hit exactly this: "Discrepancies between actions explored in Stage II and those
discovered in Stage I" due to differing regularization.

---

## 6. What the authors say is hard

**HumanUP** names three properties that separate get-up from locomotion:
> "(a) **Non-periodic behavior.** In locomotion, contacts happen in structured ways... The
> getting-up problem doesn't have such periodic behavior."
> "(b) **Richness in contact**... many other parts of the robot are likely already in touch with
> terrain... the robot may find it useful to employ its body, outside of the feet."
> "(c) **Reward sparsity.** Designing rewards for getting up is harder than other locomotion
> tasks... many parts of the body make negative progress."

(c) is the one that bites: a correct get-up requires *lowering* the head to fold into a base of
support before raising it. Any monotone height reward fights the correct solution in its first phase.
HumanUP's answer is the pair of Δ-indicator terms plus a large enough exponential payoff at the top.
Tao's answer is a curriculum that lets a strong character brute-force through the trough first.

**HoST**: "time-varying contact points, multi-stage motor skills, and precise angular momentum
control, making RL exploration challenging." Their #1 practical difficulty is **reward balancing
across stages without explicit stage separation** — solved by multi-critic.

**Jiang et al.** independently list the same three: non-periodic, diverse contacts (traditional
simplifications "completely fail"), and reward sparsity because "many initial movements may appear
'negative'".

**UniReLo 2026** adds one that is specific to unifying get-up with walking:
> "A policy spanning nonperiodic whole-body recovery, transient support reorganization, and
> periodic locomotion must represent motion distributions with markedly different temporal
> structures."

and, importantly for detection:
> "successful rising cannot be determined by body height or uprightness alone."

**KungFuAthlete 2026**: "failures often occur at aerial or high-kinetic-energy phases, where the
robot state is physically unrecoverable within a single episode."

---

## 7. Failure modes and reward hacks that were actually observed

This is the section to weigh most heavily.

**"Keeps jumping after getting up."** HumanUP, describing the Tao et al. baseline verbatim:
> "learns a very fast getting-up motion that **keeps jumping after getting up**."

This is precisely the hack in the user's request. It was observed, in a MuJoCo humanoid, by the
paper that introduced the get-up task. It is not hypothetical.

**Ballistic sub-second motion.** HumanUP Stage I, without strong regularization,
> "discovers a fast but unsafe getting-up motion (**< 1 s**)."

A get-up completed in under a second on a human-scale body is not a get-up, it is a snap. Their fix
is not a penalty on speed; it is to retime the discovered trajectory **8× slower** and train a second
policy to track it under heavy regularization.

**Violent ground hitting and bouncing.** HoST, without action bounds:
> "movements are excessively violent, as indicated by three performance metrics", including
> "violent ground hitting and rapid bouncing movements."

**Oscillation.** HoST, without L2C2 smoothness regularization:
> "motion oscillations are observed in all scenes... often leading to standing-up failures."

Their smoothness loss: L_L2C2 = λ_π D(π_θ(s_t), π_θ(s̄_t)) + λ_V Σ D(V_{φ_i}(s_t), V_{φ_i}(s̄_t)),
with s̄_t = s_t + (s_{t+1} − s_t)·u, u ~ U(·), λ_π = 1, λ_V = 0.1.

**Kneeling local optimum.** Tao et al.: fixed low torque limits cause policies to converge on
"variants of a kneeling motion" rather than standing.

**Unnatural balancing limbs.** HumanUP: "our learned motions sometimes include unnatural hand
raising for balance" / "our discovered motion tends to raise hands for balance." A hand thrown
outward is a cheap way to buy angular momentum without doing leg work.

**Got up, then fell.** UniReLo 2026, on hard-gated (as opposed to continuously gated) mode switching:
> "abrupt changes in the dominant motion prior repeatedly drive the robot toward recovery-like
> actions **after it has temporarily become upright**", causing oscillation and secondary falls.

Also: without terrain-conditioned guidance, the robot "may **reach the target body height**, but
poorly positioned feet and insufficient terrain-dependent postural adaptation fail to provide a
locomotion-ready support configuration." I.e. **height was reached and the pose was still wrong** —
an explicit statement that a height threshold alone is a hackable success criterion.

**Scoring the failure of standing rather than standing.** Tao et al.: the learned motions
> "are usually more statically stable at the beginning and become dynamic and less stable near the
> end"; characters "lose balance when asked to pause in more dynamical states."

So the last phase — exactly the phase a hold requirement tests — is where their policies are weakest.
A hold requirement is not a formality; it is the binding constraint.

**Pose interference.** HoST: training supine and prone together "negatively impacted performance due
to interference between sampled rollouts." HumanUP likewise trains supine and prone as separate
regimes and has a separate rolling-over policy to convert prone → supine.

**Single-critic collapse.** HoST-w/o-MuC: **zero** success on every terrain.

**No early termination as a deliberate choice.** VIGOR 2026:
> "Episodes are not terminated early upon failure; instead, the policy is allowed to continue
> executing until the episode horizon", to learn from "prolonged contact, partial collapses, and
> unstable intermediate configurations."

---

## 8. How the field detects "fallen", and the sign problem

The task brief flags that `torso_upright` is a sign-blind cosine. The literature's detectors are
**vector-valued**, which is what restores the sign:

- **Lu et al. 2026** gate recovery mode on the projected gravity z-component: recovery activates
  when **|g_z + 1| > 0.6**. With the standard convention g = (0,0,−1) when upright, g_z = −1 gives 0
  and lying flat gives 1, so the threshold is g_z > −0.4, i.e. tilt from vertical > arccos(0.4)
  ≈ **66°**. (A secondary summary of that paper states "≈37°", which does not follow from that
  convention — verify against the source before porting the number.)
- **HumanUP** uses `exp(−g_z^base)` as its uprightness reward and, for rolling over, per-body
  gravity errors `1 − cos θ` on **base, torso, and both knees separately** — four angles, not one.
- **UniReLo** states outright that "successful rising cannot be determined by body height or
  uprightness alone" and adds a *support-feasibility* term over contact count, support margin,
  knee error and slip.
- **HumanUP's "standing on feet"** term, 𝟙((‖F^feet‖ > 0) & (h^feet < 0.2)), is the cleanest
  keypoint-plus-contact detector in the literature: it distinguishes "upright" from "upright and
  actually supported by the feet".

The generalizable lesson: **a scalar cosine can say "not upright"; it cannot say "supine", "prone",
or "sitting".** Distinguishing those requires the full body-frame gravity vector (its x component
separates face-up from face-down) plus keypoint heights and contacts. Every paper that needs the
distinction — HumanUP for supine-vs-prone, HoST for its terrain scenes — carries either a 3-vector
or several independent angles.

---

## 9. Numbers worth having on one line

- Head-height success thresholds: HumanUP **1.1 m** (G1 stands ~1.32 m ⇒ **0.83× standing head height**);
  Tao et al. **1.2 m** (1.5 m character ⇒ **0.80×**).
- Base-height success: HoST **0.70 m** flat / **0.60 m** slope; stage thresholds **H_stage1 = 0.45 m**,
  **H_stage2 = 0.65 m**. Jiang et al. **1.0 ± 0.02 m**.
- Hold: Tao **100 steps @ 40 Hz = 2.5 s**. HoST/HumanUP: remainder of a **10 s** / fixed episode.
- Episode: HoST **500 steps @ 50 Hz = 10 s**. Tao **250 steps @ 40 Hz = 6.25 s** + 100 balance steps.
  VIGOR budget **7.5 s**.
- Give-up termination: Tao, CoM height < **0.5 m** (0.33× standing).
- Get-up duration: Stage-I discovery **< 1 s** (rejected); deployed **8 s** get-up, **4 s** roll-over.
- Force curriculum: **200 N → 0 N** in **20 N** steps, gated on near-vertical trunk.
- Action bound: **β: 1.0 → 0.25** in **0.02** steps. Torque curriculum: **β = 0.95** per stage,
  final **40–60%** of nominal.
- Initial-state bank: **20K supine + 20K prone**, drop from **0.5 m**, settle **10 s**, 50/50 train/eval split.
- Termination penalty: HumanUP **−500**.
- Regularization jump between stages: torque **−6e−7 → −0.003**, DoF velocity **−1e−4 → −1e−3**.

---

## Sources

- [arXiv:2502.12152 — Learning Getting-Up Policies for Real-World Humanoid Robots (HumanUP, RSS 2025)](https://arxiv.org/abs/2502.12152) · [project page](https://humanoid-getup.github.io/) · [code](https://github.com/RunpeiDong/humanup)
- [arXiv:2502.08378 — Learning Humanoid Standing-up Control across Diverse Postures (HoST, ICLR 2025)](https://arxiv.org/abs/2502.08378) · [project page](https://taohuang13.github.io/humanoid-standingup.github.io/)
- [arXiv:2205.00307 — Learning to Get Up (Tao, Wilson, Gou, van de Panne, SIGGRAPH 2022)](https://arxiv.org/abs/2205.00307) · [project page](https://tianxintao.github.io/get_up_control/)
- [arXiv:2602.16511 — VIGOR: Visual Goal-In-Context Inference for Unified Humanoid Fall Safety (2026)](https://arxiv.org/abs/2602.16511)
- [arXiv:2606.08922 — UniReLo / CG-MuTra: Unified Humanoid Policy from Fall Recovery to Locomotion (2026)](https://arxiv.org/abs/2606.08922)
- [arXiv:2512.12230 — Learning to Get Up Across Morphologies: Zero-Shot Recovery with a Unified Humanoid Policy](https://arxiv.org/abs/2512.12230)
- [arXiv:2605.18611 — Unified Walking, Running, and Recovery via State-Dependent Adversarial Motion Priors (2026)](https://arxiv.org/abs/2605.18611)
- [arXiv:2602.13656 — A Kung Fu Athlete Bot: Autonomous Fall-Resilient Tracking (2026)](https://arxiv.org/abs/2602.13656)
- [arXiv:2511.07407 — Unified Humanoid Fall-Safety Policy from a Few Demonstrations](https://arxiv.org/abs/2511.07407)
- [PMC12650239 — A Two-Stage RL Framework for Humanoid Robot Sitting and Standing-Up (Jiang et al.)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12650239/)
