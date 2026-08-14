# Final-task from-scratch curriculum — lead-engineer prescription (2026-08-13)

One run, from scratch, on the final task as built in `humanoid_rl/tasks/locomotion.py` +
`configs/default.yaml`. Three code changes total: (1) a per-env radial difficulty scalar in
`_draw_command`, (2) `total = max(total, 0)` on the summed reward, (3) a global penalty
scale on four regularizer terms. Everything else ships as-is.

Synthesized from the three research reports persisted at
`docs/research/topics/command-difficulty-curricula-in-shipped-legged-rl-systems.md`,
`docs/research/topics/clock-periodic-reward-gaits-trained-from-scratch-bootstrap.md`,
`docs/research/topics/reward-ramps-termination-curricula-early-training-leniency.md`.

---

## A) Command curriculum — per-env radial scalar on the polar envelope

Pattern: Booster Gym's single-radial-level grid generalized to our ellipse, gated with the
legged_gym/Isaac-Lab terrain template (per-env, promote/demote at the env's own boundary,
graduate recycling). No global mean gate anywhere.

**Mechanism.** Add `task_state["difficulty"]`, one float per env, s ∈ [0.45, 1.0],
initialized 0.45. In `_draw_command`, after computing `reach`:

```
speed  = magnitude * reach * s[indices]          # magnitude = sqrt(U) as built
yaw    = U(-1, 1) * ang_vel_yaw_max * s[indices]
```

Direction θ stays uniform on the circle and the whole ellipse scales by one number, so the
per-sector mix is 12.5% per sector at every stage of training — this is the structural
guarantee that the measured 52.7% forward skew of a box cannot reappear, and it cannot be
broken by the curriculum because the curriculum never touches θ.

**Initial envelope (s0 = 0.45):** fwd cap 0.675 m/s, back 0.36, lat 0.27, yaw ±0.45 rad/s.
Mean commanded speed ≈ 2/3 of cap (area-uniform sqrt) → ~0.2–0.45 m/s, inside the
0.3–0.5 m/s cold-start band every shipped system uses. Known side effect: with the 0.15
deadband, ~31% of lateral-sector draws at s0 collapse to standing — accepted, it is extra
standing practice early and decays automatically as s grows.

**Gate — per COMMAND SEGMENT, not per episode** (mid-episode resampling at U(2.5,6) s
makes episode-level gating misattribute falls; this was Risk 2 in the command report).
Score each segment at its end (hold-time expiry, episode timeout, or termination), using
only steps with `command_age > 1.0 s` and nonzero command:

- PROMOTE `s += 0.05` iff ALL of:
  - no termination during the segment,
  - segment-mean |vx − cmd_x| < 0.25 m/s (body frame),
  - segment-mean |vy − cmd_y| < 0.15 m/s,
  - segment-mean |wz − cmd_yaw| < 0.25 rad/s,
  - segment-mean raw gait_phase match ≥ 0.85 (the [0,1] contact-match, pre-weight) —
    the walk-these-ways rule: commands do not get harder until the gait is clean.
- DEMOTE `s -= 0.10` iff the segment ended in termination (fall).
- Standing segments (zero command): no update.
- Clip s to [0.45, 1.0].

Tolerances are deliberately TIGHTER than Booster's 0.4/0.2/0.2 because our envelope is
1.5 m/s-class, not 2 m/s-class (Risk 3: slack tolerances let s outrun competence).

**Graduate recycling** (terrain randint trick): when an env sits at s = 1.0 and earns
another promotion, resample `s ~ U(0.45, 1.0)` instead. Easy commands stay in the data mix
forever; the gait never forgets slow walking and standing transitions.

**Update site:** the per-step task hook (`base.py` ~line 256, already marked "the right
place for curriculum level updates"). Log: median s, s-histogram, per-sector promotion
rate — per-sector divergence is the tripwire for the (not-yet-needed) per-sector s[k]
upgrade; do NOT ship s[k] on day one.

**Escalation only on evidence:** if eval at s = 1.0 shows backward/lateral falling ≥ 2×
forward after breakthrough, upgrade to 8 per-sector scalars, each gated ONLY on its own
sector's segments. Never gate any sector on the mean.

## B) Clock at bootstrap

- **Weight: full from iteration 0, no ramp.** `w_gait_phase = 1.0` (≈16% of the 6.4
  max-positive budget). No shipped system ramps the clock, and ours is the SAFE kind —
  a positive [0,1] contact-match bonus (humanoid-gym +1.0-match / unitree +0.18 pattern),
  not a Siekmann penalty pair — so full strength from step 0 is exactly on-recipe.
- **Frequency range: leave the Inman coupling alone — it IS the clock curriculum.**
  `f = 0.9·sqrt(max(|v_cmd|,0.3)/1.25)` clipped to [0.60, 1.35] × jitter U(0.85,1.15).
  At s0 speeds, nominal f pins at the 0.6 floor and the jitter gives a narrow live band
  ~[0.51, 0.69] Hz; as s grows, the sampled band widens toward ~1.13 Hz automatically.
  This satisfies "start f narrow, widen on the same s schedule" (clock report rec 6) with
  zero additional code, and keeps the frequency slot a live input from step one.
- **Stance fraction U(0.55, 0.65), foot offset U(0.45, 0.55): keep as built.** Narrow,
  walk-only, alternating-only — no gait blending exposure (Siekmann's multi-gait failure
  needed a much wider parameter space to appear).
- **Soft-indicator width: widen `stance_transition_width` 0.05 → 0.07** of cycle. Matches
  walk-these-ways kappa_gait_probs = 0.07, the best-evidenced public value; 0.05 was a
  guess, 0.07 is a config.
- **`free_gait_prob = 0.0` for this run** (the `default.yaml` value, which overrides the
  0.10 dataclass default — keep the yaml). The repo's own ablation note says the clock-off
  arm collapsed and the feature is unproven; a from-scratch bootstrap is not where it gets
  proven. `clock_authority` stays in the obs at constant 1.0 — accepted dead input until a
  skill needs it.
- **Air time: keep `w_feet_air_time = 1.0` as built, do NOT zero it.** WTW/unitree zero
  air time because their clocks penalize force-in-swing continuously; our air-time target
  is DERIVED from the clock ((1−stance)/f, disagreement bug already fixed) and gated on
  `moving`, so it is the humanoid-gym pattern (air_time 1.0 + contact-match 1.2 coexist,
  hardware-validated), not double-counting.
- **Clock phase randomized at reset: already built, keep** — it is the clocked-task
  substitute for mid-gait RSI (flagged: inference, no shipped primary source, see F).

## C) Leniency: what ramps, what clips, what never moves

**1. Total-reward clip (new, one line, permanent):** `total = max(sum(terms), 0.0)` in the
reward composition (legged_gym `only_positive_rewards`, in-code comment "avoids early
termination problems"). Phase2 got away without it; the final task carries a 4×-larger
penalty budget (vertical_vel −0.8 alone) and a flat cold start — this is the field's
universal always-on answer, self-annealing, never scheduled. Not ji22: the clip is simpler,
proven at our scale by every legged_gym descendant, and our positive terms already dominate
when upright (arithmetic below).

**2. Penalty curriculum (new) on exactly four terms** — the pure noise-tax regularizers:
`ctrl (−0.005)`, `action_rate (−0.01)`, `vertical_vel (−0.8)`, `ang_vel_xy (−0.05)`.
One global scalar `p`, start 0.5, clip [0.25, 1.0], updated ONCE PER ITERATION on an EMA
(α = 0.05) of completed-episode mean length L̂ (HumanoidVerse rule, retimed for this
budget):

```
if L̂ > 550 steps (11 s):  p *= 1.0003
if L̂ < 300 steps (6 s):   p *= 0.9997
```

Arithmetic for the degree: 500M steps / 98,304 per iter = 5,086 iterations. 0.5 → 1.0
needs ln 2 / 3e-4 ≈ 2,310 promoting iterations ≈ 45% of the run — full penalties by
mid-run, exactly the target the ramps report set (HumanoidVerse's shipped 1e-5 would move
p by <5% over our whole run; their degree is tuned for billion-step Isaac runs).
NEVER scaled: lin_vel, ang_vel, lateral_vel, heading, gait_phase, feet_air_time, alive,
upright/height/torso/head (tracking + gait + posture), and also `feet_slip (−0.2)` and
`flight (−0.3)` — those two are anti-exploit gait terms (skating, hopping), not
regularizers, and the exploits they block form EARLY.

**3. Termination leniency: NONE.** terminate_height 0.55, torso_upright 0.5, head_ratio
0.65, max_tilt 0.7 — constant for the whole run, like every shipped velocity task. The
ASAP multiplicative hysteresis is the designated fallback ONLY if abort rule E-1 fires and
the reduced-envelope control also flatlines; it is not in this run.

**4. Pushes: no phase-in.** 0.7 m/s impulse every ~5 s from step one, exactly as built —
mid-range of shipped values (XBot 0.2, Isaac ±0.5, legged_gym 1.0, unitree 1.5) and
phase2 broke through under it. No shipped system phases pushes in.

**5. Alive-bonus arithmetic (the 17 terms as implemented in `reward_batch`):**

Max positive/step = 1.0 (lin) + 0.5 (ang) + 0.3 (upright) + 0.3 (height) + 0.5 (alive)
+ 1.0 (air) + 0.6 (torso) + 0.3 (head) + 1.0 (gait) + 0.4 (lat) + 0.5 (heading) = **6.4**.

Early-flail penalty estimate at full scale: vertical_vel 0.8·(1.0 m/s)² ≈ 0.80;
ang_vel_xy 0.05·(2.5² + 2.5²) ≈ 0.63; ctrl 0.005·(28·0.5) ≈ 0.07; action_rate ≈ 0.10;
slip 0.2·1.0 ≈ 0.20; flight ≈ 0.15 → **≈ 1.9/step** at p = 1, **≈ 1.3/step** at p = 0.5.

Sizing rule (Booster/unitree pattern): alive + posture positives that any merely-upright
policy collects must exceed the flail penalty. Here: 0.5 (alive) + 0.3 + 0.6 + 0.3 + 0.3
(upright/torso/head/height) = **2.0 > 1.3** at start, > 1.9 even at full penalties. So
**w_alive stays 0.5** — no change — with the zero-clip as the backstop for the worst flail
tail. (No Isaac-style −200 termination penalty: pick clip OR big terminal cost, not both.)

## D) Stays EXACTLY as built (evidence already supports it)

- Polar area-uniform command sampler, deadband 0.15, `command_hold U(2.5,6)` s,
  `zero_command_prob 0.10`, command-age obs saturating at 2 s.
- Inman frequency law + anchor 0.9 Hz @ 1.25 m/s + jitter U(0.85,1.15); stance and offset
  ranges; clock-derived air-time target; sin/cos clock + heading-error encoding; the full
  13-dim task obs / 108-dim obs layout.
- All 17 reward weights and sigmas, including w_alive 0.5 and the narrow lateral
  (σ = 0.05) and heading (σ = 0.15) kernels.
- Terminations (all four), 8 Hz action low-pass, bounds_loss_coef 10, log_std_max −0.70,
  adaptive-KL LR (desired 0.01), entropy 0, PPO batch geometry 4096×24, network 512×512.
- Domain rand: full ranges, model pool 64, obs noise, pushes 0.7 @ 5 s — all from step one.
- Standing-start resets with joint noise + random clock phase.
- Eval protocol (50-iter interval, 32 episodes, video cadence).

## E) Timeline and abort rules (this machine: 98,304 steps/iter, ~1.8 s/iter at 54k/s,
5,086 iters ≈ 2.5 h; phase2 reference: ep-len 117→412 over iters ~250→350, breakthrough
~400 on the simple task)

Expected milestones:
- **Iter ~300–500 (~10–15 min):** mean episode length accelerating through 300–400 at
  s ≈ 0.45–0.55 (the s0 task is phase2-difficulty plus clock; the clip and p = 0.5 offset
  the extra terms).
- **Iter ~500–800:** breakthrough — train ep-len > 600, eval falls < 20%, median s moving.
- **Iter ~2,300 (~1.1 h):** penalty scale p reaches 1.0; median s ≥ 0.8.
- **Iter ~3,500:** bulk of envs cycling U(0.45,1.0) after graduation; deterministic eval
  falls < 5% across all 8 sectors at s = 1.0 commands.
- **Iters 3,500–5,086:** polish at full difficulty, full penalties.

Abort/intervene BY RULE (checked at eval, every 50 iters):
1. **Iter 600, ep-len rule:** mean train episode length < 250 (not clearly off the ~150
   flatline) → ABORT. Diagnosis: difficulty was not the blocker; run the cheap control the
   command report prescribed — static envelope fixed at s = 0.45, no curriculum — before
   touching anything else. If THAT also flatlines, the fault is reward composition, and the
   ASAP termination-hysteresis fallback plus term-by-term reward audit is next, not more
   curriculum.
2. **Iter 1000, dead-gate rule:** ep-len > 500 but median s still ≤ 0.55 → gate tolerances
   are miscalibrated for this plant (too tight). ABORT, loosen toward Booster's
   0.4/0.2/0.2 in one step, restart. (Cheap: 30 min lost.)
3. **Thrash rule:** median s drops by > 0.1 over any 500-iter window → promote/demote
   oscillation (pushes shortening segments → demotions). ABORT, change demotion to −0.05
   and require 2 consecutive failed segments, restart.
4. **Iter 2000 (~1 h), competence rule:** deterministic eval falls > 50% on s = 0.6-scaled
   commands → the run will not converge in budget. ABORT.
5. **Clock-adoption rule, iter 1500:** segment-mean raw gait_phase < 0.80 at eval, or
   video shows split-stance rocking / pattering → the clock is losing again. ABORT; first
   suspect is transition width (try 0.10, the LearningHumanoidWalking relaxer) before any
   weight change.
6. **Leniency-interlock rule:** p pinned at floor 0.25 beyond iter 1500 while ep-len < 400
   → leniency is not the bottleneck; fold into rule 1's diagnosis path.

## F) Primary-source gaps flagged by the researchers (do not treat these as verified)

1. Siekmann von Mises kappa: never published; OSU configs private. The 0.07 (WTW) /
   0.1-relaxer (LearningHumanoidWalking) widths are proxies — which is why B uses 0.07.
2. Booster grid axis labeling (lin vs ang rows in the flatten arithmetic) — summarizer may
   have swapped axes; spot-check `booster_gym` before porting any literal code.
3. Whether Booster clips grid-sampled commands back to [−1,1] at high levels (10 × 0.2 =
   2.0 m/s exceeds the stated range) — unresolved in the report.
4. ASAP initial penalty scale: yaml says 1.0, README Genesis run 0.5, paper text 0.1 —
   three values, attribution uncertain; our 0.5 start is a choice, not a citation.
5. The degree retunes (their 1e-4 per reset, my 3e-4 per iteration) are budget arithmetic,
   not shipped values — the one number here most deserving of a 10-minute pilot glance.
6. Random clock-phase-at-reset "standard practice": inference from the Siekmann lineage,
   no shipped config verified (moot — the repo already does it, keep it).
7. Cassie mid-gait pose-bank detail is second-hand from arXiv 2207.07835; Siekmann
   2011.01387's own reset scheme was not re-verified.
8. arXiv 2505.20619 per-phase weight tables unpublished — "masking, not ramping" rests on
   the paper's prose.
9. legged_gym command-gate detail (exactly which env_ids the mean averages over) was read
   via WebFetch summarization — paraphrase risk; also the claim that NO shipped config
   sets `commands.curriculum=True`.
10. Time-to-first-walking is not broken out in any source — only Siekmann's full
    150M-sample budget; the E-timeline extrapolates from phase2, not from literature.
