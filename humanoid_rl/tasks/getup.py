"""Get up off the floor, and stay up.

The task is: start somewhere on the ground, reach a genuine human standing pose, and HOLD it.
Falling is not failure here; it is the starting condition. See docs/GETUP.md for the full
specification and the measurements behind every threshold.

THE DESIGN IN ONE IDEA. All the anti-cheat burden lives in a hard conjunctive predicate `U`,
and the reward only supplies a slope toward it. A conjunct has no price: it cannot be paid for
out of another term's budget. That is precisely how this repo's earlier soft posture threshold
failed, and how the symmetry term failed before it (a 61 cm two-footed brace scored 0.91 while
taking no steps).

THE THREE THINGS A PERSON ASKS FOR, and where each is enforced:

* "not just jump up, collect points and fall" -> `stand` is paid PER STEP while `U` holds,
  with no first-crossing bonus anywhere. One fall-and-recover cycle costs about 1.5 s of
  transit at ~0.3/step instead of 2.5/step and gains nothing. Success additionally requires
  the hold to have been completed AND the body to be standing when the clock runs out, so
  "got up at t=3 s then lay down" does not count.
* "the force must be adequate, no snapping upright with one joint" -> `U13` rejects
  pass-through at speed, `U9` rejects being airborne, and a scripted shove of unknown
  direction fires inside every hold attempt.
* "it should have to work out the logic" -> `rise` is convex in head height and multiplied by
  pelvis uprightness, so the payoff grows toward standing and parking in a kneel is a bad
  local deal. That convexity is the single knob to steepen if a kneel is observed. Do not add
  a term; term interactions are where this project's bugs come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_rl.tasks.base import BatchState, Task


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """0 below lo, 1 above hi, smooth in between, with ZERO slope at both edges.

    A clipped linear ramp has full slope at its foot, so a policy is paid ~2.5% of the gate
    for inching 1% past the threshold; PPO finds and sits on exactly-threshold values, which
    is the splay-and-hop mechanism seen from the pricing side. Quadratic ends make creeping
    past a boundary worth ~nothing until genuinely past it, and saturate at the top so
    stomping pays nothing extra.
    """
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class GetUpConfig:
    #: Seconds `U` must hold CONSECUTIVELY. Configured in seconds and converted with the real
    #: dt at runtime: a hardcoded step count is how episodes came to be 8 s while every
    #: comment said 20 (a 50 Hz count on a 125 Hz loop).
    hold_seconds: float = 2.0
    #: A shove of this speed, in a uniformly random direction, fires once inside every hold
    #: attempt at a step drawn from this window. Redrawn whenever the counter resets, so a
    #: policy cannot restart the counter and coast past it. Not observable, so it cannot be
    #: pre-braced directionally.
    hold_push_vel: float = 0.6
    hold_push_window: tuple[int, int] = (40, 140)
    #: Steps after a shove during which conjunct 13 (speed) alone is forgiven. E33.
    #:
    #: THIS IS THE FIX FOR THE DEFECT THAT COST TWELVE RUNS. The shove writes 0.6 m/s into
    #: qvel while conjunct 13 caps speed at 0.4, so it violated the success predicate by
    #: arithmetic on every attempt, at a step drawn from [40, 140) of the 250 needed, and the
    #: reset re-armed it for the next attempt. `held_ever` could not fire for ANY policy.
    #:
    #: 50 steps is 0.40 s, chosen as the time for a shoved body to settle: measured, |v| after
    #: a shove is 0.62 m/s and decays below the 0.40 cap within ~0.2 s once the feet are
    #: planted, so 0.40 s is roughly double what a body that catches itself needs, and far
    #: less than the 2.0 s hold. Conjuncts 1-12 are NOT forgiven for even one step, so a body
    #: that is pushed over fails immediately regardless of this window.
    hold_push_grace: int = 50

    # --- the exam ladder, E39. Sixteen runs never passed the full exam ONCE, and the only
    # published real-robot get-up (HumanUP, RSS 2025) does not ask for that on day one: it
    # lets the robot stand ANY way first, then tightens. Each level relaxes exactly two
    # things, the knee angle and the hold duration; every other conjunct stays at full
    # strictness on every level (pelvis and head height, orientation, feet loaded and level,
    # hands off, stance width, speed). Promotion happens when the CURRENT level's standing
    # predicate holds often enough in the training batch for long enough, the same
    # achievement-gated pattern as the walking task's command curriculum.
    #
    # REPORTING IS NEVER RELAXED: eval_metrics carries standing_frac_strict (the full-exam
    # predicate) alongside the current-level standing_frac, and success_batch counts ONLY
    # full-exam completions at the final level. The ladder softens what is PAID early, not
    # what is REPORTED, which is the line between a curriculum and a lie.
    exam_enabled: bool = False
    #: (u_knee, hold_seconds) per level. The LAST level must equal (u_knee, hold_seconds)
    #: below, enforced in the constructor: the ladder ends at the real exam, always.
    exam_levels: tuple[tuple[float, float], ...] = (
        (1.30, 0.5), (1.00, 1.0), (0.80, 1.5), (0.60, 2.0))
    #: Promote when the train-batch EARNED-standing fraction stays above this...
    exam_promote_frac: float = 0.01
    #: ...as an EMA over train-env steps with this timescale (on_batch_end ticks once per
    #: env step; 1200 steps = 50 PPO iterations at horizon 24). E41: the first criterion
    #: demanded the threshold hold for 1200 CONSECUTIVE steps, and E40 measured why that
    #: never fires: hold completions churn (latch-rate waves 2-86%, median streak 2
    #: iterations) even while the catch skill itself is stable (72-80 of 80 across
    #: 100-iteration checkpoints) and the run-average latch rate was 39%. A consecutive-step
    #: gauntlet on a noisy signal promotes never; an EMA promotes on the sustained average
    #: the policy actually delivers.
    exam_promote_steps: int = 1200

    # --- the standing predicate U. Thresholds measured on a settled stand, docs/GETUP.md 1.2.
    u_root_height_frac: float = 0.85     # settled stand reads 0.995
    u_head_ratio: float = 0.90           # settled 0.995; seated reads 0.450
    u_pelvis_upright: float = -0.93      # gravity_body z; settled -1.000
    u_torso_upright: float = 0.85        # inversion guard only
    u_lean: float = 0.35                 # rad of waist fold, fore and lateral
    u_force_total_bw: float = 0.60       # settled 1.00 BW; survives mass_scale 0.85
    u_force_min_bw: float = 0.20         # settled 0.50 BW per foot
    u_foot_height: float = 0.10          # m; settled 0.027
    u_hand_height: float = 0.40          # m; settled 0.835
    u_knee: float = 0.60                 # rad; settled 0.24
    u_sep_min: float = 0.08
    u_sep_max: float = 0.35              # settled 0.170
    u_lin_speed: float = 0.40
    u_ang_speed: float = 1.5

    # --- reward weights, E34. The design rule after ten shipped exploits: every positive
    # term must be unreachable from a pose that is not on the path to standing, and a term
    # must pay for something the humanoid DOES.
    #
    # `w_upright` and `w_rise` are GONE, deliberately. Orientation alone funded the E32 curl
    # (99% of the whole signal collected lying down, motionless), and E33's jumper collected
    # it in mid-air. Nothing pays for orientation any more; orientation is only a conjunct.
    #: The one dense transit term: height of the LOWER of pelvis and head, paid only through
    #: the force corridor (see gate_* below). Small on the floor by construction: measured,
    #: the settled floor bank spans 0.0025-0.0053/step against 0.59/step at a full stand.
    w_lift: float = 0.6
    #: Paid per step while all 13 conjuncts hold, times a soft quality factor in [0.6, 1.0]
    #: (stillness and nominal posture raise it; they are inside U's gate so a corpse cannot
    #: touch them, the bug this repo shipped twice before gating).
    w_stand: float = 3.0
    #: Seniority: grows linearly with CONSECUTIVE standing steps toward the 2 s hold. Makes
    #: "flash a stand and fall" pay a fraction of "keep standing": the E33 jumper's upright
    #: instants would have earned ~0 here because the counter resets on every crash.
    w_hold: float = 2.0
    #: Once per episode, on the step the 2 s hold COMPLETES. Visible to the agent only
    #: because gamma is 0.9985 (5.33 s horizon); at the old 0.99 it would be discounted to
    #: nothing, which is E31's lesson.
    w_latch: float = 40.0
    w_effort: float = -0.25
    w_smooth: float = -0.10
    #: One-sided fine on UPWARD root velocity above launch_free_vz. Zero below it.
    #:
    #: Sized from anatomy and from the E33 policy's own forensics, measured on
    #: iter_00001700: human sit-to-stand peaks at 0.5-1.0 m/s of vertical velocity and
    #: 1.1-1.3 BW of foot force; his ballistic get-up took off at 4.65-5.0 m/s with foot
    #: force to 6.2 BW, spending 63.8% of all steps airborne with a median flight apex of
    #: 1.48 m against a 0.877 m standing height. A 5 cm hop is sqrt(2g*0.05) = 0.99 m/s, so
    #: the free threshold 1.0 m/s allows every honest rise and exactly the hop the user
    #: called acceptable, while his 5 m/s take-off costs w_launch*(5-1)^2 = 16/step for the
    #: ~15-step burst: ruinous against a total honest budget of ~5.6/step.
    #: DOWNWARD velocity is never fined: falling is the starting condition, and the shove is
    #: planar so it cannot trigger this either.
    w_launch: float = -1.0
    launch_free_vz: float = 1.0

    # --- the force corridor (the gate on `lift`). Height is the payload; force is only the
    # key, and only HUMAN-RANGE force turns it.
    #
    # Lower edge, measured on the settled floor bank: a lying body reads 0.19-0.30 BW of
    # foot force (p95 0.265, and 0.265*1.15 heavy-mass draw = 0.30), so the corridor opens
    # above that noise floor and is fully open by 0.75, under the ~0.85 BW a settled stand
    # reads at the lightest mass draw.
    #
    # Upper edge, the user's mechanism ("the closer to standing, the less force he should
    # apply"): the corridor CLOSES again above forces no human get-up produces. Sit-to-stand
    # peaks at 1.1-1.3 BW; his jump take-offs ran 2.5-6.2 BW. Between 1.6 and 2.4 the gate
    # ramps to zero, so a rise must arrive at the top on soft legs near 1.0 BW to keep being
    # paid, and a jump take-off pays nothing at the exact moment it launches.
    gate_force_lo_bw: float = 0.35
    gate_force_hi_bw: float = 0.75
    gate_force_close_lo_bw: float = 1.6
    gate_force_close_hi_bw: float = 2.4
    #: Both feet must share the load for `lift` to pay in full: 2*min(F)/0.3BW, floored at
    #: 0.35 rather than 0 so a half-kneel (one foot planted) still collects a third of the
    #: term. The E30 splay-and-hop lesson, kept as a factor instead of a cliff.
    rise_foot_load_bw: float = 0.30
    #: Feet-apart factor: full pay below 0.35 m of planar separation, ramping to a 0.30
    #: floor at 0.60 m. The residual best cheat in the adversarial review was a wide brace.
    gate_sep_full: float = 0.35
    gate_sep_zero: float = 0.60

    # --- the ball. ONE heavy hit per episode, once he is genuinely up.
    #
    # Asked for as "hit him once while he is up with a mid-size heavy ball so he falls
    # differently". The purpose is not robustness drilling; it is to generate a FALL, and a
    # fall that the body's own posture produced rather than one an offline generator guessed.
    # The pose bank is built by toppling and dropping and gives 45% side-lying against the
    # 76% a real policy produces. A hit at the top of a successful rise puts him on the floor
    # the honest way, and the next attempt starts from there.
    #
    # Once per episode, so the rise is never interrupted twice and there is always time to
    # get back up afterwards. Only when he is genuinely up: a hit on a body already on the
    # floor mostly just slides it, because the ground absorbs the impulse within one control
    # step (measured: 2.5 m/s commanded reads 0.9 m/s a step later on a prone body).
    ball_enabled: bool = True
    #: How far up he must be before the ball is thrown. 1.00 is standing, sitting is ~0.45.
    #: 0.75 means "well off the floor", so it lands on a rise that mostly worked.
    ball_min_head_ratio: float = 0.75
    #: Ball MASS and SPEED are drawn separately, and the impulse follows from them, rather
    #: than picking a pelvis velocity directly. That is what guarantees genuinely different
    #: landings: a light fast ball and a heavy slow one deliver different momentum, and the
    #: spread across the two ranges is far wider than any single hand-picked number.
    #:
    #:     dv = (1 + e) * m_ball * v_ball / m_body
    #:
    #: with e = 0.5 (a partly elastic bounce) and m_body = 50.05 kg.
    #:
    #: SIZED FROM A MEASUREMENT, because the first guess was far too hard. Toppling a settled
    #: stand and letting it come to rest:
    #:
    #:     dv 0.5 m/s -> travels 1.04 m, 100% end up down
    #:     dv 1.2     -> 1.09 m, 100% down
    #:     dv 2.0     -> 1.16 m
    #:     dv 4.0     -> 1.99 m
    #:
    #: Two things fall out. Even the gentlest hit topples him every time, so extra force buys
    #: nothing. And roughly 1.0 m of that travel is the FALL ITSELF, not the impact: a pelvis
    #: starting at 0.877 m translates about a metre as the body goes over. Anything past
    #: ~1.2 m is the body being launched, which is what "do not send him flying" rules out.
    #:
    #: 2-5 kg at 5-9 m/s gives dv 0.30-1.35 m/s: enough to put him down every time, not
    #: enough to throw him.
    ball_mass_range: tuple[float, float] = (2.0, 5.0)
    ball_speed_range: tuple[float, float] = (5.0, 9.0)
    ball_restitution: float = 0.5
    #: Where the ball lands, as a height above the pelvis. This is the other half of "falls
    #: differently": the same impulse in the chest spins him far more than one in the hip,
    #: and the resulting angular velocity decides which side he ends up on.
    ball_impact_height_range: tuple[float, float] = (0.0, 0.65)

    # --- potential-based shaping on pelvis height.
    #
    # The one reward change with a proof attached. Ng, Harada & Russell (1999): adding
    #
    #     F(s, s') = gamma * Phi(s') - Phi(s)
    #
    # for ANY function Phi leaves the optimal policy unchanged. It cannot invent a new local
    # optimum, cannot be farmed, and cannot reward standing on your hands. It only makes the
    # slope continuous, so a centimetre of pelvis lift is paid for the moment it happens
    # instead of at the end.
    #
    # WHY PELVIS HEIGHT. `upright` measures pelvis ORIENTATION, which is NOT monotone along
    # the path: it is maximal sitting, drops on all fours and kneeling, and is maximal again
    # standing. The route out of a sit therefore runs downhill, which is exactly where both
    # previous runs parked. Height is monotone by geometry: an intermediate pose cannot have
    # a pelvis height outside the interval between lying (~0.15 m) and standing (0.877), so
    # it needs no verification, unlike orientation.
    #
    # Because the optimal policy is provably unchanged, this does not confound the run: if he
    # stands, the credit still belongs to the action-range fix, and this only helped find it.
    # E31 measured the cost of a MISMATCHED gamma the hard way: at shaping_gamma 1.0 against
    # ppo.gamma 0.9985 the discounted sum does not telescope, every up-down cycle nets a
    # profit, and the E33 policy funded a 5 m/s jumping machine with it. The gammas now
    # match, which restores the theorem, and the weight drops 150 -> 60: the 150 was sized
    # against a 0.30/step "dip" that the E34 gate deleted (sitting now pays ~0 instead of
    # 0.50/step, so there is no dip to out-pay). At 60 the standing drain -(1-g)*w*phi is
    # 0.09/step against a 5.6/step standing income: 1.6%, negligible, and an Oracle check
    # (shaping_gamma_matches_rl_gamma) reports it.
    shaping_weight: float = 60.0
    #: MUST equal ppo.gamma. Guarded by the Oracle; see E31 for the measured failure.
    shaping_gamma: float = 0.9985
    #: Weight of the second potential axis, "feet tucked under the pelvis", relative to the
    #: height axis (0.25 makes a full tuck worth a quarter of full height). 0.0 disables.
    #: The user's sequencing insight, E37: a supine body cannot hear the height axis (no
    #: small motion changes height), but it CAN hear "pull your feet under you", and the
    #: pose that completes the tuck is the squat the rest of the reward already pays.
    #: Through the potential only: never a term, never farmable (gammas match).
    shaping_tuck_gain: float = 0.0

    #: Pose bank built by scripts/generate_fallen_poses.py (bank_v1), optionally extended
    #: with mid-rise rungs by scripts/generate_midrise_poses.py (bank_v2).
    bank_path: str = "data/fallen/bank_v1.npz"
    #: Fraction of resets that start on a LADDER RUNG (all fours, kneel, half-kneel, squat,
    #: crouch; generator == "midrise" in the bank). 0.0 unless the bank carries the pool.
    #:
    #: The pre-registered E34 response to a floor-parked policy: reference state
    #: initialisation (DeepMimic), never a new reward term. Starting on the middle rungs
    #: lets the critic learn those states' value directly, so the floor policy climbs
    #: toward value that is KNOWN rather than rumoured. Midrise starts are excluded from
    #: the success denominator, exactly like standing starts: success is only ever counted
    #: from a genuine floor start.
    midrise_reset_frac: float = 0.0
    #: Fraction of resets that start MID-RISE with upward momentum (generator == "rising"
    #: in the bank, built by scripts/generate_rising_states.py from time-reversed descents).
    #: E40, the discovery bridge: an episode that begins 0.3-1.5 s from standing, already
    #: moving up, puts the standing salary inside exploration range, and the catch is
    #: learned directly. Excluded from the success denominator like every given start.
    rising_reset_frac: float = 0.0
    #: E42: full get-up REFERENCE imitation. A fraction of resets starts ON a synthesized
    #: reference trajectory (supine -> sit -> tuck -> squat -> stand, built by time-reversing
    #: gentle scripted descents; scripts/generate_getup_reference.py) at a random phase, and
    #: a tracking term pays for staying near the reference as it plays. This is what every
    #: published get-up system does; twenty runs proved per-step exploration cannot discover
    #: the 2-4 s sequence on its own. The term cannot be farmed: the reference is finite and
    #: monotone, pays once through, and ends at the stand where the salary takes over.
    track_reset_frac: float = 0.0
    ref_path: str = ""
    w_track: float = 2.0
    #: Denominator of the joint-error exponent: exp(-sum_sq_err / this).
    #:
    #: E44: 2.0 was far too sharp and it flattened the reward exactly where the policy
    #: lives. Measured on E43's own trajectory, per joint across 28 joints:
    #:     30 deg error -> 0.043 of 2.0 paid (2%)
    #:     20 deg       -> 0.363 (18%)
    #:     15 deg       -> 0.766 (38%)
    #: The policy sat at 27 deg, i.e. on a 2%-of-height slope, and climbed 0.012 -> 0.088
    #: in 1200 iterations: real, but far too slow to finish. At 8.0 the same 30 deg pays
    #: 0.766, so partial imitation is rewarded and the gradient exists where the policy is.
    track_sigma_sq: float = 8.0
    #: Early termination for film-riding envs: end the episode once the pose has drifted
    #: this far (sum of squared joint errors) from the reference.
    #:
    #: The other half of E43's shortfall, and standard since DeepMimic: without it an env
    #: that falls off the film in the first second keeps running for the remaining ~2400
    #: steps collecting nothing, so most experience is gathered far from the reference and
    #: the learning signal drowns. 12.0 is |dq| ~ 0.65 rad (37 deg/joint), comfortably
    #: beyond E43's working error so it does not cut healthy attempts, and well inside the
    #: range where the track term has already gone dark.
    #:
    #: E45: 12.0 was set "beyond the working error" and that was the mistake, because the
    #: working error WAS the failure. Measured on E44 at iteration 476: on-film envs sat at
    #: 29.8 deg/joint, comfortably below the 37.5 deg threshold, earning 0.774 of 2.0 for
    #: mediocrity that never got restarted. The wider kernel had paid 9x more for the same
    #: physical behaviour (E43 27.1 deg -> E44 29.8 deg: no better, slightly worse). 6.0 is
    #: 26 deg/joint, just inside the current working error, so drifting off the film is no
    #: longer survivable.
    track_fail_err_sq: float = 6.0
    #: Also terminate a rider whose PELVIS lags the reference by more than this (metres).
    #:
    #: E46, measured: with joint angles alone as the failure test, the policy rode the whole
    #: 8.8 s film to phase 0.93-1.00 while its pelvis peaked at 0.698 against the film's
    #: 0.879, i.e. it performed the choreography in a permanently crouched copy and was
    #: never restarted for it, then collapsed when the clip ended. Joint angles do not
    #: determine height: the root is a free body. 0.18 m is just inside that measured lag.
    track_fail_dz: float = 0.18
    #: Fraction of resets that start from a standing pose, so the standing terms are exercised
    #: from iteration 1. EXCLUDED from the success denominator: without that a do-nothing
    #: policy books this entire share as free successes.
    standing_reset_frac: float = 0.15


class GetUpTask(Task):
    reward_term_names = ("lift", "stand", "hold", "latch", "effort", "smooth", "launch",
                         "track")

    def __init__(self, config: GetUpConfig | None = None) -> None:
        self.cfg = config or GetUpConfig()
        # _smoothstep divides by (hi - lo), so a config with equal or inverted edges would
        # NaN the lift term silently. Nothing else validates field RELATIONS (config.py only
        # rejects unknown keys), so the constructor is the one chokepoint every env shares.
        c = self.cfg
        for lo_name, hi_name in ((("gate_force_lo_bw"), ("gate_force_hi_bw")),
                                 (("gate_force_close_lo_bw"), ("gate_force_close_hi_bw")),
                                 (("gate_sep_full"), ("gate_sep_zero"))):
            lo, hi = getattr(c, lo_name), getattr(c, hi_name)
            if not lo < hi:
                raise ValueError(f"getup.{lo_name} ({lo}) must be < {hi_name} ({hi})")
        if not c.gate_force_hi_bw <= c.gate_force_close_lo_bw:
            raise ValueError("the force corridor's opening ramp must end before its "
                             f"closing ramp starts: {c.gate_force_hi_bw} > "
                             f"{c.gate_force_close_lo_bw}")
        if c.exam_enabled:
            last_knee, last_hold = c.exam_levels[-1]
            if abs(last_knee - c.u_knee) > 1e-9 or abs(last_hold - c.hold_seconds) > 1e-9:
                raise ValueError(
                    f"the exam ladder must END at the real exam: last level "
                    f"({last_knee}, {last_hold}) != (u_knee {c.u_knee}, "
                    f"hold_seconds {c.hold_seconds})")
            for (k1, h1), (k2, h2) in zip(c.exam_levels, c.exam_levels[1:]):
                if not (k2 <= k1 and h2 >= h1):
                    raise ValueError("exam levels must tighten monotonically")
        # Global curriculum state, deliberately on the TASK and shared by every env width:
        # the exam level is a property of the run, not of an environment. Only the widest
        # (training) env advances it, see on_batch_end.
        self._exam_level = 0 if c.exam_enabled else len(c.exam_levels) - 1
        self._exam_ema = 0.0
        self._hold_steps_by_level: list[int] = [250] * len(c.exam_levels)
        self._standing_height = 0.877
        self._standing_head = 1.514
        self._body_weight = 490.99
        self._body_mass = 50.05
        self._qadr = np.arange(7, 35)
        self._knee_qadr = np.zeros(2, dtype=int)
        self._nominal = np.zeros(28)
        self._torque_limit = np.ones(28)
        self._bank_q: np.ndarray | None = None
        self._bank_v: np.ndarray | None = None
        self._bank_is_standing: np.ndarray | None = None
        self._bank_is_midrise: np.ndarray | None = None
        self._bank_is_rising: np.ndarray | None = None
        self._rng = np.random.default_rng(0)
        self._hold_steps_needed = 250

    # ------------------------------------------------------------------ setup

    def configure_for_prepared(self, prepared) -> None:
        """Take every model-derived constant from the model, and check the bank matches it."""
        import hashlib

        model = prepared.model
        self._standing_height = float(prepared.standing_height)
        self._standing_head = float(prepared.standing_head_height)
        self._body_mass = float(model.body_mass.sum())
        self._body_weight = self._body_mass * 9.81
        self._qadr = np.asarray(prepared.actuator_qpos_adr, dtype=int)
        self._nominal = np.asarray(prepared.default_joint_pos, dtype=float)
        self._torque_limit = np.abs(model.jnt_actfrcrange[:, 1])[
            model.actuator_trnid[:, 0]].astype(float)
        self._torque_limit[self._torque_limit <= 0.0] = 1.0
        knees = [i for i, n in enumerate(prepared.joint_names) if "knee" in n]
        self._knee_qadr = self._qadr[knees[:2]] if len(knees) >= 2 else self._qadr[:2]

        path = Path(self.cfg.bank_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"no fallen-pose bank at {path}. Build it with "
                f"scripts/generate_fallen_poses.py")
        bank = np.load(path, allow_pickle=False)
        # A stale bank is a real hazard here: this repo already has a run directory named
        # ABANDONED-staleClips-tracking. Fail loudly rather than train on the wrong body.
        if int(bank["nq"]) != model.nq or int(bank["nv"]) != model.nv:
            raise ValueError(f"bank has nq/nv {int(bank['nq'])}/{int(bank['nv'])}, "
                             f"model has {model.nq}/{model.nv}")
        sha = hashlib.sha1(Path(prepared.model_path).read_bytes()).hexdigest() \
            if getattr(prepared, "model_path", None) else None
        if sha is not None and str(bank["model_sha1"]) != sha:
            raise ValueError("fallen-pose bank was built for a different model file")

        train = bank["split"] == "train"
        self._bank_q = bank["qpos"][train]
        self._bank_v = bank["qvel"][train]
        self._bank_is_standing = bank["generator"][train] == "standing"
        self._bank_is_midrise = bank["generator"][train] == "midrise"
        self._bank_is_rising = bank["generator"][train] == "rising"

        self._ref_q = None
        self._ref_bounds = None
        if self.cfg.ref_path:
            rp = Path(self.cfg.ref_path)
            if not rp.is_absolute():
                rp = Path(__file__).resolve().parents[2] / rp
            refs = np.load(rp, allow_pickle=False)
            if int(refs["nq"]) != model.nq:
                raise ValueError("reference clips built for a different model")
            if str(refs["model_sha1"]) != sha:
                raise ValueError("reference clips built for a different model file")
            self._ref_q = refs["qpos"]
            self._ref_v = refs["qvel"]
            self._ref_bounds = refs["bounds"]

    @property
    def task_obs_dim(self) -> int:
        # 3 proprioceptive extras + (when a reference bank is loaded) the reference block:
        # 1 on-film flag, 1 phase, and 28 target-minus-current joint deltas.
        #
        # E43, THE FIX FOR E42'S SILENT FAILURE. E42 paid a tracking term for matching a
        # reference the policy could not observe, and measured it: from a perfect spawn the
        # joint error reached 0.35 rad/joint within 0.24 s and the film's pelvis climbed to
        # 0.48 m while the body stayed at 0.164. Not "the film is too fast": the policy had
        # no way to know a film existed. Every imitation system (DeepMimic onward) feeds the
        # phase and the target pose; without them the term is unlearnable by construction.
        # This is the same class of defect as E34's "reward depends on what the policy
        # cannot see", repeated on a new term.
        if self._ref_q is None:
            return 3
        return 3 + 1 + 1 + int(self._qadr.size)

    @property
    def _task_obs_dim_legacy(self) -> int:
        # Three floats: per-foot ground force in body weights (2) and pelvis height as a
        # fraction of standing (1). E34: the reward pays for these exact quantities, and
        # until now the policy could not see any of them; it was being graded on instruments
        # it did not have. Both are physically realisable on a robot (pressure insoles, and
        # IMU-plus-kinematics height), the same argument vec_env makes for foot_contact.
        #
        # Still deliberately NOT observed: hold_steps, push_at, push_grace. A policy that can
        # see "112 steps to go" will schedule its collapse, and one that can see the shove
        # trigger will pre-brace against it.
        return 3

    def observe_batch(self, state: BatchState, out: np.ndarray) -> None:
        # The same height mask the reward uses: a foot that is not down reports no force,
        # so the observation cannot disagree with the pay.
        fz = state.key_body_pos[:, 0:2, 2]
        F = np.where(fz <= self.cfg.u_foot_height, state.foot_force[:, :2], 0.0)
        out[:, 0:2] = np.clip(F / self._body_weight, 0.0, 1.5)
        out[:, 2] = np.clip(state.root_height / self._standing_height, 0.0, 1.2)
        if self._ref_q is None:
            return
        # The reference block. Zeroed for envs not on a film, which is itself the signal
        # "no reference right now": the flag at index 3 disambiguates a zero delta (perfect
        # tracking) from no film at all.
        n = self._qadr.size
        out[:, 3:] = 0.0
        ts = state.task_state
        on = ts["ref_clip"] >= 0
        if not on.any():
            return
        rows = np.flatnonzero(on)
        at = ts["ref_step"][rows]
        clips = ts["ref_clip"][rows]
        starts = self._ref_bounds[clips]
        lens = np.maximum(self._ref_bounds[clips + 1] - starts, 1)
        out[rows, 3] = 1.0
        out[rows, 4] = (at - starts) / lens
        # Target MINUS current: a control error, the quantity a policy can act on directly,
        # rather than an absolute pose it would have to difference itself.
        out[np.ix_(rows, np.arange(5, 5 + n))] = np.clip(
            self._ref_q[at][:, self._qadr] - state.qpos[rows][:, self._qadr], -3.0, 3.0)

    def init_state(self, state: BatchState, rng: np.random.Generator) -> None:
        n = state.num_envs
        self._rng = rng
        self._hold_steps_needed = max(1, int(round(self.cfg.hold_seconds / state.dt)))
        self._hold_steps_by_level = [
            max(1, int(round(h / state.dt))) for _k, h in self.cfg.exam_levels]
        ts = state.task_state
        ts["hold_steps"] = np.zeros(n)
        ts["held_ever"] = np.zeros(n, dtype=bool)
        ts["standing"] = np.zeros(n, dtype=bool)
        ts["push_at"] = rng.integers(*self.cfg.hold_push_window, size=n).astype(float)
        ts["pushed"] = np.zeros(n, dtype=bool)
        # Steps of remaining forgiveness for conjunct 13 only, after a shove this task
        # applied. Per-environment and kept in task_state, never on the task: the train, eval
        # and render envs share one Task instance at different widths.
        ts["push_grace"] = np.zeros(n)
        #: True once the env has been NOT-standing at some point this episode. A stand only
        #: counts toward ladder promotion if it was preceded by not-standing: placed stands
        #: and near-stand rising spawns prove nothing by persisting (both gamed the
        #: criterion before this flag; see on_batch_end).
        ts["was_down"] = np.zeros(n, dtype=bool)
        #: Reference playback: which clip this env is following (-1 none) and the playhead
        #: index into the concatenated reference array. Per-env, in task_state, as always.
        ts["ref_clip"] = np.full(n, -1, dtype=np.int64)
        ts["ref_step"] = np.zeros(n, dtype=np.int64)
        ts["track_err"] = np.zeros(n)
        ts["track_dz"] = np.zeros(n)
        #: An env that just terminated for losing the film respawns ON the film. Without
        #: this the tracking population collapses: with early termination a film episode
        #: lasts seconds while an ordinary one lasts 20 s, so returning terminated riders to
        #: the ordinary 30% draw starves the very signal the film exists to provide
        #: (measured: 69 -> 29 on-film envs within 400 steps, still falling).
        ts["respawn_film"] = np.zeros(n, dtype=bool)
        ts["from_standing"] = np.zeros(n, dtype=bool)
        ts["from_midrise"] = np.zeros(n, dtype=bool)
        ts["prev_prev_action"] = np.zeros((n, state.action.shape[1]))
        ts["ball_thrown"] = np.zeros(n)
        ts["phi_prev"] = np.zeros(n)
        #: 0 on the step straight after a reset. The body teleports to a new pose then, so a
        #: potential difference across that jump is meaningless and would be enormous; the
        #: shaping simply sits out that one step. reset_batch cannot pre-load the new value
        #: because the pose is applied later in the engine's reset sequence.
        ts["phi_valid"] = np.zeros(n)
        # Per-ENVIRONMENT, not per-task. The training, evaluation and render environments
        # share one Task instance with different widths, so a request array kept on the task
        # is sized to whichever env called init_state last and then index-errors against the
        # others. That crashed the first evaluation of every run.
        ts["push_request"] = np.zeros(n, dtype=bool)
        ts["ball_impulse"] = np.zeros((n, 6))
        ts["ball_hits"] = np.zeros(n)


    # ------------------------------------------------------------------ predicate

    def _lean(self, state: BatchState) -> tuple[np.ndarray, np.ndarray]:
        """Signed waist fold, fore and lateral, in the heading frame.

        `torso_upright` is cos(tilt) and CANNOT distinguish a forward fold from a backward one.
        That blindness already cost this project a full analysis cycle: a 48 degree backward
        arch was diagnosed and reported as a forward lean. Anything angular here is signed.
        """
        z = state.torso_zaxis
        yaw = state.heading
        c, s = np.cos(-yaw), np.sin(-yaw)
        fore = c * z[:, 0] - s * z[:, 1]
        side = s * z[:, 0] + c * z[:, 1]
        return fore, side

    def _standing(self, state: BatchState) -> np.ndarray:
        """The 13 conjuncts. All hard; none is purchasable."""
        geom, vel = self._standing_parts(state)
        return geom & vel

    def _standing_strict(self, state: BatchState) -> np.ndarray:
        """The FULL exam's predicate, whatever level the ladder is on. Reporting only."""
        saved = self._exam_level
        self._exam_level = len(self.cfg.exam_levels) - 1
        try:
            geom, vel = self._standing_parts(state)
        finally:
            self._exam_level = saved
        return geom & vel

    def _standing_parts(self, state: BatchState) -> tuple[np.ndarray, np.ndarray]:
        """The same 13, split into the twelve about POSE and the one about MOTION.

        Split for E33. The two answer different questions and only one of them can be
        destroyed by an external shove. Conjuncts 1-12 ask "is this body arranged like a
        standing human", which a shove cannot change instantaneously. Conjunct 13 asks "is it
        moving", and the task's own shove writes 0.6 m/s straight into qvel against a 0.4 m/s
        cap, so it violates 13 by arithmetic, on every attempt, regardless of the policy.

        Splitting lets the shove forgive the velocity it created without forgiving the pose.
        That is the whole point of a shove: the body has to stay ARRANGED like a stand while
        being pushed. Forgiving the geometry too would hand a toppling mannequin a pass.
        """
        cfg = self.cfg
        g = state.gravity_body
        fz = state.key_body_pos[:, 0:2, 2]
        hz = state.key_body_pos[:, 2:4, 2]
        # Height-masked, like everywhere else in E34: a foot that is not down contributes no
        # force. Changes nothing when conjunct 9 (feet on the floor) holds, and stops the
        # 13 BW mid-air sensor ghost from ever satisfying 7-8 without it.
        force = np.where(fz <= cfg.u_foot_height, state.foot_force[:, :2], 0.0)
        knee = state.qpos[:, self._knee_qadr]
        rel = state.key_body_pos[:, 0, :2] - state.key_body_pos[:, 1, :2]
        # PLANAR separation, not the lateral component. A fore-aft split at hip_y = +-0.5 puts
        # the feet 0.878 m apart while the lateral-only helper reports 0.142 m, so the
        # sagittal brace would be invisible to it.
        sep = np.linalg.norm(rel, axis=1)
        fore, side = self._lean(state)
        bw = self._body_weight

        return ((state.root_height >= cfg.u_root_height_frac * self._standing_height)
            & (state.head_height_ratio >= cfg.u_head_ratio)
            & (g[:, 2] <= cfg.u_pelvis_upright)
            & (state.torso_upright >= cfg.u_torso_upright)
            & (np.abs(fore) <= cfg.u_lean)
            & (np.abs(side) <= cfg.u_lean)
            # FORCE, never foot_contact: the contact flag fires at 9.82 N and is True for a
            # supine corpse (46 N per foot). A published "standing on feet" indicator using
            # contact evaluates True for a body lying on its back on this model.
            & (force.sum(axis=1) >= cfg.u_force_total_bw * bw)
            & (force.min(axis=1) >= cfg.u_force_min_bw * bw)
            & (fz.max(axis=1) <= cfg.u_foot_height)
            & (hz.min(axis=1) >= cfg.u_hand_height)
            # max, not mean: a mean lets one knee at 1.0 rad be paid for by the other locked.
            # The ONE conjunct the exam ladder relaxes (E39): current level's knee, ending
            # at the strict u_knee because the last level is validated to equal it.
            & (knee.max(axis=1) <= cfg.exam_levels[self._exam_level][0])
            & (sep >= cfg.u_sep_min) & (sep <= cfg.u_sep_max)
        ), (
            (np.linalg.norm(state.qvel[:, 0:3], axis=1) <= cfg.u_lin_speed)
            & (np.linalg.norm(state.qvel[:, 3:6], axis=1) <= cfg.u_ang_speed)
        )

    # ------------------------------------------------------------------ reward

    def reward_batch(self, state: BatchState, terms: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        ts = state.task_state
        U_geom, U_vel = self._standing_parts(state)
        U = U_geom & U_vel
        ts["standing"][:] = U
        ts["was_down"] |= ~U

        # --- hold bookkeeping. This is the ONLY site that mutates it, and it runs before the
        # termination and success callbacks, so counter, reward and success see one value.
        # CONSECUTIVE: hard reset to zero on any miss, never decayed. A cumulative counter is
        # satisfied by 250 separate one-step flashes, which IS the jump-collect-fall cheat.
        #
        # E33, THE GRACE WINDOW, AND WHY IT IS NOT A SOFTENING.
        #
        # For twelve runs `standing_frac` read exactly 0.0% and `held_ever` never once fired,
        # and the reason was here, not in the reward. The shove writes `hold_push_vel` = 0.6 m/s
        # directly into qvel (vec_env applies it to the root), while conjunct 13 caps speed at
        # `u_lin_speed` = 0.4. Measured over 69-72 shove events: |v| on the step after a shove
        # is 0.62 against the 0.40 cap. No policy can prevent that; the impulse is added to the
        # state, not applied through the actuators. So U went false, `hold_steps` hard-reset to
        # zero, and the lines below re-armed `pushed` and redrew `push_at` from [40, 140), which
        # is always less than the 250 needed. Every attempt was shoved before it could finish,
        # forever.
        #
        # Measured on the model's own nominal stand, held by a fixed-target servo that cannot
        # balance at all, 64 envs x 1200 steps:
        #     shove 0.6, DR on   ->  best hold 137 of 250      never completes
        #     shove 0.6, DR off  ->  best hold 186 of 250      never completes
        #     shove OFF          ->  best hold 330 of 250      COMPLETES
        # The hold was reachable the whole time. The shove that was meant to TEST it was
        # terminating it.
        #
        # The grace forgives conjunct 13 for a short window after a shove THIS TASK ITSELF
        # applied, and forgives nothing else. Conjuncts 1-12 must still hold every single step,
        # so the body must stay arranged like a stand while being pushed, which is exactly what
        # surviving a shove means. A body that topples fails U_geom and the grace does nothing
        # for it: verified against the zero-action mannequin, whose failure is toppling rather
        # than speed.
        ts["push_grace"] = np.maximum(ts["push_grace"] - 1.0, 0.0)
        keep = U | (U_geom & (ts["push_grace"] > 0.0))
        ts["hold_steps"] = np.where(keep, ts["hold_steps"] + 1.0, 0.0)
        reset_now = ~keep
        if reset_now.any():
            ts["push_at"][reset_now] = self._rng.integers(
                *cfg.hold_push_window, size=int(reset_now.sum()))
            ts["pushed"][reset_now] = False
            # A failed attempt forfeits any residual grace. Without this line, a policy could
            # break one geometry conjunct on purpose right after a shove, restart the counter,
            # and ride ~48 leftover graced steps into the fresh attempt, stacking two graces
            # into one counted hold (measured by the adversarial review: 99 of 250 counted
            # steps above the speed cap, double the intended bound). Zeroing is safe: a
            # post-shove geometry flicker just leaves the counter at 0 until velocity
            # settles, with no E33-style loop, because counting resumes on the next keep.
            ts["push_grace"][reset_now] = 0.0
        # The latch fires on the step the hold COMPLETES, once per episode: computed BEFORE
        # held_ever absorbs it, or it would never be true.
        needed = self._hold_steps_by_level[self._exam_level]
        newly_held = (ts["hold_steps"] >= needed) & ~ts["held_ever"]
        ts["held_ever"] |= ts["hold_steps"] >= needed

        # The shove: fires once per hold attempt, at an unannounced step.
        due = U & ~ts["pushed"] & (ts["hold_steps"] >= ts["push_at"])
        ts["pushed"] |= due
        # Arm the grace on the step the shove is REQUESTED. The engine applies the impulse on
        # the following step, so the window has to open now or it opens one step too late,
        # which is the one step that matters.
        ts["push_grace"][due] = float(cfg.hold_push_grace)

        # The ball: one heavy hit per episode, thrown once he is genuinely up.
        ball = np.zeros_like(due)
        if cfg.ball_enabled:
            ready = (~ts["ball_thrown"].astype(bool)) & (
                state.head_height_ratio >= cfg.ball_min_head_ratio)
            if ready.any():
                k = int(ready.sum())
                rng = self._rng
                ang = rng.uniform(-np.pi, np.pi, k)
                mass = rng.uniform(*cfg.ball_mass_range, k)
                speed = rng.uniform(*cfg.ball_speed_range, k)
                # Momentum transfer, so mass and speed both matter and neither alone fixes
                # the outcome.
                dv = (1.0 + cfg.ball_restitution) * mass * speed / self._body_mass
                # Height of the impact above the pelvis. The lever arm turns the same
                # momentum into a very different amount of tumble.
                lever = rng.uniform(*cfg.ball_impact_height_range, k)
                imp = np.zeros((k, 6))
                imp[:, 0] = dv * np.cos(ang)
                imp[:, 1] = dv * np.sin(ang)
                imp[:, 2] = rng.uniform(-0.2, 0.2, k) * dv
                # Angular velocity about the axis perpendicular to the shove, magnitude set
                # by the lever arm. A hit in the chest topples him over his feet; one at the
                # hip mostly shoves him sideways.
                imp[:, 3] = -dv * lever * np.sin(ang) * 3.0
                imp[:, 4] = dv * lever * np.cos(ang) * 3.0
                imp[:, 5] = rng.normal(0.0, 0.4, k) * dv
                ts["ball_impulse"][ready] = imp
                ts["ball_thrown"][ready] = 1.0
                ts["ball_hits"][ready] += 1.0
                ball = ready

        # The hold shove keeps its own impulse: purely planar, as specified.
        if due.any():
            ang = self._rng.uniform(-np.pi, np.pi, int(due.sum()))
            planar = np.zeros((int(due.sum()), 6))
            planar[:, 0] = cfg.hold_push_vel * np.cos(ang)
            planar[:, 1] = cfg.hold_push_vel * np.sin(ang)
            ts["ball_impulse"][due] = planar
        ts["push_request"][:] = due | ball

        # ------------------------------------------------------------------ E34 terms.
        #
        # HEIGHT IS THE PAYLOAD, FORCE IS ONLY THE KEY. Everything dense is one term: the
        # height of the LOWER of pelvis and head (the E29 anti-headstand min), paid only
        # through a corridor of HUMAN-RANGE foot force. Orientation pays nothing anywhere:
        # `upright` funded the E32 curl (99% of the signal, collected motionless on the
        # floor) and was collected mid-air by the E33 jumper.
        #
        # The same height-masked force everywhere: the raw touch sensor reads up to 13 BW in
        # mid-air self-contact (docs/GETUP.md 6.4), so a foot only counts while it is DOWN.
        fz = state.key_body_pos[:, 0:2, 2]
        F = np.where(fz <= cfg.u_foot_height, state.foot_force[:, :2], 0.0)
        bw = self._body_weight
        lift = np.clip(np.minimum(state.root_height / self._standing_height,
                                  state.head_height_ratio), 0.0, 1.0)

        # The corridor: opens above corpse readings (a lying body presses 0.19-0.30 BW, and
        # the E33 policy learned to press 1.0+ BW while lying, which is why force can never
        # be the payload), fully open across the human get-up range, and CLOSES again above
        # forces only a jump take-off produces. Smoothstep at both edges: quadratic at the
        # foot, so creeping over a threshold buys ~nothing until genuinely past it.
        corridor = self._corridor(state)
        # Both feet share the load, floored at 0.35 so a half-kneel still collects a third.
        even = np.clip(2.0 * F.min(axis=1) / (cfg.rise_foot_load_bw * bw), 0.35, 1.0)
        # Feet not splayed, floored at 0.30: the residual cheat the adversarial review found
        # was a wide brace, and nothing else prices stance width during transit.
        sep = np.linalg.norm(
            state.key_body_pos[:, 0, :2] - state.key_body_pos[:, 1, :2], axis=1)
        narrow = np.clip((cfg.gate_sep_zero - sep)
                         / (cfg.gate_sep_zero - cfg.gate_sep_full), 0.30, 1.0)
        # Convex in height (expm1), so the marginal centimetre is worth more near the top
        # and parking in a kneel is a bad deal.
        terms[:, 0] = (cfg.w_lift * corridor * even * narrow
                       * (np.expm1(3.0 * lift) / np.expm1(3.0)))

        # The standing salary, flat from the first U step, with a SOFT quality factor in
        # [0.6, 1.0]. Stillness and nominal posture are worth 40% of the salary, not gates:
        # hard-gating them is how two earlier stillness bugs shipped, and pricing them softly
        # cannot invent a floor optimum because U itself is unreachable there.
        v = np.linalg.norm(state.qvel[:, 0:3], axis=1)
        w = np.linalg.norm(state.qvel[:, 3:6], axis=1)
        joint_err = state.qpos[:, self._qadr] - self._nominal
        quality = 0.6 + 0.2 * (np.exp(-(v ** 2) / 0.25 - (w ** 2) / 4.0)
                               + np.exp(-np.sum(joint_err ** 2, axis=1) / 2.0))
        terms[:, 1] = cfg.w_stand * U * quality

        # Seniority: standing pays MORE the longer it has already been held this attempt,
        # linear to the 2 s target. A one-step flash earns w_stand alone; the 250th
        # consecutive step earns w_stand + w_hold. This is the anti-blink device: the E33
        # jumper's upright instants would have collected ~nothing here.
        terms[:, 2] = cfg.w_hold * U * np.clip(
            ts["hold_steps"] / self._hold_steps_by_level[self._exam_level], 0.0, 1.0)

        # The exam prize: once per episode, on the completing step. See w_latch for why the
        # discount horizon makes this visible at all.
        terms[:, 3] = cfg.w_latch * newly_held

        # Penalties, damped 10x on the floor. At full price a floor policy pays more for
        # STRUGGLING than for lying still, which is half of how the E32 corpse was selected;
        # searching on the floor should be nearly free.
        pen = 0.10 + 0.90 * np.clip((lift - 0.20) / 0.50, 0.0, 1.0)
        ratio = np.clip(np.abs(state.torque) / self._torque_limit, 0.0, 2.0)
        terms[:, 4] = cfg.w_effort * pen * np.mean(ratio ** 2, axis=1) / 4.0
        jerk = state.action - 2.0 * state.prev_action + ts["prev_prev_action"]
        terms[:, 5] = cfg.w_smooth * pen * np.mean(jerk ** 2, axis=1) / 16.0
        ts["prev_prev_action"][:] = state.prev_action

        # The launch fine: one-sided, upward only, zero below launch_free_vz. See the config
        # for the anatomy and the E33 forensics behind the 1.0 m/s line. Falling is never
        # fined, and the task shove is planar so it cannot trigger this.
        overspeed = np.clip(state.qvel[:, 2] - cfg.launch_free_vz, 0.0, None)
        terms[:, 6] = cfg.w_launch * overspeed ** 2

        # E42, the reference tracking term. Pays for being NEAR the playing reference, joint
        # angles and pelvis height, only for envs currently on a clip. The playhead advances
        # once per step and the clip ENDS (at the stand), so the term is a finite, monotone
        # escort into the salary's arms: not farmable by cycling, and worth nothing to a
        # policy that ignores it.
        terms[:, 7] = 0.0
        if self._ref_q is not None:
            on = ts["ref_clip"] >= 0
            if on.any():
                rows = np.flatnonzero(on)
                at = ts["ref_step"][rows]
                ref_j = self._ref_q[at][:, self._qadr]
                err = np.sum((state.qpos[rows][:, self._qadr] - ref_j) ** 2, axis=1)
                dz = state.qpos[rows, 2] - self._ref_q[at][:, 2]
                track = np.exp(-err / cfg.track_sigma_sq) * np.exp(-(dz ** 2) / 0.02)
                terms[rows, 7] = cfg.w_track * track
                # Stashed for terminated_batch, which runs after this in the engine's step.
                ts["track_err"][:] = 0.0
                ts["track_err"][rows] = err
                ts["track_dz"][:] = 0.0
                ts["track_dz"][rows] = np.abs(dz)
                # Advance; envs whose clip just finished hand off to the plain reward.
                ends = self._ref_bounds[ts["ref_clip"][rows] + 1] - 1
                nxt = at + 1
                done_clip = nxt > ends
                ts["ref_step"][rows] = np.minimum(nxt, ends)
                ts["ref_clip"][rows[done_clip]] = -1

        # Potential shaping at shaping_gamma == ppo.gamma (E31: any mismatch is a reward
        # pump, measured at 58% of the whole signal). With matched gammas the discounted sum
        # telescopes to -Phi(start): cycling pays exactly zero, and this is the ONLY signal
        # a body flat on the floor can earn, which is deliberate.
        #
        # Phi has TWO axes since E37. Height (lift) alone is nearly silent for a supine
        # body: no small motion changes it much, so the floor gradient pointed nowhere in
        # particular. The user watched the videos and named the missing instruction: "first
        # tuck the feet under yourself, then push up from that crouch". The second axis pays
        # for exactly that tuck: horizontal distance from the pelvis to the midpoint of the
        # feet, mapped to [0, 1]. Lying extended reads ~0 (feet 0.5-0.7 m out), hook-lying
        # ~0.5, a squat ~0.9, standing ~1. Because it enters through the POTENTIAL and the
        # gammas match, sliding the feet in and out farms nothing (telescopes to zero), and
        # a curled pose collects its value once on the way in and never again. This is the
        # one channel where a sequencing hint is provably unexploitable, which is why the
        # hint goes here and not into a staged bonus (the adversarially-reviewed staged
        # design died to boundary farming).
        # PER FOOT and HEIGHT-MASKED, like every foot quantity in E34. The first version
        # used the unmasked midpoint and paid a shoulder-stand: legs straight up put the
        # feet at zero HORIZONTAL distance while touching nothing, caught on camera within
        # 300 iterations. "Tucked under you" means planted under you: a foot in the air is
        # not tucked, whatever its horizontal position.
        d_feet = np.linalg.norm(
            state.key_body_pos[:, 0:2, :2] - state.qpos[:, None, 0:2], axis=2)
        foot_down = fz <= cfg.u_foot_height
        tuck = np.mean(np.clip(1.0 - d_feet / 0.55, 0.0, 1.0) * foot_down, axis=1)
        phi = lift + cfg.shaping_tuck_gain * tuck
        shaping = cfg.shaping_weight * (
            cfg.shaping_gamma * phi - ts["phi_prev"]) * ts["phi_valid"]
        ts["phi_prev"][:] = phi
        ts["phi_valid"][:] = 1.0

        # NOT clipped at zero: a prone policy's positive budget is ~0/step, so a clip would
        # bind on most steps and erase the penalty gradient entirely.
        return terms.sum(axis=1) + shaping

    def terminated_batch(self, state: BatchState) -> np.ndarray:
        # Falling is the starting condition, so a fall never terminates: that would end the
        # episode before the task begins. The ONLY behavioural termination is a film-riding
        # env that has lost the reference (E44); everything else is a NaN guard.
        bad = ~np.isfinite(state.qpos).all(axis=1)
        ts = state.task_state
        if self._ref_q is not None and self.cfg.track_fail_err_sq > 0:
            on = ts["ref_clip"] >= 0
            lost = on & ((ts["track_err"] > self.cfg.track_fail_err_sq)
                         | (ts["track_dz"] > self.cfg.track_fail_dz))
            ts["respawn_film"] |= lost
            bad |= lost
        return bad

    def success_batch(self, state: BatchState) -> np.ndarray:
        ts = state.task_state
        # BOTH halves. held_ever alone scores "stood at t=3 s, then lay down for 7 s", which is
        # a real published failure. Standing-at-the-end alone scores one upright frame at the
        # buzzer. Standing resets are masked out: they would be free successes.
        # A held_ever earned on a relaxed level is training signal, not success: success
        # exists only at the final level, where the predicate IS the full exam.
        at_final = self._exam_level == len(self.cfg.exam_levels) - 1
        return (ts["held_ever"] & ts["standing"] & at_final
                & ~ts["from_standing"] & ~ts["from_midrise"])

    def on_batch_end(self, state: BatchState, metrics: dict[str, float]) -> dict[str, float]:
        # Exam-ladder promotion, training env only. The task instance is shared by the
        # train (4096), eval (64) and render (1) envs; without the width gate an evaluation
        # would advance the global curriculum.
        cfg = self.cfg
        if cfg.exam_enabled and state.num_envs >= 1024:
            # EARNED stands only. The first version counted the whole batch and the 30%
            # placed standing starts promoted the ladder 0 -> 2 within a hundred iterations
            # of a warm start whose floor envs had never stood once (caught live: two
            # PROMOTED lines in the opening seconds, latch at zero). A start given a stand
            # proves nothing by still being in it; a stand reached from the floor or pushed
            # up from a rung is the skill the ladder exists to grow.
            ts_ = state.task_state
            # EARNED means stood after having been down this episode. ~from_standing alone
            # was gamed twice: first by placed standing starts, then by near-stand rising
            # spawns persisting in a stand they were given. was_down closes the class: any
            # start that spawns inside the predicate must LOSE it before its standing can
            # count, while floor and genuine mid-rise starts pass trivially.
            frac = float((ts_["standing"] & ts_["was_down"]
                          & ~ts_["from_standing"]).mean())
            alpha = 1.0 / float(cfg.exam_promote_steps)
            self._exam_ema = (1.0 - alpha) * self._exam_ema + alpha * frac
            if (self._exam_ema > cfg.exam_promote_frac
                    and self._exam_level < len(cfg.exam_levels) - 1):
                self._exam_level += 1
                self._exam_ema = 0.0
                k, h = cfg.exam_levels[self._exam_level]
                print(f"[exam] PROMOTED to level {self._exam_level}: "
                      f"knee <= {k}, hold {h}s", flush=True)
        metrics["exam_level"] = float(self._exam_level)
        return metrics

    def push_request(self, state: BatchState) -> np.ndarray | None:
        """Environments that should be shoved this step, for the engine's push machinery."""
        return state.task_state.get("push_request")

    def push_impulse(self, state: BatchState) -> np.ndarray | None:
        """The full 6-vector per environment: linear velocity then angular."""
        return state.task_state.get("ball_impulse")

    # ------------------------------------------------------------------ resets

    def reset_batch(self, state: BatchState, idx: np.ndarray,
                    rng: np.random.Generator) -> None:
        ts = state.task_state
        ts["hold_steps"][idx] = 0.0
        ts["held_ever"][idx] = False
        ts["standing"][idx] = False
        ts["pushed"][idx] = False
        ts["push_grace"][idx] = 0.0
        ts["from_midrise"][idx] = False
        ts["was_down"][idx] = False
        ts["ref_clip"][idx] = -1
        ts["ball_thrown"][idx] = 0.0
        ts["phi_valid"][idx] = 0.0
        ts["phi_prev"][idx] = 0.0
        ts["ball_hits"][idx] = 0.0
        ts["push_at"][idx] = rng.integers(*self.cfg.hold_push_window, size=idx.size)
        ts["prev_prev_action"][idx] = 0.0

    def reset_pose(self, state: BatchState, idx: np.ndarray, rng: np.random.Generator):
        """Draw from the bank. Always returns a pose for EVERY resetting environment.

        `_has_reset_pose` in the engine is a single global bool: if this returns non-None then
        every resetting row reads the array. The standing mix therefore cannot be expressed by
        returning None for some rows; those rows carry standing poses drawn from the bank.
        """
        if self._bank_q is None:
            return None
        pick = rng.integers(0, len(self._bank_q), size=idx.size)
        # Three pools, drawn by configured fractions: standing, mid-rise rungs (if the bank
        # carries them), and the floor for everything else.
        u = rng.random(idx.size)
        c1 = self.cfg.standing_reset_frac
        c2 = c1 + self.cfg.midrise_reset_frac
        c3 = c2 + self.cfg.rising_reset_frac
        c4 = c3 + self.cfg.track_reset_frac
        want_stand = u < c1
        want_mid = (~want_stand) & (u < c2)
        want_rise = (~want_stand) & (~want_mid) & (u < c3)
        want_track = (~want_stand) & (~want_mid) & (~want_rise) & (u < c4)
        # Riders that just lost the film go straight back onto it, overriding the draw.
        back = state.task_state["respawn_film"][idx]
        if back.any():
            want_stand = want_stand & ~back
            want_mid = want_mid & ~back
            want_rise = want_rise & ~back
            want_track = want_track | back
            state.task_state["respawn_film"][idx[np.flatnonzero(back)]] = False
        if self._ref_q is None:
            want_track[:] = False
        stand_pool = np.flatnonzero(self._bank_is_standing)
        mid_pool = np.flatnonzero(self._bank_is_midrise)
        rise_pool = np.flatnonzero(self._bank_is_rising)
        floor_pool = np.flatnonzero(
            ~self._bank_is_standing & ~self._bank_is_midrise & ~self._bank_is_rising)
        if mid_pool.size == 0:
            want_mid[:] = False
        if rise_pool.size == 0:
            want_rise[:] = False
        if stand_pool.size and floor_pool.size:
            pick = floor_pool[rng.integers(0, floor_pool.size, idx.size)]
            pick = np.where(want_stand,
                            stand_pool[rng.integers(0, stand_pool.size, idx.size)], pick)
            if mid_pool.size:
                pick = np.where(want_mid,
                                mid_pool[rng.integers(0, mid_pool.size, idx.size)], pick)
            if rise_pool.size:
                pick = np.where(want_rise,
                                rise_pool[rng.integers(0, rise_pool.size, idx.size)], pick)
        qpos_out = self._bank_q[pick].copy()
        qvel_out = self._bank_v[pick].copy()
        ts_ = state.task_state
        ts_["ref_clip"][idx] = -1
        if want_track.any():
            k = int(want_track.sum())
            n_clips = len(self._ref_bounds) - 1
            clip = rng.integers(0, n_clips, k)
            starts = self._ref_bounds[clip]
            lens = self._ref_bounds[clip + 1] - starts
            # Any phase, weighted toward the beginning so full get-ups are practiced most,
            # while late phases keep the catch-and-hold fresh.
            phase = rng.random(k) ** 1.5
            at = starts + (phase * (lens - 1)).astype(np.int64)
            rows = np.flatnonzero(want_track)
            qpos_out[rows] = self._ref_q[at]
            qvel_out[rows] = self._ref_v[at]
            ts_["ref_clip"][idx[rows]] = clip
            ts_["ref_step"][idx[rows]] = at
        state.task_state["from_standing"][idx] = self._bank_is_standing[pick]
        state.task_state["from_standing"][idx[np.flatnonzero(want_track)]] = False
        # Rising starts share the midrise flag: excluded from the SUCCESS denominator (the
        # start was given). They still count toward ladder PROMOTION, which masks only
        # from_standing: catching a given rise into a stand is precisely the skill the
        # ladder exists to grow, while merely persisting in a given stand is not.
        from_mid = self._bank_is_midrise[pick] | self._bank_is_rising[pick]
        from_mid[np.flatnonzero(want_track)] = True    # given starts, success-masked
        state.task_state["from_midrise"][idx] = from_mid
        return qpos_out, qvel_out

    def reset_noise(self, state: BatchState, idx: np.ndarray, rng: np.random.Generator):
        # None. Additive noise on top of a settled pose breaks the contact state that made it
        # a valid fixed point of the reset path in the first place.
        return None

    # ------------------------------------------------------------------ metrics

    def eval_metrics(self, state: BatchState) -> dict[str, float]:
        ts = state.task_state
        real = ~ts["from_standing"] & ~ts["from_midrise"]
        denom = max(int(real.sum()), 1)
        return {
            "standing_frac": float(ts["standing"][real].mean()) if real.any() else 0.0,
            "held_ever_frac": float(ts["held_ever"][real].sum() / denom),
            # Got up and then lost it. UniReLo's Time-to-Fall failure, reported separately so
            # it cannot hide inside the success rate.
            "time_to_fall_frac": float(
                (ts["held_ever"] & ~ts["standing"] & real).sum() / denom),
            "hold_progress": float(ts["hold_steps"][real].mean() / self._hold_steps_needed)
            if real.any() else 0.0,
            "head_height_ratio": float(state.head_height_ratio.mean()),
            "root_height": float(state.root_height.mean()),
            "pelvis_upright": float(np.clip(-state.gravity_body[:, 2], 0.0, 1.0).mean()),
            "from_standing_frac": float(ts["from_standing"].mean()),
            # The one-armed prop, made visible. Without these the failure is only findable by
            # watching a video, which is how it was found the first time.
            "foot_load_bw": float(state.foot_force[:, :2].sum(axis=1).mean() / self._body_weight),
            "hand_height_gap": float(np.abs(state.key_body_pos[:, 2, 2]
                                            - state.key_body_pos[:, 3, 2]).mean()),
            "hands_down_frac": float(
                (np.minimum(state.key_body_pos[:, 2, 2], state.key_body_pos[:, 3, 2]) < 0.15
                 ).mean()),
            "knee_max": float(state.qpos[:, self._knee_qadr].max(axis=1).mean()),
            "spin_deg_s": float(np.degrees(np.abs(state.qvel[:, 5])).mean()),
            "ball_hits": float(ts["ball_hits"].mean()),
            # E34 watch surface. gate_frac is a PRESENCE metric (how open the lift corridor
            # is); foot_sep is what would show the wide-brace cheat; launch_overspeed_frac is
            # how much of the batch is above the 1 m/s free line, i.e. is the jump dying.
            "gate_frac": float(self._corridor(state).mean()),
            # The ladder relaxes what is PAID, never what is REPORTED: the strict full-exam
            # predicate is always logged beside the current level's.
            "standing_frac_strict": float(self._standing_strict(state)[real].mean())
            if real.any() else 0.0,
            "exam_level": float(self._exam_level),
            "foot_sep": float(np.linalg.norm(
                state.key_body_pos[:, 0, :2] - state.key_body_pos[:, 1, :2], axis=1).mean()),
            "launch_overspeed_frac": float(
                (state.qvel[:, 2] > self.cfg.launch_free_vz).mean()),
        }

    def _corridor(self, state: BatchState) -> np.ndarray:
        """The lift gate's force corridor, for metrics; mirrors reward_batch exactly."""
        cfg = self.cfg
        fz = state.key_body_pos[:, 0:2, 2]
        F = np.where(fz <= cfg.u_foot_height, state.foot_force[:, :2], 0.0)
        Fsum = F.sum(axis=1) / self._body_weight
        return (_smoothstep(Fsum, cfg.gate_force_lo_bw, cfg.gate_force_hi_bw)
                * (1.0 - _smoothstep(Fsum, cfg.gate_force_close_lo_bw,
                                     cfg.gate_force_close_hi_bw)))
