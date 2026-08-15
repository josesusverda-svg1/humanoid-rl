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

    # --- reward weights. Total lies in [-0.35, +3.5].
    w_upright: float = 0.5
    w_rise: float = 1.0
    w_stand: float = 2.0
    w_quiet: float = 0.5
    w_posture: float = 0.5
    w_effort: float = -0.25
    w_smooth: float = -0.10

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
    # SIZED AGAINST THE DIP, not picked. Sitting pays about 0.50/step and the intermediate
    # poses about 0.20, so the route out costs roughly 0.30/step for the 1-2 s it takes.
    # Lifting the pelvis from 0.35 m to 0.60 m in one second moves Phi by 0.00228 per step,
    # so covering the dip needs a weight near 150. At 5 it would have been 0.011/step, three
    # percent of the trap, and would have changed nothing.
    shaping_weight: float = 150.0
    #: 1.0, NOT ppo.gamma, and this is a deliberate departure from the theorem.
    #:
    #: With gamma = 0.99 at a 125 Hz control rate the -(1-gamma)*Phi drain dominates: at a
    #: weight large enough to matter it costs 1.0/step just for being upright, which swamps
    #: every real term. Measured: even rising at 0.3 m/s scored NEGATIVE. Strict invariance
    #: was unusable here.
    #:
    #: At gamma = 1 the sum telescopes exactly, so the shaping over any episode equals
    #: weight * (Phi_end - Phi_start) and NOTHING else. Path length does not matter, and
    #: oscillating the pelvis up and down pays exactly zero, which is the hack this would
    #: otherwise invite. It is no longer provably policy-invariant, but it remains immune to
    #: the failure mode that actually bites us, and it is the only form that is both.
    shaping_gamma: float = 1.0

    #: Fraction of body weight the FEET must carry before `rise` pays in full.
    #:
    #: Added after a person watched a video and said "he puts all the pressure on one hand,
    #: lifts his hip, and drifts in circles without bending a knee". Measured, and exactly
    #: right: the left hand sat at 0.469 m while the right stayed at 0.064 m, at least one
    #: hand was on the floor 98% of the time and both only 1%, knees never exceeded 1.06 rad
    #: against the ~2.4 a kneel needs, and the body span a full turn every 6 s.
    #:
    #: The cause was a hole in the reward, not the convexity. NOTHING required the legs to do
    #: anything. `upright` pays for pelvis verticality and `rise` for head height, and a
    #: one-armed prop buys both without using a leg. The foot-force conjuncts existed but
    #: gate only the STANDING terms, which pay zero for the entire approach, so the legs were
    #: irrelevant on the whole path from lying to standing.
    #:
    #: The fix is the one the design already uses elsewhere rather than a new term: `rise` is
    #: multiplied by pelvis uprightness so that height bought by diving pays nothing, and is
    #: now also multiplied by foot load so that height bought by ARM-PROPPING pays nothing.
    rise_foot_load_bw: float = 0.30

    #: Pose bank built by scripts/generate_fallen_poses.py.
    bank_path: str = "data/fallen/bank_v1.npz"
    #: Fraction of resets that start from a standing pose, so the standing terms are exercised
    #: from iteration 1. EXCLUDED from the success denominator: without that a do-nothing
    #: policy books this entire share as free successes.
    standing_reset_frac: float = 0.15


class GetUpTask(Task):
    reward_term_names = ("upright", "rise", "stand", "quiet", "posture", "effort", "smooth")

    def __init__(self, config: GetUpConfig | None = None) -> None:
        self.cfg = config or GetUpConfig()
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

    @property
    def task_obs_dim(self) -> int:
        # Deliberately nothing. The policy sees only proprioception, which is what a real
        # humanoid has lying on the floor. Notably hold_steps is NOT observed: a policy that
        # can see "112 steps to go" will schedule its collapse, and one that can see the shove
        # trigger will pre-brace against it.
        return 0

    def observe_batch(self, state: BatchState, out: np.ndarray) -> None:
        return

    def init_state(self, state: BatchState, rng: np.random.Generator) -> None:
        n = state.num_envs
        self._rng = rng
        self._hold_steps_needed = max(1, int(round(self.cfg.hold_seconds / state.dt)))
        ts = state.task_state
        ts["hold_steps"] = np.zeros(n)
        ts["held_ever"] = np.zeros(n, dtype=bool)
        ts["standing"] = np.zeros(n, dtype=bool)
        ts["push_at"] = rng.integers(*self.cfg.hold_push_window, size=n).astype(float)
        ts["pushed"] = np.zeros(n, dtype=bool)
        ts["from_standing"] = np.zeros(n, dtype=bool)
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
        cfg = self.cfg
        g = state.gravity_body
        fz = state.key_body_pos[:, 0:2, 2]
        hz = state.key_body_pos[:, 2:4, 2]
        force = state.foot_force[:, :2]
        knee = state.qpos[:, self._knee_qadr]
        rel = state.key_body_pos[:, 0, :2] - state.key_body_pos[:, 1, :2]
        # PLANAR separation, not the lateral component. A fore-aft split at hip_y = +-0.5 puts
        # the feet 0.878 m apart while the lateral-only helper reports 0.142 m, so the
        # sagittal brace would be invisible to it.
        sep = np.linalg.norm(rel, axis=1)
        fore, side = self._lean(state)
        bw = self._body_weight

        return (
            (state.root_height >= cfg.u_root_height_frac * self._standing_height)
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
            & (knee.max(axis=1) <= cfg.u_knee)
            & (sep >= cfg.u_sep_min) & (sep <= cfg.u_sep_max)
            & (np.linalg.norm(state.qvel[:, 0:3], axis=1) <= cfg.u_lin_speed)
            & (np.linalg.norm(state.qvel[:, 3:6], axis=1) <= cfg.u_ang_speed)
        )

    # ------------------------------------------------------------------ reward

    def reward_batch(self, state: BatchState, terms: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        ts = state.task_state
        U = self._standing(state)
        ts["standing"][:] = U

        # --- hold bookkeeping. This is the ONLY site that mutates it, and it runs before the
        # termination and success callbacks, so counter, reward and success see one value.
        # CONSECUTIVE: hard reset to zero on any miss, never decayed. A cumulative counter is
        # satisfied by 250 separate one-step flashes, which IS the jump-collect-fall cheat.
        ts["hold_steps"] = np.where(U, ts["hold_steps"] + 1.0, 0.0)
        reset_now = ~U & (ts["hold_steps"] == 0.0)
        if reset_now.any():
            ts["push_at"][reset_now] = self._rng.integers(
                *cfg.hold_push_window, size=int(reset_now.sum()))
            ts["pushed"][reset_now] = False
        ts["held_ever"] |= ts["hold_steps"] >= self._hold_steps_needed

        # The shove: fires once per hold attempt, at an unannounced step.
        due = U & ~ts["pushed"] & (ts["hold_steps"] >= ts["push_at"])
        ts["pushed"] |= due

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

        upright = np.clip(-state.gravity_body[:, 2], 0.0, 1.0)
        # Head height clipped at 1.0, so throwing the head ABOVE standing height pays nothing.
        # Measured: a 3 m/s launch reaches head_ratio 1.30, and tiptoe already reads 1.008.
        h = np.clip(state.head_height_ratio, 0.0, 1.0)
        # Convex in h and multiplied by pelvis uprightness: height bought by diving or
        # handstanding pays nothing, and the marginal payoff grows toward standing, so parking
        # in a kneel is a bad deal. This convexity is the knob to steepen if a kneel appears.
        # Gated on the FEET carrying load, for the same reason it is gated on pelvis
        # uprightness: height that the legs did not pay for should not be bought. Ramped
        # rather than a hard threshold, so there is a gradient toward loading the feet at
        # all rather than a cliff the policy has to jump.
        foot_load = np.clip(
            state.foot_force[:, :2].sum(axis=1)
            / (cfg.rise_foot_load_bw * self._body_weight), 0.0, 1.0)
        rise = upright * foot_load * (np.expm1(3.0 * h) / np.expm1(3.0))

        v = np.linalg.norm(state.qvel[:, 0:3], axis=1)
        w = np.linalg.norm(state.qvel[:, 3:6], axis=1)
        joint_err = state.qpos[:, self._qadr] - self._nominal

        terms[:, 0] = cfg.w_upright * upright
        terms[:, 1] = cfg.w_rise * rise
        terms[:, 2] = cfg.w_stand * U
        # quiet and posture are GATED ON U. Ungated, a stillness term is maximised by a
        # motionless body on the floor, and this repo has shipped that exact bug twice.
        terms[:, 3] = cfg.w_quiet * U * np.exp(-(v ** 2) / 0.25 - (w ** 2) / 4.0)
        terms[:, 4] = cfg.w_posture * U * np.exp(-np.sum(joint_err ** 2, axis=1) / 2.0)
        ratio = np.clip(np.abs(state.torque) / self._torque_limit, 0.0, 2.0)
        terms[:, 5] = cfg.w_effort * np.mean(ratio ** 2, axis=1) / 4.0
        jerk = state.action - 2.0 * state.prev_action + ts["prev_prev_action"]
        terms[:, 6] = cfg.w_smooth * np.mean(jerk ** 2, axis=1) / 16.0
        ts["prev_prev_action"][:] = state.prev_action

        # Potential-based shaping, added AFTER the terms so it is not one of them: it is not
        # a preference about behaviour, it is a restatement of the same preference with a
        # smoother gradient.
        # Phi is the LOWER of pelvis height and head height, each as a fraction of standing.
        #
        # Pelvis height alone was gamed within 200 iterations, and by exactly the move the
        # design note claimed was impossible. `rise` is gated on pelvis uprightness so it
        # cannot pay for a handstand; the SHAPING had no gate at all, so the cheapest way to
        # collect it was to invert: hips up, head down, balanced on the shoulders. Measured
        # on the resulting policy: pelvis 0.71 m (81% of standing) with the head at 0.14 m.
        #
        # Taking the minimum requires BOTH ends of the body to be off the floor, which is
        # what "upright" means without going near orientation, and orientation is what is
        # not monotone along the path. Scored against the poses that policy actually found:
        #
        #     pose            pelvis-only   min(pelvis, head)
        #     inverted           0.81            0.09
        #     pike/downward dog  0.82            0.50
        #     kneeling           0.68            0.66
        #     standing           1.00            1.00
        #
        # The inversion drops from nearly-standing to nearly-nothing; the honest poses barely
        # move.
        phi = np.minimum(
            state.root_height / self._standing_height,
            state.head_height_ratio,
        )
        phi = np.clip(phi, 0.0, 1.0)
        shaping = cfg.shaping_weight * (
            cfg.shaping_gamma * phi - ts["phi_prev"]) * ts["phi_valid"]
        ts["phi_prev"][:] = phi
        ts["phi_valid"][:] = 1.0

        # NOT clipped at zero, unlike LocomotionTask. That clip is affordable there because
        # the positive budget is ~6/step; here a prone policy's positive budget is ~0/step, so
        # the clip would bind on most steps and erase the penalty gradient entirely. The
        # bounds make it unnecessary: worst penalty 0.35/step against a rise gain up to 1.5.
        return terms.sum(axis=1) + shaping

    def terminated_batch(self, state: BatchState) -> np.ndarray:
        # No early termination at all. Falling is the starting condition, so terminating on it
        # would end the episode before the task begins. NaN guard only.
        return ~np.isfinite(state.qpos).all(axis=1)

    def success_batch(self, state: BatchState) -> np.ndarray:
        ts = state.task_state
        # BOTH halves. held_ever alone scores "stood at t=3 s, then lay down for 7 s", which is
        # a real published failure. Standing-at-the-end alone scores one upright frame at the
        # buzzer. Standing resets are masked out: they would be free successes.
        return ts["held_ever"] & ts["standing"] & ~ts["from_standing"]

    def on_batch_end(self, state: BatchState, metrics: dict[str, float]) -> dict[str, float]:
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
        # Honour the configured standing fraction by re-drawing from the two sub-pools.
        want_stand = rng.random(idx.size) < self.cfg.standing_reset_frac
        stand_pool = np.flatnonzero(self._bank_is_standing)
        floor_pool = np.flatnonzero(~self._bank_is_standing)
        if stand_pool.size and floor_pool.size:
            pick = np.where(want_stand,
                            stand_pool[rng.integers(0, stand_pool.size, idx.size)],
                            floor_pool[rng.integers(0, floor_pool.size, idx.size)])
        state.task_state["from_standing"][idx] = self._bank_is_standing[pick]
        return self._bank_q[pick].copy(), self._bank_v[pick].copy()

    def reset_noise(self, state: BatchState, idx: np.ndarray, rng: np.random.Generator):
        # None. Additive noise on top of a settled pose breaks the contact state that made it
        # a valid fixed point of the reset path in the first place.
        return None

    # ------------------------------------------------------------------ metrics

    def eval_metrics(self, state: BatchState) -> dict[str, float]:
        ts = state.task_state
        real = ~ts["from_standing"]
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
        }
