"""Velocity-command locomotion: walk in a commanded direction at a commanded speed.

This is the Phase 2 task and the foundation for Phase 5. The policy receives a velocity
command (forward speed, sideways speed, turning rate) and is rewarded for matching it.

Why command-following rather than jumping straight to waypoints: a waypoint controller is
naturally expressed as a layer converting "the target is 4 m ahead and 30 degrees left"
into a velocity command. Training command-following first yields a policy that already
knows *how* to walk and turn, so the waypoint task only has to learn *where* to go.
Learning both at once is markedly slower and is a common reason humanoid navigation
training stalls.

Every tracking reward uses an exponential kernel, exp(-error^2 / sigma), rather than a
negative squared error. The exponential is bounded in (0, 1], so no single term can
dominate the total, and its gradient is strongest near the target, which is exactly where
precision matters.

All methods operate on the whole batch at once. See `humanoid_rl/tasks/base.py` for the
measurements showing why per-environment callbacks are about five times slower here.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_rl.tasks.base import BatchState, Task


def _ratio(a: float, b: float) -> float:
    """Smaller over larger, so 1.0 means the two are equal and 0.0 that one is absent."""
    return float(np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-6))


def _stance_width(state: BatchState) -> np.ndarray:
    """Lateral foot separation per environment, in each humanoid's own heading frame.

    Key bodies 0 and 1 are the left and right foot. Rotating into the heading frame first
    matters: without it a humanoid walking along y reads as having a wide stance when its
    feet are in fact directly one behind the other.
    """
    rel = state.key_body_pos[:, 0, :2] - state.key_body_pos[:, 1, :2]
    return np.abs(-np.sin(state.heading) * rel[:, 0] + np.cos(state.heading) * rel[:, 1])


@dataclass
class LocomotionConfig:
    """Every tunable number for the walking task. Mirrored in the YAML config."""

    # Command sampling. Ranges are the half-axes of an ELLIPTICAL envelope, sampled in
    # polar form, not the edges of a box.
    #
    # A box over (vx, vy) is not a neutral choice: measured over 2M samples of the old
    # ranges, 52.7% of commands fell in the forward sector and 3.7% in the lateral one, and
    # a 45 degree diagonal was capped at 0.57 m/s by the corner geometry, so "walk
    # diagonally at 1 m/s" was not merely rare, it was UNREPRESENTABLE. The roadmap needs
    # forward, backward, lateral and all four diagonals to work equally well, and a box
    # cannot express that. Polar sampling gives 12.2-12.9% in each of the eight sectors.
    #
    # Magnitudes are bounded by what this model can physically do, checked against the
    # retargeted mocap rather than guessed: on this humanoid the reference reaches forward
    # p95 1.44 m/s and lateral p95 1.09 m/s (the sidestep-run clip averages 0.88 m/s
    # lateral), so these are inside the feasible set. Human data agrees on the shape:
    # backward and lateral preferred speeds are roughly 60% and 40% of forward.
    lin_vel_x_range: tuple[float, float] = (-0.8, 1.5)  # m/s, backward and forward reach
    lin_vel_y_range: tuple[float, float] = (-0.6, 0.6)  # m/s sideways
    ang_vel_yaw_range: tuple[float, float] = (-1.0, 1.0)  # rad/s turning
    #: Fraction of commands drawn from a narrow forward cone instead of uniformly around
    #: the circle, and the cone's half-angle.
    #:
    #: Uniform direction sampling was itself a fix, for a box sampler that starved lateral
    #: commands at 3.7% and made a fast diagonal geometrically unrepresentable. It fixed
    #: that, and over-corrected: it made every heading equally important when the roadmap
    #: (walk like a human, follow a waypoint course) is almost entirely forward walking.
    #:
    #: The cost was measured, not guessed. Under uniform sampling only 4.6% of commands
    #: asked for a forward speed at or above 1.0 m/s, so the one skill the project exists to
    #: produce was practised in 1 command in 22, and the policy sensibly did not learn it.
    #: At 0.4 that becomes 20.2% while the weakest of the eight direction sectors still gets
    #: 7.5%, against the 4.2% floor the Oracle's coverage check enforces. Both properties
    #: are checked, so neither can silently regress into the other.
    forward_bias_prob: float = 0.4
    forward_cone_deg: float = 30.0
    #: Commands below this collapse to exactly zero, so there is no band of near-still
    #: commands that a policy can only satisfy by marching on the spot. Matches the 0.1
    #: threshold the reward terms already use to decide "is it moving".
    command_deadband: float = 0.2  # XBot zeroes ||cmd|| <= 0.2
    #: Seconds a command is held before being redrawn, sampled per environment.
    #:
    #: Commands used to be drawn ONLY at episode reset, so one command was held for the
    #: entire 12 to 20 second episode and the policy never once experienced a command
    #: CHANGING while it walked. A navigation layer changes the command several times a
    #: second, so the very first thing added on top would have hit a transient the walker
    #: had never seen. Every reference implementation resamples mid-episode.
    command_hold_range: tuple[float, float] = (8.0, 12.0)  # T1; XBot 8s, G1 10s
    #: Fraction of episodes given a zero command, which teaches the policy to stand still.
    #: Without it a policy often cannot stop, which breaks waypoint arrival in Phase 5.
    zero_command_prob: float = 0.10

    # Reward weights.
    w_lin_vel: float = 1.2      # XBot tracking_lin_vel
    w_ang_vel: float = 1.1      # XBot tracking_ang_vel
    w_upright: float = 0.3
    w_height: float = 0.3
    #: Uprightness of the TORSO, not the pelvis. Weighted heavily because without it the
    #: policy folds forward at the abdomen and lurches: pelvis level, at the right height,
    #: every other term satisfied, torso horizontal and head near the ground. Found by
    #: watching a video at 24.6M steps, invisible in the metrics.
    w_torso_upright: float = 0.6  # folded into w_orientation below; kept for the termination sensor only
    #: Head height relative to standing, which independently penalises stooping.
    w_head_height: float = 0.0   # no reference has it; orientation+height cover it
    w_alive: float = 0.25       # Booster T1 (G1 uses 0.15)
    #: TORQUE squared, not position-command squared. All three references penalise the
    #: physical effort; none penalises the command magnitude (a command against a spring
    #: can cost nothing, a small one against gravity a lot). Weight from Booster T1.
    #: XBot's -1e-5, not T1's -2e-4: torque scale is robot-specific, and this model's PD
    #: gains produce torques several times T1's (measured -9.4/step at -2e-4, which would
    #: have drowned the whole positive budget of 6.4). XBot is the high-torque reference.
    w_torque: float = -1.0e-5
    w_dof_vel: float = -1.0e-4   # T1; XBot -5e-4, G1 -1e-3. Unanimous mechanism.
    #: The canonical anti-vibration term, unanimous across XBot/G1/T1 and absent here
    #: until the conformance audit. Penalising joint acceleration is how the references
    #: prevent the action tremor this project instead discovered as a noise-crutch and
    #: had to remove with a plant filter. Both now exist; this one is theirs.
    w_dof_acc: float = -1.0e-7
    w_action_rate: float = -0.01
    #: Pays out once per footfall, in proportion to how long that foot was swinging. This
    #: is the term that produces actual stepping. Without it a velocity-tracking policy
    #: reliably converges on a shuffle: both feet stay in contact and it slides forward,
    #: which scores well and looks nothing like walking.
    #: Zero: G1 ships without it, the clock contact-match supersedes it, and our derived-
    #: target variant twice contradicted the clock. Code path kept, weight conforms to G1.
    w_feet_air_time: float = 0.0
    #: Penalises having neither foot on the ground, which suppresses hopping and the
    #: "ballistic dive" solution where the humanoid launches itself at the target.
    w_flight: float = 0.0        # no reference has it; the schedule licenses flight
    #: Penalises horizontal foot motion while that foot is bearing load. This is what
    #: separates walking from skating: without it a policy happily plants a foot and slides
    #: it along the ground, which tracks the commanded velocity perfectly and looks awful.
    w_feet_slip: float = -0.2
    #: Reward for matching a periodic gait schedule. This is the term that supplies RHYTHM.
    #:
    #: Without it, PPO found a gait that satisfies every other term while not walking: a
    #: fixed split stance, right foot locked 36 cm in front, both feet pattering at 7.4
    #: strikes per second while the body rocked back and forth over them. It swapped which
    #: foot was in front only 0.42 times a second, against 7.4 strikes: in a real walk those
    #: two numbers are equal, because every step trades the lead. Velocity tracking, air
    #: time, slip, uprightness and symmetry were all happy. A human watching the video was
    #: not, and described it as walking like a horse.
    #:
    #: The schedule is a clock: each foot should be loaded during its stance half and
    #: airborne during its swing half, with the two feet driven half a cycle apart. That is
    #: the definition of alternation, and no rocking motion can satisfy it.
    w_gait_phase: float = 1.2   # Booster T1 feet_contact_number weight
    #: Nominal stride frequency at `gait_speed_anchor`, in Hz. Each foot strikes once per
    #: stride, so 0.9 Hz is 1.8 foot strikes per second, human walking cadence.
    #:
    #: This is now the ANCHOR of a law rather than a constant: the commanded frequency
    #: scales with commanded speed as Inman's square-root relation, and is OBSERVED by the
    #: policy. A frequency the network cannot see is a constant it cannot adapt to, and a
    #: policy that has only ever walked at one cadence will refuse another.
    gait_frequency: float = 0.9
    gait_speed_anchor: float = 1.25          # m/s at which gait_frequency applies
    #: 0.7-1.1 Hz = 1.4-2.2 foot strikes/s, straddling the human 1.6-2.0.
    #:
    #: Was (1.0, 2.0), taken from Booster T1 in the conformance pass without checking what
    #: their "frequency" counts on a robot with different leg length. It commanded 2-4
    #: strikes/s, and the policy obeyed: 2.91 measured, forcing an 0.11 m stride against a
    #: human 0.6-0.8. We were commanding the shuffle we kept trying to tune away.
    gait_frequency_range: tuple[float, float] = (0.7, 1.1)
    #: Multiplicative jitter on the commanded frequency. Load-bearing: without variation the
    #: slot is a dead input the network learns to ignore, and it would be as unusable later
    #: as if it had never been added.
    gait_frequency_jitter: tuple[float, float] = (0.85, 1.15)
    #: Fraction of the cycle each foot spends in stance. 0.6 gives the 0.20-0.25 double
    #: support that defines a walk. Below 0.5 the two stance windows stop overlapping and a
    #: flight phase appears, which is running: the same schedule covers both.
    stance_fraction: float = 0.6
    stance_fraction_range: tuple[float, float] = (0.55, 0.65)
    #: Phase offset between the feet, in cycles. 0.5 is alternating (walk and run); 0.0 is
    #: both feet together (hop, two-footed jump take-off). Sampled narrowly for now, but it
    #: is the axis along which two-footed skills are reached THROUGH the clock rather than
    #: by switching it off.
    foot_phase_offset_range: tuple[float, float] = (0.45, 0.55)
    #: Width of the soft transition at each end of the stance window, in cycles. The old
    #: hard boolean was a step discontinuity twice per cycle.
    #: 0.07 is walk-these-ways' shipped soft-indicator width, the best-evidenced value.
    stance_transition_width: float = 0.07
    #: Fraction of segments where the clock is switched OFF (alpha = 0): the schedule is not
    #: enforced and the policy is told so. This is the mechanism by which a future jump,
    #: crouch or climb is not fighting a walking rhythm, and exercising it now means the
    #: input is alive rather than a dead slot bolted on later.
    free_gait_prob: float = 0.10
    #: Commanded body height as a fraction of standing height. Reserved capability made
    #: live: crouching to pass under something is a point in this range, and a policy
    #: trained with it as an input can crouch later without retraining.
    body_height_range: tuple[float, float] = (0.94, 1.02)

    # ---- difficulty curriculum (radial, per environment) ----
    # One scalar s per environment multiplies the whole command envelope, so difficulty
    # grows while the per-sector mix stays 12.5% at EVERY stage: the forward-bias failure
    # (52.7% forward practice) is structurally unreproducible. The pattern is Booster Gym's
    # radial level grid generalised to the ellipse, gated like the legged_gym terrain
    # curriculum: promote on demonstrated competence, demote on a fall, and at the top
    # resample down so easy commands never leave the data (the "graduate recycling" trick).
    #: 0.70, not the 0.45 first tried. At 0.45 the commands were so small that ignoring
    #: them was profitable: standing under a 0.45 m/s command still scored 0.44 on the
    #: velocity term, difficulty never promoted once in 500 iterations, and the videos
    #: showed a humanoid that simply stood. The field's practice agrees: at the ~1 m/s
    #: class, shipped humanoids (XBot, HumanoidVerse, Unitree G1) train the full range from
    #: scratch; a speed curriculum only helps at aggressive envelopes. 0.70 keeps commands
    #: meaningful from step one and leaves the curriculum its real jobs: the stretch to the
    #: full envelope, and graduate recycling so easy commands stay in the data.
    #: BACK ON, at 0.7. The reasoning that switched it off was right about the references
    #: and wrong about us, and the missing number was theirs, not ours.
    #:
    #: "The references train their full command range from step zero" is true. Their full
    #: range is XBot-L's shipped `lin_vel_x = [-0.3, 0.6]` m/s. At 0.6 m/s no curriculum is
    #: needed and none is used. Our envelope now reaches 1.5, two and a half times that,
    #: because human free walking is 1.2-1.4 and that is the deliverable. Nothing in the
    #: references speaks to an envelope that size.
    #:
    #: What does: legged_gym ships `update_command_curriculum`, which widens the range by
    #: 0.5 m/s only once mean tracking reward exceeds 0.8 of its own maximum. KSLC
    #: (arXiv:2409.16611) reaches 3.5 m/s with the identical rule, same threshold, same
    #: increment, and states the failure mode for skipping it: a robot "may struggle to earn
    #: rewards if the early training is dominated by high-velocity commands". ALMI extends
    #: its command range on the same condition. That failure mode is what the envelope run
    #: measured: the speed ratio reached 0.97 while falls hit 100% and episode length
    #: collapsed to 230, then both backed off, over and over.
    #:
    #: 0.7, because the median moving command is then 0.49 m/s and the policy already
    #: achieves 0.45-0.50. The gates are therefore passable on day one, which is the whole
    #: point: a curriculum whose first rung is out of reach is just a smaller fixed
    #: envelope. The earlier attempt at 0.45 failed for the opposite reason, but note it was
    #: measured against the OLD crushed envelope, where 0.45 difficulty meant a 0.17 m/s
    #: command that was cheaper to ignore than to follow. On this envelope it means 0.31.
    #: 0.75, which is also the floor: start at the easiest level worth training on and
    #: earn everything above it. Was 0.7, which sat BELOW difficulty_min once the floor was
    #: raised, so the very first command draw silently violated the floor.
    difficulty_init: float = 0.75
    #: The floor for demotion and for graduate recycling, not a starting point.
    #:
    #: 0.75, raised from 0.5 after 0.5 was measured to be a trap. A curriculum floor is a
    #: promise about the EASIEST COMMAND YOU ARE WILLING TO TRAIN ON, and at difficulty 0.5
    #: the median moving command is 0.37 m/s, which is the exact crawl the command-envelope
    #: fix existed to eliminate.
    #:
    #: What happened (logbook E20): with 20 s episodes a fall becomes likely in almost any
    #: episode, demotion (-0.10) fires more often than promotion (+0.05), and every
    #: environment slid to the floor within 150 iterations and stayed there for the rest of
    #: the run. The curriculum was working exactly as specified; the specification let it
    #: park somewhere we already knew was useless. At 0.75 the floor is a 0.53 m/s median,
    #: still inside the reference clip range, so the worst case is a slow walk and not a
    #: shuffle.
    difficulty_min: float = 0.75
    difficulty_promote: float = 0.05
    difficulty_demote: float = 0.10
    #: Promotion gates, judged per COMMAND SEGMENT (not per episode: with mid-episode
    #: resampling an episode spans several commands and a fall would be misattributed).
    #: Only steps older than 1 s of command age count, so transition transients are free.
    promote_vx_err: float = 0.25
    promote_vy_err: float = 0.15
    promote_wz_err: float = 0.25
    #: The walk-these-ways rule: commands do not get harder until the gait is clean.
    #:
    #: 0.68, measured, after 0.80 turned out to be the same deadlock its own comment warned
    #: about. Over the last 400 iterations of the envelope run the gait match ran 0.66 to
    #: 0.73, mean 0.700, and never once touched 0.80. With the curriculum switched off that
    #: cost nothing and so went unnoticed; switching it back on at 0.80 would have pinned
    #: difficulty at its initial value for the entire run.
    #:
    #: The scale is not 0 to 1. A humanoid standing on both feet scores 0.60, because the
    #: schedule wants stance 60% of the time and a planted foot supplies exactly that much
    #: agreement for free. So the real band is 0.60 (standing) to 1.0 (perfect alternation),
    #: and 0.700 sits a quarter of the way up it. 0.68 admits a policy that is genuinely
    #: stepping while still excluding the two-footed brace, which is all this gate was ever
    #: for. It rises on its own as the policy improves, since the gate is a floor and not a
    #: target.
    promote_gait_match: float = 0.68
    promote_min_steps: int = 60

    # ---- penalty leniency curriculum ----
    # Four regulariser terms (ctrl, action_rate, vertical_vel, ang_vel_xy) are scaled by a
    # global factor p that starts at 0.5 and drifts up only while the policy survives:
    # multiplicative nudge per step, gated on an EMA of completed-episode length. A newborn
    # policy needs to flail; a competent one gets held to the full standard. Tracking, gait,
    # slip and flight terms are never scaled, because those block exploits that form early.
    penalty_scale_init: float = 1.0  # leniency OFF: no reference anneals penalties
    penalty_scale_min: float = 1.0
    penalty_rate_per_step: float = 1.0000125   # 1.0003 per 24-step iteration
    penalty_ema_promote: float = 550.0         # steps; above this, p drifts up
    penalty_ema_demote: float = 300.0          # below this, p drifts down
    #: Penalises vertical speed of the pelvis. Its absence is why the humanoid pogoed:
    #: measured at 10.0 cm peak-to-peak bounce against a human's 4-5 cm, with vertical speed
    #: peaking at 1.17 m/s. Nothing in the reward opposed it. The height term does not: it
    #: scores POSITION, and a body can oscillate hard through the target height while
    #: averaging exactly right. Penalising the velocity is what damps the oscillation.
    #: Standard in legged-locomotion setups and simply missing here.
    w_vertical_vel: float = -2.0  # G1 and T1 agree on -2.0 exactly
    #: Penalises roll and pitch rate. The companion to the above, and the term that opposes
    #: rocking the torso back and forth in place, which is the motion the split-stance gait
    #: used to fake forward progress.
    w_ang_vel_xy: float = -0.05
    #: Sideways speed, as its OWN term rather than folded into the forward one.
    #:
    #: `lin_err` used to be `(vx-cmd_x)^2 + (vy-cmd_y)^2` inside a single exponential, so
    #: forward and sideways error traded against each other and the shared kernel saturated:
    #: once the total error was large, reducing the sideways part bought almost nothing.
    #: Measured on a policy asked to walk straight forward, sideways speed reached an RMS of
    #: 0.26 m/s, 53% of its forward speed.
    w_lateral_vel: float = 0.0   # folded into per-axis tracking, T1 sigma
    sigma_lateral_vel: float = 0.05
    #: Holding a heading, not just a turn RATE.
    #:
    #: The command's third component is a rate, so "0" means "do not turn right now" and
    #: never "face the way you started". Heading error therefore accumulated for free: the
    #: same policy drifted 47 degrees over 11 seconds and finished 2.6 m to the side of the
    #: line it set off along, while scoring well on every velocity term.
    #:
    #: The desired heading integrates the commanded turn rate, so this is consistent with
    #: turning on request: ask for a turn and the target rotates with you; ask for none and
    #: it pins you to the direction you are facing. This is what a person does when told to
    #: walk forward, and it is the reason they arrive where they were pointed.
    #: Zero: replaced by the XBot/G1 heading COMMAND. A heading target is sampled with
    #: the command and the yaw command is recomputed every step as
    #: clip(0.5 * wrap(target - yaw), -1, 1), so holding a line is achieved through the
    #: existing yaw tracking term rather than a bespoke reward. The bespoke version was
    #: this project's invention and its first formulation was unlearnable.
    w_heading: float = 0.0
    #: New terms from the audit, weights verbatim from the references.
    w_orientation: float = -1.0        # G1: squared projected gravity xy
    w_swing_height: float = -20.0      # G1: (foot_z - target)^2 on swing feet
    #: 0.09, measured on our own retargeted mocap rather than carried over from G1's body.
    #:
    #: This was going to be made a function of commanded speed, following KSLC, whose second
    #: contribution is that fixed reward targets "cause robots to default to slower walking
    #: behaviours, resulting in overly conservative strategies at higher speeds". Measuring
    #: it first killed the idea, which is the reason to measure first.
    #:
    #: Across the eight clips, MEAN foot height while airborne fits
    #:     clearance = 0.063 + 0.020 * speed
    #: and a foot resting on the ground reads foot_z = 0.027 in this frame, so the target is
    #: 0.090 + 0.020 * speed. Over our whole 0 to 1.5 m/s envelope that moves 3 cm, and at
    #: weight -20 the old fixed 0.08 costs at most 0.03/step against a positive budget near
    #: 3.5. It is not what is stopping fast walking, and a speed-dependent target here would
    #: be a new mechanism bought for nothing.
    #:
    #: The trap worth recording: measured as PEAK swing height the same fit is
    #: 0.090 + 0.085 * speed, four times the slope, and implies a 0.30/step penalty that
    #: would have looked like a smoking gun. The reward penalises every airborne step, not
    #: the apex, so the mean is the quantity that matches it. Peak would have been the wrong
    #: statistic and would have justified a change the body does not need.
    swing_height_target: float = 0.09
    #: -3.0, not T1's -1.0. The weight has to be judged against what splaying BUYS, which
    #: is stability, worth a whole episode. At -1.0 a 0.67 m brace costs 0.22/step against a
    #: ~6 positive budget, 3.7%, which the policy will happily pay forever. At -3.0 it costs
    #: 0.66, comparable to the other shaping terms, and the band itself is free. If the
    #: eval's stance_width still sits above 0.45 after this run, the term is still too weak
    #: and that is the measurement that says so.
    w_feet_distance: float = -3.0
    feet_distance_min: float = 0.2
    #: MAXIMUM lateral foot separation. XBot rewards a band, min 0.2 AND max 0.5; only the
    #: minimum side was ported in the conformance pass, and the omission is measurable: the
    #: 500M-step policy stood 0.67 m wide against a human 0.10-0.15 and shuffled 11 cm per
    #: step inside that brace. Nothing in the reward opposed splaying, so splaying was free
    #: stability. This closes the band.
    feet_distance_max: float = 0.45
    w_dof_pos_limits: float = -5.0     # G1: penalty past 90% of joint range
    sigma_heading: float = 0.15

    # Exponential kernel widths. Larger is more forgiving.
    sigma_lin_vel: float = 0.25
    sigma_ang_vel: float = 0.25
    sigma_height: float = 0.02

    #: Air time in seconds that earns zero reward. Steps longer than this are rewarded,
    #: shorter are penalised. 0.25 s is roughly a human swing phase at walking pace.
    #: Only the STANDING fallback. With the clock running, the reward derives the swing
    #: target from the schedule as (1 - stance_fraction) / frequency. Kept equal to the
    #: nominal schedule's value so the two can never silently disagree again.
    air_time_target: float = 0.44

    # Termination and shaping. Defaults are derived from the prepared model at runtime,
    # so these are only fallbacks (see LocomotionTask.configure_for_model).
    target_height: float = 0.877  # standing pelvis height of the MimicKit humanoid
    terminate_height: float = 0.55  # below this it has effectively fallen
    max_tilt: float = 0.7  # body-frame gravity z above -0.7 means severely toppled
    #: End the episode if the torso tips past this (up-axis z below the value). cos(60 deg).
    #: A hard limit rather than only a soft penalty, because a soft one can always be paid
    #: for out of the velocity reward, which is exactly what happened.
    terminate_torso_upright: float = 0.5
    #: End the episode if the head drops below this fraction of its standing height.
    terminate_head_ratio: float = 0.65

    # Initial-state randomisation, so the policy sees a spread of starting states rather
    # than memorising one trajectory from one exact pose.
    joint_pos_noise: float = 0.02
    joint_vel_noise: float = 0.02


class LocomotionTask(Task):
    """Track a commanded planar velocity and turning rate while staying upright."""

    reward_term_names = (
        "lin_vel",
        "ang_vel",
        "upright",
        "height",
        "alive",
        "torque",
        "action_rate",
        "feet_air_time",
        "flight",
        "feet_slip",
        "torso_upright",
        "head_height",
        "gait_phase",
        "vertical_vel",
        "ang_vel_xy",
        "dof_vel",
        "dof_acc",
        "orientation",
        "swing_height",
        "feet_distance",
        "dof_pos_limits",
    )

    def __init__(self, config: LocomotionConfig | None = None,
                 seed: int = 0) -> None:
        self.cfg = config or LocomotionConfig()
        # The task's OWN generator, and it is never rebound afterwards. One task object is
        # shared by the training env, the evaluation env and the render env, and
        # `init_state` used to assign `self._rng = rng`, so it ended up pointing at
        # whichever env was constructed last. After the first video render, training's
        # mid-episode command redraws and the difficulty sampler were drawing from the
        # render env's stream, and the eval env's draws at eval N depended on how much
        # training had happened to redraw in between. That makes two evaluations of two
        # checkpoints incomparable, which is the assumption the entire best-checkpoint
        # selection rests on.
        self._rng = np.random.default_rng(seed)
        #: Terrain reader, or None on flat ground. Set by `configure_for_prepared`.
        self._terrain = None
        self._foot_probe_offsets = np.zeros((0, 2))
        self._limit_lo = np.full(28, -1e9)
        self._limit_hi = np.full(28, 1e9)
        #: qpos index per ACTUATOR. Set from the prepared model; the fallback assumes the
        #: identity mapping, which is wrong on this humanoid and is why it is overwritten.
        self._joint_qpos_adr = np.arange(7, 7 + 28)

    def set_joint_qpos_adr(self, adr: np.ndarray) -> None:
        """Where each actuator's joint angle lives in qpos. See PreparedModel."""
        self._joint_qpos_adr = np.asarray(adr, dtype=int)

    def set_joint_limits(self, lo: np.ndarray, hi: np.ndarray) -> None:
        """Joint range for the soft-limit penalty, from the prepared model."""
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # "90% of range" measured from the centre, G1's soft-limit convention.
        self._limit_lo = centre - half
        self._limit_hi = centre + half

    def configure_for_model(self, standing_height: float) -> None:
        """Derive height-dependent thresholds from the actual model.

        Called by the environment after model preparation. Keeps the task correct when the
        humanoid is swapped (the Phase 1 placeholder stood at 1.28 m, the Phase 2 model at
        0.877 m) instead of silently rewarding an impossible target height.
        """
        self.cfg.target_height = standing_height
        self.cfg.terminate_height = 0.62 * standing_height

    def configure_for_prepared(self, prepared) -> None:
        """Take the terrain reader and precompute the footprint probe offsets.

        The engine already calls this hook when a task defines it (vec_env.py:137), so
        terrain reaches the task without touching the engine's constructor.

        The probes are body-frame (dx, dy) points covering BOTH feet at the nominal pose:
        each foot's centre plus the four corners of its box, which is what the spawn lift
        needs. A single probe under the root understates the surface by enough to drive tens
        of body weights into the first frame -- see `reset_noise`.
        """
        self._terrain = getattr(prepared, "terrain", None)
        if self._terrain is None:
            return
        from humanoid_rl.terrain import foot_probe_offsets

        self._foot_probe_offsets = foot_probe_offsets(prepared)

    @property
    def task_obs_dim(self) -> int:
        # The command (vx, vy, yaw_rate), the gait clock as (sin, cos), the heading error
        # as (sin, cos), then the gait parameters and command age.
        #
        # SLOT ORDER IS FROZEN. Append only, never insert or redefine: a warm start copies
        # old weights into the leading columns, so inserting a slot silently shifts every
        # later one under weights trained for a different meaning. Appending is free and
        # verified bit-identical; inserting is a fresh start with no error message.
        #
        # The heading error has to be OBSERVABLE or the policy cannot correct it. The rest
        # of the observation is deliberately heading-invariant: body-frame velocities, and a
        # gravity vector that reveals roll and pitch but not yaw. So before this the policy
        # had no way to know it had drifted off course, and punishing it for drifting would
        # have been punishing it for something it could not perceive.
        #
        #   [0:3]  velocity command      [3:5] gait clock (sin, cos)
        #   [5:7]  heading error         [7]   stride frequency, Hz
        #   [8]    stance fraction       [9]   inter-foot phase offset
        #   [10]   clock authority       [11]  commanded body height
        #   [12]   command age
        return 13

    # ------------------------------------------------------------------ lifecycle

    def init_state(self, state: BatchState, rng: np.random.Generator) -> None:
        state.task_state["command"] = np.zeros((state.num_envs, 3))
        # Gait clock, one per environment, in cycles. Randomised at reset so the batch is
        # spread over the cycle rather than every environment stepping in lockstep.
        state.task_state["phase"] = rng.random(state.num_envs)
        # Lead-foot bookkeeping. A real walk trades which foot is in front on every step;
        # the rocking gait that prompted all this kept the right foot 36 cm in front for
        # 98% of the time while pattering both feet 7.4 times a second. Counting the swaps
        # is the one measurement that separates the two, and it needs history, so it lives
        # in task state rather than being derived from a single frame.
        # Heading the humanoid is being asked to face, integrating the commanded turn rate.
        state.task_state["desired_heading"] = np.zeros(state.num_envs)
        n = state.num_envs
        # Gait parameters, all sampled with the command and all OBSERVED. A parameter the
        # network cannot see is a constant it cannot adapt to.
        state.task_state["gait_freq"] = np.full(n, self.cfg.gait_frequency)
        state.task_state["stance_frac"] = np.full(n, self.cfg.stance_fraction)
        state.task_state["foot_offset"] = np.full(n, 0.5)
        state.task_state["clock_authority"] = np.ones(n)
        state.task_state["body_height"] = np.ones(n)
        state.task_state["command_age"] = np.zeros(n)
        state.task_state["hold_time"] = np.full(n, self.cfg.command_hold_range[1])
        state.task_state["command_changed"] = np.zeros(n, dtype=bool)
        # Radial difficulty per environment, and the per-segment score that gates it.
        state.task_state["difficulty"] = np.full(n, self.cfg.difficulty_init)
        state.task_state["seg_err"] = np.zeros((n, 3))     # sums of |vx|, |vy|, |wz| error
        state.task_state["seg_gait"] = np.zeros(n)         # sum of gait match
        state.task_state["seg_steps"] = np.zeros(n)        # scored steps in the segment
        state.task_state["last_terminated"] = np.zeros(n, dtype=bool)
        # Global penalty leniency and the survival EMA that drives it.
        state.task_state["penalty_scale"] = float(self.cfg.penalty_scale_init)
        state.task_state["episode_len_ema"] = 200.0
        # The task owns a generator so mid-episode redraws do not need one threaded in
        # through `on_batch_end`, which the engine calls without one. It is created once in
        # __init__ and deliberately NOT rebound here -- see the note there.
        state.task_state["lead"] = np.zeros(state.num_envs, dtype=np.int64)
        state.task_state["lead_swaps"] = np.zeros(state.num_envs)
        state.task_state["gait_steps"] = np.zeros(state.num_envs)

    def reset_batch(
        self, state: BatchState, indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        if indices.size == 0:
            return
        # A fall demotes the difficulty before the next command is drawn at the easier
        # level. A time-limit reset is not a failure and leaves difficulty alone.
        cfg = self.cfg
        fell = indices[state.task_state["last_terminated"][indices]]
        if fell.size:
            s_ = state.task_state["difficulty"]
            s_[fell] = np.maximum(s_[fell] - cfg.difficulty_demote, cfg.difficulty_min)
        # Survival EMA for the penalty leniency, fed by every ending episode.
        lengths = state.episode_step[indices].astype(float)
        ema = state.task_state["episode_len_ema"]
        for L in lengths:
            ema += 0.01 * (L - ema)
        state.task_state["episode_len_ema"] = ema

        self._draw_command(state, indices, rng)
        state.task_state["phase"][indices] = rng.random(indices.size)
        # Start from whatever way it happens to be facing, so the target is reachable.
        state.task_state["desired_heading"][indices] = state.heading[indices]
        state.task_state["lead_swaps"][indices] = 0.0
        state.task_state["gait_steps"][indices] = 0.0

    def reset_noise(
        self, state: BatchState, indices: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cfg = self.cfg
        k = indices.size
        nq, nv = state.qpos.shape[1], state.qvel.shape[1]
        qpos_noise = np.zeros((k, nq))
        qvel_noise = np.zeros((k, nv))
        # Joints only. Perturbing the free root would drop the humanoid through the floor
        # or spawn it already rotated, neither of which is useful randomisation.
        qpos_noise[:, 7:] = rng.normal(0.0, cfg.joint_pos_noise, size=(k, nq - 7))
        qvel_noise[:, 6:] = rng.normal(0.0, cfg.joint_vel_noise, size=(k, nv - 6))

        # On terrain, scatter the spawn across the field and lift it onto the surface.
        #
        # The scatter IS the terrain randomisation: one static field sampled at thousands of
        # places is what gives 4096 envs varied ground, and it is why no per-env heightfield
        # and no terrain curriculum is needed.
        #
        # The lift is the load-bearing part, and it is not "add the height under the root".
        # Measured on the chosen field: spawning at the unmodified nominal root height drives
        # 30,295 N under one foot on the first frame -- 61.7x body weight against a 9.82 N
        # contact threshold, so `foot_contact` is trivially true and `foot_force`, which the
        # tasks read in body-weight units, is off by a factor of sixty in the first
        # observation of every episode. Offsetting by the height under the ROOT still leaves
        # 32.2 BW, because a 17.7 x 9.0 cm box foot straddles cells the root does not.
        # Taking the maximum over both footprints gives a median of 0.0 N and a peak of
        # 1.80 BW. That is the difference between training on terrain and training on an
        # instrumentation artefact.
        if self._terrain is not None:
            s = self._terrain.spawn_half_extent
            x = rng.uniform(-s, s, k)
            y = rng.uniform(-s, s, k)
            qpos_noise[:, 0] = x
            qpos_noise[:, 1] = y
            qpos_noise[:, 2] = self._terrain.probe_max(x, y, self._foot_probe_offsets)
        return qpos_noise, qvel_noise

    def _draw_command(
        self, state: BatchState, indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Draw a velocity command and its gait parameters for the given environments.

        Direction is sampled uniformly on a circle and speed uniformly over the AREA of an
        elliptical envelope, so every heading is practised equally often. Sampling a box
        over (vx, vy) instead concentrates 52.7% of commands in the forward sector and 3.7%
        in the lateral one, and caps a 45 degree diagonal at the corner distance.
        """
        cfg = self.cfg
        k = indices.size
        theta = rng.uniform(-np.pi, np.pi, k)
        # A share of commands is redrawn inside a forward cone. Applied to the ANGLE only,
        # before the envelope is evaluated, so the elliptical reach and the deadband still
        # apply exactly as they do to a uniform draw and no direction becomes unreachable.
        if cfg.forward_bias_prob > 0.0:
            half = np.deg2rad(cfg.forward_cone_deg)
            theta = np.where(rng.random(k) < cfg.forward_bias_prob,
                             rng.uniform(-half, half, k), theta)
        # sqrt makes the radius uniform by area rather than by radius. The lower bound
        # keeps every MOVING command at >= 50% of the scaled envelope: tiny commands were
        # cheap to ignore (the exponential kernel forgives a 0.3 m/s error almost fully),
        # and deliberate standing practice is already provided by zero_command_prob, so
        # near-zero moving commands taught nothing at a real cost.
        magnitude = np.sqrt(rng.uniform(0.25, 1.0, k))
        forward_reach = np.where(np.cos(theta) >= 0.0, cfg.lin_vel_x_range[1],
                                 abs(cfg.lin_vel_x_range[0]))
        lateral_reach = cfg.lin_vel_y_range[1]
        reach = 1.0 / np.sqrt(
            (np.cos(theta) / forward_reach) ** 2 + (np.sin(theta) / lateral_reach) ** 2
        )
        # The whole envelope scales by the environment's difficulty, so every direction
        # gets harder together and the per-sector mix never tilts.
        speed = magnitude * reach * state.task_state["difficulty"][indices]
        command = np.stack(
            [speed * np.cos(theta), speed * np.sin(theta),
             rng.uniform(*cfg.ang_vel_yaw_range, size=k)
             * state.task_state["difficulty"][indices]], axis=1
        )

        # Stand still, either by an explicit draw or because the speed landed in the
        # deadband. Both collapse to exactly zero so there is no near-still band that can
        # only be satisfied by marching on the spot.
        still = (rng.random(k) < cfg.zero_command_prob) | (
            np.linalg.norm(command[:, :2], axis=1) < cfg.command_deadband
        )
        command[still] = 0.0
        state.task_state["command"][indices] = command
        # XBot/G1 heading command: a target direction is drawn with the command, and the
        # yaw-rate command is recomputed every step from the wrapped error, so line-holding
        # flows through the ordinary yaw tracking term instead of a bespoke reward. Turning
        # commands come from moving the TARGET, which the sampler does here.
        state.task_state["desired_heading"][indices] = (
            state.heading[indices] + command[:, 2] / 1.0 * 2.0
        )

        # Stride frequency follows commanded speed by Inman's square-root law, so cadence
        # and speed stay consistent instead of the clock demanding one cadence at every pace.
        planar = np.linalg.norm(command[:, :2], axis=1)
        nominal = cfg.gait_frequency * np.sqrt(
            np.maximum(planar, 0.3) / cfg.gait_speed_anchor
        )
        frequency = np.clip(nominal, *cfg.gait_frequency_range) * rng.uniform(
            *cfg.gait_frequency_jitter, size=k
        )
        frequency[still] = 0.0          # zero frequency IS the standing condition
        state.task_state["gait_freq"][indices] = frequency
        state.task_state["stance_frac"][indices] = rng.uniform(*cfg.stance_fraction_range, k)
        state.task_state["foot_offset"][indices] = rng.uniform(*cfg.foot_phase_offset_range, k)
        state.task_state["clock_authority"][indices] = (
            rng.random(k) >= cfg.free_gait_prob
        ).astype(float)
        state.task_state["body_height"][indices] = rng.uniform(*cfg.body_height_range, k)
        state.task_state["command_age"][indices] = 0.0
        state.task_state["hold_time"][indices] = rng.uniform(*cfg.command_hold_range, k)
        state.task_state["seg_err"][indices] = 0.0
        state.task_state["seg_gait"][indices] = 0.0
        state.task_state["seg_steps"][indices] = 0.0

    def observe_batch(self, state: BatchState, out: np.ndarray) -> None:
        out[:, :3] = state.task_state["command"]
        # The clock enters as (sin, cos) rather than as a raw fraction, so it is continuous
        # where the cycle wraps from 1 back to 0. A raw phase has a discontinuity there,
        # and the policy would see a jump every stride at exactly the moment it matters.
        angle = 2.0 * np.pi * state.task_state["phase"]
        standing = state.task_state["gait_freq"] <= 0.0
        # Masked to (0, 0) when standing, so "no schedule" is a distinguishable input rather
        # than a phase that keeps rotating while nothing is asked of the feet.
        out[:, 3] = np.where(standing, 0.0, np.sin(angle))
        out[:, 4] = np.where(standing, 0.0, np.cos(angle))
        # As (sin, cos) for the same reason as the clock: an angle error expressed as a raw
        # number jumps by 2*pi at the wrap, right where the policy most needs it smooth.
        error = self._heading_error(state)
        out[:, 5] = np.sin(error)
        out[:, 6] = np.cos(error)
        out[:, 7] = state.task_state["gait_freq"]
        out[:, 8] = state.task_state["stance_frac"]
        out[:, 9] = state.task_state["foot_offset"]
        out[:, 10] = state.task_state["clock_authority"]
        out[:, 11] = state.task_state["body_height"]
        # Saturating at 2 s: the policy needs to know "the command just changed, expect a
        # transient" versus "steady state", not how many minutes have passed.
        out[:, 12] = np.minimum(state.task_state["command_age"], 2.0) / 2.0

    # ------------------------------------------------------------------ reward

    def reward_batch(self, state: BatchState, terms: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        command = state.task_state["command"]

        # Body-frame velocities are already computed by the engine, so "forward" means
        # forward no matter which way the humanoid is currently facing.
        vx = state.lin_vel_body[:, 0]
        vy = state.lin_vel_body[:, 1]
        yaw_rate = state.ang_vel_body[:, 2]

        lin_err = (vx - command[:, 0]) ** 2 + (vy - command[:, 1]) ** 2
        ang_err = (yaw_rate - command[:, 2]) ** 2

        # Uprightness from the body-frame gravity vector: -1 in z when perfectly upright,
        # so negating and clipping maps it to 1 upright, 0 horizontal.
        r_upright = np.clip(-state.gravity_body[:, 2], 0.0, 1.0)

        # Target height follows the commanded fraction, which is what makes crouching a
        # point in the command space rather than a future retrain.
        # Height ABOVE THE GROUND UNDER THE ROOT, not world z. `ground_z` is identically 0.0
        # on a plane, so this line is bit-identical to the flat version there; on terrain it
        # is the difference between measuring posture and measuring the hill. Standing on a
        # 2.6 cm rise otherwise reads as 2.6 cm too tall and is penalised for it.
        height_err = (
            (state.root_height - state.ground_z)
            - cfg.target_height * state.task_state["body_height"]
        ) ** 2

        torque_cost = np.sum(np.square(state.torque), axis=1)
        joint_vel = state.qvel[:, 6:6 + state.torque.shape[1]]
        dof_vel_cost = np.sum(np.square(joint_vel), axis=1)
        dof_acc_cost = np.sum(
            np.square((joint_vel - state.prev_joint_vel) / state.dt), axis=1
        )
        action_rate = np.sum(np.square(state.action - state.prev_action), axis=1)

        # Air-time reward: paid once per footfall, scaled by how long that foot swung.
        # Gated on a non-zero command so a policy asked to stand still is not pushed into
        # marching on the spot.
        # Swing duration the SCHEDULE implies, rather than a constant.
        #
        # This was a live contradiction: the constant asked for 0.25 s while the clock at
        # 0.9 Hz and 0.6 stance implies (1 - 0.6) / 0.9 = 0.44 s. Two reward terms pulled
        # in opposite directions, `feet_air_time` ran at a net penalty, and `gait_phase` sat
        # at 0.53 against a chance level of 0.5. The clock was losing to a number nobody had
        # reconciled with it.
        frequency = state.task_state["gait_freq"]
        air_target = np.where(
            frequency > 0.0,
            (1.0 - state.task_state["stance_frac"]) / np.maximum(frequency, 1e-6),
            cfg.air_time_target,
        )
        moving = frequency > 0.0
        air_time_reward = np.sum(
            (state.foot_air_time - air_target[:, None]) * state.foot_first_contact, axis=1
        ) * moving

        # Neither foot down. Brief flight is normal in running, but for Phase 2 walking it
        # is almost always the policy discovering a hop or a dive.
        flight = (~state.foot_contact).all(axis=1).astype(np.float64)
        # The schedule, needed by both the flight term below and the gait term further down.
        expected = self._expected_stance(state)

        terms[:, 0] = cfg.w_lin_vel * np.exp(-lin_err / cfg.sigma_lin_vel)
        terms[:, 1] = cfg.w_ang_vel * np.exp(-ang_err / cfg.sigma_ang_vel)
        terms[:, 2] = cfg.w_upright * r_upright
        terms[:, 3] = cfg.w_height * np.exp(-height_err / cfg.sigma_height)
        terms[:, 4] = cfg.w_alive
        # Four regulariser penalties ride the leniency scale: a newborn policy needs to
        # flail, and these are precisely the terms that fine flailing. Tracking, gait,
        # slip and flight are never scaled; those block exploits that form early.
        terms[:, 5] = cfg.w_torque * torque_cost
        terms[:, 6] = cfg.w_action_rate * action_rate
        # Foot slip: squared horizontal speed of each loaded foot. Squared rather than
        # linear so that a genuine skid is punished far more than the small unavoidable
        # motion of a foot rolling through stance.
        slip = np.sum(
            np.sum(np.square(state.foot_lin_vel[:, :, :2]), axis=2) * state.foot_contact,
            axis=1,
        )

        terms[:, 7] = cfg.w_feet_air_time * air_time_reward
        # Penalise only UNSCHEDULED flight. The flat penalty forbade running and jumping
        # outright: with a stance fraction below 0.5 the two stance windows stop overlapping
        # and the schedule itself asks for a flight phase, which the old term would then
        # punish. Identical to the old behaviour whenever the schedule asks for none.
        expected_flight = np.clip(1.0 - expected.sum(axis=1), 0.0, 1.0)
        terms[:, 8] = cfg.w_flight * np.maximum(0.0, flight - expected_flight)
        terms[:, 9] = cfg.w_feet_slip * slip
        terms[:, 10] = cfg.w_torso_upright * np.clip(state.torso_upright, 0.0, 1.0)
        terms[:, 11] = cfg.w_head_height * np.clip(state.head_height_ratio, 0.0, 1.0)

        # Gait schedule. Each foot scores 1 when its actual contact matches what the clock
        # asks for, averaged over the two feet, so the term is in [0, 1] like the other
        # shaped rewards. A rocking gait keeps both feet loaded and fails every swing
        # window, which is exactly the behaviour this exists to make unprofitable.
        actual = state.foot_contact[:, :2].astype(np.float64)
        # Scaled by clock authority, so an environment told the schedule is void is not
        # also punished by it. This is the mechanism that lets a future jump or crouch stop
        # fighting the walking rhythm, and the policy is TOLD (slot 10) rather than left to
        # infer it from a reward that silently switched off.
        terms[:, 12] = (
            cfg.w_gait_phase
            * state.task_state["clock_authority"]
            * (1.0 - np.abs(expected - actual)).mean(axis=1)
        )

        # Vertical bounce and torso rocking. Both are velocity penalties, because position
        # penalties do not damp an oscillation that averages out.
        terms[:, 13] = cfg.w_vertical_vel * np.square(state.qvel[:, 2])
        terms[:, 14] = cfg.w_ang_vel_xy * np.sum(np.square(state.ang_vel_body[:, :2]), axis=1)

        # Reference terms, adopted verbatim from the conformance audit.
        terms[:, 15] = cfg.w_dof_vel * dof_vel_cost
        terms[:, 16] = cfg.w_dof_acc * dof_acc_cost
        # G1's orientation: squared projected gravity in the horizontal plane. Replaces the
        # separate upright and torso_upright positive bonuses with one penalty.
        terms[:, 17] = cfg.w_orientation * np.sum(np.square(state.gravity_body[:, :2]), axis=1)
        # G1's swing height: feet should swing at a consistent clearance, penalised only
        # while the foot is airborne.
        # Clearance above the ground UNDER EACH FOOT. A swing foot crossing a rise is not
        # lifting higher, it has less room, and pricing that as a fault would teach the
        # policy to drag its feet uphill. Zero on flat, so this is exact there.
        foot_z = state.key_body_pos[:, :2, 2] - state.key_ground_z[:, :2]
        swinging = ~state.foot_contact[:, :2]
        terms[:, 18] = cfg.w_swing_height * np.sum(
            np.square(foot_z - cfg.swing_height_target) * swinging, axis=1
        )
        # T1's feet-distance floor: standing with the feet crossed or nearly touching is
        # unstable and precedes self-collisions.
        separation = _stance_width(state)
        terms[:, 19] = cfg.w_feet_distance * (
            np.clip(cfg.feet_distance_min - separation, 0.0, 0.1)
            + np.clip(separation - cfg.feet_distance_max, 0.0, 0.3)
        )
        # G1's soft joint-limit penalty at 90% of range.
        #
        # Indexed through actuator_qpos_adr, NOT qpos[7:7+nu]. The limits come from
        # actuator_ctrlrange in ACTUATOR order, and on this model actuator order is not qpos
        # order: hip_y and hip_z are transposed on both legs, so 4 of 28 were mismatched.
        # The effect was silent and pointed the wrong way for us: hip_y's true +-2.44 rad
        # range was scored against hip_z's +-1.05 rad one, so deep hip flexion, which is
        # exactly what a long stride needs, read as a limit violation at weight -5.0.
        angles = state.qpos[:, self._joint_qpos_adr]
        overflow = (
            np.clip(self._limit_lo * 0.9 - angles, 0, None)
            + np.clip(angles - self._limit_hi * 0.9, 0, None)
        )
        terms[:, 20] = cfg.w_dof_pos_limits * np.sum(overflow, axis=1)
        # Only-positive total, legged_gym's oldest trick and the field's universal answer
        # to penalties overwhelming a newborn policy: the floor self-anneals, because a
        # competent policy's positives dwarf its penalties and the clip stops binding.
        # Per-term logging keeps the raw values, so the clip is visible, not hidden.
        return np.maximum(terms.sum(axis=1), 0.0)

    # ------------------------------------------------------------------ termination

    def _maybe_promote(self, state: BatchState, indices: np.ndarray) -> None:
        """Raise difficulty for environments whose finished segment met every gate.

        Promotion requires demonstrated competence on THIS command: tracking errors under
        the thresholds AND a clean gait (the walk-these-ways rule: commands do not harden
        until the rhythm is right). At the top, promotion resamples difficulty uniformly
        instead of saturating, the terrain-curriculum "graduate recycling" trick, so easy
        commands never leave the training distribution.
        """
        cfg = self.cfg
        ts = state.task_state
        steps = ts["seg_steps"][indices]
        enough = steps >= cfg.promote_min_steps
        if not enough.any():
            return
        idx = indices[enough]
        n = ts["seg_steps"][idx]
        mean_err = ts["seg_err"][idx] / n[:, None]
        mean_gait = ts["seg_gait"][idx] / n
        good = (
            (mean_err[:, 0] < cfg.promote_vx_err)
            & (mean_err[:, 1] < cfg.promote_vy_err)
            & (mean_err[:, 2] < cfg.promote_wz_err)
            & (mean_gait >= cfg.promote_gait_match)
        )
        winners = idx[good]
        if winners.size == 0:
            return
        s = ts["difficulty"]
        graduated = winners[s[winners] >= 1.0 - 1e-9]
        s[winners] = np.minimum(s[winners] + cfg.difficulty_promote, 1.0)
        if graduated.size:
            s[graduated] = self._rng.uniform(cfg.difficulty_min, 1.0, graduated.size)

    def _heading_error(self, state: BatchState) -> np.ndarray:
        """Signed angle from where it faces to where it should, wrapped to [-pi, pi].

        Wrapping matters: without it, a humanoid 179 degrees off would be told to turn 181
        degrees the long way round.
        """
        error = state.task_state["desired_heading"] - state.heading
        return (error + np.pi) % (2.0 * np.pi) - np.pi

    def _expected_stance(self, state: BatchState) -> np.ndarray:
        """(N, 2) target contact for each foot at the current phase: 1 stance, 0 swing.

        The two feet run half a cycle apart, which is what "alternating" means. A rocking
        gait keeps both feet loaded and therefore fails the swing half for both of them, no
        matter how fast it patters.
        """
        phase = state.task_state["phase"]
        offset = state.task_state["foot_offset"]
        stance = state.task_state["stance_frac"]
        width = self.cfg.stance_transition_width

        def window(u: np.ndarray) -> np.ndarray:
            """Soft stance indicator, 1 inside the window and 0 outside, wrap-safe.

            The old version was a hard boolean, which put a step discontinuity in the reward
            at both ends of every cycle. Softening it costs nothing and gives a gradient
            through the transition, which is where foot placement is actually decided.
            """
            centred = (u - stance / 2.0 + 0.5) % 1.0 - 0.5
            return np.clip((stance / 2.0 - np.abs(centred)) / width + 0.5, 0.0, 1.0)

        expected = np.stack([window(phase % 1.0), window((phase + offset) % 1.0)], axis=1)
        # Zero frequency IS the standing condition, so there is no separate test on the
        # command. Standing became a point in the gait space rather than a special case.
        expected[state.task_state["gait_freq"] <= 0.0] = 1.0
        return expected

    def mirror_task_obs(self, task_obs):
        """Command is (vx, vy, yaw_rate). Reflection keeps forward speed, negates the
        sideways component and reverses the turn direction."""
        out = task_obs.clone()
        out[..., 1] = -task_obs[..., 1]
        out[..., 2] = -task_obs[..., 2]
        # Mirroring swaps the legs, so the observed clock (which drives the LEFT foot) must
        # advance to where the right foot was, a rotation by the phase offset d rather than
        # a flat half cycle. Negating sin and cos is a rotation by exactly half a cycle and
        # is therefore correct only when d is exactly 0.5; at d = 0.45 it is off by 18
        # degrees of phase on every augmented sample. Reduces to the negation at d = 0.5.
        # torch is imported lazily: this module is otherwise pure numpy and runs on the
        # physics threads, where importing torch at module scope would be dead weight.
        import torch

        phi = 2.0 * np.pi * task_obs[..., 9]
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        out[..., 3] = task_obs[..., 3] * cos_phi + task_obs[..., 4] * sin_phi
        out[..., 4] = task_obs[..., 4] * cos_phi - task_obs[..., 3] * sin_phi
        out[..., 9] = 1.0 - task_obs[..., 9]
        # Mirroring negates a heading error: being 10 degrees left of target becomes 10
        # degrees right of it. sin flips, cos does not.
        out[..., 5] = -task_obs[..., 5]
        return out

    def eval_metrics(self, state: BatchState) -> dict[str, float]:
        """How well the commanded velocity is actually being tracked.

        Reported in physical units (m/s and rad/s) rather than as the shaped exponential
        reward, because "0.08 m/s of tracking error" is interpretable and "reward 0.72"
        is not.
        """
        command = state.task_state["command"]
        lin_err = np.linalg.norm(state.lin_vel_body[:, :2] - command[:, :2], axis=1)
        ang_err = np.abs(state.ang_vel_body[:, 2] - command[:, 2])
        return {
            "lin_vel_error": float(lin_err.mean()),
            "ang_vel_error": float(ang_err.mean()),
            "commanded_speed": float(np.linalg.norm(command[:, :2], axis=1).mean()),
            "upright": float(np.clip(-state.gravity_body[:, 2], 0.0, 1.0).mean()),
            "torso_upright": float(state.torso_upright.mean()),
            "head_height_ratio": float(state.head_height_ratio.mean()),
            # Metres per second of skid under loaded feet. Should trend toward zero; a
            # visibly sliding gait shows up here long before it is obvious in a video.
            # Gait asymmetry: how unevenly the two legs share the work. 1.0 is perfectly
            # even. Tracked because a one-sided gait (one leg driving, the other a passive
            # strut) is invisible to every other metric here, all of which aggregate over
            # both legs. It was first caught by a human watching a video, at a measured
            # left/right stance ratio of 1.90.
            #
            # Defined over SWING time (foot in the air), not stance time. The stance-time
            # version scored a perfect 1.0 for a humanoid that simply never lifted either
            # foot, since both contact fractions were then 1.0, and a policy under a strong
            # symmetry loss duly found that solution: a 61 cm wide two-footed brace, dragged
            # forward by ground slip, reporting 0.91 symmetry at 0.06 m of travel per step.
            # With swing time, never lifting a foot scores 0 rather than 1.
            "gait_symmetry": float(_ratio(*(1.0 - state.foot_contact.mean(0)[:2]))),
            # How often the front foot changes, per second. In a real walk this equals the
            # step rate, because every step trades the lead. The rocking gait scored 0.42
            # against 7.4 foot strikes a second, a ratio of 17 to 1, while every other
            # metric here was satisfied. This is the number that tells walking from rocking.
            # Peak-to-peak vertical travel of the pelvis is what a viewer reads as bouncing,
            # but it needs history; the instantaneous vertical SPEED is the same defect seen
            # per-frame and is what the reward actually penalises, so report that and convert.
            "vertical_bounce": float(np.abs(state.qvel[:, 2]).mean() / max(2.0 * np.pi * 1.8, 1e-6) * 2.0),
            "vertical_speed": float(np.abs(state.qvel[:, 2]).mean()),
            # Degrees off the commanded facing. A policy told to walk forward should hold
            # this near zero; the one that prompted this drifted 47 degrees in 11 seconds.
            "heading_error_deg": float(np.degrees(np.abs(self._heading_error(state))).mean()),
            "lateral_speed": float(np.abs(state.lin_vel_body[:, 1]).mean()),
            "difficulty_median": float(np.median(state.task_state["difficulty"])),
            "penalty_scale": float(state.task_state["penalty_scale"]),
            "episode_len_ema": float(state.task_state["episode_len_ema"]),
            "lead_swaps_per_sec": float(
                (
                    state.task_state["lead_swaps"]
                    / np.maximum(state.task_state["gait_steps"], 1.0)
                ).mean()
                / state.dt
            ),
            # Fraction of time both feet are loaded. Real human walking is 0.20 to 0.25;
            # 1.0 means standing. Unlike a contact ratio this cannot be maximised by
            # refusing to step, which is exactly why it is here.
            "double_support": float(
                (state.foot_contact[:, 0] & state.foot_contact[:, 1]).mean()
            ),
            # Lateral distance between the feet, in the humanoid's own heading frame so a
            # turn does not read as a widening stance. Human walking is 0.10 to 0.15 m; a
            # bracing posture splays far wider and is otherwise easy to miss on video.
            "stance_width": float(_stance_width(state).mean()),
            "foot_slip_speed": float(
                (
                    np.linalg.norm(state.foot_lin_vel[:, :, :2], axis=2) * state.foot_contact
                ).sum(axis=1).mean()
            ),
            **self._per_direction_metrics(state),
        }

    def _per_direction_metrics(self, state: BatchState) -> dict[str, float]:
        """Tracking split by what was actually ASKED for: forward, back, sideways, turning.

        Every other metric here averages over whatever command mix the evaluation happened
        to draw, which cannot answer the only questions a person actually asks: how fast
        does it walk forward, does it turn when told to, can it walk backwards at all. A
        single `lin_vel_error` of 0.33 m/s is consistent with tracking forward perfectly and
        ignoring every turn, and with the reverse.

        Reported as SUMS divided by the environment count, not as means over the environments
        in each category. That looks odd and is deliberate. A category can be momentarily
        empty (backward commands are ~7.5% of draws, so with 64 environments there is a real
        chance of none at some step), and a per-category mean would then be undefined; a
        zero would silently drag the average down instead. With sums, `actual / target` is
        the correctly pooled tracking ratio however the category population moves, and
        `target / share` recovers the true mean command. Nothing can divide by zero because
        the division happens once, at display time, in humanoid_rl/command_report.py.

        Sign convention: `actual` is positive when moving the way it was told to and
        NEGATIVE when moving the opposite way. A policy that reverses under a backward
        command reads as negative here rather than as a small positive error.
        """
        cmd = state.task_state["command"]
        vx, vy = state.lin_vel_body[:, 0], state.lin_vel_body[:, 1]
        wz = state.ang_vel_body[:, 2]
        n = max(cmd.shape[0], 1)

        moving = np.linalg.norm(cmd[:, :2], axis=1) > 1e-6
        angle = np.arctan2(cmd[:, 1], cmd[:, 0])
        # Quadrants of the command direction, 90 degrees wide and centred on each axis.
        fwd = moving & (np.abs(angle) <= np.pi / 4)
        back = moving & (np.abs(angle) >= 3 * np.pi / 4)
        side = moving & ~fwd & ~back
        # 0.1 rad/s, the same threshold the reward terms use to decide "is it turning".
        turn = np.abs(cmd[:, 2]) > 0.1
        lat_sign = np.sign(cmd[:, 1])
        yaw_sign = np.sign(cmd[:, 2])

        out: dict[str, float] = {}
        for name, mask, target, actual in (
            ("fwd", fwd, cmd[:, 0], vx),
            ("back", back, -cmd[:, 0], -vx),
            ("side", side, np.abs(cmd[:, 1]), vy * lat_sign),
            ("turn", turn, np.abs(cmd[:, 2]), wz * yaw_sign),
        ):
            out[f"cmd_{name}_share"] = float(mask.sum() / n)
            out[f"cmd_{name}_target"] = float((target * mask).sum() / n)
            out[f"cmd_{name}_actual"] = float((actual * mask).sum() / n)
        return out

    def on_batch_end(self, state: BatchState, metrics: dict[str, float]) -> dict[str, float]:
        """Advance the gait clock and count how often the leading foot changes."""
        state.task_state["phase"] = (
            state.task_state["phase"] + state.task_state["gait_freq"] * state.dt
        ) % 1.0
        state.task_state["command_age"] += state.dt
        # Penalty leniency drifts on survival: up while the EMA of episode length shows a
        # policy that lives, down while it shows one that cannot. Multiplicative and slow,
        # reaching full strength around mid-run if progress holds.
        cfg_ = self.cfg
        pscale = state.task_state["penalty_scale"]
        if state.task_state["episode_len_ema"] > cfg_.penalty_ema_promote:
            pscale *= cfg_.penalty_rate_per_step
        elif state.task_state["episode_len_ema"] < cfg_.penalty_ema_demote:
            pscale /= cfg_.penalty_rate_per_step
        state.task_state["penalty_scale"] = float(np.clip(
            pscale, cfg_.penalty_scale_min, 1.0
        ))

        # The desired heading rotates at exactly the commanded turn rate, so asking for a
        # turn moves the target with you and asking for none pins it. That keeps the new
        # term consistent with turning rather than fighting it.
        # XBot's heading-command update, verbatim mechanism: the yaw command is half the
        # wrapped heading error, clipped to the trained range. This replaces both previous
        # heading formulations (the integrating one, which was unlearnable, and the holding
        # one, which was this project's invention): the references drive heading through
        # the yaw COMMAND and the ordinary tracking term, and now so does this.
        error = self._heading_error(state)
        state.task_state["command"][:, 2] = np.clip(0.5 * error, -1.0, 1.0)

        # Fore-aft foot positions in each humanoid's own heading frame, so a turn does not
        # register as a change of lead.
        rel = state.key_body_pos[:, :2, :2] - state.root_pos[:, None, :2]
        c, s_ = np.cos(-state.heading), np.sin(-state.heading)
        fore = c[:, None] * rel[:, :, 0] - s_[:, None] * rel[:, :, 1]
        lead = np.argmax(fore, axis=1)
        state.task_state["lead_swaps"] += lead != state.task_state["lead"]
        state.task_state["lead"] = lead
        state.task_state["gait_steps"] += 1.0

        # Score the running command segment. Only steps past 1 s of command age count, so
        # the transient after a command change is free, and standing segments score nothing.
        cfg = self.cfg
        age_ok = state.task_state["command_age"] > 1.0
        moving = np.linalg.norm(state.task_state["command"][:, :2], axis=1) > 1e-9
        scored = age_ok & moving
        if scored.any():
            err = np.stack([
                np.abs(state.lin_vel_body[:, 0] - state.task_state["command"][:, 0]),
                np.abs(state.lin_vel_body[:, 1] - state.task_state["command"][:, 1]),
                np.abs(state.ang_vel_body[:, 2] - state.task_state["command"][:, 2]),
            ], axis=1)
            expected = self._expected_stance(state)
            match = (1.0 - np.abs(expected - state.foot_contact[:, :2])).mean(axis=1)
            state.task_state["seg_err"][scored] += err[scored]
            state.task_state["seg_gait"][scored] += match[scored]
            state.task_state["seg_steps"][scored] += 1.0

        # Redraw commands mid-episode for whichever environments have held theirs long
        # enough. The gait phase, pose and desired heading deliberately survive: only the
        # command changes, which is exactly what a navigation layer above will do.
        due = np.flatnonzero(state.task_state["command_age"] >= state.task_state["hold_time"])
        if due.size:
            # A segment that ran its full hold ended WITHOUT a fall, so it is eligible for
            # promotion. Judged before the redraw wipes the accumulators.
            self._maybe_promote(state, due)
            self._draw_command(state, due, self._rng)
            # Flag them as truncated for one step so GAE bootstraps V(s) across the change.
            # Without this the critic has to predict a return across a reward discontinuity
            # it cannot see coming, which corrupts the advantages of the whole preceding
            # segment. Resampling without this is worse than not resampling at all.
            state.task_state["command_changed"][:] = False
            state.task_state["command_changed"][due] = True
        else:
            state.task_state["command_changed"][:] = False
        return {}

    def terminated_batch(self, state: BatchState) -> np.ndarray:
        cfg = self.cfg
        # The fall test, against the ground under the root rather than against z = 0. This is
        # the most consequential of the terrain fixes: on a field with 5.25 cm of relief the
        # world-z version terminates a perfectly upright humanoid standing in a dip, and
        # forgives one that has collapsed on a rise. Both errors are silent, both look like
        # policy behaviour in the fall rate, and fall rate is the headline metric.
        fallen = (state.root_height - state.ground_z) < cfg.terminate_height
        # gravity_body z near -1 is upright; rising above -max_tilt means it has toppled.
        toppled = state.gravity_body[:, 2] > -cfg.max_tilt
        # Upper-body checks. A humanoid can keep its pelvis perfectly level and upright
        # while folding at the waist, so pelvis-only termination misses the failure
        # entirely. These two catch it directly.
        # Stooped/head-height terminations removed per the audit: no reference terminates
        # on posture, and the orientation penalty now covers it. Height, tilt and NaN stay.
        stooped = np.zeros_like(fallen)
        head_down = np.zeros_like(fallen)
        # Guard against a diverged simulation producing NaNs, which would otherwise
        # silently poison the policy gradient.
        diverged = ~np.isfinite(state.qpos).all(axis=1)
        result = fallen | toppled | stooped | head_down | diverged
        # Stashed so reset_batch, which runs after the engine decides who resets, can tell
        # a fall from a time limit: falls demote the difficulty, time limits do not.
        state.task_state["last_terminated"][:] = result
        return result
