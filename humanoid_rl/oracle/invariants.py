"""Contradictions that can be found by reading the setup, before a run is ever launched.

Every serious defect this project has hit was caught by a person watching a video and asking
a blunt question, hours or days after the run that contained it started. Four of them did not
need the run at all. They were disagreements between settings, visible in the config:

* `air_time_target` demanded a 0.25 s swing while the gait clock implied 0.44 s, so two
  reward terms pulled against each other for an entire run
* the AMP reference clips topped out at 0.85 m/s while the command range asked for 1.5, so
  the discriminator had no example of fast walking and scored anything quick as fake
* 3.7% of sampled commands were lateral and a 45 degree diagonal was geometrically capped at
  0.57 m/s, so "walk diagonally" was unrepresentable rather than merely undertrained
* `log_std` was initialised at exactly its own clamp ceiling, pinning exploration noise at
  std 1.0 for 610 iterations

A check here costs milliseconds and runs before any compute is spent. That is the whole
point: the cheapest bug is the one found before the run.

DESIGN RULE. A check must compare two things that were specified INDEPENDENTLY. Verifying
that the config equals itself is worthless; verifying that the reward's idea of a swing
duration matches the clock's idea of one is not. Where a check needs a number from outside
the project (human walking speeds, cadence) it must say so and cite it, because a threshold
someone invented is how a checker becomes a rubber stamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    #: Two settings contradict each other. Training will do something, but not what was
    #: intended, and no amount of training time fixes it.
    CONTRADICTION = "CONTRADICTION"
    #: Nothing is inconsistent, but a capability that is being asked for cannot be reached
    #: given the setup. Usually a coverage gap.
    UNREACHABLE = "UNREACHABLE"
    #: Suspicious. Worth a human glance, not necessarily wrong.
    SUSPECT = "SUSPECT"
    #: Confirms something important is right. Reported so a passing run is legible, and so a
    #: check silently disappearing is noticeable.
    OK = "OK"


@dataclass
class Finding:
    severity: Severity
    check: str
    detail: str
    #: What to do about it. A finding without a remedy trains the reader to ignore findings.
    remedy: str = ""
    #: The historical defect this check exists to prevent, if there is one.
    caught_before: str = ""

    def line(self) -> str:
        mark = {"CONTRADICTION": "XX", "UNREACHABLE": "!!", "SUSPECT": "??", "OK": "ok"}
        return f"[{mark[self.severity]}] {self.check}: {self.detail}"


Check = "callable(config) -> list[Finding]"
_REGISTRY: list = []


def check(fn):
    """Register an invariant. Each takes the loaded Config and returns findings."""
    _REGISTRY.append(fn)
    return fn


# --------------------------------------------------------------------------- gait schedule


@check
def swing_duration_agrees_with_clock(config) -> list[Finding]:
    """The air-time reward and the gait clock must want the same swing duration.

    These are specified in completely different places, which is exactly why they drifted
    apart: one is a constant in seconds, the other falls out of a frequency and a duty
    factor. Nothing in the code forced them to agree.
    """
    task = config.task
    implied = (1.0 - task.stance_fraction) / max(task.gait_frequency, 1e-9)
    stated = task.air_time_target
    # The reward now derives the target from the schedule, so the constant is only a
    # fallback for the standing case. Flag it if a future edit re-introduces the conflict.
    if abs(implied - stated) / max(implied, 1e-9) > 0.25:
        return [Finding(
            Severity.SUSPECT, "swing duration",
            f"air_time_target is {stated:.2f}s while the clock implies {implied:.2f}s. "
            f"This is only harmless while the reward derives the target from the schedule.",
            remedy="Keep air_time_target as the standing fallback only, or set it to "
                   f"{implied:.2f}.",
            caught_before="Two reward terms pulled against each other for a whole run; "
                          "feet_air_time ran at a net penalty and gait_phase sat at chance.",
        )]
    return [Finding(Severity.OK, "swing duration",
                    f"reward and clock agree at {implied:.2f}s")]


@check
def schedule_describes_a_walk(config) -> list[Finding]:
    """At the nominal parameters, the schedule must produce a human walk.

    Human reference: 1.6-2.0 foot strikes/s, 0.20-0.25 double support, no flight phase.
    These are measured gait values, not thresholds chosen here.
    """
    task = config.task
    strikes = 2.0 * task.gait_frequency
    # Two stance windows of width s, half a cycle apart, overlap by (2s - 1) when s > 0.5.
    double_support = max(0.0, 2.0 * task.stance_fraction - 1.0)
    flight = max(0.0, 1.0 - 2.0 * task.stance_fraction)

    out: list[Finding] = []
    # The RANGE, not just the anchor. This check passed for months on gait_frequency=0.9
    # while the sampled band commanded 2-4 strikes/s, double human cadence, and the policy
    # dutifully delivered 2.91. Stride length is then forced arithmetic (speed / rate) and
    # the result reads as a shuffle no matter what else is tuned.
    lo, hi = task.gait_frequency_range
    if 2.0 * hi > 2.2 or 2.0 * lo < 1.2:
        out.append(Finding(
            Severity.CONTRADICTION, "cadence range",
            f"gait_frequency_range {lo}-{hi} Hz commands {2*lo:.1f}-{2*hi:.1f} foot "
            f"strikes/s; human walking is 1.6-2.0. A policy obeying this cannot produce a "
            f"human stride length at any speed it can reach.",
            remedy="Set gait_frequency_range so 2*f lands in 1.6-2.0, e.g. (0.7, 1.1).",
            caught_before="A 500M-step run delivered 2.91 strikes/s and an 0.11 m stride "
                          "against a human 0.6-0.8 m, while every per-term reward looked fine.",
        ))
    if not 1.6 - 1e-6 <= strikes <= 2.0 + 1e-6:
        out.append(Finding(
            Severity.SUSPECT, "cadence",
            f"the nominal schedule gives {strikes:.2f} foot strikes/s; human walking is "
            f"1.6-2.0",
            remedy=f"gait_frequency of {task.gait_frequency:.2f} Hz implies half that "
                   f"cadence. Set it near 0.9 for a walk.",
        ))
    if not 0.20 - 1e-6 <= double_support <= 0.25 + 1e-6:
        out.append(Finding(
            Severity.SUSPECT, "double support",
            f"the schedule implies {double_support:.2f}; human walking is 0.20-0.25",
            remedy=f"stance_fraction {task.stance_fraction:.2f} sets this as 2s-1. "
                   f"0.60-0.625 gives the human range.",
        ))
    if flight > 0.01:
        out.append(Finding(
            Severity.SUSPECT, "flight phase",
            f"stance_fraction {task.stance_fraction:.2f} is below 0.5, so the schedule asks "
            f"for a {flight:.2f} flight phase. That is running, not walking.",
            remedy="Intended for a run; for a walk keep stance_fraction above 0.5.",
        ))
    return out or [Finding(
        Severity.OK, "gait schedule",
        f"{strikes:.2f} strikes/s, {double_support:.2f} double support, no flight: human walk")]


# ------------------------------------------------------------------- command coverage


def _sample_commands(config, n: int = 200_000):
    """Draw n commands from the task's own sampler and return them as an (n, 3) array.

    Deliberately calls `_draw_command` rather than reimplementing it. A checker that
    reimplements the thing it checks verifies its own copy, and the copy is what drifts.

    The cost of that choice is this fixture: `_draw_command` reads state the probe has to
    supply, and it grows. Both `difficulty` and `heading` were added to the sampler after
    the first version of this fixture was written, and the direction check spent that whole
    period raising KeyError, being caught by the harness, and reporting SUSPECT instead of
    an answer. Anything the sampler reads must be listed here.
    """
    import numpy as np

    from humanoid_rl.tasks.locomotion import LocomotionTask

    probe = LocomotionTask(config.task)

    class _State:
        num_envs = n
        task_state: dict = {}
        #: Facing straight down +x. The sampler sets desired_heading relative to this.
        heading = np.zeros(n)

    state = _State()
    for key in ("command", "gait_freq", "stance_frac", "foot_offset", "clock_authority",
                "body_height", "command_age", "hold_time", "phase", "desired_heading",
                "lead", "lead_swaps", "gait_steps", "command_changed",
                "seg_err", "seg_gait", "seg_steps"):
        state.task_state[key] = np.zeros((n, 3) if key == "command" else n)
    # Full difficulty. If the FINAL curriculum level still asks for a crawl, no amount of
    # promotion helps, so that is the level worth checking.
    state.task_state["difficulty"] = np.ones(n)
    probe._draw_command(state, np.arange(n), np.random.default_rng(0))  # noqa: SLF001
    return state.task_state["command"]


@check
def commands_cover_every_direction(config) -> list[Finding]:
    """Every direction the roadmap needs must be both reachable and practised.

    The failure this prevents is subtle and was live for months: with a BOX over (vx, vy)
    the corner geometry caps a 45 degree diagonal at the distance to the corner, so a
    diagonal command above that speed cannot be expressed AT ALL. That is not a training
    problem and no amount of compute fixes it.
    """
    import numpy as np

    command = _sample_commands(config)
    moving = np.linalg.norm(command[:, :2], axis=1) > 1e-9
    if not moving.any():
        return [Finding(Severity.CONTRADICTION, "command coverage",
                        "every sampled command was zero")]

    angle = np.arctan2(command[moving, 1], command[moving, 0])
    sector = (((angle + np.pi / 8) % (2 * np.pi)) // (np.pi / 4)).astype(int)
    shares = np.bincount(sector, minlength=8) / sector.size
    names = ["forward", "fwd-left", "left", "back-left",
             "backward", "back-right", "right", "fwd-right"]

    worst = int(np.argmin(shares))
    # Eight sectors, so uniform is 12.5%. Below a third of that is a direction the policy
    # will barely practise.
    if shares[worst] < 0.125 / 3:
        return [Finding(
            Severity.UNREACHABLE, "command coverage",
            f"{names[worst]} gets {shares[worst]:.1%} of commands against 12.5% for uniform "
            f"(range {shares.min():.1%} to {shares.max():.1%})",
            remedy="Sample direction and speed in polar form over an elliptical envelope "
                   "rather than a box over (vx, vy).",
            caught_before="52.7% of commands were forward and 3.7% lateral; a 45 degree "
                          "diagonal was geometrically capped at 0.57 m/s.",
        )]
    return [Finding(
        Severity.OK, "command coverage",
        f"all eight directions get {shares.min():.1%} to {shares.max():.1%} of commands")]


@check
def commands_ask_for_walking_speed(config) -> list[Finding]:
    """Most MOVING commands must ask for a speed a human would call walking.

    Direction coverage and speed coverage are different questions and the direction check
    above passes while this one fails. Sampling uniformly by AREA of an elliptical envelope
    spreads headings evenly, which was the point, but it also puts most of the probability
    mass at small radii, and the envelope's lateral semi-axis (0.4 m/s) crushes the ellipse
    everywhere except straight ahead. The two effects compound.

    Human free walking is 1.2-1.4 m/s (Bohannon's normative meta-analysis). The retargeted
    mocap here spans 0.52-1.24 m/s. A policy trained on a command distribution centred well
    below both is not failing to track its command; it is tracking it, and the command is a
    crawl. Every downstream measurement then reads as a gait defect: short stride, low speed,
    and an AMP discriminator that scores the policy fake because it never moves at any speed
    the reference clips contain.
    """
    import numpy as np

    speed = np.linalg.norm(_sample_commands(config)[:, :2], axis=1)
    moving = speed > 1e-9
    if not moving.any():
        return [Finding(Severity.CONTRADICTION, "commanded speed",
                        "every sampled command was zero")]
    speed = speed[moving]
    median = float(np.median(speed))
    # 0.52-1.24 m/s: measured on this project's own retargeted clips, listed in amp.yaml.
    in_clips = float(((speed >= 0.52) & (speed <= 1.24)).mean())

    out: list[Finding] = []
    if median < 0.6:
        out.append(Finding(
            Severity.CONTRADICTION, "commanded speed",
            f"the median moving command is {median:.2f} m/s. Human free walking is 1.2-1.4 "
            f"and the reference clips are 0.52-1.24. The policy is being asked to crawl, so "
            f"a short stride and a low speed are obedience, not a gait defect.",
            remedy="Raise the envelope so the median moving command lands near 1.0 m/s. The "
                   "binding constraint is lin_vel_y_range, which sets the ellipse's lateral "
                   "semi-axis and shrinks the reach at every off-axis heading.",
            caught_before="A 500M-step run tracked its command faithfully at 0.20 m/s and "
                          "every gait metric was then read as a failure to walk.",
        ))
    if in_clips < 0.5 and config.run.task == "amp":
        out.append(Finding(
            Severity.UNREACHABLE, "commands vs clips",
            f"only {in_clips:.0%} of moving commands fall inside the reference clip range "
            f"0.52-1.24 m/s, so most of the time the discriminator has no real example at "
            f"the speed the policy was told to move.",
            remedy="Match the command distribution to the clip library before training AMP.",
            caught_before="Style reward fell 0.54 -> 0.31 while task return rose.",
        ))
    return out or [Finding(
        Severity.OK, "commanded speed",
        f"median moving command {median:.2f} m/s, {in_clips:.0%} inside the clip range")]


@check
def reference_motion_covers_the_commands(config) -> list[Finding]:
    """An adversarial motion prior can only reward speeds it has seen a human perform.

    If the command range extends past the fastest reference clip, the discriminator has no
    real example to compare against up there and correctly labels the policy's fast walking
    as fake. The style reward then collapses precisely when the policy does what it was
    asked.
    """
    import numpy as np

    if config.run.task != "amp":
        return []
    from humanoid_rl.envs.model_prep import prepare
    from humanoid_rl.motion.library import MotionLibrary

    model = prepare(Path(config.env.model_path)).model
    lib = MotionLibrary.build(
        Path(config.clip_dir), model, include=list(config.clip_include) or None,
        mirror=config.mirror_clips,
    )
    # A high percentile of INSTANTANEOUS speed, not the clip average. The discriminator
    # scores individual transitions, so what matters is whether fast transitions exist in
    # the reference at all, not whether any whole clip averages that fast. Using the mean
    # understates the reference and makes this check fire on a gap that is not there.
    planar = np.linalg.norm(lib.qvel[:, :2], axis=1)
    fastest = float(np.percentile(planar, 99))
    commanded = max(abs(config.task.lin_vel_x_range[0]), config.task.lin_vel_x_range[1])

    if commanded > fastest * 1.15:
        return [Finding(
            Severity.UNREACHABLE, "motion prior coverage",
            f"commands reach {commanded:.2f} m/s but the reference's 99th percentile "
            f"instantaneous speed is only {fastest:.2f} m/s",
            remedy="Include the faster clips in clip_include, or lower the command range. "
                   "The discriminator cannot reward a speed it has never seen a human do.",
            caught_before="FR/BR/SR were excluded while commands asked 1.5 m/s; style "
                          "reward decayed 0.336 -> 0.151 and discriminator accuracy hit 0.998.",
        )]
    return [Finding(Severity.OK, "motion prior coverage",
                    f"reference p99 {fastest:.2f} m/s covers commands of {commanded:.2f}")]


# --------------------------------------------------------------------------- exploration


@check
def exploration_noise_can_move(config) -> list[Finding]:
    """The action noise must be free to shrink AND to grow.

    `log_std` starts at log(init_noise_std) and is clamped to log_std_max. If those are
    equal, the parameter starts pinned against its own ceiling: the entropy bonus pushes it
    up, the clamp holds it, and the std never changes. It reads as a healthy constant in the
    logs and is anything but.
    """
    import numpy as np

    ppo = config.ppo
    init_log_std = float(np.log(config.network.init_noise_std))
    if init_log_std >= ppo.log_std_max - 1e-9:
        return [Finding(
            Severity.CONTRADICTION, "exploration noise",
            f"init_noise_std {config.network.init_noise_std} (log {init_log_std:+.3f}) is "
            f"at or above the clamp log_std_max {ppo.log_std_max:+.2f} (std "
            f"{np.exp(ppo.log_std_max):.2f}), so log_std starts outside its own range where "
            f"torch.clamp passes no gradient",
            remedy="Raise log_std_max above the initial value (e.g. 0.5 for an init of "
                   "1.0), or lower init_noise_std, or set entropy_coef to 0.",
            caught_before="action_std read exactly 1.0000 and entropy exactly 39.73 for "
                          "610 straight iterations while log_std sat above its ceiling.",
        )]
    return [Finding(Severity.OK, "exploration noise",
                    f"init std {config.network.init_noise_std} sits below the ceiling of "
                    f"{np.exp(ppo.log_std_max):.2f}, so log_std has live gradient")]


# ------------------------------------------------------------------------ reward shape


@check
def no_reward_term_can_be_won_by_standing_still(config) -> list[Finding]:
    """Terms that a motionless humanoid maximises are how degenerate gaits survive.

    Reported rather than computed from first principles: this one needs judgement, and its
    value is in naming the pattern so a new term is checked against it.
    """
    task = config.task
    suspects: list[str] = []
    if task.w_alive > 0 and task.w_gait_phase <= 0:
        suspects.append("w_alive pays for existing while nothing rewards stepping")
    if task.w_feet_air_time <= 0 and task.w_gait_phase <= 0:
        suspects.append("neither air time nor a gait schedule rewards lifting a foot")
    if suspects:
        return [Finding(
            Severity.SUSPECT, "degenerate optima", "; ".join(suspects),
            remedy="Keep at least one term that a motionless humanoid cannot satisfy.",
            caught_before="A 61 cm wide two-footed brace scored 0.91 on gait symmetry while "
                          "travelling 6 cm per foot strike.",
        )]
    return [Finding(Severity.OK, "degenerate optima",
                    "at least one term requires actually stepping")]


@check
def value_horizon_is_honest(config) -> list[Finding]:
    """Warn when most of the value target is the critic's own bootstrap.

    With horizon H and discount g, a fraction g**H of the discounted mass of the GAE target
    is the bootstrapped V(s) rather than observed reward. When that fraction is high, any
    "explained variance" computed on the buffer is largely the critic predicting itself, and
    will read high whatever the critic is actually doing. This is exactly how a metric added
    to detect a bad critic ended up unable to.
    """
    ppo = config.ppo
    bootstrap_share = ppo.gamma ** ppo.horizon
    if bootstrap_share > 0.5:
        return [Finding(
            Severity.SUSPECT, "value target composition",
            f"gamma^horizon = {ppo.gamma}^{ppo.horizon} = {bootstrap_share:.2f}, so "
            f"{bootstrap_share:.0%} of the value target's discounted mass is the critic's "
            f"own bootstrap rather than observed reward",
            remedy="Fine for training, but do not judge the critic on a buffer-derived "
                   "explained variance. Measure it against Monte-Carlo returns over "
                   "completed episodes instead.",
            caught_before="An explained_variance metric added to this repo reduced to "
                          "1 - Var(A)/Var(V+A) and read +0.95 where the honest value was +0.72.",
        )]
    return [Finding(Severity.OK, "value target composition",
                    f"{bootstrap_share:.0%} of the target is bootstrap")]


@check
def no_config_section_is_silently_ignored(config) -> list[Finding]:
    """A task section that the selected task does not read is a silent no-op.

    `Config` carries `task`, `amp_task` and `tracking` side by side, and `run.task` selects
    exactly one. The other two are accepted by the loader, validated, saved into the run
    directory, and never read. Every value in them is a comment that looks like a setting.

    This wasted a seven-arm ablation: six arms patched `task` on an AMP run and returned
    byte-identical results, because `amp.yaml` has no `task` section and AMP reads
    `amp_task`.
    """
    from dataclasses import fields

    selected = {"locomotion": "task", "amp": "amp_task", "tracking": "tracking"}
    active = selected.get(config.run.task)
    if active is None:
        return []
    unused = [name for name in selected.values() if name != active]

    # Only report a section that has been deliberately set away from its defaults, since
    # every section always exists as a dataclass with values.
    from humanoid_rl.config import Config

    default = Config()
    noisy = []
    for name in unused:
        current, base = getattr(config, name, None), getattr(default, name, None)
        if current is None or base is None:
            continue
        changed = [f.name for f in fields(current)
                   if getattr(current, f.name) != getattr(base, f.name)]
        if changed:
            noisy.append(f"{name} ({len(changed)} settings differ from defaults)")
    if noisy:
        return [Finding(
            Severity.SUSPECT, "unused config section",
            f"run.task is '{config.run.task}', which reads '{active}'. These are set but "
            f"never read: {', '.join(noisy)}",
            remedy=f"Move the settings into '{active}', or delete them so they stop looking "
                   f"like they do something.",
            caught_before="Six of seven ablation arms patched an unread section and "
                          "returned byte-identical results.",
        )]
    return [Finding(Severity.OK, "config sections",
                    f"run.task '{config.run.task}' reads '{active}'; nothing else is set")]


def check_checkpoint(path: Path, config) -> list[Finding]:
    """Invariants that live in the WEIGHTS, not the config.

    Added because the worst exploration bug this project had was invisible to a config
    check: `log_std` arrived above its clamp through `--init-from`, inherited from an
    earlier run, so `init_noise_std` in the YAML was silently ignored for 610 iterations.
    Whenever a parameter can be both configured and inherited, the inherited value wins and
    the config becomes a comment.
    """
    import torch

    if not path.exists():
        return []
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state.get("policy", {})
    if "log_std" not in weights:
        return []

    # A normaliser whose count is small relative to the batch adapts fast enough to pull
    # the input scaling out from under warm-started weights. Measured: a cap of 1e6 with
    # 98k-sample batches moved the statistics ~10% per iteration and destroyed a working
    # warm start between evaluations 50 and 100.
    count = weights.get("obs_rms.count")
    if count is not None and float(count) < 1.0e7 and float(count) > 1e5:
        out_norm = [Finding(
            Severity.SUSPECT, "normaliser inertia",
            f"obs_rms.count is {float(count):,.0f}; with ~1e5-sample batches the input "
            f"statistics move {98304/float(count):.0%} per iteration, which can destroy a "
            f"warm-started policy as the scaling drifts out from under its weights",
            remedy="Do not cap the normaliser count. If fresh statistics are wanted, reset "
                   "them deliberately and retrain the policy with them from the start.",
            caught_before="A warm start scored 12% falls at evaluation 50 and 100% by 100, "
                          "destroyed by 50x-accelerated normaliser drift.",
        )]
    else:
        out_norm = []

    log_std = weights["log_std"]
    ceiling = config.ppo.log_std_max
    above = int((log_std > ceiling).sum())
    if above:
        return out_norm + [Finding(
            Severity.CONTRADICTION, "frozen exploration",
            f"{above} of {log_std.numel()} log_std components sit ABOVE the clamp "
            f"({float(log_std.max()):+.5f} > {ceiling:+.2f}). torch.clamp passes NO gradient "
            f"outside its range, so these receive exactly zero gradient and are frozen "
            f"permanently. Exploration noise cannot change.",
            remedy="Clamp log_std in place after optimizer.step() rather than in the "
                   "forward pass, and clamp on load in init_policy_from. Then lower "
                   "log_std_max to actually reduce the noise.",
            caught_before="std read exactly 1.0000 for 610 iterations; the noise cost 49% "
                          "of the per-step reward and PPO was optimising the degraded policy.",
        )]
    return out_norm + [Finding(Severity.OK, "exploration reachable",
                    f"log_std max {float(log_std.max()):+.4f} is below the clamp {ceiling:+.2f}")]


@check
def value_support_covers_reachable_return(config) -> list[Finding]:
    """A distributional critic's grid must reach the returns the reward function can pay.

    This is the quietest catastrophic failure available in the whole method. C51 puts
    probability mass on a FIXED grid from v_min to v_max. If the reachable discounted return
    exceeds v_max, every good state piles its mass on the top atom, the critic returns the
    same number for "walking beautifully" and "barely upright", and the actor gets no
    gradient distinguishing them. The critic loss goes down the whole time, because
    predicting a saturated target is easy. Nothing anywhere looks wrong.

    The two sides are specified independently, which is what makes the check worth having:
    v_max is a number in the FastTD3 config, and the reachable return falls out of the task's
    reward weights and gamma. The FastTD3 authors' own IsaacLab preset is +/-10, correct for
    IsaacLab's tiny rewards and off by a factor of 30 for ours.
    """
    if getattr(config.run, "algo", "ppo") != "fasttd3":
        return []
    td3 = config.fasttd3
    task = config.task
    # Upper bound on per-step reward: every positive term at full value, penalties ignored.
    # Deliberately optimistic, because the support has to cover the best case, not the mean.
    positive = sum(max(getattr(task, name, 0.0), 0.0) for name in dir(task)
                   if name.startswith("w_"))
    ceiling = positive / max(1.0 - td3.gamma, 1e-9)

    out: list[Finding] = []
    if td3.v_max < ceiling:
        out.append(Finding(
            Severity.CONTRADICTION, "value support",
            f"v_max is {td3.v_max:.0f} but the reward weights allow a discounted return of "
            f"{ceiling:.0f} at gamma={td3.gamma}. Returns above v_max saturate on the top "
            f"atom, so the critic cannot rank good states against each other.",
            remedy=f"Set v_max to at least {ceiling * 1.25:.0f}, or use "
                   f"humanoid_rl.algos.fasttd3.suggested_support().",
            caught_before="Not yet. This check exists because the failure is invisible: the "
                          "critic loss falls normally while the value function is constant.",
        ))
    # A "value resolution" sub-check used to live here, warning when an atom spanned more
    # than half a step's reward because "one step of improvement may not move the target".
    # REMOVED: it was wrong, and a wrong check is worse than no check because it teaches the
    # reader to skim findings. The categorical projection is exactly mean-preserving, and the
    # actor consumes only E[Q] = sum(p * z), which is continuous in the probabilities at any
    # atom spacing. Coarse atoms limit how finely the SHAPE of the return distribution can be
    # represented; they do not quantise the quantity the policy gradient actually uses. The
    # saturation check above is the real failure mode and it stays.
    return out or [Finding(
        Severity.OK, "value support",
        f"[{td3.v_min:.0f}, {td3.v_max:.0f}] over {td3.num_atoms} atoms covers a reachable "
        f"{ceiling:.0f}")]


@check
def curriculum_floor_is_worth_training_on(config) -> list[Finding]:
    """The curriculum's FLOOR must still command a speed worth practising.

    A floor is a promise about the easiest command you are willing to train on, and a
    curriculum will find it. With 20 s episodes a fall is likely in almost any episode, so
    demotion outruns promotion and every environment parks on the floor. If that floor is a
    crawl, the run trains a crawler no matter what the top of the range says.

    Checked against the same measured references as the command envelope: the retargeted
    clips run 0.52-1.24 m/s, so a floor below 0.52 is outside anything we have a human
    example of.
    """
    import numpy as np

    task = config.task
    if task.difficulty_min >= 1.0:
        return [Finding(Severity.OK, "curriculum floor", "curriculum disabled")]

    out: list[Finding] = []
    if task.difficulty_init < task.difficulty_min:
        out.append(Finding(
            Severity.CONTRADICTION, "curriculum floor",
            f"difficulty_init {task.difficulty_init} is below difficulty_min "
            f"{task.difficulty_min}, so the starting level violates the floor.",
            remedy="Set difficulty_init at or above difficulty_min.",
        ))

    scaled = np.linalg.norm(_sample_commands(config)[:, :2], axis=1) * task.difficulty_min
    floor_median = float(np.median(scaled[scaled > 1e-9]))
    if floor_median < 0.52:
        out.append(Finding(
            Severity.CONTRADICTION, "curriculum floor",
            f"at difficulty_min {task.difficulty_min} the median moving command is "
            f"{floor_median:.2f} m/s, below the slowest reference clip (0.52). A curriculum "
            f"that demotes to this floor trains a crawl.",
            remedy="Raise difficulty_min until the floor median clears 0.52 m/s.",
            caught_before="Logbook E20: difficulty slid from 0.70 to the 0.50 floor within "
                          "150 iterations and stayed there, commanding 0.36-0.40 m/s for the "
                          "rest of the run.",
        ))
    return out or [Finding(
        Severity.OK, "curriculum floor",
        f"floor {task.difficulty_min} commands a median {floor_median:.2f} m/s")]


@check
def getup_hold_and_thresholds_are_reachable(config) -> list[Finding]:
    """The get-up hold, episode length and force thresholds must be mutually satisfiable.

    Four independent specifications have to agree here and nothing in the code forces them to:
    the hold duration (seconds), the control rate (physics timestep x decimation), the episode
    limit (steps), and the domain randomisation mass range. Each has been an independent
    source of failure in this project.
    """
    import numpy as np

    if getattr(config.run, "task", "") != "getup":
        return []
    cfg = config.getup
    out: list[Finding] = []

    # 1. The hold must be an exact number of control steps. E18: a step count copied from a
    # 50 Hz reference silently became 8 s on our 125 Hz loop while every comment said 20 s.
    from humanoid_rl.envs.model_prep import prepare

    prepared = prepare(Path(__file__).resolve().parents[2] / config.env.model_path)
    dt = float(prepared.model.opt.timestep) * config.env.decimation
    steps = cfg.hold_seconds / dt
    if abs(steps - round(steps)) > 1e-6:
        out.append(Finding(
            Severity.SUSPECT, "hold duration",
            f"hold_seconds {cfg.hold_seconds} is {steps:.2f} control steps at {1/dt:.0f} Hz, "
            f"not a whole number.",
            remedy=f"Use a multiple of {dt:.4f} s.",
        ))

    # 2. The episode must be long enough to fail, recover and still hold. An episode barely
    # longer than the hold makes success a matter of where the reset happened to land.
    if config.env.max_episode_steps < 3 * round(steps):
        out.append(Finding(
            Severity.CONTRADICTION, "episode vs hold",
            f"episode is {config.env.max_episode_steps} steps and the hold needs "
            f"{round(steps)}. Less than 3x leaves no room to get up, be shoved, and recover.",
            remedy=f"Set max_episode_steps to at least {3 * round(steps)}.",
        ))

    # 3. The foot-force thresholds are fractions of NOMINAL weight, but domain randomisation
    # scales real mass. At the light end a genuine stand must still clear them, or U becomes
    # unsatisfiable on part of the model pool and the policy is being asked for the impossible.
    light = min(config.domain_rand.mass_scale_range) if config.domain_rand.enabled else 1.0
    if cfg.u_force_total_bw >= light:
        out.append(Finding(
            Severity.CONTRADICTION, "foot force vs mass randomisation",
            f"u_force_total_bw {cfg.u_force_total_bw} is not below the lightest sampled mass "
            f"scale {light}. A real stand on a light model cannot satisfy it.",
            remedy=f"Keep u_force_total_bw below {light:.2f}, or narrow mass_scale_range.",
        ))
    if 2.0 * cfg.u_force_min_bw > cfg.u_force_total_bw + 1e-9:
        out.append(Finding(
            Severity.CONTRADICTION, "foot force split",
            f"2 x u_force_min_bw ({2*cfg.u_force_min_bw}) exceeds u_force_total_bw "
            f"({cfg.u_force_total_bw}), so the per-foot floor implies more than the total.",
            remedy="Set u_force_min_bw below half of u_force_total_bw.",
        ))

    # 4. The pose bank must exist and match this model. A stale artefact is a live hazard
    # here: this repo already has a directory named ABANDONED-staleClips-tracking.
    bank = Path(__file__).resolve().parents[2] / cfg.bank_path
    if not bank.exists():
        out.append(Finding(
            Severity.CONTRADICTION, "pose bank",
            f"no fallen-pose bank at {cfg.bank_path}",
            remedy="python scripts/generate_fallen_poses.py",
        ))
    else:
        data = np.load(bank, allow_pickle=False)
        if int(data["nq"]) != prepared.model.nq:
            out.append(Finding(
                Severity.CONTRADICTION, "pose bank",
                f"bank nq {int(data['nq'])} against model nq {prepared.model.nq}",
                remedy="Rebuild the bank for this model."))
        else:
            # Left/right balance. The humanoid is bilaterally symmetric, so a bank that
            # lands mostly on one shoulder trains a policy that can only rise one way, and
            # the class counts alone will not show it if the label uses abs().
            lab = data["label"]
            left = int((lab == "side_left").sum())
            right = int((lab == "side_right").sum())
            if left + right >= 20:
                share = left / (left + right)
                if not 0.35 <= share <= 0.65:
                    out.append(Finding(
                        Severity.SUSPECT, "pose bank balance",
                        f"side-lying poses are {share:.0%} left-down against {1-share:.0%} "
                        f"right-down ({left} vs {right}). A bilaterally symmetric body should "
                        f"see both roughly equally.",
                        remedy="Rebuild the bank, or check the topple generator's impulse "
                               "direction sampling."))
            floor = float((data["generator"] != "standing").mean())
            if floor < 0.5:
                out.append(Finding(
                    Severity.SUSPECT, "pose bank",
                    f"only {floor:.0%} of the bank is on the floor; the task is mostly "
                    f"resetting into a stand it does not have to earn."))

    # 5. The shaping must telescope. At gamma < 1 the -(1-gamma)*Phi drain scales with the
    # weight, and at a weight large enough to matter it swamps every real reward term.
    if cfg.shaping_weight > 0:
        drain = cfg.shaping_weight * (1.0 - cfg.shaping_gamma)
        if drain > 0.10:
            out.append(Finding(
                Severity.CONTRADICTION, "shaping drain",
                f"weight {cfg.shaping_weight} at gamma {cfg.shaping_gamma} costs "
                f"{drain:.2f}/step simply for being upright, against a standing reward near "
                f"4.3. The shaping would dominate the objective it is meant to assist.",
                remedy="Cut the weight. Do NOT reach for shaping_gamma = 1.0: this remedy "
                       "used to say that, and it is exactly the reward pump that "
                       "shaping_gamma_matches_rl_gamma now forbids (E31).",
                caught_before="Measured at weight 5 and gamma 0.99: rising at 0.3 m/s scored "
                              "NEGATIVE, because the drain exceeded the progress term."))

    return out or [Finding(
        Severity.OK, "getup setup",
        f"hold {round(steps)} steps of a {config.env.max_episode_steps}-step episode, "
        f"force floor {cfg.u_force_total_bw} under a {light} light-mass draw")]


@check
def exploration_has_a_ceiling(config) -> list[Finding]:
    """`log_std_max` must actually bound exploration, not merely exist.

    This project has now lost runs to log_std in BOTH directions, which is why the check
    covers both:

    * E05: log_std was INITIALISED above its own ceiling. `torch.clamp` passes no gradient
      strictly outside its range, so the parameter froze at std 1.0 for 610 iterations.
    * E30: log_std_max was 5.0, which is std 148 and therefore no ceiling at all. With a
      positive entropy bonus and nothing pulling back, exploration ran 0.79 -> 3.43 on an
      action range of [-1, 1], the policy drowned in its own noise, and eval return fell from
      833 at iteration 300 to 19 at iteration 900.

    A ceiling above about std 2 is not a ceiling: past that, most sampled actions clip
    against the action range and the policy is closer to noise than to a policy.
    """
    import math

    ppo = config.ppo
    ceiling = math.exp(ppo.log_std_max)
    init = config.network.init_noise_std
    out: list[Finding] = []
    if ceiling > 2.0:
        out.append(Finding(
            Severity.CONTRADICTION, "exploration ceiling",
            f"log_std_max {ppo.log_std_max} allows std {ceiling:.1f} on an action range of "
            f"[-1, 1]. Above std 2 most samples clip and the policy is mostly noise.",
            remedy="Set log_std_max near 0.0 (std 1.0). Walking trained fine at 0.4-1.4.",
            caught_before="Exploration ran 0.79 -> 3.43 and eval return collapsed 833 -> 19 "
                          "between iterations 300 and 900.",
        ))
    if init > ceiling:
        out.append(Finding(
            Severity.CONTRADICTION, "exploration ceiling",
            f"init_noise_std {init} starts ABOVE the ceiling {ceiling:.2f}. clamp passes no "
            f"gradient strictly outside its range, so log_std would be frozen from step one.",
            remedy=f"Set init_noise_std at or below {ceiling:.2f}.",
            caught_before="E05: frozen at std 1.0 for 610 iterations.",
        ))
    return out or [Finding(
        Severity.OK, "exploration ceiling",
        f"std capped at {ceiling:.2f}, starting from {init}")]


@check
def external_impulses_cannot_void_the_hold(config) -> list[Finding]:
    """No impulse the setup itself injects may violate the success predicate by arithmetic.

    The get-up hold requires |v| <= u_lin_speed CONSECUTIVELY for hold_seconds. Two mechanisms
    write velocity straight into qvel, bypassing the actuators, so no policy can resist them:
    the task's own hold shove, and the domain-randomisation push. If either exceeds the speed
    cap without a corresponding forgiveness, the predicate is unsatisfiable BY CONSTRUCTION
    and every run is optimising toward a goal that cannot be reached.

    E33: hold_push_vel 0.6 against u_lin_speed 0.4, no grace, fired inside every attempt at
    step 40-140 of the 250 needed, and the miss re-armed it. Twelve runs read standing_frac
    exactly 0.0% and the conclusion drawn each time was about the reward. Measured: a perfect
    stand reached 186/250 with the shove on and 330/250 with it off.

    The comparison is between independently specified things: an impulse magnitude (task or
    engine setting), a predicate threshold (task setting), and the forgiveness bookkeeping.
    """
    if getattr(config.run, "task", "") != "getup":
        return []
    cfg = config.getup
    out: list[Finding] = []

    grace = int(getattr(cfg, "hold_push_grace", 0))
    if cfg.hold_push_vel > cfg.u_lin_speed and grace <= 0:
        out.append(Finding(
            Severity.CONTRADICTION, "hold vs task shove",
            f"hold_push_vel {cfg.hold_push_vel} m/s is written into qvel against a "
            f"u_lin_speed cap of {cfg.u_lin_speed}, with no grace window. The shove violates "
            f"the hold by arithmetic on every attempt; no policy can complete it, ever.",
            remedy="Set hold_push_grace to ~0.4 s of steps (forgiving ONLY the velocity "
                   "conjunct), or push below the cap.",
            caught_before="E33: twelve runs with standing_frac exactly 0.0%.",
        ))

    # The engine push is invisible to the task, so NO grace can cover it. Its ceiling has to
    # clear the cap with room for the quiet-stand velocity (~0.05 m/s measured).
    if config.domain_rand.enabled and config.domain_rand.push_vel_xy > 0.85 * cfg.u_lin_speed:
        out.append(Finding(
            Severity.CONTRADICTION, "hold vs domain-rand push",
            f"domain_rand.push_vel_xy {config.domain_rand.push_vel_xy} m/s against a "
            f"u_lin_speed cap of {cfg.u_lin_speed}. The engine push is invisible to the task "
            f"(no flag reaches it), so the grace window cannot cover it and part of all hold "
            f"attempts die to a disturbance no policy could survive.",
            remedy=f"Keep push_vel_xy at or below {0.85 * cfg.u_lin_speed:.2f} "
                   f"(0.85x the cap, leaving margin for quiet-stand velocity).",
            caught_before="E33: at 0.7 roughly a third of hold windows were voided.",
        ))

    # The grace must forgive a settling transient, not the hold itself.
    if grace > 0:
        from humanoid_rl.envs.model_prep import prepare
        prepared = prepare(Path(__file__).resolve().parents[2] / config.env.model_path)
        dt = float(prepared.model.opt.timestep) * config.env.decimation
        hold_steps = cfg.hold_seconds / dt
        if grace >= 0.5 * hold_steps:
            out.append(Finding(
                Severity.CONTRADICTION, "grace vs hold",
                f"hold_push_grace {grace} steps is {grace/hold_steps:.0%} of the "
                f"{hold_steps:.0f}-step hold. Forgiving that much of the hold's own clock "
                f"stops it being a hold.",
                remedy="Keep the grace well under half the hold, ~0.4 s.",
            ))
    return out or [Finding(
        Severity.OK, "external impulses",
        f"shove {cfg.hold_push_vel} graced {grace} steps (velocity conjunct only); "
        f"engine push {config.domain_rand.push_vel_xy} under the {cfg.u_lin_speed} cap")]


@check
def discount_horizon_covers_the_task(config) -> list[Finding]:
    """The discount horizon must be longer than the longest thing the task asks for.

    `gamma` is dimensionless per STEP, so its meaning in seconds depends entirely on the
    control rate. Copying it between repos at different rates silently changes the horizon,
    and nothing anywhere in the code will complain.

    E31: `gamma = 0.99` was taken from legged_gym, which runs at 50 Hz and therefore gets a
    2.0 s horizon from it. This repo runs at 125 Hz, where the same number is 0.80 s. The
    get-up task asks the humanoid to stand and HOLD for 2.0 s: at that discount a successful
    hold is worth 0.081 of an immediate reward, and a 4 s get-up followed by the hold is worth
    0.00053. The task's own success criterion sat outside the agent's horizon for every
    get-up run in the project, so no reward change could ever have reached it.

    Same root as E19, where `max_episode_steps = 1000` was copied from 50 Hz and produced 8 s
    episodes while every comment in the repo said 20 s.

    The comparison is between two INDEPENDENTLY specified things: gamma (an algorithm setting)
    and the duration the task requires (a task setting), coupled only through the physics
    timestep and decimation.
    """
    from humanoid_rl.envs.model_prep import prepare

    ppo = config.ppo
    if ppo.gamma >= 1.0:
        return [Finding(
            Severity.CONTRADICTION, "discount horizon",
            f"gamma {ppo.gamma} is not below 1, so the discounted return need not converge.",
            remedy="Use gamma < 1.")]

    prepared = prepare(Path(__file__).resolve().parents[2] / config.env.model_path)
    dt = float(prepared.model.opt.timestep) * config.env.decimation
    horizon_s = dt / (1.0 - ppo.gamma)

    # What the task actually asks for, in seconds. Each entry is (name, seconds).
    needs: list[tuple[str, float]] = []
    if getattr(config.run, "task", "") == "getup":
        hold = float(config.getup.hold_seconds)
        # A get-up is the hold PLUS the rise that has to precede it. Measured on this project's
        # own rollouts, a rise takes 2-4 s from supine, so the episode's payoff sits at least
        # hold + 2 s away from the reset state.
        needs.append(("the hold alone", hold))
        needs.append(("a rise plus the hold", hold + 2.0))
    else:
        # Walking is cyclic: the longest thing it asks for is a full gait cycle, after which
        # the state repeats and a short horizon is genuinely enough.
        freq = getattr(config.task, "gait_frequency", None)
        if freq:
            needs.append(("one gait cycle", 1.0 / float(freq)))

    out: list[Finding] = []
    for name, seconds in needs:
        if horizon_s < seconds:
            weight = ppo.gamma ** (seconds / dt)
            out.append(Finding(
                Severity.CONTRADICTION, "discount horizon",
                f"gamma {ppo.gamma} at {1/dt:.0f} Hz is a {horizon_s:.2f} s horizon, but the "
                f"task needs {name} at {seconds:.1f} s. A reward that far away is discounted "
                f"to {weight:.4f}, so the agent is being asked for something it cannot see.",
                remedy=f"Set gamma to at least "
                       f"{1.0 - dt / (2.0 * seconds):.4f} for a horizon of 2x {seconds:.1f} s, "
                       f"or shorten what the task requires.",
                caught_before="E31: a 2 s hold discounted to 0.081. The policy learned to "
                              "cycle up and down instead, because the pump paid sooner.",
            ))
    return out or [Finding(
        Severity.OK, "discount horizon",
        f"gamma {ppo.gamma} at {1/dt:.0f} Hz is {horizon_s:.2f} s, covering "
        + (", ".join(f"{n} ({s:.1f} s)" for n, s in needs) if needs else "no stated requirement"))]


@check
def shaping_gamma_matches_rl_gamma(config) -> list[Finding]:
    """Potential-based shaping is only policy-invariant when its gamma IS the RL gamma.

    Ng, Harada & Russell (1999) prove that adding `F(s, s') = g*Phi(s') - Phi(s)` leaves the
    optimal policy unchanged. The proof is that the DISCOUNTED sum telescopes:

        sum_t g^t (g*Phi(s_{t+1}) - Phi(s_t))  =  -Phi(s_0) + lim g^T Phi(s_T)

    which depends only on the start state. That telescoping requires the g in the shaping to
    be the same g the agent discounts with. With a different one, say g_shape = 1:

        sum_t g^t (Phi(s_{t+1}) - Phi(s_t))

    does not telescope. Each rise is discounted less than the fall that undoes it, so a closed
    up-and-down loop nets a PROFIT and the shaping becomes a reward pump.

    E31, measured on the final policy over 9.6 s and 64 envs: the shaping paid out 613.8 and
    clawed back -591.5. Undiscounted that nets 22.3, which is why the code comment claiming
    "oscillating the pelvis up and down pays exactly zero" looked right. Discounted at the
    gamma PPO actually maximises it nets 44.7, larger than its own undiscounted value and
    larger than every other reward term combined: 58% of the whole signal, all of it earned by
    cycling. The humanoid learned to jump to 1.28 m with zero ground contact and land on his
    head, roughly four times per 10 seconds.

    The reason the mismatch was introduced was real: at gamma 0.99 the standing drain
    -(1-g)*Phi costs weight*0.01*Phi per step, which at weight 150 is about 1.0/step and
    swamps every honest term. That argument is a reason to raise gamma or lower the weight,
    not to break the proof. This check reports the drain so the trade is visible.
    """
    if getattr(config.run, "task", "") != "getup":
        return []
    g_rl = float(config.ppo.gamma)
    g_shape = float(config.getup.shaping_gamma)
    weight = float(config.getup.shaping_weight)
    if abs(g_shape - g_rl) > 1e-9:
        return [Finding(
            Severity.CONTRADICTION, "shaping gamma",
            f"shaping_gamma {g_shape} != ppo.gamma {g_rl}, so the potential shaping does not "
            f"telescope in the discounted sum and an up-and-down cycle pays a profit.",
            remedy=f"Set shaping_gamma to {g_rl}. The standing drain that argument was made "
                   f"against is weight*(1-gamma)*Phi = {weight*(1-g_rl)*0.7:.3f}/step at "
                   f"Phi=0.7 and weight {weight}; if that is too large, lower the weight "
                   f"rather than the gamma.",
            caught_before="E31: 58% of the reward signal was a pump earned by jumping to "
                          "1.28 m with no ground contact and landing on his head.",
        )]
    return [Finding(
        Severity.OK, "shaping gamma",
        f"matches ppo.gamma at {g_rl}; standing drain "
        f"{weight*(1-g_rl)*0.7:.3f}/step at Phi=0.7")]


@check
def terrain_field_is_bigger_than_an_episode(config) -> list[Finding]:
    """An episode must not be able to walk off the edge of the world.

    Past the heightfield's extent there is no geom at all, so the humanoid falls into the
    void. That is recorded as an ordinary termination, so it arrives in the metrics as
    `fall_rate` and reads as a policy failure -- a geometry error wearing the costume of the
    headline number. The same class as E16, where an eval that counted the wrong episodes
    invalidated every fall rate in the project.

    Budget: the largest forward command, held for a whole episode, times the measured
    achieved/commanded speed ratio, starting from the worst corner of the spawn square.
    """
    t = getattr(config, "terrain", None)
    if t is None or not t.enabled:
        return [Finding(Severity.OK, "terrain extent", "terrain disabled")]

    dt = config.env.decimation * 0.002
    episode_s = config.env.max_episode_steps * dt
    top_speed = max(abs(v) for v in config.task.lin_vel_x_range)
    # 0.79 is measured (0.586 achieved against 0.738 commanded on the flat run). Using the
    # ratio rather than the raw command is deliberate: quoting the command would size the
    # field for a policy that does not exist, and quoting the achieved speed alone would
    # size it for the policy that exists TODAY.
    reach = t.spawn_half_extent + top_speed * 0.79 * episode_s
    if reach > t.half_extent:
        return [Finding(
            Severity.CONTRADICTION, "terrain extent",
            f"an episode can reach {reach:.1f} m from the origin "
            f"(spawn {t.spawn_half_extent} m + {top_speed} m/s x 0.79 x {episode_s:.1f} s) "
            f"but the field only extends to {t.half_extent} m. Off the field there is no "
            f"geom, so the humanoid falls into the void and it is counted as a fall.",
            remedy=f"Raise terrain.half_extent above {reach:.1f}, or lower "
                   f"terrain.spawn_half_extent. Extent is nearly free: collision cost is set "
                   f"by cell size, not by grid size.",
            caught_before="Measured: hfield cost is 2.01x the plane at a 0.10 m cell whether "
                          "the field is 4 m or 40 m across.",
        )]
    return [Finding(Severity.OK, "terrain extent",
                    f"worst reach {reach:.1f} m inside a {t.half_extent} m half-extent")]


@check
def terrain_is_rough_enough_to_matter_and_not_so_rough_it_takes_over(config) -> list[Finding]:
    """The relief must be hard enough to teach and inside what has been measured.

    REVISED after the model this check originally used was refuted by measurement.

    The first version derived a ceiling from the gait clock: stride-to-stride ground change
    becomes a touchdown TIMING error, and past the stance transition width the terrain rather
    than the policy would be what loses `gait_phase` (27.8% of the reward). It put the ceiling
    near 7 cm. Measured on a trained walker, deterministic, 1.0 m/s, 64 envs x 400 steps:

        p2p     gait_phase reward/step
        0.00 cm      0.8054
        5.25 cm      0.7967
        9.00 cm      0.7802
       14.00 cm      0.7242
       20.00 cm      0.6957

    At 20 cm -- nearly 3x that ceiling -- gait_phase has fallen 13.6%, and `torso_upright`
    barely moves (0.527 -> 0.515). The clock is not taken away. What rough ground actually
    costs is SPEED: `lin_vel` 0.584 -> 0.299, a 49% loss at 20 cm. The old model predicted the
    wrong quantity would break, so it cannot set the bound.

    This version bounds on what was actually measured, in both directions:

    * TOO FLAT is the real risk on a warm start, and the first version had no opinion on it.
      Zero-shot falls for a trained walker: 5.25 cm -> 6.2%, i.e. it already solves 94% of the
      field and the run would most likely return NO EFFECT while being reported as terrain
      training. Below 8 cm is flagged.
    * TOO ROUGH is bounded at 20 cm because that is the largest relief anyone has measured
      here (64.1% zero-shot falls, at which point a warm start is barely on-distribution).
      Above it is UNMEASURED, not known-bad, and the finding says so rather than pretending
      to a physical limit.
    """
    t = getattr(config, "terrain", None)
    if t is None or not t.enabled:
        return [Finding(Severity.OK, "terrain amplitude", "terrain disabled")]

    p2p = t.amplitude_p2p
    if p2p > 0.20:
        return [Finding(
            Severity.SUSPECT, "terrain amplitude",
            f"relief {p2p*100:.1f} cm is above the 20 cm that has ever been measured on this "
            f"body. At 20 cm a trained walker already falls 64% zero-shot, so a warm start "
            f"above that is probably off-distribution -- but this is UNMEASURED, not known "
            f"to be wrong.",
            remedy="Measure zero-shot falls at this relief before spending a run on it.",
        )]
    if p2p < 0.08:
        return [Finding(
            Severity.SUSPECT, "terrain amplitude",
            f"relief {p2p*100:.2f} cm is mild: a trained walker takes only ~6% zero-shot falls "
            f"at 5.25 cm, so there may be nothing to learn and the run can return NO EFFECT "
            f"while being reported as terrain training.",
            remedy="Raise terrain.amplitude_p2p toward 0.14, where zero-shot falls are 36% and "
                   "64% of episodes still survive, or accept this as a control run.",
            caught_before="The first terrain run was configured at 5.25 cm on an argument "
                          "about the gait clock that measurement later refuted.",
        )]
    return [Finding(Severity.OK, "terrain amplitude",
                    f"{p2p*100:.1f} cm p2p, inside the measured 8-20 cm band")]


@check
def terrain_cell_supports_a_box_foot(config) -> list[Finding]:
    """A foot must rest on more than one contact point.

    Box feet were a deliberate model choice: MuJoCo's default capsule feet are line contacts
    and physically cannot produce a heel-to-toe roll (README.md:66). A heightfield cell
    coarser than the foot re-creates exactly that defect, because the foot bridges a single
    cell and can transmit no ankle torque from the ground. Measured on this body: at a
    0.15 m cell the median is 3 contacts per foot but the MINIMUM is 1; at 0.10 m the
    minimum is 2.
    """
    t = getattr(config, "terrain", None)
    if t is None or not t.enabled:
        return [Finding(Severity.OK, "terrain cell", "terrain disabled")]
    # The MimicKit foot is 0.177 x 0.090 m (geom half-sizes 0.0885 x 0.045). Contacts land
    # on grid vertices under the footprint, so the count along an axis is roughly
    # length/cell + 1. The binding requirement is on the LONG axis: it must span at least
    # 1.5 cells, so at least two grid lines cross the foot and it cannot pivot on one point.
    #
    # Calibrated against the measured contact minimum rather than asserted: at cell 0.150 m
    # the median is 3 contacts per foot but the MINIMUM is 1; at 0.100 m the minimum is 2.
    # 0.177 / 1.5 = 0.118 m puts the bound between the two measurements, which is where a
    # threshold derived from a model and checked against data should land.
    foot_long = 0.177
    max_cell = foot_long / 1.5
    if t.cell > max_cell:
        return [Finding(
            Severity.CONTRADICTION, "terrain cell",
            f"cell {t.cell:.3f} m exceeds {max_cell:.3f} m, so the foot's {foot_long:.3f} m "
            f"long axis spans fewer than 1.5 cells and can rest on a single contact point. "
            f"Measured at 0.150 m: median 3 contacts per foot, minimum 1. That is the "
            f"line-contact defect box feet were chosen to avoid.",
            remedy="Set terrain.cell at or below 0.10 m (measured minimum 2 contacts). Cost "
                   "is 2.01x the plane there, and depends on cell size only, never on the "
                   "field's extent.",
            caught_before="README.md:66 -- capsule feet are line contacts and physically "
                          "cannot produce a heel-to-toe roll, which is why this model has "
                          "box feet at all.",
        )]
    return [Finding(Severity.OK, "terrain cell",
                    f"cell {t.cell:.3f} m spans the {foot_long:.3f} m foot "
                    f"{foot_long / t.cell:.1f} times (bound {max_cell:.3f} m)")]


def run_all(config, checkpoint: Path | None = None) -> list[Finding]:
    """Every invariant. Failures inside a check are reported, never raised."""
    findings: list[Finding] = []
    if checkpoint is not None:
        try:
            findings.extend(check_checkpoint(checkpoint, config))
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding(Severity.SUSPECT, "check_checkpoint",
                                    f"check itself failed: {exc}"))
    for fn in _REGISTRY:
        try:
            findings.extend(fn(config))
        except Exception as exc:  # noqa: BLE001 - a broken check must not block a run
            findings.append(Finding(
                Severity.SUSPECT, fn.__name__,
                f"check itself failed: {type(exc).__name__}: {exc}",
                remedy="Fix the check; it is not reporting on the config.",
            ))
    order = {Severity.CONTRADICTION: 0, Severity.UNREACHABLE: 1,
             Severity.SUSPECT: 2, Severity.OK: 3}
    return sorted(findings, key=lambda f: order[f.severity])
