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
