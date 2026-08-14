"""Does it do what it was told, broken down by what it was told.

The dashboard's other panels answer "is it walking" and "is it walking like a human". This
one answers the four questions a person actually asks while watching:

    how often does it fall
    how fast does it walk forward
    does it turn when told to turn
    can it walk backwards at all

None of those are answerable from the aggregate metrics. `lin_vel_error = 0.33 m/s` is
equally consistent with tracking forward perfectly while ignoring every turn, and with the
exact reverse. The task therefore logs tracking split by commanded direction, and this module
turns those raw sums into numbers with units.

Reference values, so a number means something on its own rather than only against yesterday:
human free walking is 1.2-1.4 m/s (Bohannon's normative meta-analysis). Backward and sideways
preferred speeds run roughly 60% and 40% of forward. XBot-L, the hardware-proven humanoid this
project conforms to, ships a command range of only [-0.3, 0.6] m/s, which is worth knowing
before reading our forward number as a disappointment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Categories as logged by LocomotionTask._per_direction_metrics.
CATEGORIES = [
    ("fwd", "Forward", "m/s", "human walking 1.2-1.4; XBot-L commands at most 0.6"),
    ("back", "Backward", "m/s", "humans prefer about 60% of their forward speed"),
    ("side", "Sideways", "m/s", "humans prefer about 40% of their forward speed"),
    ("turn", "Turning", "rad/s", "commanded range is +/-1.0"),
]


@dataclass
class Direction:
    key: str
    label: str
    unit: str
    note: str
    #: Fraction of evaluation time this direction was commanded at all.
    share: float
    #: Mean speed asked for, over the time it was asked for. Zero share means unmeasured.
    commanded: float | None
    #: Mean speed delivered, same window. Negative means it moved the WRONG way.
    achieved: float | None
    #: achieved / commanded. 1.0 is perfect, 0.0 is ignoring the command entirely, and
    #: below zero is actively going the other way.
    tracking: float | None

    @property
    def verdict(self) -> str:
        if self.tracking is None:
            return "not commanded"
        if self.tracking < 0.0:
            return "moves the wrong way"
        if self.tracking < 0.4:
            return "largely ignores it"
        if self.tracking < 0.75:
            return "follows it weakly"
        if self.tracking < 1.15:
            return "follows it"
        return "overshoots"


@dataclass
class CommandReport:
    iteration: int
    #: Percentage of evaluation episodes that ended in a fall rather than at the time limit.
    fall_rate: float
    #: Seconds the humanoid stayed up on average, which is what a fall rate cannot tell you.
    seconds_upright: float | None
    episode_limit_s: float | None
    directions: list[Direction] = field(default_factory=list)


def _ratio(num: float, den: float) -> float | None:
    return None if abs(den) < 1e-9 else num / den


def build(row: dict, control_dt: float = 0.008, episode_limit: int | None = None
          ) -> CommandReport:
    """Turn one evaluation row from metrics.jsonl into a report.

    `control_dt` converts step counts to seconds. It is 8 ms here (500 Hz physics with
    decimation 4). Passing the wrong value is not hypothetical: three config files carried
    comments assuming a 50 Hz loop this project has never had, which is how episodes came to
    be 8 seconds long while every comment said 20.
    """
    directions: list[Direction] = []
    for key, label, unit, note in CATEGORIES:
        share = float(row.get(f"eval/cmd_{key}_share", 0.0) or 0.0)
        target = float(row.get(f"eval/cmd_{key}_target", 0.0) or 0.0)
        actual = float(row.get(f"eval/cmd_{key}_actual", 0.0) or 0.0)
        directions.append(Direction(
            key=key, label=label, unit=unit, note=note, share=share,
            commanded=_ratio(target, share),
            achieved=_ratio(actual, share),
            tracking=_ratio(actual, target),
        ))

    length = row.get("eval/episode_length")
    return CommandReport(
        iteration=int(row.get("iteration", 0)),
        fall_rate=float(row.get("eval/fall_rate", 0.0) or 0.0),
        seconds_upright=None if length is None else float(length) * control_dt,
        episode_limit_s=None if episode_limit is None else episode_limit * control_dt,
        directions=directions,
    )


def to_dict(report: CommandReport) -> dict:
    return {
        "iteration": report.iteration,
        "fall_rate": report.fall_rate,
        "seconds_upright": report.seconds_upright,
        "episode_limit_s": report.episode_limit_s,
        "directions": [
            {
                "key": d.key, "label": d.label, "unit": d.unit, "note": d.note,
                "share": d.share, "commanded": d.commanded, "achieved": d.achieved,
                "tracking": d.tracking, "verdict": d.verdict,
            }
            for d in report.directions
        ],
    }
