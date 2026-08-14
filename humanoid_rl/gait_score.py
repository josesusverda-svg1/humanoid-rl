"""Score a gait against measured human walking, on axes a person can actually see.

The reason this file exists is a list of failures. Every one of them scored well on episode
return, and every one was caught by a human watching a video rather than by a number:

* folding forward at the waist, pelvis perfectly level the whole time
* a 61 cm wide two-footed brace, sliding forward, 6 cm of travel per foot strike
* a one-sided gait, left foot in stance 0.58 of the time against the right's 0.30
* a fixed split stance rocking back and forth, right foot locked 36 cm in front, both feet
  pattering 7.4 times a second and the lead changing only 0.42 times a second
* pogoing 10.0 cm up and down on the leading foot

Return cannot distinguish any of those from walking. Each of the bands below can, and each
band is a measurement of real human walking rather than a threshold someone picked. A score
of 1.0 means "inside the range a person walks in", and falls off linearly outside it by the
width of the band, so 0.0 means "one full band width away from human".

The point is not the single number at the top. It is that when the number drops, the
breakdown says WHICH of these is wrong, in the units the defect actually occurs in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Band:
    """One measurable property of a gait, and the range humans walk in."""

    key: str  # metrics.jsonl key, without the "eval/" prefix
    label: str
    low: float
    high: float
    unit: str
    #: What this catches, in plain language. Shown in the dashboard on hover.
    catches: str
    #: Multiplier from the stored value to display units (e.g. metres to centimetres).
    display_scale: float = 1.0

    def score(self, value: float | None) -> float | None:
        """1.0 inside the human range, falling off linearly by one band width outside."""
        if value is None:
            return None
        width = max(self.high - self.low, 1e-9)
        if value < self.low:
            return max(0.0, 1.0 - (self.low - value) / width)
        if value > self.high:
            return max(0.0, 1.0 - (value - self.high) / width)
        return 1.0


@dataclass(frozen=True)
class Group:
    """A named cluster of bands, so a drop points at a part of the body or a behaviour."""

    name: str
    blurb: str
    bands: tuple[Band, ...]
    weight: float = 1.0


POSTURE = Group(
    "Posture",
    "Is it standing like a person, or folded over and splay-legged?",
    (
        Band("torso_upright", "Torso upright", 0.95, 1.0, "",
             "Folding forward at the waist. The pelvis can stay perfectly level and at the "
             "right height while the torso is horizontal, which is what happened at 24.6M "
             "steps and was invisible to every other metric."),
        Band("head_height_ratio", "Head height", 0.93, 1.0, "",
             "Stooping. Measured against this humanoid's own standing height, so it is a "
             "posture measure and not a size measure."),
        Band("stance_width", "Stance width", 0.10, 0.15, "cm",
             "Standing splay-legged for stability. A braced policy widened to 61 cm.",
             display_scale=100.0),
    ),
)

RHYTHM = Group(
    "Rhythm",
    "Is it alternating its legs, or holding a stance and rocking?",
    (
        Band("lead_swaps_per_sec", "Lead changes", 1.6, 2.0, "/s",
             "THE test for walking. A real walk trades which foot is in front on every "
             "step, so this equals the step rate. A gait that rocks in a fixed split "
             "stance scored 0.42 against 7.4 foot strikes a second."),
        Band("double_support", "Double support", 0.20, 0.25, "",
             "Refusing to lift a foot. 1.0 is standing still. Cannot be faked by pattering, "
             "unlike step rate, which looked healthy at 1.81 during a static brace."),
    ),
    weight=1.5,
)

SMOOTHNESS = Group(
    "Smoothness",
    "Does it move like a person, or bounce and skate?",
    (
        Band("vertical_bounce", "Vertical bounce", 0.04, 0.05, "cm",
             "Pogoing on the stance foot. Measured at 10.0 cm when nothing in the reward "
             "penalised vertical velocity.", display_scale=100.0),
        Band("foot_slip_speed", "Foot slip", 0.0, 0.10, "m/s",
             "Skating: planting a foot and sliding it. Tracks a commanded velocity "
             "perfectly and looks nothing like walking."),
    ),
)

BALANCE = Group(
    "Balance",
    "Do both legs do the same share of the work?",
    (
        Band("gait_symmetry", "Left/right evenness", 0.90, 1.0, "",
             "One leg driving while the other acts as a passive strut. Defined over SWING "
             "time, so standing on both feet scores 0 rather than a perfect 1."),
    ),
)

RELIABILITY = Group(
    "Reliability",
    "Does it stay up, and does it go where it is told?",
    (
        Band("fall_rate", "Staying upright", 0.0, 0.05, "",
             "Falling over. Scored last deliberately: a policy can be perfectly stable and "
             "still not be walking, which is how several of these failures survived."),
        Band("lin_vel_error", "Speed tracking", 0.0, 0.15, "m/s",
             "Ignoring the commanded velocity."),
        Band("heading_error_deg", "Holding a heading", 0.0, 8.0, " deg",
             "Wandering. Told to walk forward, a person holds a heading and arrives where "
             "they were pointed. This policy drifted 47 degrees in 11 seconds and finished "
             "2.6 m off the line, because the command is a turn RATE and its heading error "
             "was not even in its observation."),
        Band("lateral_speed", "Sideways drift", 0.0, 0.08, "m/s",
             "Sliding sideways while walking forward. Reached 53% of forward speed when "
             "sideways error shared one saturating kernel with forward error."),
    ),
)

GROUPS: tuple[Group, ...] = (POSTURE, RHYTHM, SMOOTHNESS, BALANCE, RELIABILITY)


@dataclass
class ScoredBand:
    band: Band
    value: float | None
    score: float | None

    def to_dict(self) -> dict:
        return {
            "key": self.band.key,
            "label": self.band.label,
            "value": None if self.value is None else self.value * self.band.display_scale,
            "low": self.band.low * self.band.display_scale,
            "high": self.band.high * self.band.display_scale,
            "unit": self.band.unit,
            "catches": self.band.catches,
            "score": self.score,
        }


@dataclass
class Scorecard:
    groups: list[dict] = field(default_factory=list)
    overall: float | None = None
    #: Bands that are furthest outside the human range, worst first. This is the part worth
    #: reading: it names the defect rather than only reporting that something is wrong.
    worst: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"overall": self.overall, "groups": self.groups, "worst": self.worst}


def score_row(row: dict, prefix: str = "eval/") -> Scorecard:
    """Score one evaluation row from metrics.jsonl."""
    card = Scorecard()
    weighted_total = 0.0
    weight_sum = 0.0
    all_scored: list[ScoredBand] = []

    for group in GROUPS:
        scored = [
            ScoredBand(band, row.get(f"{prefix}{band.key}"), band.score(row.get(f"{prefix}{band.key}")))
            for band in group.bands
        ]
        present = [s for s in scored if s.score is not None]
        all_scored.extend(present)
        group_score = sum(s.score for s in present) / len(present) if present else None
        if group_score is not None:
            weighted_total += group_score * group.weight
            weight_sum += group.weight
        card.groups.append({
            "name": group.name,
            "blurb": group.blurb,
            "score": group_score,
            "bands": [s.to_dict() for s in scored],
        })

    card.overall = weighted_total / weight_sum if weight_sum else None
    ranked = sorted(all_scored, key=lambda s: s.score)
    card.worst = [s.to_dict() for s in ranked if s.score < 0.999][:3]
    return card


def human_summary(card: Scorecard) -> str:
    """One line for a terminal, so the same scoring is usable outside the dashboard."""
    if card.overall is None:
        return "no evaluation data yet"
    parts = [f"{g['name']} {g['score']:.0%}" for g in card.groups if g["score"] is not None]
    line = f"human-likeness {card.overall:.0%}   " + "  ".join(parts)
    if card.worst:
        w = card.worst[0]
        line += f"\n  worst: {w['label']} at {w['value']:.2f}{w['unit']} " \
                f"(human {w['low']:.2f}-{w['high']:.2f})"
    return line
