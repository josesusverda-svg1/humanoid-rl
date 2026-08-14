"""Typed configuration, loaded from a single YAML file.

One file controls an entire run. It is snapshotted into the run directory at startup, so
a checkpoint always carries the exact configuration that produced it and a resumed run
cannot silently drift from the original.

Values of `null` for hardware-dependent fields (`num_workers`, `device`) mean "detect at
runtime", which is what keeps the same config portable across Apple Silicon machines with
different core counts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from humanoid_rl.algos.ppo import PPOConfig
from humanoid_rl.envs.domain_rand import DomainRandConfig
from humanoid_rl.tasks.locomotion import LocomotionConfig
from humanoid_rl.tasks.tracking import TrackingConfig
from humanoid_rl.algos.amp import AMPConfig
from humanoid_rl.tasks.amp_locomotion import AMPLocomotionConfig

T = TypeVar("T")


@dataclass
class RunConfig:
    name: str = "humanoid"
    seed: int = 0
    #: "auto" resolves to mps when available, else cpu. Phase 0 measured torch-cpu as
    #: 4x to 5x slower here, and it would also compete with the physics threads.
    device: str = "auto"
    #: Which objective to train. "locomotion" follows velocity commands (Phase 2),
    #: "tracking" reproduces reference mocap frame by frame (Phase 3 Stage 1),
    #: "amp" follows velocity commands with an adversarial motion prior (Stage 2).
    task: str = "locomotion"
    #: Learning algorithm. "ppo" is the shipped path; "fasttd3" selects the off-policy
    #: trainer. They share the environment, reward, observations and humanoid model
    #: entirely, and differ only in what consumes the transitions.
    algo: str = "ppo"
    total_env_steps: int = 500_000_000
    output_dir: str = "runs"


@dataclass
class EnvConfig:
    model_path: str = "humanoid_rl/models/humanoid.xml"
    num_envs: int = 4096
    #: null means one worker per performance core, detected at runtime.
    num_workers: int | None = None
    #: Physics steps per policy action. humanoid.xml runs at 200 Hz, so 4 gives a 50 Hz
    #: policy, the standard rate for locomotion control.
    decimation: int = 4
    max_episode_steps: int = 1000
    #: Cutoff of the one-pole low-pass on applied joint targets, in Hz. 0 disables. Exists
    #: because a policy learned to stabilise its gait with its own high-frequency action
    #: noise (vibrational stabilisation), which made deterministic evaluation collapse while
    #: noisy training looked excellent. Gait content is 1-3 Hz and passes; tremor does not.
    action_filter_hz: float = 8.0


@dataclass
class NetworkConfig:
    actor_hidden: tuple[int, ...] = (512, 512)
    critic_hidden: tuple[int, ...] = (512, 512)
    activation: str = "elu"
    init_noise_std: float = 1.0


@dataclass
class EvalConfig:
    #: Run a deterministic evaluation every N iterations. Separate from training so the
    #: reported score is never inflated by exploration noise.
    interval_iterations: int = 50
    num_episodes: int = 32
    num_envs: int = 64
    #: Render an mp4 every N evaluations, and always when a new best score is reached.
    #: Rendering costs roughly 1.3x realtime, so at the default eval interval this adds
    #: about 3 percent to total training time. Set video_every_n_evals to 0 to disable.
    video_every_n_evals: int = 5
    #: Capture a multi-view stick figure every this many million environment steps.
    #:
    #: Far cheaper than a video and meant to be frequent: 0.59 s and 80 KB against 18 s and
    #: 11 MB, because it does no rendering at all. It records body positions and projects
    #: them orthographically, which is two dot products per body, so the run accumulates a
    #: flip-book of the gait developing rather than a handful of expensive clips. 0 disables.
    skeleton_every_m_steps: float = 4.0
    video_on_best: bool = True
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 50
    #: Keep this many of the most recent periodic videos, plus every best-score video.
    #: A multi-day run would otherwise fill the disk with near-identical clips.
    keep_last_videos: int = 12


@dataclass
class LogConfig:
    checkpoint_interval_iterations: int = 100
    #: Keep this many recent checkpoints plus the best, to bound disk use on long runs.
    keep_last_checkpoints: int = 5
    log_interval_iterations: int = 1


from humanoid_rl.algos.fasttd3 import FastTD3Config  # noqa: E402


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    task: LocomotionConfig = field(default_factory=LocomotionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    amp_task: AMPLocomotionConfig = field(default_factory=AMPLocomotionConfig)
    #: Off-policy alternative to `ppo`, selected by run.algo = "fasttd3". Ignored otherwise,
    #: so a PPO run carries these defaults harmlessly and the two paths never interfere.
    fasttd3: FastTD3Config = field(default_factory=FastTD3Config)
    #: Directory of retargeted clips, and a filter over their names.
    clip_dir: str = "data/clips"
    clip_include: tuple[str, ...] = ("FW", "BW", "SW", "TR1", "ID")
    #: Also train on the left-right reflection of every clip. The performer's own gait is
    #: mildly uneven (1.08 on stance fraction, 1.14 median across joint pairs), and AMP
    #: reproduces whatever it is shown, so mirroring removes that bias from the target.
    mirror_clips: bool = False
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # ------------------------------------------------------------------ io

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return _from_dict(Config, raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, indent=2))

    # ------------------------------------------------------------------ derived

    @property
    def steps_per_iteration(self) -> int:
        """Environment steps collected per PPO iteration, across all environments."""
        return self.env.num_envs * self.ppo.horizon

    @property
    def total_iterations(self) -> int:
        return max(1, self.run.total_env_steps // self.steps_per_iteration)


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a nested dataclass from a plain dict, converting tuple-typed fields.

    Raises on unknown keys rather than ignoring them. A silently ignored typo in a config
    file is one of the more expensive ways to waste a multi-hour training run.

    Note on `get_type_hints`: this module uses `from __future__ import annotations`, so
    `dataclasses.fields()` reports each field's type as a *string* rather than the class.
    Resolving the hints properly is what allows nested sections to be recursed into.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown config keys for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)  # type: ignore[arg-type]
        elif isinstance(value, list):
            # YAML has no tuple type, so any list feeding a tuple-typed field is converted.
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


def resolve_device(requested: str) -> str:
    """Turn "auto" into a concrete torch device string, verifying it actually works.

    Checks that MPS is genuinely usable rather than just reported as built, because a
    silent fallback to CPU costs 4x to 5x and is otherwise invisible.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
