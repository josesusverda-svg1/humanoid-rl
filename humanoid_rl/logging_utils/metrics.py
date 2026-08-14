"""Structured run logging: JSONL metrics, events, and a run directory layout.

The dashboard reads these files directly, so nothing here may buffer for long. Every line
is flushed as it is written, which makes the dashboard live without needing the trainer to
serve anything itself. It also means a crashed run still leaves a complete, valid log up to
the moment it died.

Run directory layout:

    runs/<name>-<timestamp>/
        config.yaml          exact resolved config that produced this run
        metrics.jsonl        one JSON object per training iteration
        events.jsonl         checkpoints written, videos rendered, evaluations, errors
        checkpoints/         *.pt, plus best.pt
        videos/              *.mp4 with sidecar metadata

JSON Lines is used rather than a binary event format so the files stay greppable, are
readable with standard tools while training runs, and never corrupt on an abrupt kill.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class RunLogger:
    """Owns a run directory and appends to its JSONL streams."""

    def __init__(self, output_dir: str | Path, run_name: str, resume_dir: str | Path | None = None):
        if resume_dir is not None:
            self.run_dir = Path(resume_dir)
            if not self.run_dir.exists():
                raise FileNotFoundError(f"cannot resume, run directory not found: {self.run_dir}")
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_dir = Path(output_dir) / f"{run_name}-{stamp}"
            self.run_dir.mkdir(parents=True, exist_ok=False)

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.video_dir = self.run_dir / "videos"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._events_path = self.run_dir / "events.jsonl"
        self._metrics_file = self._metrics_path.open("a", buffering=1)
        self._events_file = self._events_path.open("a", buffering=1)
        self._start_time = time.time()

    # ------------------------------------------------------------------ writing

    def log_metrics(self, iteration: int, env_steps: int, metrics: dict[str, Any]) -> None:
        record = {
            "iteration": int(iteration),
            "env_steps": int(env_steps),
            "wall_time": round(time.time() - self._start_time, 3),
            "timestamp": time.time(),
            **{k: _jsonable(v) for k, v in metrics.items()},
        }
        self._metrics_file.write(json.dumps(record) + "\n")

    def log_event(self, kind: str, **payload: Any) -> None:
        record = {
            "kind": kind,
            "timestamp": time.time(),
            "wall_time": round(time.time() - self._start_time, 3),
            **{k: _jsonable(v) for k, v in payload.items()},
        }
        self._events_file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._metrics_file.close()
        self._events_file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays into plain JSON types."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        # JSON has no NaN or Infinity. Emitting them produces a file the dashboard's
        # JSON parser rejects, which would break the whole live view.
        return None
    return value


@dataclass
class EpisodeStats:
    """Rolling statistics over recently completed episodes.

    The vectorised environment reports episode results only on the steps where episodes
    happen to end, scattered across environments. This accumulates them into a stable
    window so the dashboard shows a smooth curve rather than a spiky one that depends on
    how many episodes happened to finish in a given iteration.
    """

    window: int = 200

    def __post_init__(self) -> None:
        self.returns: deque[float] = deque(maxlen=self.window)
        self.lengths: deque[int] = deque(maxlen=self.window)
        self.successes: deque[bool] = deque(maxlen=self.window)
        self.total_episodes = 0

    def add_batch(
        self, done: np.ndarray, returns: np.ndarray, lengths: np.ndarray, successes: np.ndarray
    ) -> None:
        idx = np.flatnonzero(done)
        if idx.size == 0:
            return
        self.returns.extend(returns[idx].tolist())
        self.lengths.extend(lengths[idx].tolist())
        self.successes.extend(successes[idx].astype(bool).tolist())
        self.total_episodes += int(idx.size)

    def summary(self) -> dict[str, float]:
        if not self.returns:
            return {
                "episode_return": 0.0,
                "episode_length": 0.0,
                "success_rate": 0.0,
                "episodes_total": float(self.total_episodes),
            }
        return {
            "episode_return": float(np.mean(self.returns)),
            "episode_return_std": float(np.std(self.returns)),
            "episode_length": float(np.mean(self.lengths)),
            "success_rate": float(np.mean(self.successes)),
            "episodes_total": float(self.total_episodes),
        }
