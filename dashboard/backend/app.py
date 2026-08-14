"""FastAPI backend for the training dashboard.

Design notes:

* **The trainer serves nothing.** It only appends JSONL files. This backend reads them.
  That separation means the dashboard can be started, stopped, or crash without ever
  touching a multi-day training run, and a finished run stays fully browsable afterwards.

* **Polling, not WebSockets.** Training emits roughly one metrics row every 1.6 seconds, so
  a 2 second poll is already live. Polling is dramatically simpler to reason about and
  maintain than a socket lifecycle, and it reconnects for free.

* **Downsampling happens server-side.** A multi-day run produces hundreds of thousands of
  rows. Sending them all would make the browser unusable. Series are reduced to a bounded
  number of points using min/max-preserving buckets, so spikes survive the reduction rather
  than being averaged away, which matters because spikes are usually the interesting part.

Run it with:
    .venv/bin/python scripts/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# The dashboard scores gaits with `humanoid_rl.gait_score`, so the repo root has to be
# importable no matter which directory uvicorn was started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = Path(os.environ.get("HUMANOID_RUNS_DIR", REPO_ROOT / "runs"))
FRONTEND_DIST = REPO_ROOT / "dashboard" / "frontend" / "dist"

app = FastAPI(title="Humanoid RL Dashboard", version="0.1.0")

# Permissive CORS so `npm run dev` on port 5173 can talk to this on 8000 during frontend
# development. The server binds to localhost only, so this is not an exposure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------- helpers


def _safe_run_dir(run_id: str) -> Path:
    """Resolve a run id to a directory, refusing anything that escapes RUNS_DIR.

    The run id arrives from the URL, so it is untrusted. Without this check a request for
    `../../etc` would happily read outside the runs tree.
    """
    candidate = (RUNS_DIR / run_id).resolve()
    if not candidate.is_dir() or RUNS_DIR.resolve() not in candidate.parents:
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return candidate


def _read_jsonl(path: Path, skip: int = 0) -> Iterator[dict[str, Any]]:
    """Yield parsed JSONL rows, tolerating a torn final line.

    A run that is still training may be mid-write when we read, leaving the last line
    incomplete. Skipping it is correct: the next poll will pick it up whole.
    """
    if not path.exists():
        return
    with path.open() as handle:
        for i, line in enumerate(handle):
            if i < skip:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _downsample(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    """Reduce rows to at most `max_points`, keeping extremes within each bucket.

    Plain stride sampling would drop spikes entirely and make a noisy loss curve look
    clean, which is actively misleading. This keeps the first, min, max and last row of
    each bucket, so the visible envelope of the data is preserved.
    """
    n = len(rows)
    if n <= max_points:
        return rows

    bucket = max(1, n // (max_points // 4))
    out: list[dict[str, Any]] = []
    key = "episode_return"
    for start in range(0, n, bucket):
        chunk = rows[start : start + bucket]
        if not chunk:
            continue
        values = [(r.get(key), r) for r in chunk if isinstance(r.get(key), (int, float))]
        picked = [chunk[0]]
        if values:
            picked.append(min(values, key=lambda kv: kv[0])[1])
            picked.append(max(values, key=lambda kv: kv[0])[1])
        picked.append(chunk[-1])
        # Preserve chronological order and drop duplicates within the bucket.
        seen: set[int] = set()
        for row in sorted(picked, key=lambda r: r.get("iteration", 0)):
            marker = id(row)
            if marker not in seen:
                seen.add(marker)
                out.append(row)
    return out


@dataclass
class RunSummary:
    id: str
    name: str
    started: float
    modified: float
    iterations: int
    env_steps: int
    episode_return: float | None
    success_rate: float | None
    env_steps_per_sec: float | None
    wall_hours: float | None
    checkpoints: int
    videos: int
    active: bool


def _last_line(path: Path) -> dict[str, Any] | None:
    """Read only the final JSONL row, without loading the whole file.

    Run listing touches every run directory, and on a long project those files reach
    hundreds of megabytes. Seeking from the end keeps the list endpoint fast.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        block = min(65536, size)
        handle.seek(size - block)
        tail = handle.read(block).decode("utf-8", errors="ignore")
    for line in reversed(tail.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


# ----------------------------------------------------------------------------- endpoints


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "runs_dir": str(RUNS_DIR), "exists": RUNS_DIR.exists()}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    """All runs, the live one first and the rest newest first.

    The ordering matters because the frontend takes the first entry as its default. This
    used to sort by NAME descending, which put `phase2-walking-121M` above every `amp-*`
    run purely because "p" sorts after "a", so opening the dashboard during training landed
    on an idle run from days ago. Sorting by activity and then by recency is what the old
    docstring already claimed was happening.
    """
    import time

    if not RUNS_DIR.exists():
        return []

    out: list[RunSummary] = []
    now = time.time()
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.jsonl"
        last = _last_line(metrics_path)
        modified = metrics_path.stat().st_mtime if metrics_path.exists() else run_dir.stat().st_mtime
        out.append(
            RunSummary(
                id=run_dir.name,
                name=run_dir.name.rsplit("-", 2)[0],
                started=run_dir.stat().st_ctime,
                modified=modified,
                iterations=int(last.get("iteration", 0)) if last else 0,
                env_steps=int(last.get("env_steps", 0)) if last else 0,
                episode_return=last.get("episode_return") if last else None,
                success_rate=last.get("success_rate") if last else None,
                env_steps_per_sec=last.get("env_steps_per_sec") if last else None,
                wall_hours=last.get("total_wall_hours") if last else None,
                # A run whose metrics file changed in the last 60 seconds is almost
                # certainly still training. Cheap and reliable enough for a status dot.
                active=(now - modified) < 60.0,
                checkpoints=len(list((run_dir / "checkpoints").glob("*.pt")))
                if (run_dir / "checkpoints").is_dir()
                else 0,
                videos=len(list((run_dir / "videos").glob("*.mp4")))
                if (run_dir / "videos").is_dir()
                else 0,
            )
        )
    # Live run first, then most recently written. `modified` is negated rather than using
    # reverse=True so that the `active` key keeps its ascending sense in the same tuple.
    out.sort(key=lambda r: (not r.active, -r.modified))
    return [r.__dict__ for r in out]


@app.get("/api/runs/{run_id}/metrics")
def get_metrics(
    run_id: str,
    since: int = Query(0, ge=0, description="skip this many leading rows"),
    max_points: int = Query(1500, ge=50, le=20000),
) -> dict[str, Any]:
    """Metric rows for one run, downsampled.

    `since` lets the frontend fetch only new rows while a run is live, so polling stays
    cheap regardless of how long the run has been going.
    """
    run_dir = _safe_run_dir(run_id)
    rows = list(_read_jsonl(run_dir / "metrics.jsonl", skip=since))
    total = since + len(rows)
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return {
        "run_id": run_id,
        "total_rows": total,
        "returned": len(rows),
        "keys": sorted(keys),
        "rows": _downsample(rows, max_points),
    }


@app.get("/api/runs/{run_id}/gait")
def get_gait(run_id: str, history: int = Query(500, ge=1, le=5000)) -> dict[str, Any]:
    """How human the current gait is, scored against measured human walking.

    Exists because episode return cannot tell walking from several things that are not
    walking. Every failure this project has hit scored well on return and was caught by a
    person watching a video: folding at the waist, a sliding brace, a one-sided gait, a
    rocking split stance, pogoing. `humanoid_rl.gait_score` holds the bands and the reason
    each one is there.
    """
    from humanoid_rl.gait_score import score_row

    run_dir = _safe_run_dir(run_id)
    evaluations = [
        row for row in _read_jsonl(run_dir / "metrics.jsonl")
        if "eval/episode_return" in row
    ]
    if not evaluations:
        return {"run_id": run_id, "overall": None, "groups": [], "worst": [], "history": []}

    card = score_row(evaluations[-1])

    # Full history, per group as well as overall. The per-group lines are the useful part:
    # the overall score alone tells you it dropped, and the group lines tell you whether it
    # dropped because the gait stopped alternating or because it started falling over.
    trail: list[dict[str, Any]] = []
    for row in evaluations[-history:]:
        scored = score_row(row)
        if scored.overall is None:
            continue
        point: dict[str, Any] = {
            "iteration": int(row.get("iteration", 0)),
            "env_steps": int(row.get("env_steps", 0)),
            "overall": scored.overall,
        }
        for group in scored.groups:
            point[group["name"]] = group["score"]
        trail.append(point)
    return {
        "run_id": run_id,
        "iteration": int(evaluations[-1].get("iteration", 0)),
        **card.to_dict(),
        "history": trail,
        "group_names": [g["name"] for g in card.groups],
    }


@app.get("/api/runs/{run_id}/commands")
def get_commands(run_id: str, history: int = Query(300, ge=1, le=5000)) -> dict[str, Any]:
    """Does it do what it was told, split by what it was told.

    Answers the four questions the aggregate metrics cannot: how often it falls, how fast it
    walks forward, whether it turns on command, and whether it can walk backwards. See
    humanoid_rl/command_report.py for why the raw metrics are logged as sums.
    """
    import yaml

    from humanoid_rl import command_report

    run_dir = _safe_run_dir(run_id)
    evaluations = [
        row for row in _read_jsonl(run_dir / "metrics.jsonl")
        if "eval/episode_return" in row
    ]
    if not evaluations:
        return {"run_id": run_id, "latest": None, "history": [], "supported": False}

    # The control step comes from the run's own config, never a constant here. A run started
    # before the episode-length fix has a different limit, and reporting its seconds against
    # today's number would misdate every one of them.
    limit = None
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        limit = (config.get("env") or {}).get("max_episode_steps")

    latest = command_report.build(evaluations[-1], episode_limit=limit)
    # Runs started before the per-direction metrics existed have no shares at all. Say so
    # explicitly rather than rendering a panel of zeros that looks like a policy that never
    # moves.
    supported = any(d.share > 0.0 for d in latest.directions)

    trail = [
        {
            "iteration": r.iteration,
            "fall_rate": r.fall_rate,
            "seconds_upright": r.seconds_upright,
            **{f"{d.key}_achieved": d.achieved for d in r.directions},
            **{f"{d.key}_tracking": d.tracking for d in r.directions},
        }
        for r in (command_report.build(row, episode_limit=limit)
                  for row in evaluations[-history:])
    ]
    return {
        "run_id": run_id,
        "latest": command_report.to_dict(latest),
        "history": trail,
        "supported": supported,
    }


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str, limit: int = Query(500, ge=1, le=5000)) -> list[dict[str, Any]]:
    run_dir = _safe_run_dir(run_id)
    events = list(_read_jsonl(run_dir / "events.jsonl"))
    return events[-limit:]


@app.get("/api/runs/{run_id}/config")
def get_config(run_id: str) -> dict[str, Any]:
    import yaml

    run_dir = _safe_run_dir(run_id)
    path = run_dir / "config.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail="config.yaml not found for this run")
    return yaml.safe_load(path.read_text()) or {}


@app.get("/api/runs/{run_id}/skeletons")
def list_skeletons(run_id: str) -> list[dict[str, Any]]:
    """Stick-figure captures for a run, oldest first, as a flip-book of the gait."""
    run_dir = _safe_run_dir(run_id)
    folder = run_dir / "skeletons"
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        meta = payload.get("meta", {})
        out.append({
            "name": path.stem,
            "iteration": meta.get("iteration", 0),
            "env_steps": meta.get("env_steps", 0),
            "size_kb": round(path.stat().st_size / 1024, 1),
        })
    return out


@app.get("/api/runs/{run_id}/skeletons/{name}")
def get_skeleton(run_id: str, name: str) -> dict[str, Any]:
    run_dir = _safe_run_dir(run_id)
    # Reject anything with a path separator rather than sanitising it: the filenames this
    # writes are always `iter_NNNNNNNN`, so a name containing a slash is not a typo.
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="bad name")
    path = run_dir / "skeletons" / f"{name}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such capture")
    return json.loads(path.read_text())


@app.get("/api/runs/{run_id}/videos")
def list_videos(run_id: str) -> list[dict[str, Any]]:
    """Videos rendered during evaluation, newest first, with their metadata sidecars.

    Phase 3 populates this. The endpoint exists now so the frontend gallery has a stable
    contract to build against.
    """
    run_dir = _safe_run_dir(run_id)
    video_dir = run_dir / "videos"
    if not video_dir.is_dir():
        return []

    import re

    out = []
    for path in sorted(video_dir.glob("*.mp4"), reverse=True):
        meta_path = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}

        # Derive iteration and tag from the filename (iter_00000050_best.mp4) rather than
        # relying on the sidecar. The filename is written by the same code path that
        # renders, so it is always present even if the sidecar is missing or truncated by
        # a crash mid-write.
        match = re.match(r"iter_(\d+)_(\w+)\.mp4$", path.name)
        derived: dict[str, Any] = {}
        if match:
            derived = {"iteration": int(match.group(1)), "tag": match.group(2)}

        out.append(
            {
                "name": path.name,
                "url": f"/api/runs/{run_id}/videos/{path.name}",
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
                **derived,
                **meta,
            }
        )
    return out


@app.get("/api/runs/{run_id}/videos/{filename}")
def get_video(run_id: str, filename: str) -> FileResponse:
    run_dir = _safe_run_dir(run_id)
    # Reject any path component in the filename, for the same reason as _safe_run_dir.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    path = (run_dir / "videos" / filename).resolve()
    if not path.is_file() or run_dir.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(path, media_type="video/mp4")


# The built frontend is mounted last so it never shadows an /api route.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
