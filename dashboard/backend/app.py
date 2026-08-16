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


# ----------------------------------------------------------------- getup mission control


GETUP_KEYS = [
    "iteration", "reward/track", "reward/stand", "reward/hold", "reward/latch",
    "reward/lift", "reward/launch", "action_std", "approx_kl", "lr",
    "eval/standing_frac", "eval/standing_frac_strict", "eval/held_ever_frac",
    "eval/exam_level", "eval/root_height", "eval/head_height_ratio",
    "eval/gate_frac", "eval/launch_overspeed_frac", "eval/episode_return",
    "eval/foot_load_bw", "eval/knee_max",
]


@app.get("/api/runs/{run_id}/getup")
def get_getup(run_id: str, max_points: int = Query(600, ge=50, le=4000)) -> dict[str, Any]:
    """Everything the get-up console plots, in one call.

    Train-side rows and eval rows are interleaved in metrics.jsonl; the console wants both,
    plus a windowed latch RATE (the raw latch column is a spike train that reads as noise
    when plotted directly: what matters is how often holds complete, not the batch-mean of
    a once-per-episode bonus).
    """
    run_dir = _safe_run_dir(run_id)
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.jsonl not found")
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    latch_window: list[int] = []
    # NOT an if/elif chain on the two key families: the trainer writes evaluation results
    # INTO the same row as that iteration's training metrics, so an elif on "reward/latch"
    # swallowed every eval row and the console's evaluation panels were silently empty.
    # Each row is classified independently, and a row can be both.
    for r in _read_jsonl(path):
        if "reward/latch" in r:
            latch_window.append(1 if (r.get("reward/latch") or 0) > 0 else 0)
            if len(latch_window) > 100:
                latch_window.pop(0)
            row = {k: r.get(k) for k in GETUP_KEYS if k in r}
            row["latch_rate_100"] = sum(latch_window) / max(len(latch_window), 1)
            train_rows.append(row)
        if "eval/episode_return" in r:
            eval_rows.append({k: r.get(k) for k in GETUP_KEYS if k in r})
    return {
        "train": _downsample(train_rows, max_points),
        "eval": eval_rows[-max_points:],
        "total_train_rows": len(train_rows),
    }


@app.get("/api/inspection")
def get_inspection() -> dict[str, Any]:
    """The latest visual-inspection artefacts the watch loop produces.

    The 13-conjunct breakdown and per-frame telemetry only exist in the watch script's
    stdout; the loop stores its last capture at /tmp/w.txt and its frame strip at
    /tmp/getup_watch.png. Parsing that capture here means the dashboard shows exactly what
    the last human-grade inspection saw, with its age, rather than pretending to a live
    feed it does not have.
    """
    out: dict[str, Any] = {"conjuncts": [], "frames": [], "age_seconds": None,
                           "has_strip": False, "has_reference": False}
    cap = Path("/tmp/w.txt")
    if cap.exists():
        import re
        import time
        out["age_seconds"] = round(time.time() - cap.stat().st_mtime)
        section = None
        for line in cap.read_text(errors="ignore").splitlines():
            if "standing conjuncts" in line:
                section = "conjuncts"
                continue
            if "rendering frames" in line:
                section = "frames"
                continue
            if section == "conjuncts":
                m = re.match(r"\s+(\d+)\s+(.+?)\s+([\d.]+)%", line)
                if m:
                    out["conjuncts"].append({
                        "index": int(m.group(1)),
                        "name": m.group(2).strip(),
                        "frac": float(m.group(3)) / 100.0,
                    })
            elif section == "frames":
                m = re.match(r"\s*([\d.]+)s\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
                             r"\s+([\d.-]+)\s+([\d.-]+)", line)
                if m:
                    out["frames"].append({
                        "t": float(m.group(1)), "pelvis": float(m.group(2)),
                        "head": float(m.group(3)), "feet_bw": float(m.group(4)),
                        "knee": float(m.group(5)), "hands": float(m.group(6)),
                    })
    out["has_strip"] = Path("/tmp/getup_watch.png").exists()
    out["has_reference"] = Path("/tmp/getup_reference.png").exists()
    return out


@app.get("/api/inspection/strip")
def get_inspection_strip() -> FileResponse:
    p = Path("/tmp/getup_watch.png")
    if not p.exists():
        raise HTTPException(status_code=404, detail="no frame strip yet")
    return FileResponse(p, media_type="image/png")


@app.get("/api/inspection/reference")
def get_inspection_reference() -> FileResponse:
    p = Path("/tmp/getup_reference.png")
    if not p.exists():
        raise HTTPException(status_code=404, detail="no reference strip")
    return FileResponse(p, media_type="image/png")


@app.get("/api/logbook")
def get_logbook() -> list[dict[str, Any]]:
    """docs/LOGBOOK.md parsed into experiment cards, newest first.

    The logbook is the lab's institutional memory; surfacing it beside the live run is what
    separates a console from a metrics page. Verdicts are extracted by convention (the
    logbook's own rule: every entry carries one of five verdicts).
    """
    import re
    path = REPO_ROOT / "docs" / "LOGBOOK.md"
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    entries = []
    blocks = re.split(r"^### ", text, flags=re.M)[1:]
    for b in blocks:
        lines = b.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:])
        verdict = None
        m = re.search(r"\b(WORKED|NO EFFECT|WORSE|INVALID|MIXED|RETRACTED|"
                      r"PRE-REGISTERED|TRIGGERED|NnO EFFECT)\b", title + " " + body[:800])
        if m:
            verdict = m.group(1)
        num = re.match(r"E(\d+)", title)
        entries.append({
            "title": title,
            "verdict": verdict,
            "excerpt": body[:1600],
            "_num": int(num.group(1)) if num else -1,
        })
    entries.sort(key=lambda e: e["_num"], reverse=True)
    for e in entries:
        e.pop("_num")
    return entries


_COMPARE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


@app.get("/api/getup/compare")
def get_compare() -> list[dict[str, Any]]:
    """One headline row per get-up run, for the cross-run table. Cached by file mtime."""
    out = []
    for d in sorted(RUNS_DIR.glob("getup-*")):
        path = d / "metrics.jsonl"
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        cached = _COMPARE_CACHE.get(d.name)
        if cached and cached[0] == mtime:
            out.append(cached[1])
            continue
        iters = 0
        latch_count = 0
        max_strict = 0.0
        max_stand = 0.0
        last_level = 0.0
        last_iter = 0
        for r in _read_jsonl(path):
            if "reward/latch" in r:
                iters += 1
                if (r.get("reward/latch") or 0) > 0:
                    latch_count += 1
            if "eval/episode_return" in r:
                max_strict = max(max_strict, r.get("eval/standing_frac_strict") or 0)
                max_stand = max(max_stand, r.get("eval/standing_frac") or 0)
                last_level = r.get("eval/exam_level") or last_level
                last_iter = int(r.get("iteration") or last_iter)
        row = {
            "id": d.name, "iterations": iters, "latch_iters": latch_count,
            "latch_share": round(latch_count / iters, 4) if iters else 0.0,
            "max_standing_frac": max_stand, "max_strict_frac": max_strict,
            "final_exam_level": last_level, "modified": mtime,
        }
        _COMPARE_CACHE[d.name] = (mtime, row)
        out.append(row)
    out.sort(key=lambda r: r["modified"], reverse=True)
    return out


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
