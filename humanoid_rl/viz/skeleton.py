"""Cheap multi-view stick figures of how the policy walks right now.

Rendering a video costs about 18 seconds and 11 MB, which is why it happens rarely. This
costs a few milliseconds and a few tens of kilobytes, so it can happen every few million
steps and give a flip-book of the gait developing across a whole run.

The trick is that no rendering is involved. A body's screen position under an orthographic
camera is two dot products:

    screen_x = position . right      screen_y = position . up

so a whole pose from six camera angles is a small matrix multiply. There is no rasteriser,
no OpenGL context, no encoder. The output is joint coordinates, and the dashboard draws them
as SVG lines, which also means the result is a few KB of numbers rather than a video file
and stays sharp at any size.

The view is root-centred horizontally, the way a treadmill gait analysis is presented: the
humanoid stays in frame while its limbs move, and vertical position is left absolute so
ground contact and bounce are visible against the floor line. Foot and hand trails carry the
forward travel that centring removes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

#: Camera azimuths in degrees, measured about the vertical axis. 0 looks at the humanoid
#: from straight ahead; 90 from its left. Six views because a gait defect is often invisible
#: from one angle: the one-sided gait read as normal from the front and was obvious from the
#: side, and the split-stance rocking was the other way round.
DEFAULT_VIEWS: tuple[tuple[str, float], ...] = (
    ("front", 0.0),
    ("front-left", 45.0),
    ("left", 90.0),
    ("back", 180.0),
    ("right", 270.0),
    ("front-right", 315.0),
)

#: Bodies to draw and the segments between them. Taken from the model's own body tree at
#: capture time, so this is only the fallback ordering for presentation.
TRAIL_BODIES = ("left_foot", "right_foot", "left_hand", "right_hand")


@dataclass
class SkeletonCapture:
    """Joint trajectories, projected to several 2-D views."""

    bodies: list[str]
    bones: list[tuple[int, int]]
    #: (n_views, n_frames, n_bodies, 2) screen coordinates in metres.
    views: np.ndarray
    view_names: list[str]
    fps: float
    #: Indices into `bodies` whose paths are worth drawing as trails.
    trails: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "bodies": self.bodies,
            "bones": self.bones,
            "view_names": self.view_names,
            "fps": self.fps,
            "trails": self.trails,
            # Rounded to the millimetre. The difference is invisible at any plausible size
            # and it roughly halves the payload.
            "views": np.round(self.views, 3).tolist(),
            "meta": self.meta,
        }


def body_tree(model: mujoco.MjModel) -> tuple[list[str], list[tuple[int, int]]]:
    """Every body except the world, and the parent-child segments between them."""
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        for b in range(1, model.nbody)
    ]
    index = {name: i for i, name in enumerate(names)}
    bones: list[tuple[int, int]] = []
    for b in range(1, model.nbody):
        parent = int(model.body_parentid[b])
        if parent < 1:
            continue  # child of the world, so no segment to draw
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)
        if child_name in index and parent_name in index:
            bones.append((index[parent_name], index[child_name]))
    return names, bones


def _basis(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Screen right and up vectors for an orthographic camera at this azimuth."""
    a = np.radians(azimuth_deg)
    # Camera sits at (cos a, sin a) and looks toward the origin.
    forward = np.array([-np.cos(a), -np.sin(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return right, up


def project(
    positions: np.ndarray, heading: np.ndarray | None = None, views=DEFAULT_VIEWS
) -> tuple[np.ndarray, list[str]]:
    """(n_frames, n_bodies, 3) world positions to (n_views, n_frames, n_bodies, 2).

    Horizontally centred on the root each frame, so the humanoid stays in view; vertically
    absolute, so the floor is a fixed line and the bounce is honest.

    `heading` rotates each frame into the humanoid's OWN frame first, which is what makes the
    view names mean anything. Without it the cameras sit at fixed world azimuths, so "front"
    means "camera at world +x" rather than "in front of the humanoid". A policy that wanders
    (and this one drifted 47 degrees in 11 seconds) turns under the fixed cameras until the
    labels are simply wrong, and past 90 degrees of drift left and right appear swapped.
    Passing None reproduces the old world-fixed behaviour and is only useful for debugging.
    """
    centred = positions.copy()
    # Body 0 of the list is the root (pelvis), the first body in the tree.
    centred[:, :, :2] -= centred[:, :1, :2]

    if heading is not None:
        # Rotate by -heading so the humanoid always faces the same way relative to the
        # cameras, the way a treadmill gait analysis is presented.
        c, s = np.cos(-heading), np.sin(-heading)
        x = centred[:, :, 0].copy()
        y = centred[:, :, 1].copy()
        centred[:, :, 0] = c[:, None] * x - s[:, None] * y
        centred[:, :, 1] = s[:, None] * x + c[:, None] * y

    out = np.zeros((len(views), positions.shape[0], positions.shape[1], 2))
    for v, (_, azimuth) in enumerate(views):
        right, up = _basis(azimuth)
        out[v, :, :, 0] = centred @ right
        out[v, :, :, 1] = centred @ up
    return out, [name for name, _ in views]


def capture(
    env,
    policy,
    device,
    *,
    seconds: float = 3.0,
    command=(1.0, 0.0, 0.0),
    settle: float = 1.0,
    max_frames: int = 60,
    views=DEFAULT_VIEWS,
) -> SkeletonCapture:
    """Run a short deterministic rollout and project the body positions.

    Args:
        seconds: Simulated seconds to keep. Three is two to three full strides.
        settle: Seconds discarded first, so the capture shows the steady gait rather than
            the transient from the reset pose.
        max_frames: Frames kept, evenly spaced. 60 over 3 s is 20 fps, which is enough to
            read a gait and keeps the payload around 30 KB.
    """
    import torch

    model = env.model
    data = env.datas[0]
    names, bones = body_tree(model)

    dt = env.dt
    settle_steps = int(settle / dt)
    keep_steps = int(seconds / dt)
    stride = max(1, keep_steps // max_frames)

    frames: list[np.ndarray] = []
    headings: list[float] = []
    was_training = getattr(policy, "training", False)
    if hasattr(policy, "eval"):
        policy.eval()

    env.reset()
    with torch.no_grad():
        for step in range(settle_steps + keep_steps):
            env.state.task_state["command"][0] = np.asarray(command, dtype=np.float64)
            env._compute_obs()  # noqa: SLF001 - refresh the command in the observation
            action = policy.act_deterministic(
                torch.from_numpy(env._obs.copy()).to(device)  # noqa: SLF001
            ).cpu().numpy()
            env.step(action)
            if step >= settle_steps and (step - settle_steps) % stride == 0:
                frames.append(data.xpos[1:].copy())
                headings.append(float(env.state.heading[0]))

    if was_training and hasattr(policy, "train"):
        policy.train()

    positions = np.stack(frames) if frames else np.zeros((1, len(names), 3))
    heading = np.asarray(headings) if headings else np.zeros(len(positions))
    projected, view_names = project(positions, heading, views)
    trails = [names.index(n) for n in TRAIL_BODIES if n in names]

    return SkeletonCapture(
        bodies=names,
        bones=bones,
        views=projected,
        view_names=view_names,
        fps=1.0 / (dt * stride),
        trails=trails,
        meta={
            "command": list(command),
            "seconds": len(frames) * dt * stride,
            "ground_z": 0.0,
            # Views are relative to the humanoid's own facing, not to world axes.
            "heading_relative": True,
            "mean_heading_deg": float(np.degrees(np.mean(heading))) if headings else 0.0,
        },
    )


def write(capture_result: SkeletonCapture, path: str | Path, **extra) -> Path:
    """Write one capture as JSON, with whatever run metadata the caller wants attached."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = capture_result.to_json()
    payload["meta"].update(extra)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
