"""Offscreen episode rendering to mp4.

This is the engine behind the dashboard's Videos mode. It runs a deterministic episode with
a tracking camera and writes an h264 mp4, with a telemetry overlay so a video is diagnostic
rather than merely decorative.

Three constraints shaped it, all of them recorded in DESIGN.md section 10:

* **The offscreen framebuffer has a hard size cap** set by `<visual><global offwidth
  offheight>` in the model XML, defaulting to 640x480. Rendering larger silently fails or
  clips. `humanoid_scene.xml` raises it to 1280x960, and any generated terrain scene must
  inherit that.
* **A CGL context is current on one thread at a time.** Rendering therefore happens on the
  calling thread, between vectorised steps, while the physics workers are parked on their
  barrier. Never call this concurrently with `env.step()`.
* **Video encoding uses `h264_videotoolbox`** where available, which runs on Apple's
  dedicated media engine rather than the GPU or CPU, so encoding does not steal cycles from
  a training run happening at the same time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from humanoid_rl.algos.networks import ActorCritic
from humanoid_rl.envs.vec_env import ThreadedVecEnv


@dataclass
class CameraConfig:
    """A camera that follows the humanoid.

    Defaults to a three-quarter side view. A pure side view reads foot placement and arm
    swing best, but loses turning entirely; 115 degrees keeps the gait legible while still
    showing direction changes.
    """

    distance: float = 3.6
    elevation: float = -10.0
    azimuth: float = 115.0
    height_offset: float = 0.35
    #: Low-pass factor for the camera target. The pelvis oscillates every step, and a
    #: rigidly attached camera transfers that bounce to the whole frame and reads as a
    #: much worse gait than it is.
    smoothing: float = 0.12


@dataclass
class CommandSegment:
    """A commanded velocity held for a number of seconds, with a caption for the overlay."""

    seconds: float
    command: tuple[float, float, float]  # vx (m/s), vy (m/s), yaw rate (rad/s)
    label: str


#: A scripted sequence that exercises everything Phase 2 is meant to learn. Far more
#: informative than a random command, which usually shows one behaviour for the whole clip.
DEFAULT_SCHEDULE: list[CommandSegment] = [
    CommandSegment(3.0, (0.0, 0.0, 0.0), "stand still"),
    CommandSegment(5.0, (1.0, 0.0, 0.0), "walk forward 1.0 m/s"),
    CommandSegment(4.0, (0.8, 0.0, 0.8), "turn left while walking"),
    CommandSegment(4.0, (0.8, 0.0, -0.8), "turn right while walking"),
    CommandSegment(3.0, (1.5, 0.0, 0.0), "walk fast 1.5 m/s"),
    CommandSegment(2.5, (0.0, 0.5, 0.0), "sidestep"),
    CommandSegment(2.5, (-0.4, 0.0, 0.0), "walk backward"),
    CommandSegment(3.0, (0.0, 0.0, 0.0), "stop"),
]


#: A compressed schedule for videos rendered automatically during training. Rendering is
#: not free (about 1.3x realtime), so the in-training clip trades coverage for cost: it
#: keeps the walk, both turns and the stop, and drops sidestep, backward and fast.
SHORT_SCHEDULE: list[CommandSegment] = [
    CommandSegment(1.5, (0.0, 0.0, 0.0), "stand"),
    CommandSegment(4.0, (1.0, 0.0, 0.0), "walk forward"),
    CommandSegment(3.0, (0.8, 0.0, 0.8), "turn left"),
    CommandSegment(3.0, (0.8, 0.0, -0.8), "turn right"),
    CommandSegment(1.5, (0.0, 0.0, 0.0), "stop"),
]


@dataclass
class RenderResult:
    path: Path
    frames: int
    seconds: float
    fell: bool
    fell_at: float | None
    mean_speed: float
    mean_slip: float
    metadata: dict = field(default_factory=dict)


def _load_font(size: int):
    """A monospace font if one can be found, else PIL's bitmap default."""
    from PIL import ImageFont

    for candidate in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _overlay(frame: np.ndarray, lines: list[str], font) -> np.ndarray:
    """Draw a telemetry panel onto a frame."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    pad, line_h = 10, font.size + 4 if hasattr(font, "size") else 16
    width = max(draw.textlength(line, font=font) for line in lines) + 2 * pad
    draw.rectangle([8, 8, 8 + width, 8 + pad * 2 + line_h * len(lines)], fill=(0, 0, 0, 150))
    for i, line in enumerate(lines):
        draw.text((8 + pad, 8 + pad + i * line_h), line, font=font, fill=(235, 240, 245, 255))
    return np.asarray(image)


@torch.no_grad()
def render_episode(
    env: ThreadedVecEnv,
    policy: ActorCritic,
    device: torch.device,
    out_path: str | Path,
    *,
    schedule: list[CommandSegment] | None = None,
    width: int = 960,
    height: int = 720,
    fps: int = 50,
    camera: CameraConfig | None = None,
    overlay: bool = True,
    stop_on_fall: bool = False,
) -> RenderResult:
    """Run one deterministic episode and write it to `out_path` as mp4.

    Args:
        env: A single-environment `ThreadedVecEnv`. Must not be a training environment.
        policy: Actor-critic. Only its mean action is used, so there is no exploration
            noise and the video shows what the policy actually believes is best.
        schedule: Commanded velocity segments. Defaults to DEFAULT_SCHEDULE.
        stop_on_fall: If False (the default) the episode continues after a fall, which is
            deliberate: watching *how* it recovers or fails is the most useful thing in an
            early-training video.
    """
    if env.num_envs != 1:
        raise ValueError(f"render_episode expects a 1-environment env, got {env.num_envs}")

    schedule = schedule or DEFAULT_SCHEDULE
    camera = camera or CameraConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = policy.training
    policy.eval()

    obs = env.reset()
    data = env.datas[0]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = camera.distance
    cam.elevation = camera.elevation
    cam.azimuth = camera.azimuth
    cam_target = np.array([data.qpos[0], data.qpos[1], camera.height_offset])

    font = _load_font(17) if overlay else None
    frames: list[np.ndarray] = []
    speeds: list[float] = []
    slips: list[float] = []
    fell_at: float | None = None

    # Expand the schedule into a per-control-step command track.
    steps_per_second = int(round(1.0 / env.dt))

    # Physics runs at 125 Hz here while video is encoded at `fps`. Capturing every control
    # step and then encoding at 50 gave 2.5x SLOW MOTION in every video this project has ever
    # produced, which quietly made every gait look more deliberate than it is. Capture every
    # `frame_skip`-th step instead and encode at the rate that actually implies, so playback
    # is real time. The encoded rate is reported rather than assumed to equal `fps`.
    frame_skip = max(1, round(steps_per_second / fps))
    encode_fps = steps_per_second / frame_skip
    track: list[tuple[np.ndarray, str]] = []
    for segment in schedule:
        for _ in range(int(segment.seconds * steps_per_second)):
            track.append((np.array(segment.command, dtype=np.float64), segment.label))

    with mujoco.Renderer(env.model, height=height, width=width) as renderer:
        for step, (command, label) in enumerate(track):
            # Override the sampled command so the video follows the script.
            # Tasks without a velocity command (get-up) have no such key. Writing it
            # unconditionally raised KeyError, and because visualisation failures are
            # swallowed so they cannot kill a long run, the get-up run produced ZERO
            # videos and zero flip-books for 2000 iterations while reporting nothing.
            if "command" in env.state.task_state:
                env.state.task_state["command"][0] = command
            env._compute_obs()  # noqa: SLF001 - refresh the command in the observation
            obs = env._obs.copy()  # noqa: SLF001

            action = policy.act_deterministic(torch.from_numpy(obs).to(device)).cpu().numpy()
            result = env.step(action)

            speed = float(np.linalg.norm(env.state.lin_vel_body[0, :2]))
            slip = float(
                (
                    np.linalg.norm(env.state.foot_lin_vel[0, :, :2], axis=1)
                    * env.state.foot_contact[0]
                ).sum()
            )
            speeds.append(speed)
            slips.append(slip)

            if bool(result.terminated[0]) and fell_at is None:
                fell_at = step * env.dt
                if stop_on_fall:
                    break

            # Smoothly follow the pelvis so per-step bounce does not shake the frame.
            target = np.array([data.qpos[0], data.qpos[1], camera.height_offset])
            cam_target += (target - cam_target) * camera.smoothing
            cam.lookat[:] = cam_target

            # Metrics above are collected every control step; only the picture is decimated.
            if step % frame_skip:
                continue

            renderer.update_scene(data, cam)
            frame = renderer.render()

            if overlay:
                contact = "".join("#" if c else "." for c in env.state.foot_contact[0])
                frame = _overlay(
                    frame,
                    [
                        f"t {step * env.dt:5.1f}s   {label}",
                        f"cmd  vx {command[0]:+.2f}  vy {command[1]:+.2f}  yaw {command[2]:+.2f}",
                        f"act  speed {speed:4.2f} m/s   slip {slip:4.2f} m/s",
                        f"feet [{contact}]   height {float(data.qpos[2]):4.2f} m",
                    ],
                    font,
                )
            frames.append(frame)

    if was_training:
        policy.train()

    stacked = np.asarray(frames)
    _encode(stacked, out_path, encode_fps)

    result = RenderResult(
        path=out_path,
        frames=len(frames),
        # Simulated seconds, from control steps rather than frames: frames are decimated by
        # `frame_skip` for real-time playback, so counting them would undercount the episode.
        seconds=len(speeds) * env.dt,
        fell=fell_at is not None,
        fell_at=fell_at,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        mean_slip=float(np.mean(slips)) if slips else 0.0,
    )
    # Sidecar metadata, which the dashboard gallery reads and displays on each card.
    result.metadata = {
        "frames": result.frames,
        "seconds": round(result.seconds, 2),
        "fell": result.fell,
        "fell_at": result.fell_at,
        "mean_speed": round(result.mean_speed, 3),
        "mean_slip": round(result.mean_slip, 3),
    }
    out_path.with_suffix(".json").write_text(json.dumps(result.metadata, indent=2))
    return result


#: Bits per pixel per frame. 0.09 is comfortably transparent for this content, which is a
#: smooth-shaded model on a flat background with no film grain or fine texture.
_BITS_PER_PIXEL = 0.09


def _encode(frames: np.ndarray, out_path: Path, fps: int) -> None:
    """Write frames to mp4, preferring Apple's hardware encoder.

    `h264_videotoolbox` runs on the dedicated media engine, so encoding a video while a
    training run is in progress costs neither GPU nor CPU time that training needs.

    The bitrate is derived from the frame size rather than fixed. A fixed 6 Mbps produced a
    24.7 MB file for a 13 second 640x480 clip, which on a multi-day run with periodic videos
    would have quietly consumed gigabytes.
    """
    import imageio.v3 as iio

    _, height, width, _ = frames.shape
    bitrate = max(800_000, int(width * height * fps * _BITS_PER_PIXEL))

    try:
        iio.imwrite(
            out_path,
            frames,
            fps=fps,
            codec="h264_videotoolbox",
            output_params=["-b:v", str(bitrate), "-pix_fmt", "yuv420p", "-color_range", "tv"],
        )
        return
    except Exception:  # noqa: BLE001 - fall back to the software encoder
        pass
    iio.imwrite(out_path, frames, fps=fps, codec="libx264", quality=7)


def build_render_env(
    model_path: str | Path, task, *, seed: int = 0, max_episode_steps: int = 100_000
) -> ThreadedVecEnv:
    """A single-environment env suitable for rendering.

    Randomisation is off so the video shows nominal dynamics, and the episode limit is
    effectively removed so a scripted schedule is never cut short by a timeout.
    """
    from humanoid_rl.envs.domain_rand import DomainRandConfig

    return ThreadedVecEnv(
        model_path,
        task,
        num_envs=1,
        num_workers=1,
        max_episode_steps=max_episode_steps,
        seed=seed,
        domain_rand=DomainRandConfig(enabled=False),
    )
