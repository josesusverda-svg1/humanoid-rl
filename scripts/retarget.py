"""Retarget mocap onto the humanoid and render a playback video to check it.

    .venv/bin/python scripts/retarget.py --inspect data/mocap/100STYLE     # skeleton only
    .venv/bin/python scripts/retarget.py --bvh <file.bvh> --render
    .venv/bin/python scripts/retarget.py --glob 'data/mocap/**/*BR.bvh' --limit 20

Stage 0 of the Phase 3 build order. The point is not to train anything, it is to *look at*
the reference motion on our own model before any discriminator ever sees it. A retarget that
floats, skates or folds a knee backwards will silently teach a policy to do the same, and
the resulting failure is very hard to diagnose from training curves.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from humanoid_rl.envs.model_prep import prepare  # noqa: E402
from humanoid_rl.motion import bvh  # noqa: E402
from humanoid_rl.motion.retarget import MotionClip, RetargetConfig, Retargeter  # noqa: E402  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "humanoid_rl" / "models" / "humanoid_scene.xml"


def inspect_dataset(root: Path, limit: int = 3) -> None:
    """Print the skeleton of the first few BVH files, to design the marker map."""
    files = sorted(root.rglob("*.bvh"))
    print(f"{len(files)} bvh files under {root}")
    for path in files[:limit]:
        try:
            motion = bvh.load(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.name}: FAILED {type(exc).__name__}: {exc}")
            continue
        print(f"\n  {path.relative_to(root)}")
        print(f"    frames={motion.n_frames} fps={motion.fps:.1f} duration={motion.duration:.1f}s")
        print(f"    {len(motion.joints)} joints:")
        for j in motion.joints:
            indent = "      " + "  " * _depth(motion, j)
            print(f"{indent}{j.name}{' (end)' if j.is_end_site else ''}")


def _depth(motion: bvh.BvhMotion, joint: bvh.BvhJoint) -> int:
    depth, cur = 0, joint
    while cur.parent >= 0:
        cur = motion.joints[cur.parent]
        depth += 1
    return depth


def render_clip(model: mujoco.MjModel, clip: MotionClip, out: Path, width=800, height=600) -> None:
    """Play the retargeted clip kinematically and write an mp4.

    Kinematic playback means the joint angles are set directly each frame with no physics.
    That is deliberate: it isolates retargeting quality from everything else. If the clip
    looks wrong here, no amount of controller tuning will save it.
    """
    import imageio.v3 as iio

    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 3.4, -8, 115
    target = np.array([clip.qpos[0, 0], clip.qpos[0, 1], 0.35])

    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for f in range(clip.n_frames):
            data.qpos[:] = clip.qpos[f]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            target += (np.array([data.qpos[0], data.qpos[1], 0.35]) - target) * 0.12
            cam.lookat[:] = target
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    arr = np.asarray(frames)
    bitrate = max(800_000, int(width * height * clip.fps * 0.09))
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(out, arr, fps=int(clip.fps), codec="h264_videotoolbox",
                    output_params=["-b:v", str(bitrate), "-pix_fmt", "yuv420p"])
    except Exception:  # noqa: BLE001
        iio.imwrite(out, arr, fps=int(clip.fps), codec="libx264", quality=7)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", type=Path, help="print skeletons from a dataset directory")
    ap.add_argument("--bvh", type=Path, help="retarget one file")
    ap.add_argument("--glob", type=str, help="retarget every file matching this glob")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "clips")
    ap.add_argument("--render", action="store_true", help="write a playback mp4")
    ap.add_argument("--max-seconds", type=float, default=8.0, help="trim long clips")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    if args.inspect:
        inspect_dataset(args.inspect, limit=args.limit)
        return 0

    paths: list[Path] = []
    if args.bvh:
        paths = [args.bvh]
    elif args.glob:
        paths = sorted(Path().glob(args.glob))[: args.limit]
    else:
        ap.error("give --inspect, --bvh or --glob")

    prepared = prepare(SCENE)
    retargeter = Retargeter(
        prepared.model, prepared.default_qpos, config=RetargetConfig(target_fps=args.fps)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(paths)} clip(s) -> {args.out_dir}")
    for path in paths:
        t0 = time.perf_counter()
        try:
            motion = bvh.load(path)
            if motion.duration > args.max_seconds:
                keep = int(args.max_seconds / motion.frame_time)
                motion.frames = motion.frames[:keep]
            clip = retargeter.retarget(motion, name=path.stem)
        except Exception as exc:  # noqa: BLE001
            print(f"  {path.name}: FAILED {type(exc).__name__}: {exc}")
            continue

        out = args.out_dir / f"{path.stem}.npz"
        clip.save(out)
        med = float(np.median(clip.residual))
        worst = float(np.max(clip.residual))
        flag = "ok " if med < 0.05 else "POOR"
        print(f"  [{flag}] {path.stem:<34s} {clip.n_frames:>4d} frames "
              f"{clip.duration:>5.1f}s  residual med {med * 100:>5.1f}cm max {worst * 100:>5.1f}cm "
              f"({time.perf_counter() - t0:.1f}s)")

        if args.render:
            video = args.out_dir / f"{path.stem}.mp4"
            render_clip(prepared.model, clip, video)
            print(f"         video -> {video}  ({video.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
