"""Render a strip of frames from a saved clip file, so a person can look at it.

Any npz with `qpos` and `bounds` works: the reference film, the CEM search output, a bank.
The rule this project runs on is that nothing gets trained against until someone has looked
at it, and the reference that cost E42-E46 was one nobody had watched.

    python scripts/render_clip.py data/fallen/getup_refs_v2.npz
    python scripts/render_clip.py data/fallen/getup_refs_v2.npz --clip 1 --shots 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.envs.model_prep import prepare  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("/tmp/clip.png"))
    args = ap.parse_args()

    d = np.load(args.path, allow_pickle=False)
    qpos = d["qpos"]
    bounds = d["bounds"] if "bounds" in d else np.array([0, len(qpos)])
    a, b = int(bounds[args.clip]), int(bounds[args.clip + 1])
    frames_q = qpos[a:b]
    dt = float(d["dt"]) if "dt" in d else 0.008
    idx = np.linspace(0, len(frames_q) - 1, args.shots).astype(int)

    prepared = prepare(REPO_ROOT / "humanoid_rl/models/humanoid_scene.xml",
                       action_scale_mode="full_range")
    model = prepared.model
    data = mujoco.MjData(model)

    print(f"clip {args.clip} of {len(bounds) - 1}: {len(frames_q)} frames, "
          f"{len(frames_q) * dt:.2f} s")
    print(f"{'t':>7}{'pelvis':>9}{'head':>8}")
    imgs = []
    with mujoco.Renderer(model, height=420, width=340) as renderer:
        for i in idx:
            data.qpos[:] = frames_q[i]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.azimuth = 3.0, -10.0, 110.0
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.55]
            renderer.update_scene(data, camera=cam)
            imgs.append(renderer.render().copy())
            head = float(data.xpos[prepared.head_body_id, 2]) if hasattr(
                prepared, "head_body_id") else float("nan")
            print(f"{i * dt:>6.2f}s{float(frames_q[i][2]):>9.3f}{head:>8.3f}")

    sheet = Image.new("RGB", (sum(im.shape[1] for im in imgs), imgs[0].shape[0]),
                      (255, 255, 255))
    x = 0
    for im in imgs:
        sheet.paste(Image.fromarray(im), (x, 0))
        x += im.shape[1]
    sheet.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
