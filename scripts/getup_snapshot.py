"""Render a strip of what the current get-up policy is doing, for a person to look at.

Exists because every large defect in this project was found by a human watching, and none of
them by a metric. The headstand was visible instantly in a picture and only implied by an odd
pair of numbers. The one-armed prop was described by eye before any metric showed it.

Prints the numbers alongside each frame so the picture and the telemetry can be checked
against each other, which is how the headstand was confirmed: pelvis high, head low.

    python scripts/getup_snapshot.py                      # newest get-up run
    python scripts/getup_snapshot.py --run runs/getup-...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.envs.vec_env import ThreadedVecEnv  # noqa: E402
from humanoid_rl.tasks.getup import GetUpTask  # noqa: E402

#: Seconds into the episode to capture. Weighted early, because a get-up that works at all
#: happens in the first few seconds and the rest is whether it holds.
SHOTS = (0.0, 1.2, 2.4, 4.0, 8.0, 15.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("/tmp/getup_snapshot.png"))
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    run = args.run or sorted((REPO_ROOT / "runs").glob("getup-*"))[-1]
    cfg = Config.load(run / "config.yaml")
    device = torch.device(resolve_device(cfg.run.device))
    env = ThreadedVecEnv(
        REPO_ROOT / cfg.env.model_path, GetUpTask(cfg.getup), num_envs=1, num_workers=1,
        decimation=cfg.env.decimation, max_episode_steps=cfg.env.max_episode_steps,
        action_filter_hz=cfg.env.action_filter_hz, seed=args.seed,
        domain_rand=replace(cfg.domain_rand, enabled=False),
        action_scale_mode=cfg.env.action_scale_mode,
    )
    ckpt = torch.load(run / "checkpoints" / "best.pt", map_location=device,
                      weights_only=False)
    policy = ActorCritic(
        env.obs_dim, env.nu, actor_hidden=tuple(cfg.network.actor_hidden),
        critic_hidden=tuple(cfg.network.critic_hidden), activation=cfg.network.activation,
        init_noise_std=cfg.network.init_noise_std).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    model = env.prepared.model
    data = mujoco.MjData(model)
    steps = {int(t / env.dt): t for t in SHOTS}
    obs = env.reset()
    frames = []

    print(f"{run.name}  iteration {ckpt.get('iteration', 0):,}")
    print(f"{'t':>6}{'pelvis':>9}{'head':>8}{'feet BW':>9}{'knee':>7}{'hands':>8}")
    with torch.no_grad(), mujoco.Renderer(model, height=460, width=380) as renderer:
        for step in range(max(steps) + 1):
            if step in steps:
                data.qpos[:] = env.state.qpos[0]
                data.qvel[:] = env.state.qvel[0]
                mujoco.mj_forward(model, data)
                cam = mujoco.MjvCamera()
                cam.distance, cam.elevation, cam.azimuth = 3.0, -10.0, 110.0
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.55]
                renderer.update_scene(data, camera=cam)
                frames.append((steps[step], renderer.render().copy()))
                st = env.state
                load = float(st.foot_force[0, :2].sum()) / float(model.body_mass.sum() * 9.81)
                hands = min(float(st.key_body_pos[0, 2, 2]), float(st.key_body_pos[0, 3, 2]))
                print(f"{steps[step]:>5.1f}s{float(data.qpos[2]):>9.3f}"
                      f"{float(st.head_height_ratio[0]):>8.2f}{load:>9.2f}"
                      f"{float(np.max(st.qpos[0, env.prepared.actuator_qpos_adr])):>7.2f}"
                      f"{hands:>8.2f}")
            action = policy.act_deterministic(
                torch.as_tensor(obs, dtype=torch.float32, device=device))
            obs = env.step(action.cpu().numpy()).obs
    env.close()

    width = sum(f[1].shape[1] for f in frames)
    sheet = Image.new("RGB", (width, frames[0][1].shape[0]), (255, 255, 255))
    x = 0
    for _, img in frames:
        sheet.paste(Image.fromarray(img), (x, 0))
        x += img.shape[1]
    sheet.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
