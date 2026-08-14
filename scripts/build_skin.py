"""Wrap the humanoid in a human-looking skin, and prove the physics did not change.

    .venv/bin/python scripts/build_skin.py --verify --render /tmp/skin.png

The skin is render-only: MuJoCo's own documentation says skins "do not affect the physics in
any way", and `--verify` checks that here rather than taking its word for it, by comparing
masses, inertias and a 1000-step trajectory with and without the skin attached.

That check is not ceremony. The obvious cheaper approach, hanging visual meshes off each
body, silently changes the model: `contype="0" conaffinity="0"` stops a geom colliding but
does NOT remove its inertial contribution, so a purely cosmetic edit can move a body's mass
by several kilograms and invalidate a trained policy. Skin has no such trap, and `--verify`
is how we know.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

from humanoid_rl.config import Config  # noqa: E402
from humanoid_rl.viz import makehuman, skin as skinmod  # noqa: E402

DEFAULT_ASSETS = REPO_ROOT / "third_party" / "makehuman"


def build_spec(model_path: Path, assets: Path, *, hide_capsules: bool, inflate: float):
    """Compile a model with the skin attached. Returns (spec, model, fitted)."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    plain = spec.compile()

    # Fit in the pose the humanoid actually stands in, not qpos0, which has it waist-deep
    # in the floor. `prepare` is the same function the environment uses, so the bind pose is
    # exactly the pose training and rendering start from.
    from humanoid_rl.envs.model_prep import prepare

    body = makehuman.load(assets)
    fitted = skinmod.fit(plain, body, prepare(model_path).default_qpos)
    skinmod.attach(spec, fitted, inflate=inflate)

    if hide_capsules:
        # Groups 0-2 are visible by default and 3-5 are hidden, so moving the collision
        # capsules to group 3 shows only the skin. The floor stays put, and nothing about
        # collision changes: `geom_group` is a visualisation field.
        for geom in spec.geoms:
            if geom.name != "floor":
                geom.group = 3

    return spec, spec.compile(), fitted


def verify(model_path: Path, skinned: mujoco.MjModel) -> int:
    """Assert the skin is inert, by measurement rather than by documentation."""
    plain = mujoco.MjModel.from_xml_path(str(model_path))
    print("\nPHYSICS UNCHANGED?  plain model vs skinned model\n")
    checks: list[tuple[str, bool, str]] = []

    checks.append(("nq / nv / nbody / ngeom",
                   (plain.nq, plain.nv, plain.nbody, plain.ngeom)
                   == (skinned.nq, skinned.nv, skinned.nbody, skinned.ngeom),
                   f"{plain.nq}/{plain.nv}/{plain.nbody}/{plain.ngeom} vs "
                   f"{skinned.nq}/{skinned.nv}/{skinned.nbody}/{skinned.ngeom}"))
    for field in ("body_mass", "body_inertia", "body_ipos", "geom_contype", "geom_conaffinity"):
        a, b = getattr(plain, field), getattr(skinned, field)
        same = a.shape == b.shape and np.array_equal(a, b)
        checks.append((field, same, "identical" if same else f"DIFFERS, max {np.abs(a-b).max()}"))

    # The real test: does it simulate identically? Same seed, same controls, 1000 steps.
    rng = np.random.default_rng(0)
    controls = rng.uniform(-0.3, 0.3, size=(1000, plain.nu))
    trajectories = []
    for m in (plain, skinned):
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        for u in controls:
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
        trajectories.append(d.qpos.copy())
    drift = float(np.abs(trajectories[0] - trajectories[1]).max())
    checks.append(("1000-step qpos trajectory", drift == 0.0,
                   "bitwise identical" if drift == 0.0 else f"DIVERGED by {drift:.3e}"))

    failed = 0
    for name, ok, detail in checks:
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} {detail}")
    print()
    print("Skin is inert: the trained policy and every measured metric remain valid."
          if not failed else f"{failed} check(s) FAILED. Do not use this skin.")
    return failed


def render(model: mujoco.MjModel, model_path: Path, out: Path, width: int, height: int) -> None:
    """A single offscreen frame, which also proves the skin renders in the video pipeline."""
    import imageio.v3 as iio

    from humanoid_rl.envs.model_prep import prepare

    data = mujoco.MjData(model)
    data.qpos[:] = prepare(model_path).default_qpos
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 3.0, -8.0, 135.0
    cam.lookat[:] = (0.0, 0.0, 0.9)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, cam)
        frame = renderer.render()
        n_skin = int(renderer.scene.nskin)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, frame)
    print(f"\noffscreen render: scene.nskin = {n_skin} (skin reached the renderer)")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "amp.yaml")
    ap.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "assets" / "humanoid_skinned.xml")
    ap.add_argument("--render", type=Path, default=None, help="write a preview png")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=940)  # offscreen framebuffer caps at 960
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--keep-capsules", action="store_true",
                    help="leave the collision capsules visible, to check the fit")
    ap.add_argument("--inflate", type=float, default=0.0,
                    help="push vertices out along their normals, in metres")
    args = ap.parse_args()

    model_path = REPO_ROOT / Config.load(args.config).env.model_path
    spec, model, fitted = build_spec(
        model_path, args.assets, hide_capsules=not args.keep_capsules, inflate=args.inflate
    )

    print(f"source model : {model_path.relative_to(REPO_ROOT)}")
    print(f"skin assets  : {args.assets.relative_to(REPO_ROOT)}  (CC0)")
    print()
    for key, value in fitted.report.items():
        print(f"  {key:<28}{value:,.4g}")
    print(f"\ncompiled model: nskin={model.nskin}  nskinvert={model.nskinvert:,}  "
          f"nskinbonevert={model.nskinbonevert:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(spec.to_xml())
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")

    failed = verify(model_path, model) if args.verify else 0
    if args.render:
        render(model, model_path, args.render, args.width, args.height)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
