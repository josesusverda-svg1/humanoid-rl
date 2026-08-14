"""Drive the trained humanoid yourself, with the keyboard, in an interactive viewer.

    .venv/bin/mjpython scripts/play.py

Note `mjpython`, not `python`. On macOS MuJoCo's interactive viewer has to own the main
thread, so `mujoco.viewer.launch_passive` raises outright under a normal interpreter:

    RuntimeError: `launch_passive` requires that the Python script be run under
    `mjpython` on macOS

`mjpython` ships with the mujoco wheel and is already in this project's venv.

WHY THIS WORKS WITHOUT RETRAINING
The policy is velocity-command conditioned: part of every observation is a target
(forward, sideways, turn rate) that training resampled at random each episode. This script
writes your keypresses into that same slot. Nothing about the policy changes, and nothing
about the physics changes. You are supplying the command that the training loop used to
supply, which is precisely the interface the policy was built around.

WHY THE CONTROLS ARE A JOYSTICK, NOT HOLD-TO-WALK
MuJoCo's viewer hands Python a single keycode per event, with no press/release flag and no
guarantee of auto-repeat while a key is held (`key_callback: Callable[[int], None]`). Rather
than depend on repeat behaviour that varies, each press *adjusts a persistent command*, like
nudging a throttle, and the humanoid keeps doing that until told otherwise. For inspecting a
locomotion policy this is better than twitch control anyway: you can set "walk forward at
0.75 m/s while turning left" and then just watch the gait.

The command is clamped to the range the policy was actually trained on. Outside it the policy
is extrapolating and the gait degrades, so the clamps are a feature, not a limitation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from humanoid_rl.algos.networks import ActorCritic  # noqa: E402
from humanoid_rl.config import Config, resolve_device  # noqa: E402
from humanoid_rl.render import build_render_env  # noqa: E402
from humanoid_rl.tasks.locomotion import LocomotionConfig, LocomotionTask  # noqa: E402

# GLFW key codes. Deliberately avoiding SPACE, BACKSPACE and the function keys, which
# MuJoCo's own viewer UI binds to pause, reset and rendering toggles.
KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E = 87, 65, 83, 68, 81, 69
KEY_X, KEY_Z, KEY_C = 88, 90, 67
KEY_1, KEY_2, KEY_3 = 49, 50, 51

#: How much one keypress moves each axis. Sized so a few taps span the trained range.
STEP_FORWARD = 0.25  # m/s
STEP_LATERAL = 0.20  # m/s
STEP_TURN = 0.25  # rad/s


class Controller:
    """The persistent velocity command, and the keyboard that edits it.

    Held as an object rather than globals because `key_callback` fires on the viewer's UI
    thread while the physics loop reads it on ours. A single small mutable object with plain
    float writes is the least error-prone way to share that: each field is written whole, so
    a torn read is impossible, and a stale read costs at most one control step (8 ms).
    """

    def __init__(self, limits: LocomotionConfig, tau: float = 0.30) -> None:
        # What the keys set.
        self.forward = 0.0
        self.lateral = 0.0
        self.turn = 0.0
        # What the policy is actually given, which chases the above.
        self._actual = np.zeros(3)
        self.tau = tau
        self.limits = limits
        self.reset_requested = False
        self.follow_camera = True

    def command(self) -> np.ndarray:
        """The smoothed command handed to the policy."""
        return self._actual.copy()

    def advance(self, dt: float) -> None:
        """Ease the live command toward the keyed target by one control step.

        A keypress moves the target instantly, but feeding the policy a step change would
        put it somewhere training never took it: the command is resampled only at episode
        reset, so every command the policy has ever seen was constant for a whole episode.
        A ~0.3 s first-order ramp keeps each instant close to that constant-command regime,
        and it also simply feels better than being kicked from a standstill into a jog.
        """
        target = np.array([self.forward, self.lateral, self.turn])
        self._actual += (target - self._actual) * (1.0 - np.exp(-dt / self.tau))

    def stop(self) -> None:
        self.forward = self.lateral = self.turn = 0.0

    def snap(self) -> None:
        """Drop the ramp and apply the target immediately. Used on reset and in tests."""
        self._actual = np.array([self.forward, self.lateral, self.turn])

    def on_key(self, keycode: int) -> None:
        cfg = self.limits
        if keycode == KEY_W:
            self.forward += STEP_FORWARD
        elif keycode == KEY_S:
            self.forward -= STEP_FORWARD
        elif keycode == KEY_A:
            self.turn += STEP_TURN  # positive yaw is to the left
        elif keycode == KEY_D:
            self.turn -= STEP_TURN
        elif keycode == KEY_Q:
            self.lateral += STEP_LATERAL  # strafe left, no turn
        elif keycode == KEY_E:
            self.lateral -= STEP_LATERAL
        elif keycode == KEY_X:
            self.stop()
        elif keycode == KEY_Z:
            self.reset_requested = True
        elif keycode == KEY_C:
            self.follow_camera = not self.follow_camera
        elif keycode == KEY_1:
            self.stop()
            self.forward = 0.5  # amble
        elif keycode == KEY_2:
            self.stop()
            self.forward = 1.0  # walk
        elif keycode == KEY_3:
            self.stop()
            self.forward = 1.5  # top of the trained range
        else:
            return

        # Clamp to what the policy was actually trained on. Beyond these it extrapolates.
        self.forward = float(np.clip(self.forward, *cfg.lin_vel_x_range))
        self.lateral = float(np.clip(self.lateral, *cfg.lin_vel_y_range))
        self.turn = float(np.clip(self.turn, *cfg.ang_vel_yaw_range))


HELP = """
  W / S    forward speed      -/+ {sf:.2f} m/s        (trained range {fx[0]:+.1f} to {fx[1]:+.1f})
  A / D    turn left / right  -/+ {st:.2f} rad/s      (trained range {yz[0]:+.1f} to {yz[1]:+.1f})
  Q / E    strafe left/right  -/+ {sl:.2f} m/s        (trained range {ly[0]:+.1f} to {ly[1]:+.1f})
  1 / 2 / 3   preset walk at 0.5 / 1.0 / 1.5 m/s
  X        stop (zero the command)
  Z        reset the humanoid to standing
  C        toggle camera follow
  Esc      quit

  Each press nudges a persistent command; it holds until you change it.
"""


def check_plumbing(env, control: Controller) -> int:
    """Verify each key reaches the policy's input with the right sign. No policy involved.

    This is the part that can actually be *wrong* in this script, and it is worth testing on
    its own, because the obvious test (press a key and see if the humanoid moves) conflates
    two unrelated things. An early checkpoint that cannot turn yet looks exactly like a
    reversed key, and reading a policy's weakness as a wiring bug sends you editing correct
    code. Here the policy is absent: the check is that pressing A puts a positive turn rate
    into the observation, and nothing more.

    The command occupies the last three observation values, unscaled, which the assertion
    below re-derives rather than trusting.
    """
    print(f"{'key':<8}{'axis':<10}{'expected':>10}{'in observation':>16}   verdict")
    failures = 0
    checks = [
        (KEY_W, "forward", 0, +1), (KEY_S, "forward", 0, -1),
        (KEY_A, "turn", 2, +1), (KEY_D, "turn", 2, -1),
        (KEY_Q, "lateral", 1, +1), (KEY_E, "lateral", 1, -1),
    ]
    for key, axis, index, sign in checks:
        control.stop()
        control.on_key(key)
        control.snap()
        env.state.task_state["command"][0] = control.command()
        env._compute_obs()  # noqa: SLF001
        value = float(env._obs[0, -3:][index])  # noqa: SLF001
        ok = np.sign(value) == sign and abs(value) > 1e-6
        failures += not ok
        print(f"{chr(key):<8}{axis:<10}{'positive' if sign > 0 else 'negative':>10}"
              f"{value:>16.3f}   {'ok' if ok else 'WIRED WRONG'}")
    control.stop()
    return failures


@torch.no_grad()
def selftest(env, policy, control: Controller, device: torch.device) -> int:
    """Check the controls are wired correctly, then report what the policy does with them.

    Two separate questions, reported separately, because they have different owners and
    different fixes. The first is about this script and must pass. The second is about the
    checkpoint and is informational.
    """
    print("\n1. PLUMBING: does each key reach the policy's input with the right sign?\n")
    wiring_failures = check_plumbing(env, control)
    print()
    print("2. RESPONSE: what does this checkpoint actually do when asked?")
    print("   Not a test of this script. An undertrained policy is weak on every row.\n")
    cases = [
        ("W W W    walk forward", [KEY_W, KEY_W, KEY_W], "forward"),
        ("S S      walk backward", [KEY_S, KEY_S], "forward"),
        ("A A      turn left", [KEY_A, KEY_A], "turn"),
        ("D D      turn right", [KEY_D, KEY_D], "turn"),
        ("Q Q      strafe left", [KEY_Q, KEY_Q], "lateral"),
        ("E E      strafe right", [KEY_E, KEY_E], "lateral"),
        ("2        preset walk", [KEY_2], "forward"),
        ("X        stop", [KEY_X], "forward"),
    ]
    settle, measure = 150, 250
    print(f"{'keys':<26}{'commanded':>11}{'measured':>10}   verdict")

    failures = 0
    results: dict[str, list[float]] = {}
    for label, keys, axis in cases:
        # Each case starts from a clean standing humanoid. Without this the previous case's
        # momentum and any fall it ended in leak into the next measurement, which made an
        # earlier version of this test report two failures that were entirely its own doing.
        env.reset()
        control.stop()
        for k in keys:
            control.on_key(k)
        control.snap()
        commanded = {"forward": control.forward, "lateral": control.lateral,
                     "turn": control.turn}[axis]

        samples = []
        for i in range(settle + measure):
            env.state.task_state["command"][0] = control.command()
            env._compute_obs()  # noqa: SLF001
            action = policy.act_deterministic(
                torch.from_numpy(env._obs.copy()).to(device)  # noqa: SLF001
            ).cpu().numpy()
            env.step(action)
            if i >= settle:
                s = env.state
                samples.append(
                    s.ang_vel_body[0, 2] if axis == "turn"
                    else s.lin_vel_body[0, 0 if axis == "forward" else 1]
                )
        measured = float(np.mean(samples))

        # Describes the policy, never this script. Wiring was settled above.
        if abs(commanded) < 1e-6:
            verdict = "ok" if abs(measured) < 0.25 else "drifts"
        elif np.sign(measured) == np.sign(commanded) and abs(measured) > 0.05:
            verdict = "ok"
        elif abs(measured) <= 0.05:
            verdict = "no response"
        else:
            verdict = "goes the wrong way"
        results[axis] = results.get(axis, []) + [measured]
        print(f"{label:<26}{commanded:>11.2f}{measured:>10.2f}   {verdict}")

    print()
    if wiring_failures:
        print(f"{wiring_failures} key(s) are WIRED WRONG in this script. Fix before use.")
    else:
        print("Controls verified correct. Everything above is this checkpoint's ability.")

    # One inference the response table supports that no single row does. If the two turn
    # directions come out with the SAME sign, the humanoid veers that way no matter what it
    # is asked, which is the signature of a one-sided gait, and is a training problem rather
    # than a control problem. Worth stating outright, because it is easy to misread a single
    # reversed row as a bug.
    turns = results.get("turn", [])
    if len(turns) == 2 and np.sign(turns[0]) == np.sign(turns[1]) and min(map(abs, turns)) > 0.02:
        side = "right" if turns[0] < 0 else "left"
        print(f"\nNote: both turn commands produced {side}ward rotation "
              f"({turns[0]:+.2f} and {turns[1]:+.2f} rad/s). This policy veers {side} "
              f"regardless of the command, which is what a one-sided gait looks like from "
              f"the controls. Not a wiring fault: see README lessons 6-8.")
    env.close()
    return 1 if wiring_failures else 0


def newest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()]
    if not candidates:
        raise FileNotFoundError(f"no runs with a config.yaml in {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: newest)")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skin", action="store_true",
                    help="render the human skin instead of the collision capsules "
                         "(build it first with scripts/build_skin.py)")
    ap.add_argument(
        "--speed", type=float, default=1.0, help="playback rate, 0.5 for half speed"
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="drive a scripted key sequence with no window, and report whether the humanoid "
        "actually responded. Runs under plain python, so the control path can be verified "
        "without a GUI.",
    )
    args = ap.parse_args()

    run_dir = args.run or newest_run(REPO_ROOT / "runs")
    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        available = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
        raise FileNotFoundError(f"{ckpt_path} not found. Available: {available}")

    config = Config.load(run_dir / "config.yaml")

    # The skinned model is the same physics with a render-only skin attached, verified
    # bitwise identical by `build_skin.py --verify`, so swapping it in cannot change what
    # the policy does. Only its appearance changes.
    model_path = REPO_ROOT / config.env.model_path
    if args.skin:
        skinned = REPO_ROOT / "assets" / "humanoid_skinned.xml"
        if not skinned.exists():
            raise FileNotFoundError(
                f"{skinned} not found. Build it with:\n"
                "    .venv/bin/python scripts/build_skin.py --verify"
            )
        model_path = skinned
    device = torch.device(resolve_device(config.run.device))

    # The same env the offscreen renderer uses: one environment, randomisation off, no
    # episode time limit. Critically this also means the observation is assembled by exactly
    # the same code path as in training, so the policy sees what it expects.
    task = LocomotionTask(config.task)
    env = build_render_env(model_path, task, seed=args.seed)

    policy = ActorCritic(
        env.obs_dim,
        env.nu,
        actor_hidden=tuple(config.network.actor_hidden),
        critic_hidden=tuple(config.network.critic_hidden),
        activation=config.network.activation,
        init_noise_std=config.network.init_noise_std,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(state["policy"])
    policy.eval()

    control = Controller(config.task)
    print(f"run        {run_dir.name}")
    print(f"checkpoint {args.checkpoint}  (iteration {state['iteration']:,}, "
          f"{state['env_steps']:,} env steps)")
    print(HELP.format(
        sf=STEP_FORWARD, sl=STEP_LATERAL, st=STEP_TURN,
        fx=config.task.lin_vel_x_range,
        ly=config.task.lin_vel_y_range,
        yz=config.task.ang_vel_yaw_range,
    ))

    env.reset()
    data = env.datas[0]
    dt = env.dt

    if args.selftest:
        return selftest(env, policy, control, device)

    try:
        viewer = mujoco.viewer.launch_passive(
            env.model, data, key_callback=control.on_key, show_left_ui=False
        )
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            print("\nThis needs mjpython, not python. Run:\n")
            print("    .venv/bin/mjpython scripts/play.py\n")
            return 1
        raise

    with viewer:
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0

        next_frame = time.perf_counter()
        step = 0
        with torch.no_grad():
            while viewer.is_running():
                if control.reset_requested:
                    control.reset_requested = False
                    control.stop()
                    control.snap()
                    env.reset()

                control.advance(dt)

                # Write the keyboard command into the slot training used to randomise, then
                # rebuild the observation so the policy sees the change on this very step.
                env.state.task_state["command"][0] = control.command()
                env._compute_obs()  # noqa: SLF001
                obs = env._obs.copy()  # noqa: SLF001

                action = policy.act_deterministic(
                    torch.from_numpy(obs).to(device)
                ).cpu().numpy()
                env.step(action)

                if control.follow_camera:
                    viewer.cam.lookat[:] = (data.qpos[0], data.qpos[1], 0.8)
                viewer.sync()

                # Pace to wall clock. Sleeping on a running deadline rather than a fixed
                # interval keeps the average rate correct even when a step runs long.
                step += 1
                if step % 6 == 0:
                    speed = float(np.linalg.norm(env.state.lin_vel_body[0, :2]))
                    feet = "".join("#" if c else "." for c in env.state.foot_contact[0])
                    live = control.command()
                    viewer.set_texts((
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "target fwd\ntarget turn\ntarget side\n\nactual speed\nfeet\nheight",
                        f"{control.forward:+.2f} m/s  (now {live[0]:+.2f})\n"
                        f"{control.turn:+.2f} rad/s (now {live[2]:+.2f})\n"
                        f"{control.lateral:+.2f} m/s  (now {live[1]:+.2f})\n\n"
                        f"{speed:.2f} m/s\n[{feet}]\n{float(data.qpos[2]):.2f} m",
                    ))

                next_frame += dt / max(args.speed, 1e-3)
                remaining = next_frame - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_frame = time.perf_counter()

    print()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
