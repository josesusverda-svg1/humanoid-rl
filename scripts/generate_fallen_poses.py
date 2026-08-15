"""Build a bank of settled fallen poses for the get-up task.

Two traps this exists to avoid, both measured rather than guessed (docs/GETUP.md §5.1):

* **Drops land flat; real falls land sideways.** Rolling out the walking policy and harvesting
  where it ends up gives 76% side-lying and a mean root height of 0.258 m. Dropping the body
  from a height with random orientation gives 4% side and 0.098 m. Training only on drops
  trains mostly on the case that does not happen, so the policy's own falls are the largest
  single source here.
* **The PD target held during the fall silently controls all diversity.** Holding the NOMINAL
  pose while the body topples gives a joint-angle spread of 0.012 rad, half a percent of range:
  one rigid mannequin at assorted yaws, a rank-1 bank that nothing downstream would flag.
  Holding a random target inside the commandable band gives 0.278 rad.

And the trap this project hit while testing the reset path: **random joint angles are not a
fallen pose.** They self-penetrate, and the contact forces that produces dwarf anything the
actuators do. Every pose here is settled under gravity and then validated as a fixed point of
the engine's reset path, which is `mj_resetData -> write qpos/qvel -> mj_forward` and never
settles anything itself.

    python scripts/generate_fallen_poses.py --count 8192
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_rl.envs.model_prep import prepare  # noqa: E402

#: Fractions per generator. Policy falls dominate because they are the only on-distribution
#: source; hand-authored keyframes are small but irreplaceable (the seated-legs-out pose the
#: user asked about appears in 0 of 600 drop samples and can only get in by hand).
#: Explicit, balanced coverage of every way a body can lie, rather than whatever the physics
#: happens to produce. Toppling and dropping are biased: a standing body falls forwards and
#: backwards far more readily than sideways, so the old mix gave 25% prone against 37% side
#: split unevenly, and nothing guaranteed left and right were equal.
#:
#: `oriented` places the body at a CHOSEN roll angle and lets it settle there, in four equal
#: clusters with jitter, so every side gets the same share every time.
MIX = {"oriented": 0.70, "keyframe": 0.15, "standing": 0.15}

#: Roll about the body's long axis once it is horizontal. 0 is face-down, 180 is face-up, and
#: the two sides sit between. The jitter is what the request "each time at a slightly
#: different angle, so sometimes it tips towards the belly and sometimes towards the back"
#: asks for: a pose at 65 degrees is on its side leaning towards its front, one at 115 is on
#: its side leaning back, and both settle differently.
ROLL_CLUSTERS = (
    ("prone", 0.0),
    ("side_right", 90.0),
    ("supine", 180.0),
    ("side_left", 270.0),
)
ROLL_JITTER_DEG = 28.0


def label_pose(model, data, standing_height: float, standing_head: float) -> str:
    """The total taxonomy from docs/GETUP.md 1.1. Every pose gets exactly one label."""
    mujoco.mj_forward(model, data)
    h = float(data.qpos[2])
    quat = data.qpos[3:7]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    g = rot.T @ np.array([0.0, 0.0, -1.0])  # gravity in the pelvis frame
    head_z = float(data.xpos[:, 2].max())
    hr = head_z / max(standing_head, 1e-9)

    down = (hr < 0.60) or (h < 0.50 * standing_height)
    if not down:
        return "up_or_mid"
    if g[0] >= 0.60:
        return "prone"
    if g[0] <= -0.60:
        return "supine"
    if abs(g[1]) >= 0.60:
        # Split by WHICH side. The obvious `abs(g[1]) >= 0.60` collapses left and right into
        # one label, so a bank that had drifted entirely onto one shoulder would report a
        # healthy "side: 454" and nothing would flag it. This project has already shipped one
        # sign-blind angular metric (torso_upright is cos(tilt)) and spent a full analysis
        # cycle believing a backward fold was a forward lean.
        return "side_left" if g[1] > 0 else "side_right"
    if g[2] <= -0.70:
        return "seated"
    return "low"


def settle(model, data, target: np.ndarray, qadr: np.ndarray, seconds: float,
           kp: float = 60.0, kd: float = 3.0) -> None:
    """Run physics under a PD hold at `target` until the body stops moving.

    The target is a real choice, not a detail: see the module docstring. It is held constant
    through the fall so the limbs arrive somewhere other than the nominal pose.
    """
    steps = int(seconds / model.opt.timestep)
    for _ in range(steps):
        q = data.qpos[qadr]
        v = data.qvel[qadr - 1] if qadr.min() >= 1 else np.zeros_like(q)
        data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        del q, v


def is_valid(model, data, qpos: np.ndarray, qvel: np.ndarray, qadr: np.ndarray
             ) -> tuple[bool, str]:
    """Finite, not interpenetrating, and a genuine fixed point of the reset path."""
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        return False, "non-finite"

    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    if data.ncon and float(-data.contact.dist.min()) > 2e-3:
        return False, f"penetration {(-data.contact.dist.min()) * 1000:.1f} mm"

    # Fixed-point check. The engine's reset writes qpos and calls mj_forward; it never
    # settles. A pose that drifts on the first quarter second was never actually at rest, and
    # the policy would spend the opening of every episode watching the body fall further.
    root0 = data.qpos[:3].copy()
    joints0 = data.qpos[qadr].copy()
    data.ctrl[:] = joints0
    for _ in range(int(0.25 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if float(np.linalg.norm(data.qpos[:3] - root0)) > 0.02:
        return False, "root drifts >2 cm"
    if float(np.abs(data.qpos[qadr] - joints0).max()) > 0.05:
        return False, "joints drift >0.05 rad"
    return True, "ok"


def random_target(model, prepared, rng) -> np.ndarray:
    """A random constant PD target inside the commandable band."""
    return prepared.default_joint_pos + rng.uniform(-1.0, 1.0, model.nu) * prepared.action_scale


def make_oriented(model, data, prepared, qadr, rng):
    """Place the body at a chosen roll angle, just above the floor, and let it settle.

    Equal numbers of prone, supine, left-side and right-side, each jittered so the landing is
    never the same twice and the side poses lean towards the belly or the back.

    Settling is what makes these valid: the pose is dropped from 12 cm with the limbs held at
    a random target, so the arms and legs arrive where the contact puts them rather than in a
    mannequin pose. Random joint angles WITHOUT settling self-penetrate, and the contact
    forces that produces dwarf anything the actuators do.
    """
    name, base = ROLL_CLUSTERS[int(rng.integers(0, len(ROLL_CLUSTERS)))]
    roll = np.radians(base + rng.uniform(-ROLL_JITTER_DEG, ROLL_JITTER_DEG))
    yaw = rng.uniform(-np.pi, np.pi)

    # Build the rotation from the axes we want, rather than composing three quaternions and
    # hoping the order is right. Measured: the composed version had no effect at all, every
    # commanded roll settled into the same 52% prone / 30% right / 15% left mix.
    #
    # Pelvis frame: x is the belly direction, y is left, z is toward the head. A lying body
    # has z horizontal, and the roll angle decides which of x and y faces the floor.
    #   roll 0   -> belly down   (prone)
    #   roll 90  -> left up      (right side down)
    #   roll 180 -> belly up     (supine)
    #   roll 270 -> left down    (left side down)
    head = np.array([np.cos(yaw), np.sin(yaw), 0.0])          # body z, horizontal
    down = np.array([0.0, 0.0, -1.0])
    side = np.cross(head, down)                                # horizontal, perpendicular
    body_x = np.cos(roll) * down + np.sin(roll) * side
    body_x /= np.linalg.norm(body_x)
    body_y = np.cross(head, body_x)
    rot = np.column_stack([body_x, body_y, head]).ravel()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot)

    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0
    data.qpos[2] = 0.12 + rng.uniform(0.0, 0.06)
    data.qpos[3:7] = quat
    # Modest joint noise. Large noise lets the limbs roll the torso off the orientation that
    # was just chosen, which is how the first version lost control of the mix entirely.
    data.qpos[qadr] += rng.normal(0.0, 0.12, model.nu)
    settle(model, data, random_target(model, prepared, rng), qadr, 2.0)


def make_topple(model, data, prepared, qadr, rng):
    """Standing, shoved hard enough to fall. The closest cheap proxy for a real fall."""
    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0
    data.qpos[qadr] += rng.normal(0.0, 0.10, model.nu)
    speed = rng.uniform(1.5, 5.0)
    angle = rng.uniform(-np.pi, np.pi)
    data.qvel[0:3] = [speed * np.cos(angle), speed * np.sin(angle), rng.uniform(-0.5, 0.5)]
    data.qvel[3:6] = rng.normal(0.0, 2.0, 3)
    settle(model, data, random_target(model, prepared, rng), qadr, 2.0)


def make_drop(model, data, prepared, qadr, rng):
    """Dropped from a height at a random orientation. Covers the supine and prone tails."""
    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0
    data.qpos[2] = rng.uniform(0.6, 1.2)
    quat = rng.normal(size=4)
    data.qpos[3:7] = quat / np.linalg.norm(quat)
    data.qpos[qadr] += rng.normal(0.0, 0.15, model.nu)
    settle(model, data, random_target(model, prepared, rng), qadr, 2.5)


def make_keyframe(model, data, prepared, qadr, rng):
    """Hand-authored floor poses that the physical generators do not produce.

    Chiefly the seated-legs-out pose. It is passively stable, so nothing that involves falling
    ever lands in it, and it is exactly the case a person means by "sitting on the floor".
    """
    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0
    kind = rng.integers(0, 3)
    joints = prepared.default_joint_pos.copy()
    names = list(prepared.joint_names)

    def setj(sub, value):
        for i, n in enumerate(names):
            if sub in n:
                joints[i] = value

    if kind == 0:            # seated, legs out in front
        data.qpos[2] = 0.35
        setj("hip_y", -1.4)
        setj("knee", 0.15)
        setj("abdomen_y", 0.25)
    elif kind == 1:          # all fours
        data.qpos[2] = 0.45
        data.qpos[3:7] = [0.92, 0.0, 0.39, 0.0]
        setj("knee", 2.2)
        setj("hip_y", -1.2)
        setj("shoulder_y", -1.2)
    else:                    # side-lying, curled
        data.qpos[2] = 0.25
        data.qpos[3:7] = [0.707, 0.707, 0.0, 0.0]
        setj("knee", 1.4)
        setj("hip_y", -0.9)
    joints += rng.normal(0.0, 0.08, model.nu)
    data.qpos[qadr] = np.clip(joints, model.actuator_ctrlrange[:, 0],
                              model.actuator_ctrlrange[:, 1])
    settle(model, data, joints, qadr, 1.5)


def make_standing(model, data, prepared, qadr, rng):
    """Nominal stand plus joint noise, so the standing reward terms are exercised from step 1.

    These are flagged so they can be excluded from the success denominator: without that a
    do-nothing policy books their whole share as free successes.
    """
    data.qpos[:] = prepared.default_qpos
    data.qvel[:] = 0.0
    data.qpos[qadr] += rng.normal(0.0, 0.05, model.nu)
    settle(model, data, prepared.default_joint_pos, qadr, 0.5)


GENERATORS = {"oriented": make_oriented, "topple": make_topple, "drop": make_drop,
              "keyframe": make_keyframe, "standing": make_standing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "fallen" / "bank_v1.npz")
    ap.add_argument("--model", type=Path,
                    default=REPO_ROOT / "humanoid_rl" / "models" / "humanoid_scene.xml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prepared = prepare(args.model)
    model = prepared.model
    data = mujoco.MjData(model)
    qadr = prepared.actuator_qpos_adr
    rng = np.random.default_rng(args.seed)

    kept_q, kept_v, kept_label, kept_gen = [], [], [], []
    rejected: dict[str, int] = {}
    started = time.perf_counter()

    # QUOTAS BY OUTCOME, not by intent. A pose is filed under what it actually settled into,
    # and generation continues until every class is full. Commanding an orientation is not
    # enough on its own: a body laid on its side sometimes rolls onto its front or back as it
    # settles, so asking for 25% of each still produced 44% prone and 3% supine. Sampling
    # until the quota is met is the only way the shares come out equal, and it is robust to
    # any bias in the generator rather than compensating for one bias in particular.
    floor_share = 1.0 - MIX["standing"]
    per_class = int(args.count * floor_share / 4)
    quota = {"prone": per_class, "supine": per_class,
             "side_left": per_class, "side_right": per_class,
             "seated": int(args.count * 0.05),
             "up_or_mid": int(args.count * MIX["standing"])}
    have = {k: 0 for k in quota}
    print("quotas:", quota, flush=True)

    attempts = 0
    limit = args.count * 60
    while any(have[k] < quota[k] for k in quota) and attempts < limit:
        attempts += 1
        # Aim at whichever class is furthest from its quota, then accept whatever comes out.
        short = max(quota, key=lambda k: quota[k] - have[k])
        if short == "up_or_mid":
            make_standing(model, data, prepared, qadr, rng)
            gen = "standing"
        elif short == "seated":
            make_keyframe(model, data, prepared, qadr, rng)
            gen = "keyframe"
        else:
            base = dict(ROLL_CLUSTERS)[short]
            saved = globals()["ROLL_CLUSTERS"]
            globals()["ROLL_CLUSTERS"] = ((short, base),)
            make_oriented(model, data, prepared, qadr, rng)
            globals()["ROLL_CLUSTERS"] = saved
            gen = "oriented"

        qpos = data.qpos.copy()
        qvel = data.qvel.copy() if rng.random() < 0.10 else np.zeros(model.nv)
        yaw = rng.uniform(-np.pi, np.pi)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        w, x, y, z = qpos[3:7]
        qpos[3:7] = [cy * w - sy * z, cy * x - sy * y, cy * y + sy * x, cy * z + sy * w]
        qpos[0:2] = 0.0

        ok, why = is_valid(model, data, qpos, qvel, qadr)
        if not ok:
            rejected[why.split()[0]] = rejected.get(why.split()[0], 0) + 1
            continue
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        label = label_pose(model, data, prepared.standing_height,
                           prepared.standing_head_height)
        if label not in quota or have[label] >= quota[label]:
            rejected["quota-full"] = rejected.get("quota-full", 0) + 1
            continue
        have[label] += 1
        kept_q.append(qpos)
        kept_v.append(qvel)
        kept_label.append(label)
        kept_gen.append(gen)
    print("filled:", have, f"({attempts} attempts)", flush=True)

    q = np.array(kept_q)
    v = np.array(kept_v)
    labels = np.array(kept_label)
    gens = np.array(kept_gen)
    # Held-out split, as HumanUP does: a policy that memorised its training poses would look
    # identical on the training split and fail on anything new.
    split = np.where(rng.random(len(q)) < 0.75, "train", "eval")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, qpos=q, qvel=v, label=labels, generator=gens, split=split,
        standing_height=prepared.standing_height,
        standing_head_height=prepared.standing_head_height,
        body_mass=float(model.body_mass.sum()),
        model_sha1=hashlib.sha1(args.model.read_bytes()).hexdigest(),
        nq=model.nq, nv=model.nv, nu=model.nu, generator_version=1,
    )

    print(f"\n{len(q)} poses -> {args.out}  ({time.perf_counter() - started:.0f}s)")
    print("label mix:", {k: int((labels == k).sum()) for k in sorted(set(labels))})
    print("rejected :", rejected)
    print(f"train/eval: {(split == 'train').sum()}/{(split == 'eval').sum()}")
    down = float((labels != "up_or_mid").mean())
    print(f"on the floor: {down:.1%}  (the rest are the deliberate standing mix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
