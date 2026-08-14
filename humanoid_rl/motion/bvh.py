"""BVH motion capture parser and forward kinematics.

BVH (Biovision Hierarchy) is the lingua franca of mocap: a skeleton definition followed by
one row of channel values per frame. It is a small, frozen, well-specified format, which is
why this is written here rather than taken as a dependency. The maintained PyPI options
were a name collision with a MongoDB client (`pymo`), a package last touched in 2017
(`bvh`), and one with an unstated licence (`bvhio`).

Structure of the format:

    HIERARCHY
    ROOT Hips
    {
      OFFSET 0.0 0.0 0.0
      CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
      JOINT Spine
      {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site { OFFSET 0.0 5.0 0.0 }
      }
    }
    MOTION
    Frames: 300
    Frame Time: 0.0333333
    <300 rows of floats, one value per channel in declaration order>

Two details that are easy to get wrong and silently produce garbage:

* **Rotation order is per joint and is given by the channel order**, not fixed. A joint
  declaring `Zrotation Xrotation Yrotation` composes as R = Rz @ Rx @ Ry. Assuming a global
  XYZ convention produces motion that looks almost right and drifts badly on turns.
* **Units are usually centimetres.** Feeding raw values to a model built in metres yields a
  30 metre tall human, and the resulting retarget silently optimises nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_CHANNEL_AXIS = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}
_POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")


@dataclass
class BvhJoint:
    name: str
    parent: int  # index into Skeleton.joints, -1 for the root
    offset: np.ndarray  # (3,) translation from the parent, in the file's units
    channels: list[str] = field(default_factory=list)
    #: Column indices into a motion frame for this joint's channels, in declaration order.
    channel_indices: list[int] = field(default_factory=list)
    is_end_site: bool = False

    @property
    def rotation_order(self) -> str:
        """Intrinsic Euler order for scipy, e.g. "ZXY", built from the channel order."""
        return "".join(
            _CHANNEL_AXIS[c].upper() for c in self.channels if c in _CHANNEL_AXIS
        )


@dataclass
class BvhMotion:
    """A parsed BVH file: a skeleton plus per-frame channel values."""

    joints: list[BvhJoint]
    frames: np.ndarray  # (n_frames, n_channels)
    frame_time: float  # seconds per frame
    source: Path | None = None
    #: Scale applied to every length when the file was loaded (see `load`).
    scale: float = 1.0

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 0.0

    @property
    def duration(self) -> float:
        return self.n_frames * self.frame_time

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    def index_of(self, name: str) -> int:
        """Index of a joint by exact name, then by case-insensitive substring.

        Mocap skeletons disagree endlessly on naming ("LeftFoot", "L_Foot", "lFoot"), so a
        forgiving lookup saves writing a bespoke alias table per dataset.
        """
        for i, joint in enumerate(self.joints):
            if joint.name == name:
                return i
        lowered = name.lower().replace("_", "")
        for i, joint in enumerate(self.joints):
            if lowered in joint.name.lower().replace("_", ""):
                return i
        raise KeyError(f"no joint matching {name!r} in {self.names}")

    # ------------------------------------------------------------------ kinematics

    def forward_kinematics(self, frames: slice | np.ndarray | None = None) -> np.ndarray:
        """World positions of every joint.

        Args:
            frames: Optional frame selection. Defaults to all frames.

        Returns:
            (n_selected, n_joints, 3) array of world-space joint positions.

        Computed for all requested frames at once with batched rotations, because a
        per-frame Python loop over a 100k-frame dataset is the difference between seconds
        and minutes.
        """
        data = self.frames if frames is None else self.frames[frames]
        n = data.shape[0]
        n_joints = len(self.joints)

        positions = np.zeros((n, n_joints, 3))
        # World rotation of each joint, accumulated down the chain.
        world_rot = [None] * n_joints

        for j, joint in enumerate(self.joints):
            # Local rotation from this joint's own rotation channels.
            rot_cols = [
                idx
                for c, idx in zip(joint.channels, joint.channel_indices)
                if c in _CHANNEL_AXIS
            ]
            if rot_cols:
                angles = np.deg2rad(data[:, rot_cols])
                local_rot = Rotation.from_euler(joint.rotation_order, angles)
            else:
                local_rot = Rotation.identity(n)

            # Local translation: the fixed offset, plus any position channels (root only).
            offset = np.broadcast_to(joint.offset, (n, 3)).copy()
            pos_cols = [
                idx
                for c, idx in zip(joint.channels, joint.channel_indices)
                if c in _POSITION_CHANNELS
            ]
            if pos_cols:
                axis = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
                for c, idx in zip(joint.channels, joint.channel_indices):
                    if c in _POSITION_CHANNELS:
                        offset[:, axis[c]] += data[:, idx] * self.scale

            if joint.parent < 0:
                positions[:, j] = offset
                world_rot[j] = local_rot
            else:
                parent_rot = world_rot[joint.parent]
                positions[:, j] = positions[:, joint.parent] + parent_rot.apply(offset)
                world_rot[j] = parent_rot * local_rot

        return positions


def load(path: str | Path, scale: float = 0.01) -> BvhMotion:
    """Parse a BVH file.

    Args:
        path: File to read.
        scale: Multiplier applied to every length. Defaults to 0.01 because BVH files are
            almost always authored in centimetres while physics models are in metres.
            Getting this wrong produces a 30 metre human and a retarget that optimises
            nonsense without ever erroring.
    """
    path = Path(path)
    text = path.read_text(errors="ignore")

    hierarchy, _, motion = text.partition("MOTION")
    if not motion:
        raise ValueError(f"{path} has no MOTION section; not a valid BVH file")

    joints = _parse_hierarchy(hierarchy, scale)
    frames, frame_time = _parse_motion(motion)

    expected = sum(len(j.channels) for j in joints)
    if frames.shape[1] != expected:
        raise ValueError(
            f"{path}: hierarchy declares {expected} channels but motion rows have "
            f"{frames.shape[1]}"
        )
    return BvhMotion(joints=joints, frames=frames, frame_time=frame_time, source=path, scale=scale)


def _parse_hierarchy(text: str, scale: float) -> list[BvhJoint]:
    tokens = text.replace("{", " { ").replace("}", " } ").split()
    joints: list[BvhJoint] = []
    stack: list[int] = []
    channel_cursor = 0
    i = 0
    end_site_counter = 0

    while i < len(tokens):
        token = tokens[i]

        if token in ("ROOT", "JOINT"):
            name = tokens[i + 1]
            parent = stack[-1] if stack else -1
            joints.append(BvhJoint(name=name, parent=parent, offset=np.zeros(3)))
            stack.append(len(joints) - 1)
            i += 2

        elif token == "End":
            # "End Site": a terminal marker with an offset but no channels. Kept as a
            # joint because fingertips, toe tips and head tops are often the most useful
            # retargeting targets and exist only as end sites.
            parent = stack[-1]
            end_site_counter += 1
            joints.append(
                BvhJoint(
                    name=f"{joints[parent].name}_End",
                    parent=parent,
                    offset=np.zeros(3),
                    is_end_site=True,
                )
            )
            stack.append(len(joints) - 1)
            i += 2  # skip "End" and "Site"

        elif token == "OFFSET":
            joints[stack[-1]].offset = np.array(
                [float(tokens[i + 1]), float(tokens[i + 2]), float(tokens[i + 3])]
            ) * scale
            i += 4

        elif token == "CHANNELS":
            count = int(tokens[i + 1])
            names = tokens[i + 2 : i + 2 + count]
            joint = joints[stack[-1]]
            joint.channels = names
            joint.channel_indices = list(range(channel_cursor, channel_cursor + count))
            channel_cursor += count
            i += 2 + count

        elif token == "}":
            stack.pop()
            i += 1

        else:
            i += 1

    if not joints:
        raise ValueError("no joints found in BVH hierarchy")
    return joints


def _parse_motion(text: str) -> tuple[np.ndarray, float]:
    lines = text.strip().splitlines()
    frame_time = 1.0 / 30.0
    n_frames = None
    data_start = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("frames:"):
            n_frames = int(re.findall(r"\d+", stripped)[0])
        elif stripped.lower().startswith("frame time:"):
            frame_time = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)[0])
            data_start = idx + 1
            break

    rows = [ln for ln in lines[data_start:] if ln.strip()]
    frames = np.array([[float(v) for v in ln.split()] for ln in rows], dtype=np.float64)
    if n_frames is not None and frames.shape[0] != n_frames:
        # Trust the actual data over the header, which is often wrong in the wild after
        # a clip has been trimmed by a tool that forgot to update it.
        pass
    return frames, frame_time
