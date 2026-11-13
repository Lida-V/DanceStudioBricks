"""SOMA BVH reading and writing, and the pose math the rest of the app uses.

A motion is `poses`: float64 [frame, joint, 7] in BVH space (Y up, metres, XYZW quaternions).
Joint 0 is Hips and holds its world transform; every other joint holds its local transform.
The file's `Root` joint is folded into Hips on read and written back as identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MODEL_FPS = 30.0
BONES_SPINE1 = np.array([5.003913, -0.037094, -0.013727])


class MotionError(ValueError):
    pass


@dataclass(frozen=True)
class Skeleton:
    names: tuple[str, ...]  # Hips first, no Root
    parents: tuple[int, ...]  # -1 for Hips
    offsets: np.ndarray  # [J, 3] centimetres
    hierarchy: str  # the file's HIERARCHY text, reused when writing
    file_joints: tuple[dict, ...]  # every file joint with its channels, Root included

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass(frozen=True)
class Motion:
    skeleton: Skeleton
    poses: np.ndarray
    fps: float

    @property
    def frames(self) -> int:
        return self.poses.shape[0]

    @property
    def duration(self) -> float:
        return (self.frames - 1) / self.fps


def _parse_hierarchy(lines: list[str]) -> list[dict]:
    joints: list[dict] = []
    stack: list[int] = []
    in_end_site = False
    for raw in lines:
        tokens = raw.split()
        if not tokens:
            continue
        head = tokens[0]
        if head in ("ROOT", "JOINT"):
            joints.append({"name": tokens[1].rsplit(":", 1)[-1], "parent": stack[-1] if stack else -1,
                           "offset": [0.0, 0.0, 0.0], "channels": []})
            stack.append(len(joints) - 1)
        elif tokens[:2] == ["End", "Site"]:
            in_end_site = True
        elif head == "OFFSET" and not in_end_site:
            joints[stack[-1]]["offset"] = [float(v) for v in tokens[1:4]]
        elif head == "CHANNELS":
            joints[stack[-1]]["channels"] = tokens[2:]
        elif head == "}":
            if in_end_site:
                in_end_site = False
            else:
                stack.pop()
    return joints


def _axes(channels: list[str]) -> str:
    return "".join(c[0].upper() for c in channels if c.lower().endswith("rotation"))


def read_bvh(path: str | Path) -> Motion:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    marker = text.find("MOTION")
    if not text.lstrip().startswith("HIERARCHY") or marker < 0:
        raise MotionError("not a BVH file")
    hierarchy = text[:marker].rstrip()
    file_joints = _parse_hierarchy(hierarchy.splitlines())
    names = [j["name"] for j in file_joints]
    if "Hips" not in names or (names[0] not in ("Root", "Hips")):
        raise MotionError("the skeleton has no Hips joint at the top")
    body = text[marker:].split("\n", 3)
    try:
        frames = int(body[1].split(":")[1])
        frame_time = float(body[2].split(":")[1])
    except (IndexError, ValueError) as error:
        raise MotionError("BVH motion header is malformed") from error
    width = sum(len(j["channels"]) for j in file_joints)
    data = np.fromstring(body[3], sep=" ", dtype=np.float64)
    if frames < 2 or frame_time <= 0 or data.size != frames * width:
        raise MotionError("BVH motion data does not match its header")
    data = data.reshape(frames, width)

    locals_: list[np.ndarray] = []
    column = 0
    for joint in file_joints:
        count = len(joint["channels"])
        block = data[:, column:column + count]
        column += count
        position = np.tile(np.array(joint["offset"], dtype=np.float64), (frames, 1))
        rotation_columns = []
        for index, channel in enumerate(joint["channels"]):
            lowered = channel.lower()
            if lowered.endswith("position"):
                position[:, "xyz".index(lowered[0])] = block[:, index]
            else:
                rotation_columns.append(index)
        axes = _axes(joint["channels"])
        if axes:
            quat = Rotation.from_euler(axes, block[:, rotation_columns], degrees=True).as_quat()
        else:
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1))
        locals_.append(np.concatenate([position / 100.0, quat], axis=1))

    has_root = names[0] == "Root"
    start = 1 if has_root else 0
    poses = np.stack(locals_[start:], axis=1)
    if has_root:
        root = locals_[0]
        poses[:, 0, :3] = rotate(root[:, 3:7], poses[:, 0, :3]) + root[:, :3]
        poses[:, 0, 3:7] = quat_mul(root[:, 3:7], poses[:, 0, 3:7])
    shift = start
    parents = tuple(-1 if j["parent"] < shift else j["parent"] - shift for j in file_joints[start:])
    skeleton = Skeleton(tuple(names[start:]), parents, np.array([j["offset"] for j in file_joints[start:]]),
                        hierarchy, tuple(file_joints))
    return Motion(skeleton, poses, 1.0 / frame_time)


def write_bvh(path: str | Path, skeleton: Skeleton, poses: np.ndarray, fps: float) -> None:
    index = {name: i for i, name in enumerate(skeleton.names)}
    columns = []
    count = poses.shape[0]
    for joint in skeleton.file_joints:
        source = index.get(joint["name"])
        axes = _axes(joint["channels"])
        if source is None:  # the file's Root
            translation = np.zeros((count, 3))
            euler = np.zeros((count, len(axes)))
        else:
            translation = poses[:, source, :3] * 100.0
            euler = Rotation.from_quat(poses[:, source, 3:7]).as_euler(axes, degrees=True) if axes else np.zeros((count, 0))
        rotation_column = 0
        for channel in joint["channels"]:
            lowered = channel.lower()
            if lowered.endswith("position"):
                columns.append(translation[:, "xyz".index(lowered[0])])
            else:
                columns.append(euler[:, rotation_column])
                rotation_column += 1
    motion = np.stack(columns, axis=1)
    if not np.isfinite(motion).all():
        raise MotionError("non-finite values while writing BVH")
    with Path(path).open("w", encoding="utf-8", newline="\n") as out:
        out.write(skeleton.hierarchy + "\nMOTION\n")
        out.write(f"Frames: {count}\nFrame Time: {1.0 / fps:.8f}\n")
        np.savetxt(out, motion, fmt="%.6f")


def check_soma_bones(skeleton: Skeleton, required: list[str] | None = None) -> list[str]:
    """Reasons this skeleton cannot be used; empty when it is a BONES-based SOMA skeleton."""
    problems = []
    if len(skeleton.names) != 77:
        problems.append(f"SOMA は 77 関節ですが、このファイルは {len(skeleton.names)} 関節です")
    if "Spine1" not in skeleton.names:
        return problems + ["Spine1 関節がありません"]
    offset = skeleton.offsets[skeleton.index("Spine1")]
    if np.linalg.norm(offset - BONES_SPINE1) > 1.5:
        problems.append("BONES 基準ではありません（Spine1 の OFFSET が前方向ではない）")
    for name in required or []:
        if name not in skeleton.names:
            problems.append(f"{name} 関節がありません")
    return problems


# ----------------------------------------------------------------------------------------------
# Quaternion helpers (XYZW, vectorised)

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[..., :3]
    t = 2.0 * np.cross(u, v)
    return v + q[..., 3:4] * t + np.cross(u, t)


def yaw_quat(radians: float) -> np.ndarray:
    return np.array([0.0, np.sin(radians / 2.0), 0.0, np.cos(radians / 2.0)])


def world_transforms(poses: np.ndarray, parents: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """World joint positions [F, J, 3] and rotations [F, J, 4]. Parents always come before children in BVH order."""
    frames, joints = poses.shape[:2]
    positions = np.zeros((frames, joints, 3))
    rotations = np.zeros((frames, joints, 4))
    for joint, parent in enumerate(parents):
        if parent < 0:
            positions[:, joint] = poses[:, joint, :3]
            rotations[:, joint] = poses[:, joint, 3:7]
        else:
            positions[:, joint] = rotate(rotations[:, parent], poses[:, joint, :3]) + positions[:, parent]
            rotations[:, joint] = quat_mul(rotations[:, parent], poses[:, joint, 3:7])
    return positions, rotations


def world_positions(poses: np.ndarray, parents: tuple[int, ...]) -> np.ndarray:
    return world_transforms(poses, parents)[0]


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking direction a onto direction b. Arrays are [N, 3]; returns [N, 4] XYZW."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    q = np.concatenate([np.cross(a, b), 1.0 + np.sum(a * b, axis=-1, keepdims=True)], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def heading(positions: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """Facing angle per frame in radians around +Y, 0 meaning +Z, from the hip line."""
    across = positions[:, skeleton.index("LeftLeg")] - positions[:, skeleton.index("RightLeg")]
    return np.arctan2(-across[:, 2], across[:, 0])


def sample(poses: np.ndarray, fps: float, times: np.ndarray) -> np.ndarray:
    """Poses at arbitrary times (seconds), clamped to the clip. Linear position, shortest-path nlerp rotation."""
    index = np.clip(np.asarray(times, dtype=np.float64) * fps, 0.0, poses.shape[0] - 1.0)
    lower = np.floor(index).astype(int)
    upper = np.minimum(lower + 1, poses.shape[0] - 1)
    alpha = (index - lower)[:, None, None]
    a, b = poses[lower], poses[upper]
    out = a.copy()
    out[..., :3] = a[..., :3] * (1.0 - alpha) + b[..., :3] * alpha
    sign = np.where(np.sum(a[..., 3:7] * b[..., 3:7], axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    blended = a[..., 3:7] * (1.0 - alpha) + b[..., 3:7] * sign * alpha
    out[..., 3:7] = blended / np.linalg.norm(blended, axis=-1, keepdims=True)
    return out


def place(poses: np.ndarray, skeleton: Skeleton, x: float, z: float, facing_degrees: float, source_start: float = 0.0) -> np.ndarray:
    """Move a motion so its pose at `source_start` seconds stands at (x, z) facing `facing_degrees` (0 faces +Z)."""
    anchor = poses[:1] if source_start <= 1e-6 else sample(poses, MODEL_FPS, np.array([source_start]))
    first_world = world_positions(anchor, skeleton.parents)
    yaw = np.radians(facing_degrees) - float(heading(first_world, skeleton)[0])
    q = yaw_quat(yaw)
    out = poses.copy()
    origin = np.array([anchor[0, 0, 0], 0.0, anchor[0, 0, 2]])
    out[:, 0, :3] = rotate(q, poses[:, 0, :3] - origin) + np.array([x, 0.0, z])
    out[:, 0, 3:7] = quat_mul(q, poses[:, 0, 3:7])
    return out


def pose_spot(poses: np.ndarray, skeleton: Skeleton, x: float, z: float, facing_degrees: float,
              source_start: float, at_source: float) -> tuple[float, float, float]:
    """Floor (x, z) and facing (degrees) that a placed motion reaches at `at_source` seconds."""
    placed = place(poses, skeleton, x, z, facing_degrees, source_start)
    frame = sample(placed, MODEL_FPS, np.array([at_source]))
    world = world_positions(frame, skeleton.parents)
    angle = float(np.degrees(heading(world, skeleton)[0]))
    return float(frame[0, 0, 0]), float(frame[0, 0, 2]), round(((angle + 180.0) % 360.0) - 180.0, 1)


def contact_spans(flags: np.ndarray, shortest: int = 4, bridge: int = 2) -> list[tuple[int, int]]:
    """Runs of True as (start, end), joining tiny gaps and dropping runs too short to be a real plant."""
    spans: list[list[int]] = []
    index = 0
    while index < len(flags):
        if flags[index]:
            start = index
            while index < len(flags) and flags[index]:
                index += 1
            if spans and start - spans[-1][1] <= bridge:
                spans[-1][1] = index
            else:
                spans.append([start, index])
        else:
            index += 1
    return [(a, b) for a, b in spans if b - a >= shortest]


def solve_leg(poses: np.ndarray, skeleton: Skeleton, side: str, frames: np.ndarray, goals: np.ndarray) -> None:
    """Two-bone IK: move one ankle to `goals` [len(frames), 3] in place. The knee stays in its plane and the foot keeps its world rotation."""
    leg, shin, foot = (skeleton.index(side + part) for part in ("Leg", "Shin", "Foot"))
    parent = skeleton.parents[leg]
    positions, rotations = world_transforms(poses, skeleton.parents)
    hip, knee, current = positions[frames, leg], positions[frames, shin], positions[frames, foot]
    upper, lower = np.linalg.norm(knee - hip, axis=1), np.linalg.norm(current - knee, axis=1)
    reach = goals - hip
    length = np.clip(np.linalg.norm(reach, axis=1), np.abs(upper - lower) + 1e-3, upper + lower - 1e-3)
    direction = reach / np.linalg.norm(reach, axis=1, keepdims=True)
    pole = (knee - hip) - np.sum((knee - hip) * direction, axis=1, keepdims=True) * direction
    pole /= np.linalg.norm(pole, axis=1, keepdims=True) + 1e-9
    along = (upper ** 2 - lower ** 2 + length ** 2) / (2.0 * length)
    height = np.sqrt(np.maximum(upper ** 2 - along ** 2, 0.0))
    new_knee = hip + along[:, None] * direction + height[:, None] * pole
    new_ankle = hip + length[:, None] * direction
    turn_upper = rotation_between(knee - hip, new_knee - hip)
    leg_world = quat_mul(turn_upper, rotations[frames, leg])
    turned_lower = rotate(turn_upper, current - knee)
    turn_lower = rotation_between(turned_lower, new_ankle - new_knee)
    shin_world = quat_mul(turn_lower, quat_mul(turn_upper, rotations[frames, shin]))
    poses[frames, leg, 3:7] = quat_mul(quat_inverse(rotations[frames, parent]), leg_world)
    poses[frames, shin, 3:7] = quat_mul(quat_inverse(leg_world), shin_world)
    poses[frames, foot, 3:7] = quat_mul(quat_inverse(shin_world), rotations[frames, foot])


def solve_arm(
    poses: np.ndarray,
    skeleton: Skeleton,
    side: str,
    frames: np.ndarray,
    goals: np.ndarray,
    hand_rotations: np.ndarray | None = None,
) -> None:
    """Two-bone IK: move one wrist (`Hand`) to `goals` [len(frames), 3] in place while preserving elbow plane and wrist world rotation."""
    arm, forearm, hand = (skeleton.index(side + part) for part in ("Arm", "ForeArm", "Hand"))
    parent = skeleton.parents[arm]
    positions, rotations = world_transforms(poses, skeleton.parents)
    shoulder, elbow, wrist = positions[frames, arm], positions[frames, forearm], positions[frames, hand]
    upper, lower = np.linalg.norm(elbow - shoulder, axis=1), np.linalg.norm(wrist - elbow, axis=1)
    reach = goals - shoulder
    reach_norm = np.linalg.norm(reach, axis=1, keepdims=True)
    direction = reach / np.maximum(reach_norm, 1e-9)
    length = np.clip(reach_norm[:, 0], np.abs(upper - lower) + 1e-3, upper + lower - 1e-3)

    pole = (elbow - shoulder) - np.sum((elbow - shoulder) * direction, axis=1, keepdims=True) * direction
    pole_norm = np.linalg.norm(pole, axis=1, keepdims=True)
    # Fallback elbow pole (posterior-downward relative to torso) when the arm is almost collinear.
    if (pole_norm[:, 0] < 1e-3).any():
        chest_idx = skeleton.index("Chest") if "Chest" in skeleton.names else 0
        hips_idx = 0
        up_vec = positions[frames, chest_idx] - positions[frames, hips_idx]
        up_vec /= np.linalg.norm(up_vec, axis=1, keepdims=True) + 1e-9
        l_leg = skeleton.index("LeftLeg") if "LeftLeg" in skeleton.names else arm
        r_leg = skeleton.index("RightLeg") if "RightLeg" in skeleton.names else arm
        right_vec = positions[frames, l_leg] - positions[frames, r_leg]
        right_vec /= np.linalg.norm(right_vec, axis=1, keepdims=True) + 1e-9
        fwd_vec = np.cross(right_vec, up_vec)
        fallback = -0.80 * fwd_vec - 0.60 * up_vec
        fallback = fallback - np.sum(fallback * direction, axis=1, keepdims=True) * direction
        fallback /= np.linalg.norm(fallback, axis=1, keepdims=True) + 1e-9
        pole = np.where(pole_norm < 1e-3, fallback, pole / (pole_norm + 1e-9))
    else:
        pole = pole / (pole_norm + 1e-9)

    along = (upper ** 2 - lower ** 2 + length ** 2) / (2.0 * length)
    height = np.sqrt(np.maximum(upper ** 2 - along ** 2, 0.0))
    new_elbow = shoulder + along[:, None] * direction + height[:, None] * pole
    new_wrist = shoulder + length[:, None] * direction
    turn_upper = rotation_between(elbow - shoulder, new_elbow - shoulder)
    arm_world = quat_mul(turn_upper, rotations[frames, arm])
    turned_lower = rotate(turn_upper, wrist - elbow)
    turn_lower = rotation_between(turned_lower, new_wrist - new_elbow)
    forearm_world = quat_mul(turn_lower, quat_mul(turn_upper, rotations[frames, forearm]))
    target_hand_world = hand_rotations if hand_rotations is not None else rotations[frames, hand]
    poses[frames, arm, 3:7] = quat_mul(quat_inverse(rotations[frames, parent]), arm_world)
    poses[frames, forearm, 3:7] = quat_mul(quat_inverse(arm_world), forearm_world)
    poses[frames, hand, 3:7] = quat_mul(quat_inverse(forearm_world), target_hand_world)
