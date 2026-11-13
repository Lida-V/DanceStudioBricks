"""Carry a SOMA motion onto other humanoids.

Every target is driven the same way: by how far each SOMA joint has turned, in world space, from the standard T-pose
(facing +Z, left arm along +X). A model put into its own T-pose and given those turns strikes the same pose,
whatever its rest pose or bone axes are.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .motion import Skeleton, quat_inverse, quat_mul, world_transforms
from .soma_rest import SOMA_JOINTS

TPOSE = {name: np.array(quat) / np.linalg.norm(quat) for name, _, _, quat in SOMA_JOINTS}  # the table is rounded, so make each a unit again

# VRM humanoid bone -> (SOMA joint, parent humanoid bone). Order is parents first.
_FINGERS = (("Thumb", ("Metacarpal", "Proximal", "Distal"), (1, 2, 3)), ("Index", ("Proximal", "Intermediate", "Distal"), (2, 3, 4)),
            ("Middle", ("Proximal", "Intermediate", "Distal"), (2, 3, 4)), ("Ring", ("Proximal", "Intermediate", "Distal"), (2, 3, 4)),
            ("Little", ("Proximal", "Intermediate", "Distal"), (2, 3, 4)))
HUMANOID: dict[str, tuple[str, str | None]] = {
    "hips": ("Hips", None), "spine": ("Spine1", "hips"), "chest": ("Spine2", "spine"), "upperChest": ("Chest", "chest"),
    "neck": ("Neck1", "upperChest"), "head": ("Head", "neck"),
}
for _side in ("left", "right"):
    _soma = _side.capitalize()
    HUMANOID.update({
        f"{_side}Shoulder": (f"{_soma}Shoulder", "upperChest"), f"{_side}UpperArm": (f"{_soma}Arm", f"{_side}Shoulder"),
        f"{_side}LowerArm": (f"{_soma}ForeArm", f"{_side}UpperArm"), f"{_side}Hand": (f"{_soma}Hand", f"{_side}LowerArm"),
        f"{_side}UpperLeg": (f"{_soma}Leg", "hips"), f"{_side}LowerLeg": (f"{_soma}Shin", f"{_side}UpperLeg"),
        f"{_side}Foot": (f"{_soma}Foot", f"{_side}LowerLeg"), f"{_side}Toes": (f"{_soma}ToeBase", f"{_side}Foot"),
    })
    for _finger, _parts, _numbers in _FINGERS:
        _parent = f"{_side}Hand"
        for _part, _number in zip(_parts, _numbers):
            _bone = f"{_side}{_finger}{_part}"
            HUMANOID[_bone] = (f"{_soma}Hand{'Pinky' if _finger == 'Little' else _finger}{_number}", _parent)
            _parent = _bone


def soma_skeleton() -> Skeleton:
    """The standard SOMA skeleton from the built-in table, for work that has no BVH at hand."""
    return Skeleton(tuple(j[0] for j in SOMA_JOINTS), tuple(j[1] for j in SOMA_JOINTS), np.array([j[2] for j in SOMA_JOINTS]), "", ())


def tpose(skeleton: Skeleton) -> np.ndarray:
    """One frame [1, J, 7] of the standard T-pose, standing with its lowest joint on the floor."""
    pose = np.zeros((1, len(skeleton.names), 7))
    for index, (name, parent) in enumerate(zip(skeleton.names, skeleton.parents)):
        pose[0, index, :3] = skeleton.offsets[index] / 100.0
        pose[0, index, 3:7] = TPOSE[name] if parent < 0 else quat_mul(quat_inverse(TPOSE[skeleton.names[parent]]), TPOSE[name])
    pose[0, 0, :3] = 0.0
    pose[0, 0, 1] = -world_transforms(pose, skeleton.parents)[0][0, :, 1].min()
    return pose


REST_KINDS = ("bones", "t", "a")
ARM_JOINTS = ("LeftArm", "RightArm")  # the joints an A-pose lowers; everything below them follows


def rest_pose(skeleton: Skeleton, kind: str, arm_degrees: float = 45.0) -> np.ndarray:
    """One frame [1, J, 7] of the pose a written file rests in: the standard T-pose, or an A-pose with both arms `arm_degrees` below level."""
    pose = tpose(skeleton)
    if kind == "t":
        return pose
    _, rotations = world_transforms(pose, skeleton.parents)
    half = np.radians(arm_degrees) / 2.0
    for name, sign in zip(ARM_JOINTS, (-1.0, 1.0)):  # the left arm points along +X, so turning it down is a negative turn about +Z
        joint = skeleton.index(name)
        lowered = quat_mul(np.array([0.0, 0.0, sign * np.sin(half), np.cos(half)]), rotations[0, joint])
        pose[0, joint, 3:7] = quat_mul(quat_inverse(rotations[0, skeleton.parents[joint]]), lowered)
    return pose


def _hierarchy(joints: list[dict]) -> str:
    """BVH HIERARCHY text for joints listed parents first. Leaves have no End Site, as in the SOMA files."""
    children: dict[int, list[int]] = {}
    for index, joint in enumerate(joints):
        children.setdefault(joint["parent"], []).append(index)
    lines = ["HIERARCHY"]

    def write(index: int, depth: int) -> None:
        joint, pad = joints[index], "\t" * depth
        lines.append(f"{pad}{'ROOT' if joint['parent'] < 0 else 'JOINT'} {joint['name']}")
        lines.append(pad + "{")
        lines.append(f"{pad}\tOFFSET {joint['offset'][0]:.6f} {joint['offset'][1]:.6f} {joint['offset'][2]:.6f}")
        lines.append(f"{pad}\tCHANNELS {len(joint['channels'])} {' '.join(joint['channels'])}")
        for child in children.get(index, []):
            write(child, depth + 1)
        lines.append(pad + "}")

    write(0, 0)
    return "\n".join(lines)


def rebased(poses: np.ndarray, skeleton: Skeleton, rest: np.ndarray) -> tuple[Skeleton, np.ndarray]:
    """The same motion on a skeleton whose zero pose is `rest`: (skeleton, poses) ready for `write_bvh`.

    Every joint ends up in the same place as before. Bone offsets become the joint spacing of `rest` in world axes, and each
    rotation is measured from `rest`, so software that expects a T-pose or A-pose rest reads the file as it stands.
    """
    rest_places, rest_turns = world_transforms(rest, skeleton.parents)
    _, turns = world_transforms(poses, skeleton.parents)
    out = np.zeros_like(poses)
    offsets = np.zeros_like(skeleton.offsets, dtype=np.float64)
    from_rest = [quat_mul(turns[:, joint], quat_inverse(rest_turns[0, joint])) for joint in range(len(skeleton.names))]
    for joint, parent in enumerate(skeleton.parents):
        if parent < 0:
            offsets[joint] = [0.0, rest_places[0, joint, 1] * 100.0, 0.0]
            out[:, joint, :3] = poses[:, joint, :3]
            out[:, joint, 3:7] = from_rest[joint]
        else:
            offsets[joint] = (rest_places[0, joint] - rest_places[0, parent]) * 100.0
            out[:, joint, :3] = offsets[joint] / 100.0
            out[:, joint, 3:7] = quat_mul(quat_inverse(from_rest[parent]), from_rest[joint])
    out[..., 3:7] = continuous(out[..., 3:7])
    turn = ["Zrotation", "Yrotation", "Xrotation"]
    # Hips is the top joint here. The SOMA files wrap it in a `Root` that never moves, and a reader that joins a bone to the
    # parent standing straight below it (Blender, reading the FBX) would then ignore the hips' travel.
    known = {joint["name"]: joint for joint in skeleton.file_joints}
    joints = []
    for name, parent, offset in zip(skeleton.names, skeleton.parents, offsets):
        channels = known.get(name, {}).get("channels") or (["Xposition", "Yposition", "Zposition", *turn] if parent < 0 else turn)
        joints.append({"name": name, "parent": parent, "offset": [float(v) for v in offset], "channels": list(channels)})
    return Skeleton(skeleton.names, skeleton.parents, offsets, _hierarchy(joints), tuple(joints)), out


def continuous(quats: np.ndarray) -> np.ndarray:
    """The same rotations with signs chosen so neighbouring frames never flip, for players that blend without checking."""
    dots = np.sum(quats[1:] * quats[:-1], axis=-1)
    signs = np.concatenate([np.ones((1, *dots.shape[1:])), np.where(dots < 0.0, -1.0, 1.0)], axis=0)
    return quats * np.cumprod(signs, axis=0)[..., None]


def raw_turns_from_tpose(poses: np.ndarray, skeleton: Skeleton, joints: list[str]) -> np.ndarray:
    """World rotation of each named joint measured from the T-pose, [F, len(joints), 4] XYZW, before sign smoothing."""
    _, rotations = world_transforms(poses, skeleton.parents)
    return np.stack([quat_mul(rotations[:, skeleton.index(name)], quat_inverse(TPOSE[name])) for name in joints], axis=1)


def turns_from_tpose(poses: np.ndarray, skeleton: Skeleton, joints: list[str]) -> np.ndarray:
    """World rotation of each named joint measured from the T-pose, [F, len(joints), 4] XYZW."""
    return continuous(raw_turns_from_tpose(poses, skeleton, joints))


def humanoid_bones(skeleton: Skeleton) -> list[str]:
    return [bone for bone, (joint, _) in HUMANOID.items() if joint in skeleton.names]


def humanoid_turns(poses: np.ndarray, skeleton: Skeleton) -> tuple[list[str], np.ndarray]:
    """(humanoid bone names, world turns from the T-pose [F, bones, 4]) for the bones this skeleton has."""
    bones = humanoid_bones(skeleton)
    return bones, turns_from_tpose(poses, skeleton, [HUMANOID[bone][0] for bone in bones])


def incremental_turns(poses: np.ndarray, skeleton: Skeleton, before_poses: np.ndarray | None = None,
                      before_raw: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(raw_turns, continuous_turns), recomputing only frames where `poses` differs from `before_poses`."""
    joints = [HUMANOID[bone][0] for bone in humanoid_bones(skeleton)]
    if before_poses is None or before_raw is None or before_poses.shape != poses.shape or before_raw.shape[:2] != (poses.shape[0], len(joints)):
        raw = raw_turns_from_tpose(poses, skeleton, joints)
    else:
        raw = before_raw
        changed = np.flatnonzero((poses != before_poses).any(axis=(1, 2)))
        if changed.size:
            raw = before_raw.copy()
            raw[changed] = raw_turns_from_tpose(poses[changed], skeleton, joints)
    return raw, continuous(raw)


def write_vrma(path: str | Path, poses: np.ndarray, skeleton: Skeleton, fps: float) -> None:
    """A VRM Animation (.vrma): the motion on the VRM humanoid bones, which any VRM 0.x or 1.0 model can play."""
    bones, turns = humanoid_turns(poses, skeleton)
    index = {bone: number for number, bone in enumerate(bones)}
    rest = world_transforms(tpose(skeleton), skeleton.parents)[0][0]
    frames = poses.shape[0]
    chunks: list[bytes] = []
    views, accessors, channels, samplers = [], [], [], []

    def accessor(values: np.ndarray, kind: str, bounds: bool = False) -> int:
        data = np.ascontiguousarray(values, dtype="<f4")
        offset = sum(len(chunk) for chunk in chunks)
        chunks.append(data.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": data.nbytes})
        entry = {"bufferView": len(views) - 1, "componentType": 5126, "count": int(data.shape[0]), "type": kind}
        if bounds:
            entry.update(min=[float(data.min())], max=[float(data.max())])
        accessors.append(entry)
        return len(accessors) - 1

    def track(node: int, target: str, values: np.ndarray, kind: str) -> None:
        samplers.append({"input": times, "output": accessor(values, kind), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": target}})

    times = accessor(np.arange(frames) / fps, "SCALAR", bounds=True)
    nodes = []
    for bone in bones:
        joint, parent = HUMANOID[bone]
        while parent is not None and parent not in index:
            parent = HUMANOID[parent][1]
        place = rest[skeleton.index(joint)] - (rest[skeleton.index(HUMANOID[parent][0])] if parent else 0.0)
        nodes.append({"name": bone, "translation": [round(float(v), 6) for v in place]})
        if parent:
            nodes[index[parent]].setdefault("children", []).append(index[bone])
        local = turns[:, index[bone]] if parent is None else quat_mul(quat_inverse(turns[:, index[parent]]), turns[:, index[bone]])
        track(index[bone], "rotation", continuous(local), "VEC4")
    track(index["hips"], "translation", poses[:, skeleton.index("Hips"), :3], "VEC3")
    binary = b"".join(chunks)
    document = {
        "asset": {"version": "2.0", "generator": "Dance Studio Bricks"},
        "extensionsUsed": ["VRMC_vrm_animation"],
        "extensions": {"VRMC_vrm_animation": {"specVersion": "1.0", "humanoid": {"humanBones": {bone: {"node": index[bone]} for bone in bones}}}},
        "scene": 0, "scenes": [{"nodes": [index["hips"]]}], "nodes": nodes,
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
        "animations": [{"name": "dance", "channels": channels, "samplers": samplers}],
    }
    text = json.dumps(document, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    with Path(path).open("wb") as out:
        out.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(binary)))
        out.write(struct.pack("<I4s", len(text), b"JSON") + text)
        out.write(struct.pack("<I4s", len(binary), b"BIN\0") + binary)
