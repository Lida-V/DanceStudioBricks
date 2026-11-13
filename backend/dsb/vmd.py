"""MikuMikuDance motion (VMD) from a SOMA motion, and the little of a PMX model that the conversion needs.

MMD is left-handed with the model facing -Z and 1 unit = 8 cm, so positions flip Z and rotations flip X and Y.
Its bones rest in an A-pose with no rotation, so the arms are first lifted into the T-pose the motion is measured from.
Legs are written as foot IK targets, with the leg rotations alongside as the starting pose for the IK.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .motion import Skeleton, quat_inverse, quat_mul, rotation_between, world_transforms
from .retarget import continuous, tpose, turns_from_tpose

UNITS_PER_METRE = 12.5
STANDARD_ARM_DEGREES = 37.0  # how far below level the arms of a typical MMD model rest. Used when no PMX is given
STANDARD_LEG_UNITS = 10.0  # hip to knee to ankle of a typical 158 cm MMD model. Used when no PMX is given
STILL_TURN, STILL_MOVE = 1e-7, 1e-4  # a key equal to both neighbours within these is left out

# MMD bone -> (SOMA joint, MMD parent that carries rotation). The centre bone only moves, so the trunk bones hang from the world.
BODY = {
    "下半身": ("Hips", None), "上半身": ("Spine2", None), "上半身2": ("Chest", "上半身"), "首": ("Neck1", "上半身2"), "頭": ("Head", "首"),
}
_FINGERS = (("親指", "Thumb", ("０", "１", "２"), (1, 2, 3)), ("人指", "Index", ("１", "２", "３"), (2, 3, 4)), ("中指", "Middle", ("１", "２", "３"), (2, 3, 4)),
            ("薬指", "Ring", ("１", "２", "３"), (2, 3, 4)), ("小指", "Pinky", ("１", "２", "３"), (2, 3, 4)))
ARM_CHAIN: dict[str, str] = {}  # MMD bone -> the arm bone whose lift into the T-pose it shares
for _mmd, _soma in (("左", "Left"), ("右", "Right")):
    BODY.update({
        f"{_mmd}肩": (f"{_soma}Shoulder", "上半身2"), f"{_mmd}腕": (f"{_soma}Arm", f"{_mmd}肩"), f"{_mmd}ひじ": (f"{_soma}ForeArm", f"{_mmd}腕"),
        f"{_mmd}手首": (f"{_soma}Hand", f"{_mmd}ひじ"),
        f"{_mmd}足": (f"{_soma}Leg", "下半身"), f"{_mmd}ひざ": (f"{_soma}Shin", f"{_mmd}足"), f"{_mmd}足首": (f"{_soma}Foot", f"{_mmd}ひざ"),
    })
    ARM_CHAIN.update({f"{_mmd}腕": f"{_mmd}腕", f"{_mmd}ひじ": f"{_mmd}ひじ", f"{_mmd}手首": f"{_mmd}手首"})
    for _finger, _name, _marks, _numbers in _FINGERS:
        _parent = f"{_mmd}手首"
        for _mark, _number in zip(_marks, _numbers):
            _bone = f"{_mmd}{_finger}{_mark}"
            BODY[_bone] = (f"{_soma}Hand{_name}{_number}", _parent)
            ARM_CHAIN[_bone] = f"{_mmd}手首"
            _parent = _bone
ARM_TIPS = {"腕": "ひじ", "ひじ": "手首", "手首": "中指１"}  # the bone each arm bone points at


class PmxError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.at = data, 0

    def take(self, layout: str) -> tuple:
        size = struct.calcsize(layout)
        if self.at + size > len(self.data):
            raise PmxError("PMX ファイルが途中で終わっています")
        values = struct.unpack_from(layout, self.data, self.at)
        self.at += size
        return values

    def skip(self, size: int) -> None:
        self.at += size

    def text(self, encoding: str) -> str:
        (size,) = self.take("<i")
        raw = self.data[self.at:self.at + size]
        self.at += size
        return raw.decode(encoding, errors="replace")

    def index(self, size: int, signed: bool = True) -> int:
        return self.take("<" + {1: "b" if signed else "B", 2: "h" if signed else "H", 4: "i"}[size])[0]


def read_pmx_bones(path: str | Path) -> dict[str, np.ndarray]:
    """Bone name -> rest position in this app's space (metres, Y up, facing +Z), from a PMX 2.0 / 2.1 model."""
    reader = _Reader(Path(path).read_bytes())
    magic, version, count = reader.take("<4sfB")
    if magic != b"PMX " or count < 8:
        raise PmxError("PMX ファイルではありません（PMD は読み込めません）")
    encoding_flag, extra_uv, vertex_size, texture_size, _material_size, bone_size, _morph_size, _body_size = reader.take("<8B")
    reader.skip(count - 8)
    encoding = "utf-16-le" if encoding_flag == 0 else "utf-8"
    for _ in range(4):
        reader.text(encoding)
    weights = {0: bone_size, 1: 2 * bone_size + 4, 2: 4 * bone_size + 16, 3: 2 * bone_size + 40, 4: 4 * bone_size + 16}
    for _ in range(reader.take("<i")[0]):
        reader.skip(32 + 16 * extra_uv)
        (deform,) = reader.take("<B")
        if deform not in weights:
            raise PmxError("PMX の頂点データを読み取れません")
        reader.skip(weights[deform] + 4)
    reader.skip(reader.take("<i")[0] * vertex_size)
    for _ in range(reader.take("<i")[0]):
        reader.text(encoding)
    for _ in range(reader.take("<i")[0]):
        reader.text(encoding)
        reader.text(encoding)
        reader.skip(16 + 12 + 4 + 12 + 1 + 16 + 4 + 2 * texture_size + 1)
        (shared_toon,) = reader.take("<B")
        reader.skip(1 if shared_toon else texture_size)
        reader.text(encoding)
        reader.skip(4)
    bones: dict[str, np.ndarray] = {}
    for _ in range(reader.take("<i")[0]):
        name = reader.text(encoding)
        reader.text(encoding)
        x, y, z = reader.take("<3f")
        reader.index(bone_size)
        reader.skip(4)
        (flags,) = reader.take("<H")
        reader.skip(bone_size if flags & 0x0001 else 12)
        if flags & 0x0300:
            reader.skip(bone_size + 4)
        if flags & 0x0400:
            reader.skip(12)
        if flags & 0x0800:
            reader.skip(24)
        if flags & 0x2000:
            reader.skip(4)
        if flags & 0x0020:
            reader.skip(bone_size + 8)
            for _ in range(reader.take("<i")[0]):
                reader.skip(bone_size)
                (limited,) = reader.take("<B")
                reader.skip(24 if limited else 0)
        bones.setdefault(name, np.array([x, y, -z]) / UNITS_PER_METRE)
    return bones


def _arm_lifts(bones: dict[str, np.ndarray] | None) -> dict[str, np.ndarray]:
    """World rotation that brings each resting arm bone to the T-pose: along +X on the left, -X on the right."""
    lifts = {}
    for side, sign in (("左", 1.0), ("右", -1.0)):
        level = np.array([[sign, 0.0, 0.0]])
        angle = np.radians(STANDARD_ARM_DEGREES)
        standard = rotation_between(np.array([[sign * np.cos(angle), -np.sin(angle), 0.0]]), level)[0]
        lift = standard
        for bone, tip in ARM_TIPS.items():
            base, end = (bones or {}).get(side + bone), (bones or {}).get(side + tip)
            if base is not None and end is not None and np.linalg.norm(end - base) > 1e-4:
                lift = rotation_between((end - base)[None], level)[0]
            lifts[side + bone] = lift  # a model without the tip bone keeps the lift of the bone above
    return lifts


def _leg_units(bones: dict[str, np.ndarray] | None) -> float | None:
    if bones is None or any(side + name not in bones for side in "左右" for name in ("足", "ひざ", "足首")):
        return None
    lengths = [np.linalg.norm(bones[side + "ひざ"] - bones[side + "足"]) + np.linalg.norm(bones[side + "足首"] - bones[side + "ひざ"]) for side in "左右"]
    return float(np.mean(lengths)) * UNITS_PER_METRE


def mmd_motion(poses: np.ndarray, skeleton: Skeleton, bones: dict[str, np.ndarray] | None = None) -> dict[str, tuple[np.ndarray | None, np.ndarray]]:
    """MMD bone -> (position [F, 3] or None, rotation [F, 4]) per frame, in MMD's space and units.

    `bones` are a model's rest positions from `read_pmx_bones`. With them the arm angle, the leg length and the bones to
    write follow that model; without them a typical model is assumed.
    """
    names = [name for name, (joint, _) in BODY.items() if joint in skeleton.names and (bones is None or name in bones)]
    turns = dict(zip(names, np.moveaxis(turns_from_tpose(poses, skeleton, [BODY[name][0] for name in names]), 1, 0)))
    lifts = _arm_lifts(bones)
    world = {name: quat_mul(turns[name], lifts[ARM_CHAIN[name]]) if name in ARM_CHAIN else turns[name] for name in names}
    rest = world_transforms(tpose(skeleton), skeleton.parents)[0][0]
    moving = world_transforms(poses, skeleton.parents)[0]
    legs = [skeleton.index(side + part) for side in ("Left", "Right") for part in ("Leg", "Shin", "Foot")]
    soma_leg = float(np.mean([np.linalg.norm(rest[legs[i + 1]] - rest[legs[i]]) + np.linalg.norm(rest[legs[i + 2]] - rest[legs[i + 1]]) for i in (0, 3)]))
    scale = (_leg_units(bones) or STANDARD_LEG_UNITS) / soma_leg
    flip_position, flip_rotation = np.array([1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0])
    out: dict[str, tuple[np.ndarray | None, np.ndarray]] = {}
    still = np.tile([0.0, 0.0, 0.0, 1.0], (poses.shape[0], 1))
    hips = skeleton.index("Hips")
    out["センター"] = ((moving[:, hips] - rest[hips]) * scale * flip_position, still)
    for name in names:
        parent = BODY[name][1]
        while parent is not None and parent not in world:
            parent = BODY[parent][1]
        local = world[name] if parent is None else quat_mul(quat_inverse(world[parent]), world[name])
        out[name] = (None, continuous(local) * flip_rotation)
    for mmd, soma in (("左", "Left"), ("右", "Right")):
        if bones is not None and mmd + "足ＩＫ" not in bones:
            continue
        foot = skeleton.index(soma + "Foot")
        turn = world.get(mmd + "足首")
        if turn is None:
            turn = turns_from_tpose(poses, skeleton, [soma + "Foot"])[:, 0]
        out[mmd + "足ＩＫ"] = ((moving[:, foot] - rest[foot]) * scale * flip_position, turn * flip_rotation)
    return out


_CURVE = bytes([20, 20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107,
                20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 1,
                20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 1, 0,
                20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 1, 0, 0])  # straight-line interpolation on every channel


_RECORD = np.dtype([("name", "u1", 15), ("frame", "<u4"), ("position", "<f4", 3), ("rotation", "<f4", 4), ("curve", "u1", 64)])


def _keys(position: np.ndarray | None, rotation: np.ndarray) -> np.ndarray:
    """Frames worth a key: the first, the last, and every one that differs from a neighbour."""
    frames = rotation.shape[0]
    same = 1.0 - np.abs(np.sum(rotation[1:] * rotation[:-1], axis=1)) < STILL_TURN
    if position is not None:
        same &= np.abs(position[1:] - position[:-1]).max(axis=1) < STILL_MOVE
    keep = np.ones(frames, dtype=bool)
    keep[1:-1] = ~(same[:-1] & same[1:])
    return np.flatnonzero(keep)


def write_vmd(path: str | Path, poses: np.ndarray, skeleton: Skeleton, bones: dict[str, np.ndarray] | None = None, model_name: str = "Dance Studio Bricks") -> dict:
    """Write the motion at MMD's 30 frames per second, one key per frame wherever a bone moves. Returns counts for the report."""
    motion = mmd_motion(poses, skeleton, bones)
    blocks = []
    for name, (position, rotation) in motion.items():
        label = name.encode("shift_jis")
        if len(label) > 15:
            continue
        frames = _keys(position, rotation)
        block = np.zeros(len(frames), dtype=_RECORD)
        block["name"] = np.frombuffer(label.ljust(15, b"\0"), dtype=np.uint8)
        block["frame"] = frames
        if position is not None:
            block["position"] = position[frames]
        block["rotation"] = rotation[frames]
        block["curve"] = np.frombuffer(_CURVE, dtype=np.uint8)
        blocks.append(block)
    keys = sum(len(block) for block in blocks)
    with Path(path).open("wb") as out:
        out.write(b"Vocaloid Motion Data 0002".ljust(30, b"\0"))
        out.write(model_name.encode("shift_jis", errors="replace")[:20].ljust(20, b"\0"))
        out.write(struct.pack("<I", keys))
        for block in blocks:
            out.write(block.tobytes())
        out.write(struct.pack("<4I", 0, 0, 0, 0))  # no morph, camera, light or shadow keys
    return {"bones": len(blocks), "keys": keys}


def read_vmd(path: str | Path) -> dict[str, list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]]]:
    """Bone name -> [(frame, position, rotation)] from a VMD file. Used to check what was written."""
    data = Path(path).read_bytes()
    if not data.startswith(b"Vocaloid Motion Data 0002"):
        raise ValueError("not a VMD file")
    (count,) = struct.unpack_from("<I", data, 50)
    keys: dict[str, list] = {}
    for number in range(count):
        at = 54 + number * 111
        name = data[at:at + 15].split(b"\0")[0].decode("shift_jis")
        values = struct.unpack_from("<I3f4f", data, at + 15)
        keys.setdefault(name, []).append((values[0], values[1:4], values[4:8]))
    return keys
