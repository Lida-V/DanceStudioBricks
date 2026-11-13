"""Phrase library: import validated SOMA BVH files, analyse them, and keep them at the model frame rate."""
from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import scipy.signal

from .motion import MODEL_FPS, MotionError, Skeleton, check_soma_bones, heading, read_bvh, sample, world_positions

CATEGORIES = {"pose": "ポーズ", "groove": "グルーヴ", "wave": "ウェーブ", "power": "パワー", "foot": "足技"}
ACCENT_JOINTS = ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head", "Hips")
REQUIRED = list(ACCENT_JOINTS) + ["LeftLeg", "RightLeg"]
NOMINAL_BPM = 128.0


def nominal_bars(duration: float) -> tuple[int, str]:
    """Bars this phrase is meant to fill at a typical tempo, and whether auto-arrange may use it."""
    bars = duration / (240.0 / NOMINAL_BPM)
    if bars < 1.35:
        return 1, "manual"
    if bars < 2.9:
        return 2, "auto"
    if bars < 5.6:
        return 4, "auto"
    return int(round(bars)), "manual"


def motion_accents(world: np.ndarray, skeleton: Skeleton) -> list[float]:
    """Times (seconds) of sharp stops and hits: where limb speed falls fastest after being high."""
    indexes = [skeleton.index(name) for name in ACCENT_JOINTS]
    speed = np.linalg.norm(np.diff(world[:, indexes], axis=0), axis=-1) * MODEL_FPS
    relative = speed[:, :-1] - speed[:, -1:]  # relative to the hips, so travelling is not a beat
    total = np.mean(np.abs(relative), axis=1)
    total = np.convolve(total, np.ones(3) / 3.0, mode="same")
    fall = -np.gradient(total)
    height = max(float(np.percentile(fall, 80)), 1e-3)
    peaks, _ = scipy.signal.find_peaks(fall, distance=int(0.22 * MODEL_FPS), height=height, prominence=height * 0.5)
    return [round(float(p) / MODEL_FPS, 3) for p in peaks]


def describe(world: np.ndarray, skeleton: Skeleton) -> dict:
    index = skeleton.index
    speed = np.linalg.norm(np.diff(world, axis=0), axis=-1) * MODEL_FPS
    hips = world[:, index("Hips")]
    hands = speed[:, [index("LeftHand"), index("RightHand")]].mean()
    feet = speed[:, [index("LeftFoot"), index("RightFoot")]].mean()
    head_height = world[:, index("Head"), 1]
    travel = float(np.linalg.norm(np.diff(hips[:, [0, 2]], axis=0), axis=1).sum())
    turn = float(np.ptp(np.unwrap(heading(world, skeleton))))
    level = float(0.6 * hands + 0.4 * feet)
    energy = int(np.clip(np.digitize(level, [0.35, 0.6, 0.95, 1.35]) + 1, 1, 5))
    jump = float(head_height.max() - np.median(head_height))
    if level < 0.3:
        category = "pose"
    elif feet >= 0.8 and feet > hands * 0.6:
        category = "foot"
    elif np.degrees(turn) > 200 or jump > 0.25 or energy >= 4:
        category = "power"
    elif hands > feet * 3.0 and level < 0.85:
        category = "wave"
    else:
        category = "groove"
    return {"category": category, "energy": energy, "travel": round(travel, 2), "turnDegrees": round(float(np.degrees(turn))),
            "handSpeed": round(float(hands), 2), "footSpeed": round(float(feet), 2)}


THUMB_SKIP = ("Thumb", "Index", "Middle", "Ring", "Pinky", "End", "Jaw", "Eye", "Nose", "Ear", "Tongue", "Lip")


def thumbnail(world: np.ndarray, skeleton: Skeleton, accents: list[float]) -> dict:
    """A front view of one characteristic pose for the library card."""
    frame = int(round(accents[len(accents) // 2] * MODEL_FPS)) if accents else world.shape[0] // 2
    frame = min(frame, world.shape[0] - 1)
    pose = world[frame:frame + 1]
    angle = float(heading(pose, skeleton)[0])
    pose = pose[0] - pose[0, skeleton.index("Hips")] * np.array([1.0, 0.0, 1.0])
    cos, sin = np.cos(-angle), np.sin(-angle)
    x = pose[:, 0] * cos + pose[:, 2] * sin
    keep = [i for i, name in enumerate(skeleton.names) if not any(token in name for token in THUMB_SKIP)]
    return {skeleton.names[i]: [round(float(x[i]), 3), round(float(pose[i, 1]), 3)] for i in keep}


class Library:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        self.items: dict[str, dict] = {}
        self.skeleton: Skeleton | None = None
        self._poses: dict[str, np.ndarray] = {}
        if self.index_file.is_file():
            self.items = {item["id"]: item for item in json.loads(self.index_file.read_text(encoding="utf-8"))}

    def save(self) -> None:
        temp = self.index_file.with_suffix(".tmp")
        temp.write_text(json.dumps(list(self.items.values()), ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(self.index_file)

    def template_bvh(self) -> Path | None:
        for item in self.items.values():
            return self.root / item["id"] / "source.bvh"
        return None

    def import_paths(self, paths: list[str], pack: str = "user") -> dict:
        files: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                files.extend(sorted(path.rglob("*.bvh")))
            elif path.suffix.lower() == ".bvh" and path.is_file():
                files.append(path)
        added, rejected = [], []
        for file in files:
            try:
                added.append(self._import_one(file, pack))
            except (MotionError, OSError, UnicodeError) as error:
                rejected.append({"path": str(file), "reason": str(error)})
        self.save()
        return {"added": added, "rejected": rejected}

    def _import_one(self, file: Path, pack: str) -> dict:
        motion = read_bvh(file)
        problems = check_soma_bones(motion.skeleton, REQUIRED)
        if problems:
            raise MotionError(" / ".join(problems))
        times = np.arange(0.0, motion.duration + 1e-6, 1.0 / MODEL_FPS)
        poses = sample(motion.poses, motion.fps, times)
        world = world_positions(poses, motion.skeleton.parents)
        duration = (poses.shape[0] - 1) / MODEL_FPS
        bars, usage = nominal_bars(duration)
        name = re.sub(r"[_-]+", " ", file.stem).strip().title()
        accents = motion_accents(world, motion.skeleton)
        item = {"id": uuid.uuid4().hex[:12], "name": name, "duration": round(duration, 3), "frames": poses.shape[0], "bars": bars,
                "usage": usage, "sourceBpm": round(240.0 * bars / duration, 1), **describe(world, motion.skeleton),
                "accents": accents, "thumb": thumbnail(world, motion.skeleton, accents), "tags": [], "description": "",
                "pack": pack, "provenance": {"origin": str(file), "owner": "", "license": "", "credit": ""}}
        folder = self.root / item["id"]
        folder.mkdir()
        shutil.copyfile(file, folder / "source.bvh")
        np.save(folder / "poses.npy", poses.astype(np.float32))
        self.items[item["id"]] = item
        self.skeleton = motion.skeleton
        return item

    def update(self, phrase_id: str, changes: dict) -> dict:
        item = self.items[phrase_id]
        if "category" in changes and changes["category"] not in CATEGORIES:
            raise MotionError("カテゴリが不正です")
        if "energy" in changes and not 1 <= int(changes["energy"]) <= 5:
            raise MotionError("強さは 1〜5 です")
        if "usage" in changes and changes["usage"] not in ("auto", "manual"):
            raise MotionError("用途が不正です")
        if "accents" in changes:
            values = sorted(float(v) for v in changes["accents"])
            if any(v <= 0.0 or v >= item["duration"] for v in values):
                raise MotionError(f"アクセントは 0〜{item['duration']:.2f} 秒の範囲で指定してください")
            changes = {**changes, "accents": [round(v, 3) for v in values]}
        for key in ("name", "category", "tags", "description", "energy", "usage", "provenance", "bars", "accents"):
            if key in changes:
                item[key] = changes[key]
        self.save()
        return item

    def delete(self, phrase_id: str) -> None:
        self.items.pop(phrase_id)
        self._poses.pop(phrase_id, None)
        shutil.rmtree(self.root / phrase_id, ignore_errors=True)
        self.save()

    def poses(self, phrase_id: str) -> np.ndarray:
        if phrase_id not in self._poses:
            self._poses[phrase_id] = np.load(self.root / phrase_id / "poses.npy").astype(np.float64)
        return self._poses[phrase_id]

    def get_skeleton(self) -> Skeleton:
        if self.skeleton is None:
            template = self.template_bvh()
            if template is None:
                raise MotionError("ライブラリが空です")
            self.skeleton = read_bvh(template).skeleton
        return self.skeleton

    def search_text(self, item: dict) -> str:
        from .search import impression_words

        parts = [
            item["name"],
            CATEGORIES.get(item["category"], ""),
            item.get("description", ""),
            " ".join(item.get("tags", [])),
            " ".join(impression_words(item)),
        ]
        return "。".join(part for part in parts if part)
