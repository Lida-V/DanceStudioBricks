"""Turn a clip list into one continuous motion: warped phrases joined by MotionBricks, baked and cached per join."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arrange import FAST_MOVE_SPEED, MIN_JOIN_SECONDS
from .grounding import plant_feet
from .library import Library
from .mb_client import MIN_TOKENS, Engine, EngineError, frames_from_tokens, tokens_for_frames
from .motion import MODEL_FPS, Skeleton, place, sample, world_positions
from .warp import Warp, build

MAX_SINGLE_FRAMES = frames_from_tokens(16)
MIN_SINGLE_FRAMES = frames_from_tokens(MIN_TOKENS)
MIN_VIA_FRAMES = 2 * MIN_SINGLE_FRAMES  # a join needs two generations to pass through a point on the way
JOIN_RECIPE = b"planted-feet-1"  # change this whenever joins come out differently, so cached ones are not reused

Built = tuple[dict, int, np.ndarray]  # a clip, its first timeline frame, and its poses at timeline frames
Via = tuple[float, float]


def clip_trim(clip: dict, item: dict) -> tuple[float, float]:
    """(source start, source end) seconds of `item` that `clip` plays."""
    full = float(item["duration"])
    start = max(0.0, float(clip.get("trimStart", 0.0)))
    end = min(full, float(clip.get("trimEnd", full)))
    return start, end


def warp_for(clip: dict, item: dict, sound: np.ndarray) -> Warp:
    trim_start, trim_end = clip_trim(clip, item)
    pins = [(pin["source"], clip["start"] + pin["at"]) for pin in clip.get("pins", [])]
    return build(clip["start"], clip["end"], trim_end, item.get("accents", []), sound, clip.get("match", 70) / 100.0, pins, trim_start)


def clip_frames(clip: dict, item: dict, poses: np.ndarray, skeleton: Skeleton, warp: Warp) -> tuple[int, np.ndarray]:
    """(first timeline frame, poses at timeline frames) with position, facing and time warp applied."""
    first = int(np.ceil(clip["start"] * MODEL_FPS - 1e-6))
    last = int(np.floor(clip["end"] * MODEL_FPS + 1e-6))
    placed = place(poses, skeleton, clip["x"], clip["z"], clip["facing"], warp.source_start)
    times = np.arange(first, last + 1) / MODEL_FPS
    return first, sample(placed, MODEL_FPS, warp.source_at(times))


class JoinCache:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def key(self, context: np.ndarray, target: np.ndarray, frames: int, via: Via | None = None) -> str:
        digest = hashlib.sha1(JOIN_RECIPE)
        digest.update(np.round(context, 4).tobytes())
        digest.update(np.round(target, 4).tobytes())
        digest.update(str(frames).encode())
        if via is not None:
            digest.update(f"via {via[0]:.2f} {via[1]:.2f}".encode())
        return digest.hexdigest()[:20]

    def get(self, key: str) -> np.ndarray | None:
        path = self.folder / f"{key}.npy"
        return np.load(path).astype(np.float64) if path.is_file() else None

    def put(self, key: str, poses: np.ndarray) -> None:
        np.save(self.folder / f"{key}.npy", poses.astype(np.float32))


@dataclass
class JoinJob:
    """One join that still has to be generated."""
    key: str
    context: np.ndarray
    target: np.ndarray
    frames: int
    via: Via | None = None


class BakeMemory:
    """What the last bake worked out, so the next one only recomputes what the edit touched."""

    def __init__(self) -> None:
        self.clips: dict[tuple, tuple[int, np.ndarray, dict]] = {}
        self.quality: dict[str, tuple[float, float]] = {}


def _route(begin: np.ndarray, finish: np.ndarray, frames: int, via: Via | None) -> list[tuple[float, np.ndarray]]:
    """One (frames, floor point the hips reach) per generation. A via point makes two legs, and a long leg is cut into equal pieces."""
    points = [begin, finish] if via is None else [begin, np.array(via, dtype=np.float64), finish]
    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(points[:-1], points[1:])]
    travelled = sum(lengths)
    stops: list[tuple[float, np.ndarray]] = []
    left = frames
    for number, (a, b, length) in enumerate(zip(points[:-1], points[1:], lengths)):
        legs_after = len(lengths) - 1 - number
        if legs_after:
            share = length / travelled if travelled > 1e-6 else 1.0 / len(lengths)
            leg = int(np.clip(round((frames - 1) * share) + 1, MIN_SINGLE_FRAMES, left - legs_after * (MIN_SINGLE_FRAMES - 1)))
        else:
            leg = left
        left -= leg - 1
        pieces = max(1, int(np.ceil((leg - 1) / (MAX_SINGLE_FRAMES - 1))))
        each = (leg - 1) / pieces + 1
        stops += [(each, a + (b - a) * piece / pieces) for piece in range(1, pieces + 1)]
    return stops


def _generate(engine: Engine, order: list[int], inverse: np.ndarray, context: np.ndarray, target: np.ndarray, frames: int,
              via: Via | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A join whose first frame is the last phrase pose and whose last frame is the next phrase pose, with the model's foot contacts."""
    begin, finish = context[-1, 0, [0, 2]].copy(), target[0, 0, [0, 2]].copy()
    stops = _route(begin, finish, frames, via)
    if len(stops) == 1:
        poses, contacts, _ = engine.generate(context[:, order], target[:, order], tokens_for_frames(frames))
        return poses[:, inverse], contacts
    # Long gaps and detours: walk through intermediate targets that carry the next pose along the path.
    chain, touching = [], []
    current = context
    for number, (each, point) in enumerate(stops, 1):
        waypoint = target.copy()
        waypoint[:, 0, 0] += point[0] - finish[0]
        waypoint[:, 0, 2] += point[1] - finish[1]
        poses, contacts, _ = engine.generate(current[:, order], (waypoint if number == len(stops) else waypoint[:1])[:, order], tokens_for_frames(each))
        poses = poses[:, inverse]
        chain.append(poses if not chain else poses[1:])
        touching.append(contacts if not touching else contacts[1:])
        current = poses[-4:]
    return np.concatenate(chain, axis=0), np.concatenate(touching, axis=0)


def generate_join(engine: Engine, skeleton: Skeleton, cache: JoinCache, job: JoinJob) -> np.ndarray:
    engine.start()
    order = [skeleton.index(name) for name in engine.joints]
    join, contacts = _generate(engine, order, np.argsort(order), job.context, job.target, job.frames, job.via)
    join = plant_feet(join, contacts, skeleton)
    cache.put(job.key, join)
    return join


def _blend(a: np.ndarray, b: np.ndarray, frames: int) -> np.ndarray:
    """Fallback for gaps too short to generate: a straight interpolation so preview stays continuous."""
    pair = np.stack([a, b])
    return sample(pair, 1.0, np.linspace(0.0, 1.0, frames))


def _clip_key(clip: dict, item: dict) -> tuple:
    pins = tuple((pin["source"], pin["at"]) for pin in clip.get("pins", []))
    return (clip["phraseId"], clip["start"], clip["end"], clip["x"], clip["z"], clip["facing"], clip.get("match", 70), pins,
            clip_trim(clip, item), item["duration"], tuple(item.get("accents", [])))


def bake_clip(clip: dict, library: Library, sound: np.ndarray) -> tuple[int, np.ndarray, dict]:
    """(first timeline frame, poses, report) for one phrase on the timeline."""
    item = library.items[clip["phraseId"]]
    warp = warp_for(clip, item, sound)
    first, frames = clip_frames(clip, item, library.poses(clip["phraseId"]), library.get_skeleton(), warp)
    low, high = warp.local_speed_range()
    matched = [m for m in warp.matches if m["sound"] is not None]
    return first, frames, {"speed": round(warp.speed, 3), "localSpeed": [round(low, 3), round(high, 3)], "accents": warp.matches,
                           "matched": sum(1 for m in matched if m["matched"]), "candidates": len(warp.matches), "curve": warp.curve()}


def sound_times(song: dict | None) -> np.ndarray:
    return np.array([a["t"] for a in song.get("accents", [])]) if song else np.zeros(0)


def lay_clips(clips: list[dict], library: Library, song: dict, memory: BakeMemory) -> tuple[list[Built], dict[str, dict]]:
    """Every phrase baked onto the timeline, in order, and the report for each. Unchanged phrases come from `memory`."""
    sound = sound_times(song)
    built: list[Built] = []
    report: dict[str, dict] = {}
    remembered: dict[tuple, tuple[int, np.ndarray, dict]] = {}
    for clip in sorted(clips, key=lambda c: c["start"]):
        key = _clip_key(clip, library.items[clip["phraseId"]])
        baked = memory.clips.get(key) or bake_clip(clip, library, sound)
        remembered[key] = baked
        built.append((clip, baked[0], baked[1]))
        report[clip["id"]] = baked[2]
    memory.clips = remembered
    return built, report


def gap_between(first_a: int, frames_a: np.ndarray, first_b: int, frames_b: np.ndarray) -> dict:
    """The space a join has to fill. It includes both endpoint poses."""
    last_a = first_a + frames_a.shape[0] - 1
    seconds = (first_b - last_a) / MODEL_FPS
    distance = float(np.hypot(frames_b[0, 0, 0] - frames_a[-1, 0, 0], frames_b[0, 0, 2] - frames_a[-1, 0, 2]))
    return {"last": last_a, "frames": first_b - last_a + 1, "seconds": seconds, "distance": distance, "pace": distance / max(seconds, 1e-3)}


def inspect_join(join: np.ndarray, skeleton: Skeleton) -> tuple[float, float]:
    """(fastest joint in m/s, lowest joint height in m): the numbers a join is judged by."""
    world = world_positions(join, skeleton.parents)
    return round(float((np.linalg.norm(np.diff(world, axis=0), axis=-1) * MODEL_FPS).max()), 2), round(float(world[:, :, 1].min()), 3)


def compose(clips: list[dict], library: Library, song: dict, engine: Engine, cache: JoinCache, memory: BakeMemory | None = None,
            defer: bool = False, engine_error: str | None = None, vias: dict[tuple[str, str], Via] | None = None) -> dict:
    """Bake the whole timeline. With `defer`, joins not in the cache are bridged by a straight blend and returned as jobs instead of being generated.

    `vias` gives a floor point to pass through for the join between two clip ids, so dancers can walk around each other.
    """
    memory = memory or BakeMemory()
    vias = vias or {}
    skeleton = library.get_skeleton()
    total = int(np.ceil(song["duration"] * MODEL_FPS)) + 1
    joints = len(skeleton.names)
    timeline = np.zeros((total, joints, 7))
    filled = np.zeros(total, dtype=bool)
    joins: list[dict] = []
    jobs: list[JoinJob] = []
    quality_memory: dict[str, tuple[float, float]] = {}
    built, report = lay_clips(clips, library, song, memory)
    for _, first, frames in built:
        end = min(total, first + frames.shape[0])
        if first < total:
            timeline[first:end] = frames[:end - first]
            filled[first:end] = True
    generated = reused = 0
    for (clip_a, first_a, frames_a), (clip_b, first_b, frames_b) in zip(built[:-1], built[1:]):
        gap = gap_between(first_a, frames_a, first_b, frames_b)
        last_a, gap_frames, seconds = gap["last"], gap["frames"], gap["seconds"]
        entry = {"from": clip_a["id"], "to": clip_b["id"], "start": round(last_a / MODEL_FPS, 3), "end": round(first_b / MODEL_FPS, 3),
                 "seconds": round(seconds, 3), "distance": round(gap["distance"], 2), "speed": round(gap["pace"], 2)}
        if gap_frames < 3:
            seamless = (gap_frames >= 1 and clip_a["phraseId"] == clip_b["phraseId"]
                        and float(np.linalg.norm(frames_b[0, 0, :3] - frames_a[-1, 0, :3])) < 0.15)
            entry["status"] = "cut" if seamless else "overlap"
            joins.append(entry)
            continue
        context, target = frames_a[-4:], frames_b[:4]
        via = vias.get((clip_a["id"], clip_b["id"])) if gap_frames >= MIN_VIA_FRAMES else None
        if via is not None:
            entry["via"] = [round(via[0], 2), round(via[1], 2)]
        key = cache.key(context, target, gap_frames, via)
        join = None
        if seconds < MIN_JOIN_SECONDS - 1e-3:
            join = _blend(frames_a[-1], frames_b[0], gap_frames)
            entry["status"] = "short"
        else:
            join = cache.get(key)
            if join is not None:
                reused += 1
            elif defer and engine_error is None:
                jobs.append(JoinJob(key, context, target, gap_frames, via))
                entry["status"] = "pending"
                join = _blend(frames_a[-1], frames_b[0], gap_frames)
            elif engine_error is None:
                try:
                    join = generate_join(engine, skeleton, cache, JoinJob(key, context, target, gap_frames, via))
                    generated += 1
                except EngineError as error:
                    engine_error = str(error)
            if join is None:
                join = _blend(frames_a[-1], frames_b[0], gap_frames)
                entry["status"] = "engine-error"
            elif "status" not in entry:
                entry["status"] = "fast" if gap["pace"] > FAST_MOVE_SPEED else "ok"
        resampled = sample(join, 1.0, np.linspace(0.0, join.shape[0] - 1, gap_frames)) if join.shape[0] != gap_frames else join
        if entry["status"] in ("ok", "fast"):
            quality = memory.quality.get(key) or inspect_join(resampled, skeleton)
            quality_memory[key] = quality
            entry["maxJointSpeed"], entry["minHeight"] = quality
        start, stop = last_a, min(total, first_b + 1)
        timeline[start:stop] = resampled[:stop - start]
        filled[start:stop] = True
        joins.append(entry)
    # Empty song parts hold the nearest placed pose so preview and export stay continuous.
    if filled.any():
        known = np.flatnonzero(filled)
        nearest = known[np.searchsorted(known, np.arange(total)).clip(0, len(known) - 1)]
        timeline[~filled] = timeline[nearest[~filled]]
    memory.quality = quality_memory
    return {"poses": timeline, "clips": report, "joins": joins, "jobs": jobs, "generated": generated, "reused": reused, "engineError": engine_error}


def measure_join(clip_a: dict, clip_b: dict, library: Library, song: dict, engine: Engine, cache: JoinCache) -> dict:
    """Make (or fetch) the join between two phrases as they are given, and say how it came out. Nothing on the timeline changes."""
    sound = sound_times(song)
    skeleton = library.get_skeleton()
    first_a, frames_a, _ = bake_clip(clip_a, library, sound)
    first_b, frames_b, _ = bake_clip(clip_b, library, sound)
    gap = gap_between(first_a, frames_a, first_b, frames_b)
    context, target = frames_a[-4:], frames_b[:4]
    key = cache.key(context, target, gap["frames"])
    join = cache.get(key)
    if join is None:
        join = generate_join(engine, skeleton, cache, JoinJob(key, context, target, gap["frames"]))
    if join.shape[0] != gap["frames"]:
        join = sample(join, 1.0, np.linspace(0.0, join.shape[0] - 1, gap["frames"]))
    fastest, lowest = inspect_join(join, skeleton)
    return {"seconds": round(gap["seconds"], 3), "distance": round(gap["distance"], 2), "moveSpeed": round(gap["pace"], 2), "maxJointSpeed": fastest, "minHeight": lowest}
