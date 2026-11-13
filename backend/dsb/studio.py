"""Application state: one project, one library, one baked timeline. All methods are safe to call from any thread."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .arrange import AUTO_SPEED_RANGE, DEFAULT_FLOW, FAST_MOVE_SPEED, MIN_JOIN_SECONDS, STAGE_X, STAGE_Z, TARGET_MOVE_SPEED, arrange, base_stem, candidates_for, clean_filter_tags, fill_gap_clips, filter_phrases_by_tags, has_ending_pose, join_seconds
from .compose import BakeMemory, JoinCache, clip_trim, compose, generate_join, lay_clips, measure_join, sound_times, warp_for
from .formation import close_calls, plan_vias
from .library import CATEGORIES, Library, seed_library
from .components import python_of, usable
from .mb_client import Engine, EngineError
from .motion import MODEL_FPS, MotionError, pose_spot, write_bvh, world_positions
from .retarget import HUMANOID, humanoid_turns, incremental_turns, write_vrma
from .retarget import REST_KINDS, rebased, rest_pose
from .vmd import PmxError, read_pmx_bones, write_vmd
from .warp import usable_pins

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DSB_DATA", ROOT / "data"))
SKIP_JOINTS = ("Thumb", "Index", "Middle", "Ring", "Pinky", "End", "Jaw", "Eye", "Nose", "Ear", "Tongue", "Lip")
MIN_CLIP_SECONDS = 0.3
MAX_HISTORY = 100
MAX_INSERT_SPEED = 2.0
BEAT_SPEED_ALLOWANCE = 1.05  # landing on a beat may cost this much of the top speed when phrases are squeezed
PROGRESS_SECONDS = 0.4
MAX_PINS = 32
OPTION_SPEED_RANGE = (0.5, MAX_INSERT_SPEED)  # join lengths that would play the phrase before them outside this are not offered
MAX_DANCERS = 12
NEW_DANCER_STEP = 1.5  # metres to the side of the dancer a new one is copied from
STASHED = ("dancers", "active", "formations", "flow", "seed", "match", "tempoFactor", "arrangeFilter")  # what is kept per song
EXPORT_KINDS = ("bvh", "fbx", "vmd", "vrma")
MODEL_KINDS = (".vrm", ".glb", ".fbx")


class StudioError(ValueError):
    pass


def _shorten_to_fit(lengths: list[float], durations: list[float], room: float) -> list[float] | None:
    """Lengths adding up to `room`. The slowest phrases give up time first, so the fastest ends as slow as it can. None if that takes more than MAX_INSERT_SPEED."""
    def at(speed: float) -> list[float]:
        return [min(length, max(MIN_CLIP_SECONDS, duration / speed)) for length, duration in zip(lengths, durations)]

    if sum(lengths) <= room + 1e-6:
        return lengths
    if sum(at(MAX_INSERT_SPEED)) > room + 1e-6:
        return None
    slow, fast = 1e-3, MAX_INSERT_SPEED
    for _ in range(50):
        middle = (slow + fast) / 2.0
        if sum(at(middle)) > room:
            slow = middle
        else:
            fast = middle
    return at(fast)


def carried_pins(pins: list[dict], old_start: float, old_end: float, start: float, end: float, duration: float,
                 source_start: float = 0.0) -> list[dict]:
    """Accent pins after a phrase is retimed. A move carries them along; a stretch leaves each on its timeline time. Ones that no longer fit are dropped."""
    if not pins:
        return []
    shift = 0.0 if abs((end - start) - (old_end - old_start)) < 2e-3 else old_start - start
    moved = [(pin["source"], start + pin["at"] + shift) for pin in pins]
    return [{"source": round(source, 3), "at": round(when - start, 3)} for source, when in usable_pins(moved, start, end, duration, source_start)]


def _clean_trim(clip: dict, item: dict) -> tuple[float, float]:
    """Validate and normalise `trimStart`/`trimEnd` on `clip`, stripping keys when the whole phrase is used."""
    full = float(item["duration"])
    try:
        trim_start = 0.0 if clip.get("trimStart") is None else round(float(clip["trimStart"]), 3)
        trim_end = round(full, 3) if clip.get("trimEnd") is None else round(float(clip["trimEnd"]), 3)
    except (TypeError, ValueError) as error:
        raise StudioError("使う範囲の指定が不正です") from error
    if trim_start < -1e-3 or trim_end > full + 1e-2:
        raise StudioError(f"使う範囲は 0〜{full:.2f} 秒の内側にしてください")
    trim_start, trim_end = max(0.0, trim_start), min(full, trim_end)
    if trim_end - trim_start < MIN_CLIP_SECONDS - 1e-4:
        raise StudioError("カット後のフレーズが短すぎます")
    if trim_start <= 1e-3 and trim_end >= full - 1e-3:
        clip.pop("trimStart", None)
        clip.pop("trimEnd", None)
        return 0.0, full
    clip["trimStart"], clip["trimEnd"] = round(trim_start, 3), round(trim_end, 3)
    return clip["trimStart"], clip["trimEnd"]


def _on_stage(x: float, z: float) -> tuple[float, float]:
    return round(float(min(max(x, STAGE_X[0]), STAGE_X[1])), 2), round(float(min(max(z, STAGE_Z[0]), STAGE_Z[1])), 2)


def _file_part(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text).strip("_") or "motion"


def _labels(sections: list[dict]) -> None:
    from .music import LABELS

    numbering: dict[str, int] = {}
    for section in sections:
        numbering[section["kind"]] = numbering.get(section["kind"], 0) + 1
        section["label"] = LABELS[section["kind"]] + (f" {numbering[section['kind']]}" if numbering[section["kind"]] > 1 else "")


class Studio:
    def __init__(self, data: Path = DATA, engine: Engine | None = None, background: bool = False, seed: Path | None = None) -> None:
        self.data = Path(data)
        for name in ("library", "songs", "cache/joins", "exports", "projects"):
            (self.data / name).mkdir(parents=True, exist_ok=True)
        if seed is not None:
            seed_library(self.data / "library", seed)
        self.library = Library(self.data / "library")
        self.engine = engine or Engine()
        self.cache = JoinCache(self.data / "cache" / "joins")
        self.lock = threading.RLock()
        self.project_file = self.data / "project.json"
        self.project = self._default_project()
        self.song: dict | None = None
        self.version = 0
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.background = background  # edits answer at once and MotionBricks runs on the worker thread
        self._baked: dict[str, dict] = {}  # per dancer: the finished motion, its preview and what the last bake remembered
        self._collisions: list[dict] = []
        self._jobs: list[tuple[str, object]] = []  # (dancer id, join still to generate)
        self._serial = 0  # counts edits, so the worker can tell its jobs went stale
        self._engine_error: str | None = None
        self._generated: dict[str, int] = {}
        self._edit_began = 0.0
        self._done = threading.Condition(self.lock)
        self._wake = threading.Event()
        self._closing = False
        if self.project_file.is_file():
            self.project = {**self._default_project(), **json.loads(self.project_file.read_text(encoding="utf-8"))}
            if not self.project["dancers"]:
                self.project["dancers"] = self._default_project()["dancers"]
            if self.project.get("songPath") and Path(self.project["songPath"]).is_file():
                cached = self._song_cache_file(self.project["songPath"], self.project.get("tempoFactor"))
                if cached.is_file():
                    self.song = json.loads(cached.read_text(encoding="utf-8"))
        if background:
            threading.Thread(target=self._worker, name="join-worker", daemon=True).start()

    @staticmethod
    def _default_project() -> dict:
        return {"version": 1, "songPath": None, "tempoFactor": None, "match": 70, "seed": 1, "snap": "beat",
                "flow": json.loads(json.dumps(DEFAULT_FLOW)), "dancers": [{"id": "d1", "name": "ダンサー 1", "clips": []}],
                "active": "d1", "formations": [], "models": {}, "arrangeFilter": []}

    # ---- persistence and helpers

    @property
    def dancers(self) -> list[dict]:
        return self.project["dancers"]

    @property
    def dancer(self) -> dict:
        """The dancer being edited. Every phrase edit goes to this one."""
        return next((d for d in self.dancers if d["id"] == self.project.get("active")), self.dancers[0])

    def _dancer(self, dancer_id: str | None) -> dict:
        if dancer_id is None:
            return self.dancer
        for dancer in self.dancers:
            if dancer["id"] == dancer_id:
                return dancer
        raise StudioError("そのダンサーは見つかりません")

    @property
    def clips(self) -> list[dict]:
        return self.dancer["clips"]

    @clips.setter
    def clips(self, value: list[dict]) -> None:
        self.dancer["clips"] = value

    @property
    def composed(self) -> dict | None:
        return self._baked.get(self.dancer["id"], {}).get("result")

    @property
    def preview(self) -> np.ndarray | None:
        return self._baked.get(self.dancer["id"], {}).get("preview")

    def _save(self) -> None:
        temp = self.project_file.with_suffix(".tmp")
        temp.write_text(json.dumps(self.project, ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(self.project_file)

    def _stash_file(self, path: str) -> Path:
        return self.data / "projects" / f"{hashlib.sha1(str(Path(path).resolve()).encode('utf-8')).hexdigest()[:20]}.json"

    def _stash(self) -> None:
        """Keep this song's arrangement, so loading the song again brings it back."""
        target = self._stash_file(self.project["songPath"])
        if target.is_file() or any(dancer["clips"] for dancer in self.dancers):
            target.write_text(json.dumps({key: self.project.get(key) for key in STASHED}, ensure_ascii=False), encoding="utf-8")

    def _song_cache_file(self, path: str, factor) -> Path:
        stat = Path(path).stat()
        key = hashlib.sha1(f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{factor}".encode("utf-8")).hexdigest()[:20]
        return self.data / "songs" / f"{key}.json"

    def _clip(self, clip_id: str) -> dict:
        for clip in self.clips:
            if clip["id"] == clip_id:
                return clip
        raise StudioError("そのフレーズは見つかりません")

    def _section_at(self, seconds: float) -> dict | None:
        for section in (self.song or {}).get("sections", []):
            if section["start"] <= seconds < section["end"]:
                return section
        return None

    # ---- responses

    def state(self) -> dict:
        with self.lock:
            if any(dancer["clips"] and self._baked.get(dancer["id"], {}).get("result") is None for dancer in self.dancers):
                self._recompose()  # after a restart the saved arrangement still needs baking
            return {"library": list(self.library.items.values()), "categories": CATEGORIES, "song": self._song_payload(), "previewJoints": self._preview_joints(),
                    "modelBones": list(HUMANOID), **self.changes()}

    def changes(self) -> dict:
        with self.lock:
            return {"project": self.project, "composed": self._composed_payload(), "cast": self._cast_payload(), "collisions": self._collisions,
                    "history": self._history()}

    def _history(self) -> dict:
        return {"undo": len(self.undo_stack), "redo": len(self.redo_stack)}

    # ---- undo and redo

    def _snapshot(self) -> dict:
        return copy.deepcopy({"dancers": self.dancers, "active": self.project.get("active"), "formations": self.project.get("formations", []),
                              "flow": self.project["flow"], "seed": self.project["seed"],
                              "arrangeFilter": list(self.project.get("arrangeFilter", [])),
                              "sections": self.song["sections"] if self.song else None})

    def _checkpoint(self) -> None:
        """Remember the arrangement just before a change."""
        self.undo_stack.append(self._snapshot())
        del self.undo_stack[:-MAX_HISTORY]
        self.redo_stack.clear()

    def _restore(self, snapshot: dict) -> None:
        self.project.update(dancers=snapshot["dancers"], active=snapshot["active"], formations=snapshot["formations"])
        self.project["flow"] = snapshot["flow"]
        self.project["seed"] = snapshot["seed"]
        self.project["arrangeFilter"] = list(snapshot.get("arrangeFilter", []))
        if self.song is not None and snapshot["sections"] is not None:
            self.song["sections"] = snapshot["sections"]
        self._save()
        self._recompose()

    def undo(self) -> dict:
        with self.lock:
            if self.undo_stack:
                self.redo_stack.append(self._snapshot())
                self._restore(self.undo_stack.pop())
            return self.state()

    def redo(self) -> dict:
        with self.lock:
            if self.redo_stack:
                self.undo_stack.append(self._snapshot())
                self._restore(self.redo_stack.pop())
            return self.state()

    def _song_payload(self) -> dict | None:
        if self.song is None:
            return None
        return {**self.song, "path": self.project["songPath"], "name": Path(self.project["songPath"]).name}

    def _composed_payload(self) -> dict | None:
        if self.composed is None:
            return None
        return {key: self.composed[key] for key in ("clips", "joins", "generated", "reused", "engineError", "seconds")} | {"version": self.version, "frames": self.preview.shape[0], "fps": MODEL_FPS}

    def _cast_payload(self) -> list[dict]:
        """Every dancer with the version of its motion, so the screen fetches only what changed."""
        cast = []
        for dancer in self.dancers:
            baked = self._baked.get(dancer["id"], {})
            cast.append({"id": dancer["id"], "name": dancer["name"], "clips": len(dancer["clips"]),
                         "version": baked.get("version", 0) if baked.get("result") is not None else 0})
        return cast

    def _preview_joints(self) -> dict | None:
        if not self.library.items:
            return None
        names = self.library.get_skeleton().names
        keep = [i for i, name in enumerate(names) if not any(token in name for token in SKIP_JOINTS)]
        return {"names": [names[i] for i in keep]}

    # ---- library

    def import_motion(self, paths: list[str]) -> dict:
        with self.lock:
            return self.library.import_paths(paths)

    def update_phrase(self, phrase_id: str, changes: dict) -> dict:
        with self.lock:
            if phrase_id not in self.library.items:
                raise StudioError("そのフレーズは見つかりません")
            try:
                item = self.library.update(phrase_id, changes)
            except (MotionError, TypeError) as error:
                raise StudioError(str(error)) from error
            if "accents" in changes and any(c["phraseId"] == phrase_id for dancer in self.dancers for c in dancer["clips"]):
                self._recompose(force=True)
            return item

    def delete_phrase(self, phrase_id: str) -> None:
        with self.lock:
            if any(c["phraseId"] == phrase_id for dancer in self.dancers for c in dancer["clips"]):
                raise StudioError("タイムラインで使用中のフレーズは削除できません")
            if phrase_id not in self.library.items:
                raise StudioError("そのフレーズは見つかりません")
            self.library.delete(phrase_id)

    def _cached_doc_vectors(self, items: list[dict], texts_fn) -> np.ndarray:
        cache_path = self.data / "library" / "embeddings.npz"
        cached: dict[str, np.ndarray] = {}
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as loaded:
                    keys = list(loaded["keys"])
                    vecs = loaded["vectors"]
                    cached = {str(k): vecs[idx] for idx, k in enumerate(keys)}
            except Exception:
                cached = {}

        doc_keys: list[str] = []
        missing_prompts: list[tuple[str, str]] = []
        missing_indices: list[int] = []
        for idx, item in enumerate(items):
            doc_text = self.library.search_text(item)
            digest = hashlib.sha1(doc_text.encode("utf-8")).hexdigest()[:16]
            key = f"{item['id']}:{digest}"
            doc_keys.append(key)
            if key not in cached:
                missing_prompts.append((doc_text, "Document"))
                missing_indices.append(idx)

        if missing_prompts:
            encoded = texts_fn(missing_prompts)
            for pos, idx in enumerate(missing_indices):
                cached[doc_keys[idx]] = encoded[pos]
            matrix = np.stack([cached[k] for k in doc_keys]).astype(np.float32)
            tmp_path = cache_path.with_name("embeddings.tmp.npz")
            try:
                np.savez(tmp_path, keys=np.array(doc_keys), vectors=matrix)
                tmp_path.replace(cache_path)
            except OSError:
                tmp_path.unlink(missing_ok=True)
            return matrix

        return np.stack([cached[k] for k in doc_keys]).astype(np.float32)

    def search(self, query: str, limit: int = 20) -> list[dict]:
        from .embed import EmbedUnavailable, texts
        from .search import feeling_scores

        with self.lock:
            items = list(self.library.items.values())
        if not items:
            return []
        feel = feeling_scores(query, items, CATEGORIES)
        feel_vec = np.array([feel.get(item["id"], 0.0) for item in items], dtype=np.float32)
        try:
            doc_vecs = self._cached_doc_vectors(items, texts)
            query_vec = texts([(query, "SearchQuery")])[0]
            sem_scores = doc_vecs @ query_vec
            scores = 0.45 * sem_scores + 0.55 * feel_vec
        except EmbedUnavailable as error:
            if float(feel_vec.max(initial=0.0)) > 0.0:
                order = [int(i) for i in np.argsort(-feel_vec) if feel_vec[i] > 0.0][:limit]
                return [{"id": items[i]["id"], "score": round(float(feel_vec[i]), 4)} for i in order]
            raise StudioError(f"意味検索を実行できません: {error}") from error
        order = np.argsort(-scores)[:limit]
        return [{"id": items[i]["id"], "score": round(float(scores[i]), 4)} for i in order]

    # ---- song

    def load_song(self, path: str, factor: float | None = None) -> dict:
        from .music import analyze

        if not Path(path).is_file():
            raise StudioError("音声ファイルが見つかりません")
        kept = None
        with self.lock:
            stash = self._stash_file(path)
            if self.project["songPath"] != path and stash.is_file():
                kept = json.loads(stash.read_text(encoding="utf-8"))
                if factor is None:
                    factor = kept.get("tempoFactor")  # the arrangement was made on these bars
        cached = self._song_cache_file(path, factor)
        if cached.is_file():
            song = json.loads(cached.read_text(encoding="utf-8"))
        else:
            song = analyze(path, factor)
            cached.write_text(json.dumps(song, ensure_ascii=False), encoding="utf-8")
        with self.lock:
            changed = self.project["songPath"] != path
            if changed and self.project["songPath"]:
                self._stash()
            self.song = song
            self._baked = {}
            self.undo_stack.clear()
            self.redo_stack.clear()
            if changed and kept:
                self.project.update({key: kept[key] for key in STASHED if key in kept and key != "tempoFactor"})
            elif changed:
                self.project["formations"] = []
                for dancer in self.dancers:
                    dancer["clips"] = []
            self.project["songPath"] = path
            self.project["tempoFactor"] = factor
            self._save()
            self._recompose()
            return {**self.state(), "restored": sum(len(dancer["clips"]) for dancer in self.dancers) if changed and kept else 0}

    def edit_sections(self, sections: list[dict]) -> dict:
        with self.lock:
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            bars = self.song["bars"] + [self.song["duration"]]
            cleaned = []
            for section in sorted(sections, key=lambda s: s["startBar"]):
                if section["kind"] not in DEFAULT_FLOW:
                    raise StudioError("区間の種類が不正です")
                start, end = int(section["startBar"]), int(section["endBar"])
                if not 0 <= start < end <= len(bars) - 1:
                    raise StudioError("区間の範囲が不正です")
                cleaned.append({"startBar": start, "endBar": end, "kind": section["kind"], "repeatOf": int(section.get("repeatOf", -1)),
                                "energy": float(section.get("energy", 0.0)), "start": round(float(bars[start]), 3), "end": round(float(bars[end]), 3)})
            _labels(cleaned)
            self._checkpoint()
            self.song["sections"] = cleaned
            cached = self._song_cache_file(self.project["songPath"], self.project.get("tempoFactor"))
            cached.write_text(json.dumps(self.song, ensure_ascii=False), encoding="utf-8")
            return self.state()

    # ---- arrangement

    def set_arrange_filter(self, tags: list[str] | None) -> dict:
        with self.lock:
            cleaned = clean_filter_tags(tags)
            if cleaned and self.library.items:
                matched = filter_phrases_by_tags(list(self.library.items.values()), cleaned)
                if not matched:
                    raise StudioError(f"指定されたジャンル・タグ（{'・'.join(cleaned)}）に一致するフレーズがありません")
            if cleaned != list(self.project.get("arrangeFilter") or []):
                self._checkpoint()
                self.project["arrangeFilter"] = cleaned
                self._save()
            return self.changes()

    def arrange(self, start: float | None = None, end: float | None = None, seed: int | None = None,
                tags: list[str] | None = None) -> dict:
        with self.lock:
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            effective_tags = clean_filter_tags(tags) if tags is not None else list(self.project.get("arrangeFilter") or [])
            span = (float(start), float(end)) if start is not None and end is not None else None
            before = self._snapshot()
            chosen_seed = int(seed) if seed is not None else int(self.project["seed"])
            try:
                new_clips = arrange(self.song, list(self.library.items.values()), self.clips, self.project["flow"],
                                    self.project["match"], chosen_seed, span, filter_tags=effective_tags)
            except ValueError as error:
                raise StudioError(str(error)) from error
            self.project["seed"] = chosen_seed
            if tags is not None:
                self.project["arrangeFilter"] = effective_tags
            self.clips = new_clips
            self.undo_stack.append(before)
            del self.undo_stack[:-MAX_HISTORY]
            self.redo_stack.clear()
            self._save()
            self._recompose()
            return self.changes()

    def shuffle(self, tags: list[str] | None = None) -> dict:
        with self.lock:
            return self.arrange(seed=self.project["seed"] + 1, tags=tags)

    def fill_gap(self, start: float, end: float, seed: int | None = None, tags: list[str] | None = None) -> dict:
        """Assign multiple musically matched phrases into an empty/interpolation interval `[start, end]` without moving existing clips."""
        with self.lock:
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            chosen_seed = int(seed) if seed is not None else int(self.project["seed"]) + len(self.clips) + 1
            effective_tags = clean_filter_tags(tags) if tags is not None else list(self.project.get("arrangeFilter") or [])
            try:
                created = fill_gap_clips(
                    self.song,
                    list(self.library.items.values()),
                    self.clips,
                    self.project["flow"],
                    self.project["match"],
                    chosen_seed,
                    float(start),
                    float(end),
                    filter_tags=effective_tags,
                )
            except ValueError as error:
                raise StudioError(str(error)) from error
            for clip in created:
                item = self.library.items.get(clip["phraseId"])
                if item is not None:
                    _clean_trim(clip, item)
            self._checkpoint()
            self.clips.extend(created)
            self.clips.sort(key=lambda c: c["start"])
            self._save()
            self._recompose()
            return {**self.changes(), "added": [c["id"] for c in created], "count": len(created)}

    def _trim(self, clip: dict) -> tuple[float, float]:
        return clip_trim(clip, self.library.items[clip["phraseId"]])

    def _source_span(self, clip: dict) -> float:
        trim_start, trim_end = self._trim(clip)
        return trim_end - trim_start

    def _spot_at(self, phrase_id: str, x: float, z: float, facing: float, old_trim_start: float, new_trim_start: float) -> tuple[float, float, float]:
        if abs(new_trim_start - old_trim_start) <= 1e-4:
            return x, z, facing
        if not ((self.library.root / phrase_id / "poses.npy").is_file() or phrase_id in self.library._poses):
            return x, z, facing
        return pose_spot(self.library.poses(phrase_id), self.library.get_skeleton(), x, z, facing, old_trim_start, new_trim_start)

    def update_clip(self, clip_id: str, changes: dict) -> dict:
        with self.lock:
            clip = self._clip(clip_id)
            allowed = {"start", "end", "x", "z", "facing", "locked", "match", "phraseId", "speedMode", "pins", "trimStart", "trimEnd"}
            if clip.get("locked") and any(k in allowed and k != "locked" and v != clip.get(k, [] if k == "pins" else None) for k, v in changes.items()):
                raise StudioError("ロック中のフレーズは変更できません。先にロックを解除してください")
            updated = {**clip, **{k: v for k, v in changes.items() if k in allowed}}
            if updated["phraseId"] not in self.library.items:
                raise StudioError("そのフレーズは見つかりません")
            item = self.library.items[updated["phraseId"]]
            phrase_changed = updated["phraseId"] != clip["phraseId"]
            if phrase_changed and "trimStart" not in changes and "trimEnd" not in changes:
                updated.pop("trimStart", None)
                updated.pop("trimEnd", None)
            old_trim_start, _ = self._trim(clip)
            trim_start, trim_end = _clean_trim(updated, item)
            if not phrase_changed and abs(trim_start - old_trim_start) > 1e-4 and not any(k in changes for k in ("x", "z", "facing")):
                updated["x"], updated["z"], updated["facing"] = self._spot_at(updated["phraseId"], updated["x"], updated["z"],
                                                                              updated["facing"], old_trim_start, trim_start)
            if changes.get("speedMode") == "auto":
                updated["end"] = self._auto_end(updated)
            if updated["end"] - updated["start"] < MIN_CLIP_SECONDS:
                raise StudioError("フレーズが短すぎます")
            if not 0 <= updated["match"] <= 100:
                raise StudioError("音はめの強さは 0〜100 です")
            if "pins" in changes:
                pins = self._checked_pins(changes["pins"], updated, trim_end, trim_start)
            elif phrase_changed:
                pins = []  # they marked moments of the phrase that was here
            else:
                pins = carried_pins(clip.get("pins", []), clip["start"], clip["end"], updated["start"], updated["end"], trim_end, trim_start)
            if pins or "pins" in clip:
                updated["pins"] = pins
            moved = updated["x"] != clip["x"] or updated["z"] != clip["z"]
            self._checkpoint()
            clip.update(updated)
            for key in ("trimStart", "trimEnd"):
                if key not in updated:
                    clip.pop(key, None)
            self.clips.sort(key=lambda c: c["start"])
            if phrase_changed:
                earlier = [c for c in self.clips if c["start"] < clip["start"]]
                prev = earlier[-1] if earlier else None
                if prev and not prev.get("locked") and clip["start"] - prev["end"] < MIN_JOIN_SECONDS - 1e-3:
                    latest_prev = round(clip["start"] - self._gap_needed(prev, clip, tight=True), 3)
                    if latest_prev - prev["start"] >= max(MIN_CLIP_SECONDS, self._source_span(prev) / MAX_INSERT_SPEED) - 1e-3:
                        prev["speedMode"] = "manual"
                        self._retime(prev, prev["start"], latest_prev)
            if moved:
                # The joins into and out of this clip changed length needs; clips on automatic timing follow.
                earlier = [c for c in self.clips if c["start"] < clip["start"]]
                for neighbour in ([earlier[-1]] if earlier else []) + [clip]:
                    if neighbour.get("speedMode") == "auto" and not neighbour.get("locked"):
                        self._retime(neighbour, neighbour["start"], self._auto_end(neighbour))
            self._save()
            self._recompose()
            return self.changes()

    def align_clip(self, clip_id: str, facing: bool = False) -> dict:
        """Stand a phrase where the phrase before it ends, so the join between them has no distance to cover.

        The hips' floor position at the start of this phrase goes to the hips' floor position at the end of the one before.
        With `facing`, the heading is matched too.
        """
        with self.lock:
            clip = self._clip(clip_id)
            earlier = [c for c in self.clips if c["start"] < clip["start"]]
            if not earlier:
                raise StudioError("前にフレーズがありません")
            prev = earlier[-1]
            trim_start, trim_end = self._trim(prev)
            x, z, heading_at_end = pose_spot(self.library.poses(prev["phraseId"]), self.library.get_skeleton(), prev["x"], prev["z"],
                                             prev["facing"], trim_start, trim_end)
            changes: dict = {"x": round(x, 3), "z": round(z, 3)}
            if facing:
                changes["facing"] = heading_at_end
            return self.update_clip(clip_id, changes)

    def cut_clip(self, clip_id: str, at: float, mode: str = "split") -> dict:
        """Cut a timeline phrase at timeline time `at`: split into two (`split`), drop before `at` (`before`), or drop after `at` (`after`)."""
        with self.lock:
            if mode not in ("split", "before", "after"):
                raise StudioError("カットの種類が不正です")
            clip = self._clip(clip_id)
            if clip.get("locked"):
                raise StudioError("ロック中のフレーズはカットできません。先にロックを解除してください")
            cut_time = round(float(at), 3)
            if cut_time - clip["start"] < MIN_CLIP_SECONDS - 1e-4 or clip["end"] - cut_time < MIN_CLIP_SECONDS - 1e-4:
                raise StudioError("端から 0.3 秒以上内側の位置でカットしてください")
            item = self.library.items[clip["phraseId"]]
            trim_start, trim_end = self._trim(clip)
            warp = warp_for(clip, item, sound_times(self.song))
            cut_source = round(float(warp.source_at(np.array([cut_time]))[0]), 3)
            if mode in ("split", "after") and cut_source - trim_start < MIN_CLIP_SECONDS - 1e-4:
                raise StudioError("カット後の前半フレーズが短すぎます")
            if mode in ("split", "before") and trim_end - cut_source < MIN_CLIP_SECONDS - 1e-4:
                raise StudioError("カット後の後半フレーズが短すぎます")
            self._checkpoint()
            old_start, old_end = clip["start"], clip["end"]
            old_pins = list(clip.get("pins", []))
            if mode == "after":
                clip.update(end=cut_time, speedMode="manual", trimStart=trim_start, trimEnd=cut_source)
                _clean_trim(clip, item)
                if old_pins:
                    clip["pins"] = carried_pins(old_pins, old_start, old_end, clip["start"], clip["end"], cut_source, trim_start)
                selected = clip["id"]
            elif mode == "before":
                nx, nz, nfacing = self._spot_at(clip["phraseId"], clip["x"], clip["z"], clip["facing"], trim_start, cut_source)
                clip.update(start=cut_time, x=nx, z=nz, facing=nfacing, speedMode="manual", trimStart=cut_source, trimEnd=trim_end)
                _clean_trim(clip, item)
                if old_pins:
                    clip["pins"] = carried_pins(old_pins, old_start, old_end, clip["start"], clip["end"], trim_end, cut_source)
                selected = clip["id"]
            else:
                nx, nz, nfacing = self._spot_at(clip["phraseId"], clip["x"], clip["z"], clip["facing"], trim_start, cut_source)
                second = copy.deepcopy(clip)
                second.update(id=uuid.uuid4().hex[:10], start=cut_time, end=old_end, x=nx, z=nz, facing=nfacing,
                              speedMode="manual", trimStart=cut_source, trimEnd=trim_end)
                _clean_trim(second, item)
                if old_pins:
                    second["pins"] = carried_pins(old_pins, old_start, old_end, second["start"], second["end"], trim_end, cut_source)
                clip.update(end=cut_time, speedMode="manual", trimStart=trim_start, trimEnd=cut_source)
                _clean_trim(clip, item)
                if old_pins:
                    clip["pins"] = carried_pins(old_pins, old_start, old_end, clip["start"], clip["end"], cut_source, trim_start)
                self.clips.append(second)
                self.clips.sort(key=lambda c: c["start"])
                selected = second["id"]
            self._save()
            self._recompose()
            return {**self.changes(), "selected": selected}

    @staticmethod
    def _checked_pins(pins, clip: dict, duration: float, source_start: float = 0.0) -> list[dict]:
        """Accent pins as asked for, or the reason they cannot all hold."""
        try:
            wanted = [(float(pin["source"]), clip["start"] + float(pin["at"])) for pin in pins]
        except (KeyError, TypeError, ValueError) as error:
            raise StudioError("アクセントの指定が不正です") from error
        if len(wanted) > MAX_PINS:
            raise StudioError(f"手動で合わせられるアクセントは 1 フレーズ {MAX_PINS} 個までです")
        kept = usable_pins(wanted, clip["start"], clip["end"], duration, source_start)
        if len(kept) != len(wanted):
            raise StudioError("そのアクセントはそこへ動かせません。フレーズの内側で、前後のアクセントの間に置いてください")
        return [{"source": round(source, 3), "at": round(when - clip["start"], 3)} for source, when in kept]

    def _retime(self, clip: dict, start: float, end: float) -> None:
        """Give a phrase new times. Its accent pins follow."""
        if clip.get("pins"):
            trim_start, trim_end = self._trim(clip)
            clip["pins"] = carried_pins(clip["pins"], clip["start"], clip["end"], start, end, trim_end, trim_start)
        clip["start"], clip["end"] = start, end

    def _auto_end(self, clip: dict, clips: list[dict] | None = None) -> float:
        """Where this clip ends at automatic speed: leaving room for the join to the next clip, inside the speed range."""
        span = self._source_span(clip)
        following = [c for c in (self.clips if clips is None else clips) if c["start"] > clip["start"]]
        natural = clip["start"] + span
        if not following:
            if self.song and "duration" in self.song and self.song["duration"] - clip["start"] >= MIN_CLIP_SECONDS:
                natural = min(natural, self.song["duration"])
            return round(natural, 3)
        after = min(following, key=lambda c: c["start"])
        distance = float(np.hypot(after["x"] - clip["x"], after["z"] - clip["z"]))
        latest = after["start"] - join_seconds(distance, self.song["barSeconds"])
        shortest = clip["start"] + span / AUTO_SPEED_RANGE[1]
        longest = clip["start"] + span / AUTO_SPEED_RANGE[0]
        return round(float(min(max(shortest, min(latest, longest)), after["start"] - 0.02)), 3)

    def set_flow(self, flow: dict) -> dict:
        with self.lock:
            for kind, rule in flow.items():
                if kind not in DEFAULT_FLOW or not set(rule) >= {"categories", "energy", "move", "text"}:
                    raise StudioError("流れの辞書の内容が不正です")
                if not all(c in CATEGORIES for c in rule["categories"]) or not 1 <= rule["energy"] <= 5 or not 0 <= rule["move"] <= 5:
                    raise StudioError("流れの辞書の値が範囲外です")
            self._checkpoint()
            self.project["flow"] = flow
            self._save()
            return self.changes()

    def _insert_point(self, seconds: float) -> tuple[int, float]:
        """Where a new phrase goes for a request at `seconds`: never on top of a phrase, and a sliver of a gap means the next phrase's start. A locked phrase keeps its start."""
        ordered = self.clips
        index = sum(1 for clip in ordered if clip["start"] <= seconds)
        if index and seconds < ordered[index - 1]["end"]:
            inside = ordered[index - 1]
            if seconds - inside["start"] >= (inside["end"] - inside["start"]) / 2:
                seconds = inside["end"]
            elif inside.get("locked"):
                index -= 1
                seconds = ordered[index - 1]["end"] if index else 0.0
            else:
                return index - 1, inside["start"]
        if index < len(ordered) and not ordered[index].get("locked") and ordered[index]["start"] - seconds < self.song["barSeconds"]:
            return index, ordered[index]["start"]
        earliest = ordered[index - 1]["end"] + MIN_JOIN_SECONDS if index else 0.0
        if seconds >= earliest:
            return index, seconds
        return index, next((beat for beat in self.song.get("beats", []) if beat >= earliest - 1e-3), earliest)  # pushed along, so land on a beat

    def _exact_point(self, seconds: float, place: tuple[float, float] | None) -> tuple[int, float, list[tuple[dict, float, float]]]:
        """Place a dropped phrase right at `seconds`. An unlocked phrase covering `seconds` shortens if it fits, or slides after."""
        ordered = self.clips
        begin = round(max(0.0, float(seconds)), 3)
        index = sum(1 for clip in ordered if clip["start"] <= begin)
        pre_plan: list[tuple[dict, float, float]] = []
        while index > 0:
            prev = ordered[index - 1]
            spot = place if place is not None else (prev["x"], prev["z"])
            gap = max(float(np.hypot(spot[0] - prev["x"], spot[1] - prev["z"])) / TARGET_MOVE_SPEED, MIN_JOIN_SECONDS)
            if prev["end"] + gap <= begin + 1e-3:
                break
            if prev.get("locked"):
                if begin < (prev["start"] + prev["end"]) / 2 and (index == 1 or ordered[index - 2]["end"] + MIN_JOIN_SECONDS <= prev["start"] - MIN_CLIP_SECONDS):
                    index -= 1
                    begin = round(ordered[index - 1]["end"] + MIN_JOIN_SECONDS, 3) if index else 0.0
                    continue
                begin = round(prev["end"] + gap, 3)
                break
            latest_end = round(begin - gap, 3)
            min_len = max(MIN_CLIP_SECONDS, self._source_span(prev) / MAX_INSERT_SPEED)
            if latest_end - prev["start"] >= min_len - 1e-3:
                pre_plan.append((prev, prev["start"], latest_end))
                break
            index -= 1
        return index, begin, pre_plan

    def _gap_needed(self, a: dict, b: dict, tight: bool = False) -> float:
        distance = float(np.hypot(b["x"] - a["x"], b["z"] - a["z"]))
        return max(distance / TARGET_MOVE_SPEED, MIN_JOIN_SECONDS) if tight else join_seconds(distance, self.song["barSeconds"])

    def _room_plan(self, index: int, new: dict) -> list[tuple[dict, float, float]]:
        """New (clip, start, end) times that clear the way for `new` at `index`. Nothing changes yet, so a refusal leaves the timeline alone.

        Later phrases slide back by whole bars up to the first gap that swallows the shift. A locked phrase in the way stays put, and the free ones shorten instead."""
        ordered = self.clips
        if index >= len(ordered):
            return []
        needed = new["end"] + self._gap_needed(new, ordered[index]) - ordered[index]["start"]
        if needed <= 1e-3:
            return []
        bar = self.song["barSeconds"]
        shift = math.ceil(needed / bar - 1e-6) * bar  # whole bars keep every later phrase on its beat
        pushed = []
        for position in range(index, len(ordered)):
            clip = ordered[position]
            if clip.get("locked"):
                return self._squeeze_before(position, index, new)
            pushed.append(clip)
            if position + 1 < len(ordered) and ordered[position + 1]["start"] >= clip["end"] + shift + self._gap_needed(clip, ordered[position + 1]):
                break
        return [(clip, round(clip["start"] + shift, 3), round(clip["end"] + shift, 3)) for clip in pushed]

    def _squeeze_before(self, barrier: int, index: int, new: dict) -> list[tuple[dict, float, float]]:
        """Fit `new` and the free phrases after it in front of the locked phrase at `barrier`, reaching further back only until the speeds are comfortable."""
        ordered = self.clips
        wall = ordered[barrier]
        beats = self.song.get("beats", [])
        group, first = [new, *ordered[index:barrier]], index
        while True:
            plan = self._fit_group(group, wall, new, [])
            on_beat = self._fit_group(group, wall, new, beats) if plan else None
            if on_beat and self._top_speed(on_beat) <= self._top_speed(plan) * BEAT_SPEED_ALLOWANCE:
                plan = on_beat
            if plan and self._within_auto_speed(plan):
                return plan
            if first == 0 or ordered[first - 1].get("locked"):
                break
            first -= 1
            group.insert(0, ordered[first])
        if plan is None:
            raise StudioError("ロック中のフレーズの手前に収まりません。ロックを解除するか、別の場所に入れてください")
        return plan

    def _top_speed(self, plan: list[tuple[dict, float, float]]) -> float:
        return max(self._source_span(clip) / (end - start) for clip, start, end in plan)

    def _within_auto_speed(self, plan: list[tuple[dict, float, float]]) -> bool:
        """True when every phrase that got shorter still plays inside the automatic speed range."""
        return all(end - start >= clip["end"] - clip["start"] - 2e-3 or self._source_span(clip) / (end - start) <= AUTO_SPEED_RANGE[1] + 1e-6
                   for clip, start, end in plan)

    def _fit_group(self, group: list[dict], wall: dict, new: dict, beats: list[float]) -> list[tuple[dict, float, float]] | None:
        """Times for these free phrases, laid out from the first one's start so the last ends a join before `wall`. None when they cannot fit at MAX_INSERT_SPEED."""
        lengths = [clip["end"] - clip["start"] for clip in group]
        durations = [self._source_span(clip) for clip in group]
        pairs = list(zip(group, [*group[1:], wall]))
        gaps = [self._gap_needed(a, b) for a, b in pairs]
        head = group[0]["start"]
        roomy = head + sum(lengths) + sum(gaps) <= wall["start"] + 1e-6
        if not roomy:
            gaps = [self._gap_needed(a, b, tight=True) for a, b in pairs]  # every second of join given back is a slower phrase
        cursor, plan = head, []
        for position, clip in enumerate(group):
            stays = position == 0 or (roomy and clip is not new and clip["start"] >= cursor)
            start = clip["start"] if stays else next((beat for beat in beats if beat >= cursor - 1e-3), cursor)
            fitted = _shorten_to_fit(lengths[position:], durations[position:], wall["start"] - start - sum(gaps[position:]))
            if fitted is None:
                return None
            end = start + fitted[0]
            plan.append((clip, round(start, 3), round(end, 3)))
            cursor = end + gaps[position]
        return plan

    def _apply_room(self, plan: list[tuple[dict, float, float]]) -> tuple[int, int]:
        """Put the planned times in place. Returns how many phrases changed length and how many fell off the end of the song."""
        adjusted = 0
        for clip, start, end in plan:
            if abs((end - start) - (clip["end"] - clip["start"])) > 2e-3:
                clip["speedMode"] = "manual"
                adjusted += 1
            self._retime(clip, start, end)
        kept = [clip for clip in self.clips if clip.get("locked") or clip["end"] <= self.song["duration"] + 1e-3]
        dropped = len(self.clips) - len(kept)
        self.clips = kept
        return adjusted, dropped

    def _planned_insert_span(
        self,
        item: dict,
        index: int,
        begin: float,
        end: float | None,
        place: tuple[float, float],
        fit: str | None,
        bars: float | None,
    ) -> tuple[float, float | None]:
        """Compute `(stop, trim_end)` for an inserted phrase according to `fit` (`None`/`'raw'`, `'bars'`, `'auto'`)."""
        song_dur = float(self.song["duration"])
        raw_dur = float(item["duration"])
        if end is not None or not fit or fit == "raw":
            return min(float(end) if end is not None else begin + raw_dur, song_dur), None

        bar_sec = float(self.song.get("barSeconds") or 2.0)
        next_clip = self.clips[index] if index < len(self.clips) else None
        gap_after = (
            self._gap_needed({"x": place[0], "z": place[1]}, next_clip)
            if next_clip is not None
            else join_seconds(0.0, bar_sec)
        )

        if fit == "bars":
            target_bars = max(0.5, float(bars) if bars is not None else float(item.get("bars") or 2))
            slot_len = target_bars * bar_sec
            ideal = max(MIN_CLIP_SECONDS + 0.05, slot_len - gap_after)
            stop = min(begin + min(ideal, raw_dur / 0.55), song_dur)
            played = stop - begin
            if played > 0 and raw_dur / played > 1.45:
                nat_speed = min(1.35, max(1.0, raw_dur / max(slot_len, 0.4)))
                trim_end = round(min(raw_dur, max(MIN_CLIP_SECONDS + 0.05, played * nat_speed)), 3)
                if trim_end < raw_dur - 0.05:
                    return stop, trim_end
            return stop, None

        # fit == "auto":
        # 1) When inserting into an open gap before `next_clip`, fit inside the gap if comfortable so later clips need not shift.
        if next_clip is not None and begin < next_clip["start"] - 1e-3:
            avail = next_clip["start"] - gap_after - begin
            if avail >= max(MIN_CLIP_SECONDS + 0.1, raw_dur / 1.45):
                return min(begin + min(avail, raw_dur / 0.75), song_dur), None
            if avail >= max(1.5, bar_sec * 0.9) and int(item.get("bars", 2)) == 2 and not has_ending_pose(item):
                nat_speed = min(1.35, max(1.0, raw_dur / max(avail + gap_after, 0.5)))
                trim_end = round(min(raw_dur, avail * nat_speed), 3)
                if trim_end >= max(1.6, raw_dur * 0.58) and trim_end < raw_dur - 0.05:
                    return min(begin + avail, song_dur), trim_end

        # 2) Otherwise match the phrase's musical bar count at this song's tempo.
        nom_bars = int(item.get("bars") or 2)
        if nom_bars == 1 and (bar_sec - gap_after < max(MIN_CLIP_SECONDS + 0.1, raw_dur / 1.55)):
            slot_len = 2.0 * bar_sec
        else:
            slot_len = float(nom_bars) * bar_sec
        ideal = max(MIN_CLIP_SECONDS + 0.05, slot_len - gap_after)
        if raw_dur / ideal <= 1.45:
            return min(begin + min(ideal, raw_dur / 0.78), song_dur), None
        if not has_ending_pose(item) and nom_bars == 2:
            nat_speed = min(1.35, max(1.0, raw_dur / max(slot_len, 0.5)))
            trim_end = round(min(raw_dur, ideal * nat_speed), 3)
            if trim_end >= max(1.6, raw_dur * 0.58) and trim_end < raw_dur - 0.05:
                return min(begin + ideal, song_dur), trim_end
        return min(begin + max(ideal, raw_dur / 1.50), song_dur), None

    def add_clip(self, phrase_id: str, start: float, end: float | None = None, x: float | None = None, z: float | None = None,
                 exact: bool = False, fit: str | None = None, bars: float | None = None) -> dict:
        with self.lock:
            item = self.library.items.get(phrase_id)
            if item is None:
                raise StudioError("そのフレーズは見つかりません")
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            if fit not in (None, "raw", "auto", "bars"):
                raise StudioError("長さ調整の指定が不正です")
            asked_place = (float(x), float(z)) if x is not None and z is not None else None
            if exact:
                index, begin, pre_plan = self._exact_point(float(start), asked_place)
            else:
                index, begin = self._insert_point(float(start))
                pre_plan = []
            before = self.clips[index - 1] if index else None
            place = asked_place if asked_place is not None else (before["x"], before["z"]) if before else (0.0, 3.0)
            stop, trim_end = self._planned_insert_span(item, index, begin, end, place, fit, bars)
            eff_dur = trim_end if trim_end is not None else item["duration"]
            if stop - begin < max(MIN_CLIP_SECONDS, eff_dur / MAX_INSERT_SPEED) - 1e-4:
                raise StudioError("曲の終わりに近すぎて、このフレーズは入りません")
            new = {"id": uuid.uuid4().hex[:10], "phraseId": phrase_id, "start": round(begin, 3), "end": round(stop, 3),
                   "x": place[0], "z": place[1], "facing": 0.0, "locked": False, "speedMode": "manual", "match": self.project["match"]}
            if trim_end is not None:
                new["trimEnd"] = round(trim_end, 3)
                _clean_trim(new, item)
            plan = self._room_plan(index, new)
            planned_ids = {id(clip) for clip, _, _ in plan}
            full_plan = [(c, s, e) for c, s, e in pre_plan if id(c) not in planned_ids] + plan
            self._checkpoint()
            adjusted, dropped = self._apply_room(full_plan)
            self.clips.append(new)
            self.clips.sort(key=lambda c: c["start"])
            if before is not None and before.get("speedMode") == "auto" and not before.get("locked"):
                self._retime(before, before["start"], self._auto_end(before))
            self._save()
            self._recompose()
            return {**self.changes(), "added": new["id"], "dropped": dropped, "adjusted": adjusted}

    def paste_clip(self, source: dict, start: float) -> dict:
        """Put a copy of a phrase at `start`, for the dancer being edited.

        The copy keeps the length, the part of the phrase in use, the position, the facing, the sound matching and the accent
        pins of `source` (a clip as the screen holds it, which may be gone from the timeline by now). It goes in by the same
        rules as any insert, never on top of a phrase, and starts unlocked.
        """
        with self.lock:
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            try:
                phrase_id = str(source["phraseId"])
                length = round(float(source["end"]) - float(source["start"]), 3)
                x, z = _on_stage(float(source["x"]), float(source["z"]))
                facing, match = round(float(source.get("facing", 0.0)), 1), int(source.get("match", self.project["match"]))
                pins = [(float(pin["source"]), float(pin["at"])) for pin in source.get("pins") or []]
            except (KeyError, TypeError, ValueError) as error:
                raise StudioError("貼り付ける内容が不正です") from error
            item = self.library.items.get(phrase_id)
            if item is None:
                raise StudioError("コピーしたフレーズは、もうライブラリにありません")
            if length < MIN_CLIP_SECONDS or not 0 <= match <= 100:
                raise StudioError("貼り付ける内容が不正です")
            index, begin = self._insert_point(float(start))
            new = {"id": uuid.uuid4().hex[:10], "phraseId": phrase_id, "start": round(begin, 3), "end": round(begin + length, 3),
                   "x": x, "z": z, "facing": facing, "locked": False, "speedMode": "manual", "match": match}
            for key in ("trimStart", "trimEnd"):
                if source.get(key) is not None:
                    new[key] = source[key]
            trim_start, trim_end = _clean_trim(new, item)
            if new["end"] > self.song["duration"] + 1e-3:
                raise StudioError("曲の終わりに近すぎて、このフレーズは入りません")
            kept = usable_pins([(at_source, new["start"] + at) for at_source, at in pins[:MAX_PINS]], new["start"], new["end"], trim_end, trim_start)
            if kept:
                new["pins"] = [{"source": round(at_source, 3), "at": round(when - new["start"], 3)} for at_source, when in kept]
            plan = self._room_plan(index, new)
            before = self.clips[index - 1] if index else None
            self._checkpoint()
            adjusted, dropped = self._apply_room(plan)
            self.clips.append(new)
            self.clips.sort(key=lambda c: c["start"])
            if before is not None and before.get("speedMode") == "auto" and not before.get("locked"):
                self._retime(before, before["start"], self._auto_end(before))
            self._save()
            self._recompose()
            return {**self.changes(), "added": new["id"], "dropped": dropped, "adjusted": adjusted}

    def reorder_clips(self, order: list[str]) -> dict:
        """Put the phrases in a new order. Each keeps its length and the room after it, so nothing outside the moved range shifts."""
        with self.lock:
            ordered = self.clips
            ids = [clip["id"] for clip in ordered]
            if sorted(order) != sorted(ids):
                raise StudioError("並び順の指定が不正です")
            if order == ids:
                return self.changes()
            changed = [i for i, (a, b) in enumerate(zip(ids, order)) if a != b]
            first, last = changed[0], changed[-1]
            if any(clip.get("locked") for clip in ordered[first:last + 1]):
                raise StudioError("ロック中のフレーズの時刻が変わるため、この並び替えはできません。先にロックを解除してください")
            bar = self.song["barSeconds"]
            room = {}
            for i, clip in enumerate(ordered):
                length = clip["end"] - clip["start"]
                room[clip["id"]] = max(length, ordered[i + 1]["start"] - clip["start"]) if i + 1 < len(ordered) else math.ceil((length + MIN_JOIN_SECONDS) / bar - 1e-6) * bar
            by_id = {clip["id"]: clip for clip in ordered}
            self._checkpoint()
            cursor = ordered[first]["start"]
            for clip_id in order[first:last + 1]:
                clip = by_id[clip_id]
                length = clip["end"] - clip["start"]
                self._retime(clip, round(cursor, 3), round(cursor + length, 3))
                cursor += room[clip_id]
            self.clips.sort(key=lambda c: c["start"])
            for clip in self.clips[max(0, first - 1):last + 1]:
                if clip.get("speedMode") == "auto" and not clip.get("locked"):
                    self._retime(clip, clip["start"], self._auto_end(clip))
                if clip["end"] > self.song["duration"] and self.song["duration"] - clip["start"] >= MIN_CLIP_SECONDS:
                    self._retime(clip, clip["start"], round(self.song["duration"], 3))
            self._save()
            self._recompose()
            return self.changes()

    def delete_clip(self, clip_id: str) -> dict:
        with self.lock:
            if self._clip(clip_id).get("locked"):
                raise StudioError("ロック中のフレーズは削除できません。先にロックを解除してください")
            self._checkpoint()
            self.clips = [c for c in self.clips if c["id"] != clip_id]
            self._save()
            self._recompose()
            return self.changes()

    def candidates(self, clip_id: str) -> list[dict]:
        with self.lock:
            clip = self._clip(clip_id)
            section = self._section_at(clip["start"])
            kind = section["kind"] if section else "verse"
            flow = self.project["flow"].get(kind, DEFAULT_FLOW["verse"])
            accents = self.song.get("accents", [])
            sound = np.array([a["t"] for a in accents])
            weights = np.array([1.30 if a.get("kind") == "kime" else (1.12 if a.get("onBeat") else 1.0) for a in accents]) if accents else None
            bar_sec = float(self.song.get("barSeconds") or 2.0)
            start_bar = int(clip["start"] // max(bar_sec, 0.2))
            end_bar = max(start_bar + 1, int(round(clip["end"] / max(bar_sec, 0.2))))
            epb = self.song.get("energyPerBar") or []
            local_e = float(np.mean(epb[start_bar:min(end_bar, len(epb))])) if epb and start_bar < len(epb) else None
            slot_bars = 4 if (clip["end"] - clip["start"]) >= bar_sec * 2.8 else 2
            has_kime = any(a.get("kind") == "kime" and clip["start"] - 0.15 <= a["t"] <= clip["end"] + 0.35 for a in accents)
            is_sec_end = bool(section and clip["end"] >= section["end"] - bar_sec * 1.1)
            earlier = [c for c in self.clips if c["start"] < clip["start"]][-6:]
            recent = [c["phraseId"] for c in earlier]
            recent_stems = [base_stem(self.library.items[c["phraseId"]]) for c in earlier if c["phraseId"] in self.library.items]
            mood_counts = {m: sum(1 for c in self.clips if m in self.library.items.get(c["phraseId"], {}).get("tags", [])) for m in ("女性的", "男性的", "中性的")}
            pref_mood = max(mood_counts, key=mood_counts.get) if max(mood_counts.values(), default=0) > 0 else None
            active_tags = list(self.project.get("arrangeFilter") or [])
            found = candidates_for(
                list(self.library.items.values()),
                flow,
                recent,
                clip["start"],
                clip["end"],
                sound,
                limit=8,
                section_kind=kind,
                local_energy=local_e,
                slot_bars=slot_bars,
                preferred_mood=pref_mood,
                recent_stems=recent_stems,
                sound_weights=weights,
                has_kime=has_kime,
                is_section_end=is_sec_end,
                filter_tags=active_tags or None,
            )
            swappable = [c for c in found if c["phraseId"] != clip["phraseId"]][:5]
            if swappable or not active_tags:
                return swappable
            fallback = candidates_for(
                list(self.library.items.values()),
                flow,
                recent,
                clip["start"],
                clip["end"],
                sound,
                limit=8,
                section_kind=kind,
                local_energy=local_e,
                slot_bars=slot_bars,
                preferred_mood=pref_mood,
                recent_stems=recent_stems,
                sound_weights=weights,
                has_kime=has_kime,
                is_section_end=is_sec_end,
            )
            return [c for c in fallback if c["phraseId"] != clip["phraseId"]][:5]

    def join_options(self, clip_id: str) -> dict:
        """The join after this phrase at several lengths, each made and measured, so the best one can be taken.

        A different length means the phrase before the join ends earlier or later. The next phrase stays where it is.
        """
        with self.lock:
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            a = self._clip(clip_id)
            later = [c for c in self.clips if c["start"] > a["start"]]
            if not later:
                raise StudioError("このフレーズの後ろにつなぎはありません")
            b = dict(min(later, key=lambda c: c["start"]))
            trim_start, trim_end = self._trim(a)
            duration = trim_end - trim_start
            bar = self.song["barSeconds"]
            current = round(b["start"] - a["end"], 3)
            lengths = []
            for quarters in range(1, 65):
                length = round(quarters * bar / 4.0, 3)
                played = b["start"] - length - a["start"]
                if played < max(MIN_CLIP_SECONDS, duration / OPTION_SPEED_RANGE[1]):
                    break
                if length >= MIN_JOIN_SECONDS - 1e-3 and played <= duration / OPTION_SPEED_RANGE[0] and abs(length - current) > 0.02:
                    lengths.append(length)
            chosen = [x for x in lengths if x < current][-2:] + ([current] if current >= MIN_JOIN_SECONDS - 1e-3 else []) + [x for x in lengths if x > current][:3]
            trials = []
            for length in chosen:
                end = round(b["start"] - length, 3)
                trial = {**a, "end": end}
                if a.get("pins"):
                    trial["pins"] = carried_pins(a["pins"], a["start"], a["end"], a["start"], end, trim_end, trim_start)
                trials.append((length, trial))
            song, locked = self.song, bool(a.get("locked"))
        if not trials:
            raise StudioError("このつなぎには、選べる長さがありません")
        options = []
        for length, trial in trials:  # outside the lock: each may take a generation
            speed = duration / (trial["end"] - trial["start"])
            option = {**measure_join(trial, b, self.library, song, self.engine, self.cache), "end": trial["end"], "bars": round(length / bar, 2),
                      "clipSpeed": round(speed, 3), "current": length == current}
            option["flags"] = (["fast"] if option["moveSpeed"] > FAST_MOVE_SPEED else []) + ([] if AUTO_SPEED_RANGE[0] - 1e-3 <= speed <= AUTO_SPEED_RANGE[1] + 1e-3 else ["speed"])
            options.append(option)
        min(options, key=lambda o: (len(o["flags"]), o["maxJointSpeed"]))["best"] = True
        return {"from": trials[0][1]["id"], "to": b["id"], "locked": locked, "options": options}

    # ---- dancers and formations

    def _formations(self) -> list[dict]:
        return self.project.setdefault("formations", [])

    def add_dancer(self, copy_from: str | None = None) -> dict:
        """A new dancer, empty or with another dancer's phrases one step to the side. It becomes the one being edited."""
        with self.lock:
            if len(self.dancers) >= MAX_DANCERS:
                raise StudioError(f"ダンサーは {MAX_DANCERS} 人までです")
            source = self._dancer(copy_from) if copy_from else None
            self._checkpoint()
            number = 1 + max([int(d["id"][1:]) for d in self.dancers if d["id"][1:].isdigit()] or [0])
            new = {"id": f"d{number}", "name": f"ダンサー {number}", "clips": []}
            if source is not None:
                side = NEW_DANCER_STEP * ((len(self.dancers) + 1) // 2) * (1 if len(self.dancers) % 2 else -1)  # right, left, further right ...
                for clip in source["clips"]:
                    x, z = _on_stage(clip["x"] + side, clip["z"])
                    new["clips"].append({**copy.deepcopy(clip), "id": uuid.uuid4().hex[:10], "x": x, "z": z})
                for formation in self._formations():
                    spot = formation["spots"].get(source["id"])
                    if spot:
                        x, z = _on_stage(spot["x"] + side, spot["z"])
                        formation["spots"][new["id"]] = {**spot, "x": x, "z": z}
            self.dancers.append(new)
            self.project["active"] = new["id"]
            self._save()
            self._recompose()
            return self.changes()

    def update_dancer(self, dancer_id: str, changes: dict) -> dict:
        with self.lock:
            dancer = self._dancer(dancer_id)
            name = str(changes.get("name", dancer["name"])).strip()[:24]
            if not name:
                raise StudioError("名前を入力してください")
            dancer["name"] = name
            if "hidden" in changes:  # only what the stage shows; the dancer still counts for close calls and exports
                if changes["hidden"]:
                    dancer["hidden"] = True
                else:
                    dancer.pop("hidden", None)
            self._save()
            return self.changes()

    def move_dancer(self, dancer_id: str, dx: float, dz: float) -> dict:
        """Shift every phrase of one dancer by the same distance on the floor. Locked phrases stay where they are."""
        with self.lock:
            dancer = self._dancer(dancer_id)
            dx, dz = round(float(dx), 3), round(float(dz), 3)
            free = [clip for clip in dancer["clips"] if not clip.get("locked")]
            if not free:
                raise StudioError("動かせるフレーズがありません" if dancer["clips"] else "このダンサーにはフレーズがありません")
            kept = len(dancer["clips"]) - len(free)
            if dx == 0.0 and dz == 0.0:
                return {**self.changes(), "moved": 0, "kept": kept}
            self._checkpoint()
            for clip in free:
                clip["x"], clip["z"] = round(clip["x"] + dx, 3), round(clip["z"] + dz, 3)
            self._save()
            self._recompose()
            return {**self.changes(), "moved": len(free), "kept": kept}

    def select_dancer(self, dancer_id: str) -> dict:
        with self.lock:
            self.project["active"] = self._dancer(dancer_id)["id"]
            self._save()
            return self.changes()

    def delete_dancer(self, dancer_id: str) -> dict:
        with self.lock:
            dancer = self._dancer(dancer_id)
            if len(self.dancers) < 2:
                raise StudioError("最後の 1 人は削除できません")
            self._checkpoint()
            self.project["dancers"] = [d for d in self.dancers if d is not dancer]
            for formation in self._formations():
                formation["spots"].pop(dancer["id"], None)
            self.project.get("models", {}).pop(dancer["id"], None)
            if self.project.get("active") == dancer["id"]:
                self.project["active"] = self.dancers[0]["id"]
            self._save()
            self._recompose()
            return self.changes()

    def sync_dancers(self) -> dict:
        """Give every other dancer the phrases and timing of the one being edited. Each stays as far to the side as it was, and formations still decide the places they cover."""
        with self.lock:
            source = self.dancer
            if not source["clips"]:
                raise StudioError("そろえる振りがありません。先にこのダンサーにフレーズを置いてください")
            if len(self.dancers) < 2:
                raise StudioError("ダンサーが 1 人です。先にダンサーを追加してください")
            self._checkpoint()
            lead = source["clips"][0]
            for order, dancer in enumerate(self.dancers):
                if dancer is source:
                    continue
                first = dancer["clips"][0] if dancer["clips"] else None
                dx, dz = (first["x"] - lead["x"], first["z"] - lead["z"]) if first else (NEW_DANCER_STEP * order, 0.0)
                dancer["clips"] = []
                for clip in source["clips"]:
                    x, z = _on_stage(clip["x"] + dx, clip["z"] + dz)
                    dancer["clips"].append({**copy.deepcopy(clip), "id": uuid.uuid4().hex[:10], "x": x, "z": z})
            self._apply_formations()
            self._save()
            self._recompose()
            return self.changes()

    def set_formation(self, time: float, spots: dict, name: str = "") -> dict:
        """Where each dancer stands from the bar at `time` until the next formation. Replaces the formation already on that bar."""
        with self.lock:
            if self.song is None:
                raise StudioError("先に曲を読み込んでください")
            bars = self.song["bars"]
            start = float(max((bar for bar in bars if bar <= float(time) + 1e-3), default=bars[0]))
            cleaned = {}
            for dancer in self.dancers:
                spot = spots.get(dancer["id"])
                if spot is None:
                    continue
                try:
                    x, z = _on_stage(float(spot["x"]), float(spot["z"]))
                    facing = None if spot.get("facing") is None else round(float(spot["facing"]), 1)
                except (KeyError, TypeError, ValueError) as error:
                    raise StudioError("フォーメーションの位置が不正です") from error
                cleaned[dancer["id"]] = {"x": x, "z": z, "facing": facing}
            if not cleaned:
                raise StudioError("位置を決めるダンサーがいません")
            self._checkpoint()
            formations = [f for f in self._formations() if abs(f["time"] - start) > 1e-3]
            formation = {"id": uuid.uuid4().hex[:8], "time": round(start, 3), "name": str(name).strip()[:24], "spots": cleaned}
            self.project["formations"] = sorted([*formations, formation], key=lambda f: f["time"])
            self._apply_formations(formation["id"])
            self._save()
            self._recompose()
            return {**self.changes(), "formation": formation["id"]}

    def delete_formation(self, formation_id: str) -> dict:
        """Take a formation away. The stretch it covered goes back to the formation before it, if there is one."""
        with self.lock:
            formations = self._formations()
            index = next((i for i, f in enumerate(formations) if f["id"] == formation_id), None)
            if index is None:
                raise StudioError("そのフォーメーションは見つかりません")
            self._checkpoint()
            del formations[index]
            if index:
                self._apply_formations(formations[index - 1]["id"])
            self._save()
            self._recompose()
            return self.changes()

    def _apply_formations(self, only: str | None = None) -> None:
        """Move each dancer's unlocked phrases to the spot of the formation covering them. Phrases on automatic timing refit their joins."""
        formations = self._formations()
        for index, formation in enumerate(formations):
            if only is not None and formation["id"] != only:
                continue
            until = formations[index + 1]["time"] if index + 1 < len(formations) else float("inf")
            for dancer in self.dancers:
                spot = formation["spots"].get(dancer["id"])
                if spot is None:
                    continue
                ordered = dancer["clips"]
                moved = []
                for position, clip in enumerate(ordered):
                    if clip.get("locked") or not formation["time"] - 1e-3 <= clip["start"] < until - 1e-3:
                        continue
                    facing = clip["facing"] if spot.get("facing") is None else spot["facing"]
                    if (clip["x"], clip["z"], clip["facing"]) != (spot["x"], spot["z"], facing):
                        clip.update(x=spot["x"], z=spot["z"], facing=facing)
                        moved += [position - 1, position]
                for position in sorted(set(moved)):
                    clip = ordered[position] if position >= 0 else None
                    if clip and clip.get("speedMode") == "auto" and not clip.get("locked"):
                        self._retime(clip, clip["start"], self._auto_end(clip, ordered))

    def set_model(self, dancer_id: str | None, path: str | None, mapping: dict | None = None, everyone: bool = False) -> dict:
        """The 3D model a dancer is shown with, or None for the mannequin. `mapping` is humanoid bone -> the model's bone name, for models that are not VRM."""
        with self.lock:
            targets = self.dancers if everyone else [self._dancer(dancer_id)]
            models = self.project.setdefault("models", {})
            if path is None:
                for dancer in targets:
                    models.pop(dancer["id"], None)
            else:
                file = Path(path)
                if file.suffix.lower() not in MODEL_KINDS:
                    raise StudioError("読み込めるモデルは VRM / GLB / FBX です")
                if not file.is_file():
                    raise StudioError("モデルのファイルが見つかりません")
                for dancer in targets:
                    models[dancer["id"]] = {"path": str(file), "name": file.name, "mapping": dict(mapping or {})}
            self._save()
            return self.changes()

    def model_file(self, dancer_id: str, name: str) -> Path:
        """A file of this dancer's model: the model itself, or something beside it that it refers to (textures)."""
        with self.lock:
            model = self.project.get("models", {}).get(dancer_id)
        if not model:
            raise StudioError("モデルが設定されていません")
        folder = Path(model["path"]).resolve().parent
        target = (folder / name).resolve()
        if folder not in target.parents or not target.is_file():
            raise StudioError("モデルのファイルが見つかりません")
        return target

    # ---- baking and output

    def _recompose(self, force: bool = False) -> None:
        """Rebuild the timeline after an edit. In background mode, joins that are not ready are bridged for now and handed to the worker."""
        self._serial += 1
        self._engine_error = None
        self._generated = {}
        self._edit_began = time.perf_counter()
        self._bake(force=force)
        if self._jobs:
            self._wake.set()
        else:
            with self._done:
                self._done.notify_all()

    def environment_changed(self) -> None:
        """An environment was installed: joins that failed for lack of it are made now."""
        with self.lock:
            self._recompose(force=True)

    def _memory(self, dancer_id: str) -> dict:
        return self._baked.setdefault(dancer_id, {"memory": BakeMemory(), "result": None, "preview": None, "signature": None, "version": 0, "jobs": []})

    def _plan(self) -> dict[str, dict]:
        """Per dancer, the joins that walk round another dancer and the point each passes through."""
        if len(self.dancers) < 2:
            return {}
        total = int(np.ceil(self.song["duration"] * MODEL_FPS)) + 1
        casts = [lay_clips(dancer["clips"], self.library, self.song, self._memory(dancer["id"])["memory"])[0] for dancer in self.dancers]
        return {dancer["id"]: plan for dancer, plan in zip(self.dancers, plan_vias(casts, total))}

    def _bake(self, final: bool = False, force: bool = False) -> None:
        """Compose every dancer whose phrases, detours or waiting joins changed. The others keep what they have."""
        if self.song is None or not self.library.items:
            self._baked, self._jobs, self._collisions = {}, [], []
            return
        for gone in set(self._baked) - {dancer["id"] for dancer in self.dancers}:
            del self._baked[gone]
        plans = self._plan()
        self.version += 1
        jobs: list[tuple[str, object]] = []
        for dancer in self.dancers:
            baked = self._memory(dancer["id"])
            if not dancer["clips"]:
                if baked["result"] is not None:
                    baked.update(result=None, preview=None, signature=None, jobs=[], version=self.version)
                continue
            plan = plans.get(dancer["id"], {})
            signature = json.dumps([dancer["clips"], sorted([*pair, *via] for pair, via in plan.items())], sort_keys=True)
            result = baked["result"]
            settled = result is not None and not baked["jobs"] and result["engineError"] is None
            if settled and signature == baked["signature"] and not force:
                continue
            fresh = compose(dancer["clips"], self.library, self.song, self.engine, self.cache, baked["memory"],
                            defer=self.background and not final, engine_error=self._engine_error, vias=plan)
            baked["jobs"] = fresh.pop("jobs")
            made = self._generated.get(dancer["id"], 0)
            if made:
                fresh["generated"], fresh["reused"] = made, max(0, fresh["reused"] - made)
            fresh["seconds"] = round(time.perf_counter() - self._edit_began, 3)
            baked["preview"] = self._preview_of(fresh["poses"], result["poses"] if result else None, baked["preview"])
            baked.update(result=fresh, signature=signature, version=self.version)
        for dancer in self.dancers:
            jobs += [(dancer["id"], job) for job in self._memory(dancer["id"])["jobs"]]
        self._jobs = jobs
        tracks = [(dancer["id"], self._baked[dancer["id"]]["result"]["poses"][:, 0][:, [0, 2]]) for dancer in self.dancers if self._baked[dancer["id"]]["result"] is not None]
        self._collisions = close_calls(tracks) if len(tracks) > 1 else []

    def _preview_of(self, poses: np.ndarray, before: np.ndarray | None, shown: np.ndarray | None) -> np.ndarray:
        """Joint positions for the screen. Only frames that changed are worked out again."""
        skeleton = self.library.get_skeleton()
        keep = [i for i, name in enumerate(skeleton.names) if not any(token in name for token in SKIP_JOINTS)]
        if before is None or shown is None or before.shape != poses.shape:
            return world_positions(poses, skeleton.parents)[:, keep].astype(np.float32)
        changed = np.flatnonzero((poses != before).any(axis=(1, 2)))
        if changed.size:
            shown[changed] = world_positions(poses[changed], skeleton.parents)[:, keep].astype(np.float32)
        return shown

    def _worker(self) -> None:
        try:
            self.engine.start()  # so the first edit does not wait for the model to load
        except EngineError:
            pass  # reported on the join that needs it
        while not self._closing:
            self._wake.wait()
            self._wake.clear()
            if not self._closing:
                self._generate_pending()

    def _generate_pending(self) -> None:
        while not self._closing:
            with self.lock:
                jobs, serial = self._jobs, self._serial
                skeleton = self.library.get_skeleton() if jobs else None
            if not jobs:
                return
            made: dict[str, int] = {}
            failure = None
            shown = time.perf_counter()
            for count, (dancer_id, job) in enumerate(jobs, 1):
                if self._serial != serial or self._closing:
                    break
                try:
                    generate_join(self.engine, skeleton, self.cache, job)
                    made[dancer_id] = made.get(dancer_id, 0) + 1
                except EngineError as error:
                    failure = str(error)
                    break
                if time.perf_counter() - shown > PROGRESS_SECONDS and count < len(jobs):
                    with self.lock:
                        if self._serial == serial:  # finished joins stop saying "generating" while the rest carry on
                            self._generated = dict(made)
                            self._bake()
                            self._done.notify_all()
                    shown = time.perf_counter()
            with self.lock:
                if self._serial != serial:
                    continue  # an edit arrived meanwhile and queued its own jobs
                self._engine_error, self._generated = failure, made
                self._bake(final=True)
                self._done.notify_all()
                return

    def wait_for_joins(self, version: int | None = None, timeout: float = 25.0) -> dict:
        """Blocks until no join is waiting, or until the timeline differs from `version`, and returns it."""
        with self.lock:
            self._done.wait_for(lambda: not self._jobs or (version is not None and self.version != version), timeout)
            return self.changes()

    def preview_bytes(self, dancer_id: str | None = None) -> bytes:
        with self.lock:
            preview = self._baked.get(self._dancer(dancer_id)["id"], {}).get("preview")
            if preview is None:
                raise StudioError("プレビューがまだありません")
            return np.ascontiguousarray(preview, dtype="<f4").tobytes()

    def turn_bytes(self, dancer_id: str | None = None) -> bytes:
        """How far each humanoid bone has turned from the T-pose, per frame, for showing the motion on a model. Order is `modelBones`."""
        with self.lock:
            baked = self._baked.get(self._dancer(dancer_id)["id"], {})
            if baked.get("result") is None:
                raise StudioError("プレビューがまだありません")
            if baked.get("turns", (None, None))[0] != baked["version"]:
                poses = baked["result"]["poses"]
                raw, turns = incremental_turns(poses, self.library.get_skeleton(), baked.get("turnsPoses"), baked.get("turnsRaw"))
                baked["turnsPoses"], baked["turnsRaw"] = poses, raw
                baked["turns"] = (baked["version"], np.ascontiguousarray(turns, dtype="<f4").tobytes())
            return baked["turns"][1]

    def export(self, kind: str, path: str | None = None, pmx: str | None = None, rest: str = "bones", arm_degrees: float = 45.0) -> dict:
        """Write every dancer's finished motion, one file each. `pmx` fits a VMD to that MMD model.

        `rest` is the pose a BVH or FBX rests in: "bones" (the SOMA files' own), "t" (T-pose) or "a" (A-pose, arms `arm_degrees` below level).
        """
        if kind not in EXPORT_KINDS:
            raise StudioError("書き出し形式が不正です")
        if rest not in REST_KINDS or not 0.0 < float(arm_degrees) < 90.0:
            raise StudioError("基準の姿勢の指定が不正です（T ポーズ、A ポーズ、SOMA のまま。A ポーズの腕の角度は 0〜90 度の間）")
        bones = None
        if kind == "vmd" and pmx:
            try:
                bones = read_pmx_bones(pmx)
            except (OSError, PmxError, struct.error) as error:
                raise StudioError(f"PMX を読み込めません: {error}") from error
        with self.lock:
            self._done.wait_for(lambda: not self._jobs, 600.0)
            cast = [(dancer["name"], self._baked[dancer["id"]]["result"]["poses"].copy()) for dancer in self.dancers
                    if self._baked.get(dancer["id"], {}).get("result") is not None]
            if not cast:
                raise StudioError("書き出すモーションがありません")
            skeleton = self.library.get_skeleton()
            song_name = _file_part(Path(self.project["songPath"]).stem)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if path and Path(path).suffix.lower() != f".{kind}":
            raise StudioError(f"保存先の拡張子は .{kind} にしてください")
        targets = []
        for number, (name, _) in enumerate(cast, 1):
            part = "" if len(cast) == 1 else f"-{number}-{_file_part(name)}"
            target = Path(path).with_name(f"{Path(path).stem}{part}.{kind}") if path else self.data / "exports" / f"{song_name}-{stamp}{part}.{kind}"
            if target.exists():
                raise StudioError("同じ名前のファイルが既にあります")
            targets.append(target)
        files = []
        for target, (name, poses) in zip(targets, cast):
            target.parent.mkdir(parents=True, exist_ok=True)
            if kind in ("bvh", "fbx") and rest != "bones":
                skeleton, poses = rebased(poses, self.library.get_skeleton(), rest_pose(self.library.get_skeleton(), rest, float(arm_degrees)))
            if kind == "bvh":
                write_bvh(target, skeleton, poses, MODEL_FPS)
            elif kind == "vmd":
                write_vmd(target, poses, skeleton, bones)
            elif kind == "vrma":
                write_vrma(target, poses, skeleton, MODEL_FPS)
            else:
                bvh = self.data / "exports" / f".tmp-{uuid.uuid4().hex[:8]}.bvh"
                write_bvh(bvh, skeleton, poses, MODEL_FPS)
                try:
                    self._bvh_to_fbx(bvh, target, keep_rest=rest != "bones")
                finally:
                    bvh.unlink(missing_ok=True)
            files.append({"path": str(target), "dancer": name, "bytes": target.stat().st_size})
        return {"path": files[0]["path"], "files": files, "frames": cast[0][1].shape[0], "fps": MODEL_FPS, "bytes": sum(f["bytes"] for f in files)}

    def save_export(self, name: str, data: bytes) -> dict:
        """Keep a file the screen made (a model with its motion as GLB) with the other exports."""
        stem, suffix = _file_part(Path(name).stem), Path(name).suffix.lower()
        if suffix != ".glb" or not data:
            raise StudioError("保存できるのは GLB です")
        target = self.data / "exports" / f"{stem}-{time.strftime('%Y%m%d-%H%M%S')}{suffix}"
        if target.exists():
            raise StudioError("同じ名前のファイルが既にあります")
        target.write_bytes(data)
        return {"path": str(target), "files": [{"path": str(target), "dancer": "", "bytes": len(data)}], "bytes": len(data)}

    @staticmethod
    def _bvh_to_fbx(bvh: Path, target: Path, keep_rest: bool = False) -> None:
        if not usable("blender"):
            raise StudioError("FBX の書き出しに必要な Blender 環境が見つかりません")
        script = ROOT / "backend" / "export_fbx.py"
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        done = subprocess.run([str(python_of("blender")), "-X", "utf8", str(script), str(bvh), str(target), *(["--rest"] if keep_rest else [])], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=600, check=False, creationflags=flags)
        if done.returncode or not target.is_file():
            raise StudioError(f"FBX の書き出しに失敗しました: {(done.stderr or done.stdout)[-300:]}")

    def close(self) -> None:
        self._closing = True
        self._wake.set()
        self.engine.stop()
