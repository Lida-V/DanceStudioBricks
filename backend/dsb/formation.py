"""More than one dancer: detours that keep their paths apart, and the moments they still come too close.

MotionBricks does not see other dancers, so the app plans the way round and hands it over as a point to pass through.
"""
from __future__ import annotations

import numpy as np

from .arrange import STAGE_X, STAGE_Z
from .compose import MIN_SINGLE_FRAMES, MIN_VIA_FRAMES, Built, Via
from .motion import MODEL_FPS, contact_spans

CLEARANCE = 0.7  # metres between two dancers' hips that a planned path keeps
TOO_CLOSE = 0.45  # metres between hips that is reported as a collision
DETOURS = (0.6, 0.9, 1.2, 1.6)  # sideways steps tried at the middle of a join, metres
WORTH_IT = 0.15  # a detour has to open at least this much more room than the straight path
MAX_DETOUR_PACE = 2.2  # metres per second. A detour that needs more is not taken,
DETOUR_STRETCH = 1.15  # unless the straight walk is already that fast: then it may be this much longer
STAGE_MARGIN = 0.5
STILL = 0.3  # a join that travels less than this has no side to step to


def root_track(built: list[Built], total: int) -> np.ndarray | None:
    """Hips on the floor per timeline frame, [total, 2]: phrases as baked, joins as straight lines, the nearest pose held elsewhere."""
    track = np.full((total, 2), np.nan)
    for _, first, frames in built:
        end = min(total, first + frames.shape[0])
        if first < total:
            track[first:end] = frames[:end - first, 0][:, [0, 2]]
    known = np.flatnonzero(~np.isnan(track[:, 0]))
    if not known.size:
        return None
    steps = np.arange(total)
    return np.column_stack([np.interp(steps, known, track[known, axis]) for axis in (0, 1)])


def _path(begin: np.ndarray, finish: np.ndarray, via: np.ndarray | None, count: int) -> np.ndarray:
    """`count` points from begin to finish at an even pace, through `via` when there is one."""
    points = np.array([begin, finish] if via is None else [begin, via, finish])
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if along[-1] < 1e-9:
        return np.tile(begin, (count, 1))
    at = np.linspace(0.0, along[-1], count)
    return np.column_stack([np.interp(at, along, points[:, axis]) for axis in (0, 1)])


def _on_stage(point: np.ndarray) -> bool:
    return STAGE_X[0] - STAGE_MARGIN <= point[0] <= STAGE_X[1] + STAGE_MARGIN and STAGE_Z[0] - STAGE_MARGIN <= point[1] <= STAGE_Z[1] + STAGE_MARGIN


def plan_vias(casts: list[list[Built]], total: int) -> list[dict[tuple[str, str], Via]]:
    """For each dancer, the joins that go round someone: (from clip id, to clip id) -> floor point to pass through.

    The first dancer always walks straight. Each later one gives way to everyone, taking the smallest sideways step that
    keeps CLEARANCE, or the best one found when none does. Joins too short for two generations are left straight.
    """
    tracks = [root_track(built, total) for built in casts]
    plans: list[dict[tuple[str, str], Via]] = [{} for _ in casts]
    for number, built in enumerate(casts):
        own = tracks[number]
        others = [track for index, track in enumerate(tracks) if index != number and track is not None]
        if number == 0 or own is None or not others:
            continue
        for (clip_a, first_a, frames_a), (clip_b, first_b, _) in zip(built[:-1], built[1:]):
            last_a = first_a + frames_a.shape[0] - 1
            count = first_b - last_a + 1
            if count < MIN_VIA_FRAMES or first_b >= total:
                continue
            begin, finish = own[last_a].copy(), own[first_b].copy()
            reach = float(np.linalg.norm(finish - begin))
            if reach < STILL:
                continue
            span = slice(last_a, first_b + 1)

            def room(path: np.ndarray) -> float:
                return min(float(np.linalg.norm(path - other[span], axis=1).min()) for other in others)

            line = _path(begin, finish, None, count)
            straight = room(line)
            if straight >= CLEARANCE:
                continue
            side = np.array([-(finish - begin)[1], (finish - begin)[0]]) / reach
            seconds = (count - 1) / MODEL_FPS
            fastest = max(MAX_DETOUR_PACE, DETOUR_STRETCH * reach / seconds)
            # Step aside where the path comes nearest to someone, as far as both legs still have room for a generation, or else in the middle.
            nearest = int(np.argmin(np.min([np.linalg.norm(line - other[span], axis=1) for other in others], axis=0)))
            edge = MIN_SINGLE_FRAMES / count
            turns = sorted({round(float(np.clip(nearest / (count - 1), edge, 1.0 - edge)), 3), 0.5})
            best, chosen = straight, None
            for step in DETOURS:
                for along in turns:
                    for sign in (1.0, -1.0):
                        via = begin + (finish - begin) * along + sign * step * side
                        path = _path(begin, finish, via, count)
                        pace = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) / seconds
                        if not _on_stage(via) or pace > fastest:
                            continue
                        found = room(path)
                        if found > best + 1e-6:
                            best, chosen = found, via
                if best >= CLEARANCE:
                    break
            if chosen is not None and best >= straight + WORTH_IT:
                plans[number][(clip_a["id"], clip_b["id"])] = (round(float(chosen[0]), 2), round(float(chosen[1]), 2))
                own[span] = _path(begin, finish, chosen, count)
    return plans


def close_calls(tracks: list[tuple[str, np.ndarray]], limit: int = 40) -> list[dict]:
    """Stretches where two dancers' hips are closer than TOO_CLOSE, from the finished motion: [{a, b, start, end, distance}]."""
    calls = []
    for index, (name_a, track_a) in enumerate(tracks):
        for name_b, track_b in tracks[index + 1:]:
            frames = min(track_a.shape[0], track_b.shape[0])
            distance = np.linalg.norm(track_a[:frames] - track_b[:frames], axis=1)
            for start, end in contact_spans(distance < TOO_CLOSE, shortest=2, bridge=int(0.3 * MODEL_FPS)):
                calls.append({"a": name_a, "b": name_b, "start": round(start / MODEL_FPS, 3), "end": round((end - 1) / MODEL_FPS, 3),
                              "distance": round(float(distance[start:end].min()), 2)})
    calls.sort(key=lambda call: call["start"])
    return calls[:limit]
