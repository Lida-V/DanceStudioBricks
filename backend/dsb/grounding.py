"""Keep the feet on the floor in MotionBricks joins. The model says when a foot is planted; this holds it there."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .motion import Skeleton, contact_spans, solve_leg, world_positions

CONTACT_CONFIDENCE = 0.5
SHORTEST_PLANT, BRIDGED_GAP = 3, 2  # frames
EASE_FRAMES = 6  # a correction fades out over this many frames around a plant
END_TAPER = 2  # frames over which every correction reaches zero at the first and last frame
MAX_SHIFT = 0.10  # metres. A foot the model drags further slides for the rest: holding it would snap the leg
SOFT_REACH = 0.97  # share of the leg length a correction may stretch to, so it does not lock the knee


def _pin(path: np.ndarray, near_start: bool, near_end: bool) -> np.ndarray:
    """Where the planted point stays, [n, 2] on the floor plane.

    With no room to ease in before an end of the join, it stays where it lands or lifts off.
    """
    count = path.shape[0]
    if near_start and near_end:
        return path[0] + np.linspace(0.0, 1.0, count)[:, None] * (path[-1] - path[0])
    if near_start:
        return np.tile(path[0], (count, 1))
    if near_end:
        return np.tile(path[-1], (count, 1))
    return np.tile(np.median(path, axis=0), (count, 1))


def _floor_shift(ankle: np.ndarray, toe: np.ndarray, heel_on: np.ndarray, toe_on: np.ndarray) -> np.ndarray:
    """Per-frame [x, z] move that holds the planted part of the foot still. Fades out around each plant."""
    frames = ankle.shape[0]
    shift, planted = np.zeros((frames, 2)), np.zeros(frames, dtype=bool)
    for start, end in contact_spans(heel_on | toe_on, SHORTEST_PLANT, BRIDGED_GAP):
        planted[start:end] = True
        anchor = toe if 2 * int(toe_on[start:end].sum()) >= end - start else ankle
        path = anchor[start:end][:, [0, 2]]
        shift[start:end] = _pin(path, start <= EASE_FRAMES, end >= frames - EASE_FRAMES) - path
    if not planted.any():
        return shift
    shift *= np.minimum(1.0, MAX_SHIFT / np.maximum(np.linalg.norm(shift, axis=1, keepdims=True), 1e-9))
    steps, known = np.arange(frames), np.flatnonzero(planted)
    filled = np.column_stack([np.interp(steps, known, shift[known, axis]) for axis in (0, 1)])
    fade = np.clip(1.0 - np.abs(steps[:, None] - known[None]).min(axis=1) / EASE_FRAMES, 0.0, 1.0)
    return filled * (fade * fade * (3.0 - 2.0 * fade))[:, None]


def _floor_lift(ankle: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """How far this foot has to rise to stay above the floor. Never below what the phrases at either end already do."""
    low = np.minimum(ankle[:, 1], toe[:, 1])
    deficit = np.maximum(min(0.0, float(low[0]), float(low[-1])) - low, 0.0)
    return np.maximum(gaussian_filter1d(maximum_filter1d(deficit, 3), 1.0, mode="nearest"), deficit)


def _within_reach(goal: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    """Pull goals the leg could only reach by locking the knee back toward the hip. A leg the model already straightened stays as it was."""
    full = np.linalg.norm(knee - hip, axis=1) + np.linalg.norm(ankle - knee, axis=1)
    limit = np.maximum(SOFT_REACH * full, np.linalg.norm(ankle - hip, axis=1))
    return hip + (goal - hip) * np.minimum(1.0, limit / np.maximum(np.linalg.norm(goal - hip, axis=1), 1e-9))[:, None]


def plant_feet(poses: np.ndarray, contacts: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """A copy with planted feet held still and no foot under the floor.

    `contacts` is MotionBricks' [frames, 4] confidence for left heel, left toe, right heel, right toe. The first and last
    frame come back unchanged, so the join still meets both phrases exactly.
    """
    out = poses.copy()
    frames = poses.shape[0]
    positions = world_positions(poses, skeleton.parents)
    steps = np.arange(frames)
    taper = np.clip(np.minimum(steps, steps[::-1]) / END_TAPER, 0.0, 1.0)
    for number, side in enumerate(("Left", "Right")):
        ankle, toe = positions[:, skeleton.index(side + "Foot")], positions[:, skeleton.index(side + "ToeBase")]
        shift = _floor_shift(ankle, toe, contacts[:, 2 * number] > CONTACT_CONFIDENCE, contacts[:, 2 * number + 1] > CONTACT_CONFIDENCE)
        correction = taper[:, None] * np.column_stack([shift[:, 0], _floor_lift(ankle, toe), shift[:, 1]])
        active = np.flatnonzero(np.abs(correction).sum(axis=1) > 1e-6)
        if active.size:
            hip, knee = positions[active, skeleton.index(side + "Leg")], positions[active, skeleton.index(side + "Shin")]
            solve_leg(out, skeleton, side, active, _within_reach(ankle[active] + correction[active], hip, knee, ankle[active]))
    return out
