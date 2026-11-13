"""Time mapping for a clip: uniform speed plus partial stretching that lands motion accents on sound accents."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator

LOCAL_SPEED_RANGE = (0.7, 1.6)
SEARCH_SECONDS = 0.16
PIN_EDGE = 0.02  # a pin stays this far from the phrase ends and from the pins beside it, in both source and timeline seconds
ON_SOUND_SECONDS = 0.03  # a pin this close to a sound accent is shown as joined to it


@dataclass
class Warp:
    start: float
    end: float
    source_duration: float
    knots_time: np.ndarray
    knots_source: np.ndarray
    matches: list[dict] = field(default_factory=list)
    source_start: float = 0.0

    @property
    def speed(self) -> float:
        return (self.source_duration - self.source_start) / max(1e-6, self.end - self.start)

    def source_at(self, times: np.ndarray) -> np.ndarray:
        return np.clip(PchipInterpolator(self.knots_time, self.knots_source)(times), self.source_start, self.source_duration)

    def local_speed_range(self) -> tuple[float, float]:
        return _rate_range(self.knots_time, self.knots_source)

    def curve(self, count: int = 48) -> list[float]:
        samples = np.linspace(self.start, self.end, count)
        rate = PchipInterpolator(self.knots_time, self.knots_source).derivative()(samples)
        return [round(float(v), 3) for v in rate]


def _rate_range(times: np.ndarray, sources: np.ndarray) -> tuple[float, float]:
    rate = PchipInterpolator(times, sources).derivative()(np.linspace(times[0], times[-1], 240))
    return float(rate.min()), float(rate.max())


def usable_pins(pins, start: float, end: float, source_duration: float, source_start: float = 0.0) -> list[tuple[float, float]]:
    """The (source, timeline) pins that can hold together, in order. One that is out of the phrase, or not after the one before it, is left out."""
    kept: list[tuple[float, float]] = []
    for source, when in sorted((float(s), float(t)) for s, t in pins):
        last_source, last_when = kept[-1] if kept else (source_start, start)
        if last_source + PIN_EDGE < source < source_duration - PIN_EDGE and last_when + PIN_EDGE < when < end - PIN_EDGE:
            kept.append((source, when))
    return kept


def build(start: float, end: float, source_duration: float, motion_accents: list[float], sound_times: np.ndarray, strength: float,
          pins=(), source_start: float = 0.0) -> Warp:
    """`strength` is 0..1. Accents that would need a local speed outside the range keep their uniform time.

    `pins` are (source seconds, timeline seconds) pairs placed by hand. They always hold, at whatever local speed they need,
    and the automatic matching works in the stretches between them. `source_start`..`source_duration` trims the phrase.
    """
    speed = max(1e-6, source_duration - source_start) / max(1e-6, end - start)
    fixed = usable_pins(pins, start, end, source_duration, source_start)
    base_time = np.array([start, *(when for _, when in fixed), end])
    base_source = np.array([source_start, *(source for source, _ in fixed), source_duration])

    def unmatched_time(sources: np.ndarray) -> np.ndarray:
        """Where source times land with only the pins in place."""
        if not fixed:
            return start + (sources - source_start) / speed
        grid = np.linspace(start, end, 1200)
        return np.interp(sources, PchipInterpolator(base_time, base_source)(grid), grid)

    in_trim = [accent for accent in motion_accents if source_start <= accent <= source_duration]
    free = [accent for accent in in_trim if all(abs(accent - source) > 1e-3 for source, _ in fixed)]
    candidates = []
    for accent, uniform in zip(free, unmatched_time(np.array(free, dtype=np.float64))):
        uniform = float(uniform)
        sound = None
        if len(sound_times):
            nearest = float(sound_times[int(np.argmin(np.abs(sound_times - uniform)))])
            sound = nearest if abs(nearest - uniform) <= SEARCH_SECONDS else None
        candidates.append({"source": accent, "uniform": uniform, "sound": sound})

    def assemble(active: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        times, sources = list(base_time), list(base_source)
        for item in sorted(active, key=lambda c: c["source"]):
            when = item["uniform"] + strength * (item["sound"] - item["uniform"])
            at = bisect_right(sources, item["source"])
            if times[at - 1] + 1e-3 < when < times[at] - 1e-3 and sources[at - 1] + 1e-3 < item["source"] < sources[at] - 1e-3:
                times.insert(at, when)
                sources.insert(at, item["source"])
        return np.array(times), np.array(sources)

    active = [c for c in candidates if c["sound"] is not None and source_start < c["source"] < source_duration] if strength > 0 else []
    pinned_low, pinned_high = _rate_range(base_time, base_source) if fixed else (speed, speed)
    low, high = min(LOCAL_SPEED_RANGE[0], speed, pinned_low), max(LOCAL_SPEED_RANGE[1], speed, pinned_high)
    # Keep whatever stays inside the local speed range, removing the largest displacement first.
    while active:
        slowest, fastest = _rate_range(*assemble(active))
        if slowest >= low - 1e-6 and fastest <= high + 1e-6:
            break
        worst = max(active, key=lambda c: abs(c["sound"] - c["uniform"]))
        active.remove(worst)
    times, sources = assemble(active)
    used = {id(c) for c in active}
    matches = []
    for item in candidates:
        applied = id(item) in used
        at = item["uniform"] + (strength * (item["sound"] - item["uniform"]) if applied else 0.0)
        matches.append({"t": round(at, 3), "sound": None if item["sound"] is None else round(item["sound"], 3), "matched": applied and strength > 0,
                        "source": round(item["source"], 3), "pinned": False})
    for source, when in fixed:
        nearest = float(sound_times[int(np.argmin(np.abs(sound_times - when)))]) if len(sound_times) else None
        on_sound = nearest is not None and abs(nearest - when) <= ON_SOUND_SECONDS
        matches.append({"t": round(when, 3), "sound": round(nearest, 3) if on_sound else None, "matched": True, "source": round(source, 3), "pinned": True})
    matches.sort(key=lambda m: m["t"])
    return Warp(start, end, source_duration, times, sources, matches, source_start)
