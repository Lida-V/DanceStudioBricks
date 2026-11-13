"""Song analysis: tempo, beats, bars, sections, energy and sound accents."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile
import soxr

from .components import BEAT_CHECKPOINT, usable

SR = 22050
HOP = 512


def load_mono(path: str | Path) -> np.ndarray:
    with soundfile.SoundFile(str(path)) as handle:
        rate = handle.samplerate
        data = handle.read(dtype="float32", always_2d=True).mean(axis=1)
    return soxr.resample(data, rate, SR).astype(np.float32) if rate != SR else data


CHECKPOINT = BEAT_CHECKPOINT
_tracker = None


def track_beats(y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Neural beats and downbeats (MIT, CPJKU Beat This!), or None when the checkpoint is not installed."""
    global _tracker
    if not usable("beat_this") or CHECKPOINT.stat().st_size < 81058141:
        return None
    if _tracker is None:
        import torch
        from beat_this.inference import Audio2Beats
        _tracker = Audio2Beats(checkpoint_path=str(CHECKPOINT), device="cuda" if torch.cuda.is_available() else "cpu", dbn=False)
    beats, downbeats = _tracker(y, SR)
    return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)


def _regular_grid(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit an even beat grid; returns (grid, tempo). Falls back to the tracked beats when the song drifts."""
    index = np.arange(len(times))
    slope, intercept = np.polyfit(index, times, 1)
    residual = times - (slope * index + intercept)
    # Robust refit without outliers
    keep = np.abs(residual) < max(0.05, 2.5 * float(np.std(residual)))
    if keep.sum() > 8:
        slope, intercept = np.polyfit(index[keep], times[keep], 1)
        residual = times - (slope * index + intercept)
    if float(np.percentile(np.abs(residual), 90)) < 0.06:
        return slope * index + intercept, 60.0 / float(slope)
    return times, 60.0 / float(np.median(np.diff(times)))


def _steady_grid(times: np.ndarray, period: float, duration: float) -> np.ndarray:
    """An even grid through the detected times when the song keeps a steady tempo; otherwise the times themselves.

    Missed bars in quiet passages would otherwise shift every later section boundary.
    """
    # Drift-tolerant phase: fit a line through (grid index, time) using each time's nearest grid index.
    index = np.round((times - times[0]) / period)
    slope, intercept = np.polyfit(index, times, 1)
    for _ in range(3):
        index = np.round((times - intercept) / slope)
        keep = np.abs(times - (slope * index + intercept)) < 0.2 * period
        if keep.sum() < 8:
            return _extend_grid(times, duration)
        slope, intercept = np.polyfit(index[keep], times[keep], 1)
    if float(keep.mean()) < 0.7:
        return _extend_grid(times, duration)
    start = intercept - np.floor(intercept / slope) * slope
    if start > 0.9 * slope:  # just under a whole period: the grid really starts at the top of the song
        start = max(0.0, start - slope)
    return np.arange(start, duration, slope)


def _extend_grid(beats: np.ndarray, duration: float) -> np.ndarray:
    period = float(np.median(np.diff(beats)))
    before = np.arange(beats[0] - period, -period * 0.5, -period)[::-1]
    before = before[before >= 0.0]
    after = np.arange(beats[-1] + period, duration, period)
    return np.concatenate([before, beats, after])


def _downbeat_phase(beats: np.ndarray, y: np.ndarray) -> int:
    """Which of the four beat positions is the bar start: strong low end plus harmony changing there."""
    low = librosa.feature.rms(y=librosa.effects.preemphasis(y, coef=-0.97), frame_length=2048, hop_length=HOP)[0]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=200)
    chroma = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP)
    change = np.concatenate([[0.0], np.linalg.norm(np.diff(scipy.ndimage.uniform_filter1d(chroma, 9, axis=1), axis=1), axis=0)])
    frames = np.minimum(librosa.time_to_frames(beats, sr=SR, hop_length=HOP), len(onset) - 1)
    size = min(len(frames), len(low), len(change))
    frames = frames[frames < min(len(low), len(change))]
    scores = []
    def normal(values):
        return values / (np.mean(values) + 1e-9)
    strength = normal(onset[frames]) + 0.5 * normal(low[frames]) + normal(change[frames])
    for phase in range(4):
        scores.append(float(np.mean(strength[phase::4])) if len(strength[phase::4]) else 0.0)
    _ = size
    return int(np.argmax(scores))


def _segments(bars: np.ndarray, y: np.ndarray, duration: float) -> tuple[list[int], np.ndarray]:
    """Section boundaries as bar indexes, from a self-similarity novelty curve over bars."""
    spans = np.concatenate([bars, [duration]])
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)
    frames = librosa.time_to_frames(spans, sr=SR, hop_length=HOP)
    def per_bar(feature):
        out = []
        for start, stop in zip(frames[:-1], frames[1:]):
            out.append(feature[:, start:max(start + 1, stop)].mean(axis=1))
        return np.array(out)
    bar_rms = per_bar(rms)[:, 0]
    features = np.hstack([per_bar(chroma), per_bar(mfcc)[:, 1:]])
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)
    bars_total = len(features)
    level = np.log10(bar_rms + 1e-5)[:, None]
    combined = np.hstack([features, 2.5 * (level - level.mean()) / (level.std() + 1e-9)])
    combined /= np.linalg.norm(combined, axis=1, keepdims=True) + 1e-9
    similarity = combined @ combined.T
    width = 8
    kernel = np.ones((2 * width, 2 * width))
    kernel[:width, width:] = -1
    kernel[width:, :width] = -1
    novelty = np.zeros(bars_total)
    for bar in range(bars_total):
        low, high = max(0, bar - width), min(bars_total, bar + width)
        window = similarity[low:high, low:high]
        k = kernel[width - (bar - low):width + (high - bar), width - (bar - low):width + (high - bar)]
        novelty[bar] = float(np.sum(window * k)) / k.size
    novelty = novelty - scipy.ndimage.minimum_filter1d(novelty, 33)
    # Prefer musically even bar counts.
    weights = np.ones(bars_total)
    weights[::2] = 1.15
    weights[::4] = 1.35
    weights[::8] = 1.5
    scored = novelty * weights
    peaks, _ = scipy.signal.find_peaks(scored, distance=4, prominence=max(1e-6, 0.15 * float(scored.max())))
    boundaries = sorted({0, *[int(p) for p in peaks if 3 <= p <= bars_total - 3]})
    return boundaries, bar_rms


def _label_sections(boundaries: list[int], bars_total: int, bar_rms: np.ndarray) -> list[dict]:
    edges = boundaries + [bars_total]
    sections = []
    for start, stop in zip(edges[:-1], edges[1:]):
        sections.append({"startBar": start, "endBar": stop, "energy": float(np.mean(bar_rms[start:stop]))})
    energies = np.array([s["energy"] for s in sections])
    top = np.percentile(energies, 65)
    low = np.percentile(energies, 30)
    for index, section in enumerate(sections):
        if index == 0 and section["energy"] <= np.median(energies):
            kind = "intro"
        elif index == len(sections) - 1 and section["energy"] <= np.median(energies):
            kind = "outro"
        elif section["energy"] >= top:
            kind = "chorus"
        elif section["energy"] <= low:
            kind = "interlude"
        else:
            kind = "verse"
        section["kind"] = kind
    for index in range(len(sections) - 1):
        if sections[index]["kind"] == "verse" and sections[index + 1]["kind"] == "chorus":
            if sections[index]["endBar"] - sections[index]["startBar"] <= 12:
                sections[index]["kind"] = "build"
    return sections


LABELS = {"intro": "イントロ", "verse": "Aメロ", "build": "Bメロ", "chorus": "サビ", "interlude": "間奏", "outro": "アウトロ"}


def _accents(y: np.ndarray, beats: np.ndarray) -> list[dict]:
    """Sound accents to dance on: strong onsets on the beat grid, and only very strong ones off it."""
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, aggregate=np.median)
    low = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=180)
    high = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmin=1500)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=SR, hop_length=HOP)
    local = scipy.ndimage.uniform_filter1d(onset, 87) + 1e-6  # about two seconds
    ratio = onset / local
    period = float(np.median(np.diff(beats)))
    peaks, _ = scipy.signal.find_peaks(onset, distance=max(1, int(0.45 * period * SR / HOP)), height=0.0)
    out = []
    for peak in peaks:
        strength = float(ratio[peak])
        nearest = float(np.min(np.abs(beats - times[peak])))
        on_beat = nearest < 0.12 * period
        if strength < (1.6 if on_beat else 2.6):
            continue
        out.append({"t": round(float(times[peak]), 3), "strength": round(strength, 2), "kind": "low" if low[peak] >= high[peak] else "high",
                    "onBeat": bool(on_beat), "kime": bool(strength >= 3.2)})
    return out


def _embedded_sections(path, bars: np.ndarray, duration: float, bar_rms: np.ndarray) -> list[dict] | None:
    """Sections from EmbeddingGemma 2 similarity over two-bar steps; None when the supporter is unavailable."""
    from .embed import EmbedUnavailable, audio_spans

    bar_seconds = float(np.median(np.diff(bars)))
    length = max(8.0, 2.0 * bar_seconds)
    steps = [i for i in range(0, len(bars), 2) if bars[i] + 2.0 < duration]
    spans = [(float(bars[i]), min(duration, float(bars[i]) + length)) for i in steps]
    try:
        vectors = audio_spans(str(path), spans)
    except EmbedUnavailable:
        return None
    similarity = vectors @ vectors.T
    half = 4
    count = len(steps)
    novelty = np.zeros(count)
    for center in range(1, count - 1):
        before = range(max(0, center - half), center)
        after = range(center, min(count, center + half))
        cross = np.mean([similarity[a, b] for a in before for b in after])
        inner = np.mean([similarity[a, b] for group in (before, after) for a in group for b in group if a != b] or [1.0])
        novelty[center] = inner - cross
    level = np.log10(bar_rms[steps] + 1e-5)
    change = np.abs(np.array([level[min(count, i + 2):min(count, i + 4)].mean() - level[max(0, i - 2):i].mean()
                              if 2 <= i <= count - 3 else 0.0 for i in range(count)]))
    def scaled(values):
        return (values - np.median(values)) / (np.std(values) + 1e-9)
    score = scaled(novelty) + 0.35 * scaled(change)
    weights = np.array([1.2 if steps[i] % 8 == 0 else (1.1 if steps[i] % 4 == 0 else 1.0) for i in range(count)])
    score = (score - score.min()) * weights
    peaks, _ = scipy.signal.find_peaks(score, distance=2, prominence=0.45)
    boundaries = sorted({0, *[steps[p] for p in peaks if 2 <= steps[p] <= len(bars) - 3]})
    edges = boundaries + [len(bars)]
    sections = []
    for start, stop in zip(edges[:-1], edges[1:]):
        members = [i for i, step in enumerate(steps) if start <= step < stop]
        center = vectors[members].mean(axis=0) if members else vectors[0]
        sections.append({"startBar": start, "endBar": stop, "energy": float(np.mean(bar_rms[start:stop])), "vector": center / np.linalg.norm(center)})
    return _label_with_repeats(sections)


def _label_with_repeats(sections: list[dict]) -> list[dict]:
    count = len(sections)
    matrix = np.array([[float(a["vector"] @ b["vector"]) for b in sections] for a in sections])
    off = matrix[~np.eye(count, dtype=bool)] if count > 1 else np.array([1.0])
    energies = np.array([s["energy"] for s in sections])
    lengths = np.array([s["endBar"] - s["startBar"] for s in sections], dtype=float)
    median = float(np.median(energies))
    groups = list(range(count))
    threshold = float(off.max())
    chorus_group = None
    # Tighten the similarity needed until the loudest repeated group is a plausible share of the song.
    for percentile in (75, 80, 85, 90, 95):
        threshold = float(np.percentile(off, percentile))
        groups = list(range(count))
        for i in range(count):
            for j in range(i):
                if j != i - 1 and matrix[i, j] >= threshold:
                    old, new = groups[i], groups[j]
                    groups = [new if g == old else g for g in groups]
        candidates = {}
        for group in set(groups):
            members = [i for i, g in enumerate(groups) if g == group]
            if len(members) >= 2 and lengths[members].sum() <= 0.5 * lengths.sum():
                candidates[group] = float(np.mean(energies[members]))
        if candidates:
            chorus_group = max(candidates, key=candidates.get)
            break
    for index, section in enumerate(sections):
        previous = [j for j in range(index) if matrix[index, j] >= threshold and j != index - 1]
        section["repeatOf"] = int(max(previous, key=lambda j: matrix[index, j])) if previous else -1
        if chorus_group is not None and groups[index] == chorus_group:
            kind = "chorus"
        elif chorus_group is None and section["energy"] >= np.percentile(energies, 70):
            kind = "chorus"
        elif index == 0:
            kind = "intro"
        elif index == count - 1 and section["energy"] <= median:
            kind = "outro"
        elif section["endBar"] - section["startBar"] <= 6 and section["energy"] <= median:
            kind = "interlude"
        else:
            kind = "verse"
        section["kind"] = kind
    for index in range(count - 1):
        if sections[index]["kind"] == "verse" and sections[index + 1]["kind"] == "chorus" and sections[index]["endBar"] - sections[index]["startBar"] <= 12:
            sections[index]["kind"] = "build"
    for section in sections:
        section.pop("vector", None)
    return sections


def _bar_levels(bars: np.ndarray, y: np.ndarray, duration: float) -> np.ndarray:
    spans = np.concatenate([bars, [duration]])
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    frames = librosa.time_to_frames(spans, sr=SR, hop_length=HOP)
    return np.array([rms[a:max(a + 1, b)].mean() for a, b in zip(frames[:-1], frames[1:])])


def analyze(path: str | Path, tempo_factor: float | None = None, points: int = 2400, use_embeddings: bool = True) -> dict:
    """tempo_factor: None chooses half time for very fast songs; 1, 2 or 0.5 force a reading."""
    y = load_mono(path)
    duration = len(y) / SR
    tracked = track_beats(y)
    engine_name = "beat_this"
    if tracked is not None and len(tracked[0]) >= 16 and len(tracked[1]) >= 4:
        beats, downbeats = tracked
        # The tracker's times sit on a 20 ms grid, so measure the period across eight beats.
        tempo = 60.0 * 8.0 / float(np.median(beats[8:] - beats[:-8]))
    else:
        engine_name = "librosa"
        onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, start_bpm=120.0, tightness=120)
        raw = librosa.frames_to_time(frames, sr=SR, hop_length=HOP)
        if len(raw) < 16:
            raise ValueError("拍を検出できませんでした")
        beats, tempo = _regular_grid(raw)
        downbeats = beats[_downbeat_phase(beats, y)::4]
    factor = tempo_factor if tempo_factor else (0.5 if tempo >= 170.0 else 1.0)
    if factor == 2.0:
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2.0]))
        tempo *= 2.0
    elif factor == 0.5:
        beats = beats[::2]
        downbeats = downbeats[::2]
        tempo /= 2.0
    beat_period = 60.0 / tempo
    beats = _steady_grid(beats, beat_period, duration)
    bars = _steady_grid(downbeats, beat_period * 4.0, duration)
    bars = bars[bars < duration - 0.5]
    if len(bars) < 8:
        raise ValueError("小節を検出できませんでした")
    bar_rms = _bar_levels(bars, y, duration)
    sections = _embedded_sections(path, bars, duration, bar_rms) if use_embeddings else None
    section_engine = "embeddinggemma2" if sections else "self-similarity"
    if not sections:
        boundaries, bar_rms = _segments(bars, y, duration)
        sections = _label_sections(boundaries, len(bars), bar_rms)
        for section in sections:
            section["repeatOf"] = -1
    bar_seconds = float(np.median(np.diff(bars)))
    spans = np.concatenate([bars, [duration]])
    for section in sections:
        section["start"] = round(float(spans[section["startBar"]]), 3)
        section["end"] = round(float(spans[section["endBar"]]), 3)
        section["label"] = LABELS[section["kind"]]
    numbering: dict[str, int] = {}
    for section in sections:
        numbering[section["kind"]] = numbering.get(section["kind"], 0) + 1
        if numbering[section["kind"]] > 1:
            section["label"] += f" {numbering[section['kind']]}"
    # Waveform summary for drawing: peak per column.
    step = max(1, len(y) // points)
    wave = np.abs(y[: len(y) - len(y) % step]).reshape(-1, step).max(axis=1)
    level = bar_rms / (bar_rms.max() + 1e-9)
    return {
        "engines": {"beats": engine_name, "sections": section_engine}, "tempoFactor": factor,
        "duration": round(duration, 3), "tempo": round(float(tempo), 2), "beatSeconds": round(60.0 / float(tempo), 5),
        "barSeconds": round(bar_seconds, 5), "meter": 4,
        "beats": [round(float(t), 4) for t in beats], "bars": [round(float(t), 4) for t in bars],
        "sections": sections, "energyPerBar": [round(float(v), 4) for v in level],
        "accents": _accents(y, beats), "wave": [round(float(v), 3) for v in wave],
    }
