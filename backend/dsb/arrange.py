"""Automatic placement: choose phrases from the flow rules, sound accents, local energy and repeated sections."""
from __future__ import annotations

import random
import re
import uuid

import numpy as np

from .library import CATEGORIES
from .search import _score_token, clean_query_token, normalize_token

STAGE_X = (-4.0, 4.0)
STAGE_Z = (0.0, 6.0)
TARGET_MOVE_SPEED = 1.4
FAST_MOVE_SPEED = 1.6
MIN_JOIN_SECONDS = 0.6
AUTO_SPEED_RANGE = (0.8, 1.35)

# The default flow dictionary: section kind -> preferred categories in order, energy goal, travel per reposition (metres).
DEFAULT_FLOW = {
    "intro": {"categories": ["pose", "wave", "groove"], "energy": 2, "move": 0.2, "text": "静 → 立ち上がり"},
    "verse": {"categories": ["groove", "foot", "wave"], "energy": 3, "move": 0.45, "text": "リズム刻み ・ 左右にステップ"},
    "build": {"categories": ["wave", "groove", "power"], "energy": 3, "move": 0.3, "text": "溜め ・ うねりで上昇"},
    "chorus": {"categories": ["power", "groove", "foot"], "energy": 5, "move": 0.7, "text": "大技 ・ 見せ場で左右へ展開"},
    "interlude": {"categories": ["foot", "groove", "wave"], "energy": 3, "move": 0.5, "text": "足技ソロ"},
    "outro": {"categories": ["pose", "wave", "groove"], "energy": 2, "move": 0.25, "text": "余韻 ・ 中央へ戻す"},
}

GENRE_TAGS = (
    "K-POP", "アイドル系", "J-POP", "アニソン", "ヒップホップ", "ストリート",
    "ハウス", "EDM", "ファンク", "バラード", "ワッキング", "ヴォーグ",
)
_MOOD_TAGS = ("女性的", "男性的", "中性的")
_NORM_GENRES = {normalize_token(g) for g in GENRE_TAGS}
_NORM_MOODS = {normalize_token(m) for m in _MOOD_TAGS}
_STEM_SUFFIX_RE = re.compile(r"(?:[\s_-]+[mfn])$", re.IGNORECASE)


def clean_filter_tags(filter_tags: list[str] | None) -> list[str]:
    """Normalize and deduplicate user-supplied genre/mood/action filter tags."""
    if not filter_tags:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in filter_tags:
        for part in re.split(r"[,、，・/／|｜]+", str(raw)):
            s = part.strip()
            if not s:
                continue
            cleaned = clean_query_token(s) or s
            norm = normalize_token(cleaned)
            if norm and norm not in seen:
                seen.add(norm)
                out.append(cleaned)
    return out


def _matches_single_filter_tag(item: dict, tag: str, categories: dict[str, str] = CATEGORIES) -> bool:
    cleaned = clean_query_token(tag) or tag.strip()
    norm = normalize_token(cleaned)
    if not norm:
        return False
    raw_tags = item.get("tags", [])
    norm_tags = {normalize_token(t) for t in raw_tags}
    cat = str(item.get("category", ""))
    cat_label = categories.get(cat, "")
    if norm in norm_tags or norm == normalize_token(cat) or (cat_label and norm == normalize_token(cat_label)):
        return True
    # Canonical genre / mood chip names match strictly by item tags so "K-POP only" never leaks unrelated genres.
    if norm in _NORM_GENRES or norm in _NORM_MOODS:
        return False
    hay = normalize_token(f"{item.get('name', '')} {item.get('description', '')}")
    if norm in hay:
        return True
    return _score_token(item, cleaned, cat_label) >= 0.60


def filter_phrases_by_tags(
    items: list[dict],
    filter_tags: list[str] | None,
    categories: dict[str, str] = CATEGORIES,
) -> list[dict]:
    """Filter library phrases by genre, mood, and/or action/feature tags.

    Tags within the same facet (genres / moods / actions) combine with OR, while different facets combine with AND
    (e.g. `["K-POP", "キック"]` -> K-POP kick motions; `["K-POP", "アイドル系"]` -> K-POP or Idol motions).
    """
    cleaned = clean_filter_tags(filter_tags)
    if not cleaned:
        return list(items)
    genres = [t for t in cleaned if normalize_token(t) in _NORM_GENRES]
    moods = [t for t in cleaned if normalize_token(t) in _NORM_MOODS]
    others = [t for t in cleaned if normalize_token(t) not in _NORM_GENRES and normalize_token(t) not in _NORM_MOODS]

    matched: list[dict] = []
    for item in items:
        if genres and not any(_matches_single_filter_tag(item, g, categories) for g in genres):
            continue
        if moods and not any(_matches_single_filter_tag(item, m, categories) for m in moods):
            continue
        if others and not any(_matches_single_filter_tag(item, o, categories) for o in others):
            continue
        matched.append(item)
    return matched


def base_stem(item: dict) -> str:
    """Motion family key stripping M/F/N mood suffixes so variants do not repeat back-to-back."""
    raw = str(item.get("name") or item.get("id") or "").strip().lower()
    return _STEM_SUFFIX_RE.sub("", raw)


def has_ending_pose(item: dict) -> bool:
    """True when the phrase culminates in a held pose, spin-stop, turn-finish or outro settle (tail should not be trimmed)."""
    if item.get("category") == "pose":
        return True
    tags = set(item.get("tags", []))
    if tags & {"キメ", "アウトロ", "スピン", "ターン"}:
        return True
    desc = item.get("description", "")
    if any(k in desc for k in ("止まる", "動きを収めて", "フィニッシュ", "ポーズで止", "静止")):
        return True
    name_low = str(item.get("name", "")).lower()
    return any(k in name_low for k in ("finish", "settle", "spin stop", "spin_stop", "slow rise hit", "slow_rise_hit"))


def join_seconds(distance: float, bar_seconds: float, speed: float = TARGET_MOVE_SPEED) -> float:
    """Length of a join for this distance, rounded up to quarter bars and never below one generation."""
    quarter = bar_seconds / 4.0
    needed = max(distance / speed, MIN_JOIN_SECONDS)
    return float(np.ceil(needed / quarter - 1e-6) * quarter)


def accent_alignment(accents: list[float], source_duration: float, start: float, end: float, sound: np.ndarray,
                     sound_weights: np.ndarray | None = None) -> float:
    valid = [a for a in accents if a <= source_duration - 0.02]
    if not valid or len(sound) == 0 or end <= start:
        return 0.5
    speed = source_duration / (end - start)
    scores = []
    for accent in valid:
        when = start + accent / speed
        diffs = np.abs(sound - when)
        idx = int(np.argmin(diffs))
        err = float(diffs[idx])
        base = 1.0 if err < 0.05 else max(0.0, 1.0 - (err - 0.05) / 0.2)
        if sound_weights is not None and idx < len(sound_weights):
            base = min(1.0, base * float(sound_weights[idx]))
        scores.append(base)
    return float(np.mean(scores))


def _role_affinity(item: dict, section_kind: str | None, target_energy: float, has_kime: bool, is_section_end: bool) -> float:
    if section_kind is None:
        return 0.0
    tags = set(item.get("tags", []))
    desc = item.get("description", "")
    name_low = str(item.get("name", "")).lower()
    bonus = 0.0

    is_outro_clip = "アウトロ" in tags or "outro" in name_low or "動きを収めて" in desc
    is_intro_clip = "イントロ" in tags or "intro" in name_low

    if is_outro_clip:
        if section_kind == "outro":
            bonus += 0.34 if is_section_end else 0.20
        else:
            bonus -= 0.55
    if is_intro_clip:
        if section_kind == "intro":
            bonus += 0.32
        elif section_kind in ("chorus", "build", "interlude"):
            bonus -= 0.35
        else:
            bonus -= 0.18

    if section_kind == "build" and ("ビルドアップ" in tags or "溜め" in tags or "徐々に" in desc or "build" in name_low):
        bonus += 0.26
    if section_kind == "chorus" and ("サビ" in tags or "コンボ" in desc or "爆発" in tags or "combo" in name_low):
        bonus += 0.20
    if "バラード" in tags:
        bonus += 0.16 if target_energy <= 2.4 else (-0.22 if target_energy >= 4.0 else 0.0)
    if (has_kime or is_section_end) and has_ending_pose(item) and not is_outro_clip and not is_intro_clip:
        bonus += 0.18
    return bonus


def score_phrase(
    item: dict,
    flow: dict,
    recent: list[str],
    start: float,
    end: float,
    sound: np.ndarray,
    *,
    section_kind: str | None = None,
    local_energy: float | None = None,
    slot_bars: int | None = None,
    preferred_mood: str | None = None,
    recent_stems: list[str] | None = None,
    sound_weights: np.ndarray | None = None,
    has_kime: bool = False,
    is_section_end: bool = False,
) -> float:
    categories = flow["categories"]
    if item["category"] in categories:
        category = (len(categories) - categories.index(item["category"])) / len(categories)
    elif (is_section_end or has_kime) and item["category"] == "pose":
        category = 0.65
    else:
        category = 0.0

    target_energy = float(flow["energy"])
    if local_energy is not None:
        song_energy_5 = 1.0 + 4.0 * float(np.clip(local_energy, 0.0, 1.0))
        target_energy = 0.55 * target_energy + 0.45 * song_energy_5
    energy = max(0.0, 1.0 - abs(float(item["energy"]) - target_energy) / 4.0)

    duration = float(item["duration"])
    alignment = accent_alignment(item.get("accents", []), duration, start, end, sound, sound_weights)

    if item["id"] in recent:
        novelty = 0.0 if item["id"] == (recent[-1] if recent else None) else 0.15
    elif recent_stems:
        stem = base_stem(item)
        if stem == recent_stems[-1]:
            novelty = 0.05
        elif stem in recent_stems[-5:]:
            novelty = 0.35
        else:
            novelty = 1.0
    else:
        novelty = 1.0

    base_score = 0.36 * category + 0.24 * energy + 0.25 * alignment + 0.15 * novelty
    if section_kind is None and slot_bars is None and preferred_mood is None and recent_stems is None:
        return base_score

    extras = _role_affinity(item, section_kind, target_energy, has_kime, is_section_end)

    # Strong anti-repetition penalty across recent 6 slots
    if item["id"] in recent[-3:]:
        extras -= 0.32
    elif item["id"] in recent:
        extras -= 0.18
    elif recent_stems and base_stem(item) in recent_stems[-3:]:
        extras -= 0.24

    # Reward filling the slot naturally (minimizing leftover join gap and speed distortion)
    if end > start:
        speed = duration / (end - start)
        extras -= 0.14 * min(1.5, abs(float(np.log(max(speed, 0.2) / 1.02))))
    if slot_bars is not None and item.get("bars"):
        item_bars = int(item["bars"])
        if item_bars == slot_bars:
            extras += 0.16
        elif slot_bars >= 4 and item_bars < 4 and section_kind not in ("intro", "outro"):
            extras -= 0.18

    if preferred_mood:
        tags = set(item.get("tags", []))
        if preferred_mood in tags:
            extras += 0.10
        elif "中性的" in tags:
            extras += 0.03
        elif (preferred_mood == "女性的" and "男性的" in tags) or (preferred_mood == "男性的" and "女性的" in tags):
            extras -= 0.14

    return base_score + extras


def candidates_for(
    library_items: list[dict],
    flow: dict,
    recent: list[str],
    start: float,
    end: float,
    sound: np.ndarray,
    limit: int = 6,
    *,
    section_kind: str | None = None,
    local_energy: float | None = None,
    slot_bars: int | None = None,
    preferred_mood: str | None = None,
    recent_stems: list[str] | None = None,
    sound_weights: np.ndarray | None = None,
    has_kime: bool = False,
    is_section_end: bool = False,
    filter_tags: list[str] | None = None,
) -> list[dict]:
    cleaned_tags = clean_filter_tags(filter_tags)
    pool_items = filter_phrases_by_tags(library_items, cleaned_tags) if cleaned_tags else library_items
    has_action_filter = any(
        normalize_token(t) not in _NORM_GENRES and normalize_token(t) not in _NORM_MOODS
        for t in cleaned_tags
    )
    allow_all_filtered = bool(cleaned_tags) and (
        has_action_filter or sum(1 for i in pool_items if i.get("usage") == "auto") < 4
    )
    scored = []
    for item in pool_items:
        if not allow_all_filtered and item.get("usage") != "auto" and not ((is_section_end or has_kime) and item.get("category") == "pose"):
            continue
        s = score_phrase(
            item,
            flow,
            recent,
            start,
            end,
            sound,
            section_kind=section_kind,
            local_energy=local_energy,
            slot_bars=slot_bars,
            preferred_mood=preferred_mood,
            recent_stems=recent_stems,
            sound_weights=sound_weights,
            has_kime=has_kime,
            is_section_end=is_section_end,
        )
        scored.append({"phraseId": item["id"], "score": round(float(s), 3)})
    scored.sort(key=lambda c: -c["score"])
    return scored[:limit]


def next_position(current: tuple[float, float], move: float, rng: random.Random, kind: str = "verse") -> tuple[float, float]:
    if move <= 0.05:
        return current
    if kind == "outro":
        dx, dz = -current[0], 3.0 - current[1]
        dist = float(np.hypot(dx, dz))
        if dist <= move or dist < 1e-6:
            return 0.0, 3.0
        scale = move / dist
        return round(float(current[0] + dx * scale), 2), round(float(current[1] + dz * scale), 2)
    if kind == "build" and abs(current[0]) > 0.8:
        step_x = -np.sign(current[0]) * min(abs(current[0]), move * 0.85)
        step_z = (3.0 - current[1]) * 0.35
        return round(float(current[0] + step_x), 2), round(float(np.clip(current[1] + step_z, 2.3, 3.7)), 2)

    x_lo, x_hi = max(STAGE_X[0] + 0.8, -2.4), min(STAGE_X[1] - 0.8, 2.4)
    z_lo, z_hi = max(STAGE_Z[0] + 1.8, 2.2), min(STAGE_Z[1] - 1.8, 3.8)
    for _ in range(24):
        direction = -1.0 if current[0] > 0.9 else (1.0 if current[0] < -0.9 else rng.choice((-1.0, 1.0)))
        dx = direction * move * rng.uniform(0.65, 1.0)
        dz = move * rng.uniform(-0.35, 0.35)
        x, z = current[0] + dx, current[1] + dz
        if x_lo <= x <= x_hi and z_lo <= z <= z_hi:
            return round(float(x), 2), round(float(z), 2)
    x = current[0] * 0.5
    z = 3.0 + (current[1] - 3.0) * 0.5
    return round(float(x), 2), round(float(z), 2)


def _infer_preferred_mood(usable: list[dict], existing: list[dict], by_id: dict[str, dict], song: dict, rng: random.Random) -> str | None:
    counts = {m: 0 for m in _MOOD_TAGS}
    for clip in existing:
        item = by_id.get(clip.get("phraseId", ""))
        if item:
            for m in _MOOD_TAGS:
                if m in item.get("tags", []):
                    counts[m] += 1
    if max(counts.values(), default=0) > 0:
        return max(_MOOD_TAGS, key=lambda m: counts[m])

    available = [m for m in _MOOD_TAGS if any(m in i.get("tags", []) for i in usable)]
    if not available:
        return None
    accents = song.get("accents", [])
    high_hits = sum(1 for a in accents if a.get("kind") == "high")
    low_hits = sum(1 for a in accents if a.get("kind") == "low")
    if high_hits > low_hits * 1.25 and "女性的" in available:
        weights = [0.55 if m == "女性的" else (0.30 if m == "中性的" else 0.15) for m in available]
    elif low_hits > high_hits * 1.25 and "男性的" in available:
        weights = [0.55 if m == "男性的" else (0.30 if m == "中性的" else 0.15) for m in available]
    else:
        weights = [1.0 for _ in available]
    return rng.choices(available, weights=weights, k=1)[0]


def _sound_arrays(song: dict) -> tuple[np.ndarray, np.ndarray, set[int]]:
    accents = song.get("accents", [])
    if not accents:
        return np.zeros(0), np.zeros(0), set()
    times = np.array([float(a["t"]) for a in accents])
    weights = np.array([
        1.30 if a.get("kind") == "kime" else (1.12 if a.get("onBeat") else 1.0)
        for a in accents
    ])
    bar_sec = float(song.get("barSeconds") or 2.0)
    kime_bars = {int(float(a["t"]) // max(bar_sec, 0.2)) for a in accents if a.get("kind") == "kime"}
    return times, weights, kime_bars


def _local_bar_energy(song: dict, start_bar: int, end_bar: int) -> float | None:
    epb = song.get("energyPerBar")
    if not epb:
        return None
    lo = max(0, min(start_bar, len(epb) - 1))
    hi = max(lo + 1, min(end_bar, len(epb)))
    return float(np.mean(epb[lo:hi]))


def _fit_pool(candidates: list[dict], start: float, latest: float) -> list[tuple[dict, float]]:
    if latest - start <= 0.25:
        return []
    fitted = [(i, min(latest, start + i["duration"] / AUTO_SPEED_RANGE[0])) for i in candidates]
    return [
        (i, e)
        for i, e in fitted
        if i["duration"] / (e - start) <= AUTO_SPEED_RANGE[1]
        # Avoid picking ultra-short clips that stretch to 0.8x and leave >42% of the slot as interpolation
        and (latest - e) <= max(0.65, 0.38 * (latest - start))
    ]


def arrange(song: dict, items: list[dict], existing: list[dict], flow_table: dict, match: int, seed: int,
            range_seconds: tuple[float, float] | None = None, filter_tags: list[str] | None = None) -> list[dict]:
    """Returns the new clip list. Locked clips and clips outside the range stay exactly as they are."""
    rng = random.Random(seed)
    cleaned_tags = clean_filter_tags(filter_tags)
    pool_items = filter_phrases_by_tags(items, cleaned_tags) if cleaned_tags else list(items)
    if cleaned_tags and not pool_items:
        raise ValueError(f"指定されたジャンル・タグ（{'・'.join(cleaned_tags)}）に一致するフレーズがありません")
    has_action_filter = any(
        normalize_token(t) not in _NORM_GENRES and normalize_token(t) not in _NORM_MOODS
        for t in cleaned_tags
    )
    usable = [i for i in pool_items if i.get("usage") == "auto"]
    if cleaned_tags and (has_action_filter or len(usable) < 4):
        usable = list(pool_items)
    if not usable:
        raise ValueError("自動配置に使えるフレーズがありません（2小節と4小節のフレーズを読み込んでください）")
    usable_ids = {i["id"] for i in usable}
    pose_punct = [
        i for i in pool_items
        if i["id"] not in usable_ids and i.get("usage") == "manual" and i.get("category") == "pose" and 1.8 <= float(i.get("duration", 0.0)) <= 3.2
    ]
    bars = list(song["bars"]) + [song["duration"]]
    bar_seconds = float(song.get("barSeconds") or (bars[1] - bars[0] if len(bars) > 1 else 2.0))
    min_join = join_seconds(0.0, bar_seconds)
    two_bar_room = max(0.2, 2.0 * bar_seconds - min_join)
    fitting_two_count = sum(1 for i in usable if float(i["duration"]) / two_bar_room <= AUTO_SPEED_RANGE[1])
    auto_fits_two = fitting_two_count >= max(3, len(usable) // 5)
    punct_fits_two = auto_fits_two or any(
        float(i["duration"]) / two_bar_room <= AUTO_SPEED_RANGE[1] and float(i["duration"]) / AUTO_SPEED_RANGE[0] >= two_bar_room * 0.65
        for i in pose_punct
    )
    has_four_bar = any(int(i.get("bars", 2)) == 4 for i in usable)

    sound, sound_weights, kime_bars = _sound_arrays(song)
    by_id = {i["id"]: i for i in items}
    low, high = range_seconds if range_seconds else (-1.0, song["duration"] + 1.0)
    kept = [c for c in existing if c.get("locked") or c["end"] <= low or c["start"] >= high]
    kept.sort(key=lambda c: c["start"])
    preferred_mood = _infer_preferred_mood(usable, kept, by_id, song, rng)

    clips = list(kept)
    position = (0.0, 3.0)
    recent: list[str] = []
    recent_stems: list[str] = []
    repeat_memory: dict[int, list[tuple[str, int]]] = {}
    bars_since_move = 0

    def blocked(start: float, end: float) -> bool:
        return any(c["start"] < end and start < c["end"] for c in kept)

    cursor = 0
    for section_index, section in enumerate(song["sections"]):
        kind = section["kind"]
        flow = flow_table.get(kind, DEFAULT_FLOW["verse"])
        bar = max(section["startBar"], cursor)
        stop = section["endBar"]
        chosen_here: list[tuple[str, int]] = []
        reuse = repeat_memory.get(section.get("repeatOf", -1), [])
        slot_number = 0
        while bar < stop and bar < len(bars) - 1:
            remaining = stop - bar
            if remaining < 2:
                break
            start = bars[bar]
            earlier_kept = [c for c in kept if c["end"] <= start + 1e-3]
            if earlier_kept and slot_number == 0 and bars_since_move == 0:
                last_k = earlier_kept[-1]
                position = (float(last_k["x"]), float(last_k["z"]))

            raw_move = float(flow.get("move", 0.4))
            move_interval = 6 if kind == "chorus" else 8
            if kind == "intro":
                should_move = False
            elif kind == "outro":
                should_move = float(np.hypot(position[0], position[1] - 3.0)) > 0.35
            else:
                should_move = raw_move > 0.05 and (
                    (slot_number == 0 and section_index > 0 and bars_since_move >= 4)
                    or bars_since_move >= move_interval
                )
            step_budget = min(raw_move * 0.65, 0.95) if kind != "outro" else max(raw_move, 0.6)
            target = next_position(position, step_budget, rng, kind) if should_move else position
            distance = float(np.hypot(target[0] - position[0], target[1] - position[1]))

            reused_entry = reuse[slot_number] if slot_number < len(reuse) else None
            if reused_entry is not None and reused_entry[1] <= remaining:
                ordered_sizes = [reused_entry[1], 2, 3, 4]
            else:
                prefer_four = (
                    has_four_bar
                    and remaining >= 4
                    and (
                        (kind == "chorus" and rng.random() < 0.55)
                        or (kind == "build" and remaining == 4)
                        or (kind in ("verse", "interlude") and rng.random() < 0.22)
                    )
                )
                allow_three = (not auto_fits_two) and (
                    remaining == 3 or (remaining >= 5 and (remaining - 3 >= 3 or punct_fits_two))
                )
                if prefer_four:
                    ordered_sizes = [4, 3, 2] if allow_three else [4, 2]
                elif allow_three:
                    ordered_sizes = [3, 4, 2]
                else:
                    ordered_sizes = [2, 4]

            sizes = []
            for s in ordered_sizes:
                if s <= remaining and s not in sizes:
                    sizes.append(s)
            sizes += [s for s in (3, 4) if s not in sizes]
            if not (low <= start < high):
                bar += sizes[0]
                continue

            choice: tuple[int, list[tuple[dict, float]]] | None = None
            for size in sizes:
                slot_end = bars[min(bar + size, len(bars) - 1)]
                if blocked(start, slot_end):
                    continue
                latest = slot_end - join_seconds(distance, (slot_end - start) / size)
                is_end_two = (size == 2 and bar + size >= stop)
                is_kime_two = (size == 2 and remaining == 2 and any(b in kime_bars for b in range(bar, bar + size)))
                cand_items = usable + (pose_punct if (is_end_two or is_kime_two) else [])
                pool = _fit_pool(cand_items, start, latest)
                if not pool:
                    # Relax leftover-join filter if all candidates were slightly short (e.g. 4.2s clip in 4-bar slot)
                    fitted = [(i, min(latest, start + i["duration"] / AUTO_SPEED_RANGE[0])) for i in cand_items]
                    pool = [(i, e) for i, e in fitted if i["duration"] / max(e - start, 0.1) <= AUTO_SPEED_RANGE[1]]
                if pool:
                    choice = (size, pool)
                    break

            if choice is None:
                open_sizes = [s for s in sizes if not blocked(start, bars[min(bar + s, len(bars) - 1)])]
                if not open_sizes:
                    bar += sizes[0]
                    continue
                size = max(open_sizes)
                slot_end = bars[min(bar + size, len(bars) - 1)]
                latest = max(start + 0.35, slot_end - join_seconds(distance, (slot_end - start) / size))
                closest = min(usable, key=lambda i: abs(np.log(i["duration"] / (latest - start))))
                choice = (size, [(closest, latest)])

            size, pool = choice
            local_e = _local_bar_energy(song, bar, bar + size)
            has_kime = any(b in kime_bars for b in range(bar, bar + size))
            is_sec_end = (bar + size) >= stop - 1
            preferred_id = reused_entry[0] if reused_entry is not None else None

            scored_pool = [
                (
                    score_phrase(
                        pair[0],
                        flow,
                        recent,
                        start,
                        pair[1],
                        sound,
                        section_kind=kind,
                        local_energy=local_e,
                        slot_bars=size,
                        preferred_mood=preferred_mood,
                        recent_stems=recent_stems,
                        sound_weights=sound_weights,
                        has_kime=has_kime,
                        is_section_end=is_sec_end,
                    ),
                    pair[0],
                    pair[1],
                )
                for pair in pool
            ]
            scored_pool.sort(key=lambda t: -t[0])
            top_score = scored_pool[0][0]
            near_top = [(it, en) for sc, it, en in scored_pool[:4] if top_score - sc <= 0.04]
            picked_default = rng.choice(near_top) if len(near_top) > 1 else (scored_pool[0][1], scored_pool[0][2])
            item, end = next((pair for pair in pool if pair[0]["id"] == preferred_id), picked_default)

            clips.append({
                "id": uuid.uuid4().hex[:10],
                "phraseId": item["id"],
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "x": target[0],
                "z": target[1],
                "facing": 0.0,
                "locked": False,
                "speedMode": "auto",
                "match": match,
            })

            if distance > 0.05:
                bars_since_move = size
            else:
                bars_since_move += size
            position = target
            recent = (recent + [item["id"]])[-6:]
            recent_stems = (recent_stems + [base_stem(item)])[-6:]
            chosen_here.append((item["id"], size))
            slot_number += 1
            bar += size
            cursor = bar
        repeat_memory[section_index] = chosen_here
    clips.sort(key=lambda c: c["start"])
    return clips


def _section_bounds(song: dict, section: dict) -> tuple[float, float]:
    spans = list(song.get("bars", [])) + [float(song["duration"])]
    s_idx = max(0, min(int(section.get("startBar", 0)), len(spans) - 1))
    e_idx = max(0, min(int(section.get("endBar", len(spans) - 1)), len(spans) - 1))
    return float(section.get("start", spans[s_idx])), float(section.get("end", spans[e_idx]))


def _section_for_time(song: dict, seconds: float) -> tuple[str, dict | None, float]:
    sections = song.get("sections") or []
    for section in sections:
        sec_start, sec_end = _section_bounds(song, section)
        if sec_start <= seconds < sec_end:
            return str(section["kind"]), section, sec_end
    if sections:
        first_start, _ = _section_bounds(song, sections[0])
        chosen = sections[-1] if seconds >= first_start else sections[0]
        _, sec_end = _section_bounds(song, chosen)
        return str(chosen["kind"]), chosen, sec_end
    return "verse", None, float(song["duration"])


def _snap_near(target: float, lo: float, hi: float, bars: list[float], beats: list[float]) -> float:
    if lo > hi:
        return target
    bar_lo, bar_hi = max(lo, target - 0.42), min(hi, target + 0.42)
    near_bars = [b for b in bars if bar_lo - 1e-6 <= b <= bar_hi + 1e-6]
    if near_bars:
        return float(min(near_bars, key=lambda b: abs(b - target)))
    beat_lo, beat_hi = max(lo, target - 0.30), min(hi, target + 0.30)
    near_beats = [b for b in beats if beat_lo - 1e-6 <= b <= beat_hi + 1e-6]
    if near_beats:
        return float(min(near_beats, key=lambda b: abs(b - target)))
    return target


def fill_gap_clips(
    song: dict,
    items: list[dict],
    existing: list[dict],
    flow_table: dict,
    match: int,
    seed: int,
    start: float,
    end: float,
    filter_tags: list[str] | None = None,
) -> list[dict]:
    """Plan multiple musically matched clips inside a clip-free timeline interval `[start, end]`.

    Returns ONLY the newly created clip dicts (sorted by `start`), without mutating `existing`.
    Raises `ValueError` when `[start, end]` overlaps an existing phrase or is too narrow to hold a phrase + joins.
    """
    if not items:
        raise ValueError("割り当てに使えるフレーズがありません。先にモーションを読み込んでください")
    cleaned_tags = clean_filter_tags(filter_tags)
    pool_items = filter_phrases_by_tags(items, cleaned_tags) if cleaned_tags else list(items)
    if cleaned_tags and not pool_items:
        raise ValueError(f"指定されたジャンル・タグ（{'・'.join(cleaned_tags)}）に一致するフレーズがありません")
    has_action_filter = any(
        normalize_token(t) not in _NORM_GENRES and normalize_token(t) not in _NORM_MOODS
        for t in cleaned_tags
    )

    song_dur = float(song["duration"])
    lo = max(0.0, min(song_dur, float(min(start, end))))
    hi = max(0.0, min(song_dur, float(max(start, end))))
    if hi - lo < 0.5:
        raise ValueError("ドラッグした範囲が短すぎます（補間区間をもう少し広くドラッグしてください）")

    ordered = sorted(existing, key=lambda c: float(c["start"]))
    # Allow ~50ms boundary tolerance because 30fps join bars round start/end to frame indices.
    edge_tol = 0.05
    for clip in ordered:
        if float(clip["start"]) + edge_tol < hi and lo < float(clip["end"]) - edge_tol:
            raise ValueError("補間区間（フレーズのない区間）のみをドラッグしてください")

    prev_clip = max((c for c in ordered if float(c["end"]) <= lo + edge_tol), key=lambda c: float(c["end"]), default=None)
    next_clip = min((c for c in ordered if float(c["start"]) >= hi - edge_tol), key=lambda c: float(c["start"]), default=None)
    if prev_clip is not None:
        lo = max(lo, float(prev_clip["end"]))
    if next_clip is not None:
        hi = min(hi, float(next_clip["start"]))

    prev_pos = (float(prev_clip["x"]), float(prev_clip["z"])) if prev_clip is not None else None
    next_pos = (float(next_clip["x"]), float(next_clip["z"])) if next_clip is not None else None
    total_dist = (
        float(np.hypot(next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1]))
        if (prev_pos is not None and next_pos is not None)
        else 0.0
    )

    bar_seconds = float(song.get("barSeconds") or 2.0)
    # 0.64s guarantees >= 0.60s (18 frames) even after 30fps ceil/floor quantization in compose.py.
    base_join = MIN_JOIN_SECONDS + 0.04

    def layout_window(count: int) -> tuple[float, float, float, float]:
        step_dist = total_dist / (count + 1) if (prev_pos is not None and next_pos is not None) else 0.0
        step_join = max(base_join, step_dist / TARGET_MOVE_SPEED)
        earliest = max(lo, float(prev_clip["end"]) + step_join) if prev_clip is not None else lo
        latest = min(hi, float(next_clip["start"]) - step_join) if next_clip is not None else hi
        playable_sum = (latest - earliest) - (count - 1) * step_join
        return step_join, earliest, latest, playable_sum

    _, _, _, sum_one = layout_window(1)
    if sum_one < 0.85:
        raise ValueError("この補間区間は短すぎるため、モーションを追加できません（約 2.2 秒以上の補間区間をドラッグしてください）")

    # Prefer multiple motions (count >= 2) whenever at least two ~1.15s phrases + joins fit cleanly.
    _, ear2, lat2, sum_two = layout_window(2)
    if sum_two < 2.30:
        count = 1
    else:
        target_clip_sec = float(np.clip(1.35 * bar_seconds, 2.0, 2.9))
        desired = max(2, int(round(((lat2 - ear2) + base_join) / (target_clip_sec + base_join))))
        count = 2
        for candidate_k in range(2, desired + 3):
            _, _, _, p_sum = layout_window(candidate_k)
            if p_sum / candidate_k >= 1.15 and candidate_k <= desired:
                count = candidate_k
            elif p_sum / candidate_k < 1.15:
                break

    step_join, earliest, latest, _ = layout_window(count)
    bars = [float(b) for b in song.get("bars", [])]
    beats = [float(b) for b in song.get("beats", [])]
    min_slot = 1.02 if count >= 2 else 0.85

    # Snap slot starts to musical bars/beats while preserving >= min_slot per clip and >= step_join per join.
    max_s0 = latest - count * min_slot - (count - 1) * step_join
    s0 = _snap_near(earliest, earliest, min(earliest + 0.28, max_s0), bars, beats) if max_s0 >= earliest else earliest
    starts = [s0]
    if count >= 2:
        stride = (latest - s0 + step_join) / count
        for j in range(1, count):
            prev_s = starts[-1]
            rem_after = count - j
            lo_j = prev_s + min_slot + step_join
            hi_j = latest - rem_after * min_slot - (rem_after - 1) * step_join
            ideal_j = s0 + j * stride
            snapped_j = _snap_near(ideal_j, lo_j, hi_j, bars, beats) if hi_j >= lo_j else lo_j
            starts.append(float(np.clip(snapped_j, lo_j, max(lo_j, hi_j))))

    slots: list[tuple[float, float]] = []
    for j in range(count):
        s_j = starts[j]
        e_j = (starts[j + 1] - step_join) if j + 1 < count else latest
        if next_clip is None and count == 1 and e_j - s_j > 4.2:
            e_j = s_j + min(e_j - s_j, max(2.0 * bar_seconds, 3.6))
        slots.append((s_j, e_j))

    rng = random.Random(seed)
    usable_auto = [i for i in pool_items if i.get("usage") == "auto"]
    by_id = {i["id"]: i for i in items}
    preferred_mood = _infer_preferred_mood(usable_auto or pool_items, ordered, by_id, song, rng)
    sound, sound_weights, kime_bars = _sound_arrays(song)

    earlier_clips = [c for c in ordered if float(c["end"]) <= lo + edge_tol][-6:]
    recent = [c["phraseId"] for c in earlier_clips]
    recent_stems = [base_stem(by_id[c["phraseId"]]) for c in earlier_clips if c["phraseId"] in by_id]
    next_phrase_id = next_clip["phraseId"] if next_clip is not None else None
    next_stem = base_stem(by_id[next_phrase_id]) if next_phrase_id and next_phrase_id in by_id else None

    all_ending_pose = all(has_ending_pose(i) for i in pool_items)
    created: list[dict] = []
    anchor_pos = prev_pos if prev_pos is not None else (next_pos if next_pos is not None else (0.0, 3.0))
    cur_pos = anchor_pos
    bars_since_move = 0

    for idx, (s_k, e_k) in enumerate(slots):
        span_k = max(0.45, e_k - s_k)
        mid_t = 0.5 * (s_k + e_k)
        kind, section, sec_end = _section_for_time(song, mid_t)
        flow = flow_table.get(kind, DEFAULT_FLOW["verse"])
        start_bar = int(s_k // max(bar_seconds, 0.2))
        end_bar = max(start_bar + 1, int(round(e_k / max(bar_seconds, 0.2))))
        local_e = _local_bar_energy(song, start_bar, end_bar)
        has_kime = any(b in kime_bars for b in range(start_bar, end_bar)) or any(
            a.get("kind") == "kime" and s_k - 0.12 <= float(a["t"]) <= e_k + 0.25
            for a in song.get("accents", [])
        )
        is_sec_end = bool(section and e_k >= sec_end - bar_seconds * 0.9) or (idx == count - 1 and has_kime)
        slot_bars = max(1, int(round((span_k + step_join) / max(bar_seconds, 0.4))))

        # Build candidate (item, cand_end, trim_end, fit_bonus) options for this slot.
        options: list[tuple[dict, float, float | None, float]] = []
        for item in pool_items:
            is_auto = item.get("usage") == "auto" or has_action_filter
            is_short_manual = (
                item.get("category") == "pose"
                or int(item.get("bars") or 2) == 1
                or not usable_auto
            )
            if not is_auto and not (is_short_manual and (span_k <= 2.9 or has_kime or is_sec_end)):
                continue

            dur = float(item["duration"])
            # 1) Full untrimmed motion if speed stays natural and doesn't leave an excessive blank tail.
            cand_end = min(e_k, s_k + max(0.80, dur / 0.78))
            played = cand_end - s_k
            speed = dur / max(played, 0.1)
            leftover = e_k - cand_end
            if 0.75 <= speed <= 1.48 and (leftover <= max(0.55, 0.40 * span_k) or has_action_filter):
                options.append((item, cand_end, None, 0.08))
                continue

            # 2) Tempo-proportioned trim for continuous dance phrases on narrower slots.
            if dur / span_k > 1.38 and (not has_ending_pose(item) or all_ending_pose):
                nom_bars = max(1.0, float(item.get("bars") or 2))
                nat_speed = float(np.clip(dur / max(nom_bars * bar_seconds, 0.8), 0.95, 1.30))
                trim_end = round(min(dur, max(0.85, span_k * nat_speed)), 3)
                if 0.80 <= trim_end < dur - 0.05:
                    bonus = 0.0 if not has_ending_pose(item) else -0.10
                    options.append((item, e_k, trim_end, bonus))

        if not options:
            # Guaranteed fallback for unusual libraries (e.g. only 4-bar or pose clips).
            pool_src = usable_auto or pool_items
            for item in pool_src:
                dur = float(item["duration"])
                if dur / span_k <= 1.65:
                    cand_end = min(e_k, s_k + max(0.80, dur / 0.72))
                    options.append((item, cand_end, None, 0.0))
                else:
                    trim_end = round(min(dur, max(0.80, span_k * 1.15)), 3)
                    options.append((item, e_k, trim_end if trim_end < dur - 0.05 else None, -0.05))

        scored: list[tuple[float, dict, float, float | None]] = []
        for item, cand_end, trim_end, fit_bonus in options:
            eval_item = {**item, "duration": trim_end} if trim_end is not None else item
            sc = score_phrase(
                eval_item,
                flow,
                recent,
                s_k,
                cand_end,
                sound,
                section_kind=kind,
                local_energy=local_e,
                slot_bars=slot_bars,
                preferred_mood=preferred_mood,
                recent_stems=recent_stems,
                sound_weights=sound_weights,
                has_kime=has_kime,
                is_section_end=is_sec_end,
            ) + fit_bonus
            if idx == count - 1 and next_phrase_id:
                if item["id"] == next_phrase_id:
                    sc -= 0.34
                elif next_stem and base_stem(item) == next_stem:
                    sc -= 0.24
            scored.append((sc, item, cand_end, trim_end))

        scored.sort(key=lambda t: -t[0])
        top_sc = scored[0][0]
        near_top = [t for t in scored[:4] if top_sc - t[0] <= 0.035]
        _, chosen_item, chosen_end, chosen_trim = rng.choice(near_top) if len(near_top) > 1 else scored[0]

        # Stage placement: smoothly bridge prev_pos -> next_pos when both exist; otherwise stay calm.
        if prev_pos is not None and next_pos is not None:
            frac = (idx + 1) / (count + 1)
            pos_x = round(float(prev_pos[0] + (next_pos[0] - prev_pos[0]) * frac), 2)
            pos_z = round(float(prev_pos[1] + (next_pos[1] - prev_pos[1]) * frac), 2)
        else:
            raw_move = float(flow.get("move", 0.4))
            if idx > 0 and bars_since_move >= 6 and raw_move > 0.05 and kind != "intro":
                cur_pos = next_position(cur_pos, min(raw_move * 0.6, 0.8), rng, kind)
                bars_since_move = slot_bars
            else:
                bars_since_move += slot_bars
            pos_x, pos_z = round(float(cur_pos[0]), 2), round(float(cur_pos[1]), 2)

        clip_dict: dict = {
            "id": uuid.uuid4().hex[:10],
            "phraseId": chosen_item["id"],
            "start": round(float(s_k), 3),
            "end": round(float(chosen_end), 3),
            "x": pos_x,
            "z": pos_z,
            "facing": float(prev_clip.get("facing", 0.0)) if prev_clip is not None else 0.0,
            "locked": False,
            "speedMode": "manual",
            "match": match,
        }
        if chosen_trim is not None and chosen_trim < float(chosen_item["duration"]) - 0.05:
            clip_dict["trimEnd"] = round(float(chosen_trim), 3)

        created.append(clip_dict)
        recent = (recent + [chosen_item["id"]])[-6:]
        recent_stems = (recent_stems + [base_stem(chosen_item)])[-6:]

    created.sort(key=lambda c: c["start"])
    return created

