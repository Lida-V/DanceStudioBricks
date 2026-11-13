"""Model Context Protocol (MCP) server for Dance Studio Bricks.

Exposes song analysis, subjective/semantic phrase search, auto-arrangement, gap-fill multi-motion
placement, timeline editing (add/update/cut/reorder/delete/candidates), MotionBricks join inspection,
multi-dancer formations, 3D model assignment, undo/redo, and 30fps BVH/FBX/VRMA/VMD export over MCP.
Communicates with the shared local HTTP backend (`127.0.0.1:8765`) so MCP tools and the desktop UI stay in sync.
stdout is reserved strictly for JSON-RPC frames when running with `stdio` transport.
"""
from __future__ import annotations

import argparse
from typing import Any
import urllib.parse

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]

from .client import base_url, request

INSTRUCTIONS = (
    "Dance Studio Bricks MCP server: arrange SOMA dance phrases to musical bars/beats and connect them with MotionBricks. "
    "Start with `studio_status` or `timeline_get` to inspect the loaded song, dancers, phrases, and MotionBricks join health. "
    "All timeline edits record undo snapshots (`history_step`)."
)

mcp = MCPServer("DanceStudioBricks", instructions=INSTRUCTIONS)


# ---- Compact formatting helpers (strip large WebGL/SVG arrays like `thumb` and `wave`) ----

def _compact_phrase(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": item["id"],
        "name": item.get("name", item["id"]),
        "category": item.get("category"),
        "energy": item.get("energy"),
        "bars": item.get("bars"),
        "duration": item.get("duration"),
        "sourceBpm": item.get("sourceBpm"),
        "travel": item.get("travel"),
        "usage": item.get("usage"),
        "tags": item.get("tags", []),
        "description": item.get("description", ""),
        "accentsCount": len(item.get("accents") or []),
    }
    if "score" in item:
        out["score"] = item["score"]
    return out


def _bar_number_at(bars: list[float], t: float) -> int:
    if not bars:
        return 1
    count = 1
    for idx, bar_t in enumerate(bars):
        if bar_t <= t + 1e-3:
            count = idx + 1
        else:
            break
    return count


def _compact_song(song: dict[str, Any] | None, *, include_beats: bool = False) -> dict[str, Any] | None:
    if not song:
        return None
    bars = song.get("bars") or []
    accents = song.get("accents") or []
    summary: dict[str, Any] = {
        "name": song.get("name"),
        "path": song.get("path"),
        "duration": song.get("duration"),
        "bpm": song.get("bpm"),
        "barSeconds": song.get("barSeconds"),
        "barCount": len(bars),
        "sections": song.get("sections", []),
        "engines": song.get("engines", {}),
        "accentCount": len(accents),
        "kimeTimes": [round(float(a["t"]), 3) for a in accents if a.get("kind") == "kime" or a.get("kime")],
    }
    if include_beats:
        summary["bars"] = bars
        summary["beats"] = song.get("beats", [])
        summary["accents"] = accents
        summary["energyPerBar"] = song.get("energyPerBar", [])
    return summary


def _join_counts(joins: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for join in joins:
        st = str(join.get("status", "unknown"))
        counts[st] = counts.get(st, 0) + 1
    return counts


def _enrich_clip(
    clip: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    bars: list[float],
    clip_reports: dict[str, Any],
) -> dict[str, Any]:
    item = by_id.get(clip["phraseId"], {})
    full_dur = float(item.get("duration") or max(0.3, clip["end"] - clip["start"]))
    trim_start = float(clip.get("trimStart", 0.0))
    trim_end = float(clip.get("trimEnd", full_dur))
    played_len = max(1e-3, float(clip["end"]) - float(clip["start"]))
    speed = round((trim_end - trim_start) / played_len, 3)
    rep = clip_reports.get(clip["id"]) or {}
    entry: dict[str, Any] = {
        "id": clip["id"],
        "phraseId": clip["phraseId"],
        "phraseName": item.get("name", clip["phraseId"]),
        "category": item.get("category"),
        "start": clip["start"],
        "end": clip["end"],
        "startBar": _bar_number_at(bars, float(clip["start"])),
        "speed": speed,
        "speedMode": clip.get("speedMode", "auto"),
        "x": clip["x"],
        "z": clip["z"],
        "facing": clip.get("facing", 0.0),
        "locked": bool(clip.get("locked", False)),
        "match": clip.get("match", 70),
    }
    if "trimStart" in clip or "trimEnd" in clip:
        entry["trimStart"] = round(trim_start, 3)
        entry["trimEnd"] = round(trim_end, 3)
    if clip.get("pins"):
        entry["pins"] = clip["pins"]
    if rep:
        entry["matchedAccents"] = f"{rep.get('matched', 0)}/{rep.get('candidates', 0)}"
        entry["localSpeed"] = rep.get("localSpeed")
    return entry


def _summarize_mutation(
    changes: dict[str, Any],
    by_id: dict[str, dict[str, Any]] | None = None,
    bars: list[float] | None = None,
) -> dict[str, Any]:
    project = changes.get("project") or {}
    active_id = project.get("active", "d1")
    dancers = project.get("dancers") or []
    active_dancer = next((d for d in dancers if d["id"] == active_id), dancers[0] if dancers else {"id": active_id, "clips": []})
    composed = changes.get("composed") or {}
    joins = composed.get("joins") or []
    clip_reports = composed.get("clips") or {}
    phrase_map = by_id or {}
    bar_list = bars or []

    result: dict[str, Any] = {
        "activeDancer": active_dancer.get("id"),
        "activeDancerName": active_dancer.get("name"),
        "clipCount": len(active_dancer.get("clips") or []),
        "clips": [_enrich_clip(c, phrase_map, bar_list, clip_reports) for c in (active_dancer.get("clips") or [])],
        "joins": joins,
        "joinStatusCounts": _join_counts(joins),
        "engineError": composed.get("engineError"),
        "collisions": changes.get("collisions") or [],
        "cast": changes.get("cast") or [],
        "formations": project.get("formations") or [],
        "arrangeFilter": project.get("arrangeFilter") or [],
        "history": changes.get("history") or {},
    }
    for extra_key in ("added", "count", "dropped", "adjusted", "selected", "formation", "restored"):
        if extra_key in changes:
            result[extra_key] = changes[extra_key]
    return result


def _library_and_bars(full_state: dict[str, Any] | None = None) -> tuple[dict[str, dict[str, Any]], list[float], dict[str, Any]]:
    st = full_state if full_state is not None else request("GET", "/api/state")
    by_id = {item["id"]: item for item in (st.get("library") or [])}
    bars = (st.get("song") or {}).get("bars") or []
    return by_id, bars, st


# ---- Status & Inspection Tools ----

@mcp.tool()
def studio_status() -> dict[str, Any]:
    """Inspect Dance Studio Bricks runtime status: external environments (MotionBricks, Beat This!, EmbeddingGemma 2, Blender), loaded song, library counts, dancers, and join health."""
    st = request("GET", "/api/state")
    comp = request("GET", "/api/components")
    library = st.get("library") or []
    cat_counts: dict[str, int] = {}
    for item in library:
        cat = str(item.get("category", "other"))
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    composed = st.get("composed") or {}
    joins = composed.get("joins") or []
    project = st.get("project") or {}
    return {
        "serverUrl": base_url(),
        "components": comp.get("components", []),
        "installJob": comp.get("job"),
        "libraryTotal": len(library),
        "libraryByCategory": cat_counts,
        "categories": st.get("categories", {}),
        "song": _compact_song(st.get("song"), include_beats=False),
        "activeDancer": project.get("active"),
        "cast": st.get("cast", []),
        "models": project.get("models", {}),
        "arrangeFilter": project.get("arrangeFilter") or [],
        "formationCount": len(project.get("formations") or []),
        "joinStatusCounts": _join_counts(joins),
        "engineError": composed.get("engineError"),
        "collisionCount": len(st.get("collisions") or []),
        "history": st.get("history", {}),
    }


@mcp.tool()
def timeline_get(dancer_id: str | None = None, include_beats: bool = False) -> dict[str, Any]:
    """Read the current song sections, flow rules, ordered timeline clips, MotionBricks joins, formations, and close-call collisions. Set `include_beats=True` to include full beat/accent arrays."""
    by_id, bars, st = _library_and_bars()
    project = st.get("project") or {}
    dancers = project.get("dancers") or []
    target_id = dancer_id or project.get("active")
    target = next((d for d in dancers if d["id"] == target_id), None)
    if target is None:
        raise RuntimeError(f"ダンサー '{target_id}' は見つかりません")
    is_active = target["id"] == project.get("active")
    composed = (st.get("composed") or {}) if is_active else {}
    clip_reports = composed.get("clips") or {}
    joins = composed.get("joins") or []
    return {
        "song": _compact_song(st.get("song"), include_beats=include_beats),
        "flow": project.get("flow", {}),
        "arrangeFilter": project.get("arrangeFilter") or [],
        "dancer": {"id": target["id"], "name": target["name"], "isActive": is_active},
        "clips": [_enrich_clip(c, by_id, bars, clip_reports) for c in (target.get("clips") or [])],
        "joins": joins,
        "joinStatusCounts": _join_counts(joins),
        "formations": project.get("formations") or [],
        "models": project.get("models") or {},
        "collisions": st.get("collisions") or [],
        "history": st.get("history") or {},
    }


# ---- Song & Flow Dictionary Tools ----

@mcp.tool()
def song_load(path: str, tempo_factor: float | None = None) -> dict[str, Any]:
    """Load and analyze an audio file (WAV/FLAC/MP3/OGG/AIFF). Optional `tempo_factor` (e.g. 0.5 or 2.0) halves or doubles the bar grid. Restores saved per-song choreography when revisiting a song."""
    st = request("POST", "/api/song", {"path": path, "factor": tempo_factor}, timeout=300.0)
    by_id, bars, _ = _library_and_bars(st)
    out = _summarize_mutation(st, by_id, bars)
    out["song"] = _compact_song(st.get("song"), include_beats=False)
    return out


@mcp.tool()
def song_sections_set(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace song section boundaries. Each item requires 0-based `startBar`, `endBar` (`0 < startBar < endBar <= barCount`), and `kind` (`intro`|`verse`|`build`|`chorus`|`interlude`|`outro`)."""
    st = request("PUT", "/api/song/sections", {"sections": sections})
    return {
        "song": _compact_song(st.get("song"), include_beats=False),
        "history": st.get("history", {}),
    }


@mcp.tool()
def flow_set(flow_updates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Update the choreography flow dictionary by section kind (`intro`|`verse`|`build`|`chorus`|`interlude`|`outro`). Partial updates are merged with existing rules (`categories`, `energy` 1..5, `move` 0..5, `text`)."""
    st = request("GET", "/api/state")
    merged = dict((st.get("project") or {}).get("flow") or {})
    for kind, patch in flow_updates.items():
        base = dict(merged.get(kind) or {})
        base.update(patch)
        merged[kind] = base
    changes = request("PUT", "/api/flow", merged)
    return {"flow": (changes.get("project") or {}).get("flow", {}), "history": changes.get("history", {})}


# ---- Motion Library & Search Tools ----

@mcp.tool()
def library_list(
    category: str | None = None,
    usage: str | None = None,
    tag_or_mood: str | None = None,
    limit: int = 60,
    offset: int = 0,
) -> dict[str, Any]:
    """List dance phrases in the library. Filter by `category` (`pose`|`groove`|`wave`|`power`|`foot`), `usage` (`auto`|`manual`), or `tag_or_mood` substring (e.g. `女性的`, `アイドル系`, `キメ`)."""
    st = request("GET", "/api/state")
    items = list(st.get("library") or [])
    if category:
        items = [i for i in items if i.get("category") == category]
    if usage:
        items = [i for i in items if i.get("usage") == usage]
    if tag_or_mood:
        needle = tag_or_mood.strip().lower()
        items = [
            i for i in items
            if any(needle in str(t).lower() for t in (i.get("tags") or []))
            or needle in str(i.get("name", "")).lower()
            or needle in str(i.get("description", "")).lower()
        ]
    total = len(items)
    sliced = items[max(0, offset): max(0, offset) + max(1, min(limit, 200))]
    return {
        "total": total,
        "offset": max(0, offset),
        "returned": len(sliced),
        "items": [_compact_phrase(i) for i in sliced],
    }


@mcp.tool()
def library_search(query: str, category: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Search motions by subjective feeling (`楽しそう`, `可愛らしい感じ`, `かっこいい`, `エモい`, `決めポーズ`), tags, or natural-language description."""
    fetch_limit = max(limit * 3, 30) if category else max(1, min(limit, 60))
    resp = request("POST", "/api/library/search", {"query": query, "limit": fetch_limit}, timeout=120.0)
    items = resp.get("results") or []
    if category:
        items = [i for i in items if i.get("category") == category]
    items = items[: max(1, min(limit, 60))]
    return {"query": query, "count": len(items), "results": [_compact_phrase(i) for i in items]}


@mcp.tool()
def library_import(paths: list[str]) -> dict[str, Any]:
    """Import BONES-standard SOMA `.bvh` file(s) or folder(s) into the library. Non-SOMA BVH files are safely rejected with reasons."""
    res = request("POST", "/api/library/import", {"paths": paths}, timeout=300.0)
    return {
        "addedCount": len(res.get("added") or []),
        "added": [_compact_phrase(i) for i in (res.get("added") or [])],
        "rejected": res.get("rejected") or [],
        "libraryTotal": len((res.get("state") or {}).get("library") or []),
    }


@mcp.tool()
def library_update_phrase(phrase_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Update library phrase metadata (`name`, `category`, `energy` 1..5, `usage` `auto`|`manual`, `tags`, `description`, `accents`, `provenance`)."""
    updated = request("PATCH", f"/api/library/{urllib.parse.quote(phrase_id)}", changes)
    return _compact_phrase(updated)


@mcp.tool()
def library_delete_phrase(phrase_id: str) -> dict[str, Any]:
    """Delete a phrase from the library and remove any clips referencing it."""
    st = request("DELETE", f"/api/library/{urllib.parse.quote(phrase_id)}")
    by_id, bars, _ = _library_and_bars(st)
    out = _summarize_mutation(st, by_id, bars)
    out["libraryTotal"] = len(by_id)
    return out


# ---- Arrangement & Multi-Motion Gap Fill Tools ----

@mcp.tool()
def arrange_filter_set(tags: list[str] | None = None) -> dict[str, Any]:
    """Set or clear (`tags=[]`) persistent genre/mood/action filter tags for auto-arrange, shuffle, and gap-fill (e.g. `['K-POP']`, `['アイドル系', 'ハート']`, `['キック']`)."""
    changes = request("PUT", "/api/arrange-filter", {"tags": tags or []})
    project = changes.get("project") or {}
    return {"arrangeFilter": project.get("arrangeFilter") or [], "history": changes.get("history") or {}}


@mcp.tool()
def arrange_timeline(
    start: float | None = None,
    end: float | None = None,
    seed: int | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Auto-arrange phrases for the active dancer across the whole song (or `[start, end]` seconds), optionally restricted to `tags` (genres/moods/action tags like `['K-POP']`, `['キック']`). Locked clips remain untouched."""
    by_id, bars, _ = _library_and_bars()
    changes = request("POST", "/api/arrange", {"start": start, "end": end, "seed": seed, "tags": tags})
    return _summarize_mutation(changes, by_id, bars)


@mcp.tool()
def arrange_shuffle(tags: list[str] | None = None) -> dict[str, Any]:
    """Re-arrange unlocked clips on the active dancer with a new random seed, optionally restricted to `tags`."""
    by_id, bars, _ = _library_and_bars()
    changes = request("POST", "/api/shuffle", {"tags": tags})
    return _summarize_mutation(changes, by_id, bars)


@mcp.tool()
def clip_fill_gap(start: float, end: float, seed: int | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    """Assign and insert multiple musically matched motions into a clip-free interpolation/gap interval `[start, end]` without shifting existing clips, optionally restricted to `tags`."""
    by_id, bars, _ = _library_and_bars()
    changes = request("POST", "/api/clips/fill-gap", {"start": start, "end": end, "seed": seed, "tags": tags})
    return _summarize_mutation(changes, by_id, bars)


# ---- Clip Editing, Cutting & Join Tools ----

@mcp.tool()
def clip_add(
    phrase_id: str,
    start: float,
    end: float | None = None,
    x: float | None = None,
    z: float | None = None,
    exact: bool = True,
    fit: str = "auto",
    bars: float | None = None,
) -> dict[str, Any]:
    """Insert a phrase at `start` seconds. `fit`: `'auto'` (fit open gap / song tempo), `'bars'` (fit `bars` count, e.g. 1/2/4), or `'raw'` (unmodified source duration)."""
    by_id, bar_list, _ = _library_and_bars()
    body = {
        "phraseId": phrase_id,
        "start": start,
        "end": end,
        "x": x,
        "z": z,
        "exact": exact,
        "fit": fit,
        "bars": bars,
    }
    changes = request("POST", "/api/clips", body)
    return _summarize_mutation(changes, by_id, bar_list)


@mcp.tool()
def clip_copy(clip_id: str, start: float | None = None) -> dict[str, Any]:
    """Paste a copy of a clip of the active dancer at `start` seconds (default: right after the copied clip). The copy keeps the length, trim, stage position, facing, `match` and pins, goes in by the insert rules (never on top of a clip) and starts unlocked. This is what Ctrl+C / Ctrl+V does on screen."""
    by_id, bars, state = _library_and_bars()
    project = state.get("project") or {}
    dancers = project.get("dancers") or []
    active = next((d for d in dancers if d.get("id") == project.get("active")), dancers[0] if dancers else {})
    clip = next((c for c in active.get("clips") or [] if c.get("id") == clip_id), None)
    if clip is None:
        raise RuntimeError("そのフレーズは、編集中のダンサーのタイムラインにありません")
    res = request("POST", "/api/clips/paste", {"clip": clip, "start": clip["end"] if start is None else start})
    return _summarize_mutation(res, by_id, bars)


@mcp.tool()
def clip_update(clip_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Modify a clip on the active dancer. Allowed keys: `start`, `end`, `x`, `z`, `facing`, `locked`, `match` (0..100), `phraseId`, `speedMode` (`auto`|`manual`), `pins` (`[{"source": s, "at": offset}]`), `trimStart`, `trimEnd`."""
    by_id, bars, _ = _library_and_bars()
    res = request("PATCH", f"/api/clips/{urllib.parse.quote(clip_id)}", changes)
    return _summarize_mutation(res, by_id, bars)


@mcp.tool()
def clip_cut(clip_id: str, at: float, mode: str = "split") -> dict[str, Any]:
    """Cut or split a timeline clip at `at` seconds (`mode`: `'split'` into two clips, `'before'` to trim front, `'after'` to trim tail). Automatically updates stage origin when cutting front."""
    by_id, bars, _ = _library_and_bars()
    res = request("POST", f"/api/clips/{urllib.parse.quote(clip_id)}/cut", {"at": at, "mode": mode})
    return _summarize_mutation(res, by_id, bars)


@mcp.tool()
def clip_delete(clip_id: str) -> dict[str, Any]:
    """Delete an unlocked clip from the active dancer's timeline."""
    by_id, bars, _ = _library_and_bars()
    res = request("DELETE", f"/api/clips/{urllib.parse.quote(clip_id)}")
    return _summarize_mutation(res, by_id, bars)


@mcp.tool()
def clip_reorder(order: list[str]) -> dict[str, Any]:
    """Reorder clips on the active dancer by providing the full list of clip IDs in the desired playback order."""
    by_id, bars, _ = _library_and_bars()
    res = request("POST", "/api/clips/reorder", {"order": order})
    return _summarize_mutation(res, by_id, bars)


@mcp.tool()
def clip_candidates(clip_id: str) -> dict[str, Any]:
    """Return top musically matched replacement phrase candidates for `clip_id` at its current musical position."""
    by_id, _, _ = _library_and_bars()
    res = request("GET", f"/api/clips/{urllib.parse.quote(clip_id)}/candidates")
    enriched = []
    for cand in res.get("candidates") or []:
        item = by_id.get(cand["phraseId"], {})
        enriched.append({
            "phraseId": cand["phraseId"],
            "score": cand["score"],
            "name": item.get("name", cand["phraseId"]),
            "category": item.get("category"),
            "energy": item.get("energy"),
            "bars": item.get("bars"),
            "duration": item.get("duration"),
            "tags": item.get("tags", []),
        })
    return {"clipId": clip_id, "candidates": enriched}


@mcp.tool()
def join_options(clip_id: str, apply_best: bool = False) -> dict[str, Any]:
    """Generate and measure candidate MotionBricks join lengths after `clip_id` (moveSpeed, maxJointSpeed, minHeight). Set `apply_best=True` to automatically adopt the best candidate."""
    data = request("GET", f"/api/joins/{urllib.parse.quote(clip_id)}/options", timeout=180.0)
    if apply_best and not data.get("locked"):
        best = next((o for o in (data.get("options") or []) if o.get("best")), None)
        if best and not best.get("current"):
            by_id, bars, _ = _library_and_bars()
            applied = request(
                "PATCH",
                f"/api/clips/{urllib.parse.quote(data['from'])}",
                {"end": best["end"], "speedMode": "manual"},
            )
            data["applied"] = _summarize_mutation(applied, by_id, bars)
    return data


@mcp.tool()
def joins_wait(max_rounds: int = 8) -> dict[str, Any]:
    """Wait until background MotionBricks join generation completes (no `'pending'` joins remain) and return the updated join summary."""
    by_id, bars, _ = _library_and_bars()
    latest: dict[str, Any] = {}
    for _ in range(max(1, min(max_rounds, 24))):
        latest = request("GET", "/api/joins/wait", timeout=35.0)
        joins = ((latest.get("composed") or {}).get("joins")) or []
        if not any(j.get("status") == "pending" for j in joins):
            break
    return _summarize_mutation(latest, by_id, bars)


# ---- Multi-Dancer, Formation, Model, History & Export Tools ----

@mcp.tool()
def dancer_manage(
    action: str,
    dancer_id: str | None = None,
    name: str | None = None,
    copy_from: str | None = None,
) -> dict[str, Any]:
    """Manage dancers (`action`: `'list'`|`'add'`|`'select'`|`'rename'`|`'delete'`|`'sync'`). Up to 12 dancers; `'sync'` copies the active dancer's phrases/timing to all other dancers."""
    by_id, bars, st = _library_and_bars()
    act = action.strip().lower()
    if act == "list":
        project = st.get("project") or {}
        return {"active": project.get("active"), "cast": st.get("cast", []), "models": project.get("models", {})}
    if act == "add":
        changes = request("POST", "/api/dancers", {"copyFrom": copy_from})
    elif act == "select":
        if not dancer_id:
            raise RuntimeError("dancer_id を指定してください")
        changes = request("POST", f"/api/dancers/{urllib.parse.quote(dancer_id)}/select", {})
    elif act == "rename":
        if not dancer_id or not name:
            raise RuntimeError("dancer_id と name を指定してください")
        changes = request("PATCH", f"/api/dancers/{urllib.parse.quote(dancer_id)}", {"name": name})
    elif act == "delete":
        if not dancer_id:
            raise RuntimeError("dancer_id を指定してください")
        changes = request("DELETE", f"/api/dancers/{urllib.parse.quote(dancer_id)}")
    elif act == "sync":
        changes = request("POST", "/api/dancers/sync", {})
    else:
        raise RuntimeError("action は list / add / select / rename / delete / sync のいずれかを指定してください")
    return _summarize_mutation(changes, by_id, bars)


@mcp.tool()
def formation_manage(
    action: str,
    time: float | None = None,
    spots: dict[str, dict[str, Any]] | None = None,
    name: str = "",
    formation_id: str | None = None,
) -> dict[str, Any]:
    """Manage multi-dancer stage formations (`action`: `'list'`|`'set'`|`'delete'`). For `'set'`, pass `time` (seconds; snaps to bar) and `spots` (`{'d1': {'x': -1.0, 'z': 3.0, 'facing': 0.0}}`)."""
    by_id, bars, st = _library_and_bars()
    act = action.strip().lower()
    if act == "list":
        return {"formations": (st.get("project") or {}).get("formations") or []}
    if act == "set":
        if time is None or not spots:
            raise RuntimeError("time と spots を指定してください")
        changes = request("PUT", "/api/formations", {"time": time, "spots": spots, "name": name})
    elif act == "delete":
        if not formation_id:
            raise RuntimeError("formation_id を指定してください")
        changes = request("DELETE", f"/api/formations/{urllib.parse.quote(formation_id)}")
    else:
        raise RuntimeError("action は list / set / delete のいずれかを指定してください")
    return _summarize_mutation(changes, by_id, bars)


@mcp.tool()
def model_set(
    dancer_id: str = "d1",
    path: str | None = None,
    mapping: dict[str, str] | None = None,
    everyone: bool = False,
) -> dict[str, Any]:
    """Assign a 3D character model (`.vrm` / `.glb` / `.fbx`) to `dancer_id` (or `everyone=True`), or pass `path=None` to reset to the default fingered mannequin."""
    changes = request(
        "PUT",
        f"/api/dancers/{urllib.parse.quote(dancer_id)}/model",
        {"path": path, "mapping": mapping, "everyone": everyone},
    )
    project = changes.get("project") or {}
    return {"active": project.get("active"), "models": project.get("models", {})}


@mcp.tool()
def history_step(direction: str = "undo") -> dict[str, Any]:
    """Undo (`'undo'`) or redo (`'redo'`) the most recent edit."""
    step = direction.strip().lower()
    if step not in ("undo", "redo"):
        raise RuntimeError("direction は undo または redo を指定してください")
    st = request("POST", f"/api/{step}", {})
    by_id, bars, _ = _library_and_bars(st)
    return _summarize_mutation(st, by_id, bars)


@mcp.tool()
def export_motion(
    format: str = "bvh",
    path: str | None = None,
    pmx_path: str | None = None,
    rest: str = "bones",
    arm_degrees: float = 45.0,
) -> dict[str, Any]:
    """Wait for pending MotionBricks joins and export 30fps full-timeline motion (`format`: `'bvh'`|`'fbx'`|`'vrma'`|`'vmd'`). Optional `pmx_path` calibrates VMD arm/leg proportions to a specific MMD PMX model. `rest` chooses the rest pose of a BVH / FBX: `'bones'` (the SOMA files' own), `'t'` (T-pose) or `'a'` (A-pose with the arms `arm_degrees` below level). Defaults to `data/exports/`."""
    fmt = format.strip().lower()
    if fmt == "glb":
        raise RuntimeError("GLB（モデル＋動き）書き出しは 3D 画面上でのボーン焼き込みが必要なため、アプリの書き出し画面から実行してください（MCP からは bvh / fbx / vrma / vmd が書き出せます）")
    return request("POST", "/api/export", {"format": fmt, "path": path, "pmx": pmx_path, "rest": rest, "armDegrees": arm_degrees}, timeout=600.0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dance Studio Bricks MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    args = parser.parse_args(argv)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
