"""End-to-end and stdio protocol tests for the Dance Studio Bricks MCP server."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsb import client, mcp_server, server  # noqa: E402
from dsb.embed import EmbedUnavailable  # noqa: E402
from dsb.mb_client import EngineError  # noqa: E402
from dsb.studio import Studio  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "dev-fixtures" / "bvh"


class OfflineEngine:
    def start(self, timeout=None):
        pass

    def abort_start(self):
        pass

    def stop(self):
        pass

    @property
    def joints(self):
        raise EngineError("offline test engine")


def make_test_song(bar_sec: float = 1.9, bar_count: int = 16) -> dict:
    bars = [round(i * bar_sec, 3) for i in range(bar_count)]
    return {
        "barSeconds": bar_sec,
        "bpm": round(240.0 / bar_sec, 1),
        "duration": round(bar_count * bar_sec, 3),
        "bars": bars,
        "beats": [round(i * (bar_sec / 4.0), 3) for i in range(bar_count * 4)],
        "wave": [0.25] * 200,
        "energyPerBar": [0.3] * 4 + [0.85] * 8 + [0.25] * 4,
        "accents": [{"t": bars[7], "kind": "kime", "kime": True, "onBeat": True}],
        "engines": {"beats": "test", "sections": "test"},
        "sections": [
            {"startBar": 0, "endBar": 4, "start": bars[0], "end": bars[4], "kind": "intro", "label": "イントロ", "repeatOf": -1, "energy": 0.3},
            {"startBar": 4, "endBar": 12, "start": bars[4], "end": bars[12], "kind": "chorus", "label": "サビ", "repeatOf": -1, "energy": 0.85},
            {"startBar": 12, "endBar": 16, "start": bars[12], "end": round(bar_count * bar_sec, 3), "kind": "outro", "label": "アウトロ", "repeatOf": -1, "energy": 0.25},
        ],
    }


class McpServerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)
        self.studio = Studio(self.tmp, engine=OfflineEngine(), background=False)
        self.addCleanup(self.studio.close)

        # Import 2 real SOMA BVH fixtures so compose / cut / export run with real skeleton poses
        bvh_files = sorted(FIXTURES.glob("*.bvh"))[:3]
        if bvh_files:
            self.studio.import_motion([str(p) for p in bvh_files])
            items = list(self.studio.library.items.values())
            if items:
                items[0]["tags"] = ["女性的", "アイドル系", "キメ"]
                items[0]["description"] = "可愛らしく頬にピースを添えるポーズ"
            if len(items) > 1:
                items[1]["tags"] = ["中性的", "J-POP"]
                items[1]["description"] = "明るく楽しそうにステップとクラップを刻む"

        self.studio.song = make_test_song()
        self.studio.project["songPath"] = "test_song.wav"

        patcher = mock.patch.object(server, "studio", self.studio)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.http = TestClient(server.app)

        def local_requester(method: str, path: str, body=None, timeout: float = 180.0):
            resp = self.http.request(method, path, json=body)
            if resp.status_code >= 400:
                payload = resp.json()
                raise RuntimeError(payload.get("error") or payload.get("detail") or f"HTTP {resp.status_code}")
            return resp.json()

        ctx = client.use_requester(local_requester)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def test_registered_tools_have_valid_schemas_and_descriptions(self):
        tools = asyncio.run(mcp_server.mcp.list_tools())
        names = {t.name for t in tools}
        expected = {
            "studio_status",
            "timeline_get",
            "song_load",
            "song_sections_set",
            "flow_set",
            "library_list",
            "library_search",
            "library_import",
            "library_update_phrase",
            "library_delete_phrase",
            "arrange_timeline",
            "arrange_shuffle",
            "clip_fill_gap",
            "clip_add",
            "clip_copy",
            "clip_update",
            "clip_cut",
            "clip_delete",
            "clip_reorder",
            "clip_candidates",
            "join_options",
            "joins_wait",
            "dancer_manage",
            "formation_manage",
            "model_set",
            "history_step",
            "export_motion",
        }
        self.assertTrue(expected.issubset(names), f"Missing MCP tools: {expected - names}")
        for tool in tools:
            self.assertTrue(tool.description, f"Tool {tool.name} is missing a description")

    @unittest.skipUnless(FIXTURES.is_dir() and any(FIXTURES.glob("*.bvh")), "dev-fixtures/bvh required")
    def test_end_to_end_choreography_workflow_via_mcp_tools(self):
        status = mcp_server.studio_status()
        self.assertGreaterEqual(status["libraryTotal"], 2)
        self.assertEqual(status["song"]["barCount"], 16)

        # Compact library payloads must omit heavy UI `thumb` stick-figure dicts
        listed = mcp_server.library_list(limit=10)
        self.assertGreaterEqual(listed["total"], 2)
        self.assertNotIn("thumb", listed["items"][0])

        # Subjective feeling search via MCP
        with mock.patch("dsb.embed.texts", side_effect=EmbedUnavailable("offline")):
            found = mcp_server.library_search("可愛らしい感じ", limit=5)
        self.assertGreaterEqual(found["count"], 1)

        # Auto-arrange whole song
        arranged = mcp_server.arrange_timeline(seed=5)
        self.assertGreaterEqual(arranged["clipCount"], 4)

        # Delete middle clips to open an interpolation gap, then fill with multiple motions via `clip_fill_gap`
        clips_before = arranged["clips"]
        first_clip = clips_before[0]
        last_clip = clips_before[-1]
        for mid in clips_before[1:-1]:
            mcp_server.clip_delete(mid["id"])

        filled = mcp_server.clip_fill_gap(first_clip["end"], last_clip["start"], seed=9)
        self.assertGreaterEqual(filled["count"], 2)
        self.assertEqual(len(filled["added"]), filled["count"])

        # Candidate lookup, clip cut, partial flow update, multi-dancer sync & formation
        target_id = filled["added"][0]
        cands = mcp_server.clip_candidates(target_id)
        self.assertIn("candidates", cands)

        target_clip = next(c for c in filled["clips"] if c["id"] == target_id)
        if target_clip["end"] - target_clip["start"] >= 1.2:
            mid_t = round((target_clip["start"] + target_clip["end"]) * 0.5, 3)
            after_cut = mcp_server.clip_cut(target_id, mid_t, mode="split")
            self.assertEqual(after_cut["clipCount"], filled["clipCount"] + 1)

        flow_res = mcp_server.flow_set({"chorus": {"energy": 5, "move": 0.6}})
        self.assertEqual(flow_res["flow"]["chorus"]["energy"], 5)

        added_dancer = mcp_server.dancer_manage("add", copy_from="d1")
        self.assertEqual(len(added_dancer["cast"]), 2)
        mcp_server.dancer_manage("rename", dancer_id="d2", name="センター右")
        form_res = mcp_server.formation_manage(
            "set",
            time=0.0,
            name="開幕ペア",
            spots={"d1": {"x": -0.9, "z": 3.0, "facing": 0.0}, "d2": {"x": 0.9, "z": 3.0, "facing": 0.0}},
        )
        self.assertEqual(len(form_res["formations"]), 1)

        # Export BVH and VRMA via MCP
        bvh_out = mcp_server.export_motion("bvh")
        self.assertEqual(len(bvh_out["files"]), 2)
        self.assertTrue(Path(bvh_out["path"]).is_file())

        with self.assertRaises(RuntimeError):
            mcp_server.export_motion("glb")

        # Undo & redo via MCP
        undone = mcp_server.history_step("undo")
        self.assertEqual(len(undone["formations"]), 0)
        redone = mcp_server.history_step("redo")
        self.assertEqual(len(redone["formations"]), 1)

        # Timeline read omits `wave` unless problem occurs
        tl = mcp_server.timeline_get(include_beats=False)
        self.assertNotIn("wave", tl["song"])
        self.assertNotIn("beats", tl["song"])

    def test_stdio_jsonrpc_handshake_and_tools_list(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "backend"), "DSB_MCP_AUTOBOOT": "0"}
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-W", "ignore", "-m", "dsb.mcp_server"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        try:
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "dsb-test", "version": "0.1.0"},
                },
            }
            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            payload = "\n".join(json.dumps(msg) for msg in (init_req, notif, list_req)) + "\n"
            stdout, stderr = proc.communicate(input=payload, timeout=15)
        finally:
            if proc.poll() is None:
                proc.kill()

        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 2, f"Expected JSON-RPC responses on stdout, got stdout={stdout!r}, stderr={stderr!r}")
        responses = [json.loads(line) for line in lines]
        by_id = {r.get("id"): r for r in responses if "id" in r}
        self.assertIn(1, by_id)
        self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "DanceStudioBricks")
        self.assertIn(2, by_id)
        tool_names = {t["name"] for t in by_id[2]["result"]["tools"]}
        self.assertIn("clip_fill_gap", tool_names)
        self.assertIn("library_search", tool_names)


if __name__ == "__main__":
    unittest.main()
