"""Tests for feeling/impression search, musical arrangement matching, calm stage blocking, and length-adjusted clip insertion."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsb.arrange import DEFAULT_FLOW, MIN_JOIN_SECONDS, arrange, fill_gap_clips, filter_phrases_by_tags  # noqa: E402
from dsb.embed import EmbedUnavailable  # noqa: E402
from dsb.library import CATEGORIES  # noqa: E402
from dsb.search import clean_query_token, feeling_scores, impression_words  # noqa: E402
from dsb.studio import Studio, StudioError  # noqa: E402


def sample_library() -> list[dict]:
    return [
        {
            "id": "cute_peace",
            "name": "Peace Cheek F",
            "duration": 2.35,
            "bars": 1,
            "usage": "manual",
            "category": "pose",
            "energy": 2,
            "accents": [0.8],
            "tags": ["女性的", "アイドル系", "キメ"],
            "description": "頬にピースを添えて首をかしげる",
        },
        {
            "id": "fun_clap",
            "name": "Step Touch Clap N",
            "duration": 4.2,
            "bars": 2,
            "usage": "auto",
            "category": "groove",
            "energy": 3,
            "accents": [1.0, 2.1, 3.1],
            "tags": ["中性的", "J-POP"],
            "description": "ステップタッチとクラップで明るく刻む",
        },
        {
            "id": "cool_hit",
            "name": "Sharp Iso M",
            "duration": 4.2,
            "bars": 2,
            "usage": "auto",
            "category": "power",
            "energy": 4,
            "accents": [0.9, 2.0, 3.2],
            "tags": ["男性的", "K-POP", "ヒップホップ"],
            "description": "鋭くアイソレーションとヒットを打つ",
        },
        {
            "id": "intro_open",
            "name": "Intro Rise F",
            "duration": 4.3,
            "bars": 2,
            "usage": "auto",
            "category": "pose",
            "energy": 1,
            "accents": [1.8],
            "tags": ["女性的", "イントロ", "バラード"],
            "description": "顔を上げて腕を開く",
        },
        {
            "id": "outro_end",
            "name": "Outro Settle F",
            "duration": 4.2,
            "bars": 2,
            "usage": "auto",
            "category": "pose",
            "energy": 1,
            "accents": [1.5],
            "tags": ["女性的", "アウトロ", "バラード"],
            "description": "動きを収めて止まる",
        },
        {
            "id": "chorus_4bar",
            "name": "Chorus Combo A F",
            "duration": 7.6,
            "bars": 4,
            "usage": "auto",
            "category": "power",
            "energy": 5,
            "accents": [1.2, 3.0, 5.2, 6.8],
            "tags": ["女性的", "サビ", "J-POP"],
            "description": "サビの見せ場コンボ",
        },
    ]


class FeelingSearchTests(unittest.TestCase):
    def test_subjective_queries_match_expected_motions(self):
        items = sample_library()
        cute = feeling_scores("可愛らしい感じ", items, CATEGORIES)
        self.assertGreater(cute["cute_peace"], 0.8)
        self.assertEqual(cute["cool_hit"], 0.0)

        fun = feeling_scores("楽しそう", items, CATEGORIES)
        self.assertGreater(fun["fun_clap"], 0.8)
        self.assertEqual(fun["outro_end"], 0.0)

        cool = feeling_scores("かっこいい動き", items, CATEGORIES)
        self.assertGreater(cool["cool_hit"], cute["cool_hit"])

        emo = feeling_scores("エモい・切ない", items, CATEGORIES)
        self.assertGreater(emo["intro_open"], 0.6)

        kime = feeling_scores("決めポーズ", items, CATEGORIES)
        self.assertGreater(kime["cute_peace"], 0.6)

    def test_impression_words_enrich_metadata(self):
        words = impression_words(sample_library()[0])
        self.assertIn("可愛らしい", words)
        self.assertIn("決めポーズ", words)

    def test_studio_search_caches_embeddings_and_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            studio = Studio(Path(folder), engine=object())
            studio.library.items = {i["id"]: i for i in sample_library()}

            encode_calls: list[int] = []

            def fake_texts(pairs):
                encode_calls.append(len(pairs))
                rng = np.random.default_rng(42)
                vecs = rng.normal(size=(len(pairs), 16)).astype(np.float32)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                return vecs

            with mock.patch("dsb.embed.texts", fake_texts):
                res1 = studio.search("可愛らしい感じ", limit=5)
                res2 = studio.search("楽しそう", limit=5)
            self.assertEqual(encode_calls[0], len(sample_library()))
            # Second search should reuse cached document embeddings and only encode the 1 query!
            self.assertEqual(sum(encode_calls[1:]), 2)
            self.assertTrue(res1 and res2)

            # Fallback when EmbeddingGemma 2 is unavailable
            def fail_texts(_pairs):
                raise EmbedUnavailable("not installed")

            with mock.patch("dsb.embed.texts", fail_texts):
                fallback = studio.search("可愛らしい感じ", limit=5)
            self.assertEqual(fallback[0]["id"], "cute_peace")


class ArrangeMatchingAndMovementTests(unittest.TestCase):
    def test_fast_song_avoids_outro_in_intro_calms_movement_and_uses_poses(self):
        bar_sec = 1.70  # ~141 BPM
        bars = [round(i * bar_sec, 3) for i in range(32)]
        fast_song = {
            "bars": bars,
            "duration": 32 * bar_sec,
            "barSeconds": bar_sec,
            "energyPerBar": [0.2] * 8 + [0.5] * 8 + [0.9] * 8 + [0.25] * 8,
            "accents": [{"t": bars[7], "kind": "kime", "onBeat": True}, {"t": bars[23], "kind": "kime", "onBeat": True}],
            "sections": [
                {"startBar": 0, "endBar": 8, "kind": "intro", "repeatOf": -1},
                {"startBar": 8, "endBar": 16, "kind": "verse", "repeatOf": -1},
                {"startBar": 16, "endBar": 24, "kind": "chorus", "repeatOf": -1},
                {"startBar": 24, "endBar": 32, "kind": "outro", "repeatOf": -1},
            ],
        }
        items = sample_library()
        result = arrange(fast_song, items, [], DEFAULT_FLOW, 70, 1)
        self.assertGreaterEqual(len(result), 8)
        # Intro must never pick Outro Settle
        intro_ids = [c["phraseId"] for c in result if c["start"] < bars[8]]
        self.assertNotIn("outro_end", intro_ids)
        self.assertIn("intro_open", intro_ids)
        # Auto-arranged clips stay untrimmed per SPEC.md
        for c in result:
            self.assertNotIn("trimEnd", c)
        # Calm movement: fewer than half of transitions walk across the stage
        moves = [float(np.hypot(b["x"] - a["x"], b["z"] - a["z"])) for a, b in zip(result, result[1:])]
        self.assertLess(sum(1 for d in moves if d > 0.05), len(moves) // 2 + 1)


class LengthAdjustedInsertTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.studio = Studio(Path(folder.name), engine=object())
        self.studio._recompose = lambda force=False: None
        self.studio._preview_joints = lambda: None
        self.studio.library.items = {i["id"]: dict(i) for i in sample_library()}
        bar_sec = 1.70
        bars = [round(i * bar_sec, 3) for i in range(24)]
        self.studio.song = {
            "barSeconds": bar_sec,
            "duration": 24 * bar_sec,
            "bars": bars,
            "beats": [round(i * (bar_sec / 4), 3) for i in range(96)],
            "accents": [],
            "sections": [{"startBar": 0, "endBar": 24, "kind": "verse"}],
        }
        self.studio.project["songPath"] = "song.wav"

    def test_auto_fit_fills_open_gap_without_shifting_following_clip(self):
        base = {"phraseId": "fun_clap", "facing": 0.0, "locked": False, "speedMode": "manual", "match": 70, "x": 0.0, "z": 3.0}
        # Leave a 2.8s open gap between 3.0s and 6.8s (bar 4 = 6.8s)
        self.studio.clips = [
            {**base, "id": "c1", "start": 0.0, "end": 3.0},
            {**base, "id": "c2", "start": 6.8, "end": 10.2},
        ]
        res = self.studio.add_clip("cute_peace", 3.85, exact=True, fit="auto")
        c2_after = next(c for c in self.studio.clips if c["id"] == "c2")
        inserted = next(c for c in self.studio.clips if c["id"] == res["added"])
        self.assertEqual(c2_after["start"], 6.8)
        self.assertLessEqual(inserted["end"], 6.8 - 0.6)

    def test_bars_fit_trims_long_phrase_to_fit_one_bar_cleanly(self):
        res = self.studio.add_clip("fun_clap", 0.0, exact=True, fit="bars", bars=1.0)
        inserted = next(c for c in self.studio.clips if c["id"] == res["added"])
        self.assertLess(inserted["end"] - inserted["start"], 1.70)
        self.assertIn("trimEnd", inserted)


class GapFillTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.studio = Studio(Path(folder.name), engine=object())
        self.studio._recompose = lambda force=False: None
        self.studio._preview_joints = lambda: None
        self.studio.library.items = {i["id"]: dict(i) for i in sample_library()}
        bar_sec = 1.90
        bars = [round(i * bar_sec, 3) for i in range(24)]
        self.studio.song = {
            "barSeconds": bar_sec,
            "duration": 24 * bar_sec,
            "bars": bars,
            "beats": [round(i * (bar_sec / 4), 3) for i in range(96)],
            "energyPerBar": [0.35] * 8 + [0.85] * 8 + [0.3] * 8,
            "accents": [{"t": bars[11], "kind": "kime", "onBeat": True}],
            "sections": [
                {"startBar": 0, "endBar": 8, "kind": "verse"},
                {"startBar": 8, "endBar": 16, "kind": "chorus"},
                {"startBar": 16, "endBar": 24, "kind": "outro"},
            ],
        }
        self.studio.project["songPath"] = "song.wav"

    def test_fill_gap_between_clips_inserts_multiple_motions_and_interpolates_stage_path(self):
        base = {"facing": 0.0, "locked": True, "speedMode": "manual", "match": 70}
        # 6.8s interpolation gap between 3.2s and 10.0s, moving from x=-1.5 to x=1.5
        self.studio.clips = [
            {**base, "id": "left", "phraseId": "fun_clap", "start": 0.0, "end": 3.2, "x": -1.5, "z": 3.0},
            {**base, "id": "right", "phraseId": "chorus_4bar", "start": 10.0, "end": 16.5, "x": 1.5, "z": 3.0},
        ]
        # Include 30fps join quantization edge tolerance (-0.03s / +0.03s) as sent by UI join bars
        res = self.studio.fill_gap(3.18, 10.02, seed=7)
        self.assertGreaterEqual(res["count"], 2)
        self.assertEqual(len(res["added"]), res["count"])

        # Neighboring clips keep their exact timestamps and positions
        left = next(c for c in self.studio.clips if c["id"] == "left")
        right = next(c for c in self.studio.clips if c["id"] == "right")
        self.assertEqual((left["start"], left["end"]), (0.0, 3.2))
        self.assertEqual((right["start"], right["end"]), (10.0, 16.5))

        # Every consecutive pair has >= MIN_JOIN_SECONDS gap (no overlap, no short join)
        ordered = sorted(self.studio.clips, key=lambda c: c["start"])
        for a, b in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(b["start"] - a["end"], MIN_JOIN_SECONDS - 1e-3)

        # Stage x-positions step smoothly from -1.5 to +1.5
        xs = [c["x"] for c in ordered]
        self.assertEqual(xs, sorted(xs))
        self.assertGreater(xs[1], -1.5)
        self.assertLess(xs[-2], 1.5)

        # Single undo restores the original 2-clip timeline cleanly
        self.studio.undo()
        self.assertEqual([c["id"] for c in self.studio.clips], ["left", "right"])

    def test_fill_gap_rejects_overlapping_clip_and_too_short_gap_without_history(self):
        base = {"phraseId": "fun_clap", "facing": 0.0, "locked": False, "speedMode": "manual", "match": 70, "x": 0.0, "z": 3.0}
        self.studio.clips = [
            {**base, "id": "c1", "start": 2.0, "end": 5.5},
            {**base, "id": "c2", "start": 6.6, "end": 10.0},
        ]
        # Dragging across an existing clip must fail
        with self.assertRaises(StudioError):
            self.studio.fill_gap(1.0, 6.4)
        # Dragging a 1.1s join (too narrow to hold a clip + 2 joins) must fail
        with self.assertRaises(StudioError):
            self.studio.fill_gap(5.5, 6.6)
        self.assertEqual(len(self.studio.undo_stack), 0)
        self.assertEqual([c["id"] for c in self.studio.clips], ["c1", "c2"])

    def test_fill_open_chorus_range_assigns_three_or_more_non_repeating_motions(self):
        created = fill_gap_clips(
            self.studio.song,
            list(self.studio.library.items.values()),
            [],
            DEFAULT_FLOW,
            70,
            seed=3,
            start=15.2,
            end=26.6,
        )
        self.assertGreaterEqual(len(created), 3)
        for a, b in zip(created, created[1:]):
            self.assertNotEqual(a["phraseId"], b["phraseId"])
            self.assertGreaterEqual(b["start"] - a["end"], MIN_JOIN_SECONDS - 1e-3)


class ActionTagSearchAndFilteredArrangeTests(unittest.TestCase):
    def test_kick_conversational_and_particle_queries_match_cleanly(self):
        # Verify Katakana words ending in ト/ノ are never corrupted while particles や/で/と strip cleanly
        self.assertEqual(clean_query_token("ハート"), "ハート")
        self.assertEqual(clean_query_token("ヒット"), "ヒット")
        self.assertEqual(clean_query_token("スクワット"), "スクワット")
        self.assertEqual(clean_query_token("キックや"), "キック")
        self.assertEqual(clean_query_token("キックで"), "キック")
        self.assertEqual(clean_query_token("キックしてるモーション"), "キック")

        items = [
            *sample_library(),
            {
                "id": "kpop_kick",
                "name": "Front Kick Reach F",
                "duration": 4.2,
                "bars": 2,
                "usage": "auto",
                "category": "power",
                "energy": 4,
                "accents": [1.0, 2.5],
                "tags": ["女性的", "K-POP", "キック", "ハイキック"],
                "description": "鋭く前へフロントキックを蹴り出して手を伸ばす",
            },
        ]
        for query in ("キック", "キックや", "キックで", "キックしてるモーション", "キックやでキックしてるモーション", "蹴り"):
            scores = feeling_scores(query, items, CATEGORIES)
            self.assertGreater(scores["kpop_kick"], 0.8, f"Expected kick match for query={query!r}")
            self.assertEqual(scores["fun_clap"], 0.0)

        # Multi-token AND query: "K-POP キック" matches only kpop_kick (not cool_hit which lacks kick)
        and_scores = feeling_scores("K-POP キック", items, CATEGORIES)
        self.assertGreater(and_scores["kpop_kick"], 0.9)
        self.assertEqual(and_scores["cool_hit"], 0.0)

        # Connector split query: "ピースやクラップ" matches both cute_peace and fun_clap
        or_scores = feeling_scores("ピースやクラップ", items, CATEGORIES)
        self.assertGreater(or_scores["cute_peace"], 0.6)
        self.assertGreater(or_scores["fun_clap"], 0.6)

        # Also verify against the real 190-item library if present on disk
        real_index = Path(__file__).resolve().parents[2] / "data" / "library" / "index.json"
        if real_index.is_file():
            import json
            real_items = json.loads(real_index.read_text(encoding="utf-8"))
            for q in ("キック", "キックや", "キックしてるモーション", "キックやでキックしてるモーション"):
                sc = feeling_scores(q, real_items, CATEGORIES)
                hits = [it["id"] for it in real_items if sc.get(it["id"], 0.0) > 0.5]
                self.assertGreaterEqual(len(hits), 8, f"Real library should find >=8 kick clips for {q!r}")

    def test_genre_and_tag_filtered_auto_arrange_shuffle_gap_fill_and_undo(self):
        with tempfile.TemporaryDirectory() as folder:
            studio = Studio(Path(folder), engine=object())
            studio._recompose = lambda force=False: None
            studio._preview_joints = lambda: None
            items = [
                *sample_library(),
                {
                    "id": "kpop_wave",
                    "name": "Velvet Wave F",
                    "duration": 4.2,
                    "bars": 2,
                    "usage": "auto",
                    "category": "wave",
                    "energy": 3,
                    "accents": [1.1, 2.8],
                    "tags": ["女性的", "K-POP", "ウェーブ"],
                    "description": "しなやかなボディロールとウェーブ",
                },
            ]
            studio.library.items = {i["id"]: dict(i) for i in items}
            bar_sec = 1.90
            bars = [round(i * bar_sec, 3) for i in range(16)]
            studio.song = {
                "barSeconds": bar_sec,
                "duration": 16 * bar_sec,
                "bars": bars,
                "beats": [round(i * (bar_sec / 4), 3) for i in range(64)],
                "energyPerBar": [0.4] * 8 + [0.85] * 8,
                "accents": [],
                "sections": [
                    {"startBar": 0, "endBar": 8, "kind": "verse"},
                    {"startBar": 8, "endBar": 16, "kind": "chorus"},
                ],
            }
            studio.project["songPath"] = "song.wav"

            # Facet filtering: OR within genres, AND across genre x action
            self.assertEqual({i["id"] for i in filter_phrases_by_tags(items, ["K-POP"])}, {"cool_hit", "kpop_wave"})
            self.assertEqual({i["id"] for i in filter_phrases_by_tags(items, ["K-POP", "ウェーブ"])}, {"kpop_wave"})

            # Auto-arrange restricted to K-POP only
            res = studio.arrange(seed=3, tags=["K-POP"])
            self.assertEqual(res["project"]["arrangeFilter"], ["K-POP"])
            self.assertGreaterEqual(len(studio.clips), 4)
            self.assertTrue(all(c["phraseId"] in {"cool_hit", "kpop_wave"} for c in studio.clips))

            # Shuffle inherits the active arrangeFilter
            studio.shuffle()
            self.assertTrue(all(c["phraseId"] in {"cool_hit", "kpop_wave"} for c in studio.clips))

            # Action tag with only 1-bar manual phrases ("ピース") promotes matching phrase cleanly
            studio.arrange(seed=5, tags=["ピース"])
            self.assertGreaterEqual(len(studio.clips), 4)
            self.assertTrue(all(c["phraseId"] == "cute_peace" for c in studio.clips))

            # Non-matching filter raises StudioError cleanly without corrupting state
            with self.assertRaises(StudioError):
                studio.arrange(tags=["存在しないタグXYZ"])
            self.assertEqual(studio.project["arrangeFilter"], ["ピース"])

            # Undo restores previous K-POP filter & arrangement
            studio.undo()
            self.assertEqual(studio.project["arrangeFilter"], ["K-POP"])


if __name__ == "__main__":
    unittest.main()
