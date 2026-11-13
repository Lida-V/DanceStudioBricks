"""Core logic checks that need no GPU, model or audio. Run: backend/.venv python -m unittest discover -s backend/tests"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsb.arrange import AUTO_SPEED_RANGE, DEFAULT_FLOW, MIN_JOIN_SECONDS, arrange, join_seconds  # noqa: E402
from dsb.compose import MAX_SINGLE_FRAMES, JoinCache, JoinJob, generate_join  # noqa: E402
from dsb.grounding import MAX_SHIFT, plant_feet  # noqa: E402
from dsb.library import nominal_bars  # noqa: E402
from dsb.mb_client import EngineError, frames_from_tokens  # noqa: E402
from dsb.motion import MotionError, Skeleton, check_soma_bones, read_bvh, rotate, rotation_between, sample, world_positions, world_transforms, write_bvh  # noqa: E402
from dsb.studio import Studio, StudioError, _shorten_to_fit  # noqa: E402
from dsb.warp import LOCAL_SPEED_RANGE, build  # noqa: E402

BVH = """HIERARCHY
ROOT Root
{
OFFSET 0 0 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Hips
{
OFFSET 0 100 0
CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
JOINT Spine1
{
OFFSET 5 0 0
CHANNELS 3 Zrotation Yrotation Xrotation
End Site
{
OFFSET 1 0 0
}
}
}
}
MOTION
Frames: 3
Frame Time: 0.1
0 0 0 0 0 0 10 100 0 0 0 0 0 0 0
0 0 0 0 0 0 20 100 0 0 90 0 0 0 0
0 0 0 0 0 0 30 100 0 0 0 0 0 0 0
"""


class Motion(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "a.bvh"
        self.path.write_text(BVH, encoding="utf-8")

    def test_reads_root_into_hips_in_metres(self):
        motion = read_bvh(self.path)
        self.assertEqual(motion.skeleton.names, ("Hips", "Spine1"))
        self.assertAlmostEqual(motion.fps, 10.0)
        np.testing.assert_allclose(motion.poses[1, 0, :3], [0.2, 1.0, 0.0])

    def test_round_trip_keeps_world_positions(self):
        motion = read_bvh(self.path)
        out = self.path.with_name("b.bvh")
        write_bvh(out, motion.skeleton, motion.poses, motion.fps)
        again = read_bvh(out)
        np.testing.assert_allclose(world_positions(motion.poses, motion.skeleton.parents), world_positions(again.poses, again.skeleton.parents), atol=1e-6)

    def test_rejects_garbage_and_non_soma(self):
        bad = self.path.with_name("c.bvh")
        bad.write_text("nope", encoding="utf-8")
        with self.assertRaises(MotionError):
            read_bvh(bad)
        self.assertTrue(check_soma_bones(read_bvh(self.path).skeleton))

    def test_rotation_between_maps_direction(self):
        a = np.array([[1.0, 0.0, 0.0], [0.2, 0.9, -0.4]])
        b = np.array([[0.0, 1.0, 0.0], [-0.5, 0.1, 0.8]])
        turned = rotate(rotation_between(a, b), a)
        expected = b / np.linalg.norm(b, axis=1, keepdims=True) * np.linalg.norm(a, axis=1, keepdims=True)
        np.testing.assert_allclose(turned, expected, atol=1e-9)

    def test_sample_interpolates_and_clamps(self):
        poses = read_bvh(self.path).poses
        values = sample(poses, 10.0, np.array([0.05, 5.0]))
        self.assertAlmostEqual(values[0, 0, 0], 0.15)
        self.assertAlmostEqual(values[1, 0, 0], 0.3)
        np.testing.assert_allclose(np.linalg.norm(values[..., 3:7], axis=-1), 1.0)


class Warp(unittest.TestCase):
    def test_zero_strength_is_uniform(self):
        warp = build(10.0, 14.0, 4.0, [1.0, 3.0], np.array([11.05]), 0.0)
        np.testing.assert_allclose(warp.source_at(np.array([10.0, 12.0, 14.0])), [0.0, 2.0, 4.0], atol=1e-6)

    def test_full_strength_lands_on_the_sound(self):
        warp = build(10.0, 14.0, 4.0, [1.0], np.array([11.1]), 1.0)
        self.assertTrue(warp.matches[0]["matched"])
        self.assertAlmostEqual(float(warp.source_at(np.array([11.1]))[0]), 1.0, places=5)

    def test_stays_monotonic_and_in_range(self):
        accents = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        rng = np.random.default_rng(1)
        sound = np.sort(rng.uniform(10.0, 14.0, 20))
        warp = build(10.0, 14.0, 4.0, accents, sound, 1.0)
        low, high = warp.local_speed_range()
        self.assertGreaterEqual(low, LOCAL_SPEED_RANGE[0] - 1e-6)
        self.assertLessEqual(high, LOCAL_SPEED_RANGE[1] + 1e-6)
        values = warp.source_at(np.linspace(10.0, 14.0, 400))
        self.assertTrue((np.diff(values) >= -1e-9).all())

    def test_out_of_reach_sound_is_not_used(self):
        warp = build(10.0, 14.0, 4.0, [2.0], np.array([12.5]), 1.0)
        self.assertFalse(warp.matches[0]["matched"])


def song(bars=32, bar_seconds=1.875):
    times = [round(i * bar_seconds, 3) for i in range(bars)]
    return {"bars": times, "duration": bars * bar_seconds, "barSeconds": bar_seconds, "accents": [{"t": t} for t in times],
            "sections": [{"startBar": 0, "endBar": 8, "kind": "verse", "repeatOf": -1}, {"startBar": 8, "endBar": 16, "kind": "chorus", "repeatOf": -1},
                         {"startBar": 16, "endBar": 24, "kind": "verse", "repeatOf": 0}, {"startBar": 24, "endBar": 32, "kind": "chorus", "repeatOf": 1}]}


def phrases():
    return [{"id": str(i), "name": str(i), "usage": "auto", "bars": 2, "duration": 3.5, "category": c, "energy": e, "accents": [1.0, 2.0]}
            for i, (c, e) in enumerate([("groove", 3), ("power", 5), ("foot", 3), ("wave", 2)])]


class Arrange(unittest.TestCase):
    def test_join_length(self):
        self.assertAlmostEqual(join_seconds(0.0, 1.875), 0.9375)  # the 0.6 s minimum rounds up to two quarter bars
        self.assertGreaterEqual(join_seconds(0.0, 0.8), MIN_JOIN_SECONDS)
        self.assertAlmostEqual(join_seconds(2.5, 1.875), 1.875)

    def test_clips_do_not_overlap_and_respect_speed(self):
        result = arrange(song(), phrases(), [], DEFAULT_FLOW, 70, 1)
        self.assertGreater(len(result), 8)
        for a, b in zip(result[:-1], result[1:]):
            self.assertLess(a["end"], b["start"])
        for clip in result:
            speed = 3.5 / (clip["end"] - clip["start"])
            self.assertLessEqual(speed, AUTO_SPEED_RANGE[1] + 1e-6)
            self.assertGreaterEqual(speed, AUTO_SPEED_RANGE[0] - 1e-6)

    def test_locked_clips_stay_and_range_is_limited(self):
        first = arrange(song(), phrases(), [], DEFAULT_FLOW, 70, 1)
        first[2]["locked"] = True
        again = arrange(song(), phrases(), first, DEFAULT_FLOW, 70, 2)
        self.assertIn(first[2], again)
        limited = arrange(song(), phrases(), first, DEFAULT_FLOW, 70, 3, (15.0, 30.0))
        outside = [c for c in first if c["end"] <= 15.0 or c["start"] >= 30.0]
        for clip in outside:
            self.assertIn(clip, limited)

    def test_repeated_section_reuses_phrases(self):
        result = arrange(song(), phrases(), [], DEFAULT_FLOW, 70, 1)
        first = [c["phraseId"] for c in result if c["start"] < 15.0]
        repeat = [c["phraseId"] for c in result if 30.0 <= c["start"] < 45.0]
        self.assertEqual(first[:2], repeat[:2])

    def test_no_usable_phrases(self):
        with self.assertRaises(ValueError):
            arrange(song(), [], [], DEFAULT_FLOW, 70, 1)


class History(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.studio = Studio(Path(folder.name), engine=object())
        self.studio._recompose = lambda: None
        self.studio._preview_joints = lambda: None
        self.studio.library.items = {"a": {"id": "a", "duration": 3.5}}
        self.studio.song = {"barSeconds": 1.875, "sections": [{"startBar": 0, "endBar": 8, "kind": "verse"}]}
        self.studio.project["songPath"] = "song.wav"
        base = {"phraseId": "a", "facing": 0.0, "locked": False, "speedMode": "manual", "match": 70}
        self.studio.clips = [{**base, "id": "one", "start": 0.0, "end": 3.0, "x": 0.0, "z": 3.0}, {**base, "id": "two", "start": 3.75, "end": 6.75, "x": 1.0, "z": 3.0}]

    def clip(self, name):
        return next(c for c in self.studio.clips if c["id"] == name)

    def test_undo_and_redo_restore_edits(self):
        self.studio.update_clip("two", {"x": 2.5, "facing": 45.0})
        self.studio.undo()
        self.assertEqual((self.clip("two")["x"], self.clip("two")["facing"]), (1.0, 0.0))
        self.studio.redo()
        self.assertEqual((self.clip("two")["x"], self.clip("two")["facing"]), (2.5, 45.0))

    def test_new_edit_clears_redo(self):
        self.studio.update_clip("two", {"x": 2.5})
        self.studio.undo()
        self.studio.update_clip("two", {"z": 1.0})
        self.assertEqual(self.studio.state()["history"], {"undo": 1, "redo": 0})

    def test_deleted_clip_comes_back(self):
        self.studio.delete_clip("one")
        self.assertEqual(len(self.studio.clips), 1)
        self.studio.undo()
        self.assertEqual({c["id"] for c in self.studio.clips}, {"one", "two"})

    def test_failed_edit_leaves_no_history(self):
        with self.assertRaises(StudioError):
            self.studio.update_clip("one", {"end": 0.1})
        self.assertEqual(self.studio.state()["history"]["undo"], 0)

    def test_moving_refits_automatic_neighbour_only(self):
        self.clip("one")["speedMode"] = "auto"
        self.studio.update_clip("two", {"x": 0.7})
        self.assertAlmostEqual(self.clip("one")["end"], 3.75 - join_seconds(0.7, 1.875), places=2)
        self.assertEqual(self.clip("two")["end"], 6.75)

    def test_automatic_timing_stays_inside_the_speed_limit(self):
        self.clip("one")["speedMode"] = "auto"
        self.studio.update_clip("two", {"x": 2.5})
        self.assertAlmostEqual(self.clip("one")["end"], 3.5 / AUTO_SPEED_RANGE[1], places=2)

    def test_locked_neighbour_keeps_its_timing(self):
        self.clip("one").update(speedMode="auto", locked=True)
        self.studio.update_clip("two", {"x": 2.5})
        self.assertEqual(self.clip("one")["end"], 3.0)

    def test_facing_alone_does_not_refit(self):
        self.clip("one")["speedMode"] = "auto"
        self.studio.update_clip("two", {"facing": 90.0})
        self.assertEqual(self.clip("one")["end"], 3.0)


class Placement(unittest.TestCase):
    """Slots every 3.75 s, each phrase 3 s long, so the gaps are too small to hold anything."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.studio = Studio(Path(folder.name), engine=object())
        self.studio._recompose = lambda: None
        self.studio._preview_joints = lambda: None
        self.studio.library.items = {"a": {"id": "a", "duration": 3.5}}
        self.studio.song = {"barSeconds": 1.875, "duration": 30.0, "sections": [{"startBar": 0, "endBar": 8, "kind": "verse"}]}
        self.studio.project["songPath"] = "song.wav"
        self.lay(0.0, 3.75, 7.5, 11.25)

    def lay(self, *starts):
        base = {"phraseId": "a", "facing": 0.0, "locked": False, "speedMode": "manual", "match": 70, "x": 0.0, "z": 3.0}
        self.studio.clips = [{**base, "id": name, "start": start, "end": start + 3.0} for name, start in zip("abcd", starts)]

    def starts(self):
        return {c["id"]: c["start"] for c in self.studio.clips}

    def test_left_half_of_a_phrase_goes_before_it(self):
        added = self.studio.add_clip("a", 0.5)["added"]
        self.assertEqual(self.starts()[added], 0.0)
        self.assertEqual(self.starts()["a"], 5.625)  # 3.5 s + a 0.9375 s join needs three bars
        self.assertEqual(self.starts()["d"], 16.875)

    def test_right_half_of_a_phrase_goes_after_it(self):
        added = self.studio.add_clip("a", 2.5)["added"]
        self.assertEqual(self.starts()[added], 3.75)
        self.assertEqual(self.starts()["a"], 0.0)
        self.assertEqual(self.starts()["b"], 9.375)

    def test_every_later_phrase_moves_by_whole_bars(self):
        self.studio.add_clip("a", 2.5)
        for name, was in (("b", 3.75), ("c", 7.5), ("d", 11.25)):
            bars = (self.starts()[name] - was) / 1.875
            self.assertAlmostEqual(bars, round(bars))

    def test_a_big_enough_gap_stops_the_shift(self):
        self.lay(0.0, 3.75, 7.5, 24.0)
        self.studio.add_clip("a", 2.5)
        self.assertEqual(self.starts()["c"], 13.125)
        self.assertEqual(self.starts()["d"], 24.0)

    def test_a_wide_gap_takes_the_phrase_where_asked_without_moving_anything(self):
        self.lay(0.0, 3.75, 7.5, 24.0)
        added = self.studio.add_clip("a", 15.0)["added"]
        self.assertEqual(self.starts()[added], 15.0)
        self.assertEqual(self.starts()["d"], 24.0)

    def test_phrases_pushed_past_the_song_are_removed_and_counted(self):
        self.studio.song["duration"] = 15.0
        result = self.studio.add_clip("a", 2.5)
        self.assertEqual(result["dropped"], 2)
        self.assertNotIn("d", self.starts())

    def test_one_undo_takes_the_whole_insert_back(self):
        self.studio.add_clip("a", 2.5)
        self.studio.undo()
        self.assertEqual(self.starts(), {"a": 0.0, "b": 3.75, "c": 7.5, "d": 11.25})

    def test_position_comes_from_the_request_or_the_previous_phrase(self):
        self.studio.clips[0].update(x=1.5, z=2.0)
        spot = self.studio.add_clip("a", 2.5, x=-2.0, z=4.0)["added"]
        self.assertEqual([(c["x"], c["z"]) for c in self.studio.clips if c["id"] == spot], [(-2.0, 4.0)])
        follow = self.studio.add_clip("a", 0.5 + self.starts()["a"] + 2.0)["added"]
        self.assertEqual([(c["x"], c["z"]) for c in self.studio.clips if c["id"] == follow], [(1.5, 2.0)])

    def test_too_close_to_the_end_is_refused(self):
        self.lay(0.0)
        self.studio.clips = self.studio.clips[:1]
        with self.assertRaises(StudioError):
            self.studio.add_clip("a", 29.5)
        self.assertEqual(self.studio.state()["history"]["undo"], 0)

    def test_reorder_swaps_phrases_into_each_others_slots(self):
        self.studio.reorder_clips(["c", "a", "b", "d"])
        self.assertEqual(self.starts(), {"c": 0.0, "a": 3.75, "b": 7.5, "d": 11.25})

    def test_reorder_carries_room_and_leaves_the_rest_alone(self):
        self.lay(0.0, 7.5, 11.25, 15.0)
        self.studio.reorder_clips(["b", "a", "c", "d"])
        self.assertEqual(self.starts(), {"b": 0.0, "a": 3.75, "c": 11.25, "d": 15.0})

    def test_reorder_keeps_each_phrase_length_and_position(self):
        self.studio.clips[3].update(end=11.25 + 5.0, x=2.0)
        self.studio.reorder_clips(["d", "a", "b", "c"])
        moved = next(c for c in self.studio.clips if c["id"] == "d")
        self.assertEqual((moved["start"], moved["end"], moved["x"]), (0.0, 5.0, 2.0))

    def test_reorder_is_one_undo(self):
        self.studio.reorder_clips(["c", "a", "b", "d"])
        self.studio.undo()
        self.assertEqual([c["id"] for c in self.studio.clips], ["a", "b", "c", "d"])

    def test_reorder_refuses_a_wrong_list_and_ignores_no_change(self):
        with self.assertRaises(StudioError):
            self.studio.reorder_clips(["a", "b", "c"])
        self.studio.reorder_clips(["a", "b", "c", "d"])
        self.assertEqual(len(self.studio.undo_stack), 0)

    def lock(self, *names):
        for clip in self.studio.clips:
            clip["locked"] = clip["id"] in names

    def times(self):
        return {c["id"]: (c["start"], c["end"]) for c in self.studio.clips}

    def assert_no_overlap(self):
        ordered = self.studio.clips
        self.assertEqual(ordered, sorted(ordered, key=lambda c: c["start"]))
        for a, b in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(round(b["start"] - a["end"], 3), 0.6)

    def test_a_new_phrase_starts_unlocked(self):
        added = self.studio.add_clip("a", 25.0)["added"]
        self.assertFalse(next(c for c in self.studio.clips if c["id"] == added)["locked"])

    def test_insert_shortens_free_phrases_and_leaves_the_locked_one(self):
        self.lay(0.0, 3.75, 7.5, 15.0)
        self.lock("d")
        before = self.times()
        result = self.studio.add_clip("a", 2.5)
        after = self.times()
        self.assertEqual(after["d"], before["d"])
        self.assertEqual(after["a"], before["a"])  # the phrases after the insert were enough
        self.assertEqual(result["dropped"], 0)
        self.assertGreater(result["adjusted"], 0)
        self.assertLess(after[result["added"]][1] - after[result["added"]][0], 3.5)
        self.assert_no_overlap()

    def test_the_slowest_phrase_gives_up_time_first(self):
        self.lay(0.0, 3.75, 7.5, 15.0)
        self.studio.clips[2]["end"] = 12.0  # plays at x0.78, the others at x1.17
        self.lock("d")
        self.studio.add_clip("a", 2.5)
        after = self.times()
        self.assertLess(after["c"][1] - after["c"][0], 4.5)
        self.assertAlmostEqual(after["b"][1] - after["b"][0], 3.0, places=2)
        self.assert_no_overlap()

    def test_insert_borrows_from_earlier_free_phrases_when_needed(self):
        self.lock("c", "d")
        before = self.times()
        added = self.studio.add_clip("a", 2.5)["added"]
        after = self.times()
        self.assertEqual((after["c"], after["d"]), (before["c"], before["d"]))
        self.assertEqual(after["a"][0], 0.0)
        self.assertLess(after["a"][1], before["a"][1])
        self.assertEqual([c["id"] for c in self.studio.clips], ["a", added, "b", "c", "d"])
        self.assert_no_overlap()
        for clip in self.studio.clips:
            self.assertLessEqual(3.5 / (clip["end"] - clip["start"]), 2.0 + 1e-3)

    def test_shortened_phrases_stop_following_automatic_timing(self):
        self.lay(0.0, 3.75, 7.5, 15.0)
        self.lock("d")
        self.studio.clips[2].update(speedMode="auto", end=12.0)
        self.studio.add_clip("a", 2.5)
        self.assertEqual(self.studio._clip("c")["speedMode"], "manual")

    def test_insert_moves_free_phrases_without_shortening_when_there_is_room(self):
        self.lay(0.0, 3.75, 7.5, 17.0)
        self.lock("d")
        before = self.times()
        result = self.studio.add_clip("a", 2.5)
        after = self.times()
        self.assertEqual(result["adjusted"], 0)
        self.assertEqual(after["d"], before["d"])
        for name in "abc":
            self.assertAlmostEqual(after[name][1] - after[name][0], 3.0, places=2)
        self.assert_no_overlap()

    def test_a_shift_that_stops_short_of_the_locked_phrase_is_unchanged(self):
        self.lay(0.0, 3.75, 7.5, 24.0)
        self.lock("d")
        self.studio.add_clip("a", 2.5)
        self.assertEqual(self.starts()["c"], 13.125)
        self.assertEqual(self.starts()["d"], 24.0)

    def test_moved_phrases_land_on_beats_when_they_can(self):
        self.lay(0.0, 3.75, 7.5, 17.0)
        self.lock("d")
        self.studio.song["beats"] = [i * 0.46875 for i in range(64)]
        self.studio.add_clip("a", 2.5)
        for name in "bc":
            beats = self.starts()[name] / 0.46875
            self.assertAlmostEqual(beats, round(beats), places=2)

    def test_a_beat_is_given_up_when_it_would_make_a_phrase_much_faster(self):
        self.studio.library.items["a"]["duration"] = 4.2
        self.studio.song.update(barSeconds=1.6, beats=[round(i * 0.4 + 0.001, 3) for i in range(100)])
        base = {"phraseId": "a", "facing": 0.0, "speedMode": "manual", "match": 70, "z": 3.0}
        self.studio.clips = [{**base, "id": "free", "start": 19.201, "end": 24.401, "x": 0.0, "locked": False},
                             {**base, "id": "wall", "start": 25.601, "end": 30.801, "x": 1.5, "locked": True}]
        added = self.studio.add_clip("a", 23.0)["added"]
        self.assertEqual(self.times()["wall"], (25.601, 30.801))
        free, inserted = self.times()["free"], self.times()[added]
        self.assertAlmostEqual(4.2 / (free[1] - free[0]), 4.2 / (inserted[1] - inserted[0]), delta=0.05)

    def test_pointing_at_a_locked_phrase_goes_in_front_of_it(self):
        self.lay(0.0, 9.0, 13.0, 17.0)
        self.lock("b")
        before = self.times()
        added = self.studio.add_clip("a", 9.5)["added"]
        self.assertEqual(self.times()["b"], before["b"])
        self.assertLess(self.starts()[added], 9.0)
        self.assert_no_overlap()

    def test_a_sliver_before_a_locked_phrase_does_not_take_its_place(self):
        self.lay(0.0, 9.0, 13.0, 17.0)
        self.lock("b")
        added = self.studio.add_clip("a", 8.5)["added"]
        self.assertEqual(self.starts()["b"], 9.0)
        self.assertLess(self.starts()[added], 9.0)

    def test_no_room_between_locked_phrases_is_refused_and_changes_nothing(self):
        self.lock("a", "b", "c", "d")
        before = self.times()
        with self.assertRaises(StudioError):
            self.studio.add_clip("a", 3.3)
        self.assertEqual(self.times(), before)
        self.assertEqual(self.studio.state()["history"]["undo"], 0)

    def test_one_undo_restores_shortened_phrases(self):
        self.lay(0.0, 3.75, 7.5, 15.0)
        self.lock("d")
        before = self.times()
        self.studio.add_clip("a", 2.5)
        self.studio.undo()
        self.assertEqual(self.times(), before)

    def test_locked_phrase_refuses_edits_until_unlocked(self):
        self.lock("b")
        for change in ({"start": 4.0}, {"end": 7.0}, {"x": 2.0}, {"z": 1.0}, {"facing": 45.0}, {"match": 10}, {"speedMode": "auto"}, {"phraseId": "a", "x": 1.0}):
            with self.assertRaises(StudioError):
                self.studio.update_clip("b", change)
        self.assertEqual(self.studio.state()["history"]["undo"], 0)
        self.studio.update_clip("b", {"locked": False})
        self.studio.update_clip("b", {"x": 2.0})
        self.assertEqual(self.studio._clip("b")["x"], 2.0)

    def test_locking_is_one_undo(self):
        self.studio.update_clip("b", {"locked": True})
        self.studio.undo()
        self.assertFalse(self.studio._clip("b")["locked"])

    def test_locked_phrase_cannot_be_deleted(self):
        self.lock("b")
        with self.assertRaises(StudioError):
            self.studio.delete_clip("b")
        self.assertIn("b", self.starts())

    def test_reorder_across_a_locked_phrase_is_refused(self):
        self.lock("b")
        before = self.times()
        with self.assertRaises(StudioError):
            self.studio.reorder_clips(["c", "b", "a", "d"])
        self.assertEqual(self.times(), before)

    def test_reorder_outside_a_locked_phrase_still_works(self):
        self.lock("a")
        self.studio.reorder_clips(["a", "c", "b", "d"])
        self.assertEqual(self.starts()["a"], 0.0)
        self.assertEqual([c["id"] for c in self.studio.clips], ["a", "c", "b", "d"])


def legs_skeleton() -> Skeleton:
    names = ("Hips", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase")
    return Skeleton(names, (-1, 0, 1, 2, 3, 0, 5, 6, 7), np.zeros((9, 3)), "", ())


def bent_legs(frames: int, hips_x: np.ndarray, toe_drop: np.ndarray | None = None) -> np.ndarray:
    """Hips [frames] along x, legs bent 30 degrees at the hip and 60 at the knee, so the ankles sit below the hips and can reach."""
    poses = np.zeros((frames, 9, 7))
    poses[..., 6] = 1.0
    poses[:, 0, :3] = np.column_stack([hips_x, np.full(frames, 0.7928), np.zeros(frames)])
    half = np.radians(15.0)
    for first, across in ((1, 0.1), (5, -0.1)):
        poses[:, first, :3] = [across, 0.0, 0.0]
        poses[:, first, 3:7] = [np.sin(-half), 0.0, 0.0, np.cos(half)]
        poses[:, first + 1, :3] = [0.0, -0.4, 0.0]
        poses[:, first + 1, 3:7] = [np.sin(2 * half), 0.0, 0.0, np.cos(2 * half)]
        poses[:, first + 2, :3] = [0.0, -0.4, 0.0]
        poses[:, first + 2, 3:7] = [np.sin(-half), 0.0, 0.0, np.cos(half)]
        poses[:, first + 3, :3] = [0.0, -0.08, 0.1]
    if toe_drop is not None:
        poses[:, 0, 1] -= toe_drop
    return poses


class PlantFeet(unittest.TestCase):
    def setUp(self):
        self.skeleton = legs_skeleton()
        self.frames = 31
        self.toe = self.skeleton.index("LeftToeBase")

    def contacts(self, start: int, end: int) -> np.ndarray:
        flags = np.zeros((self.frames, 4))
        flags[start:end] = 1.0
        return flags

    def world(self, poses: np.ndarray) -> np.ndarray:
        return world_positions(poses, self.skeleton.parents)

    def test_planted_foot_stops_sliding(self):
        poses = bent_legs(self.frames, np.linspace(-0.06, 0.06, self.frames))
        fixed = plant_feet(poses, self.contacts(8, 24), self.skeleton)
        before, after = self.world(poses)[8:24, self.toe, [0, 2]], self.world(fixed)[8:24, self.toe, [0, 2]]
        self.assertGreater(float(np.ptp(before[:, 0])), 0.05)
        self.assertLess(float(np.ptp(after, axis=0).max()), 0.003)

    def test_ends_still_meet_the_phrases(self):
        poses = bent_legs(self.frames, np.linspace(-0.06, 0.06, self.frames))
        fixed = plant_feet(poses, self.contacts(4, 28), self.skeleton)
        np.testing.assert_array_equal(fixed[0], poses[0])
        np.testing.assert_array_equal(fixed[-1], poses[-1])

    def test_plant_touching_the_start_stays_at_the_phrase_pose(self):
        poses = bent_legs(self.frames, np.linspace(0.0, 0.08, self.frames))
        fixed = plant_feet(poses, self.contacts(0, 14), self.skeleton)
        world = self.world(fixed)
        self.assertLess(float(np.linalg.norm(world[:14, self.toe, [0, 2]] - world[0, self.toe, [0, 2]], axis=1).max()), 0.003)

    def test_foot_keeps_its_rotation(self):
        poses = bent_legs(self.frames, np.linspace(-0.06, 0.06, self.frames))
        fixed = plant_feet(poses, self.contacts(8, 24), self.skeleton)
        foot = self.skeleton.index("LeftFoot")
        _, before = world_transforms(poses, self.skeleton.parents)
        _, after = world_transforms(fixed, self.skeleton.parents)
        np.testing.assert_allclose(np.abs(np.sum(before[:, foot] * after[:, foot], axis=1)), 1.0, atol=1e-6)

    def test_foot_under_the_floor_is_lifted(self):
        drop = 0.05 * np.sin(np.linspace(0.0, np.pi, self.frames)) ** 2
        poses = bent_legs(self.frames, np.zeros(self.frames), toe_drop=drop)
        fixed = plant_feet(poses, self.contacts(0, 0), self.skeleton)
        self.assertLess(float(self.world(poses)[:, self.toe, 1].min()), -0.02)
        self.assertGreater(float(self.world(fixed)[:, self.toe, 1].min()), -0.001)

    def test_nothing_to_do_changes_nothing(self):
        poses = bent_legs(self.frames, np.linspace(-0.06, 0.06, self.frames))
        np.testing.assert_array_equal(plant_feet(poses, self.contacts(0, 0), self.skeleton), poses)

    def test_a_long_drag_is_only_partly_held(self):
        poses = bent_legs(self.frames, np.linspace(-0.3, 0.3, self.frames))
        fixed = plant_feet(poses, self.contacts(8, 24), self.skeleton)
        moved = np.linalg.norm(self.world(fixed)[:, self.toe] - self.world(poses)[:, self.toe], axis=1)
        self.assertLess(float(moved.max()), MAX_SHIFT + 0.001)

    def test_generate_join_uses_contacts_across_long_joins(self):
        calls = []

        class Stand:
            joints = list(self.skeleton.names)

            @staticmethod
            def start():
                pass

            @staticmethod
            def generate(context, target, tokens):
                count = frames_from_tokens(tokens)
                calls.append(count)
                return bent_legs(count, np.zeros(count)), np.ones((count, 4)), 1.0

        with tempfile.TemporaryDirectory() as folder:
            cache = JoinCache(Path(folder))
            frames = MAX_SINGLE_FRAMES + 40
            context, target = bent_legs(4, np.zeros(4)), bent_legs(4, np.zeros(4))
            job = JoinJob(cache.key(context, target, frames), context, target, frames)
            join = generate_join(Stand, self.skeleton, cache, job)
        self.assertGreater(len(calls), 1)
        self.assertEqual(join.shape[0], sum(calls) - len(calls) + 1)


class ShortenToFit(unittest.TestCase):
    def test_room_to_spare_changes_nothing(self):
        self.assertEqual(_shorten_to_fit([3.0, 2.0], [3.0, 2.0], 6.0), [3.0, 2.0])

    def test_only_the_slow_phrase_shortens_when_that_is_enough(self):
        fitted = _shorten_to_fit([6.0, 3.0], [3.0, 3.0], 7.0)
        self.assertAlmostEqual(fitted[0], 4.0, places=3)
        self.assertAlmostEqual(fitted[1], 3.0, places=3)

    def test_phrases_end_at_the_same_speed_when_all_must_shorten(self):
        fitted = _shorten_to_fit([5.2, 4.2], [4.2, 4.2], 4.72)
        self.assertAlmostEqual(fitted[0], 2.36, places=3)
        self.assertAlmostEqual(fitted[1], 2.36, places=3)

    def test_a_phrase_already_past_the_limit_is_not_lengthened(self):
        fitted = _shorten_to_fit([1.0, 4.0], [4.0, 4.0], 3.0)
        self.assertAlmostEqual(fitted[0], 1.0, places=3)
        self.assertAlmostEqual(fitted[1], 2.0, places=3)

    def test_too_little_room_is_none(self):
        self.assertIsNone(_shorten_to_fit([4.0, 4.0], [4.0, 4.0], 3.0))


class BackgroundBake(unittest.TestCase):
    """An edit answers before MotionBricks has run; only the joins it touched wait."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.gate = threading.Event()
        self.ready = set()
        self.calls = []

        def compose(clips, library, song, engine, cache, memory, defer=False, engine_error=None, vias=None):
            key = tuple(c["x"] for c in clips)
            self.calls.append((key, defer))
            status = "engine-error" if engine_error else "ok" if key in self.ready else "pending"
            return {"poses": np.zeros((4, 2, 7)) + np.array([0, 0, 0, 0, 0, 0, 1.0]), "clips": {}, "joins": [{"from": "one", "to": "two", "status": status}],
                    "jobs": [key] if status == "pending" else [], "generated": 0, "reused": 0, "engineError": engine_error}

        def generate(engine, skeleton, cache, job):
            self.assertTrue(self.gate.wait(5))
            if self.failure:
                raise EngineError(self.failure)
            self.ready.add(job)

        self.failure = None
        for name, stand_in in (("compose", compose), ("generate_join", generate)):
            patcher = mock.patch(f"dsb.studio.{name}", stand_in)
            patcher.start()
            self.addCleanup(patcher.stop)
        engine = mock.Mock(start=lambda: None, stop=lambda: None)
        self.studio = Studio(Path(folder.name), engine=engine, background=True)
        self.addCleanup(self.studio.close)
        self.addCleanup(self.gate.set)
        self.studio.library.items = {"a": {"id": "a", "duration": 3.5}}
        self.studio.library.get_skeleton = lambda: Skeleton(("Hips", "Spine1"), (-1, 0), np.zeros((2, 3)), "", ())
        self.studio.song = {"barSeconds": 1.875, "duration": 30.0, "sections": []}
        self.studio.project["songPath"] = "song.wav"
        base = {"phraseId": "a", "facing": 0.0, "locked": False, "speedMode": "manual", "match": 70, "z": 3.0, "end": 3.0}
        self.studio.clips = [{**base, "id": "one", "start": 0.0, "x": 0.0}, {**base, "id": "two", "start": 3.75, "end": 6.75, "x": 1.0}]

    def status(self, payload):
        return payload["composed"]["joins"][0]["status"]

    def test_edit_answers_first_and_the_join_finishes_later(self):
        self.assertEqual(self.status(self.studio.update_clip("two", {"x": 2.0})), "pending")
        self.gate.set()
        self.assertEqual(self.status(self.studio.wait_for_joins(timeout=5.0)), "ok")

    def test_a_newer_edit_replaces_the_work_in_progress(self):
        self.studio.update_clip("two", {"x": 2.0})
        self.assertEqual(self.status(self.studio.update_clip("two", {"x": 3.0})), "pending")
        self.gate.set()
        self.assertEqual(self.status(self.studio.wait_for_joins(timeout=5.0)), "ok")
        self.assertEqual(self.calls[-1][0], (0.0, 3.0))

    def test_engine_failure_reaches_the_join(self):
        self.failure = "no engine"
        self.studio.update_clip("two", {"x": 2.0})
        self.gate.set()
        done = self.studio.wait_for_joins(timeout=5.0)
        self.assertEqual(self.status(done), "engine-error")
        self.assertEqual(done["composed"]["engineError"], "no engine")


class Library(unittest.TestCase):
    def test_lengths(self):
        self.assertEqual(nominal_bars(2.0)[1], "manual")
        self.assertEqual(nominal_bars(4.0), (2, "auto"))
        self.assertEqual(nominal_bars(8.0), (4, "auto"))
        self.assertEqual(nominal_bars(16.0)[1], "manual")


if __name__ == "__main__":
    unittest.main()
