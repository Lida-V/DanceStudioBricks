"""Checks for accent pins, join routes, several dancers, formations and the model exports. No GPU, model or audio needed."""
import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsb.compose import MAX_SINGLE_FRAMES, MIN_SINGLE_FRAMES, MIN_VIA_FRAMES, JoinCache, JoinJob, _route, compose, generate_join  # noqa: E402
from dsb.formation import CLEARANCE, TOO_CLOSE, close_calls, plan_vias, root_track  # noqa: E402
from dsb.mb_client import EngineError, frames_from_tokens  # noqa: E402
from dsb.motion import MODEL_FPS, quat_inverse, quat_mul, rotate, world_transforms, yaw_quat  # noqa: E402
from dsb.retarget import HUMANOID, humanoid_turns, incremental_turns, soma_skeleton, tpose, turns_from_tpose, write_vrma  # noqa: E402
from dsb.studio import Studio, StudioError, carried_pins  # noqa: E402
from dsb.vmd import BODY, STANDARD_ARM_DEGREES, STANDARD_LEG_UNITS, UNITS_PER_METRE, mmd_motion, read_pmx_bones, read_vmd, write_vmd  # noqa: E402
from dsb.warp import LOCAL_SPEED_RANGE, build, usable_pins  # noqa: E402

from test_core import bent_legs, legs_skeleton  # noqa: E402


class Pins(unittest.TestCase):
    def test_a_pin_lands_exactly_where_it_is_put(self):
        warp = build(10.0, 14.0, 4.0, [1.0, 3.0], np.zeros(0), 1.0, pins=[(1.0, 11.6)])
        self.assertAlmostEqual(float(warp.source_at(np.array([11.6]))[0]), 1.0, places=6)
        self.assertTrue((np.diff(warp.source_at(np.linspace(10.0, 14.0, 400))) >= -1e-9).all())
        pinned = [m for m in warp.matches if m["pinned"]]
        self.assertEqual([(m["source"], m["t"]) for m in pinned], [(1.0, 11.6)])

    def test_a_pin_holds_even_past_the_automatic_speed_range(self):
        warp = build(10.0, 14.0, 4.0, [1.0], np.zeros(0), 1.0, pins=[(1.0, 12.4)])
        low, _ = warp.local_speed_range()
        self.assertLess(low, LOCAL_SPEED_RANGE[0])
        self.assertAlmostEqual(float(warp.source_at(np.array([12.4]))[0]), 1.0, places=6)

    def test_automatic_matching_works_around_a_pin_without_moving_it(self):
        sound = np.array([11.1, 13.1])
        warp = build(10.0, 14.0, 4.0, [1.0, 3.0], sound, 1.0, pins=[(1.0, 11.1)])
        self.assertAlmostEqual(float(warp.source_at(np.array([11.1]))[0]), 1.0, places=6)
        free = next(m for m in warp.matches if not m["pinned"])
        self.assertTrue(free["matched"])
        self.assertAlmostEqual(float(warp.source_at(np.array([13.1]))[0]), 3.0, places=5)

    def test_a_pin_on_a_sound_is_shown_joined_to_it(self):
        warp = build(10.0, 14.0, 4.0, [1.0], np.array([11.5, 13.0]), 0.0, pins=[(1.0, 11.51)])
        self.assertEqual(warp.matches[0]["sound"], 11.5)
        away = build(10.0, 14.0, 4.0, [1.0], np.array([11.5, 13.0]), 0.0, pins=[(1.0, 12.2)])
        self.assertIsNone(away.matches[0]["sound"])

    def test_no_pins_is_the_same_warp_as_before(self):
        sound = np.sort(np.random.default_rng(3).uniform(10.0, 14.0, 12))
        plain, with_none = build(10.0, 14.0, 4.0, [0.7, 1.9, 3.2], sound, 0.8), build(10.0, 14.0, 4.0, [0.7, 1.9, 3.2], sound, 0.8, pins=[])
        np.testing.assert_array_equal(plain.knots_time, with_none.knots_time)
        np.testing.assert_array_equal(plain.knots_source, with_none.knots_source)

    def test_pins_that_cannot_hold_together_are_left_out(self):
        self.assertEqual(usable_pins([(1.0, 12.0), (2.0, 11.0), (3.0, 13.0)], 10.0, 14.0, 4.0), [(1.0, 12.0), (3.0, 13.0)])
        self.assertEqual(usable_pins([(1.0, 9.0), (5.0, 12.0)], 10.0, 14.0, 4.0), [])

    def test_a_move_carries_pins_and_a_stretch_keeps_their_time(self):
        pins = [{"source": 1.0, "at": 1.5}]
        self.assertEqual(carried_pins(pins, 10.0, 14.0, 20.0, 24.0, 4.0), [{"source": 1.0, "at": 1.5}])
        self.assertEqual(carried_pins(pins, 10.0, 14.0, 10.5, 14.0, 4.0), [{"source": 1.0, "at": 1.0}])
        self.assertEqual(carried_pins(pins, 10.0, 14.0, 10.0, 11.2, 4.0), [])


def studio_with_two_clips(test: unittest.TestCase) -> Studio:
    folder = tempfile.TemporaryDirectory()
    test.addCleanup(folder.cleanup)
    studio = Studio(Path(folder.name), engine=object())
    studio._recompose = lambda force=False: None
    studio._preview_joints = lambda: None
    studio.library.items = {"a": {"id": "a", "duration": 3.5}, "b": {"id": "b", "duration": 3.5}}
    bars = [round(i * 1.875, 3) for i in range(16)]
    studio.song = {"barSeconds": 1.875, "duration": 30.0, "bars": bars, "beats": [], "sections": [{"startBar": 0, "endBar": 16, "kind": "verse"}]}
    studio.project["songPath"] = "song.wav"
    base = {"phraseId": "a", "facing": 0.0, "locked": False, "speedMode": "manual", "match": 70}
    studio.clips = [{**base, "id": "one", "start": 0.0, "end": 3.0, "x": 0.0, "z": 3.0}, {**base, "id": "two", "start": 3.75, "end": 6.75, "x": 1.0, "z": 3.0},
                    {**base, "id": "three", "start": 15.0, "end": 18.0, "x": -1.0, "z": 2.0}]
    return studio


class ClipPins(unittest.TestCase):
    def setUp(self):
        self.studio = studio_with_two_clips(self)

    def clip(self, name):
        return next(c for c in self.studio.clips if c["id"] == name)

    def test_pins_are_kept_follow_a_move_and_undo(self):
        self.studio.update_clip("two", {"pins": [{"source": 1.0, "at": 1.2}]})
        self.assertEqual(self.clip("two")["pins"], [{"source": 1.0, "at": 1.2}])
        self.studio.update_clip("two", {"start": 7.5, "end": 10.5})
        self.assertEqual(self.clip("two")["pins"], [{"source": 1.0, "at": 1.2}])
        self.studio.update_clip("two", {"start": 8.0})
        self.assertEqual(self.clip("two")["pins"], [{"source": 1.0, "at": 0.7}])
        self.studio.undo()
        self.studio.undo()
        self.studio.undo()
        self.assertNotIn("pins", self.clip("two"))

    def test_impossible_pins_are_refused_and_leave_no_history(self):
        for pins in ([{"source": 1.0, "at": 5.0}], [{"source": 9.0, "at": 1.0}], [{"source": 1.0, "at": 2.0}, {"source": 2.0, "at": 1.0}], [{"source": "x"}]):
            with self.assertRaises(StudioError):
                self.studio.update_clip("two", {"pins": pins})
        self.assertEqual(self.studio.state()["history"]["undo"], 0)

    def test_another_phrase_drops_the_pins(self):
        self.studio.update_clip("two", {"pins": [{"source": 1.0, "at": 1.2}]})
        self.studio.update_clip("two", {"phraseId": "b"})
        self.assertEqual(self.clip("two")["pins"], [])

    def test_a_locked_phrase_refuses_pins(self):
        self.studio.update_clip("two", {"locked": True})
        with self.assertRaises(StudioError):
            self.studio.update_clip("two", {"pins": [{"source": 1.0, "at": 1.2}]})


class Dancers(unittest.TestCase):
    def setUp(self):
        self.studio = studio_with_two_clips(self)

    def by_id(self, dancer_id):
        return next(d for d in self.studio.dancers if d["id"] == dancer_id)

    def test_a_copied_dancer_stands_to_the_side_and_is_the_one_edited(self):
        self.studio.add_dancer("d1")
        self.assertEqual([d["id"] for d in self.studio.dancers], ["d1", "d2"])
        self.assertEqual(self.studio.project["active"], "d2")
        first, second = self.by_id("d1")["clips"], self.by_id("d2")["clips"]
        self.assertEqual([c["start"] for c in first], [c["start"] for c in second])
        self.assertEqual([c["x"] - o["x"] for c, o in zip(second, first)], [1.5, 1.5, 1.5])
        self.assertFalse({c["id"] for c in first} & {c["id"] for c in second})
        self.studio.update_clip(second[0]["id"], {"facing": 90.0})
        self.assertEqual(first[0]["facing"], 0.0)
        with self.assertRaises(StudioError):
            self.studio.update_clip("one", {"facing": 45.0})  # belongs to the dancer that is not selected

    def test_adding_and_removing_dancers_undo(self):
        self.studio.add_dancer()
        self.assertEqual(self.by_id("d2")["clips"], [])
        self.studio.delete_dancer("d2")
        self.assertEqual(self.studio.project["active"], "d1")
        self.studio.undo()
        self.assertEqual(len(self.studio.dancers), 2)
        self.studio.undo()
        self.assertEqual(len(self.studio.dancers), 1)
        with self.assertRaises(StudioError):
            self.studio.delete_dancer("d1")

    def test_a_formation_places_unlocked_phrases_until_the_next_one(self):
        self.studio.add_dancer("d1")
        self.studio.select_dancer("d1")
        self.studio.update_clip("two", {"locked": True})
        self.studio.set_formation(0.4, {"d1": {"x": -2.0, "z": 1.0}, "d2": {"x": 2.0, "z": 1.0, "facing": 180}}, "open")
        self.studio.set_formation(15.2, {"d1": {"x": 0.0, "z": 5.0}})
        lead, other = self.by_id("d1")["clips"], self.by_id("d2")["clips"]
        self.assertEqual([(c["x"], c["z"]) for c in lead], [(-2.0, 1.0), (1.0, 3.0), (0.0, 5.0)])  # the locked one stayed
        self.assertEqual([(c["x"], c["z"], c["facing"]) for c in other[:2]], [(2.0, 1.0, 180.0), (2.0, 1.0, 180.0)])
        self.assertEqual((other[2]["x"], other[2]["z"]), (2.0, 1.0))  # d2 has no spot in the second formation, so it stays where the first put it
        self.assertEqual([f["time"] for f in self.studio.project["formations"]], [0.0, 15.0])

    def test_a_formation_on_the_same_bar_replaces_and_deleting_falls_back(self):
        self.studio.set_formation(0.0, {"d1": {"x": -2.0, "z": 1.0}})
        self.studio.set_formation(0.3, {"d1": {"x": 3.0, "z": 4.0}})
        self.assertEqual(len(self.studio.project["formations"]), 1)
        later = self.studio.set_formation(15.0, {"d1": {"x": 0.0, "z": 0.0}})["formation"]
        self.assertEqual(self.studio.clips[2]["x"], 0.0)
        self.studio.delete_formation(later)
        self.assertEqual((self.studio.clips[2]["x"], self.studio.clips[2]["z"]), (3.0, 4.0))
        self.studio.undo()
        self.assertEqual(self.studio.clips[2]["x"], 0.0)

    def test_a_formation_refits_phrases_on_automatic_timing(self):
        self.studio.clips[0]["speedMode"] = "auto"
        self.studio.set_formation(3.75, {"d1": {"x": 3.0, "z": 3.0}})
        self.assertAlmostEqual(self.studio.clips[0]["end"], 3.5 / 1.35, places=2)  # a longer walk leaves less time, down to the speed limit

    def test_spots_are_kept_on_the_stage_and_bad_ones_refused(self):
        self.studio.set_formation(0.0, {"d1": {"x": 99.0, "z": -5.0}})
        self.assertEqual((self.studio.clips[0]["x"], self.studio.clips[0]["z"]), (4.0, 0.0))
        with self.assertRaises(StudioError):
            self.studio.set_formation(0.0, {"d1": {"x": "left"}})
        with self.assertRaises(StudioError):
            self.studio.set_formation(0.0, {"nobody": {"x": 0.0, "z": 0.0}})

    def test_sync_gives_the_others_the_same_phrases_at_their_own_side(self):
        self.studio.add_dancer("d1")
        self.studio.select_dancer("d1")
        self.studio.delete_clip("three")
        self.studio.update_clip("one", {"x": 0.5})
        self.studio.sync_dancers()
        lead, other = self.by_id("d1")["clips"], self.by_id("d2")["clips"]
        self.assertEqual([c["start"] for c in other], [0.0, 3.75])
        self.assertEqual([round(c["x"] - o["x"], 2) for c, o in zip(other, lead)], [1.0, 1.0])  # as far apart as their first phrases were


class SongStash(unittest.TestCase):
    def test_an_arrangement_comes_back_with_its_song(self):
        studio = studio_with_two_clips(self)
        first, second = studio.data / "a.wav", studio.data / "b.wav"
        for file in (first, second):
            file.write_bytes(b"x")
        studio.project["songPath"] = str(first)
        analysed = {"barSeconds": 1.875, "duration": 30.0, "bars": [0.0], "beats": [], "sections": [], "accents": []}
        with mock.patch("dsb.music.analyze", return_value=analysed):
            self.assertEqual(studio.load_song(str(second))["restored"], 0)
            self.assertEqual(studio.clips, [])
            self.assertEqual(studio.load_song(str(first))["restored"], 3)
        self.assertEqual([c["id"] for c in studio.clips], ["one", "two", "three"])


class Routes(unittest.TestCase):
    def test_a_straight_join_is_one_generation_or_equal_pieces(self):
        begin, finish = np.array([0.0, 0.0]), np.array([2.0, 0.0])
        self.assertEqual([frames for frames, _ in _route(begin, finish, 40, None)], [40])
        long = _route(begin, finish, 2 * MAX_SINGLE_FRAMES, None)
        self.assertEqual(len(long), 3)
        np.testing.assert_allclose(long[-1][1], finish)

    def test_a_via_makes_two_legs_that_are_each_long_enough(self):
        stops = _route(np.array([0.0, 0.0]), np.array([2.0, 0.0]), MIN_VIA_FRAMES, (0.2, 0.9))
        self.assertEqual(len(stops), 2)
        self.assertTrue(all(frames >= MIN_SINGLE_FRAMES for frames, _ in stops))
        np.testing.assert_allclose(stops[0][1], [0.2, 0.9])
        self.assertAlmostEqual(sum(frames for frames, _ in stops) - 1, MIN_VIA_FRAMES)

    def test_a_join_with_a_via_passes_through_it(self):
        skeleton = legs_skeleton()
        seen = []

        class Stand:
            joints = list(skeleton.names)

            @staticmethod
            def start():
                pass

            @staticmethod
            def generate(context, target, tokens):
                count = frames_from_tokens(tokens)
                seen.append((target.shape[0], target[0, 0, [0, 2]].copy()))
                return bent_legs(count, np.zeros(count)), np.zeros((count, 4)), 1.0

        with tempfile.TemporaryDirectory() as folder:
            cache = JoinCache(Path(folder))
            context, target = bent_legs(4, np.zeros(4)), bent_legs(4, np.full(4, 2.0))
            self.assertNotEqual(cache.key(context, target, 60), cache.key(context, target, 60, (1.0, 0.8)))
            generate_join(Stand, skeleton, cache, JoinJob("k", context, target, 60, (1.0, 0.8)))
        self.assertEqual([poses for poses, _ in seen], [1, 4])  # one pose to pass through, then the full target
        np.testing.assert_allclose(seen[0][1], [1.0, 0.8])
        np.testing.assert_allclose(seen[1][1], [2.0, 0.0])


def standing(clip_id: str, first: int, frames: int, x: float, z: float):
    poses = np.zeros((frames, 1, 7))
    poses[:, 0, 0], poses[:, 0, 2] = x, z
    return {"id": clip_id}, first, poses


class Paths(unittest.TestCase):
    def test_the_later_dancer_walks_round_one_standing_in_the_way(self):
        lead = [standing("a1", 0, 200, 0.0, 3.0)]
        walker = [standing("b1", 0, 60, -2.0, 3.0), standing("b2", 120, 60, 2.0, 3.0)]
        plans = plan_vias([lead, walker], 200)
        self.assertEqual(plans[0], {})
        via = plans[1][("b1", "b2")]
        self.assertGreaterEqual(abs(via[1] - 3.0), 0.6)
        reversed_order = plan_vias([walker, lead], 200)
        self.assertEqual(reversed_order, [{}, {}])  # the first dancer keeps the straight path, and one standing still cannot step aside

    def test_two_crossing_dancers_only_the_later_one_detours(self):
        first = [standing("a1", 0, 60, -2.0, 3.0), standing("a2", 120, 60, 2.0, 3.0)]
        second = [standing("b1", 0, 60, 2.0, 3.0), standing("b2", 120, 60, -2.0, 3.0)]
        plans = plan_vias([first, second], 200)
        self.assertEqual(plans[0], {})
        self.assertIn(("b1", "b2"), plans[1])

    def test_paths_with_room_and_joins_too_short_stay_straight(self):
        lead = [standing("a1", 0, 200, 0.0, 0.5)]
        walker = [standing("b1", 0, 60, -2.0, 3.0), standing("b2", 120, 60, 2.0, 3.0)]
        self.assertEqual(plan_vias([lead, walker], 200), [{}, {}])
        blocked = [standing("a1", 0, 200, 0.0, 3.0)]
        hurried = [standing("b1", 0, 60, -2.0, 3.0), standing("b2", 59 + MIN_VIA_FRAMES - 2, 60, 2.0, 3.0)]
        self.assertEqual(plan_vias([blocked, hurried], 200), [{}, {}])

    def test_a_planned_path_keeps_the_clearance(self):
        lead = [standing("a1", 0, 200, 0.0, 3.0)]
        walker = [standing("b1", 0, 60, -2.0, 3.0), standing("b2", 120, 60, 2.0, 3.0)]
        via = np.array(plan_vias([lead, walker], 200)[1][("b1", "b2")])
        for point in (via, (np.array([-2.0, 3.0]) + via) / 2, (np.array([2.0, 3.0]) + via) / 2):
            self.assertGreaterEqual(np.linalg.norm(point - np.array([0.0, 3.0])), CLEARANCE - 0.05)

    def test_the_step_aside_is_taken_where_the_path_is_blocked(self):
        lead = [standing("a1", 0, 250, 1.0, 3.0)]  # a quarter of the way along the walker's path, where a detour at the middle would not help
        walker = [standing("b1", 0, 60, 2.0, 3.0), standing("b2", 150, 60, -2.0, 3.0)]
        via = np.array(plan_vias([lead, walker], 250)[1][("b1", "b2")])
        self.assertGreater(via[0], 0.0)
        path = np.concatenate([np.linspace([2.0, 3.0], via, 50), np.linspace(via, [-2.0, 3.0], 50)])
        self.assertGreaterEqual(np.linalg.norm(path - np.array([1.0, 3.0]), axis=1).min(), CLEARANCE - 0.02)

    def test_root_track_holds_the_ends_and_walks_straight_between(self):
        track = root_track([standing("a", 10, 10, 0.0, 0.0), standing("b", 30, 10, 2.0, 0.0)], 50)
        np.testing.assert_allclose(track[[0, 19, 25, 30, 49], 0], [0.0, 0.0, 2.0 * 6 / 11, 2.0, 2.0])

    def test_close_calls_report_when_and_how_near(self):
        still = np.tile([0.0, 0.0], (90, 1))
        passing = np.column_stack([np.linspace(-3.0, 3.0, 90), np.zeros(90)])
        calls = close_calls([("d1", still), ("d2", passing), ("d3", still + [0.0, 5.0])])
        self.assertEqual([(call["a"], call["b"]) for call in calls], [("d1", "d2")])
        self.assertLess(calls[0]["distance"], TOO_CLOSE)
        self.assertTrue(1.2 < calls[0]["start"] < 1.5 < calls[0]["end"] < 1.8)


def turned(skeleton, pose: np.ndarray, joint: str, quat: np.ndarray) -> np.ndarray:
    """The pose with one joint turned by `quat` in world space, keeping everything above it."""
    out = pose.copy()
    index = skeleton.index(joint)
    _, world = world_transforms(pose, skeleton.parents)
    parent = skeleton.parents[index]
    new_world = quat_mul(quat, world[0, index])
    out[0, index, 3:7] = new_world if parent < 0 else quat_mul(quat_inverse(world[0, parent]), new_world)
    return out


def axis_quat(axis, degrees: float) -> np.ndarray:
    axis = np.array(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate([axis * np.sin(np.radians(degrees) / 2.0), [np.cos(np.radians(degrees) / 2.0)]])


class Retarget(unittest.TestCase):
    def setUp(self):
        self.skeleton = soma_skeleton()
        self.rest = tpose(self.skeleton)

    def direction(self, pose, a, b):
        world = world_transforms(pose, self.skeleton.parents)[0][0]
        vector = world[self.skeleton.index(b)] - world[self.skeleton.index(a)]
        return vector / np.linalg.norm(vector)

    def test_the_tpose_stands_on_the_floor_with_arms_level(self):
        world = world_transforms(self.rest, self.skeleton.parents)[0][0]
        self.assertAlmostEqual(float(world[:, 1].min()), 0.0, places=6)
        np.testing.assert_allclose(self.direction(self.rest, "LeftArm", "LeftHand"), [1.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(self.direction(self.rest, "RightArm", "RightHand"), [-1.0, 0.0, 0.0], atol=1e-3)
        self.assertLess(self.direction(self.rest, "LeftLeg", "LeftFoot")[1], -0.99)
        self.assertGreater(self.direction(self.rest, "LeftFoot", "LeftToeBase")[2], 0.9)  # facing +Z

    def test_the_tpose_measures_as_no_turn_and_a_spin_as_that_spin(self):
        bones, turns = humanoid_turns(self.rest, self.skeleton)
        self.assertEqual(bones, list(HUMANOID))
        np.testing.assert_allclose(np.abs(turns[0, :, 3]), 1.0, atol=1e-6)
        spun = turned(self.skeleton, self.rest, "Hips", yaw_quat(np.radians(90.0)))
        _, turns = humanoid_turns(spun, self.skeleton)
        np.testing.assert_allclose(np.abs(turns[0] @ yaw_quat(np.radians(90.0))), 1.0, atol=1e-6)

    def test_vrma_holds_the_humanoid_bones_and_the_motion(self):
        raised = turned(self.skeleton, self.rest, "LeftArm", axis_quat([0, 0, 1], 60.0))
        poses = np.concatenate([self.rest, raised, raised], axis=0)
        poses[2, 0, :3] += [0.3, -0.1, 0.5]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "a.vrma"
            write_vrma(target, poses, self.skeleton, 30.0)
            data = target.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", data, 0)
        self.assertEqual((magic, version, length), (b"glTF", 2, len(data)))
        size, kind = struct.unpack_from("<I4s", data, 12)
        self.assertEqual(kind, b"JSON")
        document = json.loads(data[20:20 + size])
        binary = data[20 + size + 8:]
        bones = document["extensions"]["VRMC_vrm_animation"]["humanoid"]["humanBones"]
        self.assertEqual(set(bones), set(HUMANOID))
        self.assertEqual(document["extensions"]["VRMC_vrm_animation"]["specVersion"], "1.0")
        nodes = document["nodes"]
        self.assertEqual(sum(len(node.get("children", [])) for node in nodes), len(nodes) - 1)
        self.assertAlmostEqual(nodes[bones["hips"]["node"]]["translation"][1], world_transforms(self.rest, self.skeleton.parents)[0][0, 0, 1], places=5)

        def values(channel):
            sampler = document["animations"][0]["samplers"][channel["sampler"]]
            accessor = document["accessors"][sampler["output"]]
            view = document["bufferViews"][accessor["bufferView"]]
            width = {"VEC3": 3, "VEC4": 4}[accessor["type"]]
            return np.frombuffer(binary, dtype="<f4", count=accessor["count"] * width, offset=view["byteOffset"]).reshape(-1, width)

        channels = {(channel["target"]["node"], channel["target"]["path"]): channel for channel in document["animations"][0]["channels"]}
        arm = values(channels[(bones["leftUpperArm"]["node"], "rotation")])
        np.testing.assert_allclose(np.abs(arm[0, 3]), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.abs(arm[1] @ axis_quat([0, 0, 1], 60.0)), 1.0, atol=1e-5)  # the shoulder above did not turn, so local is world
        forearm = values(channels[(bones["leftLowerArm"]["node"], "rotation")])
        np.testing.assert_allclose(np.abs(forearm[1, 3]), 1.0, atol=1e-5)  # it only follows its parent
        hips = values(channels[(bones["hips"]["node"], "translation")])
        np.testing.assert_allclose(hips[2] - hips[0], [0.3, -0.1, 0.5], atol=1e-5)
        times = document["accessors"][document["animations"][0]["samplers"][0]["input"]]
        self.assertEqual((times["count"], times["min"], round(times["max"][0], 5)), (3, [0.0], round(2 / 30.0, 5)))

    def test_incremental_turns_match_full_recomputation_after_partial_edits(self):
        raised = turned(self.skeleton, self.rest, "LeftArm", axis_quat([0, 0, 1], 55.0))
        curled = turned(self.skeleton, raised, "LeftHandIndex2", axis_quat([0, 0, 1], 70.0))
        poses = np.concatenate([self.rest, raised, curled, self.rest, raised], axis=0)
        raw0, inc0 = incremental_turns(poses, self.skeleton)
        _, full0 = humanoid_turns(poses, self.skeleton)
        np.testing.assert_allclose(inc0, full0, atol=1e-7)
        edited = poses.copy()
        edited[2] = turned(self.skeleton, self.rest, "RightHandPinky3", axis_quat([0, 0, 1], -65.0))[0]
        raw1, inc1 = incremental_turns(edited, self.skeleton, poses, raw0)
        _, full1 = humanoid_turns(edited, self.skeleton)
        np.testing.assert_allclose(inc1, full1, atol=1e-7)
        # No-op pass keeps exact values without mutating raw1
        raw2, inc2 = incremental_turns(edited, self.skeleton, edited, raw1)
        np.testing.assert_array_equal(inc2, inc1)

    def test_bundled_default_mannequin_glb_has_all_finger_bones_and_cc0_license(self):
        root = Path(__file__).resolve().parents[2] / "frontend" / "public" / "models"
        glb = root / "default-mannequin.glb"
        lic = root / "LICENSE.txt"
        self.assertTrue(glb.is_file())
        self.assertIn("CC0", lic.read_text(encoding="utf-8"))
        data = glb.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", data, 0)
        self.assertEqual((magic, version, length), (b"glTF", 2, len(data)))
        size, kind = struct.unpack_from("<I4s", data, 12)
        self.assertEqual(kind, b"JSON")
        doc = json.loads(data[20:20 + size])
        nodes = doc["nodes"]
        joint_names = {nodes[j].get("name", "") for s in doc["skins"] for j in s["joints"]}
        for side in ("l", "r"):
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                for part in ("01", "02", "03"):
                    self.assertIn(f"{finger}_{part}_{side}", joint_names)


def a_pose_direction(side: str, degrees: float) -> np.ndarray:
    sign = 1.0 if side == "左" else -1.0
    return np.array([sign * np.cos(np.radians(degrees)), -np.sin(np.radians(degrees)), 0.0])


class Vmd(unittest.TestCase):
    def setUp(self):
        self.skeleton = soma_skeleton()
        self.rest = tpose(self.skeleton)
        pose = self.rest
        for joint, axis, degrees in (("Hips", [0, 1, 0], 40.0), ("Chest", [1, 0, 0], 20.0), ("LeftArm", [0, 0, 1], -70.0), ("LeftForeArm", [0, 1, 0], -50.0),
                                     ("RightArm", [1, 1, 0], 35.0), ("RightForeArm", [0, 0, 1], 80.0), ("Head", [0, 1, 0], -30.0), ("LeftLeg", [1, 0, 0], -45.0)):
            pose = turned(self.skeleton, pose, joint, axis_quat(axis, degrees))
        self.pose = pose

    def world_turns(self, motion):
        """Each MMD bone's world rotation in this app's space, from the local rotations that were written."""
        flip = np.array([-1.0, -1.0, 1.0, 1.0])
        world = {}
        for name in BODY:
            if name not in motion:
                continue
            local = motion[name][1][0] * flip
            parent = BODY[name][1]
            while parent is not None and parent not in motion:
                parent = BODY[parent][1]
            world[name] = local if parent is None else quat_mul(world[parent], local)
        return world

    def soma_direction(self, pose, a, b):
        positions = world_transforms(pose, self.skeleton.parents)[0][0]
        vector = positions[self.skeleton.index(b)] - positions[self.skeleton.index(a)]
        return vector / np.linalg.norm(vector)

    def test_the_tpose_lifts_the_arms_from_the_a_pose_and_moves_nothing_else(self):
        motion = mmd_motion(self.rest, self.skeleton)
        world = self.world_turns(motion)
        for side in "左右":
            np.testing.assert_allclose(rotate(world[side + "腕"], a_pose_direction(side, STANDARD_ARM_DEGREES)), [1.0 if side == "左" else -1.0, 0.0, 0.0], atol=1e-5)
            np.testing.assert_allclose(np.abs(motion[side + "ひじ"][1][0, 3]), 1.0, atol=1e-6)  # the forearm follows the upper arm
            np.testing.assert_allclose(motion[side + "足ＩＫ"][0][0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(motion["センター"][0][0], [0.0, 0.0, 0.0], atol=1e-6)
        for name in ("下半身", "上半身", "上半身2", "首", "頭", "左足", "右ひざ"):
            np.testing.assert_allclose(np.abs(motion[name][1][0, 3]), 1.0, atol=1e-6)

    def test_arm_bones_point_where_the_soma_arms_point(self):
        world = self.world_turns(mmd_motion(self.pose, self.skeleton))
        for side, soma in (("左", "Left"), ("右", "Right")):
            rest = a_pose_direction(side, STANDARD_ARM_DEGREES)
            np.testing.assert_allclose(rotate(world[side + "腕"], rest), self.soma_direction(self.pose, soma + "Arm", soma + "ForeArm"), atol=1e-3)
            np.testing.assert_allclose(rotate(world[side + "ひじ"], rest), self.soma_direction(self.pose, soma + "ForeArm", soma + "Hand"), atol=1e-3)

    def test_a_model_with_its_own_arm_angles_is_followed(self):
        def place(side, upper, lower):
            shoulder = np.array([0.1 if side == "左" else -0.1, 1.3, 0.0])
            elbow = shoulder + 0.25 * a_pose_direction(side, upper)
            wrist = elbow + 0.22 * a_pose_direction(side, lower)
            return {side + "腕": shoulder, side + "ひじ": elbow, side + "手首": wrist, side + "中指１": wrist + 0.08 * a_pose_direction(side, lower)}

        bones = {**place("左", 45.0, 30.0), **place("右", 45.0, 30.0)}
        bones.update({name: np.zeros(3) for name in BODY if name not in bones})
        world = self.world_turns(mmd_motion(self.pose, self.skeleton, bones))
        np.testing.assert_allclose(rotate(world["左腕"], a_pose_direction("左", 45.0)), self.soma_direction(self.pose, "LeftArm", "LeftForeArm"), atol=1e-3)
        np.testing.assert_allclose(rotate(world["左ひじ"], a_pose_direction("左", 30.0)), self.soma_direction(self.pose, "LeftForeArm", "LeftHand"), atol=1e-3)

    def test_trunk_and_legs_carry_the_soma_turns(self):
        world = self.world_turns(mmd_motion(self.pose, self.skeleton))
        expected = dict(zip(BODY, turns_from_tpose(self.pose, self.skeleton, [BODY[name][0] for name in BODY])[0]))
        for name in ("下半身", "上半身2", "頭", "左足", "左ひざ", "右足首"):
            np.testing.assert_allclose(np.abs(world[name] @ expected[name]), 1.0, atol=1e-6)

    def test_positions_use_mmd_space_and_the_model_leg_length(self):
        moved = self.rest.copy()
        moved[0, 0, :3] += [0.5, -0.2, 0.3]
        rest_world = world_transforms(self.rest, self.skeleton.parents)[0][0]
        legs = [self.skeleton.index(side + name) for side in ("Left", "Right") for name in ("Leg", "Shin", "Foot")]
        soma_leg = np.mean([np.linalg.norm(rest_world[legs[i + 1]] - rest_world[legs[i]]) + np.linalg.norm(rest_world[legs[i + 2]] - rest_world[legs[i + 1]]) for i in (0, 3)])
        scale = STANDARD_LEG_UNITS / soma_leg
        motion = mmd_motion(moved, self.skeleton)
        np.testing.assert_allclose(motion["センター"][0][0], np.array([0.5, -0.2, -0.3]) * scale, atol=1e-3)
        np.testing.assert_allclose(motion["左足ＩＫ"][0][0], np.array([0.5, -0.2, -0.3]) * scale, atol=1e-3)
        bones = {"左足": np.array([0.1, 1.0, 0.0]), "左ひざ": np.array([0.1, 0.5, 0.0]), "左足首": np.array([0.1, 0.1, 0.0]),
                 "右足": np.array([-0.1, 1.0, 0.0]), "右ひざ": np.array([-0.1, 0.5, 0.0]), "右足首": np.array([-0.1, 0.1, 0.0]), "左足ＩＫ": np.zeros(3), "下半身": np.zeros(3)}
        fitted = mmd_motion(moved, self.skeleton, bones)
        np.testing.assert_allclose(fitted["センター"][0][0], np.array([0.5, -0.2, -0.3]) * 0.9 * UNITS_PER_METRE / soma_leg, atol=1e-3)
        self.assertEqual(set(fitted), {"センター", "下半身", "左足ＩＫ", "左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首"})  # only bones the model has: no arms, no right foot IK

    def test_a_written_file_reads_back_with_still_bones_thinned(self):
        poses = np.concatenate([self.rest] * 4 + [self.pose], axis=0)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "a.vmd"
            report = write_vmd(target, poses, self.skeleton)
            keys = read_vmd(target)
            size = target.stat().st_size
        self.assertEqual(size, 54 + 111 * report["keys"] + 16)
        self.assertEqual([frame for frame, _, _ in keys["左腕"]], [0, 3, 4])  # frames 1 and 2 equal both neighbours
        self.assertEqual([frame for frame, _, _ in keys["左小指３"]], [0, 4])
        self.assertIn("右足ＩＫ", keys)
        motion = mmd_motion(poses, self.skeleton)
        np.testing.assert_allclose(keys["頭"][-1][2], motion["頭"][1][4], atol=1e-6)


def pmx_bytes() -> bytes:
    def text(value: str) -> bytes:
        raw = value.encode("utf-16-le")
        return struct.pack("<i", len(raw)) + raw

    out = b"PMX " + struct.pack("<fB8B", 2.0, 8, 0, 0, 1, 1, 1, 1, 1, 1) + text("model") * 4
    vertex = struct.pack("<8f", *range(8))
    out += struct.pack("<i", 3)
    out += vertex + struct.pack("<Bb", 0, 0) + struct.pack("<f", 1.0)  # BDEF1
    out += vertex + struct.pack("<Bbbf", 1, 0, 1, 0.5) + struct.pack("<f", 1.0)  # BDEF2
    out += vertex + struct.pack("<Bbbf9f", 3, 0, 1, 0.5, *range(9)) + struct.pack("<f", 1.0)  # SDEF
    out += struct.pack("<i3B", 3, 0, 1, 2)
    out += struct.pack("<i", 1) + text("tex.png")
    out += struct.pack("<i", 1) + text("mat") + text("mat") + bytes(16 + 12 + 4 + 12 + 1 + 16 + 4) + struct.pack("<bbBBb", 0, -1, 0, 0, 0) + text("") + struct.pack("<i", 3)
    bones = [("センター", (0.0, 8.0, 0.0), 0x001E, b""), ("左腕", (2.5, 16.0, 0.25), 0x001F, b""),
             ("左足ＩＫ", (1.0, 1.0, -0.5), 0x003E, struct.pack("<bifi", 1, 40, 2.0, 2) + struct.pack("<bB", 0, 0) + struct.pack("<bB6f", 1, 1, *range(6))),
             ("左ひじ", (5.0, 14.0, 0.25), 0x0D1B, struct.pack("<bf", 0, 0.5) + bytes(12) + bytes(24))]
    out += struct.pack("<i", len(bones))
    for name, position, flags, extra in bones:
        tail = struct.pack("<b", 1) if flags & 1 else struct.pack("<3f", 0.0, 1.0, 0.0)
        out += text(name) + text(name) + struct.pack("<3f", *position) + struct.pack("<bi", -1, 0) + struct.pack("<H", flags) + tail + extra
    return out + struct.pack("<i", 0)


class Pmx(unittest.TestCase):
    def test_bones_come_out_in_metres_facing_the_other_way(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "m.pmx"
            file.write_bytes(pmx_bytes())
            bones = read_pmx_bones(file)
            self.assertEqual(list(bones), ["センター", "左腕", "左足ＩＫ", "左ひじ"])
            np.testing.assert_allclose(bones["左腕"], [0.2, 1.28, -0.02])
            np.testing.assert_allclose(bones["左ひじ"], [0.4, 1.12, -0.02])
            file.write_bytes(b"Pmd\x00" + bytes(40))
            with self.assertRaises(ValueError):
                read_pmx_bones(file)
            file.write_bytes(pmx_bytes()[:200])
            with self.assertRaises(ValueError):
                read_pmx_bones(file)


class CutsAndExactInsert(unittest.TestCase):
    def setUp(self):
        self.studio = studio_with_two_clips(self)
        self.studio.library.items["b"] = {"id": "b", "duration": 2.0}

    def clip(self, name):
        return next(c for c in self.studio.clips if c["id"] == name)

    def test_split_before_after_and_undo(self):
        self.studio.update_clip("two", {"pins": [{"source": 0.7, "at": 0.6}, {"source": 2.8, "at": 2.4}]})
        res = self.studio.cut_clip("two", 5.25, "split")
        second_id = res["selected"]
        first, second = self.clip("two"), self.clip(second_id)
        self.assertEqual((first["start"], first["end"]), (3.75, 5.25))
        self.assertEqual((second["start"], second["end"]), (5.25, 6.75))
        self.assertAlmostEqual(first.get("trimStart", 0.0), 0.0, places=3)
        self.assertAlmostEqual(first["trimEnd"], second["trimStart"], places=3)
        self.assertEqual(first["pins"], [{"source": 0.7, "at": 0.6}])
        self.assertEqual(second["pins"], [{"source": 2.8, "at": 0.9}])
        self.studio.undo()
        self.assertEqual(len(self.studio.clips), 3)
        self.assertEqual((self.clip("two")["start"], self.clip("two")["end"]), (3.75, 6.75))
        self.assertNotIn("trimStart", self.clip("two"))

        self.studio.cut_clip("one", 1.2, "before")
        self.assertEqual((self.clip("one")["start"], self.clip("one")["end"]), (1.2, 3.0))
        self.assertGreater(self.clip("one")["trimStart"], 1.0)
        self.studio.cut_clip("one", 2.4, "after")
        self.assertEqual((self.clip("one")["start"], self.clip("one")["end"]), (1.2, 2.4))
        self.assertLess(self.clip("one")["trimEnd"], 3.5)

    def test_locked_or_edge_cuts_are_refused(self):
        self.studio.update_clip("one", {"locked": True})
        with self.assertRaises(StudioError):
            self.studio.cut_clip("one", 1.5, "split")
        with self.assertRaises(StudioError):
            self.studio.cut_clip("two", 3.85, "split")
        with self.assertRaises(StudioError):
            self.studio.cut_clip("two", 6.6, "after")

    def test_split_preserves_world_motion_and_reports_cut_join(self):
        skeleton = legs_skeleton()
        frames = 121  # 4.0 seconds at 30 fps
        xs = np.linspace(0.0, 1.2, frames)
        poses = bent_legs(frames, xs)
        turns = np.linspace(0.0, np.deg2rad(60.0), frames)
        poses[:, 0, 3:7] = np.array([yaw_quat(float(t)) for t in turns])
        self.studio.library.skeleton = skeleton
        self.studio.library._poses["walk"] = poses
        self.studio.library.items["walk"] = {"id": "walk", "duration": 4.0, "accents": []}
        self.studio.clips = [{"id": "w", "phraseId": "walk", "start": 1.0, "end": 5.0,
                              "x": -0.5, "z": 2.5, "facing": 30.0, "locked": False, "speedMode": "manual", "match": 0}]

        class NoEngine:
            joints = list(skeleton.names)

            @staticmethod
            def start():
                raise EngineError("offline")

        whole = compose(self.studio.clips, self.studio.library, self.studio.song, NoEngine, self.studio.cache)
        second_id = self.studio.cut_clip("w", 2.6, "split")["selected"]
        split_baked = compose(self.studio.clips, self.studio.library, self.studio.song, NoEngine, self.studio.cache)
        self.assertEqual(split_baked["joins"][0]["status"], "cut")
        np.testing.assert_allclose(split_baked["poses"][30:151, 0, :3], whole["poses"][30:151, 0, :3], atol=3e-3)

        # Resetting trim on the second half restores its original anchor spot and full span.
        self.studio.update_clip(second_id, {"start": 1.0, "end": 5.0, "trimStart": None, "trimEnd": None})
        restored = self.clip(second_id)
        self.assertNotIn("trimStart", restored)
        self.assertNotIn("trimEnd", restored)
        self.assertAlmostEqual(restored["x"], -0.5, places=2)
        self.assertAlmostEqual(restored["z"], 2.5, places=2)
        self.assertAlmostEqual(restored["facing"], 30.0, places=1)

    def test_replacing_phrase_resets_trim_and_pins_and_opens_cut_gap(self):
        second_id = self.studio.cut_clip("two", 5.25, "split")["selected"]
        self.studio.update_clip(second_id, {"pins": [{"source": 2.5, "at": 0.7}]})
        self.assertIn("trimStart", self.clip(second_id))
        self.studio.update_clip(second_id, {"phraseId": "b", "speedMode": "auto"})
        replaced = self.clip(second_id)
        self.assertEqual(replaced["phraseId"], "b")
        self.assertNotIn("trimStart", replaced)
        self.assertNotIn("trimEnd", replaced)
        self.assertEqual(replaced["pins"], [])
        self.assertAlmostEqual(replaced["end"], 5.25 + 2.0 / 0.8, places=2)
        self.assertLessEqual(self.clip("two")["end"], 5.25 - 0.4 + 1e-3)

    def test_exact_insert_lands_on_requested_spot(self):
        # Narrow gap before clip 'three' (15.0 - 14.0 = 1.0 < barSeconds 1.875) keeps exact start 14.0 instead of snapping to 15.0.
        added = self.studio.add_clip("b", 14.0, exact=True)["added"]
        self.assertEqual(self.clip(added)["start"], 14.0)
        self.assertGreaterEqual(self.clip("three")["start"], 16.0 + 0.4)

        # Dropping inside the tail of 'one' ([0.0, 3.0]) at 2.5 shortens 'one' and starts new clip right at 2.5.
        self.studio.undo()
        mid = self.studio.add_clip("b", 2.5, exact=True)["added"]
        self.assertEqual(self.clip(mid)["start"], 2.5)
        self.assertLessEqual(self.clip("one")["end"], 2.1)

        # Dropping near the head of 'one' at 0.5 puts new clip at 0.5 and slides 'one' behind it.
        self.studio.undo()
        head = self.studio.add_clip("b", 0.5, x=1.2, z=2.4, exact=True)["added"]
        self.assertEqual((self.clip(head)["start"], self.clip(head)["x"], self.clip(head)["z"]), (0.5, 1.2, 2.4))
        self.assertGreater(self.clip("one")["start"], self.clip(head)["end"])

    def test_http_endpoints_for_cut_exact_insert_and_replace(self):
        self.studio.engine = mock.Mock()
        with mock.patch("dsb.studio.Studio", return_value=self.studio), mock.patch("dsb.installer.Installer"):
            import importlib
            import dsb.server as srv
            importlib.reload(srv)

        with mock.patch.object(srv, "studio", self.studio):
            cut_res = srv.cut_clip("two", srv.CutClip(at=5.0, mode="split"))
            second_id = cut_res["selected"]
            self.assertEqual(len(self.studio.clips), 4)

            srv.update_clip(second_id, {"phraseId": "b", "speedMode": "auto"})
            self.assertEqual(self.clip(second_id)["phraseId"], "b")

            ins_res = srv.add_clip(srv.AddClip(phraseId="b", start=10.2, x=-1.5, z=2.5, exact=True))
            added = ins_res["added"]
            self.assertEqual((self.clip(added)["start"], self.clip(added)["x"], self.clip(added)["z"]), (10.2, -1.5, 2.5))


class CopyAndPaste(unittest.TestCase):
    """Clips at 0-3, 3.75-6.75 and 15-18 on a 30 s song with 1.875 s bars."""

    def setUp(self):
        self.studio = studio_with_two_clips(self)

    def clip(self, name):
        return next(c for c in self.studio.clips if c["id"] == name)

    def test_a_copy_keeps_what_was_copied_and_starts_unlocked(self):
        self.studio.update_clip("two", {"pins": [{"source": 1.0, "at": 1.2}], "facing": 90.0, "match": 40, "trimEnd": 3.2})
        self.studio.update_clip("two", {"locked": True})
        source = copy.deepcopy(self.clip("two"))
        added = self.clip(self.studio.paste_clip(source, 9.0)["added"])
        self.assertEqual((added["start"], added["end"]), (9.0, 12.0))
        self.assertNotEqual(added["id"], "two")
        self.assertFalse(added["locked"])
        for key in ("phraseId", "x", "z", "facing", "match", "trimEnd", "pins"):
            self.assertEqual(added[key], source[key], key)
        self.assertEqual(self.clip("two"), source)  # the copied phrase is untouched

    def test_a_copy_goes_in_by_the_insert_rules_and_is_one_undo(self):
        before = {c["id"]: (c["start"], c["end"]) for c in self.studio.clips}
        added = self.clip(self.studio.paste_clip(copy.deepcopy(self.clip("one")), self.clip("one")["end"])["added"])
        self.assertEqual(added["start"], 3.75)  # no room before "two", so it takes that start
        self.assertEqual(self.clip("two")["start"], 3.75 + 3 * 1.875)  # and "two" moves back by whole bars
        self.assertEqual((self.clip("three")["start"], self.clip("three")["end"]), before["three"])
        ordered = self.studio.clips
        for a, b in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(round(b["start"] - a["end"], 3), 0.6)
        self.studio.undo()
        self.assertEqual({c["id"]: (c["start"], c["end"]) for c in self.studio.clips}, before)

    def test_a_copy_can_outlive_its_phrase_and_reach_another_dancer(self):
        source = copy.deepcopy(self.clip("three"))
        self.studio.delete_clip("three")
        self.studio.add_dancer()
        added = self.studio.paste_clip(source, 20.0)["added"]
        other = next(d for d in self.studio.dancers if d["id"] == "d2")
        self.assertEqual([c["id"] for c in other["clips"]], [added])
        self.assertEqual((other["clips"][0]["x"], other["clips"][0]["z"]), (-1.0, 2.0))
        self.assertNotIn(added, [c["id"] for c in self.studio.dancers[0]["clips"]])

    def test_pins_that_no_longer_fit_are_dropped_and_bad_input_is_refused(self):
        source = {**copy.deepcopy(self.clip("one")), "pins": [{"source": 1.0, "at": 1.2}, {"source": 2.0, "at": 9.0}]}
        added = self.clip(self.studio.paste_clip(source, 22.0)["added"])
        self.assertEqual(added["pins"], [{"source": 1.0, "at": 1.2}])
        count = len(self.studio.undo_stack)
        for bad in ({}, {**source, "phraseId": "gone"}, {**source, "end": source["start"] + 0.1}, {**source, "x": "left"}, {**source, "match": 300}):
            with self.assertRaises(StudioError):
                self.studio.paste_clip(bad, 9.0)
        with self.assertRaises(StudioError):
            self.studio.paste_clip(source, 29.0)  # it would run past the end of the song
        self.assertEqual(len(self.studio.undo_stack), count)


class RestPose(unittest.TestCase):
    """A BVH written on a T-pose or A-pose rest has to hold the same motion, and rest in that pose."""

    def setUp(self):
        from dsb.retarget import rebased, rest_pose
        self.rebased, self.rest_pose = rebased, rest_pose
        self.skeleton = soma_skeleton()
        pose = tpose(self.skeleton)
        for joint, axis, degrees in (("Hips", [0, 1, 0], 40.0), ("Chest", [1, 0, 0], 20.0), ("LeftArm", [0, 0, 1], -70.0), ("RightForeArm", [0, 0, 1], 80.0), ("LeftLeg", [1, 0, 0], -45.0)):
            pose = turned(self.skeleton, pose, joint, axis_quat(axis, degrees))
        moved = pose.copy()
        moved[0, 0, :3] += [0.4, -0.1, 0.7]
        self.poses = np.concatenate([tpose(self.skeleton), pose, moved], axis=0)

    def written(self, kind, degrees=45.0):
        from dsb.motion import read_bvh, write_bvh
        skeleton, poses = self.rebased(self.poses, self.skeleton, self.rest_pose(self.skeleton, kind, degrees))
        with tempfile.TemporaryDirectory() as folder:
            write_bvh(Path(folder) / "a.bvh", skeleton, poses, 30.0)
            return read_bvh(Path(folder) / "a.bvh")

    def test_every_joint_stays_where_it_was(self):
        want = world_transforms(self.poses, self.skeleton.parents)[0]
        for kind in ("t", "a"):
            back = self.written(kind)
            self.assertEqual(back.skeleton.names, self.skeleton.names)
            np.testing.assert_allclose(world_transforms(back.poses, back.skeleton.parents)[0], want, atol=2e-5, err_msg=kind)

    def test_the_file_rests_in_the_pose_asked_for(self):
        for kind, degrees, drop in (("t", 45.0, 0.0), ("a", 45.0, 45.0), ("a", 30.0, 30.0)):
            skeleton = self.written(kind, degrees).skeleton
            zero = np.zeros((1, len(skeleton.names), 7))
            zero[..., 6] = 1.0
            zero[0, :, :3] = skeleton.offsets / 100.0
            places = world_transforms(zero, skeleton.parents)[0][0]
            for side, sign in (("Left", 1.0), ("Right", -1.0)):
                arm = places[skeleton.index(side + "Hand")] - places[skeleton.index(side + "Arm")]
                arm /= np.linalg.norm(arm)
                np.testing.assert_allclose(arm, [sign * np.cos(np.radians(drop)), -np.sin(np.radians(drop)), 0.0], atol=2e-3, err_msg=f"{kind} {degrees} {side}")
            leg = places[skeleton.index("LeftFoot")] - places[skeleton.index("LeftLeg")]
            self.assertLess(leg[1] / np.linalg.norm(leg), -0.99)
            self.assertAlmostEqual(float(places[:, 1].min()), 0.0, places=4)  # standing on the floor

    def test_the_tpose_frame_is_all_zero_rotation_on_a_tpose_rest(self):
        back = self.written("t")
        np.testing.assert_allclose(np.abs(back.poses[0, :, 6]), 1.0, atol=1e-6)
        self.assertLess(float(np.abs(self.written("a").poses[0, self.skeleton.index("LeftArm"), 6])), 0.95)  # on an A-pose rest the T-pose lifts the arms


if __name__ == "__main__":
    unittest.main()
