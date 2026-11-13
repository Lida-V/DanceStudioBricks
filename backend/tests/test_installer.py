import hashlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsb import components, installer as module  # noqa: E402
from dsb.installer import MARKER, Cancelled, InstallError, Installer, read_pointer  # noqa: E402
from dsb.mb_client import Engine, EngineError  # noqa: E402

CONTENT = b"weights" * 1000
OID = hashlib.sha256(CONTENT).hexdigest()
POINTER = f"version https://git-lfs.github.com/spec/v1\noid sha256:{OID}\nsize {len(CONTENT)}\n".encode()


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int):
        super().__init__(data)
        self.status = status


class Fake(Installer):
    """Replaces the network and the external commands."""

    def __init__(self, **options):
        super().__init__(**options)
        self.commands: list[list[str]] = []
        self.requests: list[tuple[str, int]] = []
        self.body = CONTENT

    def _gpu_report(self):
        return "NVIDIA GeForce RTX 4090, 610.88\n"

    def _open(self, url, have):
        self.requests.append((url, have))
        return Response(self.body[have:], 206) if have else Response(self.body, 200)

    def _run(self, command, cwd=None, env=None, check=True):
        self._stop_if_cancelled()
        self.commands.append(command)
        if "ls-files" in command:
            return 0, "model.onnx\nsrc.py"
        if "checkout" in command:
            folder = Path(command[2])
            (folder / "model.onnx").write_bytes(POINTER)
            (folder / "src.py").write_text("code")
        if command[1:2] == ["sync"]:
            python = cwd / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
        return 0, ""


def wait_until_finished(installer):
    deadline = time.monotonic() + 10
    while installer.status()["job"]["state"] == "running":
        if time.monotonic() > deadline:
            raise AssertionError("the job did not end")
        time.sleep(0.01)
    return installer.status()["job"]


class Base(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for target, value in ((components, "CONFIG"), (module, "LOG")):
            patcher = mock.patch.object(target, value, self.root / f"{value}.txt")
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (mock.patch.object(components, "DEFAULT_ROOT", self.root / "none"), mock.patch.object(module, "find_tool", lambda name: name), mock.patch.dict("os.environ", {"DSB_MB_PYTHON": ""})):
            patcher.start()
            self.addCleanup(patcher.stop)


class Check(Base):
    def verdict(self, root, ids=("motionbricks",)):
        return Fake().check(str(root), list(ids))

    def failed(self, verdict):
        return [item["id"] for item in verdict["items"] if not item["ok"]]

    def test_an_empty_place_passes(self):
        self.assertEqual(self.failed(self.verdict(self.root)), [])

    def test_only_plain_short_paths_are_accepted(self):
        self.assertEqual(self.failed(self.verdict("D:\\アプリ\\x")), ["root"])
        self.assertEqual(self.failed(self.verdict("relative\\x")), ["root"])
        self.assertEqual(self.failed(self.verdict("C:\\" + "a" * 130)), ["root"])

    def test_someone_elses_files_are_not_overwritten(self):
        (self.root / "MotionBricks").mkdir()
        (self.root / "MotionBricks" / "mine.txt").touch()
        self.assertEqual(self.failed(self.verdict(self.root)), ["folder-motionbricks"])

    def test_a_missing_tool_is_reported(self):
        with mock.patch.object(module, "find_tool", lambda name: None if name == "uv" else name):
            self.assertIn("uv", self.failed(self.verdict(self.root)))

    def test_old_drivers_and_missing_gpus_are_refused(self):
        with mock.patch.object(Fake, "_gpu_report", lambda self: "GTX 1080, 472.10\n"):
            self.assertIn("gpu", self.failed(self.verdict(self.root)))
        with mock.patch.object(Fake, "_gpu_report", side_effect=OSError):
            self.assertIn("gpu", self.failed(self.verdict(self.root)))

    def test_too_little_disk_is_refused(self):
        with mock.patch.object(module.shutil, "disk_usage", return_value=mock.Mock(free=1e9)):
            self.assertIn("disk", self.failed(self.verdict(self.root)))

    def test_the_gpu_is_only_needed_by_motionbricks(self):
        self.assertNotIn("gpu", [item["id"] for item in self.verdict(self.root, ("blender",))["items"]])


class Install(Base):
    def test_nothing_starts_without_agreement(self):
        with self.assertRaises(InstallError):
            Fake().start(str(self.root), ["motionbricks"], False)

    def test_full_run_places_verified_files_and_registers(self):
        warmed, ready = [], []
        installer = Fake(warm=lambda: warmed.append(1), on_ready=lambda: ready.append(1))
        installer.start(str(self.root), ["motionbricks"], True)
        job = wait_until_finished(installer)
        self.assertEqual(job["state"], "done", job.get("error"))
        folder = self.root / "MotionBricks"
        self.assertEqual((folder / "model.onnx").read_bytes(), CONTENT)
        self.assertFalse((folder / MARKER).exists())
        self.assertFalse((folder / ".dsb-download").exists())
        self.assertEqual(components.python_of("motionbricks"), folder / ".venv/Scripts/python.exe")
        self.assertTrue(components.usable("motionbricks"))
        self.assertEqual((warmed, ready), ([1], [1]))
        fetch = next(c for c in installer.commands if "fetch" in c)
        self.assertIn(module.MB_COMMIT, fetch)
        self.assertIn(module.MB_COMMIT, installer.requests[0][0])

    def test_a_corrupt_download_never_reaches_the_folder(self):
        installer = Fake()
        installer.body = b"x" * len(CONTENT)
        installer.start(str(self.root), ["motionbricks"], True)
        job = wait_until_finished(installer)
        self.assertEqual(job["state"], "failed")
        folder = self.root / "MotionBricks"
        self.assertEqual((folder / "model.onnx").read_bytes(), POINTER)
        self.assertTrue((folder / MARKER).exists())
        self.assertFalse(components.usable("motionbricks"))

    def test_an_interrupted_download_continues(self):
        installer = Fake()
        target = self.root / "file.bin"
        (self.root / "file.bin.part").write_bytes(CONTENT[:100])
        installer._download("http://x/file", target, len(CONTENT), OID)
        self.assertEqual(installer.requests, [("http://x/file", 100)])
        self.assertEqual(target.read_bytes(), CONTENT)

    def test_a_server_that_ignores_resume_starts_over(self):
        installer = Fake()
        installer._open = lambda url, have: Response(CONTENT, 200)
        target = self.root / "file.bin"
        (self.root / "file.bin.part").write_bytes(CONTENT[:100])
        installer._download("http://x/file", target, len(CONTENT), OID)
        self.assertEqual(target.read_bytes(), CONTENT)

    def test_a_finished_file_is_not_fetched_again(self):
        installer = Fake()
        target = self.root / "file.bin"
        target.write_bytes(CONTENT)
        installer._download("http://x/file", target, len(CONTENT), OID)
        self.assertEqual(installer.requests, [])

    def test_network_errors_are_retried(self):
        installer = Fake()
        calls = []

        def flaky(url, have):
            calls.append(have)
            if len(calls) == 1:
                raise OSError("reset")
            return Response(CONTENT, 200)

        installer._open = flaky
        with mock.patch.object(installer._cancel, "wait"):
            installer._download("http://x/file", self.root / "file.bin", len(CONTENT), OID)
        self.assertEqual(len(calls), 2)

    def test_cancel_stops_a_download_and_keeps_what_was_fetched(self):
        installer = Fake()
        installer._cancel.set()
        with self.assertRaises(Cancelled):
            installer._download("http://x/file", self.root / "file.bin", len(CONTENT), OID)
        self.assertFalse((self.root / "file.bin").exists())

    def test_an_existing_environment_is_only_registered(self):
        python = self.root / "MotionBricks/.venv/Scripts/python.exe"
        python.parent.mkdir(parents=True)
        python.touch()
        installer = Fake()
        installer.start(str(self.root), ["motionbricks"], True)
        self.assertEqual(wait_until_finished(installer)["state"], "done")
        self.assertEqual(installer.commands, [])
        self.assertEqual(components.python_of("motionbricks"), python)

    def test_a_second_run_continues_after_a_failure(self):
        installer = Fake()
        installer.body = b"x" * len(CONTENT)
        installer.start(str(self.root), ["motionbricks"], True)
        self.assertEqual(wait_until_finished(installer)["state"], "failed")
        installer.body = CONTENT
        installer.start(str(self.root), ["motionbricks"], True)
        self.assertEqual(wait_until_finished(installer)["state"], "done")
        self.assertEqual((self.root / "MotionBricks/model.onnx").read_bytes(), CONTENT)

    def test_a_failing_component_ends_the_job_and_says_why(self):
        installer = Fake()
        with mock.patch.object(installer, "_install_blender", side_effect=RuntimeError("boom")):
            installer.start(str(self.root), ["blender"], True)
            job = wait_until_finished(installer)
        self.assertEqual((job["state"], job["error"], job["components"]["blender"]), ("failed", "boom", "failed"))

    def test_unknown_components_are_refused(self):
        with self.assertRaises(InstallError):
            Fake().start(str(self.root), ["nothing"], True)


class Pointer(Base):
    def test_reads_lfs_pointers_only(self):
        path = self.root / "a"
        path.write_bytes(POINTER)
        self.assertEqual(read_pointer(path), (OID, len(CONTENT)))
        path.write_bytes(CONTENT)
        self.assertIsNone(read_pointer(path))


class Lookup(Base):
    def python(self, folder):
        path = self.root / folder / ".venv/Scripts/python.exe"
        path.parent.mkdir(parents=True)
        path.touch()
        return path

    def test_environment_variable_wins_over_the_registered_path(self):
        first, second = self.python("a"), self.python("b")
        components.register("motionbricks", first)
        self.assertEqual(components.python_of("motionbricks"), first)
        with mock.patch.dict("os.environ", {"DSB_MB_PYTHON": str(second)}):
            self.assertEqual(components.python_of("motionbricks"), second)
        self.assertEqual(json.loads(components.CONFIG.read_text(encoding="utf-8")), {"motionbricks": str(first)})

    def test_an_unfinished_install_is_not_used(self):
        python = self.python("a")
        components.register("motionbricks", python)
        (self.root / "a" / MARKER).touch()
        self.assertFalse(components.usable("motionbricks"))
        with self.assertRaises(EngineError):
            Engine().start()

    def test_the_model_is_part_of_embeddinggemma(self):
        components.register("embeddinggemma2", self.python("a"))
        self.assertFalse(components.usable("embeddinggemma2"))
        model = self.root / "a/models/embeddinggemma-2/model.safetensors"
        model.parent.mkdir(parents=True)
        model.touch()
        self.assertTrue(components.usable("embeddinggemma2"))

    def test_blender_lives_directly_in_its_folder(self):
        python = self.root / "bpy/Scripts/python.exe"
        python.parent.mkdir(parents=True)
        python.touch()
        components.register("blender", python)
        self.assertEqual(components.folder_of("blender"), self.root / "bpy")
        self.assertTrue(components.usable("blender"))


if __name__ == "__main__":
    unittest.main()
