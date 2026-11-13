"""Resident MotionBricks generator for Dance Studio Bricks. Runs in the official MotionBricks venv.

Keeps one SOMA engine loaded so a join costs one GPU call instead of a 3 s job. The app decides every target; this
process only answers "four context poses + one to four target poses -> generated motion" and holds no motion state.

    C:/AI/MotionBricks/.venv/Scripts/python.exe backend/mb_engine.py --session runtime/mb-session.json --parent-pid N

Local only: 127.0.0.1 on a free port, a per-launch token in the session file. Messages are 4-byte little-endian length
+ UTF-8 JSON. Poses travel as base64 float32 [pose, joint, 7] in BVH space (Y up, metres, XYZW): Hips as its world
transform, every other joint as its local transform, joints in the order `hello` returns.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import queue
import secrets
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(sys.prefix).resolve().parent  # the MotionBricks repository (its .venv lives in the root)
MAX_MESSAGE = 16 << 20
POSES_PER_TOKEN = 4


class Generator:
    def __init__(self, precision: str):
        import warp as wp
        from motionbricks.io.usd import load_skeleton_from_usd
        from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file
        from motionbricks.model.soma import contract
        from motionbricks.model.soma.engine import MotionBricksSOMAEngine
        from motionbricks.utils.space_conversion_utils import FacingDirectionType, SpaceConverter

        self.wp = wp
        wp.init()
        wp.set_device("cuda:0")
        self.skeleton = load_skeleton_from_usd(
            get_asset_file("human", "soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH)
        self.joints = list(self.skeleton.joint_names)
        self.model_joints = list(contract.SOMA_N1_CONTRACT.joint_names)
        # BVH space is Y up; the engine's world is Z up (the same converter the official apps and the supporter use).
        self.to_runtime = SpaceConverter(FacingDirectionType.MUJOCO).convert_transform(wp.transform_identity())
        self._rt_p = np.array(self.to_runtime.p, dtype=np.float64)
        self._rt_q = Rotation.from_quat(np.array(self.to_runtime.q, dtype=np.float64))
        onnx = contract.SOMA_N1_FP16_ONNX_PATH if precision == "fp16" else contract.SOMA_N1_FP32_ONNX_PATH
        kwargs = {"model_to_runtime": self.to_runtime, "cache_dir": REPO / ".trt-cache"}
        if precision == "fp16":
            from motionbricks.inference.builder import TensorRTBuildOptions
            kwargs["build_options"] = TensorRTBuildOptions(enable_fp16=True)
        started = time.perf_counter()
        self.engine = MotionBricksSOMAEngine(onnx, **kwargs)
        self.load_seconds = time.perf_counter() - started
        with wp.ScopedStream(self.engine.stream):
            self.context = self.engine.create_input(self.skeleton)
            self.target = self.engine.create_input(self.skeleton)
            self.output = self.engine.create_output(self.skeleton)
        self.count = 0

    def close(self):
        self.engine.close()

    def _to_world(self, poses: np.ndarray) -> np.ndarray:
        out = poses.astype(np.float64).copy()
        out[:, 0, :3] = self._rt_q.apply(poses[:, 0, :3]) + self._rt_p
        out[:, 0, 3:7] = (self._rt_q * Rotation.from_quat(poses[:, 0, 3:7])).as_quat()
        return out

    def _to_bvh(self, poses: np.ndarray) -> np.ndarray:
        inverse = self._rt_q.inv()
        out = poses.astype(np.float64).copy()
        out[:, 0, :3] = inverse.apply(poses[:, 0, :3] - self._rt_p)
        out[:, 0, 3:7] = (inverse * Rotation.from_quat(poses[:, 0, 3:7])).as_quat()
        return out

    def generate(self, context: np.ndarray, target: np.ndarray, tokens) -> tuple[np.ndarray, np.ndarray, float]:
        wp = self.wp
        from motionbricks.model.config import MotionBricksInferenceConfig
        config = MotionBricksInferenceConfig(allowed_token_counts=tuple(int(t) for t in tokens)) if tokens else None
        ctx = wp.array(np.ascontiguousarray(self._to_world(context), dtype=np.float32), dtype=wp.transform)
        tgt = wp.array(np.ascontiguousarray(self._to_world(target), dtype=np.float32), dtype=wp.transform)
        started = time.perf_counter()
        with wp.ScopedStream(self.engine.stream):
            self.context.update_from_poses(ctx, stream=self.engine.stream)
            self.target.update_from_poses(tgt, stream=self.engine.stream)
            self.engine.generate_animation(self.context, self.target, self.output, config=config)
            self.engine.synchronize()
            frames = int(self.output.num_frames)
            poses = self.output.poses[:frames].numpy()
            contacts = self.output.foot_contacts[:frames].numpy()
        elapsed = time.perf_counter() - started
        self.count += 1
        return self._to_bvh(poses).astype(np.float32), contacts.astype(np.float32), elapsed

    def reference(self) -> np.ndarray:
        """The skeleton's reference pose in the wire layout (model space is BVH space), one pose."""
        pose = self.skeleton.reference_local_transforms.numpy().astype(np.float64).reshape(1, -1, 7)
        return pose


# ----------------------------------------------------------------------------------------------
# Wire

def decode_poses(text: str, joints: int, lo: int, hi: int) -> np.ndarray:
    raw = base64.b64decode(text, validate=True)
    if len(raw) % (joints * 7 * 4):
        raise ValueError("pose data is not [pose, joint, 7] float32")
    poses = np.frombuffer(raw, dtype="<f4").reshape(-1, joints, 7).astype(np.float64)
    if not lo <= len(poses) <= hi:
        raise ValueError(f"expected {lo} to {hi} poses, got {len(poses)}")
    if not np.isfinite(poses).all():
        raise ValueError("non-finite pose values")
    norms = np.linalg.norm(poses[..., 3:7], axis=-1)
    if np.abs(norms - 1).max() > 1e-2:
        raise ValueError("quaternions are not unit length")
    poses[..., 3:7] /= norms[..., None]
    return poses


def encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def recv(stream) -> dict | None:
    head = stream.read(4)
    if len(head) < 4:
        return None
    size = struct.unpack("<I", head)[0]
    if size > MAX_MESSAGE:
        raise ValueError("message too large")
    body = stream.read(size)
    if len(body) < size:
        return None
    return json.loads(body.decode("utf-8"))


def send(stream, message: dict) -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(struct.pack("<I", len(body)) + body)
    stream.flush()


def handle(gen: Generator, request: dict) -> dict:
    op = request.get("op")
    rid = request.get("id")
    if op == "ping":
        return {"id": rid, "ok": True, "count": gen.count}
    if op != "generate":
        return {"id": rid, "ok": False, "error": "unknown op"}
    joints = len(gen.joints)
    context = decode_poses(request["context"], joints, POSES_PER_TOKEN, POSES_PER_TOKEN)
    target = decode_poses(request["target"], joints, 1, POSES_PER_TOKEN)
    use = int(request.get("posesToUse", len(target)))
    if not 1 <= use <= len(target):
        raise ValueError("posesToUse out of range")
    tokens = request.get("tokens") or None
    if tokens is not None and not all(6 <= int(t) <= 16 for t in tokens):
        raise ValueError("tokens must be 6 to 16")
    poses, contacts, elapsed = gen.generate(context, target[:use], tokens)
    return {"id": rid, "ok": True, "frames": int(poses.shape[0]), "fps": 30.0, "poses": encode(poses),
            "contacts": encode(contacts), "generateMs": round(elapsed * 1000, 3)}


def other_helper_running(session: Path) -> bool:
    try:
        info = json.loads(session.read_text(encoding="utf-8"))
        with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=.5):
            return True
    except (OSError, ValueError, KeyError):
        return False


class Job:
    def __init__(self, request: dict):
        self.request, self.reply, self.done = request, None, threading.Event()


def client(conn: socket.socket, gen: Generator, token: str, jobs: queue.Queue, stop: threading.Event) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with conn, conn.makefile("rwb") as stream:
        try:
            hello = recv(stream)
            if not hello or hello.get("op") != "hello" or not hmac.compare_digest(str(hello.get("token", "")), token):
                send(stream, {"ok": False, "error": "hello with the session token first"})
                return
            send(stream, {"id": hello.get("id"), "ok": True, "joints": gen.joints, "modelJoints": gen.model_joints,
                          "fps": 30.0, "posesPerToken": POSES_PER_TOKEN, "pid": os.getpid()})
            while not stop.is_set():
                request = recv(stream)
                if request is None:
                    break
                if request.get("op") == "shutdown":
                    send(stream, {"id": request.get("id"), "ok": True})
                    stop.set()
                    break
                # The engine runs on the main thread (CUDA/TensorRT stay on the thread that created them).
                job = Job(request)
                jobs.put(job)
                reply = job.reply if job.done.wait(10) else {"id": request.get("id"), "ok": False, "error": "timeout"}
                send(stream, reply)
                if reply.get("ok") and "generateMs" in reply:
                    print(f"mb_engine: #{gen.count} {reply['frames']} frames in {reply['generateMs']} ms at {time.time():.3f}", flush=True)
        except (OSError, ValueError) as error:
            print(f"mb_engine: client dropped ({type(error).__name__})", flush=True)


def process_alive(pid: int) -> bool:
    """Windows: whether a process is still running (SYNCHRONIZE handle, zero wait); never signals it."""
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x102   # WAIT_TIMEOUT: not signalled, still running
    finally:
        kernel32.CloseHandle(handle)


def serve(gen: Generator, session: Path, port: int, parent: int = 0) -> None:
    token = secrets.token_hex(16)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", port))
    listener.listen(4)
    listener.settimeout(.5)
    port = listener.getsockname()[1]
    session.parent.mkdir(parents=True, exist_ok=True)
    temp = session.with_suffix(".tmp")
    temp.write_text(json.dumps({"version": 1, "port": port, "token": token, "pid": os.getpid(),
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "joints": len(gen.joints)}),
                    encoding="utf-8")
    os.replace(temp, session)
    print(f"mb_engine: listening on 127.0.0.1:{port}, engine {gen.load_seconds:.2f}s", flush=True)
    jobs, stop = queue.Queue(), threading.Event()

    def accept():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(None)
            threading.Thread(target=client, args=(conn, gen, token, jobs, stop), daemon=True).start()

    def watch():
        # Started by another program (the operator app, a smoke test): exit with it, even when it was killed.
        while not stop.wait(1.0):
            if not process_alive(parent):
                print("mb_engine: parent process ended", flush=True)
                stop.set()

    threading.Thread(target=accept, daemon=True).start()
    if parent:
        threading.Thread(target=watch, daemon=True).start()
    try:
        while not stop.is_set():
            try:
                job = jobs.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                job.reply = handle(gen, job.request)
            except (ValueError, KeyError, TypeError) as error:
                job.reply = {"id": job.request.get("id"), "ok": False, "error": f"{type(error).__name__}: {error}"}
            job.done.set()
    finally:
        stop.set()
        listener.close()
        try:
            if json.loads(session.read_text(encoding="utf-8")).get("pid") == os.getpid():
                session.unlink()
        except (OSError, ValueError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=ROOT / "runtime/mb-session.json")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit when this process ends")
    args = parser.parse_args()
    if not (REPO / "motionbricks").is_dir():
        print("mb_engine: run with the MotionBricks venv python (C:/AI/MotionBricks/.venv/Scripts/python.exe)", file=sys.stderr)
        return 2
    session = args.session.resolve()
    if other_helper_running(session):
        print("mb_engine: another helper already answers on this session file", file=sys.stderr)
        return 3
    gen = Generator(args.precision)
    try:
        # Warm up (CUDA graph, first-call setup: ~94 ms) so the player's first transition is fast.
        ref = gen.reference()
        gen.generate(np.repeat(ref, POSES_PER_TOKEN, axis=0), ref, None)
        serve(gen, session, args.port, args.parent_pid)
    except KeyboardInterrupt:
        pass
    finally:
        gen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
