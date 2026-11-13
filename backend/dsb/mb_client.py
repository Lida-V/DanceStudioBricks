"""Client for backend/mb_engine.py. Starts the resident engine on first use and talks to it over local TCP."""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .components import ROOT, python_of, usable

ENGINE = ROOT / "backend" / "mb_engine.py"
SESSION = Path(os.environ.get("DSB_MB_SESSION", ROOT / "runtime" / "mb-session.json"))
POSES_PER_TOKEN = 4
MIN_TOKENS, MAX_TOKENS = 6, 16


class EngineError(RuntimeError):
    pass


def frames_from_tokens(tokens: int) -> int:
    """Generated frames for a stopping target, measured from the engine's output."""
    return POSES_PER_TOKEN * tokens - 6


def tokens_for_frames(frames: float) -> int:
    return int(np.clip(round((frames + 6) / POSES_PER_TOKEN), MIN_TOKENS, MAX_TOKENS))


class Engine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._stream = None
        self._socket: socket.socket | None = None
        self.joints: list[str] = []

    def start(self, timeout: float = 240.0) -> None:
        with self._lock:
            if self._stream is not None:
                return
            if not usable("motionbricks"):
                raise EngineError("MotionBricks の環境が見つかりません")
            python = python_of("motionbricks")
            SESSION.parent.mkdir(parents=True, exist_ok=True)
            SESSION.unlink(missing_ok=True)
            log = (SESSION.parent / "mb-engine.log").open("wb")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(
                [str(python), str(ENGINE), "--session", str(SESSION), "--parent-pid", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags, cwd=str(python.parents[2]))
            log.close()
            deadline = time.monotonic() + timeout
            while not SESSION.is_file():
                if self._process.poll() is not None:
                    raise EngineError("MotionBricks の起動に失敗しました。runtime/mb-engine.log を確認してください")
                if time.monotonic() > deadline:
                    self._process.kill()
                    raise EngineError("MotionBricks の起動がタイムアウトしました")
                time.sleep(0.2)
            info = json.loads(SESSION.read_text(encoding="utf-8"))
            self._socket = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=60)
            self._socket.settimeout(60)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._stream = self._socket.makefile("rwb")
            hello = self._call({"op": "hello", "token": info["token"]})
            self.joints = list(hello["joints"])

    def abort_start(self) -> None:
        """Ends a start that is still loading. Does not take the lock, which that start holds."""
        process = self._process
        if self._stream is None and process is not None and process.poll() is None:
            process.kill()

    def _call(self, message: dict) -> dict:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        self._stream.write(struct.pack("<I", len(body)) + body)
        self._stream.flush()
        head = self._stream.read(4)
        if len(head) < 4:
            raise EngineError("MotionBricks との接続が切れました")
        reply = json.loads(self._stream.read(struct.unpack("<I", head)[0]).decode("utf-8"))
        if not reply.get("ok"):
            raise EngineError(str(reply.get("error", "generation failed")))
        return reply

    def generate(self, context: np.ndarray, target: np.ndarray, tokens: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        """context [4, J, 7] and target [1..4, J, 7] in the engine's joint order. Returns (poses, contacts, ms)."""
        self.start()
        request = {"op": "generate", "context": _encode(context), "target": _encode(target),
                   "tokens": [int(tokens)] if tokens else None}
        with self._lock:
            reply = self._call(request)
        joints = len(self.joints)
        poses = np.frombuffer(base64.b64decode(reply["poses"]), dtype="<f4").reshape(-1, joints, 7).astype(np.float64)
        contacts = np.frombuffer(base64.b64decode(reply["contacts"]), dtype="<f4").reshape(poses.shape[0], -1).copy()
        return poses, contacts, float(reply["generateMs"])

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._call({"op": "shutdown"})
                except (OSError, EngineError):
                    pass
                self._stream.close()
                self._socket.close()
                self._stream = None
            if self._process is not None:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._process = None


def _encode(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")
