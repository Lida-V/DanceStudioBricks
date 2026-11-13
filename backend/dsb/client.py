"""Shared local HTTP client for Dance Studio Bricks MCP and CLI helpers.

Talks to the running Dance Studio Bricks backend (`127.0.0.1:8765` by default) so MCP edits and the
Electron/Web UI stay synchronized on the same live Studio instance. Automatically boots the local backend
in the background when it is not yet running (unless `DSB_MCP_AUTOBOOT=0`).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
RUNTIME_DIR = ROOT / "runtime"

Requester = Callable[[str, str, Any | None, float], Any]
_custom_requester: Requester | None = None


def server_port() -> int:
    return int(os.environ.get("DSB_PORT", "8765"))


def base_url() -> str:
    explicit = os.environ.get("DSB_URL", "").strip().rstrip("/")
    return explicit if explicit else f"http://127.0.0.1:{server_port()}"


def _is_alive(url: str) -> bool:
    for probe in ("/api/health", "/api/components"):
        try:
            with urllib.request.urlopen(f"{url}{probe}", timeout=1.5) as resp:
                if resp.status != 200:
                    continue
                payload = json.load(resp)
                if payload.get("app") == "Dance Studio Bricks" or "components" in payload:
                    return True
        except Exception:
            continue
    return False


def ensure_server() -> None:
    """Ensure the local Dance Studio Bricks FastAPI server is reachable."""
    url = base_url()
    if _is_alive(url):
        return
    if os.environ.get("DSB_MCP_AUTOBOOT", "1") == "0":
        raise RuntimeError(f"Dance Studio Bricks サーバー ({url}) が起動していません")

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or server_port()
    if host not in ("127.0.0.1", "localhost"):
        raise RuntimeError(f"リモートサーバー ({url}) に接続できません")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME_DIR / "backend.log"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log_path.open("a", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                "-W",
                "ignore",
                "-m",
                "uvicorn",
                "dsb.server:app",
                "--app-dir",
                str(BACKEND_DIR),
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            creationflags=flags,
        )

    for _ in range(160):
        if _is_alive(url):
            return
        if proc.poll() is not None:
            raise RuntimeError(f"バックエンドが起動できませんでした。{log_path} を確認してください")
        time.sleep(0.25)
    raise RuntimeError(f"バックエンドの起動がタイムアウトしました ({log_path})")


def _extract_error(raw_bytes: bytes, status: int) -> str:
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    if text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if "error" in parsed:
                    return str(parsed["error"])
                if "detail" in parsed:
                    return str(parsed["detail"])
        except ValueError:
            pass
        return text
    return f"HTTP {status}"


def _http_request(method: str, path: str, body: Any | None = None, timeout: float = 180.0) -> Any:
    ensure_server()
    url = f"{base_url()}{path}"
    headers: dict[str, str] = {"Accept": "application/json"}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as error:
        raise RuntimeError(_extract_error(error.read(), error.code)) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Dance Studio Bricks サーバーへの通信に失敗しました: {error.reason}") from error


@contextmanager
def use_requester(requester: Requester):
    """Override the HTTP transport in unit tests without binding a network port."""
    global _custom_requester
    previous = _custom_requester
    _custom_requester = requester
    try:
        yield
    finally:
        _custom_requester = previous


def request(method: str, path: str, body: Any | None = None, timeout: float = 180.0) -> Any:
    active = _custom_requester or _http_request
    return active(method, path, body, timeout)
