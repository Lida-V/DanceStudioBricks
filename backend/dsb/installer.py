"""Downloads and sets up the external environments on the user's own PC, after the user has agreed.

Every source is pinned and every large file is verified. An unfinished install keeps a marker in its folder: it is not
used by the app, and starting again continues where it stopped.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from .components import BEAT_CHECKPOINT, COMPONENTS, DEFAULT_ROOT, MARKER, ROOT, Component, python_of, register, usable

MB_REPO = "NVlabs/MotionBricks"
MB_COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
EG2_REPO = "google/embeddinggemma-2"
EG2_COMMIT = "f9972ac22e696a105c482202646979e33e4b5a1e"
# (path, size, sha256 or "" for the small files that have no published hash)
EG2_FILES = (
    (".gitattributes", 1301, ""), ("1_Pooling/config.json", 90, ""), ("2_Normalize/config.json", 97, ""), ("README.md", 22234, ""),
    ("chat_template.jinja", 1016, ""), ("config.json", 4455, ""), ("config_sentence_transformers.json", 1565, ""),
    ("model.safetensors", 1488915288, "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"),
    ("modules.json", 413, ""), ("preprocessor_config.json", 511, ""), ("processor_config.json", 1788, ""), ("sentence_bert_config.json", 747, ""),
    ("tokenizer.json", 32170510, "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
    ("tokenizer.model", 4689013, "e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03"), ("tokenizer_config.json", 1599, ""),
)
BEAT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BEAT_SIZE = 81058141
BEAT_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
TORCH_INDEX = "https://download.pytorch.org/whl/cu134"
BLENDER_PACKAGES = ("bpy==5.0.1", "numpy==1.26.4")
POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
MIN_DRIVER = 525  # CUDA 12
MAX_ROOT_LENGTH = 120  # deep files in these environments would otherwise pass the Windows path limit
SPARE_GIGABYTES = 3.0
PREPARE_TIMEOUT = 4 * 3600.0  # the first load builds a GPU plan, which is slow when another app holds the GPU
ATTEMPTS = 8
CHUNK = 1 << 20
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")  # uv colors its output even into a pipe
LOG =ROOT / "runtime" / "install.log"
HELP = {
    "uv": "winget install --id astral-sh.uv -e で入れられます",
    "git": "winget install --id Git.Git -e で入れられます",
    "gpu": "NVIDIA のドライバを更新してください",
}


class InstallError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def find_tool(name: str) -> str | None:
    """Also looks where winget and uv put themselves, which a running app's PATH does not know about yet."""
    found = shutil.which(name)
    if found:
        return found
    home, local = Path.home(), Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    for place in (home / ".local" / "bin", local / "Microsoft" / "WinGet" / "Links", Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd"):
        if (place / f"{name}.exe").is_file():
            return str(place / f"{name}.exe")
    return None


def read_pointer(path: Path) -> tuple[str, int] | None:
    """(sha256, size) when this file is still a Git LFS pointer."""
    with path.open("rb") as handle:
        data = handle.read(512)
    if not data.startswith(POINTER_HEADER):
        return None
    fields = dict(line.split(" ", 1) for line in data.decode().splitlines() if " " in line)
    return fields["oid"].removeprefix("sha256:"), int(fields["size"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class Installer:
    def __init__(self, warm: Callable[[], None] | None = None, abort_warm: Callable[[], None] | None = None, on_ready: Callable[[], None] | None = None) -> None:
        self._warm, self._abort_warm, self._on_ready = warm, abort_warm, on_ready
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._process: subprocess.Popen | None = None
        self._job: dict = {"state": "idle"}

    # ---- what the screen asks

    def status(self) -> dict:
        with self._lock:
            job = dict(self._job)
        return {"defaultRoot": str(DEFAULT_ROOT), "job": job, "components": [
            {"id": c.id, "label": c.label, "purpose": c.purpose, "required": c.required, "gigabytes": c.gigabytes,
             "licenses": [{"name": name, "url": url} for name, url in c.licenses], "installed": usable(c.id)} for c in COMPONENTS.values()]}

    def check(self, root: str, ids: list[str]) -> dict:
        chosen = self._chosen(ids)
        needs = {need for item in chosen for need in item.needs}
        items = []
        if "gpu" in needs:
            items.append(self._check_gpu())
        for tool in sorted(needs & {"git", "uv"}):
            found = find_tool(tool)
            items.append({"id": tool, "label": {"git": "Git", "uv": "uv"}[tool], "ok": bool(found), "detail": found or HELP[tool]})
        items += self._check_root(root, chosen)
        return {"ok": all(item["ok"] for item in items), "items": items}

    def start(self, root: str, ids: list[str], accepted: bool) -> dict:
        if not accepted:
            raise InstallError("ライセンスへの同意が必要です")
        chosen = self._chosen(ids)
        if not chosen:
            raise InstallError("導入する環境を選んでください")
        verdict = self.check(root, ids)
        if not verdict["ok"]:
            raise InstallError("導入の条件を満たしていません: " + "、".join(item["label"] for item in verdict["items"] if not item["ok"]))
        with self._lock:
            if self._job.get("state") == "running":
                raise InstallError("すでに導入中です")
            self._cancel.clear()
            self._job = {"state": "running", "root": root, "components": {item.id: "waiting" for item in chosen}, "current": None, "step": "", "detail": "", "done": 0, "total": 0, "error": None}
        threading.Thread(target=self._execute, args=(Path(root), chosen), name="installer", daemon=True).start()
        return self.status()

    def cancel(self) -> None:
        self._cancel.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.kill()
        if self._abort_warm is not None and self._job.get("step") == "初回の準備":
            self._abort_warm()

    # ---- checks

    @staticmethod
    def _chosen(ids: list[str]) -> list[Component]:
        unknown = [i for i in ids if i not in COMPONENTS]
        if unknown:
            raise InstallError(f"不明な環境です: {unknown[0]}")
        return [item for key, item in COMPONENTS.items() if key in ids]

    @staticmethod
    def _gpu_report() -> str:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=True, creationflags=flags).stdout

    def _check_gpu(self) -> dict:
        label = "NVIDIA GPU と CUDA 12 対応ドライバ"
        try:
            name, driver = [part.strip() for part in self._gpu_report().splitlines()[0].split(",")[:2]]
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return {"id": "gpu", "label": label, "ok": False, "detail": "NVIDIA GPU が見つかりません。MotionBricks には必要です"}
        if int(driver.split(".")[0]) < MIN_DRIVER:
            return {"id": "gpu", "label": label, "ok": False, "detail": f"{name}、ドライバ {driver}。{HELP['gpu']}"}
        return {"id": "gpu", "label": label, "ok": True, "detail": f"{name}、ドライバ {driver}"}

    @staticmethod
    def _check_root(root: str, chosen: list[Component]) -> list[dict]:
        label = "インストール先"
        path = Path(root)
        if not root.isascii() or not path.is_absolute():
            return [{"id": "root", "label": label, "ok": False, "detail": "英数字だけのフルパスを指定してください（例: C:\\AI）"}]
        if len(root) > MAX_ROOT_LENGTH:
            return [{"id": "root", "label": label, "ok": False, "detail": "パスが長すぎます。短いフォルダを指定してください"}]
        items = []
        need = 0.0
        for item in chosen:
            if item.folder is None:
                need += item.gigabytes
                continue
            folder = path / item.folder
            occupied = folder.is_dir() and any(folder.iterdir())
            resumable = (folder / MARKER).exists()
            existing = (folder / item.python).is_file()
            if occupied and not resumable and not existing:
                items.append({"id": f"folder-{item.id}", "label": f"{item.label} のフォルダ", "ok": False, "detail": f"{folder} にすでに別のファイルがあります"})
            elif not existing or resumable:
                need += item.gigabytes
        anchor = next(p for p in (path, *path.parents) if p.exists()) if any(p.exists() for p in (path, *path.parents)) else None
        if anchor is None:
            return [{"id": "root", "label": label, "ok": False, "detail": "このドライブは使えません"}, *items]
        free = shutil.disk_usage(anchor).free / 1e9
        return [{"id": "disk", "label": "空き容量", "ok": free >= need + SPARE_GIGABYTES, "detail": f"必要 約 {need + SPARE_GIGABYTES:.0f} GB、空き {free:.0f} GB"}, *items]

    # ---- running

    def _update(self, **values) -> None:
        with self._lock:
            self._job.update(values)

    def _mark(self, component: str, state: str) -> None:
        with self._lock:
            self._job["components"] = {**self._job["components"], component: state}

    def _step(self, text: str) -> None:
        self._log(f"== {text}")
        self._update(step=text, detail="", done=0, total=0)

    @staticmethod
    def _log(line: str) -> None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as output:
            output.write(line + "\n")

    def _execute(self, root: Path, chosen: list[Component]) -> None:
        current = None
        try:
            for item in chosen:
                current = item.id
                self._update(current=current)
                self._mark(current, "running")
                getattr(self, f"_install_{current}")(root, item)
                self._mark(current, "done")
            importlib.invalidate_caches()
            if self._on_ready is not None:
                self._on_ready()
            self._update(state="done", step="", detail="")
        except Cancelled:
            if current:
                self._mark(current, "waiting")
            self._update(state="cancelled", step="", detail="")
        except Exception as error:  # the job must always end, and say why
            self._log(f"!! {error!r}")
            if current:
                self._mark(current, "failed")
            self._update(state="failed", error=str(error))
        finally:
            self._process = None

    def _stop_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise Cancelled

    def _run(self, command: list[str], cwd: Path | None = None, env: dict | None = None, check: bool = True) -> tuple[int, str]:
        self._stop_if_cancelled()
        self._log("$ " + " ".join(command))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        merged = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"} | (env or {})
        process = subprocess.Popen(command, cwd=str(cwd) if cwd else None, env=merged, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", creationflags=flags)
        self._process = process
        lines: list[str] = []
        for raw in process.stdout:
            line = ANSI.sub("", raw).strip()
            if line:
                lines.append(line)
                self._log(line)
                self._update(detail=line[:160])
        code = process.wait()
        self._process = None
        self._stop_if_cancelled()
        if code and check:
            raise InstallError(f"{Path(command[0]).stem} が失敗しました: {' / '.join(lines[-3:])[-400:]}（詳細は runtime/install.log）")
        return code, "\n".join(lines)

    def _open(self, url: str, have: int):
        headers = {"User-Agent": "DanceStudioBricks"} | ({"Range": f"bytes={have}-"} if have else {})
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)

    def _download(self, url: str, target: Path, size: int, expected: str, base: int = 0) -> None:
        """Places `target` only once it is complete and verified. An interrupted download continues from its .part file."""
        if target.is_file() and target.stat().st_size == size and (not expected or sha256(target) == expected):
            self._update(done=base + size)
            return
        part = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        failure: Exception | None = None
        for _ in range(ATTEMPTS):
            have = part.stat().st_size if part.exists() else 0
            if have == size:
                break
            if have > size:
                part.unlink()
                have = 0
            try:
                with self._open(url, have) as response, part.open("ab" if have and response.status == 206 else "wb") as output:
                    have = have if response.status == 206 else 0
                    while True:
                        self._stop_if_cancelled()
                        block = response.read(CHUNK)
                        if not block:
                            break
                        output.write(block)
                        have += len(block)
                        self._update(done=base + have)
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                failure = error
                self._log(f"retry {target.name}: {error!r}")
                self._cancel.wait(2.0)
                self._stop_if_cancelled()
        if not part.exists() or part.stat().st_size != size:
            raise InstallError(f"{target.name} をダウンロードできませんでした: {failure}")
        if expected and sha256(part) != expected:
            part.unlink()
            raise InstallError(f"{target.name} の内容が一致しませんでした。もう一度お試しください")
        os.replace(part, target)

    def _download_all(self, files: list[tuple[str, Path, int, str]]) -> None:
        """files: (url, target, size, sha256). Progress counts bytes across all of them."""
        self._update(total=sum(entry[2] for entry in files), done=0)
        base = 0
        for url, target, size, expected in files:
            self._update(detail=target.name)
            self._download(url, target, size, expected, base)
            base += size

    def _begin(self, root: Path, item: Component) -> Path | None:
        """The folder to build in, or None when a finished environment is already there."""
        folder = root / item.folder
        if (folder / item.python).is_file() and not (folder / MARKER).exists():
            register(item.id, folder / item.python)
            return None
        folder.mkdir(parents=True, exist_ok=True)
        (folder / MARKER).touch()
        return folder

    def _finish(self, item: Component, folder: Path, module: str) -> None:
        self._step("動作の確認")
        self._run([str(folder / item.python), "-c", f"import {module}"])
        (folder / MARKER).unlink()
        register(item.id, folder / item.python)

    def _new_venv(self, folder: Path, python: Path) -> None:
        if not python.is_file():
            self._run([find_tool("uv"), "venv", "--python", "3.11", str(python.parents[1])], cwd=folder)

    def _pip(self, python: Path, arguments: list[str]) -> None:
        self._run([find_tool("uv"), "pip", "install", "--python", str(python), "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match", *arguments])

    # ---- one method per component, named _install_<id>

    def _install_motionbricks(self, root: Path, item: Component) -> None:
        folder = self._begin(root, item)
        if folder is not None:
            self._step("MotionBricks の取得")
            self._checkout(folder)
            self._step("モデルのダウンロード")
            self._fetch_lfs(folder)
            self._step("Python 環境の作成")
            self._run([find_tool("uv"), "sync", "--locked"], cwd=folder)
            self._finish(item, folder, "motionbricks")
            shutil.rmtree(folder / ".dsb-download", ignore_errors=True)
        if self._warm is not None:
            self._step("初回の準備")
            self._update(detail="GPU 用のプランを作っています。他のアプリが GPU を使っていると長くかかります")
            self._warm()

    def _checkout(self, folder: Path) -> None:
        git, env = find_tool("git"), {"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        if (folder / ".git").is_dir() and self._run([git, "-C", str(folder), "rev-parse", "HEAD"], check=False)[1].strip().endswith(MB_COMMIT):
            return
        self._run([git, "init", str(folder)], env=env)
        self._run([git, "-C", str(folder), "remote", "add", "origin", f"https://github.com/{MB_REPO}"], env=env, check=False)
        self._run([git, "-C", str(folder), "fetch", "--depth", "1", "origin", MB_COMMIT], env=env)
        self._run([git, "-C", str(folder), "checkout", "--force", "FETCH_HEAD"], env=env)

    def _fetch_lfs(self, folder: Path) -> None:
        names = self._run([find_tool("git"), "-C", str(folder), "ls-files"])[1].splitlines()
        files = []
        for name in names:
            path = folder / name.strip('"')
            pointer = read_pointer(path) if path.is_file() else None
            if pointer:
                url = f"https://media.githubusercontent.com/media/{MB_REPO}/{MB_COMMIT}/{urllib.parse.quote(name.strip(chr(34)))}"
                files.append((url, path, pointer[1], pointer[0]))
        # Staged by hash under the folder, so a pointer file is only replaced by its verified content.
        self._update(total=sum(entry[2] for entry in files), done=0)
        base = 0
        for url, path, size, expected in files:
            staged = folder / ".dsb-download" / expected
            self._update(detail=path.name)
            self._download(url, staged, size, expected, base)
            os.replace(staged, path)
            base += size

    def _install_beat_this(self, root: Path, item: Component) -> None:
        self._step("Beat This! の導入")
        self._pip(python_of(item.id), ["-r", str(ROOT / "backend" / "requirements-beat.lock.txt")])
        self._step("重みのダウンロード")
        self._download_all([(BEAT_URL, BEAT_CHECKPOINT, BEAT_SIZE, BEAT_SHA256)])

    def _install_embeddinggemma2(self, root: Path, item: Component) -> None:
        folder = self._begin(root, item)
        if folder is None:
            return
        python = folder / item.python
        self._step("Python 環境の作成")
        self._new_venv(folder, python)
        self._pip(python, ["-r", str(ROOT / "backend" / "requirements-eg2.lock.txt")])
        self._step("モデルのダウンロード")
        base = f"https://huggingface.co/{EG2_REPO}/resolve/{EG2_COMMIT}"
        self._download_all([(f"{base}/{urllib.parse.quote(name)}", folder / "models" / "embeddinggemma-2" / name, size, expected) for name, size, expected in EG2_FILES])
        self._finish(item, folder, "sentence_transformers")

    def _install_blender(self, root: Path, item: Component) -> None:
        folder = self._begin(root, item)
        if folder is None:
            return
        python = folder / item.python
        self._step("Blender の Python 環境を作成")
        self._new_venv_here(folder)
        self._pip(python, list(BLENDER_PACKAGES))
        self._finish(item, folder, "bpy")

    def _new_venv_here(self, folder: Path) -> None:
        """This environment is the folder itself, which already holds the marker."""
        if not (folder / "pyvenv.cfg").is_file():
            self._run([find_tool("uv"), "venv", "--allow-existing", "--python", "3.11", str(folder)], cwd=folder)
