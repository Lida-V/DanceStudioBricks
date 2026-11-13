"""Where the external environments live, and whether each one can be used. Standard library only."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get("DSB_COMPONENTS", ROOT / "runtime" / "components.json"))
DEFAULT_ROOT = Path(r"C:\AI")
BEAT_CHECKPOINT = ROOT / "runtime" / "models" / "beat_this-final0.ckpt"
MARKER = ".dsb-installing"  # present while an install is unfinished, so a half-built environment is never used


@dataclass(frozen=True)
class Component:
    id: str
    label: str
    purpose: str
    required: bool
    gigabytes: float
    folder: str | None  # under the install root; None for what goes into the app's own environment
    python: str  # relative to the folder
    env: str
    needs: tuple[str, ...]
    licenses: tuple[tuple[str, str], ...]


COMPONENTS = {c.id: c for c in (
    Component("motionbricks", "MotionBricks", "フレーズ同士のつなぎを生成します", True, 7.0, "MotionBricks", ".venv/Scripts/python.exe", "DSB_MB_PYTHON", ("gpu", "git", "uv"), (
        ("コード: Apache License 2.0", "https://github.com/NVlabs/MotionBricks/blob/main/LICENSE"),
        ("モデル: NVIDIA Open Model Agreement", "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/"),
        ("TensorRT: NVIDIA TensorRT SLA", "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html"),
        ("同梱サンプル: NVIDIA Sample Data Evaluation License（評価・試験用。アプリの素材には使いません）", "https://github.com/NVlabs/MotionBricks/blob/main/assets/motions/LICENSE.txt"),
    )),
    Component("beat_this", "Beat This!", "曲の拍と小節を検出します", False, 4.0, None, "", "", ("uv",), (
        ("コードと重み: MIT License（学習データの一部に権利上の制限があると配布元が注記）", "https://github.com/CPJKU/beat_this/blob/main/LICENSE"),
        ("PyTorch: BSD-3-Clause", "https://github.com/pytorch/pytorch/blob/main/LICENSE"),
    )),
    Component("embeddinggemma2", "EmbeddingGemma 2", "曲の反復区間の検出と、文章でのフレーズ検索に使います", False, 5.0, "EmbeddingGemma2", ".venv/Scripts/python.exe", "DSB_EG2_PYTHON", ("uv",), (
        ("モデル: Apache License 2.0", "https://huggingface.co/google/embeddinggemma-2"),
        ("PyTorch: BSD-3-Clause", "https://github.com/pytorch/pytorch/blob/main/LICENSE"),
    )),
    Component("blender", "Blender（bpy）", "FBX の書き出しに使います", False, 0.7, "Mocap/bpy311", "Scripts/python.exe", "DSB_BPY_PYTHON", ("uv",), (
        ("Blender: GNU GPL v3 以降", "https://www.blender.org/about/license/"),
    )),
)}


def registered() -> dict[str, str]:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def register(component_id: str, python: Path) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({**registered(), component_id: str(python)}, ensure_ascii=False, indent=1), encoding="utf-8")


def python_of(component_id: str) -> Path:
    item = COMPONENTS[component_id]
    if item.folder is None:
        return Path(sys.executable)
    chosen = os.environ.get(item.env) or registered().get(component_id)
    return Path(chosen) if chosen else DEFAULT_ROOT / item.folder / item.python


def folder_of(component_id: str) -> Path:
    item = COMPONENTS[component_id]
    return python_of(component_id).parents[len(Path(item.python).parts) - 1]


def eg2_model() -> Path:
    return folder_of("embeddinggemma2") / "models" / "embeddinggemma-2"


def usable(component_id: str) -> bool:
    if component_id == "beat_this":
        return BEAT_CHECKPOINT.is_file() and importlib.util.find_spec("beat_this") is not None
    if not python_of(component_id).is_file() or (folder_of(component_id) / MARKER).exists():
        return False
    return component_id != "embeddinggemma2" or (eg2_model() / "model.safetensors").is_file()
