"""Calls EmbeddingGemma 2 in its own environment. Every feature here is optional: callers handle EmbedUnavailable."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .components import ROOT, eg2_model, python_of, usable

WORKER = ROOT / "backend" / "eg2_worker.py"


class EmbedUnavailable(RuntimeError):
    pass


def _run(items: list[dict], timeout: float = 300.0) -> np.ndarray:
    if not usable("embeddinggemma2"):
        raise EmbedUnavailable("EmbeddingGemma 2 の環境が見つかりません")
    with tempfile.TemporaryDirectory() as folder:
        request, output = Path(folder) / "request.json", Path(folder) / "embeddings.npz"
        request.write_text(json.dumps({"items": items, "batch_size": 4}, ensure_ascii=False), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            done = subprocess.run([str(python_of("embeddinggemma2")), "-X", "utf8", str(WORKER), "--request", str(request), "--output", str(output), "--model", str(eg2_model())],
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=timeout, check=False, creationflags=flags)
        except subprocess.TimeoutExpired as error:
            raise EmbedUnavailable("EmbeddingGemma 2 がタイムアウトしました") from error
        if done.returncode or not output.is_file():
            raise EmbedUnavailable((done.stderr or done.stdout)[-400:])
        with np.load(output, allow_pickle=False) as data:
            ids, vectors = data["ids"], data["vectors"].astype(np.float64)
    if list(ids) != [item["id"] for item in items]:
        raise EmbedUnavailable("出力の順序が要求と一致しません")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def audio_spans(path: str, spans: list[tuple[float, float]]) -> np.ndarray:
    return _run([{"id": str(i), "type": "audio", "path": str(Path(path).resolve()), "start": float(a), "end": float(b)} for i, (a, b) in enumerate(spans)])


def texts(values: list[tuple[str, str]]) -> np.ndarray:
    """values: (text, prompt name)."""
    return _run([{"id": str(i), "type": "text", "text": text, "prompt": prompt} for i, (text, prompt) in enumerate(values)])
