"""Embeds text and audio spans with EmbeddingGemma 2. Runs in that environment, offline.

    python backend/eg2_worker.py --request request.json --output embeddings.npz --model models/embeddinggemma-2

request.json: {"items": [{"id", "type": "text", "text", "prompt"} | {"id", "type": "audio", "path", "start", "end"}], "batch_size": 4}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

SAMPLE_RATE = 16000
DIMENSIONS = 768


def read_audio(path: str, start: float, end: float) -> np.ndarray:
    import soundfile
    import soxr

    with soundfile.SoundFile(path) as handle:
        rate = handle.samplerate
        first = int(round(start * rate))
        handle.seek(min(first, len(handle)))
        data = handle.read(frames=max(0, int(round(end * rate)) - first), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        raise ValueError(f"no audio in the requested span: {path}")
    return mono if rate == SAMPLE_RATE else soxr.resample(mono, rate, SAMPLE_RATE, quality="HQ").astype(np.float32)


def load_model(folder: str, kinds: set[str]):
    import torch
    from sentence_transformers import SentenceTransformer

    # Encoders for inputs this request does not have stay unloaded.
    config = {"vision_config": None}
    if "audio" not in kinds:
        config["audio_config"] = None
    cuda = torch.cuda.is_available()
    return SentenceTransformer(folder, device="cuda" if cuda else "cpu", local_files_only=True, config_kwargs=config,
                               model_kwargs={"dtype": torch.bfloat16 if cuda else torch.float32})


def embed(items: list[dict], folder: str, batch: int) -> np.ndarray:
    model = load_model(folder, {item["type"] for item in items})
    vectors = np.zeros((len(items), DIMENSIONS), dtype=np.float32)
    groups: dict[str | None, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item.get("prompt", "Document") if item["type"] == "text" else None, []).append(index)
    for prompt, indexes in groups.items():
        inputs = [items[i]["text"] if items[i]["type"] == "text" else {"audio": read_audio(items[i]["path"], items[i]["start"], items[i]["end"])} for i in indexes]
        result = model.encode(inputs, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, truncate_dim=DIMENSIONS,
                              show_progress_bar=False, **({"prompt_name": prompt} if prompt else {}))
        vectors[indexes] = np.asarray(result, dtype=np.float32)
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    items = request["items"]
    vectors = embed(items, args.model, int(request.get("batch_size", 4)))
    if not np.isfinite(vectors).all():
        raise ValueError("the model returned non-finite vectors")
    np.savez(args.output, ids=np.array([item["id"] for item in items]), vectors=vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
