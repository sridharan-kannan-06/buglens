"""Local text embeddings with fastembed (ONNX on CPU, no torch)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from buglens import config

ProgressFn = Callable[[int, int], None]  # (items done, items total)


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised row vectors."""

    model_name: str

    def embed_documents(self, texts: Sequence[str], on_progress: ProgressFn | None = None) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1 so that a dot product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix.reshape(0, 0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 through fastembed. The model is downloaded on first use."""

    def __init__(self, model_name: str = config.EMBED_MODEL, cache_folder: Path | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily: it is slow to import

        self.model_name = model_name
        folder = cache_folder or config.cache_dir() / "fastembed"
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(folder))

    def embed_documents(self, texts: Sequence[str], on_progress: ProgressFn | None = None) -> np.ndarray:
        """Embed code chunks in batches, reporting progress after each batch."""
        vectors: list[np.ndarray] = []
        batch = config.EMBED_BATCH_SIZE
        for start in range(0, len(texts), batch):
            vectors.extend(self._model.embed(list(texts[start:start + batch]), batch_size=batch))
            if on_progress:
                on_progress(min(start + batch, len(texts)), len(texts))
        return normalize(np.array(vectors, dtype=np.float32))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed search queries (fastembed adds the model's query prefix when it has one)."""
        return normalize(np.array(list(self._model.query_embed(list(texts))), dtype=np.float32))


@lru_cache(maxsize=1)
def get_embedder() -> FastEmbedEmbedder:
    """Shared embedder, so the model is loaded once per process."""
    return FastEmbedEmbedder()
