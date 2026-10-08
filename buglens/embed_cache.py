"""Remember chunk embeddings per repository so that other commits can reuse them.

Two commits of the same repository share almost all of their code. The index
is keyed by commit, so without this cache every new commit would embed every
chunk again. Here vectors are stored by a hash of the embedded text: only
chunks whose text actually changed are sent to the embedding model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from buglens import config


def text_key(text: str) -> str:
    """Stable identifier of an embedded text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """A folder with `keys.json` and `vectors.npy`. Row i of the matrix belongs to keys[i]."""

    def __init__(self, folder: Path, model_name: str) -> None:
        self._folder = folder
        self._model_name = model_name
        self._keys: list[str] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        """Read the cache. A missing, damaged or other-model cache simply counts as empty."""
        try:
            meta = json.loads((self._folder / "keys.json").read_text(encoding="utf-8"))
            vectors = np.load(self._folder / "vectors.npy", allow_pickle=False)
        except (OSError, ValueError):
            return
        if meta.get("embed_model") == self._model_name and len(meta.get("keys", [])) == vectors.shape[0]:
            self._keys, self._vectors = meta["keys"], vectors

    def lookup(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Vectors already known for these keys."""
        position = {key: row for row, key in enumerate(self._keys)}
        return {key: self._vectors[position[key]] for key in set(keys) if key in position}

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        """Store new vectors and write the cache. The oldest entries are dropped beyond the cap."""
        if not keys:
            return
        known = set(self._keys)
        fresh = [row for row, key in enumerate(keys) if key not in known]
        if not fresh:
            return
        new_vectors = np.asarray(vectors, dtype=np.float32)[fresh]
        self._vectors = new_vectors if self._vectors.size == 0 else np.vstack([self._vectors, new_vectors])
        self._keys = self._keys + [keys[row] for row in fresh]
        overflow = len(self._keys) - config.EMBED_CACHE_MAX_VECTORS
        if overflow > 0:
            self._keys, self._vectors = self._keys[overflow:], self._vectors[overflow:]
        self._folder.mkdir(parents=True, exist_ok=True)
        np.save(self._folder / "vectors.npy", self._vectors, allow_pickle=False)
        meta = {"embed_model": self._model_name, "keys": self._keys}
        (self._folder / "keys.json").write_text(json.dumps(meta), encoding="utf-8")
