"""Build, save and load the per-repo search index (chunks + embedding vectors).

The index is stored on disk as `chunks.json`, `vectors.npy` and `meta.json`,
in a folder keyed by repo and commit sha. No vector database is involved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from buglens import config
from buglens.chunking import chunk_files, embedding_text
from buglens.embed_cache import EmbeddingCache, text_key
from buglens.embeddings import Embedder, ProgressFn
from buglens.models import Chunk
from buglens.repo_loader import read_source_files


@dataclass
class RepoIndex:
    """Chunks and their vectors. Row i of `vectors` belongs to `chunks[i]`."""

    chunks: list[Chunk]
    vectors: np.ndarray
    files_total: int
    files_indexed: int
    reused_chunks: int = 0  # chunks whose vectors came from another commit of the same repo

    @property
    def truncated(self) -> bool:
        """True when the chunk cap cut the repo short."""
        return self.files_indexed < self.files_total


def _settings_fingerprint(embedder: Embedder, max_chunks: int) -> dict[str, object]:
    """Everything that, if changed, makes a cached index stale."""
    return {
        "embed_model": embedder.model_name,
        "chunk_lines": config.CHUNK_LINES,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "max_chunks": max_chunks,
    }


def embed_with_reuse(
    texts: list[str],
    embedder: Embedder,
    cache: EmbeddingCache | None,
    on_progress: ProgressFn | None = None,
) -> tuple[np.ndarray, int]:
    """Embed texts, taking vectors from the cache where possible. Returns (vectors, reused count)."""
    if cache is None:
        return embedder.embed_documents(texts, on_progress), 0
    keys = [text_key(text) for text in texts]
    text_by_key = dict(zip(keys, texts))  # identical texts are embedded once
    known = cache.lookup(keys)
    reused = sum(key in known for key in keys)
    missing = [key for key in text_by_key if key not in known]
    if missing:
        fresh = embedder.embed_documents([text_by_key[key] for key in missing], on_progress)
        known.update(zip(missing, fresh))
        cache.add(missing, fresh)
    vectors = np.array([known[key] for key in keys], dtype=np.float32)
    return vectors, reused


def build_index(
    repo_dir: Path,
    embedder: Embedder,
    max_chunks: int,
    on_progress: ProgressFn | None = None,
    cache: EmbeddingCache | None = None,
) -> RepoIndex:
    """Read, chunk and embed a repo folder."""
    files = read_source_files(repo_dir)
    chunks, files_indexed = chunk_files(files, max_chunks)
    reused = 0
    if chunks:
        texts = [embedding_text(chunk) for chunk in chunks]
        vectors, reused = embed_with_reuse(texts, embedder, cache, on_progress)
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)
    return RepoIndex(
        chunks=chunks, vectors=vectors, files_total=len(files), files_indexed=files_indexed, reused_chunks=reused
    )


def save_index(index: RepoIndex, folder: Path, embedder: Embedder, max_chunks: int) -> None:
    """Write the index to disk. meta.json is written last and marks the index as complete."""
    folder.mkdir(parents=True, exist_ok=True)
    chunk_dicts = [chunk.model_dump() for chunk in index.chunks]
    (folder / "chunks.json").write_text(json.dumps(chunk_dicts), encoding="utf-8")
    np.save(folder / "vectors.npy", index.vectors, allow_pickle=False)
    meta = _settings_fingerprint(embedder, max_chunks)
    meta.update(files_total=index.files_total, files_indexed=index.files_indexed, chunks=len(index.chunks))
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(folder: Path, embedder: Embedder, max_chunks: int) -> RepoIndex | None:
    """Load a cached index, or return None if it is missing, incomplete or stale."""
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        fingerprint = _settings_fingerprint(embedder, max_chunks)
        if any(meta.get(key) != value for key, value in fingerprint.items()):
            return None
        chunk_dicts = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if len(chunk_dicts) != vectors.shape[0]:
        return None
    return RepoIndex(
        chunks=[Chunk(**item) for item in chunk_dicts],
        vectors=vectors,
        files_total=int(meta.get("files_total", 0)),
        files_indexed=int(meta.get("files_indexed", 0)),
    )


def load_or_build_index(
    repo_dir: Path,
    index_folder: Path,
    embedder: Embedder,
    max_chunks: int,
    on_progress: ProgressFn | None = None,
    reuse_folder: Path | None = None,
) -> tuple[RepoIndex, bool]:
    """Return (index, came_from_cache).

    `reuse_folder` holds the repo-wide embedding cache, shared by all commits of the repo.
    """
    cached = load_index(index_folder, embedder, max_chunks)
    if cached is not None:
        return cached, True
    cache = EmbeddingCache(reuse_folder, embedder.model_name) if reuse_folder else None
    index = build_index(repo_dir, embedder, max_chunks, on_progress, cache)
    save_index(index, index_folder, embedder, max_chunks)
    return index, False
