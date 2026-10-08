"""Test doubles. These exist for the test suite only: BugLens has no mock mode."""
from __future__ import annotations

import re
import zipfile
import zlib
from pathlib import Path

import numpy as np

from buglens.embeddings import normalize
from buglens.errors import RepoNotFound
from buglens.llm.base import LLMClient
from buglens.models import RepoRef

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeLLM(LLMClient):
    """Replays canned replies in order and records every call. An exception in the list is raised."""

    name = "fake"
    model = "fake-model"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, bytes | None]] = []

    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        self.calls.append((prompt, image_png))
        if not self.replies:
            raise AssertionError("FakeLLM was called more often than expected")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeEmbedder:
    """Deterministic bag-of-words embedding: texts that share words get similar vectors."""

    model_name = "fake-bag-of-words"

    def __init__(self, size: int = 256) -> None:
        self.size = size
        self.document_calls = 0

    def _embed(self, texts) -> np.ndarray:
        matrix = np.zeros((len(texts), self.size), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]{3,}", text.lower()):
                matrix[row, zlib.crc32(word.encode()) % self.size] += 1.0
        return normalize(matrix)

    def embed_documents(self, texts, on_progress=None) -> np.ndarray:
        self.document_calls += 1
        return self._embed(texts)

    def embed_queries(self, texts) -> np.ndarray:
        return self._embed(texts)


def zip_folder(source: Path, dest: Path, top_folder: str = "owner-repo-0123456") -> None:
    """Zip a folder the way GitHub does: everything sits under one top-level folder."""
    with zipfile.ZipFile(dest, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, f"{top_folder}/{path.relative_to(source).as_posix()}")


class FakeGitHub:
    """Serves a local folder as if it were a GitHub repository. No network."""

    authenticated = False

    def __init__(
        self,
        source: Path,
        labels: list[str] | None = None,
        exists: bool = True,
        issues: list[dict] | None = None,
    ) -> None:
        self.source = source
        self.labels = labels
        self.exists = exists
        self.issues = issues or []
        self.downloads = 0
        self.searches: list[list[str]] = []

    def resolve_ref(self, repo: RepoRef) -> tuple[str, str]:
        if not self.exists:
            raise RepoNotFound("GitHub returned 404: the repository or ref does not exist, or it is private.")
        return repo.ref or "main", FAKE_SHA

    def download_zipball(self, repo: RepoRef, sha: str, dest: Path, max_bytes: int) -> None:
        self.downloads += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        zip_folder(self.source, dest)

    def list_labels(self, repo: RepoRef, limit: int = 100) -> list[str]:
        if self.labels is None:
            raise RepoNotFound("labels unavailable")
        return self.labels[:limit]

    def list_issues(self, repo: RepoRef, limit: int = 100) -> list[dict]:
        return self.issues[:limit]

    def search_issues(self, repo: RepoRef, keywords: list[str], limit: int = 20) -> list[dict]:
        self.searches.append(keywords)
        return []
