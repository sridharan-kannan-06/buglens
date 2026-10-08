"""Split source files into overlapping line windows."""
from __future__ import annotations

from collections.abc import Iterable

from buglens import config
from buglens.models import Chunk
from buglens.repo_loader import SourceFile


def chunk_text(
    path: str,
    text: str,
    size: int = config.CHUNK_LINES,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Cut one file into windows of `size` lines; consecutive windows share `overlap` lines."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than the chunk size")
    lines = text.splitlines()
    chunks: list[Chunk] = []
    for start in range(0, len(lines), size - overlap):
        window = lines[start:start + size]
        if any(line.strip() for line in window):
            chunks.append(
                Chunk(path=path, start_line=start + 1, end_line=start + len(window), text="\n".join(window))
            )
        if start + size >= len(lines):
            break
    return chunks


def chunk_files(files: Iterable[SourceFile], max_chunks: int) -> tuple[list[Chunk], int]:
    """Chunk files in the given (priority) order until `max_chunks` is reached.

    Returns the chunks and the number of files that were fully indexed.
    """
    chunks: list[Chunk] = []
    files_indexed = 0
    for source in files:
        file_chunks = chunk_text(source.path, source.text)
        room = max_chunks - len(chunks)
        if len(file_chunks) > room:
            chunks.extend(file_chunks[:room])
            break
        chunks.extend(file_chunks)
        files_indexed += 1
    return chunks, files_indexed


def embedding_text(chunk: Chunk) -> str:
    """Text sent to the embedding model: the file path gives the chunk its context."""
    return f"File: {chunk.path}\n{chunk.text}"
