from __future__ import annotations

import pytest

from buglens.chunking import chunk_files, chunk_text, embedding_text
from buglens.repo_loader import SourceFile


def numbered_lines(count: int) -> str:
    return "\n".join(f"line {number}" for number in range(1, count + 1))


def test_short_file_is_one_chunk():
    chunks = chunk_text("a.py", numbered_lines(10))
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 10)


def test_windows_overlap_by_ten_lines():
    chunks = chunk_text("a.py", numbered_lines(130), size=60, overlap=10)
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 60), (51, 110), (101, 130)]
    assert chunks[1].text.splitlines()[0] == "line 51"
    assert chunks[0].text.splitlines()[-10:] == chunks[1].text.splitlines()[:10]


def test_exact_multiple_does_not_add_an_empty_chunk():
    chunks = chunk_text("a.py", numbered_lines(60), size=60, overlap=10)
    assert len(chunks) == 1


def test_blank_files_give_no_chunks():
    assert chunk_text("a.py", "\n\n   \n") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a.py", "x", size=10, overlap=10)


def test_embedding_text_starts_with_the_path():
    chunk = chunk_text("src/app/login.py", "def login(): pass")[0]
    assert embedding_text(chunk).startswith("File: src/app/login.py\n")


def test_chunk_files_stops_at_the_cap_in_priority_order():
    files = [
        SourceFile(path="code.py", text=numbered_lines(130), priority=0),
        SourceFile(path="docs.md", text=numbered_lines(130), priority=2),
    ]
    chunks, files_indexed = chunk_files(files, max_chunks=4)
    assert len(chunks) == 4
    assert [chunk.path for chunk in chunks] == ["code.py", "code.py", "code.py", "docs.md"]
    assert files_indexed == 1  # docs.md was cut short


def test_chunk_files_without_truncation():
    files = [SourceFile(path="code.py", text=numbered_lines(20), priority=0)]
    chunks, files_indexed = chunk_files(files, max_chunks=100)
    assert (len(chunks), files_indexed) == (1, 1)
