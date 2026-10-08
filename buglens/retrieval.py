"""Stage 6: hybrid retrieval. Three signals are combined into one score per file.

    score = best semantic chunk score                      (embeddings, cosine similarity)
          + EXTRA_HIT_BONUS  * other top chunks in the file (capped)
          + LEXICAL_WEIGHT   * on-screen strings found in the file, IDF-weighted (capped)
          + PATH_HINT_BONUS  * file names from the screenshot that point at the file (capped)

All weights are named constants in config.py.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from buglens import config
from buglens.embeddings import Embedder
from buglens.hints import match_hint, parse_hint
from buglens.index import RepoIndex
from buglens.models import CandidateFile, Chunk, ScreenshotAnalysis


@dataclass
class PathEvidence:
    """What the file names on the screenshot say about the repository's files."""

    files: dict[str, float] = field(default_factory=dict)  # weight per file; 1.0 = hint fits only this file
    chunks: dict[int, float] = field(default_factory=dict)  # chunks that contain a line named in a hint
    hints: dict[str, list[str]] = field(default_factory=dict)  # hint texts per file
    lines: dict[str, list[int]] = field(default_factory=dict)  # line numbers named per file


def semantic_chunk_scores(vectors: np.ndarray, query_vectors: np.ndarray, top_k: int) -> dict[int, float]:
    """For every query take its `top_k` chunks. Returns {chunk index: best cosine similarity}.

    Rows of both matrices are L2-normalised, so the dot product is the cosine similarity.
    """
    scores: dict[int, float] = {}
    if vectors.shape[0] == 0 or query_vectors.shape[0] == 0:
        return scores
    similarities = query_vectors @ vectors.T  # shape: (queries, chunks)
    keep = min(top_k, vectors.shape[0])
    for row in similarities:
        for index in np.argsort(-row)[:keep]:
            scores[int(index)] = max(scores.get(int(index), -1.0), float(row[index]))
    return scores


def idf_weight(files_with_string: int, total_files: int) -> float:
    """1.0 for a string found in a single file, falling towards 0 as it appears in more files."""
    if files_with_string <= 0 or total_files <= 0:
        return 0.0
    return math.log((total_files + 1) / files_with_string) / math.log(total_files + 1)


def lexical_chunk_scores(
    chunks: list[Chunk],
    strings: dict[str, float],
) -> tuple[dict[int, float], dict[str, float], dict[str, list[str]]]:
    """Case-insensitive substring search of screenshot strings across all chunks.

    `strings` maps each search string to a boost (error messages count more).
    Returns (per-chunk weight, per-file weight, per-file matched strings).
    """
    lowered = [chunk.text.lower() for chunk in chunks]
    total_files = len({chunk.path for chunk in chunks})
    chunk_weights: dict[int, float] = defaultdict(float)
    file_weights: dict[str, float] = defaultdict(float)
    file_matches: dict[str, list[str]] = defaultdict(list)

    for string, boost in strings.items():
        needle = string.lower().strip()
        if len(needle) < config.MIN_LEXICAL_CHARS:
            continue
        hit_chunks = [index for index, text in enumerate(lowered) if needle in text]
        hit_files = {chunks[index].path for index in hit_chunks}
        if not hit_files:
            continue
        weight = boost * idf_weight(len(hit_files), total_files)
        for index in hit_chunks:
            chunk_weights[index] += weight
        for path in hit_files:
            file_weights[path] += weight
            file_matches[path].append(string)
    return dict(chunk_weights), dict(file_weights), dict(file_matches)


def path_hint_scores(chunks: list[Chunk], file_hints: list[str]) -> PathEvidence:
    """Match file names seen on screen against the repository's paths.

    A hint that fits one file gives it weight 1.0. A hint that fits several
    files (a bare 'index.js') is shared between them, and ignored when it fits
    more than MAX_PATH_HINT_MATCHES.
    """
    evidence = PathEvidence()
    repo_paths = sorted({chunk.path for chunk in chunks})
    for text in file_hints:
        hint = parse_hint(text)
        matches = match_hint(hint, repo_paths) if hint else []
        if not matches or len(matches) > config.MAX_PATH_HINT_MATCHES:
            continue
        for path in matches:
            evidence.files[path] = evidence.files.get(path, 0.0) + 1.0 / len(matches)
            evidence.hints.setdefault(path, []).append(text)
            if hint.line is None:
                continue
            evidence.lines.setdefault(path, []).append(hint.line)
            for index, chunk in enumerate(chunks):
                if chunk.path == path and chunk.start_line <= hint.line <= chunk.end_line:
                    evidence.chunks[index] = evidence.chunks.get(index, 0.0) + 1.0
                    break
    return evidence


def aggregate_files(
    chunks: list[Chunk],
    semantic: dict[int, float],
    lexical_chunks: dict[int, float],
    lexical_files: dict[str, float],
    lexical_matches: dict[str, list[str]],
    path_evidence: PathEvidence | None = None,
    top_files: int = config.TOP_CANDIDATE_FILES,
) -> list[CandidateFile]:
    """Combine chunk-level evidence into one score per file and return the best files."""
    paths = path_evidence or PathEvidence()
    semantic_by_file: dict[str, list[float]] = defaultdict(list)
    for index, score in semantic.items():
        semantic_by_file[chunks[index].path].append(score)

    candidates = []
    for path in set(semantic_by_file) | set(lexical_files) | set(paths.files):
        hits = semantic_by_file.get(path, [])
        best = max(hits) if hits else 0.0
        hit_bonus = config.EXTRA_HIT_BONUS * min(max(len(hits) - 1, 0), config.MAX_EXTRA_HITS)
        lexical_bonus = min(config.LEXICAL_WEIGHT * lexical_files.get(path, 0.0), config.LEXICAL_BONUS_CAP)
        path_bonus = min(config.PATH_HINT_BONUS * paths.files.get(path, 0.0), config.PATH_HINT_BONUS_CAP)
        candidates.append(
            CandidateFile(
                path=path,
                score=best + hit_bonus + lexical_bonus + path_bonus,
                semantic_score=best,
                hit_bonus=hit_bonus,
                lexical_bonus=lexical_bonus,
                path_bonus=path_bonus,
                matched_strings=lexical_matches.get(path, []),
                path_hints=paths.hints.get(path, []),
                hint_lines=paths.lines.get(path, []),
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.path))
    candidates = candidates[:top_files]

    # Attach the best chunks of each file as evidence for the reranker.
    chunk_scores: dict[int, float] = defaultdict(float)
    for index, score in semantic.items():
        chunk_scores[index] += score
    for index, weight in lexical_chunks.items():
        chunk_scores[index] += config.LEXICAL_WEIGHT * weight
    for index, weight in paths.chunks.items():
        chunk_scores[index] += config.PATH_HINT_BONUS * weight
    for candidate in candidates:
        own = [index for index in chunk_scores if chunks[index].path == candidate.path]
        own.sort(key=lambda index: (-chunk_scores[index], index))
        if not own:  # named on screen without a line number: show the top of the file
            own = [index for index, chunk in enumerate(chunks) if chunk.path == candidate.path][:1]
        candidate.snippets = [chunks[index] for index in own[: config.SNIPPETS_PER_FILE]]
    return candidates


def build_queries(user_text: str, analysis: ScreenshotAnalysis) -> list[str]:
    """The user's sentence plus the model's search queries, without duplicates."""
    queries: list[str] = []
    for query in [user_text, *analysis.search_queries]:
        query = query.strip()
        if query and query.lower() not in {existing.lower() for existing in queries}:
            queries.append(query)
    return queries


def lexical_strings(analysis: ScreenshotAnalysis) -> dict[str, float]:
    """Strings to search for exactly, mapped to their boost."""
    strings = {text: 1.0 for text in analysis.visible_text}
    strings.update({text: config.IDENTIFIER_BOOST for text in analysis.identifiers})
    strings.update({text: config.ERROR_MESSAGE_BOOST for text in analysis.error_messages})
    return strings


def retrieve(
    index: RepoIndex,
    embedder: Embedder,
    user_text: str,
    analysis: ScreenshotAnalysis,
) -> list[CandidateFile]:
    """Return up to TOP_CANDIDATE_FILES files, best first, with their best snippets."""
    if not index.chunks:
        return []
    query_vectors = embedder.embed_queries(build_queries(user_text, analysis))
    semantic = semantic_chunk_scores(index.vectors, query_vectors, config.TOP_CHUNKS_PER_QUERY)
    lexical_chunks, lexical_files, lexical_matches = lexical_chunk_scores(index.chunks, lexical_strings(analysis))
    path_evidence = path_hint_scores(index.chunks, analysis.file_hints)
    return aggregate_files(index.chunks, semantic, lexical_chunks, lexical_files, lexical_matches, path_evidence)
