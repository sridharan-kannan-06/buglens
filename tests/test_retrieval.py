from __future__ import annotations

import numpy as np
import pytest

from buglens import config
from buglens.index import RepoIndex
from buglens.models import Chunk, ScreenshotAnalysis
from buglens.retrieval import (
    aggregate_files,
    build_queries,
    idf_weight,
    lexical_chunk_scores,
    retrieve,
    semantic_chunk_scores,
)
from tests.fakes import FakeEmbedder


def make_chunk(path: str, text: str, start: int = 1) -> Chunk:
    return Chunk(path=path, start_line=start, end_line=start + 9, text=text)


def test_semantic_scores_take_top_k_per_query():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = semantic_chunk_scores(vectors, queries, top_k=1)
    assert scores == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}

    scores = semantic_chunk_scores(vectors, queries, top_k=2)
    assert scores[2] == pytest.approx(0.8)  # best similarity over both queries


def test_idf_weight_down_weights_common_strings():
    assert idf_weight(1, 50) == pytest.approx(1.0)
    assert idf_weight(1, 50) > idf_weight(5, 50) > idf_weight(40, 50) > 0
    assert idf_weight(0, 50) == 0.0


def test_lexical_match_is_case_insensitive_and_idf_weighted():
    chunks = [
        make_chunk("a.py", 'raise Error("Invalid Username or password")'),
        make_chunk("b.py", "label = 'Submit'"),
        make_chunk("c.py", "button = 'Submit'"),
        make_chunk("d.py", "nothing here"),
    ]
    chunk_weights, file_weights, matches = lexical_chunk_scores(
        chunks, {"invalid username or password": 1.0, "Submit": 1.0, "OK": 1.0}
    )
    assert file_weights["a.py"] == pytest.approx(1.0)  # unique string: full weight
    assert 0 < file_weights["b.py"] < file_weights["a.py"]  # appears in two files: down-weighted
    assert file_weights["b.py"] == pytest.approx(file_weights["c.py"])
    assert "d.py" not in file_weights
    assert matches["a.py"] == ["invalid username or password"]
    assert set(chunk_weights) == {0, 1, 2}  # "OK" is too short to be searched


def test_aggregate_adds_hit_bonus_and_lexical_bonus():
    chunks = [
        make_chunk("multi.py", "one", 1),
        make_chunk("multi.py", "two", 51),
        make_chunk("multi.py", "three", 101),
        make_chunk("single.py", "four"),
        make_chunk("lexical_only.py", "five"),
    ]
    semantic = {0: 0.50, 1: 0.40, 2: 0.30, 3: 0.55}
    candidates = aggregate_files(
        chunks,
        semantic,
        lexical_chunks={4: 1.0},
        lexical_files={"lexical_only.py": 1.0},
        lexical_matches={"lexical_only.py": ["Save changes"]},
    )
    by_path = {candidate.path: candidate for candidate in candidates}

    multi = by_path["multi.py"]
    assert multi.semantic_score == pytest.approx(0.50)
    assert multi.hit_bonus == pytest.approx(2 * config.EXTRA_HIT_BONUS)
    assert multi.score == pytest.approx(0.50 + 2 * config.EXTRA_HIT_BONUS)
    assert [snippet.start_line for snippet in multi.snippets] == [1, 51]  # best chunks first, capped

    assert by_path["single.py"].score == pytest.approx(0.55)
    lexical_only = by_path["lexical_only.py"]
    assert lexical_only.score == pytest.approx(config.LEXICAL_WEIGHT)
    assert lexical_only.matched_strings == ["Save changes"]
    assert lexical_only.snippets[0].text == "five"

    assert [candidate.path for candidate in candidates] == ["multi.py", "single.py", "lexical_only.py"]


def test_aggregate_caps_bonuses_and_number_of_files():
    chunks = [make_chunk(f"f{number}.py", "x") for number in range(30)]
    chunks += [make_chunk("big.py", "y", start) for start in range(1, 1000, 50)]
    semantic = {index: 0.1 for index in range(len(chunks))}
    candidates = aggregate_files(
        chunks, semantic, lexical_chunks={}, lexical_files={"f0.py": 100.0}, lexical_matches={}
    )
    assert len(candidates) == config.TOP_CANDIDATE_FILES
    by_path = {candidate.path: candidate for candidate in candidates}
    assert by_path["f0.py"].lexical_bonus == pytest.approx(config.LEXICAL_BONUS_CAP)
    assert by_path["big.py"].hit_bonus == pytest.approx(config.MAX_EXTRA_HITS * config.EXTRA_HIT_BONUS)


def test_build_queries_deduplicates():
    analysis = ScreenshotAnalysis(search_queries=["login form handler", "Login form handler", "password check"])
    assert build_queries("login form handler", analysis) == ["login form handler", "password check"]


def test_retrieve_with_fake_embeddings_combines_semantic_and_lexical():
    embedder = FakeEmbedder()
    chunks = [
        make_chunk("src/login.py", "def login(request): check password for username and sign in"),
        make_chunk("src/cart.py", "def cart_total(items): add prices and apply discount"),
        make_chunk("src/errors.py", 'MESSAGES = {"auth": "Invalid username or password"}'),
        make_chunk("src/profile.py", "def profile(user): show avatar and biography"),
    ]
    index = RepoIndex(
        chunks=chunks,
        vectors=embedder.embed_documents([chunk.text for chunk in chunks]),
        files_total=4,
        files_indexed=4,
    )
    analysis = ScreenshotAnalysis(
        error_messages=["Invalid username or password"],
        search_queries=["login request password check"],
    )
    candidates = retrieve(index, embedder, "cannot sign in", analysis)
    paths = [candidate.path for candidate in candidates]

    assert set(paths[:2]) == {"src/login.py", "src/errors.py"}
    errors_file = next(candidate for candidate in candidates if candidate.path == "src/errors.py")
    assert errors_file.matched_strings == ["Invalid username or password"]
    assert errors_file.lexical_bonus == pytest.approx(config.LEXICAL_WEIGHT * config.ERROR_MESSAGE_BOOST)
    assert all(candidate.snippets for candidate in candidates)


def test_retrieve_on_empty_index_returns_nothing():
    index = RepoIndex(chunks=[], vectors=np.zeros((0, 0), dtype=np.float32), files_total=0, files_indexed=0)
    assert retrieve(index, FakeEmbedder(), "anything", ScreenshotAnalysis()) == []
