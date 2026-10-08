"""Tests for the evidence features: file-name hints, code evidence, similar issues,
the prefilled issue link, embedding reuse and retry notices."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from buglens import config, pipeline
from buglens.config import Settings, check_llm_settings
from buglens.embed_cache import EmbeddingCache, text_key
from buglens.errors import ConfigError, GitHubRateLimit
from buglens.evidence import attach_evidence, build_evidence, focus_line_in
from buglens.hints import PathHint, match_hint, parse_hint
from buglens.index import RepoIndex, embed_with_reuse, load_or_build_index
from buglens.llm.base import RetryableLLMError, call_with_backoff
from buglens.models import CandidateFile, Chunk, RankedFile, RepoRef, ScreenshotAnalysis
from buglens.report import new_issue_url
from buglens.retrieval import lexical_strings, path_hint_scores, retrieve
from buglens.similar_issues import find_similar_issues, issue_text, rank_similar, search_keywords
from tests.fakes import FAKE_SHA, FakeEmbedder, FakeGitHub, FakeLLM
from tests.test_pipeline import RERANK_REPLY, REPO_URL, USER_TEXT, VISION_REPLY, WRITER_REPLY


# ---------------------------------------------------------------- file hints
@pytest.mark.parametrize(
    ("text", "path", "line"),
    [
        ("src/views/login.py:42", "src/views/login.py", 42),
        ("src\\views\\login.py:42:7", "src/views/login.py", 42),
        ('File "app/views/login.py", line 42, in login', "app/views/login.py", 42),
        ("login.py, line 9", "login.py", 9),
        ("at LoginForm (LoginForm.tsx:12)", "LoginForm.tsx", 12),
        ("webpack-internal:///./src/components/LoginForm.jsx:18:5", "src/components/LoginForm.jsx", 18),
        ("http://localhost:3000/static/js/main.chunk.js:120", "static/js/main.chunk.js", 120),
        ("C:\\Users\\me\\project\\app\\views.py", "Users/me/project/app/views.py", None),
        ("app/(shop)/[id]/page.tsx", "app/(shop)/[id]/page.tsx", None),
        ("`./settings.py`", "settings.py", None),
    ],
)
def test_parse_hint(text, path, line):
    assert parse_hint(text) == PathHint(path=path, line=line)


@pytest.mark.parametrize("text", ["", "README", ".env", "Save changes", "v1.2", "a.b", "line 42"])
def test_parse_hint_rejects_things_that_are_not_files(text):
    assert parse_hint(text) is None


def test_match_hint_prefers_the_deepest_path_match():
    paths = ["app/views/login.py", "tests/login.py", "app/models/user.py"]
    assert match_hint(PathHint("views/login.py"), paths) == ["app/views/login.py"]
    assert match_hint(PathHint("login.py"), paths) == ["app/views/login.py", "tests/login.py"]
    assert match_hint(PathHint("/home/ci/build/APP/Views/Login.py"), paths) == ["app/views/login.py"]
    assert match_hint(PathHint("missing.py"), paths) == []
    assert match_hint(PathHint("ogin.py"), paths) == []  # the file name must match completely


def chunk(path: str, text: str, start: int = 1, end: int | None = None) -> Chunk:
    return Chunk(path=path, start_line=start, end_line=end or start + text.count("\n"), text=text)


def test_path_hints_point_at_files_and_lines():
    chunks = [
        chunk("app/views/login.py", "a\n" * 59 + "a", 1, 60),
        chunk("app/views/login.py", "b\n" * 59 + "b", 51, 110),
        chunk("tests/login.py", "c"),
        chunk("web/index.js", "d"),
        chunk("docs/index.js", "e"),
    ]
    evidence = path_hint_scores(chunks, ["views/login.py:70", "index.js", "nothing.py", "not a file"])

    assert evidence.files["app/views/login.py"] == pytest.approx(1.0)  # the hint fits one file only
    assert evidence.files["web/index.js"] == pytest.approx(0.5)  # a bare name is shared between its matches
    assert "tests/login.py" not in evidence.files
    assert evidence.chunks == {1: 1.0}  # line 70 lies in the second chunk
    assert evidence.lines == {"app/views/login.py": [70]}
    assert evidence.hints["app/views/login.py"] == ["views/login.py:70"]


def test_a_name_that_fits_too_many_files_is_ignored():
    chunks = [chunk(f"pkg{number}/index.js", "x") for number in range(config.MAX_PATH_HINT_MATCHES + 1)]
    assert path_hint_scores(chunks, ["index.js"]).files == {}


def test_a_file_named_in_a_stack_trace_outranks_semantic_matches():
    embedder = FakeEmbedder()
    chunks = [
        chunk("src/billing/invoice.py", "def total(items): return sum(prices)"),
        chunk("src/cart/cart.py", "def cart_total(items): add up the cart prices and discount"),
        chunk("src/cart/view.py", "def render_cart(cart): show cart total prices"),
    ]
    index = RepoIndex(
        chunks=chunks, vectors=embedder.embed_documents([item.text for item in chunks]), files_total=3, files_indexed=3
    )
    plain = ScreenshotAnalysis(search_queries=["cart total prices discount"])
    hinted = plain.model_copy(update={"file_hints": ["billing/invoice.py:1"]})

    assert retrieve(index, embedder, "cart total is wrong", plain)[0].path != "src/billing/invoice.py"
    top = retrieve(index, embedder, "cart total is wrong", hinted)[0]
    assert top.path == "src/billing/invoice.py"
    assert top.path_bonus == pytest.approx(config.PATH_HINT_BONUS)
    assert (top.path_hints, top.hint_lines) == (["billing/invoice.py:1"], [1])
    assert top.snippets[0].path == "src/billing/invoice.py"


def test_identifiers_are_searched_exactly_with_a_boost():
    analysis = ScreenshotAnalysis(visible_text=["Save"], identifiers=["handleSubmit"], error_messages=["Boom"])
    assert lexical_strings(analysis) == {
        "Save": 1.0,
        "handleSubmit": config.IDENTIFIER_BOOST,
        "Boom": config.ERROR_MESSAGE_BOOST,
    }


# ------------------------------------------------------------------ evidence
def numbered(count: int, start: int = 1) -> str:
    return "\n".join(f"line {number}" for number in range(start, start + count))


def test_focus_line_prefers_the_most_specific_string():
    snippet = chunk("a.py", 'title = "Username"\nprint("Invalid username or password")', 10)
    assert focus_line_in(snippet, ["Username", "Invalid username or password"]) == 11
    assert focus_line_in(snippet, ["USERNAME"]) == 10  # case is ignored
    assert focus_line_in(snippet, ["missing", ""]) is None


def test_evidence_centres_on_a_line_named_in_a_stack_trace():
    candidate = CandidateFile(
        path="a.py",
        score=1.0,
        snippets=[chunk("a.py", numbered(60), 1, 60)],
        matched_strings=["line 3"],
        path_hints=["a.py:40"],
        hint_lines=[40],
    )
    evidence = build_evidence(candidate)
    assert evidence.focus_line == 40  # the named line wins over the matched string on line 3
    assert (evidence.start_line, evidence.end_line) == (40 - config.EVIDENCE_CONTEXT_LINES, 40 + config.EVIDENCE_CONTEXT_LINES)
    assert evidence.snippet.splitlines()[0] == "line 34" and evidence.snippet.splitlines()[-1] == "line 46"
    assert evidence.path_hints == ["a.py:40"]


def test_evidence_without_a_match_shows_the_start_of_the_best_chunk():
    candidate = CandidateFile(path="a.py", score=0.5, snippets=[chunk("a.py", numbered(60, 101), 101, 160)])
    evidence = build_evidence(candidate)
    assert evidence.focus_line is None
    assert (evidence.start_line, evidence.end_line) == (101, 100 + config.EVIDENCE_MAX_LINES)
    assert build_evidence(CandidateFile(path="b.py", score=0.1)) is None


def test_attach_evidence_keeps_the_ranking_and_never_trusts_model_supplied_evidence():
    candidates = [CandidateFile(path="a.py", score=1.0, snippets=[chunk("a.py", "x = 1")])]
    ranked = attach_evidence([RankedFile(path="a.py", reason="r", confidence=0.7)], candidates)
    assert ranked[0].evidence.snippet == "x = 1" and ranked[0].confidence == 0.7

    from buglens.rerank import rerank

    reply = json.dumps(
        {"files": [{"path": "a.py", "reason": "r", "confidence": 0.9,
                    "evidence": {"start_line": 1, "end_line": 1, "snippet": "INVENTED BY THE MODEL"}}]}
    )
    ranked, _ = rerank(FakeLLM([reply]), "text", "{}", candidates)
    assert ranked[0].evidence.snippet == "x = 1"


# ------------------------------------------------------------ similar issues
def issue(number: int, title: str, body: str = "", state: str = "open") -> dict:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
    }


ISSUES = [
    issue(1, "Dark mode toggle resets after reload", "theme preference is lost"),
    issue(2, "Login fails with valid password", "invalid username or password shown for a correct login", "closed"),
    issue(3, "Add export to CSV", "feature request for the reports page"),
]


def test_search_keywords_picks_long_uncommon_words():
    assert search_keywords("The login page does not work properly with a valid password") == ["password", "login", "valid"]
    assert search_keywords("it is not ok") == []


def test_issue_text_is_title_plus_capped_body():
    text = issue_text(issue(1, "Title", "b" * 5000))
    assert text.startswith("Title\n") and len(text) == len("Title\n") + config.SIMILAR_ISSUE_BODY_CHARS


def test_rank_similar_orders_by_similarity_and_applies_the_threshold():
    query = "Login fails with a valid password: invalid username or password"
    ranked = rank_similar(FakeEmbedder(), query, ISSUES, min_score=0.3)
    assert [item.number for item in ranked] == [2]
    assert ranked[0].state == "closed" and ranked[0].url.endswith("/issues/2")
    assert 0.3 <= ranked[0].similarity <= 1.0

    assert rank_similar(FakeEmbedder(), query, ISSUES, min_score=0.99) == []
    assert rank_similar(FakeEmbedder(), query, []) == []
    assert len(rank_similar(FakeEmbedder(), query, ISSUES, top=2, min_score=-1.0)) == 2


def test_find_similar_merges_recent_and_searched_issues(fixture_repo):
    github = FakeGitHub(fixture_repo, issues=ISSUES + [ISSUES[1]])  # a duplicate entry is ignored
    similar, warnings = find_similar_issues(
        github, FakeEmbedder(), RepoRef(owner="owner", repo="repo"), "Login fails with valid password"
    )
    assert warnings == []
    assert github.searches == [["password", "login", "fails"]]
    assert [item.number for item in similar][:1] == [2]


def test_similar_issue_check_survives_github_errors(fixture_repo):
    class Limited(FakeGitHub):
        def list_issues(self, repo, limit=100):
            raise GitHubRateLimit("GitHub API rate limit reached.")

    similar, warnings = find_similar_issues(
        Limited(fixture_repo), FakeEmbedder(), RepoRef(owner="owner", repo="repo"), "login password"
    )
    assert similar == []
    assert len(warnings) == 1 and "rate limit" in warnings[0]


def run_pipeline(tmp_path, fixture_repo, png_bytes, **kwargs):
    github = kwargs.pop("github", None) or FakeGitHub(fixture_repo, labels=["bug"])
    return pipeline.analyze(
        REPO_URL,
        USER_TEXT,
        png_bytes,
        llm=FakeLLM([VISION_REPLY, RERANK_REPLY, WRITER_REPLY]),
        github=github,
        embedder=FakeEmbedder(),
        cache_root=tmp_path / "cache",
        **kwargs,
    )


def test_pipeline_reports_similar_issues_without_an_extra_llm_call(tmp_path, fixture_repo, png_bytes, monkeypatch):
    monkeypatch.setattr(config, "SIMILAR_ISSUE_MIN_SCORE", 0.3)  # the bag-of-words test embedder scores lower
    result = run_pipeline(tmp_path, fixture_repo, png_bytes, github=FakeGitHub(fixture_repo, labels=["bug"], issues=ISSUES))
    assert result.similar_issues and result.similar_issues[0].number == 2
    assert "similar" in result.timings


def test_similar_issue_check_can_be_switched_off(tmp_path, fixture_repo, png_bytes):
    result = run_pipeline(tmp_path, fixture_repo, png_bytes, check_similar=False)
    assert result.similar_issues == [] and "similar" not in result.timings


# ------------------------------------------------------- prefilled issue link
def test_new_issue_url_prefills_title_labels_and_body(tmp_path, fixture_repo, png_bytes):
    result = run_pipeline(tmp_path, fixture_repo, png_bytes)
    url, body_included = new_issue_url(result)
    parsed = urlparse(url)
    fields = parse_qs(parsed.query)

    assert body_included is True
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == f"{REPO_URL}/issues/new"
    assert fields["title"] == [result.issue.title]
    assert fields["labels"] == ["bug"]
    assert fields["body"] == [result.issue.body_markdown]  # survives the encoding unchanged


def test_new_issue_url_drops_a_body_that_is_too_long(tmp_path, fixture_repo, png_bytes):
    result = run_pipeline(tmp_path, fixture_repo, png_bytes)
    result.issue.body_markdown = "long line of text\n" * 2000
    result.issue.labels = []
    url, body_included = new_issue_url(result)

    assert body_included is False
    assert len(url) < config.MAX_PREFILL_URL_CHARS
    assert set(parse_qs(urlparse(url).query)) == {"title"}


# ---------------------------------------------------------- embedding reuse
class CountingEmbedder(FakeEmbedder):
    """Remembers how many texts it was asked to embed."""

    def __init__(self) -> None:
        super().__init__()
        self.texts_embedded = 0

    def embed_documents(self, texts, on_progress=None):
        self.texts_embedded += len(texts)
        return super().embed_documents(texts, on_progress)


def test_embedding_cache_round_trip(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb", "model-a")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache.add(["k1", "k2"], vectors)

    reloaded = EmbeddingCache(tmp_path / "emb", "model-a")
    known = reloaded.lookup(["k2", "k3"])
    assert list(known) == ["k2"] and known["k2"].tolist() == [0.0, 1.0]
    assert EmbeddingCache(tmp_path / "emb", "model-b").lookup(["k1"]) == {}  # another model: start empty
    assert text_key("abc") == text_key("abc") != text_key("abd")


def test_embedding_cache_drops_the_oldest_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBED_CACHE_MAX_VECTORS", 3)
    cache = EmbeddingCache(tmp_path / "emb", "m")
    cache.add(["a", "b"], np.ones((2, 2), dtype=np.float32))
    cache.add(["c", "d"], np.ones((2, 2), dtype=np.float32))
    assert sorted(EmbeddingCache(tmp_path / "emb", "m").lookup(["a", "b", "c", "d"])) == ["b", "c", "d"]


def test_only_new_texts_are_embedded(tmp_path):
    embedder = CountingEmbedder()
    cache = EmbeddingCache(tmp_path / "emb", embedder.model_name)

    first, reused = embed_with_reuse(["alpha beta", "gamma delta", "alpha beta"], embedder, cache)
    assert (reused, embedder.texts_embedded) == (0, 2)  # the repeated text is embedded once
    assert np.allclose(first[0], first[2])

    second, reused = embed_with_reuse(["gamma delta", "epsilon zeta"], embedder, cache)
    assert (reused, embedder.texts_embedded) == (1, 3)
    assert np.allclose(second[0], first[1])
    assert np.allclose(second, FakeEmbedder().embed_documents(["gamma delta", "epsilon zeta"]))


def test_a_new_commit_reuses_embeddings_of_unchanged_files(tmp_path, fixture_repo):
    embedder = CountingEmbedder()
    reuse = tmp_path / "cache" / "embeddings" / "owner__repo"
    first, _ = load_or_build_index(fixture_repo, tmp_path / "index_a", embedder, 8000, reuse_folder=reuse)
    assert (first.reused_chunks, embedder.texts_embedded) == (0, 6)

    (fixture_repo / "src/app/cart.py").write_text("def cart_total(items):\n    return 0\n", encoding="utf-8")
    second, from_cache = load_or_build_index(fixture_repo, tmp_path / "index_b", embedder, 8000, reuse_folder=reuse)

    assert from_cache is False
    assert second.reused_chunks == 5  # only the changed file is embedded again
    assert embedder.texts_embedded == 7


def test_pipeline_uses_the_repo_wide_embedding_cache(tmp_path, fixture_repo, png_bytes):
    run_pipeline(tmp_path, fixture_repo, png_bytes)
    folder = tmp_path / "cache" / "embeddings" / "owner__repo"
    assert {path.name for path in folder.iterdir()} == {"keys.json", "vectors.npy"}
    assert (tmp_path / "cache" / "index" / f"owner__repo__{FAKE_SHA}" / "meta.json").is_file()


# ------------------------------------------------------ retries and settings
def test_backoff_reports_each_retry():
    notices: list[tuple[int, float]] = []
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RetryableLLMError(429, "slow down")
        return "ok"

    result = call_with_backoff(flaky, base_delay=1.0, sleep=lambda seconds: None, on_retry=lambda *args: notices.append(args))
    assert result == "ok"
    assert [status for status, _ in notices] == [429, 429]
    assert notices[0][1] < notices[1][1]


def test_pipeline_forwards_retry_notices_as_progress(tmp_path, fixture_repo, png_bytes):
    class RetryingLLM(FakeLLM):
        def generate(self, prompt, image_png=None):
            if not self.calls and self.on_retry:
                self.on_retry(429, 6.0)
            return super().generate(prompt, image_png)

    events = []
    pipeline.analyze(
        REPO_URL,
        USER_TEXT,
        png_bytes,
        llm=RetryingLLM([VISION_REPLY, RERANK_REPLY, WRITER_REPLY]),
        github=FakeGitHub(fixture_repo, labels=[]),
        embedder=FakeEmbedder(),
        cache_root=tmp_path / "cache",
        on_progress=lambda stage, event, detail: events.append((stage, event, detail)),
    )
    assert ("vision", "progress", "model API answered 429; retrying in 6s") in events


def settings(**overrides) -> Settings:
    values = dict(
        gemini_api_key="key", gemma_model="m", llm_backend="gemini", github_token="",
        ollama_host="http://localhost:11434", native_json=False, max_chunks=8000,
    )
    values.update(overrides)
    return Settings(**values)


def test_thinking_level_is_optional_and_validated():
    check_llm_settings(settings())  # default: not set
    check_llm_settings(settings(thinking_level="minimal"))
    with pytest.raises(ConfigError, match="BUGLENS_THINKING_LEVEL"):
        check_llm_settings(settings(thinking_level="turbo"))


def test_a_rejected_thinking_level_explains_how_to_fix_it():
    from google.genai import errors

    from buglens.llm.gemini import _translate

    rejected = errors.APIError(400, {"error": {"code": 400, "message": "Thinking level is not supported for this model.", "status": "INVALID_ARGUMENT"}})
    error = _translate(rejected, "some-model", "key")
    assert isinstance(error, ConfigError) and "BUGLENS_THINKING_LEVEL" in str(error)
    assert "Remove BUGLENS_THINKING_LEVEL" in error.hint


def test_vision_prompt_asks_for_file_names_and_identifiers():
    from buglens.prompts import load_prompt

    prompt = load_prompt("vision")
    assert '"file_hints"' in prompt and '"identifiers"' in prompt
    assert "Never guess a file name" in prompt
