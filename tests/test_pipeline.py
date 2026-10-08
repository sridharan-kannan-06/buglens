"""The full pipeline, offline: FakeLLM, FakeEmbedder and a fixture repo served by FakeGitHub."""
from __future__ import annotations

import json

import pytest

from buglens import pipeline
from buglens.cli import main as cli_main
from buglens.errors import BugLensError, EmptyRetrieval, RepoNotFound, UnsupportedImage
from buglens.report import brief_markdown, issue_markdown, save_result, summary_text
from tests.fakes import FAKE_SHA, FakeEmbedder, FakeGitHub, FakeLLM

REPO_URL = "https://github.com/owner/repo"
USER_TEXT = "Login fails with a valid password"

VISION_REPLY = """```json
{
  "visible_text": ["Sign in", "Username"],
  "ui_components": ["login form"],
  "error_messages": ["Invalid username or password"],
  "apparent_problem": "The login form rejects a valid user.",
  "framework_hints": [],
  "search_queries": ["login view that checks the password", "sign in form template", "password check"]
}
```"""

RERANK_REPLY = json.dumps(
    {
        "files": [
            {"path": "src/app/login.py", "reason": "Returns the error shown in the screenshot.", "confidence": 0.9},
            {"path": "src/app/auth_service.py", "reason": "Invented file.", "confidence": 0.8},
            {"path": "src/app/templates/login.html", "reason": "Renders the form.", "confidence": 0.4},
        ]
    }
)

WRITER_REPLY = json.dumps(
    {
        "issue": {
            "title": "[Bug]: Login rejects valid credentials",
            "body_markdown": (
                "**Describe the bug**\nThe login form shows \"Invalid username or password\".\n\n"
                "**To Reproduce**\n[please confirm]\n\n"
                "**Expected behavior**\n[please confirm]"
            ),
            "labels": ["bug", "priority: critical"],
        },
        "brief": {
            "summary": "Signing in fails even when the password is right.",
            "where_to_start": [
                {"path": "src/app/login.py", "what_to_look_for": "How the password is checked."},
                {"path": "src/app/auth_service.py", "what_to_look_for": "Invented."},
            ],
            "how_to_verify": ["Sign in with a valid account [please confirm]"],
            "caveats": ["The file list is a suggestion."],
        },
    }
)


def run(tmp_path, fixture_repo, replies, **kwargs):
    llm = FakeLLM(replies)
    github = kwargs.pop("github", None) or FakeGitHub(fixture_repo, labels=["bug", "enhancement"])
    embedder = kwargs.pop("embedder", None) or FakeEmbedder()
    result = pipeline.analyze(
        REPO_URL, USER_TEXT, llm=llm, github=github, embedder=embedder, cache_root=tmp_path / "cache", **kwargs
    )
    return result, llm, github, embedder


def test_full_pipeline_with_screenshot(tmp_path, fixture_repo, png_bytes):
    events = []
    result, llm, _, _ = run(
        tmp_path,
        fixture_repo,
        [VISION_REPLY, RERANK_REPLY, WRITER_REPLY],
        screenshot=png_bytes,
        on_progress=lambda stage, event, detail: events.append((stage, event)),
    )

    # Exactly three LLM calls; only the first carries the image.
    assert len(llm.calls) == 3
    assert llm.calls[0][1] is not None and llm.calls[0][1].startswith(b"\x89PNG")
    assert llm.calls[1][1] is None and llm.calls[2][1] is None

    assert (result.repo, result.ref, result.sha) == ("owner/repo", "main", FAKE_SHA)
    assert result.used_image is True
    assert result.template_name == "Bug report"
    assert result.screenshot_analysis.error_messages == ["Invalid username or password"]

    # Retrieval never proposes skipped files.
    assert "src/app/login.py" in result.candidate_paths
    assert not any("node_modules" in path or path.endswith((".png", ".min.js")) for path in result.candidate_paths)
    assert not any(path.startswith(".github/") for path in result.candidate_paths)

    # Hallucination guard: the invented file is gone, order is kept.
    assert [item.path for item in result.ranked_files] == ["src/app/login.py", "src/app/templates/login.html"]
    assert [point.path for point in result.brief.where_to_start] == ["src/app/login.py"]
    assert any("src/app/auth_service.py" in warning for warning in result.warnings)

    # Labels are a subset of the repo's real labels.
    assert result.issue.labels == ["bug"]
    assert any("priority: critical" in warning for warning in result.warnings)
    assert any("GITHUB_TOKEN" in warning for warning in result.warnings)

    assert list(result.timings) == list(pipeline.STAGE_LABELS)
    assert ("vision", "start") in events and ("write", "done") in events
    assert result.file_url("src/app/login.py") == f"{REPO_URL}/blob/{FAKE_SHA}/src/app/login.py"

    # Evidence is attached by code: the excerpt holds the on-screen error, and the link jumps to that line.
    top = result.ranked_files[0]
    assert top.evidence is not None
    assert top.evidence.focus_line == 9 and "Invalid username or password" in top.evidence.snippet
    assert top.evidence.matched_strings[0] == "Invalid username or password"  # most specific string first
    assert result.evidence_url(top) == f"{REPO_URL}/blob/{FAKE_SHA}/src/app/login.py#L9"
    assert (result.files_indexed, result.chunks_indexed) == (6, 6)
    assert result.similar_issues == []

    # The rerank prompt shows the candidates; the writer prompt shows the repo's template and labels.
    assert "FILE: src/app/login.py" in llm.calls[1][0]
    assert "**Describe the bug**" in llm.calls[2][0]
    assert '"evidence"' not in llm.calls[2][0]  # excerpts of untrusted code are not repeated to the writer
    assert '["bug", "enhancement"]' in llm.calls[2][0]


def test_text_only_mode_skips_the_vision_call(tmp_path, fixture_repo):
    result, llm, _, _ = run(tmp_path, fixture_repo, [RERANK_REPLY, WRITER_REPLY], use_image=False)
    assert len(llm.calls) == 2
    assert all(image is None for _, image in llm.calls)
    assert result.used_image is False
    assert result.screenshot_analysis.search_queries == [USER_TEXT]
    assert "No screenshot was analysed" in llm.calls[0][0]
    assert result.ranked_files[0].path == "src/app/login.py"


def test_second_run_reuses_repo_and_index_cache(tmp_path, fixture_repo, png_bytes):
    github = FakeGitHub(fixture_repo, labels=[])
    embedder = FakeEmbedder()
    for _ in range(2):
        run(
            tmp_path,
            fixture_repo,
            [VISION_REPLY, RERANK_REPLY, WRITER_REPLY],
            screenshot=png_bytes,
            github=github,
            embedder=embedder,
        )
    assert github.downloads == 1
    assert embedder.document_calls == 1
    index_folder = tmp_path / "cache" / "index" / f"owner__repo__{FAKE_SHA}"
    assert {path.name for path in index_folder.iterdir()} == {"chunks.json", "vectors.npy", "meta.json"}


def test_labels_unavailable_means_no_labels(tmp_path, fixture_repo, png_bytes):
    result, _, _, _ = run(
        tmp_path,
        fixture_repo,
        [VISION_REPLY, RERANK_REPLY, WRITER_REPLY],
        screenshot=png_bytes,
        github=FakeGitHub(fixture_repo, labels=None),
    )
    assert result.issue.labels == []
    assert any("labels" in warning for warning in result.warnings)


def test_bad_input_fails_before_any_download(tmp_path, fixture_repo, png_bytes):
    github = FakeGitHub(fixture_repo)
    with pytest.raises(BugLensError, match="describe the bug"):
        pipeline.analyze(REPO_URL, "   ", png_bytes, llm=FakeLLM([]), github=github, embedder=FakeEmbedder())
    with pytest.raises(UnsupportedImage):
        pipeline.analyze(REPO_URL, USER_TEXT, b"not an image", llm=FakeLLM([]), github=github, embedder=FakeEmbedder())
    with pytest.raises(UnsupportedImage):
        pipeline.analyze(REPO_URL, USER_TEXT, None, llm=FakeLLM([]), github=github, embedder=FakeEmbedder())
    assert github.downloads == 0


def test_private_or_missing_repo(tmp_path, fixture_repo, png_bytes):
    with pytest.raises(RepoNotFound):
        run(tmp_path, fixture_repo, [], screenshot=png_bytes, github=FakeGitHub(fixture_repo, exists=False))


def test_repo_without_source_files_raises_empty_retrieval(tmp_path, png_bytes):
    empty_repo = tmp_path / "empty"
    (empty_repo / "static").mkdir(parents=True)
    (empty_repo / "static" / "logo.png").write_bytes(b"\x89PNG\x00")
    with pytest.raises(EmptyRetrieval):
        run(tmp_path, empty_repo, [VISION_REPLY], screenshot=png_bytes)


def test_reports_and_saved_files(tmp_path, fixture_repo, png_bytes):
    result, _, _, _ = run(tmp_path, fixture_repo, [VISION_REPLY, RERANK_REPLY, WRITER_REPLY], screenshot=png_bytes)

    issue = issue_markdown(result)
    assert issue.splitlines()[1] == "# [Bug]: Login rejects valid credentials"
    assert "**Suggested labels:** `bug`" in issue and "[please confirm]" in issue

    brief = brief_markdown(result)
    assert "Suggestions only. Verify before filing." in brief
    assert f"[src/app/login.py]({REPO_URL}/blob/{FAKE_SHA}/src/app/login.py#L9)" in brief

    assert "| 1 | [src/app/login.py](" in brief and "#L9) | 3-10 |" in brief
    assert brief.count("The file list is a suggestion") == 1  # the standard caveat is not repeated

    summary = summary_text(result)
    assert "1. src/app/login.py  (confidence 0.90)" in summary
    assert 'evidence: line 9; on screen: "Invalid username or password"' in summary

    folder = save_result(result, tmp_path / "out")
    assert {path.name for path in folder.iterdir()} == {"issue.md", "brief.md", "result.json"}
    saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert saved["sha"] == FAKE_SHA and saved["ranked_files"][0]["path"] == "src/app/login.py"


def test_cli_analyze_writes_files_and_prints_summary(tmp_path, fixture_repo, png_bytes, monkeypatch, capsys):
    result, _, _, _ = run(tmp_path, fixture_repo, [VISION_REPLY, RERANK_REPLY, WRITER_REPLY], screenshot=png_bytes)
    monkeypatch.setattr(pipeline, "analyze", lambda *args, **kwargs: result)
    shot = tmp_path / "shot.png"
    shot.write_bytes(png_bytes)

    code = cli_main(
        ["analyze", "--repo", REPO_URL, "--screenshot", str(shot), "--text", USER_TEXT, "--out", str(tmp_path / "out")]
    )

    assert code == 0
    assert "Suspected files:" in capsys.readouterr().out
    assert len(list((tmp_path / "out").glob("*/result.json"))) == 1


def test_cli_reports_errors_with_a_hint(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RepoNotFound("GitHub returned 404.")

    monkeypatch.setattr(pipeline, "analyze", fail)
    code = cli_main(["analyze", "--repo", REPO_URL, "--text", USER_TEXT, "--no-image"])
    captured = capsys.readouterr()
    assert code == 1
    assert "error: GitHub returned 404." in captured.err and "hint:" in captured.err


def test_cli_requires_a_screenshot_unless_no_image(capsys):
    assert cli_main(["analyze", "--repo", REPO_URL, "--text", USER_TEXT]) == 2
    assert "--screenshot is required" in capsys.readouterr().err
