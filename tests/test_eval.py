"""Eval harness tests: metrics, the runner (with a fake pipeline) and the collect heuristics."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from buglens.errors import RepoNotFound
from buglens.eval import collect as collect_module
from buglens.eval.collect import (
    closes_issue,
    collect,
    find_image_urls,
    is_ground_truth_file,
    parse_issue_url,
    yaml_stanza,
)
from buglens.eval.metrics import first_hit_rank, hit_at_k, reciprocal_rank, summarize
from buglens.eval.run import NO_IMAGE, WITH_IMAGE, load_cases, markdown_summary, run_eval, write_report

EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "eval" / "cases.example.yaml"


# ------------------------------------------------------------------ metrics
def test_first_hit_rank_uses_exact_paths():
    ranked = ["src/a.py", "src/b.py", "src/c.py"]
    assert first_hit_rank(ranked, ["src/b.py"]) == 2
    assert first_hit_rank(ranked, ["src/c.py", "src/a.py"]) == 1  # any ground-truth file counts
    assert first_hit_rank(ranked, ["b.py"]) is None  # no partial matches
    assert first_hit_rank(ranked, ["SRC/b.py"]) is None
    assert first_hit_rank(["./src\\b.py"], ["src/b.py"]) == 1  # only slashes and ./ are normalised
    assert first_hit_rank([], ["src/a.py"]) is None


def test_hit_at_k_and_reciprocal_rank():
    assert hit_at_k(1, 1) and hit_at_k(3, 3) and hit_at_k(3, 5)
    assert not hit_at_k(2, 1) and not hit_at_k(None, 5)
    assert reciprocal_rank(1) == 1.0 and reciprocal_rank(4) == 0.25 and reciprocal_rank(None) == 0.0


def test_summarize():
    summary = summarize([1, 3, None, 5])
    assert summary["cases"] == 4
    assert summary["hit@1"] == pytest.approx(0.25)
    assert summary["hit@3"] == pytest.approx(0.50)
    assert summary["hit@5"] == pytest.approx(0.75)
    assert summary["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 5) / 4)


def test_summarize_with_no_cases():
    assert summarize([]) == {"cases": 0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}


# ------------------------------------------------------------------- runner
def write_cases(tmp_path: Path) -> Path:
    (tmp_path / "shot.png").write_bytes(b"fake image bytes")
    cases = {
        "cases": [
            {
                "repo": "https://github.com/o/one",
                "ref": "aaa",
                "issue_url": "https://github.com/o/one/issues/1",
                "screenshot": "shot.png",
                "text": "first bug",
                "fixed_files": ["src/fix.py"],
            },
            {
                "repo": "https://github.com/o/two",
                "ref": "bbb",
                "screenshot": "missing.png",
                "text": "second bug",
                "fixed_files": ["src/other.py"],
            },
            {"repo": "https://github.com/o/broken", "ref": "ccc", "text": "third bug", "fixed_files": ["x.py"]},
        ]
    }
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(cases), encoding="utf-8")
    return path


def fake_analyze(repo_url, user_text, screenshot=None, *, use_image=True, ref=None, check_similar=True):
    """Stands in for pipeline.analyze: the screenshot moves the right file from rank 3 to rank 1."""
    if repo_url.endswith("broken"):
        raise RepoNotFound("GitHub returned 404.")
    paths = ["src/fix.py", "src/a.py", "src/b.py"] if use_image else ["src/a.py", "src/b.py", "src/fix.py"]
    return SimpleNamespace(
        ranked_files=[SimpleNamespace(path=path) for path in paths],
        candidate_paths=paths + ["src/other.py"],
        llm_model="fake-model",
        llm_backend="fake",
    )


def test_run_eval_scores_both_variants(tmp_path):
    calls = []

    def recording_analyze(repo_url, user_text, screenshot=None, **kwargs):
        calls.append((repo_url, screenshot, kwargs))
        return fake_analyze(repo_url, user_text, screenshot, **kwargs)

    report = run_eval(write_cases(tmp_path), [WITH_IMAGE, NO_IMAGE], analyze=recording_analyze)

    with_image = report["variants"][WITH_IMAGE]
    assert [case["rank"] for case in with_image["cases"]] == [1, None, None]
    assert "Screenshot not found" in with_image["cases"][1]["error"]  # missing file -> error, not a crash
    assert "has no screenshot" in with_image["cases"][2]["error"]
    assert with_image["metrics"]["errors"] == 2
    assert with_image["metrics"]["hit@1"] == pytest.approx(1 / 3)  # errors count as misses

    no_image = report["variants"][NO_IMAGE]
    assert [case["rank"] for case in no_image["cases"]] == [3, None, None]
    assert no_image["cases"][1]["in_candidates"] is True  # retrieval found it, the reranker did not
    assert "RepoNotFound" in no_image["cases"][2]["error"]  # pipeline errors are recorded per case
    assert no_image["metrics"]["hit@1"] == 0.0
    assert no_image["metrics"]["hit@3"] == pytest.approx(1 / 3)
    assert no_image["metrics"]["mrr"] == pytest.approx((1 / 3) / 3)
    assert no_image["metrics"]["candidate_recall"] == pytest.approx(2 / 3)

    # Each case is pinned to its pre-fix ref; the text-only variant never gets a screenshot.
    assert calls[0][2] == {"use_image": True, "ref": "aaa", "check_similar": False}
    assert calls[0][1] == tmp_path / "shot.png"
    assert all(screenshot is None for _, screenshot, kwargs in calls if not kwargs["use_image"])
    assert report["model"] == "fake-model via fake"


def test_report_files_and_markdown(tmp_path):
    report = run_eval(write_cases(tmp_path), [WITH_IMAGE, NO_IMAGE], analyze=fake_analyze, limit=1)
    assert report["case_count"] == 1

    json_path, markdown_path = write_report(report, tmp_path / "results")
    assert json.loads(json_path.read_text(encoding="utf-8"))["variants"][WITH_IMAGE]["metrics"]["hit@1"] == 1.0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == markdown_summary(report)
    assert "| Screenshot + text | 1 | 0 | 1.00 | 1.00 | 1.00 | 1.000 | 1.00 |" in markdown
    assert "| Text only (--no-image) | 1 | 0 | 0.00 | 1.00 | 1.00 | 0.333 | 1.00 |" in markdown
    assert "| https://github.com/o/one/issues/1 | 1 | 3 |" in markdown


def test_load_cases_accepts_a_plain_list_and_rejects_bad_files(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([{"repo": "r", "ref": "s", "text": "t", "fixed_files": ["a.py"]}]), encoding="utf-8")
    assert load_cases(path)[0].fixed_files == ["a.py"]

    path.write_text(yaml.safe_dump([{"repo": "r", "ref": "s", "text": "t", "fixed_files": []}]), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid case"):
        load_cases(path)

    path.write_text("cases: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain a list"):
        load_cases(path)


def test_the_example_file_is_marked_as_an_example_and_cannot_be_run():
    text = EXAMPLE_FILE.read_text(encoding="utf-8")
    assert "EXAMPLE ONLY" in text and "PLACEHOLDER" in text
    assert yaml.safe_load(text)["cases"][0]["repo"] == "https://github.com/OWNER/REPO"
    with pytest.raises(ValueError, match="example file"):
        load_cases(EXAMPLE_FILE)


def test_no_real_eval_data_is_shipped():
    eval_dir = EXAMPLE_FILE.parent
    assert not (eval_dir / "cases.yaml").exists()
    assert not (eval_dir / "results").exists()


# ------------------------------------------------------------------ collect
ISSUE_BODY = """The total is wrong.

![screenshot](https://github.com/user-attachments/assets/1111-2222)
<img width="400" src="https://user-images.githubusercontent.com/1/abc.png">
![tracker](https://evil.example.com/pixel.png)
"""


class FakeCollectClient:
    """Answers GitHub API paths from a dictionary."""

    authenticated = True

    def __init__(self, responses: dict[str, object], linked_prs: list[dict] | None = None) -> None:
        self.responses = responses
        self.linked_prs = linked_prs or []
        self.searches: list[str] = []

    def get_json(self, path: str, params=None):
        if path == "/search/issues":
            self.searches.append(params["q"])
            return self.responses.get(path, {"items": []})
        if params and params.get("page", 1) > 1:
            return []
        return self.responses[path]

    def graphql(self, query: str, variables: dict):
        return {"repository": {"issue": {"closedByPullRequestsReferences": {"nodes": self.linked_prs}}}}


def collect_responses(pr_body: str = "Fixes #7", commits: int = 1) -> dict[str, object]:
    base = "/repos/acme/shop"
    return {
        f"{base}/issues/7": {"title": "Cart total is wrong", "state": "closed", "body": ISSUE_BODY},
        f"{base}/issues/7/timeline": [
            {"event": "cross-referenced", "source": {"issue": {"number": 3}}},  # another issue, not a PR
            {"event": "cross-referenced", "source": {"issue": {"number": 9, "pull_request": {"merged_at": None}}}},
            {"event": "cross-referenced", "source": {"issue": {"number": 12, "pull_request": {"merged_at": "2026-01-01"}}}},
            {"event": "closed", "commit_id": None},
        ],
        f"{base}/pulls/12": {
            "number": 12,
            "title": "Round the cart total",
            "body": pr_body,
            "merged": True,
            "merge_commit_sha": "merge123",
            "commits": commits,
            "base": {"sha": "base456"},
        },
        f"{base}/commits/merge123": {"parents": [{"sha": "parent789"}, {"sha": "branchtip"}]},
        f"{base}/pulls/12/files": [
            {"filename": "src/cart.py", "status": "modified"},
            {"filename": "src/money/rounding.py", "status": "renamed", "previous_filename": "src/rounding.py"},
            {"filename": "src/new_helper.py", "status": "added"},
            {"filename": "tests/test_cart.py", "status": "modified"},
            {"filename": "docs/changelog.md", "status": "modified"},
            {"filename": "package-lock.json", "status": "modified"},
        ],
    }


def test_parse_issue_url():
    repo, number = parse_issue_url("https://github.com/acme/shop/issues/7#issuecomment-1")
    assert (repo.full_name, number) == ("acme/shop", 7)
    with pytest.raises(ValueError):
        parse_issue_url("https://github.com/acme/shop/pull/7")


def test_find_image_urls_only_accepts_github_hosts():
    assert find_image_urls(ISSUE_BODY) == [
        "https://github.com/user-attachments/assets/1111-2222",
        "https://user-images.githubusercontent.com/1/abc.png",
    ]
    assert find_image_urls("") == []


def test_ground_truth_excludes_tests_docs_and_lockfiles():
    assert is_ground_truth_file("src/cart.py")
    assert is_ground_truth_file("config/settings.toml")
    assert not is_ground_truth_file("tests/test_cart.py")
    assert not is_ground_truth_file("docs/changelog.md")
    assert not is_ground_truth_file("README.md")
    assert not is_ground_truth_file("package-lock.json")
    assert not is_ground_truth_file("static/logo.png")


def test_closes_issue_matches_closing_keywords_for_this_issue_only():
    assert closes_issue("This PR fixes #7.", 7)
    assert closes_issue("Closes: acme/shop#7", 7)
    assert closes_issue("resolved #7 and closes #8", 8)
    assert not closes_issue("Related to #7", 7)
    assert not closes_issue("Fixes #70", 7)


def test_collect_builds_a_case(tmp_path, monkeypatch):
    saved = []

    def fake_download(urls, folder, prefix):
        saved.append((urls, prefix))
        return [f"{folder.as_posix()}/{prefix}_1.png"], []

    monkeypatch.setattr(collect_module, "download_images", fake_download)
    found = collect("https://github.com/acme/shop/issues/7", FakeCollectClient(collect_responses()), tmp_path / "shots")

    assert found.ref == "parent789"  # first parent of the merge commit
    assert found.fixed_files == ["src/cart.py", "src/rounding.py"]  # pre-fix path of the renamed file
    assert found.screenshots == [f"{(tmp_path / 'shots').as_posix()}/acme__shop__7_1.png"]
    assert saved[0][1] == "acme__shop__7" and len(saved[0][0]) == 2
    assert any("PR #12 says it closes this issue" in note for note in found.notes)
    assert any("4 changed file(s) left out" in note for note in found.notes)

    stanza = yaml_stanza(found)
    assert stanza.startswith("# REVIEW BY HAND")
    case = yaml.safe_load(stanza)[0]
    assert case == {
        "repo": "https://github.com/acme/shop",
        "ref": "parent789",
        "issue_url": "https://github.com/acme/shop/issues/7",
        "screenshot": found.screenshots[0],
        "text": "Cart total is wrong",
        "fixed_files": ["src/cart.py", "src/rounding.py"],
    }


def test_collect_flags_low_confidence_and_possible_rebase(tmp_path):
    client = FakeCollectClient(collect_responses(pr_body="Rounding cleanup, see #7", commits=3))
    responses = client.responses
    responses["/repos/acme/shop/commits/merge123"] = {"parents": [{"sha": "parent789"}]}
    found = collect("https://github.com/acme/shop/issues/7", client, tmp_path, download=False)

    assert found.ref == "parent789"
    assert found.screenshots == []
    assert any("LOW CONFIDENCE" in note for note in found.notes)
    assert any("rebase-merged" in note for note in found.notes)


def test_collect_prefers_githubs_own_issue_to_pr_link(tmp_path):
    responses = collect_responses(pr_body="No closing keyword here")
    responses["/repos/acme/shop/issues/7/timeline"] = [{"event": "closed", "commit_id": None}]  # no cross-reference
    client = FakeCollectClient(responses, linked_prs=[{"number": 12, "merged": True}, {"number": 99, "merged": False}])
    found = collect("https://github.com/acme/shop/issues/7", client, tmp_path, download=False)

    assert found.ref == "parent789" and found.fixed_files == ["src/cart.py", "src/rounding.py"]
    assert any("PR #12 is linked on GitHub" in note for note in found.notes)
    assert client.searches == []  # the search fallback was not needed


def test_collect_falls_back_to_searching_for_closing_keywords(tmp_path):
    responses = collect_responses()
    responses["/repos/acme/shop/issues/7/timeline"] = [{"event": "closed", "commit_id": None}]
    responses["/search/issues"] = {
        "items": [
            {"number": 12, "title": "Round the cart total", "body": "Fixes #7"},
            {"number": 30, "title": "Unrelated", "body": "See also #7"},
        ]
    }
    client = FakeCollectClient(responses)
    found = collect("https://github.com/acme/shop/issues/7", client, tmp_path, download=False)

    assert found.ref == "parent789"
    assert any("PR #12 was found by search" in note for note in found.notes)
    assert client.searches == ["repo:acme/shop is:pr is:merged 7 in:body"]


def test_collect_without_a_merged_pr_leaves_todos(tmp_path):
    responses = collect_responses()
    responses["/repos/acme/shop/issues/7/timeline"] = [{"event": "closed", "commit_id": None}]
    found = collect("https://github.com/acme/shop/issues/7", FakeCollectClient(responses), tmp_path, download=False)

    assert found.ref == "" and found.fixed_files == []
    case = yaml.safe_load(yaml_stanza(found))[0]
    assert case["ref"] == "TODO" and case["fixed_files"] == ["TODO"]
    assert "No merged pull request that closes this issue was found" in yaml_stanza(found)
