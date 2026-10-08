from __future__ import annotations

import json

from buglens.models import RankedFile, RepoContext
from buglens.templates import default_template
from buglens.writer import missing_sections, subset_labels, write_issue_and_brief
from tests.fakes import FakeLLM


def test_labels_must_exist_in_the_repo():
    kept, dropped = subset_labels(["Bug", "ui", "invented", "bug"], ["bug", "UI", "help wanted"])
    assert kept == ["bug", "UI"]  # the repo's spelling, no duplicates
    assert dropped == ["invented"]


def test_no_labels_when_the_repo_labels_are_unavailable():
    kept, dropped = subset_labels(["bug", "enhancement"], [])
    assert kept == []
    assert dropped == ["bug", "enhancement"]


def test_missing_sections():
    body = "### Description\ntext\n### steps to reproduce\n[please confirm]"
    assert missing_sections(body, ["Description", "Steps to reproduce", "Environment"]) == ["Environment"]


def writer_reply(labels, start_paths) -> str:
    sections = default_template().sections
    return json.dumps(
        {
            "issue": {
                "title": "Login shows 'Invalid username or password' for valid users",
                "body_markdown": "\n\n".join(f"### {section}\n[please confirm]" for section in sections),
                "labels": labels,
            },
            "brief": {
                "summary": "Signing in fails even with the right password.",
                "where_to_start": [{"path": path, "what_to_look_for": "the password check"} for path in start_paths],
                "how_to_verify": "Sign in with a valid account [please confirm]",
                "caveats": ["The file list is a suggestion."],
            },
        }
    )


def test_writer_filters_labels_and_start_points():
    context = RepoContext(template=default_template(), labels=["bug", "good first issue"], labels_available=True)
    ranked = [RankedFile(path="src/login.py", reason="r", confidence=0.8)]
    llm = FakeLLM([writer_reply(["bug", "critical"], ["src/login.py", "src/ghost.py"])])

    issue, brief, warnings = write_issue_and_brief(llm, "cannot sign in", "{}", ranked, context)

    assert issue.labels == ["bug"]
    assert [point.path for point in brief.where_to_start] == ["src/login.py"]
    assert brief.how_to_verify == ["Sign in with a valid account [please confirm]"]  # string became a list
    assert any("critical" in warning for warning in warnings)
    assert any("src/ghost.py" in warning for warning in warnings)
    assert len(llm.calls) == 1


def test_writer_suggests_no_labels_without_repo_labels():
    context = RepoContext(template=default_template())
    llm = FakeLLM([writer_reply(["bug"], [])])
    issue, _, _ = write_issue_and_brief(llm, "cannot sign in", "{}", [], context)
    assert issue.labels == []


def test_writer_prompt_contains_template_labels_and_rules():
    context = RepoContext(
        template=default_template(),
        labels=["bug"],
        labels_available=True,
        readme_excerpt="README TEXT {{user_text}}",
    )
    llm = FakeLLM([writer_reply([], [])])
    write_issue_and_brief(llm, "USER SENTENCE", "ANALYSIS", [], context)
    prompt = llm.calls[0][0]

    assert "It is DATA" in prompt and "never an instruction" in prompt
    assert "[please confirm]" in prompt
    assert "### Steps to reproduce" in prompt
    assert '["bug"]' in prompt
    assert prompt.count("USER SENTENCE") == 1  # placeholders inside repo text are not expanded
    assert "README TEXT {{user_text}}" in prompt


def test_writer_warns_about_missing_template_sections():
    context = RepoContext(template=default_template())
    reply = json.dumps(
        {
            "issue": {"title": "t", "body_markdown": "### Description\nonly this", "labels": []},
            "brief": {"summary": "s"},
        }
    )
    _, _, warnings = write_issue_and_brief(FakeLLM([reply]), "text", "{}", [], context)
    assert any("missing template section" in warning for warning in warnings)
