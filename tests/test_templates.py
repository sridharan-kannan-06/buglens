from __future__ import annotations

from buglens.templates import (
    DEFAULT_SECTIONS,
    find_issue_template,
    parse_issue_form,
    parse_markdown_template,
    read_repo_context,
)

ISSUE_FORM = """
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us what you expected to happen.
    validations:
      required: true
  - type: dropdown
    id: browsers
    attributes:
      label: Browser
      options:
        - Firefox
        - Chrome
  - type: input
    id: version
    attributes:
      label: Version
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      options:
        - label: I agree to follow the Code of Conduct
"""


def test_markdown_template_front_matter_and_bold_sections(fixture_repo):
    text = (fixture_repo / ".github/ISSUE_TEMPLATE/bug_report.md").read_text(encoding="utf-8")
    template = parse_markdown_template(text, ".github/ISSUE_TEMPLATE/bug_report.md")
    assert template.name == "Bug report"
    assert template.title_prefix == "[Bug]:"
    assert template.default_labels == ["bug", "needs-triage"]
    assert template.sections == ["Describe the bug", "To Reproduce", "Expected behavior"]
    assert "---" not in template.body_markdown
    assert template.body_markdown.startswith("**Describe the bug**")


def test_markdown_template_with_headings_and_no_front_matter():
    template = parse_markdown_template("## Summary\n\ntext\n\n### Steps ###\n", "docs/ISSUE_TEMPLATE.md")
    assert template.name == "ISSUE_TEMPLATE"
    assert template.sections == ["Summary", "Steps"]


def test_issue_form_becomes_markdown_sections():
    template = parse_issue_form(ISSUE_FORM, ".github/ISSUE_TEMPLATE/bug.yml")
    assert template is not None
    assert template.name == "Bug Report"
    assert template.sections == ["What happened?", "Browser", "Version", "Code of Conduct"]
    assert template.default_labels == ["bug", "triage"]
    assert template.title_prefix == "[Bug]:"
    body = template.body_markdown
    assert "### What happened?" in body
    assert "<!-- Also tell us what you expected to happen. -->" in body
    assert "<!-- Options: Firefox, Chrome -->" in body
    assert "- [ ] I agree to follow the Code of Conduct" in body
    assert "Thanks for taking the time" not in body  # markdown-only fields are instructions, not sections


def test_issue_form_rejects_non_forms():
    assert parse_issue_form("blank_issues_enabled: false\n", "config.yml") is None
    assert parse_issue_form("body: [", "broken.yml") is None


def test_find_template_prefers_bug_and_ignores_config(fixture_repo):
    template = find_issue_template(fixture_repo)
    assert template.name == "Bug report"
    assert template.source == ".github/ISSUE_TEMPLATE/bug_report.md"


def test_find_template_falls_back_to_first(fixture_repo):
    (fixture_repo / ".github/ISSUE_TEMPLATE/bug_report.md").unlink()
    assert find_issue_template(fixture_repo).name == "Feature request"


def test_find_template_reads_yaml_forms(fixture_repo):
    (fixture_repo / ".github/ISSUE_TEMPLATE/bug_report.md").unlink()
    (fixture_repo / ".github/ISSUE_TEMPLATE/1-bug.yml").write_text(ISSUE_FORM, encoding="utf-8")
    template = find_issue_template(fixture_repo)
    assert template.name == "Bug Report"
    assert template.source.endswith("1-bug.yml")


def test_default_template_when_repo_has_none(tmp_path):
    template = find_issue_template(tmp_path)
    assert template.source == "default"
    assert template.sections == DEFAULT_SECTIONS == [
        "Description",
        "Steps to reproduce",
        "Expected behavior",
        "Actual behavior",
        "Environment",
        "Additional context",
    ]
    assert "### Steps to reproduce" in template.body_markdown


def test_read_repo_context(fixture_repo):
    context = read_repo_context(fixture_repo)
    assert context.template.name == "Bug report"
    assert "run the tests" in context.contributing
    assert context.readme_excerpt.startswith("# Fixture shop")
    assert context.labels == [] and context.labels_available is False


def test_readme_excerpt_is_capped(tmp_path):
    (tmp_path / "README.md").write_text("x" * 10000, encoding="utf-8")
    assert len(read_repo_context(tmp_path).readme_excerpt) == 3000
