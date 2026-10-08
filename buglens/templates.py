"""Read the repo's own conventions: issue template, CONTRIBUTING and README.

Supports classic markdown templates (with YAML front matter) and YAML issue
forms. Issue forms are converted to markdown sections, one `###` heading per
field label, which is how GitHub renders a submitted form.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from buglens import config
from buglens.models import IssueTemplate, RepoContext

DEFAULT_SECTIONS = [
    "Description",
    "Steps to reproduce",
    "Expected behavior",
    "Actual behavior",
    "Environment",
    "Additional context",
]
TEMPLATE_PARENTS = (".github", "", "docs")  # where GitHub looks for ISSUE_TEMPLATE
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*:?\s*$")
_FORM_INPUT_TYPES = {"textarea", "input", "dropdown", "checkboxes"}


def default_template() -> IssueTemplate:
    """Fallback used when the repo has no issue template."""
    body = "\n\n".join(f"### {section}" for section in DEFAULT_SECTIONS)
    return IssueTemplate(name="BugLens default", source="default", body_markdown=body, sections=DEFAULT_SECTIONS)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into (front matter dict, body). Bad YAML gives an empty dict."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            try:
                meta = yaml.safe_load("\n".join(lines[1:index])) or {}
            except yaml.YAMLError:
                meta = {}
            body = "\n".join(lines[index + 1:]).strip()
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text


def find_sections(markdown: str) -> list[str]:
    """Section titles: markdown headings and lines that are entirely bold."""
    sections = []
    for line in markdown.splitlines():
        match = _HEADING.match(line.strip()) or _BOLD_LINE.match(line.strip())
        if match:
            sections.append(match.group(1).strip())
    return sections


def _label_list(value: Any) -> list[str]:
    """Template labels may be a YAML list or a comma-separated string."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_markdown_template(text: str, source: str) -> IssueTemplate:
    """Parse a classic `.md` issue template."""
    meta, body = split_front_matter(text)
    return IssueTemplate(
        name=str(meta.get("name") or Path(source).stem),
        source=source,
        body_markdown=body,
        sections=find_sections(body),
        title_prefix=str(meta.get("title") or "").strip(),
        default_labels=_label_list(meta.get("labels")),
    )


def _form_field_to_markdown(field: dict[str, Any]) -> tuple[str, str] | None:
    """Convert one issue-form field to (label, markdown block). Non-input fields give None."""
    attributes = field.get("attributes") or {}
    label = str(attributes.get("label") or "").strip()
    if field.get("type") not in _FORM_INPUT_TYPES or not label:
        return None
    lines = [f"### {label}", ""]
    description = str(attributes.get("description") or "").strip()
    if description:
        lines.append(f"<!-- {description} -->")
    options = attributes.get("options") or []
    if field.get("type") == "dropdown" and options:
        lines.append("<!-- Options: " + ", ".join(str(option) for option in options) + " -->")
    if field.get("type") == "checkboxes":
        for option in options:
            text = option.get("label") if isinstance(option, dict) else option
            lines.append(f"- [ ] {text}")
    return label, "\n".join(lines).rstrip()


def parse_issue_form(text: str, source: str) -> IssueTemplate | None:
    """Parse a `.yml` issue form. Returns None if it is not a usable form."""
    try:
        form = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(form, dict) or not isinstance(form.get("body"), list):
        return None
    blocks, sections = [], []
    for field in form["body"]:
        converted = _form_field_to_markdown(field) if isinstance(field, dict) else None
        if converted:
            sections.append(converted[0])
            blocks.append(converted[1])
    if not blocks:
        return None
    return IssueTemplate(
        name=str(form.get("name") or Path(source).stem),
        source=source,
        body_markdown="\n\n".join(blocks),
        sections=sections,
        title_prefix=str(form.get("title") or "").strip(),
        default_labels=_label_list(form.get("labels")),
    )


def _child(parent: Path, name: str) -> Path | None:
    """Find a direct child by name, ignoring case (GitHub does the same)."""
    if not parent.is_dir():
        return None
    for entry in sorted(parent.iterdir()):
        if entry.name.lower() == name.lower():
            return entry
    return None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _templates_in_folder(folder: Path, repo_dir: Path) -> list[IssueTemplate]:
    """Parse every template file in an ISSUE_TEMPLATE folder (config.yml is not a template)."""
    templates = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if not path.is_file() or path.stem.lower() == "config":
            continue
        source = path.relative_to(repo_dir).as_posix()
        if suffix == ".md":
            templates.append(parse_markdown_template(_read(path), source))
        elif suffix in (".yml", ".yaml"):
            form = parse_issue_form(_read(path), source)
            if form:
                templates.append(form)
    return templates


def find_issue_template(repo_dir: Path) -> IssueTemplate:
    """Pick the repo's bug template: one with 'bug' in its name, else the first, else a default."""
    for parent_name in TEMPLATE_PARENTS:
        parent = _child(repo_dir, parent_name) if parent_name else repo_dir
        if parent is None:
            continue
        folder = _child(parent, "ISSUE_TEMPLATE")
        if folder and folder.is_dir():
            templates = _templates_in_folder(folder, repo_dir)
            for template in templates:
                if "bug" in template.name.lower() or "bug" in Path(template.source).name.lower():
                    return template
            if templates:
                return templates[0]
        single_file = _child(parent, "ISSUE_TEMPLATE.md")
        if single_file and single_file.is_file():
            return parse_markdown_template(_read(single_file), single_file.relative_to(repo_dir).as_posix())
    return default_template()


def _first_existing(repo_dir: Path, names: tuple[str, ...]) -> str:
    """Text of the first of these files found in the root, .github/ or docs/."""
    for parent_name in ("", ".github", "docs"):
        parent = _child(repo_dir, parent_name) if parent_name else repo_dir
        if parent is None:
            continue
        for name in names:
            path = _child(parent, name)
            if path and path.is_file():
                return _read(path)
    return ""


def read_repo_context(repo_dir: Path) -> RepoContext:
    """Collect the template, CONTRIBUTING and the start of the README. Labels are added later."""
    contributing = _first_existing(repo_dir, ("CONTRIBUTING.md", "CONTRIBUTING.rst", "CONTRIBUTING"))
    readme = _first_existing(repo_dir, ("README.md", "README.rst", "README.txt", "README"))
    return RepoContext(
        template=find_issue_template(repo_dir),
        contributing=contributing[: config.CONTRIBUTING_CHARS],
        readme_excerpt=readme[: config.README_CHARS],
    )
