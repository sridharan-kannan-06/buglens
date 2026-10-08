"""Turn an AnalysisResult into markdown files, a console summary and a prefilled GitHub link."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from buglens import config
from buglens.models import AnalysisResult

DISCLAIMER = "Suggestions only. Verify before filing."
STANDARD_CAVEAT = "The file list is a suggestion produced by search and a language model. It has not been verified."


def new_issue_url(result: AnalysisResult) -> tuple[str, bool]:
    """Link to GitHub's "new issue" form with the draft filled in. Returns (url, body_included).

    Nothing is filed by opening the link: the user still has to press Submit on GitHub.
    If the link would be too long, the body is left out and has to be pasted.
    """
    base = f"{result.repo_url}/issues/new"
    fields = {"title": result.issue.title}
    if result.issue.labels:
        fields["labels"] = ",".join(result.issue.labels)
    with_body = f"{base}?{urlencode({**fields, 'body': result.issue.body_markdown})}"
    if len(with_body) <= config.MAX_PREFILL_URL_CHARS:
        return with_body, True
    return f"{base}?{urlencode(fields)}", False


def issue_markdown(result: AnalysisResult) -> str:
    """The issue draft as one markdown document (title, labels, body)."""
    labels = ", ".join(f"`{label}`" for label in result.issue.labels) or "none"
    return (
        f"<!-- BugLens draft. {DISCLAIMER} -->\n"
        f"# {result.issue.title}\n\n"
        f"**Suggested labels:** {labels}\n\n"
        f"{result.issue.body_markdown}\n"
    )


def _table_cell(text: str) -> str:
    """Make free text safe inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _caveats(result: AnalysisResult) -> list[str]:
    """The model's caveats plus the standard one, unless the model already said the same."""
    caveats = list(result.brief.caveats)
    already_said = any("suggestion" in caveat.lower() and "file" in caveat.lower() for caveat in caveats)
    return caveats if already_said else caveats + [STANDARD_CAVEAT]


def brief_markdown(result: AnalysisResult) -> str:
    """The contributor brief, including suspected files, code evidence and similar issues."""
    lines = [
        f"# Contributor brief: {result.issue.title}",
        "",
        f"> {DISCLAIMER}",
        "",
        f"Repository: [{result.repo}]({result.repo_url}) at commit `{result.sha[:12]}`",
        "",
        "## Summary",
        "",
        result.brief.summary,
        "",
        "## Suspected files",
        "",
        "| # | File | Lines | Confidence | Why |",
        "| - | ---- | ----- | ---------- | --- |",
    ]
    for rank, item in enumerate(result.ranked_files, start=1):
        where = "" if item.evidence is None else f"{item.evidence.start_line}-{item.evidence.end_line}"
        link = f"[{item.path}]({result.evidence_url(item)})"
        lines.append(f"| {rank} | {link} | {where} | {item.confidence:.2f} | {_table_cell(item.reason)} |")

    lines += ["", "## Where to start", ""]
    start_points = [f"- `{point.path}`: {point.what_to_look_for}" for point in result.brief.where_to_start]
    lines += start_points or ["- (no starting point suggested; begin with the suspected files above)"]
    lines += ["", "## How to verify", ""]
    steps = [f"{number}. {step}" for number, step in enumerate(result.brief.how_to_verify, start=1)]
    lines += steps or ["1. [please confirm]"]

    if result.similar_issues:
        lines += ["", "## Similar existing issues", "", "Check these before filing, in case the bug is already reported.", ""]
        lines += [
            f"- [#{issue.number}]({issue.url}) {_table_cell(issue.title)} ({issue.state}, similarity {issue.similarity:.2f})"
            for issue in result.similar_issues
        ]

    lines += ["", "## Caveats", ""]
    lines += [f"- {caveat}" for caveat in _caveats(result)]
    return "\n".join(lines) + "\n"


def summary_text(result: AnalysisResult) -> str:
    """Readable plain-text summary for the terminal."""
    lines = [
        f"Repository : {result.repo} @ {result.sha[:12]} ({result.ref})",
        f"Model      : {result.llm_model} via {result.llm_backend}" + ("" if result.used_image else " (text only)"),
        f"Template   : {result.template_name}",
        "",
        f"Issue title: {result.issue.title}",
        f"Labels     : {', '.join(result.issue.labels) or 'none'}",
        "",
        "Suspected files:",
    ]
    for rank, item in enumerate(result.ranked_files, start=1):
        lines.append(f"  {rank}. {item.path}  (confidence {item.confidence:.2f})")
        lines.append(f"     {item.reason}")
        if item.evidence is not None:
            evidence = item.evidence
            details = [f"line {evidence.focus_line}" if evidence.focus_line else f"lines {evidence.start_line}-{evidence.end_line}"]
            if evidence.matched_strings:
                details.append("on screen: " + ", ".join(f'"{text}"' for text in evidence.matched_strings[:3]))
            if evidence.path_hints:
                details.append("named on screen: " + ", ".join(evidence.path_hints[:2]))
            lines.append("     evidence: " + "; ".join(details))
    lines += ["", "Brief:", f"  {result.brief.summary}", ""]
    if result.similar_issues:
        lines.append("Similar existing issues (check before filing):")
        lines += [
            f"  #{issue.number} [{issue.state}] {issue.title}  (similarity {issue.similarity:.2f})"
            for issue in result.similar_issues
        ]
        lines.append("")
    lines.append("Timings    : " + ", ".join(f"{stage} {seconds:.1f}s" for stage, seconds in result.timings.items()))
    if result.warnings:
        lines.append("Warnings:")
        lines += [f"  - {warning}" for warning in result.warnings]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def save_result(result: AnalysisResult, out_root: Path) -> Path:
    """Write issue.md, brief.md and result.json into a new timestamped folder and return it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = out_root / f"{stamp}_{result.repo.replace('/', '_')}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "issue.md").write_text(issue_markdown(result), encoding="utf-8")
    (folder / "brief.md").write_text(brief_markdown(result), encoding="utf-8")
    (folder / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return folder
