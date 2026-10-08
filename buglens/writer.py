"""Stage 8: one Gemma call that writes both the issue draft and the contributor brief."""
from __future__ import annotations

import json

from buglens import config
from buglens.llm.base import LLMClient
from buglens.llm.json_utils import generate_structured
from buglens.models import ContributorBrief, IssueDraft, RankedFile, RepoContext, WriterReply
from buglens.prompts import render
from buglens.rerank import clean_path


def subset_labels(proposed: list[str], allowed: list[str]) -> tuple[list[str], list[str]]:
    """Keep only labels that really exist in the repo.

    Matching ignores case; the repo's own spelling is returned. Returns (kept, dropped).
    With an empty `allowed` list nothing is kept, so BugLens never invents labels.
    """
    real_names = {label.lower(): label for label in allowed}
    kept: list[str] = []
    dropped: list[str] = []
    for label in proposed:
        real = real_names.get(label.strip().lower())
        if real is None:
            dropped.append(label)
        elif real not in kept:
            kept.append(real)
    return kept, dropped


def missing_sections(body_markdown: str, sections: list[str]) -> list[str]:
    """Template sections whose title does not appear in the issue body."""
    lowered = body_markdown.lower()
    return [section for section in sections if section.lower() not in lowered]


def keep_ranked_start_points(brief: ContributorBrief, ranked_paths: set[str]) -> tuple[ContributorBrief, list[str]]:
    """Same hallucination guard as the reranker: the brief may only point at ranked files."""
    kept, dropped = [], []
    for point in brief.where_to_start:
        path = clean_path(point.path)
        if path in ranked_paths:
            kept.append(point.model_copy(update={"path": path}))
        else:
            dropped.append(point.path)
    return brief.model_copy(update={"where_to_start": kept}), dropped


def write_issue_and_brief(
    llm: LLMClient,
    user_text: str,
    analysis_text: str,
    ranked_files: list[RankedFile],
    context: RepoContext,
) -> tuple[IssueDraft, ContributorBrief, list[str]]:
    """LLM call 3 of 3. Returns (issue, brief, warnings)."""
    template = context.template
    prompt = render(
        "writer",
        user_text=user_text,
        analysis_json=analysis_text,
        ranked_json=json.dumps(
            [item.model_dump(include={"path", "reason", "confidence"}) for item in ranked_files], indent=2
        ),
        template_name=template.name,
        title_prefix=template.title_prefix,
        template_body=template.body_markdown[: config.TEMPLATE_CHARS],
        labels_json=json.dumps(context.labels),
        default_labels_json=json.dumps(template.default_labels),
        contributing=context.contributing or "(none found)",
        readme=context.readme_excerpt or "(none found)",
    )
    reply = generate_structured(llm, prompt, WriterReply)
    warnings: list[str] = []

    labels, dropped_labels = subset_labels(reply.issue.labels, context.labels)
    if dropped_labels:
        warnings.append("Dropped label(s) that do not exist in the repository: " + ", ".join(dropped_labels[:8]))
    issue = reply.issue.model_copy(update={"labels": labels})

    missing = missing_sections(issue.body_markdown, template.sections)
    if missing:
        warnings.append("The issue draft is missing template section(s): " + ", ".join(missing[:8]))

    brief, dropped_paths = keep_ranked_start_points(reply.brief, {item.path for item in ranked_files})
    if dropped_paths:
        warnings.append(
            "Dropped 'where to start' path(s) that are not among the suspected files: " + ", ".join(dropped_paths[:5])
        )
    return issue, brief, warnings
