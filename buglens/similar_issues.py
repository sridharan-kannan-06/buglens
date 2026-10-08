"""Look for existing issues that read like the new report, so duplicates are noticed before filing.

Issues are fetched through the GitHub API (the most recently updated ones plus
a keyword search), then ranked locally with the same embedding model that is
used for code search. No LLM call is made, and issue text is never put into a prompt.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from buglens import config
from buglens.embeddings import Embedder
from buglens.errors import BugLensError
from buglens.github_client import GitHubClient
from buglens.models import RepoRef, SimilarIssue

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")
_COMMON_WORDS = {
    "this", "that", "with", "when", "from", "have", "does", "what", "where", "which", "there", "their",
    "about", "after", "before", "should", "would", "could", "being", "into", "than", "then", "them",
    "properly", "correctly", "issue", "error", "problem", "work", "works", "working", "doesn", "cannot",
}


def search_keywords(text: str, limit: int = config.SEARCH_KEYWORDS) -> list[str]:
    """The longest distinct uncommon words of a text. Longer words tend to be more specific."""
    words: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word not in _COMMON_WORDS and word not in words:
            words.append(word)
    return sorted(words, key=lambda word: (-len(word), words.index(word)))[:limit]


def issue_text(issue: dict[str, Any]) -> str:
    """Title plus the start of the body: the part that is compared."""
    body = str(issue.get("body") or "")[: config.SIMILAR_ISSUE_BODY_CHARS]
    return f"{issue.get('title') or ''}\n{body}".strip()


def fetch_issue_pool(github: GitHubClient, repo: RepoRef, query_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Recent issues plus keyword-search hits, without duplicates. Returns (issues, warnings)."""
    warnings: list[str] = []
    pool: dict[int, dict[str, Any]] = {}
    keywords = search_keywords(query_text)
    fetchers = [("recent issues", lambda: github.list_issues(repo))]
    if keywords:
        fetchers.append(("issue search", lambda: github.search_issues(repo, keywords)))
    for name, fetch in fetchers:
        try:
            for issue in fetch():
                if isinstance(issue.get("number"), int):
                    pool.setdefault(issue["number"], issue)
        except BugLensError as error:
            warnings.append(f"Could not fetch {name} for the similar-issue check ({error}).")
    return list(pool.values()), warnings


def rank_similar(
    embedder: Embedder,
    query_text: str,
    issues: list[dict[str, Any]],
    top: int | None = None,
    min_score: float | None = None,
) -> list[SimilarIssue]:
    """Embed the report and every issue, and keep the most similar issues above the threshold."""
    if not issues:
        return []
    top = config.SIMILAR_ISSUES_TOP if top is None else top
    min_score = config.SIMILAR_ISSUE_MIN_SCORE if min_score is None else min_score
    vectors = embedder.embed_documents([query_text] + [issue_text(issue) for issue in issues])
    scores = vectors[1:] @ vectors[0]
    ranked = []
    for row in np.argsort(-scores)[:top]:
        if float(scores[row]) < min_score:
            break
        issue = issues[int(row)]
        ranked.append(
            SimilarIssue(
                number=issue["number"],
                title=str(issue.get("title") or ""),
                url=str(issue.get("html_url") or ""),
                state=str(issue.get("state") or ""),
                similarity=round(float(scores[row]), 3),
            )
        )
    return ranked


def find_similar_issues(
    github: GitHubClient,
    embedder: Embedder,
    repo: RepoRef,
    query_text: str,
) -> tuple[list[SimilarIssue], list[str]]:
    """Return (similar issues, warnings). Never raises for GitHub problems: the check is optional."""
    issues, warnings = fetch_issue_pool(github, repo, query_text)
    return rank_similar(embedder, query_text, issues), warnings
