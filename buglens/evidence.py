"""Attach a code excerpt to every suspected file.

The reranker gives a reason in words. The evidence is the part BugLens can
prove: the lines that search actually matched, with the on-screen strings and
file names that led there. It is built by code from the retrieval results, so
the model cannot make it up.
"""
from __future__ import annotations

from buglens import config
from buglens.models import CandidateFile, Chunk, Evidence, RankedFile


def focus_line_in(chunk: Chunk, needles: list[str]) -> int | None:
    """Number of the line in the chunk that contains one of the strings (ignoring case).

    Longer strings are tried first: "Invalid username or password" says more
    about where to look than "Username".
    """
    lines = [line.lower() for line in chunk.text.splitlines()]
    for needle in sorted({needle.lower().strip() for needle in needles}, key=lambda text: (-len(text), text)):
        if not needle:
            continue
        for offset, line in enumerate(lines):
            if needle in line:
                return chunk.start_line + offset
    return None


def build_evidence(candidate: CandidateFile) -> Evidence | None:
    """Pick the most telling excerpt of a candidate file.

    Preference: a line named in a stack trace, then a line holding an on-screen
    string, then simply the start of the best-matching chunk.
    """
    if not candidate.snippets:
        return None
    chosen, focus = candidate.snippets[0], None
    for snippet in candidate.snippets:
        named = [line for line in candidate.hint_lines if snippet.start_line <= line <= snippet.end_line]
        line = named[0] if named else focus_line_in(snippet, candidate.matched_strings)
        if line is not None:
            chosen, focus = snippet, line
            break

    if focus is not None:
        first = max(chosen.start_line, focus - config.EVIDENCE_CONTEXT_LINES)
        last = min(chosen.end_line, focus + config.EVIDENCE_CONTEXT_LINES)
    else:
        first = chosen.start_line
        last = min(chosen.end_line, first + config.EVIDENCE_MAX_LINES - 1)
    lines = chosen.text.splitlines()[first - chosen.start_line: last - chosen.start_line + 1]
    return Evidence(
        start_line=first,
        end_line=last,
        focus_line=focus,
        snippet="\n".join(lines),
        matched_strings=sorted(candidate.matched_strings, key=lambda text: (-len(text), text))[:8],  # most specific first
        path_hints=candidate.path_hints[:4],
    )


def attach_evidence(ranked: list[RankedFile], candidates: list[CandidateFile]) -> list[RankedFile]:
    """Return the ranked files with the evidence of their retrieval candidate attached."""
    by_path = {candidate.path: candidate for candidate in candidates}
    return [
        item.model_copy(update={"evidence": build_evidence(by_path[item.path]) if item.path in by_path else None})
        for item in ranked
    ]
