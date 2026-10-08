"""Stage 7: ask Gemma to pick the most likely files among the retrieval candidates."""
from __future__ import annotations

from buglens import config
from buglens.errors import EmptyReply, StructuredOutputError
from buglens.evidence import attach_evidence
from buglens.llm.base import LLMClient
from buglens.llm.json_utils import generate_structured
from buglens.models import CandidateFile, RankedFile, RerankReply
from buglens.prompts import render


def clean_path(path: str) -> str:
    """Normalise a path written by the model: quotes, backslashes and a leading ./ are removed."""
    path = path.strip().strip("`'\"").replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def format_candidates(
    candidates: list[CandidateFile],
    per_file_chars: int = config.RERANK_CHARS_PER_FILE,
    total_chars: int = config.RERANK_TOTAL_CHARS,
) -> str:
    """Render candidates for the prompt. Snippet text is hard-capped per file and in total.

    Every candidate path is always listed, even when the budget for excerpts has run out.
    """
    blocks = []
    remaining = total_chars
    for candidate in candidates:
        lines = [f"FILE: {candidate.path}"]
        if candidate.path_hints:
            quoted = ", ".join(f'"{text}"' for text in candidate.path_hints[:4])
            lines.append(f"FILE NAMES IN THE SCREENSHOT THAT POINT HERE: {quoted}")
        if candidate.matched_strings:
            quoted = ", ".join(f'"{text}"' for text in candidate.matched_strings[:8])
            lines.append(f"SCREENSHOT STRINGS FOUND IN THIS FILE: {quoted}")
        budget = min(per_file_chars, remaining)
        for snippet in candidate.snippets:
            if budget <= 0:
                break
            excerpt = snippet.text[:budget]
            lines.append(f"--- lines {snippet.start_line}-{snippet.end_line} ---")
            lines.append(excerpt)
            budget -= len(excerpt)
            remaining -= len(excerpt)
        if not any(line.startswith("--- lines") for line in lines):
            lines.append("(excerpt omitted to save space)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def keep_known_paths(
    ranked: list[RankedFile],
    candidate_paths: set[str],
    limit: int = config.MAX_RANKED_FILES,
) -> tuple[list[RankedFile], list[str]]:
    """Hallucination guard: keep only files that retrieval actually proposed.

    Returns (kept files in the model's order, dropped paths). Duplicates are removed.
    """
    kept: list[RankedFile] = []
    dropped: list[str] = []
    for item in ranked:
        path = clean_path(item.path)
        if path not in candidate_paths:
            dropped.append(item.path)
        elif all(path != existing.path for existing in kept):
            # Rebuilt from the three fields the model is asked for; evidence is attached by code later.
            kept.append(RankedFile(path=path, reason=item.reason, confidence=item.confidence))
    return kept[:limit], dropped


def retrieval_order_fallback(candidates: list[CandidateFile]) -> list[RankedFile]:
    """Used when the reranker gives nothing usable: keep the retrieval order."""
    return [
        RankedFile(
            path=candidate.path,
            reason="Ranked by search score only; the model did not provide a reason.",
            confidence=max(0.0, min(1.0, candidate.score)),
        )
        for candidate in candidates[: config.MAX_RANKED_FILES]
    ]


def rerank(
    llm: LLMClient,
    user_text: str,
    analysis_text: str,
    candidates: list[CandidateFile],
) -> tuple[list[RankedFile], list[str]]:
    """LLM call 2 of 3. Returns (up to 5 ranked files with their code evidence, warnings)."""
    prompt = render(
        "rerank",
        user_text=user_text,
        analysis_json=analysis_text,
        candidates=format_candidates(candidates),
        max_files=str(config.MAX_RANKED_FILES),
    )
    fallback_note = " Showing the search order instead; confidence values are search scores, not model judgements."
    try:
        reply = generate_structured(llm, prompt, RerankReply)
    except (StructuredOutputError, EmptyReply) as error:
        fallback = attach_evidence(retrieval_order_fallback(candidates), candidates)
        return fallback, [f"The reranker gave no usable reply ({error})." + fallback_note]

    kept, dropped = keep_known_paths(reply.files, {candidate.path for candidate in candidates})
    warnings = []
    if dropped:
        warnings.append(
            f"Dropped {len(dropped)} file(s) named by the model that are not among the candidates: "
            + ", ".join(dropped[:5])
        )
    if not kept:
        warnings.append("The reranker did not select any candidate file." + fallback_note)
        kept = retrieval_order_fallback(candidates)
    return attach_evidence(kept, candidates), warnings
