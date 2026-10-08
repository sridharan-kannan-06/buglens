"""Ranking metrics for file localisation.

A case is a *hit at k* when any ground-truth file (a file changed by the real
fix) appears among the first k suspected files. Paths must match exactly.
MRR is the mean of 1/rank of the first ground-truth file (0 when none is found).
"""
from __future__ import annotations

from collections.abc import Sequence

K_VALUES = (1, 3, 5)


def normalize_path(path: str) -> str:
    """Use forward slashes and drop a leading ./ so equal paths compare equal."""
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def first_hit_rank(ranked_paths: Sequence[str], fixed_files: Sequence[str]) -> int | None:
    """1-based position of the first ground-truth file in the ranking, or None."""
    truth = {normalize_path(path) for path in fixed_files}
    for position, path in enumerate(ranked_paths, start=1):
        if normalize_path(path) in truth:
            return position
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarize(ranks: Sequence[int | None]) -> dict[str, float]:
    """hit@1, hit@3, hit@5 and MRR over all cases. A failed case counts as a miss."""
    total = len(ranks)
    summary: dict[str, float] = {"cases": total}
    for k in K_VALUES:
        summary[f"hit@{k}"] = sum(hit_at_k(rank, k) for rank in ranks) / total if total else 0.0
    summary["mrr"] = sum(reciprocal_rank(rank) for rank in ranks) / total if total else 0.0
    return summary
