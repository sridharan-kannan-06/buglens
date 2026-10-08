"""Run BugLens over a file of real, already-fixed bugs and score the suspected files.

    python -m buglens.eval run eval/cases.yaml            # with-image and text-only variants
    python -m buglens.eval run eval/cases.yaml --no-image # text-only variant only

Results are written to eval/results/ as JSON plus a markdown summary. Nothing
here invents numbers: every figure comes from an actual pipeline run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from buglens import pipeline
from buglens.errors import BugLensError
from buglens.eval.metrics import first_hit_rank, summarize

WITH_IMAGE = "with_image"
NO_IMAGE = "no_image"
VARIANT_TITLES = {WITH_IMAGE: "Screenshot + text", NO_IMAGE: "Text only (--no-image)"}


class EvalCase(BaseModel):
    """One real bug: the repo before the fix, what the user saw, and the files the fix changed."""

    repo: str
    ref: str
    issue_url: str = ""
    screenshot: str = ""
    text: str
    fixed_files: list[str] = Field(min_length=1)


class CaseOutcome(BaseModel):
    """What happened for one case in one variant."""

    repo: str
    ref: str
    issue_url: str
    fixed_files: list[str]
    ranked_paths: list[str] = Field(default_factory=list)
    rank: int | None = None  # position of the first ground-truth file among the suspected files
    in_candidates: bool = False  # was a ground-truth file among the retrieval candidates?
    model: str = ""
    error: str = ""
    seconds: float = 0.0


def load_cases(path: Path) -> list[EvalCase]:
    """Read a cases file: either a YAML list, or a mapping with a `cases:` list."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if data.get("example"):
            raise ValueError(
                f"{path} is the example file with placeholder values. "
                "Create eval/cases.yaml with real cases (see RUN_AND_TEST.md)."
            )
        data = data.get("cases")
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} does not contain a list of cases.")
    try:
        return [EvalCase.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"{path} has an invalid case:\n{exc}") from exc


def find_screenshot(case: EvalCase, cases_file: Path) -> Path | None:
    """Screenshot paths are relative to the working directory, or else to the cases file."""
    if not case.screenshot:
        return None
    for candidate in (Path(case.screenshot), cases_file.parent / case.screenshot):
        if candidate.is_file():
            return candidate
    return None


def run_case(case: EvalCase, use_image: bool, cases_file: Path, analyze: Callable[..., Any]) -> CaseOutcome:
    """Run the pipeline for one case pinned to its pre-fix ref, and score it."""
    outcome = CaseOutcome(repo=case.repo, ref=case.ref, issue_url=case.issue_url, fixed_files=case.fixed_files)
    started = time.perf_counter()
    try:
        screenshot = find_screenshot(case, cases_file) if use_image else None
        if use_image and screenshot is None:
            reason = f"Screenshot not found: '{case.screenshot}'" if case.screenshot else "This case has no screenshot."
            raise BugLensError(reason)
        # The similar-issue check is skipped: it does not affect the ranking that is being scored.
        result = analyze(case.repo, case.text, screenshot, use_image=use_image, ref=case.ref, check_similar=False)
        outcome.ranked_paths = [item.path for item in result.ranked_files]
        outcome.rank = first_hit_rank(outcome.ranked_paths, case.fixed_files)
        outcome.in_candidates = first_hit_rank(result.candidate_paths, case.fixed_files) is not None
        outcome.model = f"{result.llm_model} via {result.llm_backend}"
    except BugLensError as error:
        outcome.error = f"{type(error).__name__}: {error}"
    outcome.seconds = round(time.perf_counter() - started, 2)
    return outcome


def run_variant(
    cases: list[EvalCase],
    variant: str,
    cases_file: Path,
    analyze: Callable[..., Any],
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run every case in one variant and compute the metrics."""
    outcomes = []
    for number, case in enumerate(cases, start=1):
        print(f"[{variant}] case {number}/{len(cases)}: {case.issue_url or case.repo}", file=sys.stderr, flush=True)
        outcomes.append(run_case(case, variant == WITH_IMAGE, cases_file, analyze))
        if sleep_seconds and number < len(cases):
            time.sleep(sleep_seconds)
    metrics = summarize([outcome.rank for outcome in outcomes])
    metrics["errors"] = sum(bool(outcome.error) for outcome in outcomes)
    metrics["candidate_recall"] = (
        sum(outcome.in_candidates for outcome in outcomes) / len(outcomes) if outcomes else 0.0
    )
    return {"metrics": metrics, "cases": [outcome.model_dump() for outcome in outcomes]}


def markdown_summary(report: dict[str, Any]) -> str:
    """Human-readable summary of a report, ready to paste into the README."""
    lines = [
        "# BugLens evaluation",
        "",
        f"- Date: {report['created_at']}",
        f"- Cases file: `{report['cases_file']}` ({report['case_count']} cases)",
        f"- Model: {report['model'] or 'unknown (no case finished)'}",
        "- A hit at k means: a file changed by the real fix is among the top k suspected files (exact path).",
        "- Cases that ended in an error count as misses.",
        "",
        "| Variant | Cases | Errors | hit@1 | hit@3 | hit@5 | MRR | Fix among the 12 candidates |",
        "| ------- | ----- | ------ | ----- | ----- | ----- | --- | --------------------------- |",
    ]
    for variant, data in report["variants"].items():
        metrics = data["metrics"]
        lines.append(
            f"| {VARIANT_TITLES[variant]} | {metrics['cases']} | {metrics['errors']} | "
            f"{metrics['hit@1']:.2f} | {metrics['hit@3']:.2f} | {metrics['hit@5']:.2f} | "
            f"{metrics['mrr']:.3f} | {metrics['candidate_recall']:.2f} |"
        )
    lines += ["", "## Per case", ""]
    variants = list(report["variants"])
    lines.append("| Case | " + " | ".join(f"Rank: {VARIANT_TITLES[variant]}" for variant in variants) + " |")
    lines.append("| ---- | " + " | ".join("----" for _ in variants) + " |")
    for position in range(report["case_count"]):
        first = report["variants"][variants[0]]["cases"][position]
        cells = []
        for variant in variants:
            outcome = report["variants"][variant]["cases"][position]
            cells.append("error" if outcome["error"] else str(outcome["rank"] or "miss"))
        lines.append(f"| {first['issue_url'] or first['repo']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write <timestamp>.json and <timestamp>.md into the results folder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"eval_{stamp}.json"
    markdown_path = out_dir / f"eval_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_summary(report), encoding="utf-8")
    return json_path, markdown_path


def run_eval(
    cases_file: Path,
    variants: list[str],
    analyze: Callable[..., Any] = pipeline.analyze,
    limit: int | None = None,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    """Load the cases, run the requested variants and return the report dictionary."""
    cases = load_cases(cases_file)[:limit]
    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cases_file": cases_file.as_posix(),
        "case_count": len(cases),
        "model": "",
        "variants": {},
    }
    for variant in variants:
        report["variants"][variant] = run_variant(cases, variant, cases_file, analyze, sleep_seconds)
    models = {case["model"] for data in report["variants"].values() for case in data["cases"] if case["model"]}
    report["model"] = ", ".join(sorted(models))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m buglens.eval run", description=__doc__.splitlines()[0])
    parser.add_argument("cases", type=Path, help="YAML file with eval cases (see eval/cases.example.yaml).")
    parser.add_argument("--no-image", action="store_true", help="Run only the text-only variant.")
    parser.add_argument("--limit", type=int, help="Only run the first N cases.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between cases (rate limits).")
    parser.add_argument("--out", type=Path, default=Path("eval/results"), help="Folder for the result files.")
    args = parser.parse_args(argv)

    variants = [NO_IMAGE] if args.no_image else [WITH_IMAGE, NO_IMAGE]
    try:
        report = run_eval(args.cases, variants, limit=args.limit, sleep_seconds=args.sleep)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    json_path, markdown_path = write_report(report, args.out)
    print(markdown_summary(report))
    print(f"Written: {json_path} and {markdown_path}")
    return 0
