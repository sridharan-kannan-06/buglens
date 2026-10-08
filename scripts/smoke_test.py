"""End-to-end smoke test against the real GitHub API and the real Gemma model.

    python scripts/smoke_test.py --repo https://github.com/owner/repo --screenshot shot.png --text "what went wrong"

Needs the keys in .env. It prints a summary and a few sanity checks; nothing is written to disk
except the cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `buglens` importable

from buglens import pipeline  # noqa: E402
from buglens.cli import print_progress  # noqa: E402
from buglens.errors import BugLensError  # noqa: E402
from buglens.report import summary_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real BugLens pipeline once and print a summary.")
    parser.add_argument("--repo", required=True, help="Public GitHub repository URL.")
    parser.add_argument("--screenshot", required=True, type=Path, help="Path to a png/jpg/webp screenshot.")
    parser.add_argument("--text", required=True, help="One-line description of the bug.")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        result = pipeline.analyze(args.repo, args.text, args.screenshot, on_progress=print_progress)
    except BugLensError as error:
        print(f"SMOKE TEST FAILED: {error}", file=sys.stderr)
        if error.hint:
            print(f"hint: {error.hint}", file=sys.stderr)
        return 1

    print()
    print(summary_text(result))
    print()
    checks = {
        "screenshot analysis has search queries": bool(result.screenshot_analysis.search_queries),
        "at least one suspected file": bool(result.ranked_files),
        "every suspected file was a retrieval candidate": all(
            item.path in result.candidate_paths for item in result.ranked_files
        ),
        "issue has a title and a body": bool(result.issue.title and result.issue.body_markdown),
        "brief has a summary": bool(result.brief.summary),
    }
    for name, passed in checks.items():
        print(f"[{'pass' if passed else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
