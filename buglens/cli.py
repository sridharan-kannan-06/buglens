"""Command line interface: `python -m buglens analyze ...` and `python -m buglens index ...`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from buglens import pipeline
from buglens.errors import BugLensError
from buglens.report import save_result, summary_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buglens",
        description="Turn a bug screenshot into an issue draft and a list of suspected files. "
        "Suggestions only. Verify before filing.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Analyse one bug and write issue.md, brief.md and result.json.")
    analyze.add_argument("--repo", required=True, help="Public GitHub repository URL.")
    analyze.add_argument("--screenshot", type=Path, help="Path to a png/jpg/webp screenshot.")
    analyze.add_argument("--text", required=True, help="One-line description of the bug.")
    analyze.add_argument("--out", type=Path, default=Path("out"), help="Output folder (default: out/).")
    analyze.add_argument("--no-image", action="store_true", help="Skip vision and use only the text.")
    analyze.add_argument("--json", action="store_true", help="Print the full result as JSON instead of a summary.")

    index = commands.add_parser("index", help="Download and index a repository ahead of time.")
    index.add_argument("--repo", required=True, help="Public GitHub repository URL.")
    return parser


def print_progress(stage: str, event: str, detail: str) -> None:
    """Progress goes to stderr so that stdout stays clean for --json."""
    label = pipeline.STAGE_LABELS.get(stage, stage)
    if event == "start":
        print(f"[ .. ] {label}", file=sys.stderr, flush=True)
    elif event == "progress":
        print(f"       {detail}", file=sys.stderr, flush=True)
    else:
        print(f"[ ok ] {label}" + (f": {detail}" if detail else ""), file=sys.stderr, flush=True)


def run_analyze(args: argparse.Namespace) -> int:
    if not args.no_image and args.screenshot is None:
        print("error: --screenshot is required unless --no-image is given.", file=sys.stderr)
        return 2
    result = pipeline.analyze(
        args.repo,
        args.text,
        None if args.no_image else args.screenshot,
        use_image=not args.no_image,
        on_progress=print_progress,
    )
    folder = save_result(result, args.out)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print()
        print(summary_text(result))
        print(f"\nFiles written to {folder}")
    return 0


def run_index(args: argparse.Namespace) -> int:
    prepared, run = pipeline.build_index_only(args.repo, on_progress=print_progress)
    print(f"Indexed {prepared.repo.full_name} @ {prepared.sha[:12]}: {len(prepared.index.chunks)} chunks.")
    for warning in run.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Model output can contain any character; do not crash on consoles that are not UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return run_analyze(args) if args.command == "analyze" else run_index(args)
    except BugLensError as error:
        print(f"error: {error}", file=sys.stderr)
        if error.hint:
            print(f"hint: {error.hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
