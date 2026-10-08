"""Thin wrapper for agents: run the BugLens CLI and print its result as JSON on stdout.

    python scripts/run_buglens.py --repo URL --screenshot PATH --text "..."

The BugLens project is expected three folders above this file
(<project>/skill/buglens/scripts/). Set BUGLENS_HOME if it lives elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    """Folder that contains the `buglens` package."""
    return Path(os.environ.get("BUGLENS_HOME") or Path(__file__).resolve().parents[3])


def project_python(root: Path) -> str:
    """Prefer the project's virtualenv so its dependencies are available."""
    for candidate in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def fail(error: str, hint: str = "", code: int = 1) -> int:
    print(json.dumps({"ok": False, "error": error, "hint": hint}))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BugLens and print the result as JSON.")
    parser.add_argument("--repo", required=True, help="Public GitHub repository URL.")
    parser.add_argument("--screenshot", help="Path to a png/jpg/webp screenshot.")
    parser.add_argument("--text", required=True, help="One-line description of the bug.")
    parser.add_argument("--no-image", action="store_true", help="Use only the text.")
    args = parser.parse_args(argv)

    root = project_root()
    if not (root / "buglens" / "cli.py").is_file():
        return fail(f"BugLens project not found at {root}.", "Set BUGLENS_HOME to the project folder.")
    if not args.no_image and not args.screenshot:
        return fail("A screenshot is required.", "Pass --screenshot PATH or use --no-image.")

    command = [project_python(root), "-m", "buglens", "analyze", "--repo", args.repo, "--text", args.text, "--json"]
    if args.no_image:
        command.append("--no-image")
    else:
        command += ["--screenshot", str(Path(args.screenshot).resolve())]  # the CLI runs in another folder

    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        lines = completed.stderr.splitlines()
        error = next((line[len("error: "):] for line in lines if line.startswith("error: ")), "BugLens failed.")
        hint = next((line[len("hint: "):] for line in lines if line.startswith("hint: ")), "")
        return fail(error, hint, completed.returncode)
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
