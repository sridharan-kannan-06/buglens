"""Entry point for `python -m buglens.eval run ...` and `python -m buglens.eval collect ...`."""
from __future__ import annotations

import sys

USAGE = """usage:
  python -m buglens.eval run CASES.yaml [--no-image] [--limit N] [--sleep SECONDS] [--out DIR]
  python -m buglens.eval collect ISSUE_URL [--include-all] [--no-download]
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command == "run":
        from buglens.eval.run import main as run_main

        return run_main(rest)
    if command == "collect":
        from buglens.eval.collect import main as collect_main

        return collect_main(rest)
    print(f"unknown command '{command}'\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
