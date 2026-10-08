"""File names read from the screenshot (stack traces, error overlays, editor tabs).

A screenshot that shows `File "app/views/login.py", line 42` names the file
directly. This module turns such strings into (path, line) and finds the
matching files of the repository.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_URL_PREFIX = re.compile(r"^[a-z][a-z0-9+.-]*://(?:/\.?/?|[^/]*/)", re.IGNORECASE)  # webpack:///./, http://host/
_DRIVE = re.compile(r"^[A-Za-z]:/")  # C:/Users/...
_LINE_SUFFIX = re.compile(r":(\d+)(?::\d+)?$")  # :42 or :42:7
_LINE_WORDS = re.compile(r"[\s,\"']*\bline\s+(\d+).*$", re.IGNORECASE)  # ", line 42, in login"
_FILE_WORD = re.compile(r"^file\s+", re.IGNORECASE)  # Python tracebacks: File "app.py", line 3
_PATH_CHARS = re.compile(r"^[\w./@+\-\[\]()]+$")
_WRAPPERS = "`'\"(),[]<>"


@dataclass(frozen=True)
class PathHint:
    """A file path (as far as it was visible) and an optional line number."""

    path: str
    line: int | None = None


def parse_hint(text: str) -> PathHint | None:
    """Parse strings such as 'src\\\\views\\\\login.py:42:7', 'login.py, line 42' or 'at Form (Form.tsx:12)'.

    Returns None for anything that does not look like a file name with an extension.
    """
    text = text.strip()
    line = None
    words = _LINE_WORDS.search(text)
    if words:
        line = int(words.group(1))
        text = text[: words.start()]
    tokens = _FILE_WORD.sub("", text.strip()).split()
    if not tokens:
        return None
    # In a stack frame such as "at Form (Form.tsx:12)" the file is the last word.
    text = tokens[-1].strip(_WRAPPERS).replace("\\", "/").split("?")[0]
    text = _DRIVE.sub("", _URL_PREFIX.sub("", text))
    number = _LINE_SUFFIX.search(text)
    if number:
        line = line or int(number.group(1))
        text = text[: number.start()]
    while text.startswith("./"):
        text = text[2:]
    text = text.strip("/")
    name = text.rsplit("/", 1)[-1]
    stem, _, extension = name.rpartition(".")
    looks_like_a_file = bool(stem) and extension[:1].isalpha() and len(extension) <= 10  # not "v1.2"
    if len(name) < 4 or not looks_like_a_file or not _PATH_CHARS.match(text):
        return None
    return PathHint(path=text, line=line)


def match_hint(hint: PathHint, repo_paths: list[str]) -> list[str]:
    """Repository files the hint points at.

    The file name must be equal. Among files with that name, the ones sharing
    the most trailing folders with the hint win, so 'views/login.py' prefers
    'app/views/login.py' over 'tests/login.py'.
    """
    wanted = hint.path.lower().split("/")
    best: list[str] = []
    best_depth = 0
    for path in repo_paths:
        parts = path.lower().split("/")
        depth = 0
        while depth < min(len(parts), len(wanted)) and parts[-1 - depth] == wanted[-1 - depth]:
            depth += 1
        if depth == 0:
            continue
        if depth > best_depth:
            best, best_depth = [path], depth
        elif depth == best_depth:
            best.append(path)
    return best
