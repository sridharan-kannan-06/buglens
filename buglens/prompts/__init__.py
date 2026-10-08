"""Prompt templates live in the .txt files next to this module.

Placeholders look like {{name}}. They are filled in a single pass, so a value
that itself contains "{{something}}" (for example text from an untrusted repo)
is never expanded again.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read prompts/<name>.txt."""
    return (_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render(name: str, **values: str) -> str:
    """Fill the {{placeholders}} of a prompt. A missing value is a programming error."""
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], load_prompt(name))
