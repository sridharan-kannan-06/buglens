"""Get a validated pydantic object out of a free-text LLM reply.

Gemma is not guaranteed to support JSON mode or schema-constrained decoding,
so we ask for JSON in the prompt and then parse defensively:

1. strip a markdown code fence wrapped around the reply,
2. cut out the outermost `{...}`,
3. forgive trailing commas,
4. validate with pydantic,
5. on failure, retry once with the validation error sent back to the model.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from buglens import config
from buglens.errors import EmptyReply, StructuredOutputError
from buglens.llm.base import LLMClient
from buglens.prompts import render

ModelT = TypeVar("ModelT", bound=BaseModel)

_OPENING_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n?")
_CLOSING_FENCE = re.compile(r"\r?\n?```$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


class JSONParseError(ValueError):
    """The reply did not contain a parseable JSON object."""


def strip_code_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper around the whole reply.

    Only a fence at the very start and end is removed. Fences inside the reply
    are left alone because the issue body may legitimately contain code blocks.
    """
    text = text.strip()
    if text.startswith("```"):
        text = _OPENING_FENCE.sub("", text, count=1)
        text = _CLOSING_FENCE.sub("", text, count=1)
    return text


def extract_outermost_object(text: str) -> str:
    """Return the first complete `{...}` in the text, respecting strings and escapes."""
    start = text.find("{")
    if start == -1:
        raise JSONParseError("no '{' found in the reply")
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:position + 1]
    raise JSONParseError("the JSON object is not closed (the reply may have been cut off)")


def parse_json_object(text: str) -> dict[str, Any]:
    """Tolerantly parse a JSON object from an LLM reply."""
    candidate = extract_outermost_object(strip_code_fences(text or ""))
    error: Exception | None = None
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            data = json.loads(attempt, strict=False)  # strict=False allows raw newlines in strings
        except json.JSONDecodeError as exc:
            error = exc
            continue
        if not isinstance(data, dict):
            raise JSONParseError("the reply is JSON but not an object")
        return data
    raise JSONParseError(f"invalid JSON: {error}")


def parse_reply(text: str, schema: type[ModelT]) -> ModelT:
    """Parse and validate a reply. Raises JSONParseError or pydantic.ValidationError."""
    return schema.model_validate(parse_json_object(text))


def generate_structured(
    llm: LLMClient,
    prompt: str,
    schema: type[ModelT],
    image_png: bytes | None = None,
) -> ModelT:
    """Call the LLM and return a validated object, with exactly one repair retry.

    The retry also covers a reply that the API withheld (EmptyReply). If the
    retry is withheld too, EmptyReply is raised so the caller sees the real cause.
    """
    reply = ""
    try:
        reply = llm.generate(prompt, image_png)
        return parse_reply(reply, schema)
    except EmptyReply as empty:
        problem = f"{empty} Keep every string short and do not repeat long passages of text word for word."
    except (JSONParseError, ValidationError) as first_error:
        problem = str(first_error)[:1500]

    repair_prompt = render(
        "repair",
        error=problem,
        previous=reply[: config.REPAIR_REPLY_CHARS] or "(no reply)",
        original_prompt=prompt,
    )
    repaired = llm.generate(repair_prompt, image_png)
    try:
        return parse_reply(repaired, schema)
    except (JSONParseError, ValidationError) as second_error:
        reason = " ".join(str(second_error).split())[:200]
        raise StructuredOutputError(
            f"The model did not return valid JSON for {schema.__name__} after one repair retry ({reason})."
        ) from second_error
