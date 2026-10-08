from __future__ import annotations

import pytest

from buglens.errors import EmptyReply, StructuredOutputError
from buglens.llm.json_utils import (
    JSONParseError,
    extract_outermost_object,
    generate_structured,
    parse_json_object,
    strip_code_fences,
)
from buglens.models import RankedFile, RerankReply, ScreenshotAnalysis
from tests.fakes import FakeLLM


def test_plain_json():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_code_fences_are_stripped():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_prose_around_the_object_is_ignored():
    text = 'Sure! Here is the JSON:\n```json\n{"a": {"b": [1, 2]}}\n```\nLet me know if you need more.'
    assert parse_json_object(text) == {"a": {"b": [1, 2]}}


def test_braces_and_quotes_inside_strings_do_not_confuse_the_parser():
    text = 'Result: {"title": "Crash on } key", "body": "say \\"hi\\" {x}"} trailing }'
    assert parse_json_object(text) == {"title": "Crash on } key", "body": 'say "hi" {x}'}


def test_code_block_inside_a_string_is_kept():
    text = '{"body_markdown": "Steps:\\n```\\nnpm start\\n```"}'
    assert parse_json_object(text)["body_markdown"] == "Steps:\n```\nnpm start\n```"


def test_trailing_commas_are_forgiven():
    assert parse_json_object('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_raw_newlines_in_strings_are_allowed():
    assert parse_json_object('{"a": "line one\nline two"}') == {"a": "line one\nline two"}


@pytest.mark.parametrize("text", ["", "no json here", '{"a": 1', "[1, 2, 3]", '{"a": nope}'])
def test_unparseable_replies_raise(text):
    with pytest.raises(JSONParseError):
        parse_json_object(text)


def test_extract_outermost_object_returns_the_whole_nested_object():
    assert extract_outermost_object('x {"a": {"b": 1}} y {"c": 2}') == '{"a": {"b": 1}}'


def test_lenient_models_accept_small_slips():
    analysis = ScreenshotAnalysis.model_validate(
        {"visible_text": "Save", "error_messages": None, "apparent_problem": None, "search_queries": [" a ", ""]}
    )
    assert analysis.visible_text == ["Save"]
    assert analysis.error_messages == []
    assert analysis.apparent_problem == ""
    assert analysis.search_queries == ["a"]


def test_confidence_is_clamped_to_zero_one():
    assert RankedFile(path="a.py", confidence=85).confidence == pytest.approx(0.85)
    assert RankedFile(path="a.py", confidence=-3).confidence == 0.0
    assert RankedFile(path="a.py", confidence=1000).confidence == 1.0


def test_generate_structured_happy_path_uses_one_call():
    llm = FakeLLM(['{"files": [{"path": "a.py", "reason": "r", "confidence": 0.5}]}'])
    reply = generate_structured(llm, "PROMPT", RerankReply)
    assert reply.files[0].path == "a.py"
    assert len(llm.calls) == 1


def test_generate_structured_repairs_invalid_json_once():
    llm = FakeLLM(["I think the answer is a.py", '{"files": [{"path": "a.py", "confidence": 0.4}]}'])
    reply = generate_structured(llm, "ORIGINAL PROMPT", RerankReply, image_png=b"png")

    assert reply.files[0].path == "a.py"
    assert len(llm.calls) == 2
    repair_prompt, repair_image = llm.calls[1]
    assert "ORIGINAL PROMPT" in repair_prompt
    assert "I think the answer is a.py" in repair_prompt
    assert "no '{' found" in repair_prompt  # the parse error is sent back
    assert repair_image == b"png"  # the image is sent again


def test_generate_structured_repairs_schema_errors_with_the_validation_message():
    llm = FakeLLM(['{"files": [{"path": "a.py", "confidence": "very high"}]}', '{"files": []}'])
    reply = generate_structured(llm, "PROMPT", RerankReply)
    assert reply.files == []
    assert "confidence" in llm.calls[1][0]


def test_generate_structured_gives_up_after_one_repair():
    llm = FakeLLM(["nope", "still nope"])
    with pytest.raises(StructuredOutputError, match="no '{' found"):  # the reason is part of the message
        generate_structured(llm, "PROMPT", RerankReply)
    assert len(llm.calls) == 2


def test_withheld_reply_is_retried_once_with_the_real_reason():
    blocked = EmptyReply("The Gemini API returned no text (finish reason: RECITATION).")
    llm = FakeLLM([blocked, '{"files": []}'])
    assert generate_structured(llm, "ORIGINAL PROMPT", RerankReply).files == []

    repair_prompt = llm.calls[1][0]
    assert "finish reason: RECITATION" in repair_prompt
    assert "do not repeat long passages" in repair_prompt
    assert "(no reply)" in repair_prompt and "ORIGINAL PROMPT" in repair_prompt


def test_reply_withheld_twice_surfaces_as_empty_reply_not_as_bad_json():
    blocked = EmptyReply("The Gemini API returned no text (finish reason: RECITATION).")
    llm = FakeLLM([blocked, blocked])
    with pytest.raises(EmptyReply, match="RECITATION") as caught:
        generate_structured(llm, "PROMPT", RerankReply)
    assert "--no-image" in caught.value.hint
    assert len(llm.calls) == 2
