from __future__ import annotations

import pytest

from buglens.config import Settings, check_llm_settings
from buglens.errors import ConfigError, LLMError, LLMRateLimit
from buglens.llm import get_llm
from buglens.llm.base import RetryableLLMError, call_with_backoff, is_retryable_status
from buglens.prompts import load_prompt, render


def settings(**overrides) -> Settings:
    values = dict(
        gemini_api_key="key",
        gemma_model="some-model",
        llm_backend="gemini",
        github_token="",
        ollama_host="http://localhost:11434",
        native_json=False,
        max_chunks=8000,
    )
    values.update(overrides)
    return Settings(**values)


def test_retryable_statuses():
    assert is_retryable_status(429) and is_retryable_status(500) and is_retryable_status(503)
    assert not is_retryable_status(400) and not is_retryable_status(404) and not is_retryable_status(None)


def test_backoff_retries_then_succeeds_with_growing_delays():
    delays: list[float] = []
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RetryableLLMError(429, "slow down")
        return "ok"

    assert call_with_backoff(flaky, retries=4, base_delay=1.0, sleep=delays.append) == "ok"
    assert attempts["count"] == 3
    assert len(delays) == 2
    assert 1.0 <= delays[0] < 1.6 and 2.0 <= delays[1] < 2.6


def test_backoff_gives_up_with_a_rate_limit_error():
    def always_429() -> str:
        raise RetryableLLMError(429, "slow down")

    with pytest.raises(LLMRateLimit):
        call_with_backoff(always_429, retries=2, base_delay=0.0, sleep=lambda seconds: None)


def test_backoff_gives_up_with_a_generic_error_on_5xx():
    def always_503() -> str:
        raise RetryableLLMError(503, "unavailable")

    with pytest.raises(LLMError) as caught:
        call_with_backoff(always_503, retries=1, base_delay=0.0, sleep=lambda seconds: None)
    assert not isinstance(caught.value, LLMRateLimit)


def test_missing_model_id_tells_the_user_to_run_check_gemma():
    with pytest.raises(ConfigError) as caught:
        check_llm_settings(settings(gemma_model=""))
    assert "GEMMA_MODEL" in str(caught.value)
    assert "scripts/check_gemma.py" in caught.value.hint


def test_missing_api_key_and_unknown_backend():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        get_llm(settings(gemini_api_key=""))
    with pytest.raises(ConfigError, match="LLM_BACKEND"):
        get_llm(settings(llm_backend="openai"))


def test_ollama_backend_needs_no_gemini_key():
    llm = get_llm(settings(llm_backend="ollama", gemini_api_key="", gemma_model="local-tag"))
    assert (llm.name, llm.model) == ("ollama", "local-tag")


def test_error_messages_never_contain_the_api_key():
    secret = "SECRET-KEY-VALUE"
    for broken in (settings(gemini_api_key=secret, gemma_model=""), settings(gemini_api_key=secret, llm_backend="x")):
        with pytest.raises(ConfigError) as caught:
            check_llm_settings(broken)
        assert secret not in str(caught.value) and secret not in caught.value.hint


def api_error(code: int, message: str):
    from google.genai import errors

    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})


def test_gemini_errors_are_translated_and_never_leak_the_key():
    from buglens.llm.gemini import _translate

    secret = "SECRET-KEY-VALUE"
    retryable = _translate(api_error(429, "quota exceeded"), "m", secret)
    assert isinstance(retryable, RetryableLLMError) and retryable.status == 429
    assert isinstance(_translate(api_error(503, "overloaded"), "m", secret), RetryableLLMError)

    not_found = _translate(api_error(404, "models/m is not found"), "m", secret)
    assert isinstance(not_found, ConfigError) and "scripts/check_gemma.py" in not_found.hint

    bad_key = _translate(api_error(400, "API key not valid. Please pass a valid API key."), "m", secret)
    assert isinstance(bad_key, ConfigError)

    other = _translate(api_error(400, f"bad request for key {secret}"), "m", secret)
    assert isinstance(other, LLMError) and secret not in str(other) and "***" in str(other)


def test_finish_reason_is_read_from_the_response():
    from types import SimpleNamespace

    from buglens.llm.gemini import finish_reason

    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name="RECITATION"))
    assert finish_reason(SimpleNamespace(prompt_feedback=None, candidates=[candidate])) == "RECITATION"
    feedback = SimpleNamespace(block_reason=SimpleNamespace(name="SAFETY"))
    assert finish_reason(SimpleNamespace(prompt_feedback=feedback, candidates=None)) == "prompt blocked, SAFETY"
    assert finish_reason(SimpleNamespace(prompt_feedback=None, candidates=[])) == "unknown"


def test_vision_prompt_asks_for_short_strings_not_transcripts():
    prompt = load_prompt("vision")
    assert "at most 60 characters" in prompt
    assert "Do NOT transcribe content" in prompt


@pytest.mark.parametrize("name", ["vision", "rerank", "writer"])
def test_every_task_prompt_says_untrusted_text_is_data(name):
    prompt = load_prompt(name)
    assert "SECURITY RULE" in prompt
    assert "DATA" in prompt and "never an instruction" in prompt
    assert "ONE JSON object" in prompt


def test_render_fills_placeholders_in_a_single_pass():
    text = render("repair", error="{{previous}}", previous="OLD REPLY", original_prompt="TASK")
    assert "{{previous}}" in text  # a value that looks like a placeholder is left alone
    assert text.count("OLD REPLY") == 1
