"""Gemma through the Gemini API, using the google-genai SDK."""
from __future__ import annotations

from google import genai
from google.genai import errors, types

from buglens import config
from buglens.errors import ConfigError, EmptyReply, LLMError
from buglens.llm.base import LLMClient, RetryableLLMError, call_with_backoff, is_retryable_status


class GeminiLLM(LLMClient):
    """Sends one user turn (optional image + prompt) and returns the text reply."""

    name = "gemini"

    def __init__(self, api_key: str, model: str, native_json: bool = False, thinking_level: str = "") -> None:
        self.model = model
        self._api_key = api_key
        self._native_json = native_json
        self._thinking_level = thinking_level
        timeout_ms = config.LLM_TIMEOUT_SECONDS * 1000
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))

    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        return call_with_backoff(lambda: self._generate_once(prompt, image_png), on_retry=self.on_retry)

    def _generate_once(self, prompt: str, image_png: bytes | None) -> str:
        parts: list[types.Part | str] = []
        if image_png is not None:
            parts.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))
        parts.append(prompt)
        # No system_instruction: Gemma may reject it. JSON mode and the thinking level are
        # opt-in settings, because not every model accepts them.
        thinking = None
        if self._thinking_level:
            thinking = types.ThinkingConfig(thinking_level=self._thinking_level.upper())
        generation_config = types.GenerateContentConfig(
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json" if self._native_json else None,
            thinking_config=thinking,
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=parts, config=generation_config
            )
        except (ValueError, TypeError) as exc:  # the SDK rejected a setting before sending anything
            raise ConfigError(f"The Gemini SDK rejected the request settings ({exc}).") from exc
        except errors.APIError as exc:
            raise _translate(exc, self.model, self._api_key) from exc
        except Exception as exc:  # network trouble, timeouts, ...
            raise LLMError(f"Could not reach the Gemini API ({type(exc).__name__}).") from exc
        text = response.text or ""
        if not text.strip():
            raise EmptyReply(f"The Gemini API returned no text (finish reason: {finish_reason(response)}).")
        return text


def finish_reason(response: object) -> str:
    """Why generation ended: STOP, MAX_TOKENS, RECITATION, SAFETY, ... or a blocked prompt."""
    block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if block_reason:
        return f"prompt blocked, {getattr(block_reason, 'name', block_reason)}"
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    return str(getattr(reason, "name", None) or reason or "unknown")


def _translate(exc: errors.APIError, model: str, api_key: str) -> Exception:
    """Turn an SDK error into a retryable error or a user-facing one. The key is blanked out."""
    status = getattr(exc, "code", None)
    message = getattr(exc, "message", "") or ""
    if api_key:
        message = message.replace(api_key, "***")
    message = message[:300]
    if is_retryable_status(status):
        return RetryableLLMError(status, message)
    if status == 400 and "api key" in message.lower():
        return ConfigError("The Gemini API says the API key is not valid (400).")
    if status == 400 and "thinking" in message.lower():
        return ConfigError(
            f"The model '{model}' rejected the BUGLENS_THINKING_LEVEL setting: {message}",
            hint="Remove BUGLENS_THINKING_LEVEL from .env, or try the value 'minimal'.",
        )
    if status == 404:
        return ConfigError(
            f"The Gemini API does not know the model '{model}' (404).",
            hint="Run `python scripts/check_gemma.py` and copy an exact model id into GEMMA_MODEL.",
        )
    if status in (401, 403):
        return ConfigError(f"The Gemini API rejected the API key or its permissions ({status}).")
    return LLMError(f"Gemini API error {status}: {message}")
