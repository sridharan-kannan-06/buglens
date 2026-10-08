"""EXPERIMENTAL: Gemma served locally by Ollama, over its HTTP API.

Enable with LLM_BACKEND=ollama. GEMMA_MODEL is then the Ollama model tag
(whatever `ollama list` shows). This backend has had far less testing than
the Gemini one.
"""
from __future__ import annotations

import base64

import requests

from buglens import config
from buglens.errors import ConfigError, LLMError
from buglens.llm.base import LLMClient, RetryableLLMError, call_with_backoff, is_retryable_status


class OllamaLLM(LLMClient):
    """Calls POST /api/generate with the image as a base64 string."""

    name = "ollama"

    def __init__(self, host: str, model: str, native_json: bool = False) -> None:
        self.model = model
        self._url = host.rstrip("/") + "/api/generate"
        self._native_json = native_json

    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        return call_with_backoff(lambda: self._generate_once(prompt, image_png), on_retry=self.on_retry)

    def _generate_once(self, prompt: str, image_png: bytes | None) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE, "num_predict": config.LLM_MAX_OUTPUT_TOKENS},
        }
        if image_png is not None:
            payload["images"] = [base64.b64encode(image_png).decode("ascii")]
        if self._native_json:
            payload["format"] = "json"
        try:
            response = requests.post(self._url, json=payload, timeout=config.LLM_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise LLMError(
                f"Could not reach Ollama ({type(exc).__name__}).",
                hint="Is Ollama running? Check OLLAMA_HOST.",
            ) from exc
        if is_retryable_status(response.status_code):
            raise RetryableLLMError(response.status_code, response.text[:300])
        if response.status_code == 404:
            raise ConfigError(
                f"Ollama does not have the model '{self.model}'.",
                hint="Run `ollama list` and set GEMMA_MODEL to one of the tags shown.",
            )
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
        try:
            return str(response.json().get("response") or "")
        except ValueError as exc:
            raise LLMError("Ollama returned a reply that is not JSON.") from exc
