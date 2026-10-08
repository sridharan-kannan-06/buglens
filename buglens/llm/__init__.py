"""LLM backends. `get_llm` picks one from the settings."""
from __future__ import annotations

from buglens.config import Settings, check_llm_settings
from buglens.llm.base import LLMClient


def get_llm(settings: Settings) -> LLMClient:
    """Create the configured backend, or raise ConfigError if the settings are incomplete."""
    check_llm_settings(settings)
    if settings.llm_backend == "ollama":
        from buglens.llm.ollama import OllamaLLM

        return OllamaLLM(settings.ollama_host, settings.gemma_model, native_json=settings.native_json)
    from buglens.llm.gemini import GeminiLLM

    return GeminiLLM(
        settings.gemini_api_key,
        settings.gemma_model,
        native_json=settings.native_json,
        thinking_level=settings.thinking_level,
    )
