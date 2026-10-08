"""The LLM interface every backend implements, plus the shared retry helper."""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

from buglens import config
from buglens.errors import LLMError, LLMRateLimit

T = TypeVar("T")


class LLMClient(ABC):
    """One prompt in, one text reply out. An optional PNG image goes with the prompt.

    There is no system prompt and no JSON mode in this interface on purpose:
    Gemma may support neither, so every instruction lives in the user prompt.
    """

    name: str = "base"
    model: str = ""
    # Optional callback (http status, seconds to wait) called before each backoff wait.
    on_retry: Callable[[int, float], None] | None = None

    @abstractmethod
    def generate(self, prompt: str, image_png: bytes | None = None) -> str:
        """Return the model's raw text reply."""


class RetryableLLMError(Exception):
    """Raised by a backend for failures worth retrying: HTTP 429 and 5xx."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def is_retryable_status(status: int | None) -> bool:
    return status == 429 or (status is not None and 500 <= status < 600)


def call_with_backoff(
    call: Callable[[], T],
    retries: int = config.LLM_MAX_RETRIES,
    base_delay: float = config.LLM_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float], None] | None = None,
) -> T:
    """Run `call`, retrying RetryableLLMError with exponential backoff and a little jitter."""
    for attempt in range(retries + 1):
        try:
            return call()
        except RetryableLLMError as error:
            if attempt == retries:
                if error.status == 429:
                    raise LLMRateLimit(f"The LLM API kept answering 429 (rate limit) after {retries} retries.") from error
                raise LLMError(f"The LLM API kept failing (HTTP {error.status}) after {retries} retries.") from error
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            if on_retry:
                on_retry(error.status, delay)
            sleep(delay)
    raise AssertionError("unreachable")
