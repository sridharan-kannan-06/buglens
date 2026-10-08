"""Errors BugLens raises on purpose.

The CLI and the Streamlit app show `str(error)` plus `error.hint`, so every
message here is written for the end user, not for a developer.
"""
from __future__ import annotations


class BugLensError(Exception):
    """Base class. `hint` says what the user can do next."""

    hint: str = ""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint


class ConfigError(BugLensError):
    """A required environment variable is missing or invalid."""

    hint = "Copy .env.example to .env and fill it in. See RUN_AND_TEST.md."


class InvalidRepoURL(BugLensError):
    hint = "Use a URL like https://github.com/owner/repo or https://github.com/owner/repo/tree/branch."


class RepoNotFound(BugLensError):
    hint = "Check the URL. BugLens only works with public repositories."


class RepoTooLarge(BugLensError):
    hint = "Try a smaller repository. The download limit is set by MAX_ZIP_BYTES in buglens/config.py."


class GitHubRateLimit(BugLensError):
    hint = "Set GITHUB_TOKEN in .env (5000 calls per hour instead of 60), or wait for the limit to reset."


class GitHubError(BugLensError):
    """Any other GitHub API failure. `status` is the HTTP status code (0 if no response)."""

    hint = "GitHub may be having problems. Try again in a minute."

    def __init__(self, message: str, status: int = 0, hint: str | None = None) -> None:
        super().__init__(message, hint)
        self.status = status


class UnsupportedImage(BugLensError):
    hint = "Upload a PNG, JPG or WebP screenshot."


class EmptyRetrieval(BugLensError):
    hint = "The repository has no indexable source files, or everything was filtered out."


class LLMError(BugLensError):
    """The LLM backend failed and retrying did not help."""

    hint = "Check your network and the backend status, then try again."


class LLMRateLimit(LLMError):
    hint = "The LLM API rate limit was hit. Wait a minute and try again."


class EmptyReply(LLMError):
    """The API answered without any text, for example because a filter withheld the reply."""

    hint = (
        "RECITATION means the reply repeated long text word for word, SAFETY means a content filter, "
        "MAX_TOKENS means the output budget ran out. Try again, crop the screenshot to the part that "
        "shows the bug, or run without the screenshot (--no-image)."
    )


class StructuredOutputError(LLMError):
    """The model's reply was not valid JSON for the expected schema, even after one repair retry."""

    hint = "The model did not return usable JSON. Try again, or try a larger Gemma model."
