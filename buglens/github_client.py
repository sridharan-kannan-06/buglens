"""Small GitHub REST client: parse repo URLs, resolve refs, download zipballs, list labels.

BugLens never runs git. Everything goes through https://api.github.com.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from buglens import config
from buglens.errors import GitHubError, GitHubRateLimit, InvalidRepoURL, RepoNotFound, RepoTooLarge
from buglens.models import RepoRef

_REPO_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9][A-Za-z0-9-]*)/"
    r"(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?"
    r"(?:/(?P<rest>.*))?$"
)


def parse_repo_url(url: str) -> RepoRef:
    """Turn a GitHub URL into a RepoRef.

    Accepts https://github.com/o/r, with or without `.git`, a trailing slash,
    or `/tree/<branch>`. Anything after `/tree/` is taken as the ref.
    """
    cleaned = (url or "").strip().split("#")[0].split("?")[0].rstrip("/")
    match = _REPO_URL.match(cleaned)
    if not match:
        raise InvalidRepoURL(f"'{url}' is not a GitHub repository URL.")
    rest = match.group("rest") or ""
    ref = None
    if rest.startswith("tree/"):
        ref = rest[len("tree/"):].strip("/") or None
    elif rest.startswith("commit/"):
        ref = rest[len("commit/"):].strip("/") or None
    return RepoRef(owner=match.group("owner"), repo=match.group("repo"), ref=ref)


class GitHubClient:
    """Thin wrapper over the GitHub REST API. The token is optional."""

    def __init__(self, token: str = "", session: requests.Session | None = None) -> None:
        self._token = token
        self._session = session or requests.Session()

    @property
    def authenticated(self) -> bool:
        return bool(self._token)

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "buglens"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
        stream: bool = False,
    ) -> requests.Response:
        """GET an API path (or a full URL) and raise a friendly error on failure."""
        url = path if path.startswith("https://") else config.GITHUB_API_ROOT + path
        try:
            response = self._session.get(
                url,
                headers=self._headers(accept),
                params=params,
                stream=stream,
                timeout=config.GITHUB_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise GitHubError(f"Could not reach GitHub ({type(exc).__name__}).") from exc
        _raise_for_status(response)
        return response

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET an API path and decode the JSON body."""
        return self._get(path, params=params).json()

    def graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Run a GraphQL query and return its `data`. GitHub's GraphQL API always needs a token."""
        if not self._token:
            raise GitHubError("This lookup needs GITHUB_TOKEN: GitHub's GraphQL API does not accept anonymous calls.")
        try:
            response = self._session.post(
                config.GITHUB_API_ROOT + "/graphql",
                json={"query": query, "variables": variables},
                headers=self._headers("application/vnd.github+json"),
                timeout=config.GITHUB_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise GitHubError(f"Could not reach GitHub ({type(exc).__name__}).") from exc
        _raise_for_status(response)
        payload = response.json()
        if payload.get("errors"):
            message = str(payload["errors"][0].get("message", ""))[:200]
            raise GitHubError(f"GitHub GraphQL error: {message}")
        return payload.get("data") or {}

    def get_repo(self, repo: RepoRef) -> dict[str, Any]:
        """Repository metadata (default branch, size, ...)."""
        return self.get_json(f"/repos/{repo.owner}/{repo.repo}")

    def _commit_sha(self, repo: RepoRef, ref: str) -> str:
        """Resolve a branch, tag or sha to a full commit sha."""
        path = f"/repos/{repo.owner}/{repo.repo}/commits/{quote(ref, safe='/')}"
        return self._get(path, accept="application/vnd.github.sha").text.strip()

    def resolve_ref(self, repo: RepoRef) -> tuple[str, str]:
        """Return (ref name, commit sha). Uses the default branch when no ref was given.

        A URL like /tree/main/src is ambiguous (branch `main/src`, or folder `src`
        on `main`?), so if the full ref is unknown we retry with its first segment.
        """
        ref = repo.ref or self.get_repo(repo)["default_branch"]
        attempts = [ref]
        if "/" in ref:
            attempts.append(ref.split("/")[0])
        for attempt in attempts:
            try:
                return attempt, self._commit_sha(repo, attempt)
            except (RepoNotFound, GitHubError) as exc:
                if isinstance(exc, GitHubError) and exc.status not in (404, 422):
                    raise
        raise RepoNotFound(f"Could not find '{ref}' in {repo.full_name} (or the repository is private).")

    def download_zipball(self, repo: RepoRef, sha: str, dest: Path, max_bytes: int) -> None:
        """Stream the repository zipball to `dest`, refusing anything over `max_bytes`."""
        response = self._get(f"/repos/{repo.owner}/{repo.repo}/zipball/{sha}", stream=True)
        too_large = RepoTooLarge(
            f"{repo.full_name} is larger than the {max_bytes // (1024 * 1024)} MB download limit."
        )
        if int(response.headers.get("Content-Length") or 0) > max_bytes:
            response.close()
            raise too_large
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        written = 0
        try:
            with open(partial, "wb") as handle:
                for block in response.iter_content(chunk_size=1 << 16):
                    written += len(block)
                    if written > max_bytes:
                        raise too_large
                    handle.write(block)
        except requests.RequestException as exc:
            partial.unlink(missing_ok=True)
            raise GitHubError(f"The download was interrupted ({type(exc).__name__}).") from exc
        except RepoTooLarge:
            partial.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        partial.replace(dest)

    def list_labels(self, repo: RepoRef, limit: int = config.MAX_LABELS) -> list[str]:
        """Names of up to `limit` labels defined in the repository."""
        data = self.get_json(f"/repos/{repo.owner}/{repo.repo}/labels", params={"per_page": min(limit, 100)})
        return [item["name"] for item in data if isinstance(item, dict) and item.get("name")][:limit]

    def list_issues(self, repo: RepoRef, limit: int = config.SIMILAR_ISSUES_POOL) -> list[dict[str, Any]]:
        """The most recently updated issues, open and closed. Pull requests are left out."""
        params = {"state": "all", "sort": "updated", "direction": "desc", "per_page": min(limit, 100)}
        data = self.get_json(f"/repos/{repo.owner}/{repo.repo}/issues", params=params)
        return [item for item in data if isinstance(item, dict) and "pull_request" not in item]

    def search_issues(self, repo: RepoRef, keywords: list[str], limit: int = config.SIMILAR_ISSUES_SEARCH) -> list[dict[str, Any]]:
        """Issues of this repository that contain all the keywords."""
        query = f"repo:{repo.full_name} is:issue " + " ".join(keywords)
        data = self.get_json("/search/issues", params={"q": query, "per_page": min(limit, 100)})
        return [item for item in data.get("items", []) if isinstance(item, dict) and "pull_request" not in item]


def _raise_for_status(response: requests.Response) -> None:
    """Map GitHub HTTP errors to BugLens errors."""
    status = response.status_code
    if status < 400:
        return
    remaining = response.headers.get("X-RateLimit-Remaining")
    if status in (403, 429) and (remaining == "0" or "rate limit" in response.text.lower()):
        raise GitHubRateLimit(f"GitHub API rate limit reached.{_reset_note(response)}")
    if status == 404:
        raise RepoNotFound("GitHub returned 404: the repository or ref does not exist, or it is private.")
    if status == 401:
        raise GitHubError("GitHub rejected GITHUB_TOKEN (401).", status, hint="Check or remove GITHUB_TOKEN in .env.")
    raise GitHubError(f"GitHub API returned HTTP {status}.", status)


def _reset_note(response: requests.Response) -> str:
    """Human-readable 'resets in N minutes' note from the rate-limit header."""
    try:
        seconds = int(response.headers.get("X-RateLimit-Reset", "0")) - int(time.time())
    except ValueError:
        return ""
    if seconds <= 0:
        return ""
    return f" It resets in about {seconds // 60 + 1} minute(s)."
