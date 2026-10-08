"""Best-effort helper that turns a fixed GitHub issue into an eval case.

    python -m buglens.eval.collect https://github.com/owner/repo/issues/123

It looks for the merged pull request that closed the issue, lists the files that
pull request changed, picks the commit just before the fix, downloads images
from the issue body, and prints a YAML stanza.

The pull request is found in three ways, most reliable first: GitHub's own
"closed by" link (needs GITHUB_TOKEN), then the issue's timeline, then a search
for merged pull requests that say "fixes #N".

THIS IS A HEURISTIC. It can pick the wrong pull request, and a pull request can
contain unrelated changes. Always review the stanza by hand before using it.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from PIL import Image

from buglens import config
from buglens.errors import BugLensError
from buglens.github_client import GitHubClient
from buglens.models import RepoRef
from buglens.repo_loader import PRIORITY_DOCS_TESTS, file_priority, is_wanted

MAX_CANDIDATE_PRS = 5
MAX_IMAGES = 5
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}

_ISSUE_URL = re.compile(r"^https?://github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\((https://[^)\s]+)")
_HTML_IMAGE = re.compile(r"<img[^>]+src=[\"'](https://[^\"']+)[\"']", re.IGNORECASE)
_CLOSING_KEYWORD = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s+(?:[\w.-]+/[\w.-]+)?#(\d+)", re.IGNORECASE
)


_CLOSING_PRS_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      closedByPullRequestsReferences(first: 10, includeClosedPrs: true) { nodes { number merged } }
    }
  }
}
"""


@dataclass
class Collected:
    """Everything found for one issue, plus notes about how sure we are."""

    repo_url: str
    issue_url: str
    title: str
    ref: str = ""
    fixed_files: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse_issue_url(url: str) -> tuple[RepoRef, int]:
    match = _ISSUE_URL.match(url.strip())
    if not match:
        raise ValueError(f"'{url}' is not a GitHub issue URL (expected https://github.com/owner/repo/issues/123).")
    return RepoRef(owner=match.group(1), repo=match.group(2)), int(match.group(3))


def find_image_urls(body: str) -> list[str]:
    """Image URLs in an issue body (markdown and <img> tags), hosted by GitHub only."""
    urls: list[str] = []
    for url in _MARKDOWN_IMAGE.findall(body or "") + _HTML_IMAGE.findall(body or ""):
        host = (urlparse(url).hostname or "").lower()
        if (host == "github.com" or host.endswith(".githubusercontent.com")) and url not in urls:
            urls.append(url)
    return urls


def is_ground_truth_file(path: str) -> bool:
    """True for files BugLens could find: source or config, not tests, docs, lockfiles or binaries."""
    return is_wanted(path, 1) and file_priority(path) != PRIORITY_DOCS_TESTS


def closes_issue(text: str, number: int) -> bool:
    """Does this pull request text say 'fixes #N' (or closes/resolves) for our issue?"""
    return any(int(found) == number for found in _CLOSING_KEYWORD.findall(text or ""))


def linked_closing_prs(client: GitHubClient, repo: RepoRef, number: int) -> list[int]:
    """Merged pull requests that GitHub itself lists as having closed the issue. Needs a token."""
    data = client.graphql(_CLOSING_PRS_QUERY, {"owner": repo.owner, "name": repo.repo, "number": number})
    issue = (data.get("repository") or {}).get("issue") or {}
    nodes = (issue.get("closedByPullRequestsReferences") or {}).get("nodes") or []
    return [node["number"] for node in nodes if node and node.get("merged")]


def search_closing_prs(client: GitHubClient, repo: RepoRef, number: int) -> list[int]:
    """Merged pull requests whose text says they fix this issue, found with the search API."""
    query = f"repo:{repo.full_name} is:pr is:merged {number} in:body"
    found = client.get_json("/search/issues", params={"q": query, "per_page": 10})
    return [
        item["number"]
        for item in found.get("items", [])
        if closes_issue(f"{item.get('title', '')}\n{item.get('body') or ''}", number)
    ]


def find_closing_pr(client: GitHubClient, repo: RepoRef, number: int) -> tuple[dict[str, Any] | None, str]:
    """Find the merged pull request that most likely fixed the issue. Returns (pull request, note)."""
    base = f"/repos/{repo.owner}/{repo.repo}"

    # 1. GitHub's own link between the issue and the pull request that closed it.
    linked: list[int] = []
    if client.authenticated:
        try:
            linked = linked_closing_prs(client, repo, number)
        except BugLensError:
            linked = []
    if linked:
        note = f"PR #{linked[0]} is linked on GitHub as the pull request that closed this issue."
        if len(linked) > 1:
            others = ", ".join(f"#{pr_number}" for pr_number in linked[1:])
            note += f" Other linked PRs: {others}. CHECK which one holds the fix."
        return client.get_json(f"{base}/pulls/{linked[0]}"), note

    # 2. The issue timeline: cross-references from merged pull requests.
    events = client.get_json(f"{base}/issues/{number}/timeline", params={"per_page": 100})

    closing_commit = None
    pr_numbers: list[int] = []
    for event in events:
        if event.get("event") == "closed" and event.get("commit_id"):
            closing_commit = event["commit_id"]
        source = (event.get("source") or {}).get("issue") or {}
        same_repo = (source.get("repository") or {}).get("full_name", repo.full_name).lower() == repo.full_name.lower()
        merged = (source.get("pull_request") or {}).get("merged_at")
        if event.get("event") == "cross-referenced" and merged and same_repo and source["number"] not in pr_numbers:
            pr_numbers.append(source["number"])

    pulls = [client.get_json(f"{base}/pulls/{pr_number}") for pr_number in pr_numbers[-MAX_CANDIDATE_PRS:]]
    for pull in pulls:
        if closing_commit and pull.get("merge_commit_sha") == closing_commit:
            return pull, f"PR #{pull['number']} was merged as the commit that closed the issue."
    for pull in pulls:
        if closes_issue(f"{pull.get('title', '')}\n{pull.get('body') or ''}", number):
            return pull, f"PR #{pull['number']} says it closes this issue."
    if len(pulls) == 1:
        return pulls[0], f"PR #{pulls[0]['number']} is the only merged PR that mentions the issue. LOW CONFIDENCE."
    if pulls:
        return None, "Several merged PRs mention this issue and none clearly closes it."

    # 3. Search for merged pull requests that say "fixes #N".
    try:
        searched = search_closing_prs(client, repo, number)
    except BugLensError:
        searched = []
    if len(searched) == 1:
        return client.get_json(f"{base}/pulls/{searched[0]}"), f"PR #{searched[0]} was found by search and says it closes this issue."
    if searched:
        found = ", ".join(f"#{pr_number}" for pr_number in searched)
        return None, f"Several merged PRs say they close this issue ({found}). Pick one by hand."
    hint = "" if client.authenticated else " Set GITHUB_TOKEN: GitHub's own issue-to-PR link can only be read with a token."
    return None, "No merged pull request that closes this issue was found." + hint


def pre_fix_ref(client: GitHubClient, repo: RepoRef, pull: dict[str, Any]) -> tuple[str, str]:
    """The commit just before the fix: first parent of the merge commit, else the PR base."""
    merge_sha = pull.get("merge_commit_sha")
    if pull.get("merged") and merge_sha:
        commit = client.get_json(f"/repos/{repo.owner}/{repo.repo}/commits/{merge_sha}")
        parents = commit.get("parents") or []
        if parents:
            note = "ref = first parent of the merge commit."
            if len(parents) == 1 and (pull.get("commits") or 1) > 1:
                note += " The PR had several commits: if it was rebase-merged, this ref already contains part of the fix. CHECK."
            return parents[0]["sha"], note
    return pull["base"]["sha"], "ref = base commit of the PR (merge commit unavailable). CHECK."


def changed_files(client: GitHubClient, repo: RepoRef, pr_number: int, include_all: bool) -> tuple[list[str], int]:
    """Paths (as they were before the fix) changed by the PR. Returns (kept, number skipped)."""
    kept: list[str] = []
    skipped = 0
    for page in (1, 2, 3):
        files = client.get_json(
            f"/repos/{repo.owner}/{repo.repo}/pulls/{pr_number}/files", params={"per_page": 100, "page": page}
        )
        for item in files:
            path = item.get("previous_filename") or item["filename"]
            # A file added by the fix does not exist before the fix, so it can never be found.
            if item.get("status") == "added" or not (include_all or is_ground_truth_file(path)):
                skipped += 1
            else:
                kept.append(path)
        if len(files) < 100:
            break
    return kept, skipped


def download_images(urls: list[str], folder: Path, prefix: str) -> tuple[list[str], list[str]]:
    """Download up to MAX_IMAGES images. Returns (saved paths, notes about failures)."""
    saved, notes = [], []
    for number, url in enumerate(urls[:MAX_IMAGES], start=1):
        try:
            response = requests.get(url, timeout=config.GITHUB_TIMEOUT_SECONDS, stream=True)
            response.raise_for_status()
            data = response.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("larger than 10 MB")
            extension = IMAGE_EXTENSIONS.get(Image.open(io.BytesIO(data)).format or "")
            if extension is None:
                raise ValueError("not a png, jpg or webp image")
        except (requests.RequestException, OSError, ValueError) as exc:
            notes.append(f"Could not use image {number} ({type(exc).__name__}: {exc}).")
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{prefix}_{number}.{extension}"
        path.write_bytes(data)
        saved.append(path.as_posix())
    return saved, notes


def collect(
    issue_url: str,
    client: GitHubClient,
    screenshots_dir: Path,
    include_all: bool = False,
    download: bool = True,
) -> Collected:
    """Gather everything for one issue. Network errors surface as BugLensError."""
    repo, number = parse_issue_url(issue_url)
    issue = client.get_json(f"/repos/{repo.owner}/{repo.repo}/issues/{number}")
    found = Collected(repo_url=repo.url, issue_url=issue_url.strip(), title=str(issue.get("title") or "").strip())
    if issue.get("state") != "closed":
        found.notes.append("The issue is not closed, so there may be no fix yet.")

    pull, note = find_closing_pr(client, repo, number)
    found.notes.append(note)
    if pull is not None:
        found.ref, ref_note = pre_fix_ref(client, repo, pull)
        found.notes.append(ref_note)
        found.fixed_files, skipped = changed_files(client, repo, pull["number"], include_all)
        if skipped:
            found.notes.append(f"{skipped} changed file(s) left out (tests, docs, lockfiles, binaries or new files).")
        if not found.fixed_files:
            found.notes.append("No usable ground-truth file: this issue cannot be an eval case.")

    image_urls = find_image_urls(issue.get("body") or "")
    if not image_urls:
        found.notes.append("No image in the issue body: this case only works for the text-only variant.")
    elif download:
        prefix = f"{repo.owner}__{repo.repo}__{number}"
        found.screenshots, image_notes = download_images(image_urls, screenshots_dir, prefix)
        found.notes.extend(image_notes)
        if len(found.screenshots) > 1:
            found.notes.append("Several images were downloaded; the first is used. Pick the one that shows the bug.")
    return found


def yaml_stanza(found: Collected) -> str:
    """A YAML list item for eval/cases.yaml, with the notes as comments on top."""
    comments = ["# REVIEW BY HAND before using this case. It was collected by a heuristic."]
    comments += [f"# - {note}" for note in found.notes]
    comments.append("# - `text` is the issue title. Rewrite it as the one-line description a user would give.")
    case = {
        "repo": found.repo_url,
        "ref": found.ref or "TODO",
        "issue_url": found.issue_url,
        "screenshot": found.screenshots[0] if found.screenshots else "",
        "text": found.title,
        "fixed_files": found.fixed_files or ["TODO"],
    }
    return "\n".join(comments) + "\n" + yaml.safe_dump([case], sort_keys=False, allow_unicode=True, width=1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m buglens.eval.collect", description=__doc__.split("\n\n")[0])
    parser.add_argument("issue_url", help="URL of a closed GitHub issue that was fixed by a pull request.")
    parser.add_argument("--screenshots-dir", type=Path, default=Path("eval/screenshots"))
    parser.add_argument("--include-all", action="store_true", help="Keep tests, docs and lockfiles in fixed_files.")
    parser.add_argument("--no-download", action="store_true", help="Do not download images.")
    args = parser.parse_args(argv)

    settings = config.load_settings()
    client = GitHubClient(settings.github_token)
    if not client.authenticated:
        print(f"warning: {config.UNAUTHENTICATED_WARNING}", file=sys.stderr)
    try:
        found = collect(args.issue_url, client, args.screenshots_dir, args.include_all, not args.no_download)
    except (BugLensError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(yaml_stanza(found))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
