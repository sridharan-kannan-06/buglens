from __future__ import annotations

import pytest

from buglens.errors import InvalidRepoURL
from buglens.github_client import parse_repo_url


@pytest.mark.parametrize(
    ("url", "owner", "repo", "ref"),
    [
        ("https://github.com/pallets/flask", "pallets", "flask", None),
        ("https://github.com/pallets/flask/", "pallets", "flask", None),
        ("https://github.com/pallets/flask.git", "pallets", "flask", None),
        ("http://www.github.com/pallets/flask.git/", "pallets", "flask", None),
        ("github.com/pallets/flask", "pallets", "flask", None),
        ("  https://github.com/pallets/flask  ", "pallets", "flask", None),
        ("https://github.com/pallets/flask/tree/main", "pallets", "flask", "main"),
        ("https://github.com/pallets/flask/tree/feature/login-fix/", "pallets", "flask", "feature/login-fix"),
        ("https://github.com/o/my.repo-name_1", "o", "my.repo-name_1", None),
        ("https://github.com/o/r/commit/abc123", "o", "r", "abc123"),
        ("https://github.com/o/r/issues/12", "o", "r", None),
        ("https://github.com/o/r?tab=readme#usage", "o", "r", None),
    ],
)
def test_parse_repo_url(url, owner, repo, ref):
    parsed = parse_repo_url(url)
    assert (parsed.owner, parsed.repo, parsed.ref) == (owner, repo, ref)
    assert parsed.full_name == f"{owner}/{repo}"
    assert parsed.url == f"https://github.com/{owner}/{repo}"


@pytest.mark.parametrize(
    "url",
    ["", "not a url", "https://gitlab.com/o/r", "https://github.com/onlyowner", "https://github.com/"],
)
def test_parse_repo_url_rejects_bad_input(url):
    with pytest.raises(InvalidRepoURL):
        parse_repo_url(url)
