from __future__ import annotations

import zipfile

from buglens import config
from buglens.models import RepoRef
from buglens.repo_loader import (
    PRIORITY_CODE,
    PRIORITY_DOCS_TESTS,
    PRIORITY_OTHER,
    ensure_repo,
    extract_zipball,
    file_priority,
    is_wanted,
    looks_minified,
    read_source_files,
)
from tests.fakes import FAKE_SHA, FakeGitHub, zip_folder


def test_is_wanted_keeps_source_files():
    assert is_wanted("src/app/login.py", 500)
    assert is_wanted("web/components/Button.tsx", 500)
    assert is_wanted(".github/ISSUE_TEMPLATE/bug_report.yml", 500)
    assert is_wanted("README.md", 500)


def test_is_wanted_skips_junk():
    assert not is_wanted("node_modules/react/index.js", 500)
    assert not is_wanted("web/vendor/lib.js", 500)
    assert not is_wanted("app/dist/bundle.js", 500)
    assert not is_wanted("pkg/__pycache__/mod.py", 500)
    assert not is_wanted(".next/server/page.js", 500)
    assert not is_wanted(".git/config", 500)
    assert not is_wanted("package-lock.json", 500)
    assert not is_wanted("frontend/yarn.lock", 500)
    assert not is_wanted("static/app.min.js", 500)
    assert not is_wanted("static/logo.png", 500)
    assert not is_wanted("fonts/Inter.woff2", 500)
    assert not is_wanted("src/empty.py", 0)
    assert not is_wanted("src/huge.py", config.MAX_FILE_BYTES + 1)


def test_file_priority_prefers_code_over_docs_and_tests():
    assert file_priority("src/app/login.py") == PRIORITY_CODE
    assert file_priority("web/styles/main.css") == PRIORITY_CODE
    assert file_priority("settings.toml") == PRIORITY_OTHER
    assert file_priority("locales/en.json") == PRIORITY_OTHER
    assert file_priority("README.md") == PRIORITY_DOCS_TESTS
    assert file_priority("docs/guide/setup.py") == PRIORITY_DOCS_TESTS
    assert file_priority("tests/test_login.py") == PRIORITY_DOCS_TESTS
    assert file_priority("src/Button.test.tsx") == PRIORITY_DOCS_TESTS


def test_looks_minified():
    assert looks_minified("x" * 5000)
    assert not looks_minified("short line\n" * 100)


def test_read_source_files_filters_and_sorts(fixture_repo):
    files = read_source_files(fixture_repo)
    paths = [item.path for item in files]
    assert paths == [
        "src/app/cart.py",
        "src/app/login.py",
        "src/app/templates/login.html",
        "CONTRIBUTING.md",
        "README.md",
        "tests/test_login.py",
    ]
    assert all("\\" not in path for path in paths)


def test_extract_zipball_strips_top_folder_and_blocks_path_traversal(tmp_path, fixture_repo):
    archive_path = tmp_path / "repo.zip"
    zip_folder(fixture_repo, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("owner-repo-0123456/../../escaped.py", "print('escaped')\n")
        archive.writestr("owner-repo-0123456/sub/../../also_escaped.py", "print('escaped')\n")

    dest = tmp_path / "extracted"
    extract_zipball(archive_path, dest)

    assert (dest / "src/app/login.py").is_file()
    assert (dest / ".github/ISSUE_TEMPLATE/bug_report.md").is_file()
    assert not (dest / "node_modules").exists()
    assert not (dest / "static/logo.png").exists()
    assert not (dest / "package-lock.json").exists()
    assert not list(tmp_path.rglob("escaped.py"))
    assert not list(tmp_path.rglob("also_escaped.py"))


def test_ensure_repo_downloads_once(tmp_path, fixture_repo):
    github = FakeGitHub(fixture_repo)
    repo = RepoRef(owner="owner", repo="repo")

    first = ensure_repo(github, repo, FAKE_SHA, tmp_path / "cache")
    second = ensure_repo(github, repo, FAKE_SHA, tmp_path / "cache")

    assert first == second == tmp_path / "cache" / "repos" / f"owner__repo__{FAKE_SHA}"
    assert (first / "src/app/cart.py").is_file()
    assert github.downloads == 1
    assert not list((tmp_path / "cache" / "repos").glob("*.zip"))
