"""Download a repository zipball, extract the useful text files and list them for indexing."""
from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from buglens import config
from buglens.errors import GitHubError
from buglens.github_client import GitHubClient
from buglens.models import RepoRef

PRIORITY_CODE = 0
PRIORITY_OTHER = 1
PRIORITY_DOCS_TESTS = 2


@dataclass(frozen=True)
class SourceFile:
    """One text file from the repo. `path` is POSIX-style and relative to the repo root."""

    path: str
    text: str
    priority: int


def is_wanted(path: str, size: int) -> bool:
    """Decide from the path and size alone whether a file is worth extracting."""
    pure = PurePosixPath(path)
    name = pure.name.lower()
    if any(part in config.SKIP_DIRS for part in pure.parts[:-1]):
        return False
    if name in config.LOCKFILES:
        return False
    if pure.suffix.lower() in config.BINARY_EXTENSIONS:
        return False
    if ".min." in name or name.endswith("-min.js"):
        return False
    return 0 < size <= config.MAX_FILE_BYTES


def file_priority(path: str) -> int:
    """0 for source code, 2 for docs and tests, 1 for everything else (config, i18n, ...)."""
    pure = PurePosixPath(path)
    name = pure.name.lower()
    suffix = pure.suffix.lower()
    folders = {part.lower() for part in pure.parts[:-1]}
    is_test = (
        bool(folders & config.TEST_DIR_NAMES)
        or name.startswith("test_")
        or any(marker in name for marker in ("_test.", ".test.", ".spec.", "_spec."))
    )
    if is_test or suffix in config.DOC_EXTENSIONS or folders & config.DOC_DIR_NAMES:
        return PRIORITY_DOCS_TESTS
    if suffix in config.CODE_EXTENSIONS:
        return PRIORITY_CODE
    return PRIORITY_OTHER


def looks_binary(data: bytes) -> bool:
    """Text files do not contain NUL bytes."""
    return b"\0" in data[:4096]


def looks_minified(text: str) -> bool:
    """Generated or minified files have very long lines."""
    line_count = text.count("\n") + 1
    return len(text) / line_count > config.MINIFIED_AVG_LINE_CHARS


def _safe_relative_path(member_name: str) -> str | None:
    """Strip the zipball's top-level folder and reject paths that could escape the target."""
    parts = PurePosixPath(member_name.replace("\\", "/")).parts[1:]
    if not parts or any(part in ("..", "") or ":" in part for part in parts):
        return None
    return "/".join(parts)


def extract_zipball(zip_path: Path, dest_dir: Path) -> int:
    """Extract the wanted files of a GitHub zipball into `dest_dir`. Returns the file count."""
    extracted = 0
    total_bytes = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = _safe_relative_path(info.filename)
            if relative is None or not is_wanted(relative, info.file_size):
                continue
            total_bytes += info.file_size
            if total_bytes > config.MAX_EXTRACTED_BYTES:
                break
            target = dest_dir / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
            except OSError:
                continue  # for example a path too long for Windows; skip that file
            extracted += 1
    return extracted


def repo_cache_key(repo: RepoRef, sha: str) -> str:
    """Folder name used for both the repo cache and the index cache."""
    return f"{repo.owner}__{repo.repo}__{sha}"


def ensure_repo(client: GitHubClient, repo: RepoRef, sha: str, cache_root: Path) -> Path:
    """Return the folder holding the extracted repo, downloading it only if needed."""
    repos_dir = cache_root / "repos"
    dest_dir = repos_dir / repo_cache_key(repo, sha)
    done_marker = repos_dir / (dest_dir.name + ".ok")
    if done_marker.exists() and dest_dir.is_dir():
        return dest_dir

    shutil.rmtree(dest_dir, ignore_errors=True)  # remove a half-finished earlier attempt
    zip_path = repos_dir / (dest_dir.name + ".zip")
    client.download_zipball(repo, sha, zip_path, config.MAX_ZIP_BYTES)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        extract_zipball(zip_path, dest_dir)
    except zipfile.BadZipFile as exc:
        raise GitHubError("GitHub returned a file that is not a valid zip archive.") from exc
    finally:
        zip_path.unlink(missing_ok=True)
    done_marker.write_text("extracted", encoding="utf-8")
    return dest_dir


def read_source_files(repo_dir: Path) -> list[SourceFile]:
    """Read every indexable text file, sorted so code comes before config, docs and tests."""
    files: list[SourceFile] = []
    skip_dirs = config.SKIP_DIRS | config.NO_INDEX_DIRS
    for folder, dir_names, file_names in os.walk(repo_dir):
        dir_names[:] = sorted(name for name in dir_names if name not in skip_dirs)
        for file_name in sorted(file_names):
            full_path = Path(folder) / file_name
            relative = full_path.relative_to(repo_dir).as_posix()
            try:
                if not is_wanted(relative, full_path.stat().st_size):
                    continue
                data = full_path.read_bytes()
            except OSError:
                continue
            if looks_binary(data):
                continue
            text = data.decode("utf-8", errors="replace")
            priority = file_priority(relative)
            # Prose has long lines too, so the minified check is only applied to non-docs.
            if not text.strip() or (priority != PRIORITY_DOCS_TESTS and looks_minified(text)):
                continue
            files.append(SourceFile(path=relative, text=text, priority=priority))
    files.sort(key=lambda item: (item.priority, item.path))
    return files
