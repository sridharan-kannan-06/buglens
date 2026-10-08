"""The BugLens pipeline: repo URL + screenshot + one sentence -> AnalysisResult.

Stages: resolve -> load_repo -> index -> context -> vision -> retrieval -> rerank -> write -> similar.
At most three LLM calls are made (vision, rerank, write). A JSON repair retry or a
backoff retry after a 429 repeats the same step; it is not a fourth step.
The last stage (similar existing issues) uses the GitHub API and local embeddings only.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from buglens import config
from buglens.embeddings import Embedder, get_embedder
from buglens.errors import BugLensError, EmptyRetrieval, UnsupportedImage
from buglens.github_client import GitHubClient, parse_repo_url
from buglens.index import RepoIndex, load_or_build_index
from buglens.llm import get_llm
from buglens.llm.base import LLMClient
from buglens.models import AnalysisResult, IssueDraft, RepoContext, RepoRef, ScreenshotAnalysis, SimilarIssue
from buglens.repo_loader import ensure_repo, repo_cache_key
from buglens.rerank import rerank
from buglens.retrieval import retrieve
from buglens.similar_issues import find_similar_issues
from buglens.templates import read_repo_context
from buglens.vision import analyze_screenshot, prepare_image, text_only_analysis
from buglens.writer import write_issue_and_brief

# (stage name, "start" | "progress" | "done", detail text)
ProgressFn = Callable[[str, str, str], None]

STAGE_LABELS = {
    "resolve": "Resolving the repository",
    "load_repo": "Downloading the repository",
    "index": "Indexing source files",
    "context": "Reading the issue template and labels",
    "vision": "Reading the screenshot with Gemma",
    "retrieval": "Searching for related code",
    "rerank": "Ranking suspected files with Gemma",
    "write": "Writing the issue draft and brief",
    "similar": "Checking for similar existing issues",
}


class Run:
    """Collects stage timings and warnings, and reports progress to the caller."""

    def __init__(self, on_progress: ProgressFn | None = None) -> None:
        self.timings: dict[str, float] = {}
        self.warnings: list[str] = []
        self.detail = ""
        self.current = ""
        self._on_progress = on_progress

    def notify(self, stage: str, event: str, detail: str = "") -> None:
        if self._on_progress:
            self._on_progress(stage, event, detail)

    def notify_retry(self, status: int, delay: float) -> None:
        """Called by the LLM backend before it waits and retries (rate limit or server error)."""
        self.notify(self.current, "progress", f"model API answered {status}; retrying in {delay:.0f}s")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage. Set `self.detail` inside the block to describe the outcome."""
        self.detail = ""
        self.current = name
        self.notify(name, "start")
        started = time.perf_counter()
        yield
        self.timings[name] = round(time.perf_counter() - started, 3)
        self.notify(name, "done", self.detail)


@dataclass
class PreparedRepo:
    """A repo that has been resolved, downloaded and indexed."""

    repo: RepoRef
    ref_name: str
    sha: str
    repo_dir: Path
    index: RepoIndex


def prepare_repo(
    run: Run,
    repo_url: str,
    ref: str | None,
    github: GitHubClient,
    embedder: Embedder | None,
    cache_root: Path,
    max_chunks: int,
) -> PreparedRepo:
    """Stages 1-4: resolve the ref, download the zipball, chunk and embed (all cached)."""
    with run.stage("resolve"):
        repo = parse_repo_url(repo_url)
        if ref:
            repo = repo.model_copy(update={"ref": ref})
        ref_name, sha = github.resolve_ref(repo)
        run.detail = f"{repo.full_name} @ {sha[:7]}"

    with run.stage("load_repo"):
        repo_dir = ensure_repo(github, repo, sha, cache_root)

    with run.stage("index"):
        embedder = embedder or get_embedder()
        index, from_cache = load_or_build_index(
            repo_dir,
            cache_root / "index" / repo_cache_key(repo, sha),
            embedder,
            max_chunks,
            on_progress=lambda done, total: run.notify("index", "progress", f"embedded {done}/{total} chunks"),
            reuse_folder=cache_root / "embeddings" / f"{repo.owner}__{repo.repo}",
        )
        run.detail = f"{len(index.chunks)} chunks from {index.files_indexed} files"
        if from_cache:
            run.detail += " (cached)"
        elif index.reused_chunks:
            run.detail += f" ({index.reused_chunks} reused from another commit)"
        if index.truncated:
            run.warnings.append(
                f"Large repository: only {index.files_indexed} of {index.files_total} files were indexed "
                f"(chunk cap {max_chunks}). Source code is indexed first; docs and tests may be missing."
            )
    return PreparedRepo(repo=repo, ref_name=ref_name, sha=sha, repo_dir=repo_dir, index=index)


def _read_context(run: Run, github: GitHubClient, prepared: PreparedRepo) -> RepoContext:
    """Stage: issue template, CONTRIBUTING, README and the repo's real labels."""
    context = read_repo_context(prepared.repo_dir)
    try:
        context.labels = github.list_labels(prepared.repo)
        context.labels_available = True
    except BugLensError as exc:  # labels are optional: degrade instead of failing the run
        run.warnings.append(f"Could not fetch the repository's labels, so none are suggested ({exc}).")
    run.detail = f"template: {context.template.name}; {len(context.labels)} labels"
    return context


def analysis_text(analysis: ScreenshotAnalysis, used_image: bool) -> str:
    """How the screenshot analysis is shown to the rerank and write prompts."""
    if not used_image:
        return "No screenshot was analysed. Only USER_DESCRIPTION is available."
    return analysis.model_dump_json(indent=2)


def similar_issue_query(issue: IssueDraft, user_text: str, analysis: ScreenshotAnalysis) -> str:
    """The text that existing issues are compared with."""
    parts = [issue.title, user_text, analysis.apparent_problem, *analysis.error_messages[:3]]
    return "\n".join(part for part in parts if part)


def _check_similar(
    run: Run,
    github: GitHubClient,
    embedder: Embedder,
    repo: RepoRef,
    query: str,
) -> list[SimilarIssue]:
    """Stage: existing issues that read like this report. Optional, so problems become warnings."""
    similar, warnings = find_similar_issues(github, embedder, repo, query)
    run.warnings.extend(warnings)
    run.detail = f"{len(similar)} similar issue(s)"
    return similar


def analyze(
    repo_url: str,
    user_text: str,
    screenshot: bytes | str | Path | None = None,
    *,
    use_image: bool = True,
    ref: str | None = None,
    check_similar: bool = True,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    github: GitHubClient | None = None,
    cache_root: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> AnalysisResult:
    """Run the whole pipeline. Raises a BugLensError subclass with a friendly message on failure."""
    settings = config.load_settings()
    run = Run(on_progress)
    cache_root = cache_root or config.cache_dir()

    # Cheap checks first, so the user hears about bad input before any download starts.
    user_text = (user_text or "").strip()
    if not user_text:
        raise BugLensError("Please describe the bug in one sentence.")
    image_png = None
    if use_image:
        if screenshot is None:
            raise UnsupportedImage("No screenshot was provided.")
        image_png = prepare_image(screenshot)
    else:
        run.warnings.append("The screenshot was not analysed (text-only mode).")
    llm = llm or get_llm(settings)
    llm.on_retry = run.notify_retry
    github = github or GitHubClient(settings.github_token)
    if not github.authenticated:
        run.warnings.append(config.UNAUTHENTICATED_WARNING)

    prepared = prepare_repo(run, repo_url, ref, github, embedder, cache_root, settings.max_chunks)
    embedder = embedder or get_embedder()

    with run.stage("context"):
        context = _read_context(run, github, prepared)

    with run.stage("vision"):
        if image_png is not None:
            analysis = analyze_screenshot(llm, image_png, user_text)  # LLM call 1
            run.detail = (
                f"{len(analysis.visible_text)} strings, {len(analysis.file_hints)} file names, "
                f"{len(analysis.search_queries)} search queries"
            )
        else:
            analysis = text_only_analysis(user_text)
            run.detail = "skipped"

    with run.stage("retrieval"):
        candidates = retrieve(prepared.index, embedder, user_text, analysis)
        if not candidates:
            raise EmptyRetrieval(f"No searchable source files were found in {prepared.repo.full_name}.")
        run.detail = f"{len(candidates)} candidate files"

    prompt_analysis = analysis_text(analysis, use_image)
    with run.stage("rerank"):
        ranked_files, warnings = rerank(llm, user_text, prompt_analysis, candidates)  # LLM call 2
        run.warnings.extend(warnings)
        run.detail = f"{len(ranked_files)} suspected files"

    with run.stage("write"):
        issue, brief, warnings = write_issue_and_brief(llm, user_text, prompt_analysis, ranked_files, context)  # LLM call 3
        run.warnings.extend(warnings)

    similar: list[SimilarIssue] = []
    if check_similar:
        with run.stage("similar"):
            query = similar_issue_query(issue, user_text, analysis)
            similar = _check_similar(run, github, embedder, prepared.repo, query)

    return AnalysisResult(
        repo=prepared.repo.full_name,
        repo_url=prepared.repo.url,
        ref=prepared.ref_name,
        sha=prepared.sha,
        user_text=user_text,
        used_image=use_image,
        llm_backend=llm.name,
        llm_model=llm.model,
        template_name=context.template.name,
        screenshot_analysis=analysis,
        candidate_paths=[candidate.path for candidate in candidates],
        ranked_files=ranked_files,
        issue=issue,
        brief=brief,
        similar_issues=similar,
        files_indexed=prepared.index.files_indexed,
        chunks_indexed=len(prepared.index.chunks),
        timings=run.timings,
        warnings=run.warnings,
    )


def build_index_only(
    repo_url: str,
    *,
    ref: str | None = None,
    embedder: Embedder | None = None,
    github: GitHubClient | None = None,
    cache_root: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> tuple[PreparedRepo, Run]:
    """Download and index a repo without calling the LLM (used by `buglens index`)."""
    settings = config.load_settings()
    run = Run(on_progress)
    github = github or GitHubClient(settings.github_token)
    if not github.authenticated:
        run.warnings.append(config.UNAUTHENTICATED_WARNING)
    prepared = prepare_repo(
        run, repo_url, ref, github, embedder, cache_root or config.cache_dir(), settings.max_chunks
    )
    return prepared, run
