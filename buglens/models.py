"""Pydantic models shared by every stage of the pipeline.

The models that describe LLM replies are deliberately forgiving (a single string
is accepted where a list is expected, missing lists become empty) so that small
formatting slips by the model do not cost a repair retry.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, field_validator


def _as_text(value: Any) -> str:
    """None becomes an empty string; everything else is stringified and trimmed."""
    return "" if value is None else str(value).strip()


def _as_text_list(value: Any) -> list[str]:
    """Accept None, a single string or a list, and return a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    return [text for text in (_as_text(item) for item in value) if text]


Text = Annotated[str, BeforeValidator(_as_text)]
TextList = Annotated[list[str], BeforeValidator(_as_text_list)]


class RepoRef(BaseModel):
    """A GitHub repository, optionally pinned to a branch, tag or commit."""

    owner: str
    repo: str
    ref: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


class Chunk(BaseModel):
    """A window of lines from one file."""

    path: str
    start_line: int
    end_line: int
    text: str


class ScreenshotAnalysis(BaseModel):
    """What the vision model saw in the screenshot (stage 5)."""

    visible_text: TextList = Field(default_factory=list)
    ui_components: TextList = Field(default_factory=list)
    error_messages: TextList = Field(default_factory=list)
    apparent_problem: Text = ""
    framework_hints: TextList = Field(default_factory=list)
    search_queries: TextList = Field(default_factory=list)
    file_hints: TextList = Field(default_factory=list)  # e.g. "src/views/login.py:42" from a stack trace
    identifiers: TextList = Field(default_factory=list)  # e.g. "handleSubmit", "LoginForm"


class CandidateFile(BaseModel):
    """A file proposed by hybrid retrieval, with the evidence behind its score."""

    path: str
    score: float
    semantic_score: float = 0.0
    hit_bonus: float = 0.0
    lexical_bonus: float = 0.0
    path_bonus: float = 0.0
    matched_strings: list[str] = Field(default_factory=list)
    path_hints: list[str] = Field(default_factory=list)  # screenshot file names that point at this file
    hint_lines: list[int] = Field(default_factory=list)  # line numbers named with those file names
    snippets: list[Chunk] = Field(default_factory=list)


class Evidence(BaseModel):
    """The code excerpt that made BugLens propose a file. It comes from search, not from the model."""

    start_line: int
    end_line: int
    focus_line: int | None = None  # the line that holds an on-screen string or was named in a stack trace
    snippet: str
    matched_strings: list[str] = Field(default_factory=list)
    path_hints: list[str] = Field(default_factory=list)


class RankedFile(BaseModel):
    """One suspected file chosen by the reranker. `evidence` is attached afterwards by code."""

    path: Text
    reason: Text = ""
    confidence: float = 0.0
    evidence: Evidence | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        """Keep confidence in 0..1. Values such as 85 are read as percentages."""
        if 1.0 < value <= 100.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))


class RerankReply(BaseModel):
    """Raw reply of the rerank call."""

    files: list[RankedFile] = Field(default_factory=list)


class IssueDraft(BaseModel):
    title: Text = Field(min_length=1)
    body_markdown: Text = Field(min_length=1)
    labels: TextList = Field(default_factory=list)


class StartPoint(BaseModel):
    path: Text
    what_to_look_for: Text = ""


class ContributorBrief(BaseModel):
    summary: Text = Field(min_length=1)
    where_to_start: list[StartPoint] = Field(default_factory=list)
    how_to_verify: TextList = Field(default_factory=list)
    caveats: TextList = Field(default_factory=list)


class WriterReply(BaseModel):
    """Raw reply of the write call: the issue and the brief together."""

    issue: IssueDraft
    brief: ContributorBrief


class IssueTemplate(BaseModel):
    """The repo's issue template, normalised to markdown."""

    name: str
    source: str  # path inside the repo, or "default"
    body_markdown: str
    sections: list[str] = Field(default_factory=list)
    title_prefix: str = ""
    default_labels: list[str] = Field(default_factory=list)


class RepoContext(BaseModel):
    """Everything the writer needs to know about the repo's conventions."""

    template: IssueTemplate
    contributing: str = ""
    readme_excerpt: str = ""
    labels: list[str] = Field(default_factory=list)
    labels_available: bool = False


class SimilarIssue(BaseModel):
    """An existing issue in the repository that reads like the new report."""

    number: int
    title: str
    url: str
    state: str  # "open" or "closed"
    similarity: float  # cosine similarity of the embeddings, 0..1


class AnalysisResult(BaseModel):
    """The full output of one BugLens run."""

    repo: str  # "owner/name"
    repo_url: str
    ref: str  # branch, tag or sha that was analysed
    sha: str
    user_text: str
    used_image: bool
    llm_backend: str
    llm_model: str
    template_name: str
    screenshot_analysis: ScreenshotAnalysis
    candidate_paths: list[str]  # the files retrieval handed to the reranker
    ranked_files: list[RankedFile]
    issue: IssueDraft
    brief: ContributorBrief
    similar_issues: list[SimilarIssue] = Field(default_factory=list)
    files_indexed: int = 0
    chunks_indexed: int = 0
    timings: dict[str, float]  # seconds per stage
    warnings: list[str]

    def file_url(self, path: str) -> str:
        """Permanent GitHub link to a file at the analysed commit."""
        return f"{self.repo_url}/blob/{self.sha}/{path}"

    def evidence_url(self, item: RankedFile) -> str:
        """Like file_url, but jumps to the lines of the evidence when there is any."""
        url = self.file_url(item.path)
        if item.evidence is None:
            return url
        if item.evidence.focus_line:
            return f"{url}#L{item.evidence.focus_line}"
        return f"{url}#L{item.evidence.start_line}-L{item.evidence.end_line}"
