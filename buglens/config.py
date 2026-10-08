"""All tunable numbers and environment settings live here.

Constants are plain module-level names so they are easy to find and explain.
Secrets are only ever read from the environment (or a local .env file).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from buglens.errors import ConfigError

# ---------------------------------------------------------------- repo loading
MAX_ZIP_BYTES = 60 * 1024 * 1024  # refuse zipballs larger than 60 MB
MAX_FILE_BYTES = 200 * 1024  # skip single files larger than 200 KB
MAX_EXTRACTED_BYTES = 300 * 1024 * 1024  # stop extracting after this much text (zip-bomb guard)
DEFAULT_MAX_CHUNKS = 8000  # cap on indexed chunks per repo (override: BUGLENS_MAX_CHUNKS)

# Directories that are never extracted.
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", ".next", "__pycache__",
    ".venv", "venv", "target", "out", "coverage", ".idea", ".vscode", "bower_components",
}
# Directories that are extracted (we need the issue templates) but not indexed for search.
NO_INDEX_DIRS = {".github"}

LOCKFILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "pipfile.lock",
    "cargo.lock", "composer.lock", "gemfile.lock", "go.sum", "uv.lock", "bun.lockb",
    "npm-shrinkwrap.json", "pdm.lock", "flake.lock", "packages.lock.json",
}

# Images, fonts, archives, media, compiled files and bulk data are never text we want.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".tiff", ".psd", ".avif",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".whl",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".ogg", ".webm", ".flac",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".pyc", ".pyd", ".wasm",
    ".db", ".sqlite", ".sqlite3", ".pkl", ".npy", ".npz", ".parquet", ".onnx", ".pt", ".h5",
    ".csv", ".tsv", ".ipynb", ".map", ".lock", ".snap",
}

# Priority 0: source code, including the markup and styles where UI bugs usually live.
CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte", ".astro",
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".java", ".kt", ".kts", ".go",
    ".rs", ".rb", ".php", ".cs", ".c", ".h", ".cc", ".cpp", ".hpp", ".swift", ".m",
    ".dart", ".scala", ".sh", ".sql", ".ex", ".exs", ".erb", ".ejs", ".hbs", ".jinja",
    ".j2", ".twig", ".lua", ".r", ".pl", ".elm", ".clj", ".hs", ".graphql", ".proto",
}
# Priority 2: documentation. Everything else that is text gets priority 1 (config, i18n, ...).
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".adoc"}
TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs", "e2e", "fixtures", "testdata"}
DOC_DIR_NAMES = {"docs", "doc", "documentation", "examples", "example"}

MINIFIED_AVG_LINE_CHARS = 400  # a file whose average line is longer than this looks generated

# -------------------------------------------------------------------- chunking
CHUNK_LINES = 60
CHUNK_OVERLAP = 10

# ------------------------------------------------------------------ embeddings
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 64
EMBED_CACHE_MAX_VECTORS = 40000  # vectors remembered per repo so other commits can reuse them

# ------------------------------------------------------------------ screenshot
MAX_IMAGE_SIDE = 1600  # longest side in pixels after resizing
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_SEARCH_QUERIES = 6
MAX_LEXICAL_STRINGS = 40  # cap on screenshot strings used for exact matching
MAX_FILE_HINTS = 10  # file names read from the screenshot (stack traces, editor tabs)
MAX_IDENTIFIERS = 20  # function, class and component names read from the screenshot

# ------------------------------------------------------------------- retrieval
TOP_CHUNKS_PER_QUERY = 20  # semantic hits kept for each query
EXTRA_HIT_BONUS = 0.03  # added per extra top-chunk hit in the same file
MAX_EXTRA_HITS = 5  # cap so huge files do not win on size alone
LEXICAL_WEIGHT = 0.25  # bonus for one screenshot string found in exactly one file
ERROR_MESSAGE_BOOST = 1.5  # error messages count more than ordinary visible text
LEXICAL_BONUS_CAP = 0.6  # cap on the total lexical bonus of one file
MIN_LEXICAL_CHARS = 4  # ignore very short strings such as "OK"
IDENTIFIER_BOOST = 1.2  # code names seen on screen count a little more than UI text
PATH_HINT_BONUS = 0.5  # a file named on screen (stack trace, editor tab) is strong evidence
PATH_HINT_BONUS_CAP = 0.75  # cap on the total path bonus of one file
MAX_PATH_HINT_MATCHES = 8  # a name such as "index.js" that fits more files than this is ignored
TOP_CANDIDATE_FILES = 12  # files passed on to the reranker
SNIPPETS_PER_FILE = 2  # best chunks kept per candidate file

# -------------------------------------------------------------------- evidence
EVIDENCE_CONTEXT_LINES = 6  # lines shown above and below the line that matched
EVIDENCE_MAX_LINES = 25  # excerpt length when no single line stands out

# -------------------------------------------------------------- similar issues
SIMILAR_ISSUES_POOL = 100  # most recently updated issues that are compared
SIMILAR_ISSUES_SEARCH = 20  # extra issues fetched with a keyword search
SIMILAR_ISSUES_TOP = 5  # how many similar issues are shown
SIMILAR_ISSUE_MIN_SCORE = 0.70  # cosine similarity below this is not shown (tuned by eye on one repo)
SIMILAR_ISSUE_BODY_CHARS = 600  # how much of an issue body is compared
SEARCH_KEYWORDS = 3  # words used in the keyword search (GitHub requires all of them to match)

# ----------------------------------------------------------------- issue link
MAX_PREFILL_URL_CHARS = 7000  # longer "new issue" links are cut down to title and labels

# ---------------------------------------------------------------------- rerank
RERANK_CHARS_PER_FILE = 1800  # hard cap on snippet characters per candidate file
RERANK_TOTAL_CHARS = 16000  # hard cap on snippet characters in the whole prompt
MAX_RANKED_FILES = 5

# ---------------------------------------------------------------------- writer
README_CHARS = 3000
CONTRIBUTING_CHARS = 2000
TEMPLATE_CHARS = 4000
MAX_LABELS = 100

# ------------------------------------------------------------------------- LLM
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 4096
LLM_MAX_RETRIES = 4  # retries after the first attempt, on 429 and 5xx only
LLM_BACKOFF_SECONDS = 3.0  # wait 3s, 6s, 12s, 24s (plus jitter)
LLM_TIMEOUT_SECONDS = 120
REPAIR_REPLY_CHARS = 4000  # how much of a bad reply is sent back in the repair prompt

# ---------------------------------------------------------------------- GitHub
GITHUB_API_ROOT = "https://api.github.com"
GITHUB_TIMEOUT_SECONDS = 60
UNAUTHENTICATED_WARNING = (
    "GITHUB_TOKEN is not set: unauthenticated GitHub API calls are limited to 60 per hour."
)

SUPPORTED_BACKENDS = ("gemini", "ollama")
THINKING_LEVELS = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class Settings:
    """Values read from the environment. Never log or print the secret fields."""

    gemini_api_key: str
    gemma_model: str
    llm_backend: str
    github_token: str
    ollama_host: str
    native_json: bool
    max_chunks: int
    thinking_level: str = ""  # "" keeps the model's default; e.g. "minimal" for faster replies


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_settings() -> Settings:
    """Read settings from the environment (and .env if present). Does not validate."""
    load_dotenv()
    try:
        max_chunks = int(_env("BUGLENS_MAX_CHUNKS") or DEFAULT_MAX_CHUNKS)
    except ValueError as exc:
        raise ConfigError("BUGLENS_MAX_CHUNKS must be a whole number.") from exc
    return Settings(
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemma_model=_env("GEMMA_MODEL"),
        llm_backend=(_env("LLM_BACKEND") or "gemini").lower(),
        github_token=_env("GITHUB_TOKEN"),
        ollama_host=_env("OLLAMA_HOST") or "http://localhost:11434",
        native_json=_env("BUGLENS_NATIVE_JSON").lower() in {"1", "true", "yes", "on"},
        max_chunks=max_chunks,
        thinking_level=_env("BUGLENS_THINKING_LEVEL").lower(),
    )


def check_llm_settings(settings: Settings) -> None:
    """Raise ConfigError with a clear message if the LLM settings are unusable."""
    if settings.llm_backend not in SUPPORTED_BACKENDS:
        raise ConfigError(
            f"LLM_BACKEND must be one of {', '.join(SUPPORTED_BACKENDS)} (got '{settings.llm_backend}')."
        )
    if settings.llm_backend == "gemini" and not settings.gemini_api_key:
        raise ConfigError("GEMINI_API_KEY is not set.")
    if settings.thinking_level and settings.thinking_level not in THINKING_LEVELS:
        raise ConfigError(
            f"BUGLENS_THINKING_LEVEL must be empty or one of {', '.join(THINKING_LEVELS)} "
            f"(got '{settings.thinking_level}')."
        )
    if not settings.gemma_model:
        raise ConfigError(
            "GEMMA_MODEL is not set. BugLens does not guess a model id.",
            hint="Run `python scripts/check_gemma.py` to list the exact Gemma model ids, then set GEMMA_MODEL in .env.",
        )


def cache_dir() -> Path:
    """Root folder for downloaded repos, indexes and the embedding model."""
    return Path(_env("BUGLENS_CACHE_DIR") or ".cache")
