"""BugLens web UI. Run with: streamlit run app.py"""
from __future__ import annotations

import logging
import os
from pathlib import PurePosixPath

import streamlit as st
from dotenv import load_dotenv

from buglens import config, pipeline
from buglens.errors import (
    BugLensError,
    ConfigError,
    EmptyReply,
    EmptyRetrieval,
    GitHubRateLimit,
    LLMRateLimit,
    RepoNotFound,
    RepoTooLarge,
    UnsupportedImage,
)
from buglens.models import AnalysisResult, RankedFile
from buglens.report import DISCLAIMER, brief_markdown, issue_markdown, new_issue_url

ERROR_TITLES = {
    ConfigError: "Configuration is incomplete",
    LLMRateLimit: "The model API is rate limited",
    EmptyReply: "The model API returned no text",
    GitHubRateLimit: "GitHub rate limit reached",
    RepoNotFound: "Repository not found or private",
    RepoTooLarge: "Repository is too large",
    UnsupportedImage: "Unsupported screenshot",
    EmptyRetrieval: "Nothing to search",
}

# File extension -> syntax highlighting name for st.code.
CODE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx",
    ".html": "html", ".vue": "html", ".css": "css", ".scss": "scss", ".json": "json",
    ".java": "java", ".kt": "kotlin", ".go": "go", ".rb": "ruby", ".php": "php", ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp", ".swift": "swift", ".sh": "bash",
    ".sql": "sql", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml", ".md": "markdown",
}


def show_sidebar() -> None:
    """Show which environment variables are set. Values are never displayed."""
    st.sidebar.header("Configuration")
    uses_gemini = os.environ.get("LLM_BACKEND", "").strip().lower() != "ollama"
    variables = (
        ("GEMINI_API_KEY", uses_gemini),
        ("GEMMA_MODEL", True),
        ("LLM_BACKEND", False),
        ("GITHUB_TOKEN", False),
    )
    for name, required in variables:
        if os.environ.get(name, "").strip():
            st.sidebar.markdown(f":green[set] `{name}`")
        elif required:
            st.sidebar.markdown(f":red[missing] `{name}`")
        else:
            st.sidebar.markdown(f":gray[not set] `{name}` (optional)")
    st.sidebar.caption(
        "Values are read from the environment or a local .env file and are never shown. "
        "LLM_BACKEND defaults to gemini. "
        "Without GITHUB_TOKEN, GitHub allows 60 API calls per hour."
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "BugLens finds files that are likely related to a bug. It does not fix the bug, "
        "and it can be wrong."
    )


def show_error(error: BugLensError) -> None:
    """Friendly error box with a hint about what to do next."""
    title = next((text for kind, text in ERROR_TITLES.items() if isinstance(error, kind)), "BugLens could not finish")
    st.error(f"**{title}.** {error}")
    if error.hint:
        st.info(error.hint)


def run_analysis(repo_url: str, screenshot: bytes, user_text: str) -> AnalysisResult | None:
    """Run the pipeline and show one line per finished stage."""
    with st.status("Analysing...", expanded=True) as status:

        def on_progress(stage: str, event: str, detail: str) -> None:
            label = pipeline.STAGE_LABELS.get(stage, stage)
            if event == "start":
                status.update(label=f"{label}...")
            elif event == "progress":
                status.update(label=f"{label}: {detail}")
            else:
                st.write(f"{label}" + (f": {detail}" if detail else ""))

        try:
            result = pipeline.analyze(repo_url, user_text, screenshot, on_progress=on_progress)
        except BugLensError as error:
            status.update(label="Stopped", state="error")
            show_error(error)
            return None
        except Exception as error:  # last resort: log the traceback to the terminal, keep the page tidy
            logging.getLogger("buglens").exception("Unexpected error during analysis")
            status.update(label="Stopped", state="error")
            st.error(f"**Unexpected error.** {type(error).__name__}. The details are in the terminal running Streamlit.")
            return None
        status.update(label="Done", state="complete", expanded=False)
    return result


def show_metrics(result: AnalysisResult) -> None:
    """One row of numbers about this run."""
    files, chunks, suspects, similar, seconds = st.columns(5)
    files.metric("Files searched", result.files_indexed)
    chunks.metric("Code chunks", result.chunks_indexed)
    suspects.metric("Suspected files", len(result.ranked_files))
    similar.metric("Similar issues", len(result.similar_issues))
    seconds.metric("Total time", f"{sum(result.timings.values()):.0f} s")


def show_issue(result: AnalysisResult) -> None:
    st.subheader(result.issue.title)
    if result.issue.labels:
        st.markdown("Suggested labels: " + " ".join(f"`{label}`" for label in result.issue.labels))
    else:
        st.caption("No labels suggested.")
    st.caption(f"Follows the repository's issue template: {result.template_name}")
    st.markdown(result.issue.body_markdown)
    st.divider()

    url, body_included = new_issue_url(result)
    st.link_button("Open this draft on GitHub", url)
    filled = "title, labels and body" if body_included else "title and labels (the body is too long for a link: paste it from below)"
    st.caption(
        f"Opens GitHub's new-issue form with the {filled} filled in. Nothing is filed until you press "
        "Submit there. Labels only apply if you may label issues in that repository, and a repository "
        "that does not allow blank issues shows its template chooser instead."
    )
    if result.similar_issues:
        st.info(f"{len(result.similar_issues)} existing issue(s) look similar. Check the Similar issues tab before filing.")

    st.caption("Copy-ready title")
    st.code(result.issue.title, language=None)
    st.caption("Copy-ready body (markdown)")
    st.code(result.issue.body_markdown, language="markdown")
    st.download_button("Download issue.md", issue_markdown(result), file_name="issue.md", mime="text/markdown")


def show_evidence(result: AnalysisResult, rank: int, item: RankedFile) -> None:
    """Why one file is suspected: the model's reason plus the code that search matched."""
    with st.expander(f"{rank}. {item.path}", expanded=rank == 1):
        st.write(item.reason)
        evidence = item.evidence
        if evidence is None:
            st.caption("No code excerpt is available for this file.")
            return
        if evidence.path_hints:
            st.text("Named in the screenshot: " + ", ".join(evidence.path_hints))
        if evidence.matched_strings:
            st.text("On-screen text found in this file: " + " | ".join(evidence.matched_strings))
        where = f"lines {evidence.start_line}-{evidence.end_line}"
        if evidence.focus_line:
            where += f", match on line {evidence.focus_line}"
        st.caption(f"Excerpt ({where}). Found by code search, not written by the model.")
        st.code(evidence.snippet, language=CODE_LANGUAGES.get(PurePosixPath(item.path).suffix.lower()))
        st.markdown(f"[Open these lines on GitHub]({result.evidence_url(item)})")


def show_files(result: AnalysisResult) -> None:
    st.caption("Files that may be related to the bug, most likely first. These are suggestions, not a diagnosis.")
    rows = [
        {
            "#": rank,
            "Path": item.path,
            "Reason": item.reason,
            "Confidence": item.confidence,
            "Link": result.evidence_url(item),
        }
        for rank, item in enumerate(result.ranked_files, start=1)
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "Link": st.column_config.LinkColumn("Link", display_text="Open on GitHub"),
        },
    )
    st.caption(f"Links point at commit `{result.sha[:12]}` of {result.repo}.")
    st.markdown("**Evidence**")
    for rank, item in enumerate(result.ranked_files, start=1):
        show_evidence(result, rank, item)


def show_brief(result: AnalysisResult) -> None:
    markdown = brief_markdown(result)
    st.markdown(markdown)
    st.download_button("Download brief.md", markdown, file_name="brief.md", mime="text/markdown")


def show_similar(result: AnalysisResult) -> None:
    if not result.similar_issues:
        st.write(
            "No similar issue was found among the most recently updated issues and a keyword search. "
            "That does not prove the bug is new."
        )
        return
    st.caption("Existing issues that read like this report. Check them before filing, in case it is a duplicate.")
    rows = [
        {
            "Issue": f"#{issue.number}",
            "Title": issue.title,
            "State": issue.state,
            "Similarity": issue.similarity,
            "Link": issue.url,
        }
        for issue in result.similar_issues
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.2f"),
            "Link": st.column_config.LinkColumn("Link", display_text="Open on GitHub"),
        },
    )
    st.caption(
        f"Compared with a local embedding model; issues scoring under {config.SIMILAR_ISSUE_MIN_SCORE:.2f} are not shown. "
        "A high score means similar wording, not a confirmed duplicate."
    )


def show_result(result: AnalysisResult, screenshot: bytes | None) -> None:
    st.warning("Suggestions, verify before filing. BugLens points at likely files; it does not fix the bug.")
    show_metrics(result)
    tabs = st.tabs(
        ["Issue draft", "Suspected files", "Contributor brief", f"Similar issues ({len(result.similar_issues)})"]
    )
    with tabs[0]:
        show_issue(result)
    with tabs[1]:
        show_files(result)
    with tabs[2]:
        show_brief(result)
    with tabs[3]:
        show_similar(result)

    with st.expander("What Gemma saw in your screenshot"):
        if result.used_image:
            image_column, analysis_column = st.columns(2)
            if screenshot:
                image_column.image(screenshot)
            analysis_column.json(result.screenshot_analysis.model_dump())
        else:
            st.write("The screenshot was not analysed in this run.")
    with st.expander("Stage timings and warnings"):
        st.table([{"Stage": stage, "Seconds": f"{seconds:.2f}"} for stage, seconds in result.timings.items()])
        st.caption(f"Model: {result.llm_model} via {result.llm_backend}")
        for warning in result.warnings:
            st.warning(warning)
        if not result.warnings:
            st.write("No warnings.")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="BugLens", page_icon=":mag:", layout="wide")
    show_sidebar()

    st.title("BugLens")
    st.write(
        "Give BugLens a public GitHub repository, a screenshot of a bug and one sentence. "
        "It drafts an issue in the repository's own template, lists the files most likely involved "
        "with the code that points there, checks for similar existing issues and writes a short "
        "brief for a new contributor."
    )

    repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")
    upload = st.file_uploader("Bug screenshot (PNG, JPG or WebP)", type=["png", "jpg", "jpeg", "webp"])
    user_text = st.text_input("What went wrong, in one sentence", placeholder="The save button does nothing on the settings page")

    if st.button("Analyze", type="primary"):
        if not repo_url.strip() or upload is None or not user_text.strip():
            st.error("Please fill in the repository URL, upload a screenshot and describe the bug.")
        else:
            result = run_analysis(repo_url, upload.getvalue(), user_text)
            if result is not None:
                st.session_state["result"] = result
                st.session_state["screenshot"] = upload.getvalue()
            else:
                st.session_state.pop("result", None)

    if "result" in st.session_state:
        show_result(st.session_state["result"], st.session_state.get("screenshot"))

    st.divider()
    st.caption(DISCLAIMER)


main()
