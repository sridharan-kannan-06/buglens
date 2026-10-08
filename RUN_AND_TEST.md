# Run and test BugLens

A checklist for the team. Work through it in order. Commands are written for a
terminal opened in the project folder. Where Windows differs, the PowerShell
form is given.

> BugLens never runs git, and nothing in this checklist needs git.

## Checklist

1. **Create the virtualenv.**

   ```bash
   python -m venv .venv
   ```

   Activate it. macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Install the dependencies** (inside the virtualenv, never globally).

   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` and fill in the keys.** macOS / Linux:

   ```bash
   cp .env.example .env
   ```

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Edit `.env`: set `GEMINI_API_KEY`. Optionally set `GITHUB_TOKEN` (any GitHub
   personal access token; no scopes are needed for public repositories). Leave
   `GEMMA_MODEL` empty for now. Never commit `.env`.

4. **Go/no-go: find the Gemma model id.**

   ```bash
   python scripts/check_gemma.py
   ```

   It lists every model with "gemma" in its name that your key can use. Copy
   the exact id of the vision-capable model you want into `GEMMA_MODEL` in
   `.env`. Then test an image prompt with any screenshot:

   ```bash
   python scripts/check_gemma.py --image path/to/screenshot.png --prompt "List the text visible in this screenshot."
   ```

   It prints the raw reply and ends with `GO` or `NO-GO`. If this step fails,
   nothing else will work, so fix it first.

5. **Run the offline tests.** No key and no network are needed.

   ```bash
   pytest -q
   ```

   Expected: all tests pass.

6. **Run the smoke test** on a small public repository, with a screenshot of a
   bug from that project (an image attached to one of its issues works well).

   ```bash
   python scripts/smoke_test.py --repo https://github.com/OWNER/REPO --screenshot path/to/screenshot.png --text "One sentence about the bug"
   ```

   It runs the real pipeline, prints the summary and five `[pass]` / `[FAIL]`
   checks. The first run on a repository downloads the embedding model (about
   65 MB, once) and embeds every chunk, which is the slow part.

7. **Try the command line.**

   ```bash
   python -m buglens analyze --repo https://github.com/OWNER/REPO --screenshot path/to/screenshot.png --text "One sentence about the bug"
   ```

   Check the new folder under `out/`: `issue.md`, `brief.md`, `result.json`.
   Add `--no-image` to see the text-only baseline.

8. **Launch the web app.**

   ```bash
   streamlit run app.py
   ```

   Open http://localhost:8501. Check that the sidebar shows the variables as
   set, run one analysis and look at all four tabs and both expanders. In the
   Suspected files tab, open the evidence of each file and follow the line
   link. Try a screenshot that shows a stack trace: the file names on it should
   appear under "Named in the screenshot".

   To make the demo faster, set `BUGLENS_THINKING_LEVEL=minimal` in `.env` and
   restart. In one comparison run with `gemma-4-31b-it` the three model calls
   took about 33 s instead of 106 s and ranked the same file first. That is a
   single run, so compare the answers on your own demo case before relying on it.

9. **Pre-index the demo repositories** so the demo does not wait for embedding.

   ```bash
   python -m buglens index --repo https://github.com/OWNER/REPO
   ```

10. **Build an eval set.** For each closed issue that has a screenshot and was
    fixed by a merged pull request:

    ```bash
    python -m buglens.eval.collect https://github.com/OWNER/REPO/issues/123
    ```

    It prints a YAML stanza and saves images to `eval/screenshots/`. Set
    `GITHUB_TOKEN` first: with a token the helper reads GitHub's own link
    between an issue and the pull request that closed it, which is far more
    reliable than its fallbacks. The helper is still a heuristic: **review every stanza by hand** (is it the right pull
    request, is `ref` really before the fix, are the files the real fix, does
    the screenshot show the bug, does `text` avoid giving the answer away).
    Create `eval/cases.yaml` in the format documented in
    `eval/cases.example.yaml` (a top-level `cases:` list, and no `example: true`
    line) and paste the reviewed stanzas under `cases:`.

11. **Run the eval.**

    ```bash
    python -m buglens.eval run eval/cases.yaml
    ```

    This runs every case twice (with the screenshot, and text only), prints
    hit@1, hit@3, hit@5 and MRR, and writes a JSON file and a markdown summary
    to `eval/results/`. Paste the markdown table into the README's Evaluation
    section. Use `--no-image` to run only the text-only variant, `--limit N` to
    try a few cases first, and `--sleep SECONDS` if you hit rate limits.
    Cases from the same repository share embeddings, so only the first one
    pays the full indexing time.

12. **Before submitting:** replace every `[TODO: ...]` in `README.md`, tick the
    checklist items that are now true, and make sure `.env`, `.cache/` and
    `out/` are not committed (they are in `.gitignore`).

## Troubleshooting

| Symptom | Likely cause | What to do |
| ------- | ------------ | ---------- |
| `The LLM API kept answering 429` or "The model API is rate limited" | Gemini API rate limit or quota for your key. BugLens already retried with backoff. | Wait a minute and retry. For the eval use `--sleep 10` or more, or `--limit`. Check the quota of your key in Google AI Studio. |
| `GitHub API rate limit reached` | No `GITHUB_TOKEN`: only 60 calls per hour per IP address, shared with everyone on the same network. | Put a token in `GITHUB_TOKEN` in `.env`, or wait for the reset time shown in the message. |
| `GEMMA_MODEL is not set` | No model id configured. BugLens does not guess one. | Run `python scripts/check_gemma.py` and copy an id into `.env`. |
| `The Gemini API does not know the model '...' (404)` | Wrong model id (typo, or an id from another provider such as an Ollama tag). | Run `python scripts/check_gemma.py` and copy the id exactly. |
| `The Gemini API rejected the API key` | Wrong or expired `GEMINI_API_KEY`. | Create a new key and update `.env`. |
| `GitHub returned 404 ... or it is private` | Typo in the URL, or a private repository. | BugLens only supports public repositories. Check the URL in a private browser window. |
| `... is larger than the 60 MB download limit` | Huge repository. | Use a smaller repository. The limit is `MAX_ZIP_BYTES` in `buglens/config.py`. |
| Warning `Large repository: only N of M files were indexed` | The 8000-chunk cap was reached. Source code is indexed first, so docs and tests are dropped first. | Usually fine. Raise `BUGLENS_MAX_CHUNKS` in `.env` if important code was cut (indexing takes longer). |
| The three Gemma steps take a minute or more | The model "thinks" before answering. | Set `BUGLENS_THINKING_LEVEL=minimal` in `.env`, or try the smaller Gemma model. Lines such as `model API answered 429; retrying in 6s` in the progress output mean the delay is rate limiting instead. |
| `rejected the BUGLENS_THINKING_LEVEL setting` | The model does not accept that thinking level. | Remove the variable, or use `minimal`. |
| The Similar issues tab is empty | No issue scored above the similarity threshold, the repository has no issues, or the GitHub calls failed (then there is a warning). | Nothing to do. It does not prove the bug is new. The threshold is `SIMILAR_ISSUE_MIN_SCORE` in `buglens/config.py`. |
| "Open this draft on GitHub" shows a template chooser | The repository does not allow blank issues. | Pick the template there and paste the body from the copy-ready box. |
| Indexing takes many minutes | Embedding runs on the CPU. On the laptop used to scaffold this project it embedded roughly 5 to 6 chunks per second; your machine will differ. | Run `python -m buglens index --repo URL` before the demo. Lower `BUGLENS_MAX_CHUNKS` for a faster, shallower index. |
| `No searchable source files were found` | The repository has no text files BugLens indexes (for example only binaries or notebooks). | Try another repository. |
| Suspected files look unrelated | The screenshot has little readable text, or the sentence is vague. | Open "What Gemma saw in your screenshot" in the app. Use a sharper screenshot that shows the error text, and a more specific sentence. |
| Warning `Dropped N file(s) named by the model that are not among the candidates` | The model named a file that retrieval did not find. The guard removed it. | Nothing to do. This is the hallucination guard working. |
| `The Gemini API returned no text (finish reason: RECITATION)` | The API withheld the reply because it would have repeated long text word for word, typically a screenshot full of prose. BugLens already retried once. | Retry. Crop the screenshot to the part that shows the bug (labels and error text, not paragraphs). `--no-image` skips the screenshot. Other reasons: `SAFETY` is a content filter, `MAX_TOKENS` means the output budget (`LLM_MAX_OUTPUT_TOKENS` in `buglens/config.py`) ran out. |
| Warning `The reranker gave no usable reply` or error "did not return usable JSON" | The model did not answer with valid JSON, even after one repair retry. | Retry. Try a larger Gemma model. As an experiment, set `BUGLENS_NATIVE_JSON=1` (may be rejected by Gemma). |
| `Unsupported image format` | The screenshot is not PNG, JPG or WebP. | Convert it. |
| Streamlit shows variables as missing although `.env` is filled in | Streamlit was started from another folder, or `.env` was edited after start. | Start `streamlit run app.py` from the project folder and restart after editing `.env`. |
| `Could not reach Ollama` | `LLM_BACKEND=ollama` but Ollama is not running. | Start Ollama, check `OLLAMA_HOST`, and set `GEMMA_MODEL` to a tag from `ollama list`. This backend is experimental. |
