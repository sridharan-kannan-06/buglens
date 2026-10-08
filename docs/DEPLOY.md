# Deploying BugLens

> **Status: not tested.** These steps were written from the platforms'
> documented behaviour. Nobody has deployed BugLens with them yet, and the
> Docker image has not been built. Expect to adjust details, and update this
> file once a deployment works.

BugLens is a single Streamlit process. It needs outbound HTTPS (GitHub API,
Gemini API) and a writable disk for its cache (`.cache/`).

## Environment variables

| Variable | Required | Notes |
| -------- | -------- | ----- |
| `GEMINI_API_KEY` | yes | Secret. Set it in the platform's secret store, never in the repository. |
| `GEMMA_MODEL` | yes | Exact model id. Find it with `python scripts/check_gemma.py`. |
| `GITHUB_TOKEN` | strongly recommended | Secret. A hosted app shares one IP address between all visitors, so the unauthenticated limit of 60 calls per hour is used up quickly. |
| `LLM_BACKEND` | no | Leave unset (`gemini`). The `ollama` backend is experimental and needs an Ollama server. |
| `BUGLENS_MAX_CHUNKS` | no | Lower it (for example `2000`) on small instances to shorten indexing. |
| `BUGLENS_THINKING_LEVEL` | no | `minimal` makes the model answer faster. Leave unset if the model rejects it. |
| `BUGLENS_CACHE_DIR` | no | The Dockerfile sets it to `/app/.cache`. |

Things to know before choosing an instance size:

- Indexing a repository embeds every chunk on the CPU. On a small shared CPU
  this can take minutes for a mid-size repository.
- The cache lives on the container's disk. On most platforms it is lost on
  every redeploy or restart, so repositories are indexed again afterwards.
- The embedding model and its runtime need memory. Start with at least 1 GB RAM
  and increase it if the process is killed while indexing.

## Option A: DigitalOcean App Platform (uses the Dockerfile)

1. Push the repository to GitHub (the team does this manually).
2. In the DigitalOcean control panel choose **Create > Apps** and connect the GitHub repository and branch.
3. App Platform should detect the `Dockerfile` in the repository root. Keep the resource type **Web Service**.
4. Set the **HTTP port** to `8501`. (The container also honours a `PORT` variable if the platform provides one.)
5. Under **Environment Variables** add `GEMINI_API_KEY` and `GITHUB_TOKEN` and mark them as **encrypted**. Add `GEMMA_MODEL` as a plain variable.
6. Pick an instance size (see the notes above) and create the app.
7. When the build finishes, open the app URL. If a health check is configured, use the path `/_stcore/health`.
8. Run one analysis on a small repository and check the "Stage timings and warnings" expander.
9. Put the URL into the README under **Working Application**.

## Option B: Streamlit Community Cloud (no Docker)

1. Push the repository to GitHub (the team does this manually).
2. On https://share.streamlit.io choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
3. In **Advanced settings** choose a Python version that is 3.10 or newer.
4. In **Advanced settings > Secrets** paste the variables in TOML form. Top-level secrets are made available to the app as environment variables, which is how BugLens reads them:

   ```toml
   GEMINI_API_KEY = "your key"
   GEMMA_MODEL = "exact model id"
   GITHUB_TOKEN = "your token"
   ```

5. Deploy. Dependencies are installed from `requirements.txt`. The embedding model is downloaded on the first analysis, because the Dockerfile's pre-download step does not apply here.
6. Run one analysis on a small repository. Community Cloud instances are small, so set `BUGLENS_MAX_CHUNKS` to a lower value in the secrets if indexing is too slow or the app runs out of memory.
7. Put the URL into the README under **Working Application**.

## Local Docker run (to check the image before deploying)

```bash
docker build -t buglens .
docker run --rm -p 8501:8501 --env-file .env buglens
```

Then open http://localhost:8501. The health endpoint is http://localhost:8501/_stcore/health.

Note: `--env-file` does not strip quotes or inline comments, so keep the values in `.env` plain (`NAME=value`).
