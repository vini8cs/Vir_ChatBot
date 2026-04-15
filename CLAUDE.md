# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Vir_ChatBot is a RAG chatbot for virology/bioinformatics. It ingests PDFs (plus docx, images, txt, tsv) via Docling, indexes chunks in a FAISS vectorstore, and answers questions using a LangGraph agent backed by **Google Gemini** (LLM + embeddings).

## Commands

Python is managed with **uv** (pinned to 3.12, linux/x86_64 only — see `pyproject.toml`). Torch is installed from the `pytorch-cpu` index.

```bash
uv sync                                  # install dependencies
uv sync --group dev                      # install + test dependencies
uv run pre-commit run --all-files        # ruff check + format, line-length 78
uv run ruff check --fix --select E,W,F,I,B,C4,SIM --line-length 78 .
uv run ruff format --line-length 78 .
```

Local dev (three processes, each in its own terminal — the backend won't function without Redis + Celery):

```bash
docker run -d -p 6379:6379 --name redis-vir redis:7
uv run celery -A agents.vir_chatbot.tasks worker -l info
uv run uvicorn backend.api:app --host 0.0.0.0 --port 8000
cd frontend && uv run streamlit run app.py
```

Containerized:

```bash
docker compose up --build                   # full stack
docker compose --profile dev up --build     # + redis-commander on :8081
```

## Testing

Tests live in `tests/` mirroring the source tree (e.g. `tests/routers/test_config.py`).

```bash
uv run pytest                        # run all tests
uv run pytest -v                     # verbose output
uv run pytest --cov=. --cov-report=term-missing   # with coverage
uv run pytest tests/test_models.py   # single file
```

Rules:
- Every new feature and every bug fix **must** include a corresponding test.
- After any code change, run the full test suite and fix all failures before considering the task done.
- Prefer testing real behaviour over mocks. Only mock at system boundaries (Gemini API, FAISS I/O) — never mock the module under test itself.
- `tests/conftest.py` stubs `google.genai`, `langchain_google_genai`, and `langchain_google_vertexai` in `sys.modules`, and sets fake env vars so `config.py` can be imported without real credentials. Tests that need a writable config path use the `tmp_config_path` fixture.
- Priority areas: `backend/models.py`, `backend/state.py`, `agents/vir_chatbot/vectorstore.py` (static methods), `llms/gemini.py` (static methods), `backend/routers/`.

## Architecture

### Process layout

Four runtime services — they do not share memory; all coordination is through Redis (Celery broker) and the SQLite file at `SQLITE_MEMORY_DATABASE`:

- **web** (FastAPI, `backend/api.py`) — chat streaming, config, vectorstore CRUD, thread CRUD. On startup (`lifespan`), it loads the FAISS retriever into `backend/state.py::global_resources`.
- **worker** (Celery, `agents/vir_chatbot/tasks.py`) — runs heavy PDF ingestion jobs (Docling OCR, chunking, embedding, FAISS write).
- **redis** — Celery broker/result backend.
- **streamlit** (`frontend/`) — UI; talks to the backend via `frontend/api_client.py` over HTTP (no direct Python imports across the boundary).

### Config is three-layered — don't confuse them

1. **`config.py`** — static defaults loaded from `.env` via `pydantic-settings`. Requires `GEMINI_API_KEY`, `GCP_CREDENTIALS`, `GCP_PROJECT`, `GCP_REGION`. Also initializes `CLIENT_GEMINI` (via `genai.Client()`) and exports constants like `GEMINI_MODEL`, `EMBEDDING_MODEL`, `TEMPERATURE`, etc.
2. **`backend/models.py::RuntimeConfig`** — the subset of config that can be mutated at runtime through `PUT /config`. Persisted to `runtime_config.json` next to the SQLite DB (`state._save_persisted_config`), loaded in `lifespan`.
3. **Per-request overrides** — `create_graph()` in `agents/vir_chatbot/vir_chatbot.py` accepts overrides (`llm_model`, `temperature`, `system_prompt`, …) that win over `RuntimeConfig`. The chat router passes these from `state.runtime_config` on each request.

When adding a new tunable, it almost always needs to be threaded through all three.

### LLM and embeddings: Gemini-only

Both the LLM and the embeddings are Google Gemini:

- **LLM**: `ChatGoogleGenerativeAI` (via `langchain_google_genai`) — instantiated directly in `Vir_ChatBot.__init__` inside `agents/vir_chatbot/vir_chatbot.py`.
- **Embeddings**: `GoogleGenerativeAIEmbeddings` (via `langchain_google_genai`) — instantiated in `load_global_vectorstore` (query path) and in `Gemini.__init__` (ingestion path via `VectorStoreCreator`).

### Critical FAISS/embeddings invariant

A FAISS index must be queried with the exact same embedding model it was built with. If you change `EMBEDDING_MODEL` after a vectorstore exists, retrieval silently returns garbage — the store must be rebuilt. The ingestion path (`agents/vir_chatbot/vectorstore.py::VectorStoreCreator`) and the query path (`agents/vir_chatbot/vir_chatbot.py::load_global_vectorstore`) must both use `_.EMBEDDING_MODEL`.

### LangGraph agent

`Vir_ChatBot.build_graph()` builds a minimal two-node graph: `query_or_respond` (the LLM with a bound `retrieve` tool) and `tools` (the `ToolNode`), wired by `tools_condition`. Conversation state is persisted by `AsyncSqliteSaver` on `SQLITE_MEMORY_DATABASE` with `PRAGMA journal_mode=WAL` + `busy_timeout=30000` — this matters because Celery, the web server, and checkpointer may all touch the same DB.

### VectorStoreCreator inherits from Gemini

`VectorStoreCreator(Gemini)` in `agents/vir_chatbot/vectorstore.py` subclasses the Gemini summarization wrapper because the **optional** text/image summarization step uses Gemini. The `__init__` calls `super().__init__()` which sets `self.embeddings = GoogleGenerativeAIEmbeddings(...)`. Both the ingestion embeddings and the summarization chain use the same Gemini credentials.

### Docling ingestion

`_start_chunking_process` walks `pdf_paths`, dispatches `.txt` / `.tsv` to simple chunkers and everything else to Docling (`HybridChunker` over `DocumentConverter`). Results are cached in `CACHE_FOLDER/cache.csv` keyed by filename; `_diff_vs_cache` prevents re-ingesting PDFs already in the store, and `delete_pdfs()` uses the cached UUIDs to remove vectors from FAISS.

## Conventions

- **Line length 78**, enforced by the ruff pre-commit hook with rule set `E,W,F,I,B,C4,SIM`. Match it.
- `import config as _` is the project-wide idiom for accessing config constants (e.g. `_.VECTORSTORE_PATH`). Follow it — this is a codebase convention, not a typo.
- Paths in containers are hard-coded to `/app/vectorstore`, `/app/cache`, `/app/db_data` and bind-mounted from the host via `.env` values. The `entrypoint.sh` remaps the `appuser` UID/GID to `PUID`/`PGID` at container start so bind-mount permissions line up.
