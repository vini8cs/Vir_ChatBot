FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
       --no-install-recommends \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
       poppler-utils=22.12.0-2+deb12u1 \
       tesseract-ocr=5.3.0-2 \
       curl=7.88.1-10+deb12u14 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/vectorstore /app/pdfs /app/cache /app/db_data /tmp/temp_uploads \
    && chown -R appuser:appuser /app /tmp/temp_uploads

USER appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY config.py /app/
COPY llms/ /app/llms/
COPY templates/ /app/templates/
COPY backend/ /app/backend/
COPY agents/vir_chatbot/ /app/agents/vir_chatbot/

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
