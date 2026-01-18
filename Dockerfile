FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    tesseract-ocr \
    curl

ENV UV_COMPILE_BYTECODE=1

ENV UV_LINK_MODE=copy

ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY config.py langgraph_functions.py gemini.py prompts.py schemas.py tokenizer.py /app/
COPY backend/ /app/backend/
COPY agents/vir_chatbot/ /app/agents/vir_chatbot/

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
