import logging
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI

import backend.state as state
import config as _
from agents.vir_chatbot.vir_chatbot import load_global_vectorstore
from backend.models import HealthCheckFilter
from backend.routers import chat, config, tasks, threads, vectorstore

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


async def _turn_wal_mode_on():
    logging.info("Initializing SQLite with WAL mode...")
    try:
        db_dir = Path(_.SQLITE_MEMORY_DATABASE).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA busy_timeout=30000;")
        logging.info(
            f"SQLite WAL mode enabled for {_.SQLITE_MEMORY_DATABASE}"
        )
    except Exception as e:
        logging.warning(f"Could not enable WAL mode: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.runtime_config = state._load_persisted_config()
    logging.info("Runtime config loaded.")

    await _turn_wal_mode_on()

    logging.info("Loading VectorStore into memory...")
    try:
        state.global_resources["retriever"] = await load_global_vectorstore(
            retriever_limit=state.runtime_config.retriever_limit
        )
        if state.global_resources["retriever"]:
            logging.info("VectorStore loaded successfully!")
        else:
            logging.info("VectorStore not found. Create one first.")
    except Exception as e:
        logging.error(
            f"Error loading VectorStore: {e}. Try creating it first."
        )
        state.global_resources["retriever"] = None

    yield

    state.global_resources.clear()
    logging.info("Resources cleared.")


app = FastAPI(lifespan=lifespan)

app.include_router(config.router)
app.include_router(vectorstore.router)
app.include_router(tasks.router)
app.include_router(threads.router)
app.include_router(chat.router)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}
