import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import aiofiles
import aiosqlite
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel, Field

import config as _
from agents.vir_chatbot.tasks import app as celery_app
from agents.vir_chatbot.tasks import (
    create_vectorstore_from_folder,
    create_vectorstore_uploaded_pdfs,
    delete_pdfs_from_vectorstore,
)
from agents.vir_chatbot.vir_chatbot import create_graph, load_global_vectorstore


class RuntimeConfig(BaseModel):
    """Runtime configuration that can be modified via API."""

    gemini_model: str = _.GEMINI_MODEL
    embedding_model: str = _.EMBEDDING_MODEL
    temperature: float = _.TEMPERATURE
    max_output_tokens: int = _.MAX_OUTPUT_TOKENS
    token_size: int = _.TOKEN_SIZE
    max_retries: int = _.MAX_RETRIES
    tokenizer_model: str = _.TOKENIZER_MODEL
    threads: int = _.THREADS
    summarize: bool = _.SUMMARIZE
    retriever_limit: int = _.RETRIEVER_LIMIT


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration."""

    gemini_model: str | None = None
    embedding_model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    token_size: int | None = None
    max_retries: int | None = None
    tokenizer_model: str | None = None
    threads: int | None = None
    summarize: bool | None = None
    retriever_limit: int | None = None


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


class DeleteFileRequest(BaseModel):
    filenames: List[str]


class TaskIDRequest(BaseModel):
    task_id: str


class ChatRequest(BaseModel):
    message: str
    thread_id: str = Field(..., description="Unique ID for the conversation thread")
    user_id: str = Field("default_user", description="User ID for personalization")


class CreateThreadRequest(BaseModel):
    user_id: str = Field(..., description="User ID to associate with the thread")


class ThreadResponse(BaseModel):
    thread_id: str
    user_id: str


global_resources = {}
runtime_config: RuntimeConfig = RuntimeConfig()
TEMP_UPLOAD_DIR = "/tmp/temp_uploads"
Path(TEMP_UPLOAD_DIR).mkdir(exist_ok=True)


async def turn_wal_mode_on():
    logging.info("Initializing SQLite with WAL mode...")
    try:
        db_dir = os.path.dirname(_.SQLITE_MEMORY_DATABASE)
        Path(db_dir).mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA busy_timeout=30000;")
        logging.info(f"SQLite WAL mode enabled for {_.SQLITE_MEMORY_DATABASE}")
    except Exception as e:
        logging.warning(f"Could not enable WAL mode: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await turn_wal_mode_on()

    logging.info("Loading VectorStore into memory...")
    try:
        global_resources["retriever"] = await load_global_vectorstore()
        if global_resources["retriever"]:
            logging.info("VectorStore loaded successfully!")
        else:
            logging.info("VectorStore not found. Create one first.")
    except Exception as e:
        logging.error(f"Error loading VectorStore: {e}. Try creating it first.")
        global_resources["retriever"] = None

    yield

    global_resources.clear()
    logging.info("Resources cleared.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.get("/config")
async def get_config():
    """Get current runtime configuration settings."""
    return runtime_config.model_dump()


@app.put("/config")
async def update_config(request: ConfigUpdateRequest):
    """
    Update runtime configuration. Only provided fields will be updated.
    Changes take effect immediately for new chat sessions.
    """
    global runtime_config

    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No configuration fields provided")

    current_config = runtime_config.model_dump()
    current_config.update(update_data)
    runtime_config = RuntimeConfig(**current_config)

    return {
        "status": "success",
        "message": "Configuration updated successfully",
        "config": runtime_config.model_dump(),
    }


@app.post("/config/reset")
async def reset_config():
    """
    Reset configuration to default values from config.py.
    Optionally reload the vectorstore if retriever_limit changed.
    """
    global runtime_config
    runtime_config = RuntimeConfig()

    return {
        "status": "success",
        "message": "Configuration reset to defaults",
        "config": runtime_config.model_dump(),
    }


@app.post("/config/reset-and-reload")
async def reset_config_and_reload():
    """
    Reset configuration to defaults AND reload the vectorstore.
    Use this when you want a complete fresh start.
    """
    global runtime_config
    runtime_config = RuntimeConfig()

    try:
        global_resources["retriever"] = await load_global_vectorstore()
        return {
            "status": "success",
            "message": "Configuration reset and VectorStore reloaded",
            "config": runtime_config.model_dump(),
            "vectorstore_loaded": global_resources["retriever"] is not None,
        }
    except Exception as e:
        logging.error(f"Error reloading VectorStore: {e}")
        return {
            "status": "partial",
            "message": "Configuration reset but VectorStore reload failed",
            "config": runtime_config.model_dump(),
            "error": str(e),
        }


@app.post("/vectorstore/reload")
async def reload_vectorstore():
    """Endpoint to reload the VectorStore into memory."""
    try:
        global_resources["retriever"] = await load_global_vectorstore()
        if global_resources["retriever"]:
            return {
                "status": "success",
                "message": "VectorStore reloaded successfully.",
            }
        return {
            "status": "warning",
            "message": "VectorStore not found. Create one first.",
        }
    except Exception as e:
        logging.info(f"Error reloading VectorStore: {e}")
        return {"status": "error", "message": "Error reloading VectorStore"}


@app.post("/create-vectorstore-based-on-selected-pdfs/")
async def upload_pdf(
    files: list[UploadFile] = File(..., media_type="application/pdf"),  # noqa: B008
):
    """Endpoint to create or update the vectorstore based on selected PDF(s)"""
    request_id = str(uuid.uuid4())
    upload_dir = os.path.join(TEMP_UPLOAD_DIR, request_id)
    Path(upload_dir).mkdir(exist_ok=True)

    pdf_files_to_upload = []
    try:
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            pdf_files_to_upload.append(file_path)

    except Exception as e:
        await asyncio.to_thread(shutil.rmtree, upload_dir)
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

    task = create_vectorstore_uploaded_pdfs.delay(
        pdf_files_to_upload,
        summarize=runtime_config.summarize,
        gemini_model=runtime_config.gemini_model,
    )

    return {
        "message": "VectorStore creation started in background.",
        "task_id": task.id,
        "upload_directory": str(upload_dir),
    }


@app.post("/delete-selected-pdfs/")
async def delete_pdfs(request: DeleteFileRequest):
    """
    Endpoint to trigger the deletion of specific PDFs from the VectorStore.
    """
    if not request.filenames:
        raise HTTPException(status_code=400, detail="The list of files is empty.")

    logging.info(f"Requesting deletion for: {request.filenames}")

    task = delete_pdfs_from_vectorstore.delay(request.filenames)

    return {
        "message": "Deletion process started in background.",
        "task_id": task.id,
        "files_to_delete": request.filenames,
    }


@app.post("/create-vectorstore-from-folder/")
async def create_vectorstore_from_folder_endpoint():
    """
    Create vectorstore from zero using PDF_FOLDER path in .env
    """
    task = create_vectorstore_from_folder.delay(
        summarize=runtime_config.summarize,
        gemini_model=runtime_config.gemini_model,
    )

    return {
        "message": "VectorStore creation from folder started in background.",
        "task_id": task.id,
    }


def get_filenames_from_metadata(metadata_str: str) -> List[str]:
    try:
        data = json.loads(metadata_str)
        return data.get("filename")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


@app.get("/pdfs/list")
async def list_pdfs():
    """
    List all PDFs in the vectorstore cache.
    """
    cache_path = os.path.join(_.CACHE_FOLDER, "cache.csv")

    if not os.path.exists(cache_path):
        return {"pdfs": []}

    try:
        df = pd.read_csv(cache_path)
        if "metadata" not in df.columns:
            return {"pdfs": []}

        filenames = df["metadata"].apply(get_filenames_from_metadata).dropna().unique()
        return {"pdfs": sorted(filenames)}

    except Exception as e:
        logging.error(f"Error reading cache: {e}")
        return {"pdfs": []}


@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """
    Get the status and progress of a Celery task.
    """
    try:

        def _handle_progress():
            info = task_result.info or {}
            return {
                "current": info.get("current", 0),
                "total": info.get("total", 0),
                "percent": info.get("percent", 0),
                "step": info.get("step", ""),
                "details": info.get("details", ""),
            }

        def _handle_success():
            result = task_result.result or {}
            return {"result": result, "percent": 100}

        def _handle_failure():
            return {
                "error": (str(task_result.result) if task_result.result else "Unknown error"),
                "percent": 0,
            }

        def _handle_pending():
            return {"percent": 0, "step": "Waiting", "details": "Task queued..."}

        task_result = celery_app.AsyncResult(task_id)

        response = {
            "task_id": task_id,
            "status": task_result.status,
            "ready": task_result.ready(),
            "successful": task_result.successful() if task_result.ready() else None,
        }

        _handlers = {
            "PROGRESS": _handle_progress,
            "SUCCESS": _handle_success,
            "FAILURE": _handle_failure,
            "PENDING": _handle_pending,
        }

        handler = _handlers.get(task_result.status)
        if handler:
            response.update(handler())

        return response

    except Exception as e:
        logging.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@app.post("/threads/create", response_model=ThreadResponse)
async def create_thread(request: CreateThreadRequest):
    """
    Create a new thread for a user.
    Returns a unique thread_id that can be used in chat requests.
    """
    thread_id = str(uuid.uuid4())

    return ThreadResponse(thread_id=thread_id, user_id=request.user_id)


@app.get("/threads/{user_id}")
async def get_user_threads(user_id: str):
    """
    Get all threads associated with a specific user.
    """
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        return {"threads": []}

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            query = """
            SELECT DISTINCT thread_id 
            FROM checkpoints 
            WHERE json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            """  # noqa: W291
            async with conn.execute(query, (user_id,)) as cur:
                rows = await cur.fetchall()

            return {"threads": [{"thread_id": r[0]} for r in rows]}

    except Exception as e:
        if "no such table" in str(e).lower():
            return {"threads": []}
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@app.delete("/threads/{user_id}/{thread_id}")
async def delete_thread(user_id: str, thread_id: str):
    """Delete a specific thread and its associated messages."""
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            query = """
            SELECT 1 FROM checkpoints 
            WHERE thread_id = ? 
            AND json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            LIMIT 1
            """  # noqa W291
            async with conn.execute(query, (thread_id, user_id)) as cur:
                row = await cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail="Thread not found or access denied",
                )

            await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))

            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))

            await conn.commit()
            return {"message": f"Thread {thread_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        if "no such table" in str(e).lower():
            raise HTTPException(status_code=404, detail="Thread not found or access denied") from e
        logging.error(f"CRITICAL DB ERROR: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """
    Get the message history for a specific thread.
    Extracts messages from the LangGraph checkpointer.
    """
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        return {"messages": []}

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            query = """
            SELECT checkpoint, type
            FROM checkpoints 
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """  # noqa W291
            async with conn.execute(query, (thread_id,)) as cur:
                row = await cur.fetchone()

            if not row:
                return {"messages": []}

            checkpoint_data, checkpoint_type = row[0], row[1]

            if not isinstance(checkpoint_data, bytes):
                return {"messages": []}

            serde = JsonPlusSerializer()
            checkpoint = serde.loads_typed((checkpoint_type, checkpoint_data))

            messages = []
            channel_values = checkpoint.get("channel_values", {})
            msg_list = channel_values.get("messages", [])

            for msg in msg_list:
                if not (hasattr(msg, "type") and hasattr(msg, "content")):
                    continue
                msg_type = msg.type
                msg_content = msg.content
                if isinstance(msg_content, list):
                    continue
                if msg_type in ["human", "ai"] and msg_content:
                    role = "user" if msg_type == "human" else "assistant"
                    messages.append({"role": role, "content": msg_content})

            return {"messages": messages}

    except Exception as e:
        if "no such table" in str(e).lower():
            return {"messages": []}
        logging.error(f"Error loading messages for thread {thread_id}: {e}")
        return {"messages": []}


async def create_graph_retriever(retriever, config: RuntimeConfig, max_retries=3):
    """Create graph using runtime configuration."""
    for attempt in range(max_retries):
        try:
            return await create_graph(
                global_retriever=retriever,
                llm_model=config.gemini_model,
                temperature=config.temperature,
                max_retries=config.max_retries,
            )
        except Exception as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                await asyncio.sleep(3)
                continue
            return None
    return None


async def chat_generator(user_input: str, thread_id: str, user_id: str):
    """
    Generates the streaming response and ensures the database connection is closed.
    Uses runtime_config for LLM settings.
    """
    retriever = global_resources.get("retriever")
    if not retriever:
        yield f"data: {json.dumps({'error': 'VectorStore not initialized. Please create or reload the VectorStore.'})}\n\n"  # noqa E501
        return

    result = await create_graph_retriever(retriever, runtime_config, 3)

    if result is None:
        yield f"data: {json.dumps({'error': 'Error creating graph'})}\n\n"
        return

    graph, checkpointer_cm = result

    graph_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    try:
        async for event in graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            graph_config,
            stream_mode="values",
        ):
            messages = event.get("messages", [])
            if not messages:
                continue
            last_msg = messages[-1]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                continue

            if last_msg.type != "ai":
                continue

            if last_msg.content:
                payload = {"content": last_msg.content, "type": "ai_response"}
                yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error during chat generation: {error_msg}")
        if "database is locked" in error_msg.lower():
            error_msg = "The system is busy updating. Please try again in a few seconds."
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

    finally:
        if checkpointer_cm:
            try:
                await checkpointer_cm.__aexit__(None, None, None)
            except Exception as close_err:
                logging.error(f"Error closing SQLite connection: {close_err}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Call the chat generator and return a streaming response."""
    return StreamingResponse(
        chat_generator(request.message, request.thread_id, request.user_id),
        media_type="text/event-stream",
    )
