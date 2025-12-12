import json
import logging
import os
import shutil
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import aiofiles
import aiosqlite
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import config as _
from agents.vir_chatbot.vir_chatbot import create_graph, load_global_vectorstore
from tasks import app as celery_app
from tasks import (
    create_vectorstore_from_folder,
    create_vectorstore_uploaded_pdfs,
    delete_pdfs_from_vectorstore,
)


# Filtro para não logar requisições do healthcheck
class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


class DeleteFileRequest(BaseModel):
    filenames: List[str]


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

    logging.info("Carregando VectorStore na memória...")
    try:
        global_resources["retriever"] = await load_global_vectorstore()
        if global_resources["retriever"]:
            logging.info("VectorStore load sucessfully!")
        else:
            logging.info("VectorStore não encontrado. Crie um primeiro.")
    except Exception as e:
        logging.error(f"Erro ao carregar VectorStore: {e}. Tente criar primeiramente.")
        global_resources["retriever"] = None

    yield

    global_resources.clear()
    logging.info("Recursos limpos.")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.post("/vectorstore/reload")
async def reload_vectorstore():
    """Endpoint to reload the VectorStore into memory."""
    try:
        global_resources["retriever"] = await load_global_vectorstore()
        if global_resources["retriever"]:
            return {
                "status": "success",
                "message": "VectorStore recarregado com sucesso.",
            }
        return {
            "status": "warning",
            "message": "VectorStore não encontrado. Crie um primeiro.",
        }
    except Exception as e:
        logging.info(f"Erro ao recarregar VectorStore: {e}")
        return {"status": "error", "message": f"Erro ao recarregar: {e}"}


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
        shutil.rmtree(upload_dir)
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivos: {e}")

    task = create_vectorstore_uploaded_pdfs.delay(pdf_files_to_upload)

    return {
        "message": "A criação do VectorStore foi iniciada em segundo plano.",
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

    print(f"Requesting deletion for: {request.filenames}")

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
    task = create_vectorstore_from_folder.delay()

    return {
        "message": "A criação do VectorStore do zero foi iniciada em segundo plano.",
        "task_id": task.id,
    }


@app.get("/pdfs/list")
async def list_pdfs():
    """
    List all PDFs in the vectorstore cache.
    """
    import pandas as pd

    # CACHE_FOLDER is a directory, the actual file is cache.csv inside it
    cache_path = os.path.join(_.CACHE_FOLDER, "cache.csv")

    if not os.path.exists(cache_path):
        return {"pdfs": []}

    try:
        df = pd.read_csv(cache_path)
        if "metadata" in df.columns:
            import json

            filenames = set()
            for metadata_str in df["metadata"]:
                try:
                    metadata = json.loads(metadata_str)
                    if "filename" in metadata:
                        filenames.add(metadata["filename"])
                except (json.JSONDecodeError, TypeError):
                    continue
            return {"pdfs": sorted(filenames)}
        return {"pdfs": []}
    except Exception as e:
        print(f"Error reading cache: {e}")
        return {"pdfs": []}


@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """
    Get the status and progress of a Celery task.
    """
    try:
        task_result = celery_app.AsyncResult(task_id)

        response = {
            "task_id": task_id,
            "status": task_result.status,
            "ready": task_result.ready(),
            "successful": task_result.successful() if task_result.ready() else None,
        }

        if task_result.status == "PROGRESS":
            # Task is in progress
            info = task_result.info or {}
            response.update(
                {
                    "current": info.get("current", 0),
                    "total": info.get("total", 0),
                    "percent": info.get("percent", 0),
                    "step": info.get("step", ""),
                    "details": info.get("details", ""),
                }
            )
        elif task_result.status == "SUCCESS":
            # Task completed successfully
            result = task_result.result or {}
            response.update(
                {
                    "result": result,
                    "percent": 100,
                }
            )
        elif task_result.status == "FAILURE":
            # Task failed
            response.update(
                {
                    "error": (str(task_result.result) if task_result.result else "Unknown error"),
                    "percent": 0,
                }
            )
        elif task_result.status == "PENDING":
            # Task is pending
            response.update(
                {
                    "percent": 0,
                    "step": "Aguardando",
                    "details": "Tarefa na fila...",
                }
            )

        return response

    except Exception as e:
        print(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting task status: {e}")


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
    sqlite_db_path = _.SQLITE_MEMORY_DATABASE

    if not os.path.exists(sqlite_db_path):
        return {"threads": []}

    try:
        async with aiosqlite.connect(sqlite_db_path) as conn:
            query = """
            SELECT DISTINCT thread_id 
            FROM checkpoints 
            WHERE json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            """  # noqa: W291
            async with conn.execute(query, (str(user_id),)) as cur:
                rows = await cur.fetchall()

            return {"threads": [{"thread_id": r[0]} for r in rows]}

    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.delete("/threads/{user_id}/{thread_id}")
async def delete_thread(user_id: str, thread_id: str):
    sqlite_db_path = _.SQLITE_MEMORY_DATABASE

    if not os.path.exists(sqlite_db_path):
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        async with aiosqlite.connect(sqlite_db_path) as conn:
            query = """
            SELECT COUNT(*) FROM checkpoints 
            WHERE thread_id = ? 
            AND json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            """  # noqa: W291
            async with conn.execute(query, (str(thread_id), str(user_id))) as cur:
                row = await cur.fetchone()
                count = row[0] if row else 0

            if count == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Thread {thread_id} not found for user {user_id}",
                )

            # Delete the thread
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
            await conn.commit()

            return {"message": f"Thread {thread_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """
    Get the message history for a specific thread.
    Extracts messages from the LangGraph checkpointer.
    """
    sqlite_db_path = _.SQLITE_MEMORY_DATABASE

    if not os.path.exists(sqlite_db_path):
        return {"messages": []}

    try:
        async with aiosqlite.connect(sqlite_db_path) as conn:
            # Get the latest checkpoint for this thread
            query = """
            SELECT checkpoint, type
            FROM checkpoints 
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """
            async with conn.execute(query, (str(thread_id),)) as cur:
                row = await cur.fetchone()

            if not row:
                return {"messages": []}

            checkpoint_data, checkpoint_type = row[0], row[1]

            if not isinstance(checkpoint_data, bytes):
                return {"messages": []}

            # Use LangGraph's serializer to properly deserialize
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            serde = JsonPlusSerializer()
            checkpoint = serde.loads_typed((checkpoint_type, checkpoint_data))

            # Extract messages from checkpoint
            messages = []
            channel_values = checkpoint.get("channel_values", {})
            msg_list = channel_values.get("messages", [])

            for msg in msg_list:
                # Messages should be proper LangChain message objects
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    msg_type = msg.type
                    msg_content = msg.content
                    # Skip tool messages and empty/list content
                    if isinstance(msg_content, list):
                        continue
                    if msg_type in ["human", "ai"] and msg_content:
                        role = "user" if msg_type == "human" else "assistant"
                        messages.append({"role": role, "content": msg_content})

            return {"messages": messages}

    except Exception as e:
        logging.error(f"Error loading messages for thread {thread_id}: {e}")
        return {"messages": []}


async def chat_generator(user_input: str, thread_id: str, user_id: str):
    """
    Gera a resposta em streaming e garante o fechamento da conexão do banco.
    """
    retriever = global_resources.get("retriever")
    if not retriever:
        yield f"data: {json.dumps({'error': 'VectorStore não inicializado. Por favor, crie ou recarregue o VectorStore.'})}\n\n"
        return

    graph_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    graph = None
    checkpointer_cm = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            graph, checkpointer_cm = await create_graph(global_retriever=retriever)
            break
        except Exception as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                import asyncio

                await asyncio.sleep(1)
                continue
            else:
                yield f"data: {json.dumps({'error': f'Erro ao criar grafo: {str(e)}'})}\n\n"
                return

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

            # Check for tool calls (AI decided to use a tool)
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                continue  # Skip tool call messages, wait for final response

            if last_msg.type != "ai":
                continue

            # Only yield if there's actual content
            if last_msg.content:
                payload = {"content": last_msg.content, "type": "ai_response"}
                yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_msg = str(e)
        if "database is locked" in error_msg.lower():
            error_msg = "O sistema está ocupado atualizando. Por favor, tente novamente em alguns segundos."
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

    finally:
        if checkpointer_cm:
            try:
                await checkpointer_cm.__aexit__(None, None, None)
            except Exception as close_err:
                logging.error(f"Erro ao fechar conexão SQLite: {close_err}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        chat_generator(request.message, request.thread_id, request.user_id),
        media_type="text/event-stream",
    )
