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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite with WAL mode for better concurrency
    print("Initializing SQLite with WAL mode...")
    try:
        db_dir = os.path.dirname(_.SQLITE_MEMORY_DATABASE)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(_.SQLITE_MEMORY_DATABASE)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")  # 30 second timeout
        conn.close()
        logging.info(f"SQLite WAL mode enabled for {_.SQLITE_MEMORY_DATABASE}")
    except Exception as e:
        logging.warning(f"Could not enable WAL mode: {e}")

    print("Carregando VectorStore na memória...")
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
    Cria o vectorstore do zero a partir de uma pasta de PDFs.

    - **pdf_folder**: Caminho absoluto para a pasta contendo os PDFs.
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


async def chat_generator(user_input: str, thread_id: str, user_id: str):
    """
    Gera a resposta em streaming e garante o fechamento da conexão do banco.
    """
    print(f"[DEBUG] chat_generator called with thread_id={thread_id}, user_id={user_id}")

    retriever = global_resources.get("retriever")
    if not retriever:
        yield "data: "
        f"{json.dumps({'error': 'VectorStore não inicializado. Por favor, crie ou recarregue o VectorStore.'})}\n\n"
        return

    graph_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    print(f"[DEBUG] graph_config: {graph_config}")

    graph = None
    checkpointer_cm = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            graph, checkpointer_cm = await create_graph(global_retriever=retriever)
            print("[DEBUG] Graph created successfully")
            break
        except Exception as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                print(f"[DEBUG] Database locked, retrying... (attempt {attempt + 1})")
                import asyncio

                await asyncio.sleep(1)  # Wait before retry
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
            if last_msg.type != "ai":
                continue
            payload = {"content": last_msg.content, "type": "ai_response"}
            yield f"data: {json.dumps(payload)}\n\n"

        print(f"[DEBUG] Stream completed for thread_id={thread_id}")
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"[DEBUG] Error in chat_generator: {e}")
        error_msg = str(e)
        if "database is locked" in error_msg.lower():
            error_msg = "O sistema está ocupado atualizando. Por favor, tente novamente em alguns segundos."
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

    finally:
        if checkpointer_cm:
            try:
                print(f"[DEBUG] Closing checkpointer connection for thread_id={thread_id}")
                await checkpointer_cm.__aexit__(None, None, None)
                print("[DEBUG] Checkpointer connection closed successfully")
            except Exception as close_err:
                print(f"Erro ao fechar conexão SQLite: {close_err}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        chat_generator(request.message, request.thread_id, request.user_id),
        media_type="text/event-stream",
    )
