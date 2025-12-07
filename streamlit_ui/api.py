import json
import os
import shutil
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

import config as _
from agents.vir_chatbot.vir_chatbot import create_graph, load_global_vectorstore
from tasks import (
    create_vectorstore_from_folder,
    create_vectorstore_uploaded_pdfs,
    delete_pdfs_from_vectorstore,
)

global_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Carregando VectorStore na memória...")
    try:
        global_resources["retriever"] = await load_global_vectorstore()
        print("VectorStore carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar VectorStore: {e}")
        global_resources["retriever"] = None

    yield

    global_resources.clear()
    print("Recursos limpos.")


app = FastAPI(lifespan=lifespan)


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


TEMP_UPLOAD_DIR = "/tmp/temp_uploads"
Path(TEMP_UPLOAD_DIR).mkdir(exist_ok=True)


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
        with sqlite3.connect(sqlite_db_path) as conn:
            cur = conn.cursor()
            query = """
            SELECT DISTINCT thread_id 
            FROM checkpoints 
            WHERE json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            """  # noqa: W291
            cur.execute(query, (str(user_id),))
            rows = cur.fetchall()

            return {"threads": [{"thread_id": r[0]} for r in rows]}

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


async def chat_generator(user_input: str, thread_id: str, user_id: str):
    """
    Gera a resposta em streaming e garante o fechamento da conexão do banco.
    """
    print(f"[DEBUG] chat_generator called with thread_id={thread_id}, user_id={user_id}")

    retriever = global_resources.get("retriever")
    if not retriever:
        yield f"data: {json.dumps({'error': 'VectorStore não inicializado.'})}\n\n"
        return

    graph_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    print(f"[DEBUG] graph_config: {graph_config}")

    graph = None
    checkpointer_cm = None
    try:
        graph, checkpointer_cm = await create_graph(global_retriever=retriever)
        print("[DEBUG] Graph created successfully")

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
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
