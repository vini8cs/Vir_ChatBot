import os
import shutil
import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from tasks import (
    create_vectorstore_from_folder,
    create_vectorstore_uploaded_pdfs,
    delete_pdfs_from_vectorstore,
)


class DeleteFileRequest(BaseModel):
    filenames: List[str]


app = FastAPI(
    title="Vir_ChatBot API",
    description="API para interagir com as funcionalidades do Vir_ChatBot.",
    version="0.1.0",
)

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
