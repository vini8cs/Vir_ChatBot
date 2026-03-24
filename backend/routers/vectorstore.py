import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List

import aiofiles
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

import backend.state as state
import config as _
from agents.vir_chatbot.tasks import (
    create_vectorstore_uploaded_pdfs,
    delete_pdfs_from_vectorstore,
)
from agents.vir_chatbot.vir_chatbot import load_global_vectorstore
from backend.models import DeleteFileRequest

router = APIRouter(tags=["vectorstore"])

TEMP_UPLOAD_DIR = "/tmp/temp_uploads"
Path(TEMP_UPLOAD_DIR).mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".jpg",
    ".jpeg",
    ".png",
    ".txt",
    ".tsv",
}


@router.post("/vectorstore/reload")
async def reload_vectorstore():
    """Endpoint to reload the VectorStore into memory."""
    try:
        state.global_resources["retriever"] = await load_global_vectorstore(
            retriever_limit=state.runtime_config.retriever_limit
        )
        if state.global_resources["retriever"]:
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


@router.post("/create-vectorstore-based-on-selected-pdfs/")
async def upload_pdf(
    files: list[UploadFile] = File(...),  # noqa: B008
):
    """Endpoint to create or update the vectorstore based on selected file(s).
    Supported formats: PDF, DOCX, DOC, JPEG, PNG, TXT, TSV.
    """
    unsupported = [
        f.filename
        for f in files
        if not any(
            f.filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS
        )
    ]
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type(s): {', '.join(unsupported)}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    request_id = str(uuid.uuid4())
    upload_dir = os.path.join(TEMP_UPLOAD_DIR, request_id)
    Path(upload_dir).mkdir(exist_ok=True)

    files_to_upload = []
    try:
        for file in files:
            safe_name = Path(file.filename).name
            file_path = os.path.join(upload_dir, safe_name)
            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)
            files_to_upload.append(file_path)

    except Exception as e:
        await asyncio.to_thread(shutil.rmtree, upload_dir)
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        ) from e

    task = create_vectorstore_uploaded_pdfs.delay(
        files_to_upload,
        summarize=state.runtime_config.summarize,
        gemini_model=state.runtime_config.gemini_model,
    )

    return {
        "message": "VectorStore creation started in background.",
        "task_id": task.id,
        "upload_directory": str(upload_dir),
    }


@router.post("/delete-selected-pdfs/")
async def delete_pdfs(request: DeleteFileRequest):
    """
    Endpoint to trigger the deletion of specific PDFs from the VectorStore.
    """
    if not request.filenames:
        raise HTTPException(
            status_code=400, detail="The list of files is empty."
        )

    logging.info(f"Requesting deletion for: {request.filenames}")

    task = delete_pdfs_from_vectorstore.delay(request.filenames)

    return {
        "message": "Deletion process started in background.",
        "task_id": task.id,
        "files_to_delete": request.filenames,
    }


def _get_filenames_from_metadata(metadata_str: str) -> List[str]:
    try:
        data = json.loads(metadata_str)
        return data.get("filename")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


@router.get("/pdfs/list")
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

        filenames = (
            df["metadata"]
            .apply(_get_filenames_from_metadata)
            .dropna()
            .unique()
        )
        return {"pdfs": sorted(filenames)}

    except Exception as e:
        logging.error(f"Error reading cache: {e}")
        return {"pdfs": []}
