import logging
import os
from typing import List

from celery import Celery

from agents.vir_chatbot.vectorstore import (
    NoCacheFoundError,
    NoNewPDFError,
    NoVectorStoreFoundError,
    VectorAlreadyCreatedError,
    VectorStoreCreator,
)

app = Celery(
    "tasks",
    broker="redis://redis:6379",
    backend="redis://redis:6379",
)


def update_task_progress(task, current: int, total: int, step: str, details: str = ""):
    task.update_state(
        state="PROGRESS",
        meta={
            "current": current,
            "total": total,
            "percent": int((current / total) * 100) if total > 0 else 0,
            "step": step,
            "details": details,
        },
    )


@app.task(bind=True)
def create_vectorstore_uploaded_pdfs(self, pdfs_to_add: list[str]):
    total_steps = 5
    logging.info("Building/adding new PDFs to vectorstore...")
    try:
        update_task_progress(
            self,
            1,
            total_steps,
            "Starting",
            f"Processing {len(pdfs_to_add)} PDF(s)...",
        )

        vector_store_creator = VectorStoreCreator(pdfs_to_add=pdfs_to_add)
        vector_store_creator.pdf_paths = vector_store_creator.pdfs_to_add.copy()

        update_task_progress(
            self,
            2,
            total_steps,
            "Checking Cache",
            "Checking existing PDF(s)...",
        )
        if vector_store_creator._check_chache():
            vector_store_creator._load_cache()
            vector_store_creator._diff_vs_cache()

        update_task_progress(self, 3, total_steps, "Chunking", "Chunking PDF(s) with Docling...")
        vector_store_creator._start_chunking_process()

        update_task_progress(self, 4, total_steps, "VectorStore", "Updating/Building VectorStore...")
        vector_store_creator._processing_faiss_vectorstore_data()

        if vector_store_creator._check_vectorstore_exists():
            vector_store_creator._load_faiss_vectorstore()
            vector_store_creator._adding_chunks_to_vectorstore()
        else:
            vector_store_creator._save_faiss_vectorstore()

        update_task_progress(self, 5, total_steps, "Saving in cache", "Saving cache...")
        vector_store_creator._save_cache()

        return {
            "status": "Success",
            "message": "VectorStore created/updated.",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }

    except Exception as e:
        logging.info(f"Error in Celery task: {e}")
        return {"status": "Error", "error": str(e)}
    except NoNewPDFError:
        return {
            "status": "Success",
            "message": "All the PDF(s) already exist in Vectorstore.",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }
    finally:
        for file_temp in pdfs_to_add:
            if not os.path.isfile(file_temp):
                continue
            os.remove(file_temp)


@app.task(bind=True)
def create_vectorstore_from_folder(self):
    total_steps = 6
    try:
        update_task_progress(self, 1, total_steps, "Starting", "Searching for PDFs in folder...")

        vector_store_creator = VectorStoreCreator()
        if vector_store_creator._check_chache() or vector_store_creator._check_vectorstore_exists():
            raise VectorAlreadyCreatedError
        vector_store_creator._find_pdf()

        pdf_count = len(vector_store_creator.pdf_paths)
        update_task_progress(
            self,
            2,
            total_steps,
            "Chunking",
            f"Chunking {pdf_count} PDF(s) with Docling...",
        )
        vector_store_creator._start_chunking_process()

        update_task_progress(self, 3, total_steps, "VectorStore", "Creating VectorStore...")
        vector_store_creator._processing_faiss_vectorstore_data()
        vector_store_creator._save_faiss_vectorstore()

        update_task_progress(self, 4, total_steps, "Finishing", "Saving cache...")
        vector_store_creator._save_cache()

        return {
            "status": "Success",
            "message": f"VectorStore created from scratch with {pdf_count} PDF(s).",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }
    except VectorAlreadyCreatedError as e:
        logging.info(f"Error in Celery task: {e}")
        return {"status": "Failure", "error": str(e)}


@app.task(bind=True)
def delete_pdfs_from_vectorstore(self, filenames: List[str]):
    total_steps = 4
    try:
        update_task_progress(
            self,
            1,
            total_steps,
            "Starting",
            f"Starting process to delete {len(filenames)} file(s)...",
        )

        vector_store_creator = VectorStoreCreator(pdfs_to_delete=filenames)
        if not vector_store_creator._check_chache():
            raise NoCacheFoundError

        update_task_progress(
            self,
            2,
            total_steps,
            "Loading",
            "Loading cache and identifying registers",
        )
        vector_store_creator._load_cache()
        vector_store_creator.recover_deleted_pdfs_from_cache()

        update_task_progress(self, 3, total_steps, "Deleting", "Deleting from VectorStore...")

        if not vector_store_creator._check_vectorstore_exists():
            raise NoVectorStoreFoundError

        vector_store_creator._load_faiss_vectorstore()
        vector_store_creator.delete_uuids_from_vectorstore()

        update_task_progress(self, 4, total_steps, "Finishing", "Updating cache...")
        vector_store_creator._reformat_chache_after_deletion()

        return {
            "status": "Success",
            "message": f"{len(filenames)} file(s) deleted successfully: {', '.join(filenames)}",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }
    except (NoCacheFoundError, NoVectorStoreFoundError) as e:
        logging.info(f"Error in Celery task: {e}")
        return {"status": "Error", "error": str(e)}
