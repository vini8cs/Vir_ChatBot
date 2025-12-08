import logging
import os
from typing import List

from celery import Celery

from agents.vir_chatbot.vectorstore import VectorStoreCreator

app = Celery(
    "tasks",
    broker="redis://redis:6379",
    backend="redis://redis:6379",
)


def update_task_progress(task, current: int, total: int, step: str, details: str = ""):
    """Helper function to update task progress metadata."""
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
    total_steps = 6
    try:
        update_task_progress(
            self,
            1,
            total_steps,
            "Iniciando",
            f"Processando {len(pdfs_to_add)} PDF(s)...",
        )

        vector_store_creator = VectorStoreCreator(pdfs_to_add=pdfs_to_add)
        vector_store_creator.pdf_paths = vector_store_creator.pdfs_to_add.copy()

        update_task_progress(
            self,
            2,
            total_steps,
            "Verificando Cache",
            "Verificando PDFs já existentes...",
        )
        if vector_store_creator._check_chache():
            vector_store_creator._load_cache()
            try:
                vector_store_creator._diff_vs_cache()
            except Exception:
                # NoNewPDFError - all PDFs already exist
                return {
                    "status": "Sucesso",
                    "message": "Todos os PDFs já existem no VectorStore.",
                    "current": total_steps,
                    "total": total_steps,
                    "percent": 100,
                }

        update_task_progress(
            self, 3, total_steps, "Chunking", "Extraindo chunks dos PDFs com Docling..."
        )
        vector_store_creator._chunking_documents_with_docling()

        update_task_progress(
            self, 4, total_steps, "Sumarizando", "Processando e sumarizando conteúdo..."
        )
        vector_store_creator._summarization_process()
        if not vector_store_creator.dont_summarize:
            vector_store_creator._filter_reference_info()
        else:
            vector_store_creator.filtered_df = vector_store_creator.merged_df.copy()

        update_task_progress(
            self, 5, total_steps, "VectorStore", "Atualizando VectorStore..."
        )
        vector_store_creator._processing_faiss_vectorstore_data()
        if vector_store_creator._check_vectorstore_exists():
            vector_store_creator._load_faiss_vectorstore()
            vector_store_creator._adding_chunks_to_vectorstore()
        else:
            vector_store_creator._save_faiss_vectorstore()

        update_task_progress(self, 6, total_steps, "Finalizando", "Salvando cache...")
        vector_store_creator._save_cache()

        return {
            "status": "Sucesso",
            "message": "VectorStore criado ou atualizado.",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }
    except Exception as e:
        logging.info(f"Erro na tarefa Celery: {e}")
        return {"status": "Falha", "error": str(e)}
    finally:
        for file_temp in pdfs_to_add:
            if not os.path.isfile(file_temp):
                continue
            os.remove(file_temp)


@app.task(bind=True)
def create_vectorstore_from_folder(self):
    """Cria o vectorstore do zero a partir de uma pasta de PDFs."""
    total_steps = 6
    try:
        update_task_progress(
            self, 1, total_steps, "Iniciando", "Buscando PDFs na pasta..."
        )

        vector_store_creator = VectorStoreCreator()
        vector_store_creator._find_pdf()

        pdf_count = len(vector_store_creator.pdf_paths)
        update_task_progress(
            self,
            2,
            total_steps,
            "Chunking",
            f"Extraindo chunks de {pdf_count} PDF(s) com Docling...",
        )
        vector_store_creator._chunking_documents_with_docling()

        update_task_progress(
            self, 3, total_steps, "Sumarizando", "Processando e sumarizando conteúdo..."
        )
        vector_store_creator._summarization_process()

        update_task_progress(
            self, 4, total_steps, "Filtrando", "Filtrando referências..."
        )
        if not vector_store_creator.dont_summarize:
            vector_store_creator._filter_reference_info()
        else:
            vector_store_creator.filtered_df = vector_store_creator.merged_df.copy()

        update_task_progress(
            self, 5, total_steps, "VectorStore", "Criando VectorStore..."
        )
        vector_store_creator._processing_faiss_vectorstore_data()
        vector_store_creator._save_faiss_vectorstore()

        update_task_progress(self, 6, total_steps, "Finalizando", "Salvando cache...")
        vector_store_creator._save_cache()

        return {
            "status": "Sucesso",
            "message": f"VectorStore criado do zero com {pdf_count} PDF(s).",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }
    except Exception as e:
        logging.info(f"Erro na tarefa Celery: {e}")
        return {"status": "Falha", "error": str(e)}


@app.task(bind=True)
def delete_pdfs_from_vectorstore(self, filenames: List[str]):
    total_steps = 4
    try:
        update_task_progress(
            self,
            1,
            total_steps,
            "Iniciando",
            f"Preparando exclusão de {len(filenames)} arquivo(s)...",
        )

        vector_store_creator = VectorStoreCreator(pdfs_to_delete=filenames)

        update_task_progress(
            self,
            2,
            total_steps,
            "Carregando",
            "Carregando cache e identificando registros...",
        )
        vector_store_creator._load_cache()
        vector_store_creator.recover_deleted_pdfs_from_cache()

        update_task_progress(
            self, 3, total_steps, "Deletando", "Removendo do VectorStore..."
        )
        vector_store_creator._load_faiss_vectorstore()
        vector_store_creator.delete_uuids_from_vectorstore()

        update_task_progress(
            self, 4, total_steps, "Finalizando", "Atualizando cache..."
        )
        vector_store_creator._reformat_chache_after_deletion()

        return {
            "status": "Sucesso",
            "message": f"Arquivo(s) {filenames} deletado(s) com sucesso.",
            "current": total_steps,
            "total": total_steps,
            "percent": 100,
        }
    except Exception as e:
        logging.info(f"Erro na tarefa Celery: {e}")
        return {"status": "Falha", "error": str(e)}
