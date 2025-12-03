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


@app.task
def create_vectorstore_uploaded_pdfs(pdfs_to_add: list[str]):
    try:
        vector_store_creator = VectorStoreCreator(pdfs_to_add=pdfs_to_add)
        vector_store_creator.add_from_folder()
        return {
            "status": "Sucesso",
            "message": "VectorStore criado ou atualizado.",
        }
    except Exception as e:
        logging.info(f"Erro na tarefa Celery: {e}")
        return {"status": "Falha", "error": str(e)}
    finally:
        for file_temp in pdfs_to_add:
            if not os.path.isfile(file_temp):
                continue
            os.remove(file_temp)


@app.task
def create_vectorstore_from_folder():
    """Cria o vectorstore do zero a partir de uma pasta de PDFs."""
    try:
        vector_store_creator = VectorStoreCreator()
        vector_store_creator.build_vectorstore_from_zero()
        return {
            "status": "Sucesso",
            "message": "VectorStore criado do zero com sucesso.",
        }
    except Exception as e:
        logging.info(f"Erro na tarefa Celery: {e}")
        return {"status": "Falha", "error": str(e)}


@app.task
def delete_pdfs_from_vectorstore(filenames: List[str]):
    try:
        vector_store_creator = VectorStoreCreator(pdfs_to_delete=filenames)
        vector_store_creator.delete_pdfs()
        return {
            "status": "Sucesso",
            "message": f"Arquivo(s) {filenames} deletado(s) com sucesso.",
        }
    except Exception as e:
        logging.info(f"Erro na tarefa Celery: {e}")
        return {"status": "Falha", "error": str(e)}
