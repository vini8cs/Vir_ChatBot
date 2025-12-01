import logging
import os

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
