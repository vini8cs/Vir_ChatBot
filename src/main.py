import os
import shutil
import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile

from tasks import create_vectorstore_uploaded_pdfs

# from typing import List


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


# # (Opcional) Endpoint para verificar o status da tarefa
# def get_task_status(task_id: str):
#     """Verifica o status de uma tarefa Celery pelo seu ID."""
#     task_result = create_vectorstore_from_folder_task.AsyncResult(task_id)

#     response = {
#         "task_id": task_id,
#         "status": task_result.status,
#         "result": task_result.result if task_result.ready() else None
#     }
#     return response


# @app.post("/create-vectorstore-based-on-selected-folder/")
# async def create_vectorstore():
#     cache = os.path.join(VECTOR_DATABASE, PROCESSED_CACHE)
#     processed = get_processed_pdfs(cache)
#     pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf") and f not in processed]
#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#     if not pdfs:
#         if os.path.exists(VECTOR_DATABASE) and os.listdir(VECTOR_DATABASE):
#             logging.info("No new PDFs to process. Loading existing vector store.")
#             app.state.vectorstore = FAISS.load_local(VECTOR_DATABASE,
# embeddings, allow_dangerous_deserialization=True)
#         return {"status": "No new PDFs to process"}
#     try:
#         logging.info("Creating/updating vector store...")
#         result = create_vectorstore_task.delay(
#             pdf_folder=PDF_FOLDER,
#             output_folder=OUTPUT_FOLDER,
#             languages=["eng", "pt"],
#             temperature=0.1,
#             max_output_tokens=2048,
#             max_concurrency=3,
#             chunk_size=10000,
#             chunk_overlap=1000,
#             pdf_files=pdfs,
#         )
#         return {"status": "Vector store task started", "task_id": result.id}

#     except Exception as e:
#         logging.error(f"Error creating/updating vector store: {e}")
#         raise HTTPException(status_code=500, detail="Error creating/updating vector store")


# @app.get("/vectorstore-status/{task_id}")
# async def vectorstore_status(task_id: str):
#     result = AsyncResult(task_id)
#     cache = os.path.join(VECTOR_DATABASE, PROCESSED_CACHE)
#     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#     if result.status == "SUCCESS":
#         pdfs = result.result
#         add_to_processed(pdfs, cache)
#         app.state.vectorstore = FAISS.load_local(VECTOR_DATABASE, embeddings, allow_dangerous_deserialization=True)
#         return {"status": "Vector store loaded", "pdfs": pdfs}
#     return {"status": result.status}


# @app.post("/ask/")
# def ask_question(request: QuestionRequest):
#     try:
#         chain = app.state.chain
#         vectorstore = app.state.vectorstore
#         if vectorstore is None:
#             raise HTTPException(status_code=404, detail="Vector store not loaded. Process PDFs first.")
#         docs = vectorstore.similarity_search(request.question)
#         retriever = vectorstore.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, chain)
#         response = retrieval_chain.invoke({"input": request.question,
# "context": app.state.context, "documents": docs})
#         answer = response["answer"]
#         app.state.context += f"\nQ: {request.question}\nA: {answer}\n"
#         return {"answer": answer}
#     except Exception as e:
#         logging.error(f"Error processing question: {e}")
#         raise HTTPException(status_code=500, detail="Error processing question")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     vectorstore_path = "faiss_data"
#     Path(vectorstore_path).mkdir(exist_ok=True)

#         create_vectorstore(
#             vectorstore_path=vectorstore_path,
#             text_json=text_json,
#             table_json=table_json,
#             image_json=image_json,
#         )
#         add_to_processed(pdfs)
#     yield

# app = FastAPI(lifespan=lifespan)

# @app.post("/ask")
# async def ask_question(request: QuestionRequest):

#     vectorstore_path = "faiss_data"
#     if not os.path.exists(vectorstore_path):
#         return {"error": "Vector store not found. Please process PDFs first."}

#     response = query_vectorstore(
#         vectorstore_path=vectorstore_path,
#         question=question,
#         model_name=GEMINI_MODEL,
#         temperature=0.1,
#         max_output_tokens=1024,
#         top_k=3
#     )
#     return {"response": response}
