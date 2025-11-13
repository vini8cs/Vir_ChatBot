# import json
# import uuid
# import base64
# import io
import json
import logging
import os

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from PIL import Image
# import re
from datetime import datetime

# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from summarize_pdf_vertexai import summarize_pdfs_vertexai
# from langsmith import Client
import pandas as pd
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import FAISS

# import logging
# from unstructured_client.models import operations, shared
# from langchain_google_vertexai import ChatVertexAI, VertexAI
from timescale_vector.client import uuid_from_time

# import uuid
from unstructured.documents.elements import CompositeElement, Image, Table

# from dotenv import load_dotenv
# from pathlib import Path
# import os
# from unstructured_client import UnstructuredClient
# from os.path import join
from unstructured.partition.pdf import partition_pdf

import config as _
from gemini import Gemini
from prompts import PROMPT_IMAGE, PROMPT_TEXT
from schemas import RESPONSE_SCHEMA
from tokenizer import TokenizerWrapper

# from unstructured_client import UnstructuredClient

# from unstructured_client.models import operations, shared

# from os.path import join, relpath


# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# from psycopg_pool import AsyncConnectionPool
# from psycopg.rows import dict_row
# from typing import List, Tuple, Any
# from google import genai
# from google.genai import types
# import numpy as np
# from numpy.linalg import norm
# import time
# from timescale_vector import client


# import PyPDF2


class VectorStoreCreator(Gemini):
    def __init__(
        self,
        pdf_folder: str,
        output_folder: str,
        cache: str,
        vectorstore_path: str,
        gemini_model: str = _.GEMINI_MODEL,
        embedding_model: str = _.EMBEDDING_MODEL,
        temperature: float = _.TEMPERATURE,
        max_output_tokens: int = _.MAX_OUTPUT_TOKENS,
        token_size: int = _.TOKEN_SIZE,
        languages: list[str] = _.LANGUAGES,
        max_retries: int = _.MAX_RETRIES,
        tokenizer_model: str = _.TOKENIZER_MODEL,
    ):
        super().__init__(
            gemini_model=gemini_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_schema=RESPONSE_SCHEMA,
            max_retries=max_retries,
            prompt_text=PROMPT_TEXT,
            prompt_image=PROMPT_IMAGE,
            gemini_embedding_model=embedding_model,
        )
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        self.embedding_model = embedding_model
        self.token_size = token_size
        self.languages = languages
        self.cache = cache
        self.vectorstore_path = vectorstore_path
        self.tokenizer_tool = TokenizerWrapper(model_name=tokenizer_model, max_length=8191)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer_tool,
            max_tokens=self.token_size,
            merge_peers=True,
        )
        self.converter = DocumentConverter()

    def _save_cache(self):
        file_exists = os.path.exists(self.cache)
        self.merged_df.to_csv(self.cache, mode="a", index=False, header=not file_exists)

    def _load_cache(self):
        logging.info(f"Loading {self.cache}...")
        self.cache_df = pd.read_csv(self.cache)
        self.cache_df["metadata"] = self.cache_df["metadata"].apply(json.loads)

        self.pdf_list = set(self.cache_df["metadata"].apply(lambda x: os.path.join(x["file_directory"], x["filename"])))

    def _check_chache(self):
        logging.info(f"Checking if cache exists {self.cache}...")
        if os.path.exists(self.cache):
            return True
        return False

    def _diff_vs_cache(self):
        print("testing pdf cache")
        print(self.pdf_list)
        pdf_path_set = set(self.pdf_paths)
        print(pdf_path_set)
        self.pdf_paths = list(pdf_path_set - self.pdf_list)
        if len(self.pdf_paths) == 0:
            raise ValueError("No new PDF was found...")

    def _find_pdf(self):
        logging.info("Searching for PDF files...")
        self.pdf_paths = []
        for root, _loop, files in os.walk(self.pdf_folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    self.pdf_paths.append(full_path)

    def _chunking_documents_with_docling(self):
        def process_chunk(chunks, directory):
            return [
                {
                    "text": chunk.text,
                    "metadata": json.dumps(
                        {
                            "filename": os.path.join(directory, chunk.meta.origin.filename),
                            "page_numbers": [
                                page_no
                                for page_no in sorted(
                                    set(prov.page_no for item in chunk.meta.doc_items for prov in item.prov)
                                )
                            ]
                            or None,
                            "title": (chunk.meta.headings[0] if chunk.meta.headings else None),
                        }
                    ),
                }
                for chunk in chunks
            ]

        documents_chunks = []
        for pdf_path in self.pdf_paths:
            result = self.converter.convert(source=pdf_path)
            chunk_iter = self.chunker.chunk(dl_doc=result.document)
            chunks = list(chunk_iter)
            documents_chunks.extend(process_chunk(chunks, os.path.dirname(pdf_path)))

        self._document_chunks = pd.DataFrame(documents_chunks)

    def _pdf_chunking_process(self, pdf_path):
        return partition_pdf(
            filename=pdf_path,
            infer_table_structure=True,
            strategy="hi_res",
            languages=self.languages,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=self.chunk_size,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_characters=self.new_after_n_characters,
        )

    def _chunking_pdfs(self):
        def separate_data(chunks):
            tables_data = []
            texts_data = []
            images_data = []

            for chunk in chunks:
                if isinstance(chunk, Table):
                    tables_data.append(
                        {
                            "id": str(uuid_from_time(datetime.now())),
                            "metadata": json.dumps(chunk.metadata.to_dict()),
                            "contents": chunk.text,
                        }
                    )

                elif isinstance(chunk, CompositeElement):
                    texts_data.append(
                        {
                            "id": str(uuid_from_time(datetime.now())),
                            "contents": chunk.text,
                            "metadata": json.dumps(chunk.metadata.to_dict()),
                        }
                    )
                    if hasattr(chunk.metadata, "orig_elements"):
                        for element in chunk.metadata.orig_elements:
                            if isinstance(element, Image):
                                images_data.append(
                                    {
                                        "id": str(uuid_from_time(datetime.now())),
                                        "contents": element.metadata.image_base64,
                                        "metadata": json.dumps(chunk.metadata.to_dict()),
                                    }
                                )

                elif isinstance(chunk, Image):
                    images_data.append(
                        {
                            "id": str(uuid_from_time(datetime.now())),
                            "metadata": json.dumps(chunk.metadata.to_dict()),
                            "contents": chunk.metadata.image_base64,
                        }
                    )

            logging.info(
                f"Separated {len(tables_data)} tables, {len(texts_data)} texts, and {len(images_data)} images."
            )

            tables_df = pd.DataFrame(tables_data)
            texts_df = pd.DataFrame(texts_data)
            images_df = pd.DataFrame(images_data)

            return tables_df, texts_df, images_df

        def update_metadata_and_summary(df):
            flat = pd.json_normalize(df["summarized_content"]).reset_index(drop=True)
            df = df.reset_index(drop=True)
            df = df.join(flat)
            df["metadata"] = df.apply(
                lambda r: json.dumps({**json.loads(r["metadata"]), "isReference": r["isReference"]}),
                axis=1,
            )
            return df.drop(columns=["isReference"])

        def process_df(df, content="text"):
            processed_df = df.copy()
            processed_df["summarized_content"] = processed_df["contents"].apply(
                self._generate_text_summaries if content == "text" else self._genenate_image_summaries
            )
            processed_df = processed_df.dropna(subset=["summarized_content"])
            processed_df["summarized_content"] = processed_df["summarized_content"].apply(lambda x: json.loads(x)[0])
            return update_metadata_and_summary(processed_df)

        PROCESSED_DFS = {
            "text": [],
            "table": [],
            "image": [],
        }

        logging.info("Starting PDF chunking process...")
        for pdf_path in self.pdf_paths:
            chunks = self._pdf_chunking_process(pdf_path)
            logging.info(f"Chunked {pdf_path} into {len(chunks)} chunks.")
            tables_df, texts_df, images_df = separate_data(chunks)

            print(f"tables_df: {tables_df.shape}, texts_df: {texts_df.shape}, images_df: {images_df.shape}")
            for name, df in [
                ("text", texts_df),
                ("table", tables_df),
                ("image", images_df),
            ]:
                if df.empty:
                    logging.warning(f"Dataframe extracted with {name} type for {pdf_path} is empty. Continuing...")
                    continue

                logging.info(f"Processing {name} dataframe for {pdf_path}...")
                df = process_df(df, content="image") if name == "image" else process_df(df)
                PROCESSED_DFS[name].append(df)

        texts_all_df = pd.concat(PROCESSED_DFS["text"], ignore_index=True) if PROCESSED_DFS["text"] else None
        tables_all_df = pd.concat(PROCESSED_DFS["table"], ignore_index=True) if PROCESSED_DFS["table"] else None
        images_all_df = pd.concat(PROCESSED_DFS["image"], ignore_index=True) if PROCESSED_DFS["image"] else None

        dfs_to_merge = [d for d in [texts_all_df, tables_all_df, images_all_df] if d is not None]

        self.merged_df = pd.concat(
            dfs_to_merge,
            ignore_index=True,
        )

    def _check_vectorstore_exists(self):
        logging.info("Checking if vectorstore exist...")
        return os.path.exists(self.vectorstore_path) and os.listdir(self.vectorstore_path)

    def _load_faiss_vectorstore(self):
        logging.info("Loading vectorstore...")
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _processing_faiss_vectorstore_data(self):
        logging.info("Creating vectorstore with faiss...")

        self.texts_for_vectorstore = self.merged_df["summary"].tolist()
        self.metadatas_for_vectorstore = [
            json.loads(m) if isinstance(m, str) else m for m in self.merged_df["metadata"]
        ]
        self.ids_for_vectorstore = self.merged_df["id"].tolist()

    def _adding_chunks_to_vectorstore(self):
        logging.info("Adding chunks to vectorstore...")
        self.vectorstore.from_texts(
            self.texts_for_vectorstore,
            self.embeddings,
            metadatas=self.metadatas_for_vectorstore,
            ids=self.ids_for_vectorstore,
        )

    def _save_faiss_vectorstore(self):
        logging.info("Saving vectorstore...")
        self.vectorstore = FAISS.from_texts(
            self.texts_for_vectorstore,
            self.embeddings,
            metadatas=self.metadatas_for_vectorstore,
            ids=self.ids_for_vectorstore,
        )
        self.vectorstore.save_local(self.vectorstore_path)

    def build_vectorstore_from_folder(self):
        logging.info("Bulding a new vectorstore from zero...")
        self._find_pdf()
        self._chunking_documents_with_docling()
        # self._chunking_pdfs()
        # self._processing_faiss_vectorstore_data()
        # self._save_faiss_vectorstore()
        # self._save_cache()

    def add_from_folder(self):
        self._find_pdf()
        if not self._check_chache():
            raise ValueError("No cache found. Try creating the vectorstor first!")
        self._load_cache()
        self._diff_vs_cache()
        self._chunking_pdfs()
        self._load_faiss_vectorstore()
        self._processing_faiss_vectorstore_data()
        self._adding_chunks_to_vectorstore()
        self._save_cache()
