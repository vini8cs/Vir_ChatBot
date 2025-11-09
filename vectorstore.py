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
        chunk_size: int = _.CHUNK_SIZE,
        combine_text_under_n_chars: int = _.COMBINE_TEXT_UNDER_N_CHARS,
        new_after_n_characters: int = _.NEW_AFTER_N_CHARACTERS,
        languages: list[str] = _.LANGUAGES,
        max_retries: int = _.MAX_RETRIES,
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
        self.chunk_size = chunk_size
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_characters = new_after_n_characters
        self.languages = languages
        self.cache = cache
        self.vectorstore_path = vectorstore_path
        self.pdf_list = []
        self.cache_df = pd.DataFrame()
        self.merged_df = None

    def _save_cache(self):
        logging.info(f"Saving cache to {self.cache}...")
        self.merged_df.to_csv(self.cache, index=False)

    def _load_cache(self):
        logging.info(f"Loading {self.cache}...")
        self.cache_df = pd.read_csv(self.cache)
        self.cache_df["metadata"] = self.cache_df["metadata"].apply(json.loads)

        self.pdf_list.extend(
            [
                os.path.join(row["metadata"].get("file_directory", ""), row["metadata"].get("filename", ""))
                for _, row in self.cache_df.iterrows()
                if "metadata" in row and row["metadata"]
            ]
        )

    def _check_chache(self):
        logging.info(f"Checking if cache exists {self.cache}...")
        if os.path.exists(self.cache):
            return True
        return False

    def _find_pdf(self):
        logging.info("Searching for PDF files...")
        self.pdf_paths = []
        for root, _loop, files in os.walk(self.pdf_folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    self.pdf_paths.append(full_path)

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
                lambda r: json.dumps({**json.loads(r["metadata"]), "isReference": r["isReference"]}), axis=1
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

        self._find_pdf()

        PROCESSED_DFS = {
            "text": [],
            "table": [],
            "image": [],
        }

        logging.info("Starting PDF chunking process...")
        for pdf_path in self.pdf_paths:
            if pdf_path in self.pdf_list:
                continue
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

        self.texts_all_df = pd.concat(PROCESSED_DFS["text"], ignore_index=True) if PROCESSED_DFS["text"] else None
        self.tables_all_df = pd.concat(PROCESSED_DFS["table"], ignore_index=True) if PROCESSED_DFS["table"] else None
        self.images_all_df = pd.concat(PROCESSED_DFS["image"], ignore_index=True) if PROCESSED_DFS["image"] else None

        dfs_to_merge = [d for d in [self.texts_all_df, self.tables_all_df, self.images_all_df] if d is not None]

        if len(dfs_to_merge) != 0:

            self.merged_df = pd.concat(
                dfs_to_merge,
                ignore_index=True,
            )

            if not self.cache_df.empty:
                logging.info("Merging new pdf(s) with cache")
                self.merged_df = pd.concat([self.cache_df, self.merged_df], ignore_index=True)

            self._save_cache()
        else:
            logging.info("No new pdf was added to the vectorstore cache...")

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

    def _create_faiss_vectorstore(self):
        logging.info("Creating vectorstore with faiss...")
        texts = self.merged_df["summary"].tolist()
        metadatas = self.merged_df["metadata"].tolist()
        ids = self.merged_df["id"].tolist()

        self.vectorstore = FAISS.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        self.vectorstore.save_local(self.vectorstore_path)

    def create_vectorstore(self):
        if self._check_chache():
            self._load_cache()

        self._chunking_pdfs()

        if self._check_vectorstore_exists():
            self._load_faiss_vectorstore()
            return

        if not self.merged_df:
            return

        self._create_faiss_vectorstore()
        self._load_faiss_vectorstore()
