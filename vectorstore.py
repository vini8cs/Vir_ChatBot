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

from config import (
    CHUNK_SIZE,
    COMBINE_TEXT_UNDER_N_CHARS,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    LANGUAGES,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    NEW_AFTER_N_CHARACTERS,
    PROMPT_IMAGE,
    PROMPT_TEXT,
    RESPONSE_SCHEMA,
    TEMPERATURE,
)
from gemini import Gemini

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
        gemini_model: str = GEMINI_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        temperature: float = TEMPERATURE,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        chunk_size: int = CHUNK_SIZE,
        combine_text_under_n_chars: int = COMBINE_TEXT_UNDER_N_CHARS,
        new_after_n_characters: int = NEW_AFTER_N_CHARACTERS,
        languages: list[str] = LANGUAGES,
        max_retries: int = MAX_RETRIES,
    ):
        super().__init__(
            gemini_model=gemini_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_schema=RESPONSE_SCHEMA,
            max_retries=max_retries,
            prompt_text=PROMPT_TEXT,
            prompt_image=PROMPT_IMAGE,
        )
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_characters = new_after_n_characters
        self.languages = languages
        self.cache = cache

    def _save_cache(self):
        self.merged_df.to_csv(self.cache, index=False)

    def _load_cache(self):
        self.cache = pd.read_csv(self.cache)

    def _find_pdf(self):
        self.pdf_paths = []
        for root, _, files in os.walk(self.pdf_folder):
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

        def process_row(row):
            summarized = row["summarized_content"]
            if isinstance(summarized, dict):
                metadata = json.loads(row["metadata"])
                if "isReference" in summarized:
                    metadata["isReference"] = summarized["isReference"]
                row["metadata"] = json.dumps(metadata)
                row["summary"] = summarized.get("summary")
            else:
                row["summary"] = None
            return row

        def update_metadata_and_summary(df):
            return df.apply(process_row, axis=1)

        texts_df_list = []
        tables_df_list = []
        images_df_list = []

        self._find_pdf()
        for pdf_path in self.pdf_paths:
            chunks = self._pdf_chunking_process(pdf_path)
            tables_df, texts_df, images_df = separate_data(chunks)

            texts_df["summarized_content"] = texts_df["contents"].apply(self._generate_text_summaries)
            tables_df["summarized_content"] = tables_df["contents"].apply(self._generate_text_summaries)
            images_df["summarized_content"] = images_df["contents"].apply(self._genenate_image_summaries)

            texts_df = update_metadata_and_summary(texts_df)
            tables_df = update_metadata_and_summary(tables_df)
            images_df = update_metadata_and_summary(images_df)

            texts_df = texts_df.dropna(subset=["summarized_content"])
            tables_df = tables_df.dropna(subset=["summarized_content"])
            images_df = images_df.dropna(subset=["summarized_content"])

            texts_df_list.append(texts_df)
            tables_df_list.append(tables_df)
            images_df_list.append(images_df)

        self.texts_all_df = pd.concat(texts_df_list, ignore_index=True)
        self.tables_all_df = pd.concat(tables_df_list, ignore_index=True)
        self.images_all_df = pd.concat(images_df_list, ignore_index=True)

        self.merged_df = pd.concat([self.texts_all_df, self.tables_all_df, self.images_all_df], ignore_index=True)
