import json
import logging
import os
import uuid

import pandas as pd
from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.vectorstores import FAISS

import config as _
from gemini import Gemini
from prompts import PROMPT_IMAGE, PROMPT_TEXT
from schemas import RESPONSE_SCHEMA
from tokenizer import TokenizerWrapper


class NoNewPDFError(Exception):
    """Custom exception for no new PDF found."""


class NoCacheFoundError(Exception):
    """Custom exception for no cache found."""


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
        pdfs_to_delete: list[str] = _.PDF_TO_DELETE,
        threads: int = _.THREADS,
        dont_summarize: bool = _.DONT_SUMMARIZE,
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
        self.tokenizer_model = tokenizer_model
        self.threads = threads
        self.dont_summarize = dont_summarize

        self._start_docling_config()

        if pdfs_to_delete:
            self.pdf_to_delete = pdfs_to_delete

    def _start_docling_config(self):
        logging.info("Starting docling configuration...")
        self.tokenizer_tool = TokenizerWrapper(model_name=self.tokenizer_model, max_length=self.token_size)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer_tool,
            max_tokens=self.token_size,
            merge_peers=True,
        )
        accelerator_options = AcceleratorOptions(num_threads=self.threads, device=AcceleratorDevice.CPU)
        pipeline_options = PdfPipelineOptions(generate_picture_images=True)
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.enable_remote_services = True
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def _save_cache(self):
        logging.info(f"Saving cache to {self.cache}...")
        file_exists = os.path.exists(self.cache)
        self.filtered_df.to_csv(self.cache, mode="a", index=False, header=not file_exists)

    def _load_cache(self):
        logging.info(f"Loading {self.cache}...")
        self.cache_df = pd.read_csv(self.cache)
        self.cache_df["metadata"] = self.cache_df["metadata"].apply(json.loads)

        self.pdf_list = set(self.cache_df["metadata"].apply(lambda x: x["filename"]))

    def _check_chache(self):
        logging.info(f"Checking if cache exists {self.cache}...")
        if os.path.exists(self.cache):
            return True
        return False

    def _diff_vs_cache(self):
        logging.info("Diffing PDFs vs Cache...")
        pdf_path_set = set(self.pdf_paths)
        self.pdf_paths = list(pdf_path_set - self.pdf_list)
        if len(self.pdf_paths) == 0:
            raise NoNewPDFError("No new PDF was found...")

    def recover_deleted_pdfs_from_cache(self):
        logging.info("Recovering deleted PDFs from cache...")
        self.cache_to_delete = self.cache_df[
            self.cache_df["metadata"].apply(lambda x: x["filename"] in self.pdf_to_delete)
        ]
        if self.cache_to_delete.empty:
            raise NoCacheFoundError("No matching PDFs found in cache to delete.")
        self.merged_df = self.cache_df[~self.cache_df["metadata"].apply(lambda x: x["filename"] in self.pdf_to_delete)]
        self.uudis_to_remove = self.cache_to_delete["id"].tolist()

    def delete_uuids_from_vectorstore(self):
        logging.info("Deleting UUIDs from vectorstore...")
        self.vectorstore.delete(ids=self.uudis_to_remove)
        logging.info("Saving vectorstore after deletions...")
        self.vectorstore.save_local(self.vectorstore_path)

    def _find_pdf(self):
        logging.info("Searching for PDF files...")
        self.pdf_paths = []
        for root, _loop, files in os.walk(self.pdf_folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    self.pdf_paths.append(full_path)

    def _chunking_documents_with_docling(self):
        def extract_text_from_chunk(chunks, directory):
            return [
                {
                    "id": str(uuid.uuid1()),
                    "contents": chunk.text,
                    "metadata": json.dumps(
                        {
                            "filename": os.path.join(directory, chunk.meta.origin.filename),
                            "page_numbers": sorted(
                                {prov.page_no for item in chunk.meta.doc_items for prov in item.prov}
                            )
                            or None,
                            "title": (chunk.meta.headings[0] if chunk.meta.headings else None),
                        }
                    ),
                }
                for chunk in chunks
            ]

        def extract_image_from_chunk(picture, filename):
            image_uri = str(
                picture.image.get("uri") if isinstance(picture.image, dict) else getattr(picture.image, "uri", None)
            )

            if image_uri.startswith("data:") and "base64," in image_uri:
                image_uri = image_uri.split("base64,")[-1]

            page_numbers = sorted({prov.page_no for prov in getattr(picture, "prov", [])}) or None

            return {
                "id": str(uuid.uuid1()),
                "contents": image_uri,
                "metadata": json.dumps(
                    {
                        "filename": filename,
                        "page_numbers": page_numbers,
                        "title": None,
                    }
                ),
            }

        image_chunks = []
        text_chunks = []
        for pdf_path in self.pdf_paths:
            result = self.converter.convert(source=pdf_path)
            chunk_iter = self.chunker.chunk(dl_doc=result.document)
            chunks = list(chunk_iter)
            text_chunks.extend(extract_text_from_chunk(chunks, os.path.dirname(pdf_path)))
            for picture in result.document.pictures:
                image_chunks.append(extract_image_from_chunk(picture, pdf_path))

        self.text_chunks_df = pd.DataFrame(text_chunks)
        self.image_chunks_df = pd.DataFrame(image_chunks)

    def _summarize_df(self, df, content="text"):
        def update_metadata_and_summary(df):
            flat = pd.json_normalize(df["summarized_content"]).reset_index(drop=True)
            df = df.reset_index(drop=True)
            df = df.join(flat)
            return df.drop(columns=["summarized_content"])

        processed_df = df.copy()
        processed_df["summarized_content"] = processed_df["contents"].apply(
            self._generate_text_summaries if content == "text" else self._genenate_image_summaries
        )
        processed_df = processed_df.dropna(subset=["summarized_content"])
        processed_df["summarized_content"] = processed_df["summarized_content"].apply(
            lambda x: json.loads(x)[0] if len(json.loads(x)) > 0 else None
        )
        return update_metadata_and_summary(processed_df)

    def _filter_reference_info(self):
        filtered_df = self.merged_df.copy()
        filtered_df = filtered_df[filtered_df["isReference"].astype(str).str.lower() == "false"]
        self.filtered_df = filtered_df.drop(columns="isReference").reset_index(drop=True)

    def _summarization_process(self):
        if not self.dont_summarize:
            logging.info("Summarizing chunks...")
            self.text_processed = self._summarize_df(self.text_chunks_df, content="text")
            self.image_processed = self._summarize_df(self.image_chunks_df, content="image")
            merged_df = pd.concat([self.text_processed, self.image_processed]).reset_index(drop=True)
        else:
            logging.info("Skipping summarization...")
            merged_df = self.text_chunks_df.copy()
            merged_df["summary"] = merged_df["contents"]

        self.merged_df = merged_df.dropna(subset=["summary"])

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

        self.texts_for_vectorstore = self.filtered_df["summary"].tolist()
        self.metadatas_for_vectorstore = [
            json.loads(m) if isinstance(m, str) else m for m in self.filtered_df["metadata"]
        ]
        self.ids_for_vectorstore = self.filtered_df["id"].tolist()

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
        if self._check_chache():
            self._load_cache()
            self._diff_vs_cache()
        self._chunking_documents_with_docling()
        self._summarization_process()
        if not self.dont_summarize:
            self._filter_reference_info()
        else:
            self.filtered_df = self.merged_df.copy()
        self._processing_faiss_vectorstore_data()
        self._save_faiss_vectorstore()
        self._save_cache()

    def add_from_folder(self):
        self._find_pdf()
        if not self._check_chache():
            raise NoCacheFoundError("No cache found. Try creating the vectorstor first!")
        self._load_cache()
        self._diff_vs_cache()
        self._chunking_documents_with_docling()
        self._summarization_process()
        if not self.dont_summarize:
            self._filter_reference_info()
        else:
            self.filtered_df = self.merged_df.copy()
        self._load_faiss_vectorstore()
        self._processing_faiss_vectorstore_data()
        self._adding_chunks_to_vectorstore()
        self._save_cache()

    def delete_pdfs(self):
        if not self._check_chache():
            raise NoCacheFoundError("No cache found. Try creating the vectorstor first!")
        self._load_cache()
        self.recover_deleted_pdfs_from_cache()
        self._load_faiss_vectorstore()
        self.delete_uuids_from_vectorstore()
        self._save_cache()
