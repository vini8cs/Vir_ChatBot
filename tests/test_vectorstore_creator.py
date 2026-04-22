"""Tests for VectorStoreCreator instance methods in
agents/vir_chatbot/vectorstore.py.

Heavy deps (Docling converter/chunker, FAISS I/O, HuggingFace tokenizer)
are mocked. VectorStoreCreator inherits from Gemini, whose
``GoogleGenerativeAIEmbeddings`` is already stubbed in conftest.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.vir_chatbot.vectorstore import (
    NoCacheFoundError,
    NoNewPDFError,
    NoVectorStoreFoundError,
    VectorStoreCreator,
)


@pytest.fixture()
def creator(tmp_path):
    """Return a VectorStoreCreator with Docling + tokenizer stubs."""
    with (
        patch(
            "agents.vir_chatbot.vectorstore.TokenizerWrapper"
        ) as mock_tokenizer,
        patch("agents.vir_chatbot.vectorstore.HybridChunker") as mock_chunker,
        patch(
            "agents.vir_chatbot.vectorstore.DocumentConverter"
        ) as mock_converter,
    ):
        mock_tokenizer.return_value = MagicMock()
        mock_chunker.return_value = MagicMock()
        mock_converter.return_value = MagicMock()
        c = VectorStoreCreator(
            pdfs_to_add=["/fake/path/one.pdf"],
            pdfs_to_delete=["/fake/path/delete.pdf"],
            cache=str(tmp_path),
            vectorstore_path=str(tmp_path / "vs"),
        )
    return c


class TestInit:
    def test_pdfs_to_add_is_stored(self, creator):
        assert creator.pdfs_to_add == ["/fake/path/one.pdf"]

    def test_pdfs_to_delete_is_stored(self, creator):
        assert creator.pdf_to_delete == ["/fake/path/delete.pdf"]

    def test_cache_path_is_joined_with_cache_csv(self, creator, tmp_path):
        assert creator.cache == os.path.join(str(tmp_path), "cache.csv")

    def test_start_docling_config_attaches_converter(self, creator):
        assert creator.converter is not None
        assert creator.chunker is not None
        assert creator.tokenizer_tool is not None


class TestCacheHelpers:
    def test_check_chache_returns_false_when_missing(self, creator):
        assert creator._check_chache() is False

    def test_check_chache_returns_true_when_file_exists(
        self, creator, tmp_path
    ):
        path = tmp_path / "cache.csv"
        path.write_text("id,contents,metadata\n")
        creator.cache = str(path)
        assert creator._check_chache()

    def test_save_cache_writes_expected_columns(self, creator):
        creator.filtered_df = pd.DataFrame(
            [
                {
                    "id": "1",
                    "contents": "x",
                    "summary": "s",
                    "metadata": {"filename": "a.pdf"},
                }
            ]
        )
        creator._save_cache()
        df = pd.read_csv(creator.cache)
        assert df.loc[0, "id"] == 1 or df.loc[0, "id"] == "1"
        assert "a.pdf" in df.loc[0, "metadata"]

    def test_save_cache_appends_without_duplicating_header(self, creator):
        creator.filtered_df = pd.DataFrame(
            [{"id": "1", "summary": "s", "metadata": {"filename": "a"}}]
        )
        creator._save_cache()
        creator.filtered_df = pd.DataFrame(
            [{"id": "2", "summary": "s", "metadata": {"filename": "b"}}]
        )
        creator._save_cache()
        df = pd.read_csv(creator.cache)
        assert len(df) == 2

    def test_reformat_chache_overwrites_file(self, creator):
        creator.filtered_df = pd.DataFrame(
            [{"id": "1", "metadata": {"filename": "a"}}]
        )
        creator._save_cache()
        creator.filtered_df = pd.DataFrame(
            [{"id": "99", "metadata": {"filename": "z"}}]
        )
        creator._reformat_chache_after_deletion()
        df = pd.read_csv(creator.cache)
        assert len(df) == 1
        assert int(df.loc[0, "id"]) == 99

    def test_load_cache_parses_metadata_json(self, creator):
        pd.DataFrame(
            [
                {
                    "id": "1",
                    "metadata": json.dumps({"filename": "a.pdf"}),
                }
            ]
        ).to_csv(creator.cache, index=False)
        creator._load_cache()
        assert creator.cache_df.loc[0, "metadata"] == {"filename": "a.pdf"}
        assert creator.pdf_list == {"a.pdf"}


class TestDiffVsCache:
    def test_filters_out_already_cached_pdfs(self, creator):
        creator.pdf_paths = ["/x/a.pdf", "/x/b.pdf"]
        creator.pdf_list = {"a.pdf"}
        creator._diff_vs_cache()
        assert creator.pdf_paths == ["/x/b.pdf"]

    def test_raises_when_all_already_cached(self, creator):
        creator.pdf_paths = ["/x/a.pdf"]
        creator.pdf_list = {"a.pdf"}
        with pytest.raises(NoNewPDFError):
            creator._diff_vs_cache()


class TestRecoverDeletedPdfs:
    def test_sets_uuids_and_filtered_df(self, creator):
        creator.cache_df = pd.DataFrame(
            [
                {"id": "u1", "metadata": {"filename": "a.pdf"}},
                {"id": "u2", "metadata": {"filename": "b.pdf"}},
                {"id": "u3", "metadata": {"filename": "a.pdf"}},
            ]
        )
        creator.pdf_to_delete = ["a.pdf"]
        creator.recover_deleted_pdfs_from_cache()
        assert sorted(creator.uudis_to_remove) == ["u1", "u3"]
        assert creator.filtered_df["id"].tolist() == ["u2"]

    def test_raises_when_no_match(self, creator):
        creator.cache_df = pd.DataFrame(
            [{"id": "u1", "metadata": {"filename": "a.pdf"}}]
        )
        creator.pdf_to_delete = ["nonexistent.pdf"]
        with pytest.raises(NoCacheFoundError):
            creator.recover_deleted_pdfs_from_cache()


class TestDeleteUuidsFromVectorstore:
    def test_calls_delete_and_save(self, creator, tmp_path):
        vs = MagicMock()
        creator.vectorstore = vs
        creator.uudis_to_remove = ["u1", "u2"]
        creator.vectorstore_path = str(tmp_path / "vs")
        creator.delete_uuids_from_vectorstore()
        vs.delete.assert_called_once_with(ids=["u1", "u2"])
        vs.save_local.assert_called_once_with(creator.vectorstore_path)

    def test_save_still_called_when_delete_raises(self, creator):
        vs = MagicMock()
        vs.delete.side_effect = Exception("boom")
        creator.vectorstore = vs
        creator.uudis_to_remove = ["u1"]
        creator.delete_uuids_from_vectorstore()
        vs.save_local.assert_called_once()


class TestChunkTxt:
    def test_splits_into_chunks_of_token_size(self, creator, tmp_path):
        creator.token_size = 3
        f = tmp_path / "doc.txt"
        f.write_text("one two three four five six seven")
        chunks = creator._chunk_txt(str(f))
        assert len(chunks) == 3
        assert all("filename" in json.loads(c["metadata"]) for c in chunks)

    def test_handles_small_files(self, creator, tmp_path):
        creator.token_size = 10
        f = tmp_path / "tiny.txt"
        f.write_text("only few words")
        chunks = creator._chunk_txt(str(f))
        assert len(chunks) == 1
        assert chunks[0]["contents"] == "only few words"

    def test_empty_file_returns_no_chunks(self, creator, tmp_path):
        creator.token_size = 10
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert creator._chunk_txt(str(f)) == []


class TestChunkTsv:
    def test_each_non_empty_row_becomes_a_chunk(self, creator, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("col1\tcol2\nhello\tworld\nfoo\tbar\n")
        chunks = creator._chunk_tsv(str(f))
        assert len(chunks) == 2
        assert "col1: hello" in chunks[0]["contents"]
        assert "col2: world" in chunks[0]["contents"]

    def test_skips_rows_with_only_nan(self, creator, tmp_path):
        f = tmp_path / "sparse.tsv"
        f.write_text("col1\tcol2\n\t\nfoo\tbar\n")
        chunks = creator._chunk_tsv(str(f))
        assert len(chunks) == 1


class TestChunkingDocumentsWithDocling:
    def test_text_chunks_extracted_from_docling_output(self, creator):
        chunk_obj = MagicMock()
        chunk_obj.text = "some text"
        prov_item = MagicMock()
        prov_item.page_no = 1
        doc_item = MagicMock()
        doc_item.prov = [prov_item]
        chunk_obj.meta.doc_items = [doc_item]
        chunk_obj.meta.headings = ["Intro"]

        result = MagicMock()
        result.document.pictures = []
        creator.converter.convert.return_value = result
        creator.chunker.chunk.return_value = iter([chunk_obj])

        text_chunks, image_chunks = creator._chunking_documents_with_docling(
            "/tmp/paper.pdf"
        )
        assert len(text_chunks) == 1
        meta = json.loads(text_chunks[0]["metadata"])
        assert meta["filename"] == "paper.pdf"
        assert meta["page_numbers"] == [1]
        assert meta["title"] == "Intro"
        assert image_chunks == []

    def test_pictures_become_image_chunks(self, creator):
        picture = MagicMock()
        picture.image = {"uri": "data:image/jpeg;base64,SGVsbG8="}
        picture.prov = []

        chunk_obj = MagicMock()
        chunk_obj.text = "t"
        chunk_obj.meta.doc_items = []
        chunk_obj.meta.headings = []

        result = MagicMock()
        result.document.pictures = [picture]
        creator.converter.convert.return_value = result
        creator.chunker.chunk.return_value = iter([chunk_obj])

        _, image_chunks = creator._chunking_documents_with_docling(
            "/tmp/paper.pdf"
        )
        assert len(image_chunks) == 1
        assert image_chunks[0]["contents"] == "SGVsbG8="

    def test_image_file_ext_adds_base64_chunk(self, creator, tmp_path):
        img_path = tmp_path / "pic.jpg"
        from PIL import Image as PILImage

        PILImage.new("RGB", (2, 2), "red").save(img_path, format="JPEG")

        chunk_obj = MagicMock()
        chunk_obj.text = "caption text"
        chunk_obj.meta.doc_items = []
        chunk_obj.meta.headings = []
        result = MagicMock()
        result.document.pictures = []
        creator.converter.convert.return_value = result
        creator.chunker.chunk.return_value = iter([chunk_obj])

        _, image_chunks = creator._chunking_documents_with_docling(
            str(img_path)
        )
        assert len(image_chunks) == 1
        assert image_chunks[0]["contents"]  # base64 string

    def test_non_rgb_image_is_converted(self, creator, tmp_path):
        """Covers vectorstore.py:365 — img.convert('RGB') for RGBA images."""
        img_path = tmp_path / "pic.png"
        from PIL import Image as PILImage

        # RGBA mode is not in {"RGB", "L"} → triggers convert("RGB")
        PILImage.new("RGBA", (2, 2), (255, 0, 0, 128)).save(
            img_path, format="PNG"
        )

        chunk_obj = MagicMock()
        chunk_obj.text = "t"
        chunk_obj.meta.doc_items = []
        chunk_obj.meta.headings = []
        result = MagicMock()
        result.document.pictures = []
        creator.converter.convert.return_value = result
        creator.chunker.chunk.return_value = iter([chunk_obj])

        _, image_chunks = creator._chunking_documents_with_docling(
            str(img_path)
        )
        assert len(image_chunks) == 1
        assert image_chunks[0]["contents"]


class TestSummarizationProcess:
    def test_returns_df_unchanged_when_empty(self, creator):
        empty_df = pd.DataFrame()
        assert creator.summarization_process(empty_df, "text").empty

    def test_skip_summarization_for_text_when_disabled(self, creator):
        creator.summarize = False
        df = pd.DataFrame([{"contents": "one two three four five", "id": 1}])
        result = creator.summarization_process(df, "text")
        assert "summary" in result.columns
        assert result["summary"].iloc[0] == "one two three four five"

    def test_filters_references_after_summarizing(self, creator):
        creator.summarize = True
        creator._summarize_df = MagicMock(
            return_value=pd.DataFrame(
                [
                    {"id": 1, "summary": "keep", "isReference": "false"},
                    {"id": 2, "summary": "drop", "isReference": "true"},
                ]
            )
        )
        df = pd.DataFrame([{"contents": "x", "id": 1}])
        result = creator.summarization_process(df, "text")
        assert list(result["summary"]) == ["keep"]
        assert "isReference" not in result.columns


class TestSummarizeDf:
    # pandas ``.apply`` treats dict-like callables oddly — use real
    # functions (not MagicMock) as the summarizer stubs.

    def test_joins_summary_columns_and_drops_intermediate(self, creator):
        payload = json.dumps([{"summary": "s", "isReference": False}])
        creator._generate_text_summaries = lambda _content: payload
        df = pd.DataFrame([{"contents": "x", "id": 1}])
        result = creator._summarize_df(df, content="text")
        assert "summary" in result.columns
        assert result.loc[0, "summary"] == "s"
        assert "summarized_content" not in result.columns

    def test_uses_image_generator_when_content_is_image(self, creator):
        payload = json.dumps([{"summary": "desc", "isReference": False}])
        creator._genenate_image_summaries = lambda _content: payload
        df = pd.DataFrame([{"contents": "b64", "id": 1}])
        result = creator._summarize_df(df, content="image")
        assert result.loc[0, "summary"] == "desc"

    def test_drops_rows_where_summary_is_none(self, creator):
        creator._generate_text_summaries = lambda _content: None
        df = pd.DataFrame([{"contents": "x", "id": 1}])
        result = creator._summarize_df(df, content="text")
        assert result.empty


class TestStartChunkingProcess:
    def test_dispatches_txt_and_tsv_and_pdf(self, creator, tmp_path):
        txt = tmp_path / "a.txt"
        txt.write_text("alpha beta gamma delta")
        tsv = tmp_path / "b.tsv"
        tsv.write_text("c1\tc2\nx\ty\n")
        pdf_path = str(tmp_path / "c.pdf")

        creator.pdf_paths = [str(txt), str(tsv), pdf_path]
        creator.token_size = 10
        creator._chunking_documents_with_docling = MagicMock(
            return_value=([{"id": "1", "metadata": "{}"}], [])
        )
        creator.summarize = False
        creator._start_chunking_process()
        assert hasattr(creator, "filtered_df")


class TestVectorstoreExistsAndIO:
    def test_check_vectorstore_exists_false_when_missing(
        self, creator, tmp_path
    ):
        creator.vectorstore_path = str(tmp_path / "missing")
        assert not creator._check_vectorstore_exists()

    def test_check_vectorstore_exists_false_when_empty_dir(
        self, creator, tmp_path
    ):
        p = tmp_path / "vs"
        p.mkdir()
        creator.vectorstore_path = str(p)
        assert not creator._check_vectorstore_exists()

    def test_check_vectorstore_exists_true_when_populated(
        self, creator, tmp_path
    ):
        p = tmp_path / "vs"
        p.mkdir()
        (p / "index.faiss").write_bytes(b"x")
        creator.vectorstore_path = str(p)
        assert creator._check_vectorstore_exists()

    def test_load_faiss_calls_load_local(self, creator, tmp_path):
        creator.vectorstore_path = str(tmp_path / "vs")
        creator.embeddings = MagicMock()
        fake_vs = MagicMock()
        with patch(
            "agents.vir_chatbot.vectorstore.FAISS.load_local",
            return_value=fake_vs,
        ) as mock_load:
            creator._load_faiss_vectorstore()
        assert creator.vectorstore is fake_vs
        mock_load.assert_called_once()

    def test_processing_faiss_builds_parallel_lists(self, creator):
        creator.filtered_df = pd.DataFrame(
            [
                {
                    "id": "u1",
                    "summary": "s1",
                    "metadata": {"filename": "a.pdf"},
                },
                {
                    "id": "u2",
                    "summary": "s2",
                    "metadata": json.dumps({"filename": "b.pdf"}),
                },
            ]
        )
        creator._processing_faiss_vectorstore_data()
        assert creator.texts_for_vectorstore == ["s1", "s2"]
        assert creator.ids_for_vectorstore == ["u1", "u2"]
        assert creator.metadatas_for_vectorstore[0]["filename"] == "a.pdf"
        assert creator.metadatas_for_vectorstore[1]["filename"] == "b.pdf"

    def test_adding_chunks_invokes_add_and_save(self, creator, tmp_path):
        creator.vectorstore = MagicMock()
        creator.texts_for_vectorstore = ["s"]
        creator.metadatas_for_vectorstore = [{"filename": "a"}]
        creator.ids_for_vectorstore = ["id"]
        creator.vectorstore_path = str(tmp_path / "vs")
        creator._adding_chunks_to_vectorstore()
        creator.vectorstore.add_texts.assert_called_once()
        creator.vectorstore.save_local.assert_called_once()

    def test_save_faiss_builds_from_texts(self, creator, tmp_path):
        creator.texts_for_vectorstore = ["s"]
        creator.metadatas_for_vectorstore = [{"filename": "a"}]
        creator.ids_for_vectorstore = ["id"]
        creator.vectorstore_path = str(tmp_path / "vs")
        creator.embeddings = MagicMock()
        fake_vs = MagicMock()
        with patch(
            "agents.vir_chatbot.vectorstore.FAISS.from_texts",
            return_value=fake_vs,
        ) as mock_from_texts:
            creator._save_faiss_vectorstore()
        mock_from_texts.assert_called_once()
        fake_vs.save_local.assert_called_once()


class TestAddFromFolder:
    def test_builds_new_vectorstore_when_nothing_exists(
        self, creator, tmp_path
    ):
        creator.vectorstore_path = str(tmp_path / "vs-new")
        creator.pdfs_to_add = ["/fake/x.pdf"]
        creator._check_chache = MagicMock(return_value=False)
        creator._start_chunking_process = MagicMock()
        creator._processing_faiss_vectorstore_data = MagicMock()
        creator._check_vectorstore_exists = MagicMock(return_value=False)
        creator._save_faiss_vectorstore = MagicMock()
        creator._load_faiss_vectorstore = MagicMock()
        creator._adding_chunks_to_vectorstore = MagicMock()
        creator._save_cache = MagicMock()

        creator.add_from_folder()
        creator._save_faiss_vectorstore.assert_called_once()
        creator._load_faiss_vectorstore.assert_not_called()

    def test_adds_to_existing_vectorstore(self, creator):
        creator.pdfs_to_add = ["/fake/x.pdf"]
        creator._check_chache = MagicMock(return_value=True)
        creator._load_cache = MagicMock()
        creator._diff_vs_cache = MagicMock()
        creator._start_chunking_process = MagicMock()
        creator._processing_faiss_vectorstore_data = MagicMock()
        creator._check_vectorstore_exists = MagicMock(return_value=True)
        creator._load_faiss_vectorstore = MagicMock()
        creator._adding_chunks_to_vectorstore = MagicMock()
        creator._save_faiss_vectorstore = MagicMock()
        creator._save_cache = MagicMock()

        creator.add_from_folder()
        creator._load_faiss_vectorstore.assert_called_once()
        creator._adding_chunks_to_vectorstore.assert_called_once()
        creator._save_faiss_vectorstore.assert_not_called()


class TestDeletePdfs:
    def test_raises_when_no_cache(self, creator):
        creator._check_chache = MagicMock(return_value=False)
        with pytest.raises(NoCacheFoundError):
            creator.delete_pdfs()

    def test_raises_when_no_vectorstore(self, creator):
        creator._check_chache = MagicMock(return_value=True)
        creator._load_cache = MagicMock()
        creator.recover_deleted_pdfs_from_cache = MagicMock()
        creator._check_vectorstore_exists = MagicMock(return_value=False)
        with pytest.raises(NoVectorStoreFoundError):
            creator.delete_pdfs()

    def test_happy_path(self, creator):
        creator._check_chache = MagicMock(return_value=True)
        creator._load_cache = MagicMock()
        creator.recover_deleted_pdfs_from_cache = MagicMock()
        creator._check_vectorstore_exists = MagicMock(return_value=True)
        creator._load_faiss_vectorstore = MagicMock()
        creator.delete_uuids_from_vectorstore = MagicMock()
        creator._reformat_chache_after_deletion = MagicMock()

        creator.delete_pdfs()
        creator.delete_uuids_from_vectorstore.assert_called_once()
        creator._reformat_chache_after_deletion.assert_called_once()
