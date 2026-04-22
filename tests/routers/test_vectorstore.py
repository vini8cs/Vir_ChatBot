"""Tests for backend/routers/vectorstore.py."""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.state as state
from backend.models import RuntimeConfig
from backend.routers.vectorstore import _get_filenames_from_metadata, router

app = FastAPI()
app.include_router(router)


@pytest.fixture(autouse=True)
def reset_state():
    state.global_resources = {}
    state.runtime_config = RuntimeConfig()
    yield
    state.global_resources = {}


@pytest.fixture()
def client():
    return TestClient(app)


class TestGetFilenamesFromMetadata:
    def test_returns_filename_from_valid_json(self):
        meta = json.dumps({"filename": "paper.pdf"})
        assert _get_filenames_from_metadata(meta) == "paper.pdf"

    def test_returns_none_on_invalid_json(self):
        assert _get_filenames_from_metadata("not-json") is None

    def test_returns_none_when_filename_key_missing(self):
        meta = json.dumps({"other": "value"})
        assert _get_filenames_from_metadata(meta) is None


class TestReloadVectorstore:
    def test_returns_success_when_retriever_loaded(self, client):
        fake_retriever = MagicMock()
        with patch(
            "backend.routers.vectorstore.load_global_vectorstore",
            new=AsyncMock(return_value=fake_retriever),
        ):
            resp = client.post("/vectorstore/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert state.global_resources["retriever"] is fake_retriever

    def test_returns_warning_when_no_vectorstore(self, client):
        with patch(
            "backend.routers.vectorstore.load_global_vectorstore",
            new=AsyncMock(return_value=None),
        ):
            resp = client.post("/vectorstore/reload")
        assert resp.json()["status"] == "warning"

    def test_returns_error_on_exception(self, client):
        with patch(
            "backend.routers.vectorstore.load_global_vectorstore",
            new=AsyncMock(side_effect=RuntimeError("faiss error")),
        ):
            resp = client.post("/vectorstore/reload")
        assert resp.json()["status"] == "error"


class TestUploadPdf:
    def _upload(self, client, filename, content=b"data"):
        return client.post(
            "/create-vectorstore-based-on-selected-pdfs/",
            files={
                "files": (filename, io.BytesIO(content), "application/pdf")
            },
        )

    def test_rejects_unsupported_extension(self, client):
        resp = self._upload(client, "file.exe")
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_accepts_pdf(self, client):
        fake_task = MagicMock()
        fake_task.id = "task-123"
        with patch(
            "backend.routers.vectorstore"
            ".create_vectorstore_uploaded_pdfs.delay",
            return_value=fake_task,
        ):
            resp = self._upload(client, "paper.pdf")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "task-123"

    def test_accepts_txt(self, client):
        fake_task = MagicMock()
        fake_task.id = "t"
        with patch(
            "backend.routers.vectorstore"
            ".create_vectorstore_uploaded_pdfs.delay",
            return_value=fake_task,
        ):
            resp = self._upload(client, "data.txt", b"text")
        assert resp.status_code == 200

    def test_accepts_tsv(self, client):
        fake_task = MagicMock()
        fake_task.id = "t"
        with patch(
            "backend.routers.vectorstore"
            ".create_vectorstore_uploaded_pdfs.delay",
            return_value=fake_task,
        ):
            resp = self._upload(client, "table.tsv", b"col\tval\na\tb")
        assert resp.status_code == 200

    def test_returns_500_on_write_error(self, client):
        with patch(
            "backend.routers.vectorstore.aiofiles.open",
            side_effect=OSError("disk full"),
        ):
            resp = self._upload(client, "paper.pdf")
        assert resp.status_code == 500


class TestDeletePdfs:
    def test_rejects_empty_filenames_list(self, client):
        resp = client.post("/delete-selected-pdfs/", json={"filenames": []})
        assert resp.status_code == 400

    def test_returns_task_id_for_valid_request(self, client):
        fake_task = MagicMock()
        fake_task.id = "del-task"
        with patch(
            "backend.routers.vectorstore.delete_pdfs_from_vectorstore.delay",
            return_value=fake_task,
        ):
            resp = client.post(
                "/delete-selected-pdfs/",
                json={"filenames": ["old.pdf"]},
            )
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "del-task"


class TestListPdfs:
    def test_returns_empty_when_no_cache(self, client, tmp_path, monkeypatch):
        import config as _

        monkeypatch.setattr(_, "CACHE_FOLDER", str(tmp_path / "no-cache"))
        resp = client.get("/pdfs/list")
        assert resp.json() == {"pdfs": []}

    def test_returns_sorted_filenames(self, client, tmp_path, monkeypatch):
        import config as _

        monkeypatch.setattr(_, "CACHE_FOLDER", str(tmp_path))
        cache = tmp_path / "cache.csv"
        df = pd.DataFrame(
            {
                "metadata": [
                    json.dumps({"filename": "z.pdf"}),
                    json.dumps({"filename": "a.pdf"}),
                    json.dumps({"filename": "a.pdf"}),
                ]
            }
        )
        df.to_csv(cache, index=False)
        resp = client.get("/pdfs/list")
        assert resp.json() == {"pdfs": ["a.pdf", "z.pdf"]}

    def test_returns_empty_when_no_metadata_column(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        monkeypatch.setattr(_, "CACHE_FOLDER", str(tmp_path))
        cache = tmp_path / "cache.csv"
        pd.DataFrame({"id": ["1"]}).to_csv(cache, index=False)
        resp = client.get("/pdfs/list")
        assert resp.json() == {"pdfs": []}

    def test_returns_empty_on_read_error(self, client, tmp_path, monkeypatch):
        import config as _

        monkeypatch.setattr(_, "CACHE_FOLDER", str(tmp_path))
        cache = tmp_path / "cache.csv"
        cache.write_text("bad,csv,data\n[[[")
        with patch(
            "backend.routers.vectorstore.pd.read_csv",
            side_effect=Exception("parse error"),
        ):
            resp = client.get("/pdfs/list")
        assert resp.json() == {"pdfs": []}
