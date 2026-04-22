"""Tests for the /config router (backend/routers/config.py)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.state as state
from backend.models import RuntimeConfig
from backend.routers.config import router


@pytest.fixture()
def client(monkeypatch):
    """TestClient with an isolated FastAPI app containing only the
    config router. Config saves are patched to avoid filesystem I/O."""
    monkeypatch.setattr(
        "backend.state._save_persisted_config", lambda _cfg: None
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_runtime_config():
    """Restore default runtime_config around every test."""
    state.runtime_config = RuntimeConfig()
    yield
    state.runtime_config = RuntimeConfig()


class TestGetConfig:
    def test_returns_200(self, client):
        assert client.get("/config").status_code == 200

    def test_response_contains_expected_keys(self, client):
        data = client.get("/config").json()
        assert "temperature" in data
        assert "llm_model" in data
        assert "max_retries" in data

    def test_reflects_current_state(self, client):
        state.runtime_config = RuntimeConfig(temperature=0.99)
        data = client.get("/config").json()
        assert data["temperature"] == pytest.approx(0.99)


class TestUpdateConfig:
    def test_updates_single_field(self, client):
        resp = client.put("/config", json={"temperature": 0.5})
        assert resp.status_code == 200
        assert resp.json()["config"]["temperature"] == pytest.approx(0.5)
        assert state.runtime_config.temperature == pytest.approx(0.5)

    def test_response_contains_status_success(self, client):
        resp = client.put("/config", json={"temperature": 0.3})
        assert resp.json()["status"] == "success"

    def test_empty_body_returns_400(self, client):
        assert client.put("/config", json={}).status_code == 400

    def test_partial_update_preserves_other_fields(self, client):
        original_retries = state.runtime_config.max_retries
        client.put("/config", json={"temperature": 0.1})
        assert state.runtime_config.max_retries == original_retries

    def test_multiple_fields_updated_at_once(self, client):
        resp = client.put(
            "/config", json={"temperature": 0.2, "max_retries": 7}
        )
        cfg = resp.json()["config"]
        assert cfg["temperature"] == pytest.approx(0.2)
        assert cfg["max_retries"] == 7


class TestResetConfig:
    def test_returns_200(self, client):
        assert client.post("/config/reset").status_code == 200

    def test_resets_overridden_values_to_defaults(self, client):
        state.runtime_config = RuntimeConfig(temperature=0.99)
        client.post("/config/reset")
        assert state.runtime_config.temperature == pytest.approx(
            RuntimeConfig().temperature
        )

    def test_response_status_is_success(self, client):
        assert client.post("/config/reset").json()["status"] == "success"


class TestResetConfigAndReload:
    @pytest.fixture()
    def client_with_reload(self, monkeypatch):
        monkeypatch.setattr(
            "backend.state._save_persisted_config", lambda _cfg: None
        )
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_returns_200_when_vectorstore_loads(
        self, client_with_reload, monkeypatch
    ):
        from unittest.mock import AsyncMock, MagicMock

        fake_retriever = MagicMock()
        monkeypatch.setattr(
            "backend.routers.config.load_global_vectorstore",
            AsyncMock(return_value=fake_retriever),
        )
        resp = client_with_reload.post("/config/reset-and-reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["vectorstore_loaded"] is True

    def test_returns_partial_when_vectorstore_not_found(
        self, client_with_reload, monkeypatch
    ):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "backend.routers.config.load_global_vectorstore",
            AsyncMock(return_value=None),
        )
        resp = client_with_reload.post("/config/reset-and-reload")
        data = resp.json()
        assert data["status"] == "success"
        assert data["vectorstore_loaded"] is False

    def test_returns_partial_on_reload_exception(
        self, client_with_reload, monkeypatch
    ):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            "backend.routers.config.load_global_vectorstore",
            AsyncMock(side_effect=RuntimeError("faiss error")),
        )
        resp = client_with_reload.post("/config/reset-and-reload")
        data = resp.json()
        assert data["status"] == "partial"
        assert "faiss error" in data["error"]
