"""Tests for backend/api.py (lifespan, health check, WAL setup)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestHealthCheck:
    def test_health_endpoint_returns_healthy(self):
        from backend.api import app

        with (
            patch(
                "backend.api.load_global_vectorstore",
                new=AsyncMock(return_value=None),
            ),
            TestClient(app) as client,
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


class TestLifespan:
    def test_loads_persisted_config_on_startup(self):
        from backend.models import RuntimeConfig

        custom_cfg = RuntimeConfig(temperature=0.42)
        with (
            patch(
                "backend.api.state._load_persisted_config",
                return_value=custom_cfg,
            ),
            patch(
                "backend.api.load_global_vectorstore",
                new=AsyncMock(return_value=None),
            ),
            patch("backend.api._turn_wal_mode_on", new=AsyncMock()),
        ):
            import backend.state as state
            from backend.api import app

            with TestClient(app):
                assert state.runtime_config.temperature == pytest.approx(0.42)

    def test_stores_retriever_when_vectorstore_found(self):
        fake_retriever = MagicMock()
        with (
            patch(
                "backend.api.load_global_vectorstore",
                new=AsyncMock(return_value=fake_retriever),
            ),
            patch("backend.api._turn_wal_mode_on", new=AsyncMock()),
        ):
            import backend.state as state
            from backend.api import app

            with TestClient(app):
                assert state.global_resources["retriever"] is fake_retriever

    def test_global_resources_cleared_on_shutdown(self):
        with (
            patch(
                "backend.api.load_global_vectorstore",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch("backend.api._turn_wal_mode_on", new=AsyncMock()),
        ):
            import backend.state as state
            from backend.api import app

            with TestClient(app):
                pass  # lifespan exit happens here
            assert state.global_resources == {}

    def test_vectorstore_load_error_sets_none(self):
        with (
            patch(
                "backend.api.load_global_vectorstore",
                new=AsyncMock(side_effect=RuntimeError("faiss broken")),
            ),
            patch("backend.api._turn_wal_mode_on", new=AsyncMock()),
        ):
            import backend.state as state
            from backend.api import app

            with TestClient(app):
                assert state.global_resources["retriever"] is None


class TestTurnWalModeOn:
    @pytest.mark.asyncio
    async def test_executes_pragma_commands(self, tmp_path, monkeypatch):
        import config as _

        db = str(tmp_path / "test.sqlite")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db)

        from backend.api import _turn_wal_mode_on

        await _turn_wal_mode_on()
