"""Tests for backend/routers/tasks.py (/tasks router)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.tasks import router

PATCH_APP = "backend.routers.tasks.celery_app"


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _make_result(
    status, ready=False, successful=None, info=None, result=None
):
    r = MagicMock()
    r.status = status
    r.ready.return_value = ready
    r.successful.return_value = successful
    r.info = info or {}
    r.result = result
    return r


class TestGetTaskStatus:
    def test_returns_200_for_pending_task(self, client):
        task_result = _make_result("PENDING", ready=False)
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.return_value = task_result
            resp = client.get("/tasks/fake-task-id/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "fake-task-id"
        assert data["status"] == "PENDING"
        assert data["step"] == "Waiting"

    def test_progress_status_returns_progress_fields(self, client):
        task_result = _make_result(
            "PROGRESS",
            ready=False,
            info={
                "current": 2,
                "total": 5,
                "percent": 40,
                "step": "Chunking",
                "details": "processing",
            },
        )
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.return_value = task_result
            data = client.get("/tasks/t/status").json()
        assert data["percent"] == 40
        assert data["step"] == "Chunking"

    def test_success_status_returns_result_and_100_percent(self, client):
        task_result = _make_result(
            "SUCCESS", ready=True, successful=True, result={"message": "done"}
        )
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.return_value = task_result
            data = client.get("/tasks/t/status").json()
        assert data["percent"] == 100
        assert data["result"] == {"message": "done"}

    def test_failure_status_returns_error(self, client):
        task_result = _make_result(
            "FAILURE",
            ready=True,
            successful=False,
            result=Exception("something went wrong"),
        )
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.return_value = task_result
            data = client.get("/tasks/t/status").json()
        assert data["percent"] == 0
        assert "something went wrong" in data["error"]

    def test_failure_with_none_result_returns_unknown(self, client):
        task_result = _make_result("FAILURE", ready=True, result=None)
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.return_value = task_result
            data = client.get("/tasks/t/status").json()
        assert data["error"] == "Unknown error"

    def test_unknown_status_has_no_extra_fields(self, client):
        task_result = _make_result("STARTED", ready=False)
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.return_value = task_result
            data = client.get("/tasks/t/status").json()
        assert data["status"] == "STARTED"
        assert "percent" not in data

    def test_returns_500_on_exception(self, client):
        with patch(PATCH_APP) as mock_app:
            mock_app.AsyncResult.side_effect = RuntimeError("broker down")
            resp = client.get("/tasks/bad/status")
        assert resp.status_code == 500


class TestCancelTask:
    def test_returns_cancelled_status(self, client):
        with patch(PATCH_APP) as mock_app:
            resp = client.delete("/tasks/task-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"
        assert data["task_id"] == "task-123"
        mock_app.control.revoke.assert_called_once_with(
            "task-123", terminate=True
        )

    def test_returns_500_when_revoke_raises(self, client):
        with patch(PATCH_APP) as mock_app:
            mock_app.control.revoke.side_effect = RuntimeError("broker down")
            resp = client.delete("/tasks/t")
        assert resp.status_code == 500
