"""Tests for Pydantic request/response models (backend/models.py)."""

import logging

import pytest
from pydantic import ValidationError

from backend.models import (
    ChatRequest,
    ConfigUpdateRequest,
    CreateThreadRequest,
    DeleteFileRequest,
    HealthCheckFilter,
    RenameThreadRequest,
    RuntimeConfig,
    ThreadResponse,
)


class TestRuntimeConfig:
    def test_default_values_are_populated(self):
        cfg = RuntimeConfig()
        assert cfg.temperature >= 0
        assert cfg.max_retries > 0
        assert cfg.retriever_limit > 0
        assert isinstance(cfg.system_prompt, str)
        assert isinstance(cfg.gemini_model, str)

    def test_accepts_valid_overrides(self):
        cfg = RuntimeConfig(temperature=0.9, max_retries=5)
        assert cfg.temperature == 0.9
        assert cfg.max_retries == 5

    def test_model_dump_roundtrip(self):
        cfg = RuntimeConfig(temperature=0.5, summarize=True)
        restored = RuntimeConfig(**cfg.model_dump())
        assert restored == cfg

    def test_summarize_default_is_bool(self):
        assert isinstance(RuntimeConfig().summarize, bool)


class TestConfigUpdateRequest:
    def test_all_fields_are_optional(self):
        req = ConfigUpdateRequest()
        assert req.temperature is None
        assert req.gemini_model is None
        assert req.summarize is None

    def test_exclude_none_returns_only_provided_fields(self):
        req = ConfigUpdateRequest(temperature=0.7)
        data = req.model_dump(exclude_none=True)
        assert data == {"temperature": 0.7}

    def test_multiple_fields(self):
        req = ConfigUpdateRequest(temperature=0.3, max_retries=2)
        data = req.model_dump(exclude_none=True)
        assert data["temperature"] == 0.3
        assert data["max_retries"] == 2


class TestChatRequest:
    def test_requires_message_and_thread_id(self):
        req = ChatRequest(message="hello", thread_id="abc-123")
        assert req.message == "hello"
        assert req.thread_id == "abc-123"

    def test_user_id_defaults_to_default_user(self):
        req = ChatRequest(message="hi", thread_id="t-1")
        assert req.user_id == "default_user"

    def test_custom_user_id(self):
        req = ChatRequest(message="hi", thread_id="t-1", user_id="alice")
        assert req.user_id == "alice"

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(thread_id="t-1")

    def test_missing_thread_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="hello")


class TestDeleteFileRequest:
    def test_valid_list(self):
        req = DeleteFileRequest(filenames=["a.pdf", "b.pdf"])
        assert len(req.filenames) == 2

    def test_empty_list_is_valid(self):
        req = DeleteFileRequest(filenames=[])
        assert req.filenames == []

    def test_missing_filenames_raises(self):
        with pytest.raises(ValidationError):
            DeleteFileRequest()


class TestThreadModels:
    def test_create_thread_request_requires_user_id(self):
        req = CreateThreadRequest(user_id="user-1")
        assert req.user_id == "user-1"

    def test_create_thread_request_missing_user_id_raises(self):
        with pytest.raises(ValidationError):
            CreateThreadRequest()

    def test_rename_thread_request(self):
        req = RenameThreadRequest(name="My Thread")
        assert req.name == "My Thread"

    def test_rename_thread_request_missing_name_raises(self):
        with pytest.raises(ValidationError):
            RenameThreadRequest()

    def test_thread_response(self):
        resp = ThreadResponse(thread_id="t-1", user_id="u-1")
        assert resp.thread_id == "t-1"
        assert resp.user_id == "u-1"


class TestHealthCheckFilter:
    def _make_record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_allows_non_health_log_records(self):
        f = HealthCheckFilter()
        assert f.filter(self._make_record("GET /chat")) is True

    def test_blocks_health_log_records(self):
        f = HealthCheckFilter()
        assert f.filter(self._make_record("GET /health HTTP/1.1")) is False

    def test_allows_unrelated_records(self):
        f = HealthCheckFilter()
        assert f.filter(self._make_record("Worker started")) is True
