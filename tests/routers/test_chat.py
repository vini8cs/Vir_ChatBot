"""Tests for backend/routers/chat.py (/chat/stream endpoint)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.state as state
from backend.models import RuntimeConfig
from backend.routers.chat import _create_graph_retriever, router

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


def _collect_sse(response):
    """Parse SSE lines from streaming response text."""
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(line[len("data: ") :])
    return events


# ---------------------------------------------------------------------------
# helpers to build fake async-generator streams
# ---------------------------------------------------------------------------


def _stream_events(*events):
    """Return an astream_events stub that yields the given events."""

    async def _gen(*_, **__):
        for ev in events:
            yield ev

    return _gen


def _stream_then_raise(exc, *, preamble=None):
    """Yield an optional non-matching preamble event, then raise ``exc``."""

    async def _gen(*_, **__):
        if preamble:
            yield preamble
        raise exc

    return _gen


# ---------------------------------------------------------------------------
# TestChatStream
# ---------------------------------------------------------------------------


class TestChatStream:
    def test_returns_error_when_no_vectorstore(self, client):
        state.global_resources = {}
        resp = client.post(
            "/chat/stream",
            json={"message": "hi", "thread_id": "t1"},
        )
        assert resp.status_code == 200
        assert any("VectorStore" in e for e in _collect_sse(resp))

    def test_returns_error_when_graph_creation_fails(self, client):
        state.global_resources["retriever"] = MagicMock()
        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=None),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )
        assert any("error" in e for e in _collect_sse(resp))

    def test_streams_ai_response_chunks(self, client):
        state.global_resources["retriever"] = MagicMock()

        chunk = MagicMock()
        chunk.content = "Hello!"
        chunk.tool_call_chunks = []

        graph = MagicMock()
        graph.astream_events = _stream_events(
            {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
        )
        cm = MagicMock()
        cm.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=(graph, cm)),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )

        events = _collect_sse(resp)
        ai_events = [json.loads(e) for e in events if e != "[DONE]"]
        assert any(e.get("type") == "ai_response" for e in ai_events)
        assert any(e.get("content") == "Hello!" for e in ai_events)
        assert "[DONE]" in events

    def test_filters_out_tool_call_chunks(self, client):
        state.global_resources["retriever"] = MagicMock()

        chunk = MagicMock()
        chunk.content = "ignored"
        chunk.tool_call_chunks = [MagicMock()]  # non-empty → skip

        graph = MagicMock()
        graph.astream_events = _stream_events(
            {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
        )
        cm = MagicMock()
        cm.__aexit__ = AsyncMock()

        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=(graph, cm)),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )
        events = _collect_sse(resp)
        ai_events = [
            json.loads(e)
            for e in events
            if e != "[DONE]" and "error" not in e
        ]
        assert not any(e.get("content") == "ignored" for e in ai_events)

    def test_handles_list_content_with_thinking_filtered(self, client):
        state.global_resources["retriever"] = MagicMock()

        chunk = MagicMock()
        chunk.tool_call_chunks = []
        chunk.content = [
            {"type": "thinking", "text": "internal thought"},
            {"type": "text", "text": "visible text"},
        ]

        graph = MagicMock()
        graph.astream_events = _stream_events(
            {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
        )
        cm = MagicMock()
        cm.__aexit__ = AsyncMock()

        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=(graph, cm)),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )
        events = _collect_sse(resp)
        ai_events = [json.loads(e) for e in events if e != "[DONE]"]
        contents = [e.get("content", "") for e in ai_events]
        assert any("visible text" in c for c in contents)
        assert all("internal thought" not in c for c in contents)

    def test_skips_non_chat_model_stream_events(self, client):
        state.global_resources["retriever"] = MagicMock()

        graph = MagicMock()
        graph.astream_events = _stream_events(
            {"event": "on_tool_start", "data": {}},
            {"event": "on_chain_end", "data": {}},
        )
        cm = MagicMock()
        cm.__aexit__ = AsyncMock()

        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=(graph, cm)),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )
        assert _collect_sse(resp) == ["[DONE]"]

    def test_returns_error_event_on_stream_exception(self, client):
        state.global_resources["retriever"] = MagicMock()

        graph = MagicMock()
        graph.astream_events = _stream_then_raise(
            RuntimeError("unexpected error"),
            preamble={"event": "on_tool_start", "data": {}},
        )
        cm = MagicMock()
        cm.__aexit__ = AsyncMock()

        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=(graph, cm)),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )
        assert any("error" in e for e in _collect_sse(resp))

    def test_db_locked_error_uses_friendly_message(self, client):
        state.global_resources["retriever"] = MagicMock()

        graph = MagicMock()
        graph.astream_events = _stream_then_raise(
            RuntimeError("database is locked"),
            preamble={"event": "on_tool_start", "data": {}},
        )
        cm = MagicMock()
        cm.__aexit__ = AsyncMock()

        with patch(
            "backend.routers.chat._create_graph_retriever",
            new=AsyncMock(return_value=(graph, cm)),
        ):
            resp = client.post(
                "/chat/stream",
                json={"message": "hi", "thread_id": "t1"},
            )
        error_events = [
            json.loads(e)
            for e in _collect_sse(resp)
            if e != "[DONE]" and "error" in e
        ]
        assert any("busy" in e.get("error", "").lower() for e in error_events)


# ---------------------------------------------------------------------------
# TestCreateGraphRetriever
# ---------------------------------------------------------------------------


class TestCreateGraphRetriever:
    @pytest.mark.asyncio
    async def test_returns_result_on_first_attempt(self):
        fake_result = (MagicMock(), MagicMock())
        with patch(
            "backend.routers.chat.create_graph",
            new=AsyncMock(return_value=fake_result),
        ):
            result = await _create_graph_retriever(
                MagicMock(), RuntimeConfig()
            )
        assert result is fake_result

    @pytest.mark.asyncio
    async def test_retries_on_db_locked_then_succeeds(self):
        fake_result = (MagicMock(), MagicMock())
        call_count = 0

        async def _flaky(*_, **__):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("database is locked")
            return fake_result

        with (
            patch("backend.routers.chat.create_graph", side_effect=_flaky),
            patch("backend.routers.chat.asyncio.sleep", new=AsyncMock()),
        ):
            result = await _create_graph_retriever(
                MagicMock(), RuntimeConfig(), max_retries=3
            )
        assert result is fake_result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_returns_none_after_all_retries_exhausted(self):
        async def _always_locked(*_, **__):
            raise Exception("database is locked")

        with (
            patch(
                "backend.routers.chat.create_graph",
                side_effect=_always_locked,
            ),
            patch("backend.routers.chat.asyncio.sleep", new=AsyncMock()),
        ):
            result = await _create_graph_retriever(
                MagicMock(), RuntimeConfig(), max_retries=2
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_max_retries_is_zero(self):
        """Empty retry range never enters the loop; returns None directly."""
        result = await _create_graph_retriever(
            MagicMock(), RuntimeConfig(), max_retries=0
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_non_db_error_returns_none_immediately(self):
        async def _boom(*_, **__):
            raise RuntimeError("other error")

        with patch("backend.routers.chat.create_graph", side_effect=_boom):
            result = await _create_graph_retriever(
                MagicMock(), RuntimeConfig(), max_retries=3
            )
        assert result is None
