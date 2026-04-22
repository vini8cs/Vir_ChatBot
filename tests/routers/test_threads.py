"""Tests for backend/routers/threads.py (/threads router).

Uses an in-process aiosqlite database to avoid filesystem dependencies
while still exercising the real SQL paths.
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers.threads import router

app = FastAPI()
app.include_router(router)


@pytest.fixture()
def client():
    return TestClient(app)


class TestCreateThread:
    def test_returns_thread_id_and_user_id(self, client):
        resp = client.post("/threads/create", json={"user_id": "user-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user-1"
        assert uuid.UUID(data["thread_id"])

    def test_each_call_returns_unique_thread_id(self, client):
        r1 = client.post("/threads/create", json={"user_id": "u"}).json()
        r2 = client.post("/threads/create", json={"user_id": "u"}).json()
        assert r1["thread_id"] != r2["thread_id"]


class TestGetUserThreads:
    def test_returns_empty_list_when_db_not_found(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        monkeypatch.setattr(
            _, "SQLITE_MEMORY_DATABASE", str(tmp_path / "missing.sqlite")
        )
        resp = client.get("/threads/user-1")
        assert resp.status_code == 200
        assert resp.json() == {"threads": []}

    def test_returns_empty_list_when_no_checkpoints_table(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db = tmp_path / "mem.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", str(db))
        resp = client.get("/threads/user-1")
        assert resp.status_code == 200
        assert resp.json()["threads"] == []

    def test_returns_threads_for_user(self, client, tmp_path, monkeypatch):
        import asyncio
        import json

        import aiosqlite

        import config as _

        db_path = str(tmp_path / "mem.sqlite")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE checkpoints (thread_id TEXT, metadata TEXT)"
                )
                meta = json.dumps({"user_id": "user-1"})
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?,?)",
                    ("thread-abc", meta),
                )
                await conn.commit()

        asyncio.run(_setup())
        resp = client.get("/threads/user-1")
        assert resp.status_code == 200
        threads = resp.json()["threads"]
        assert any(t["thread_id"] == "thread-abc" for t in threads)

    def test_returns_500_on_unexpected_error(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db = tmp_path / "mem.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", str(db))
        with patch(
            "backend.routers.threads.aiosqlite.connect",
            side_effect=Exception("unexpected db error"),
        ):
            resp = client.get("/threads/user-1")
        assert resp.status_code == 500


class TestRenameThread:
    def test_returns_404_when_db_missing(self, client, tmp_path, monkeypatch):
        import config as _

        monkeypatch.setattr(
            _, "SQLITE_MEMORY_DATABASE", str(tmp_path / "missing.sqlite")
        )
        resp = client.patch("/threads/t1/name", json={"name": "My Thread"})
        assert resp.status_code == 404

    def test_renames_thread_in_db(self, client, tmp_path, monkeypatch):
        import asyncio

        import aiosqlite

        import config as _

        db_path = str(tmp_path / "mem.sqlite")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS thread_names "
                    "(thread_id TEXT PRIMARY KEY, name TEXT NOT NULL)"
                )
                await conn.commit()

        asyncio.run(_setup())
        resp = client.patch("/threads/t1/name", json={"name": "Cool Thread"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Cool Thread"

    def test_returns_500_on_db_error(self, client, tmp_path, monkeypatch):
        import config as _

        db = tmp_path / "mem.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", str(db))
        with patch(
            "backend.routers.threads.aiosqlite.connect",
            side_effect=Exception("write error"),
        ):
            resp = client.patch("/threads/t1/name", json={"name": "X"})
        assert resp.status_code == 500


class TestDeleteThread:
    def _db_with_checkpoint(self, tmp_path, thread_id, user_id):
        import asyncio
        import json

        import aiosqlite

        db_path = str(tmp_path / "mem.sqlite")

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE checkpoints (thread_id TEXT, metadata TEXT)"
                )
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS writes (thread_id TEXT)"
                )
                meta = json.dumps({"user_id": user_id})
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?,?)",
                    (thread_id, meta),
                )
                await conn.commit()

        asyncio.run(_setup())
        return db_path

    def test_returns_404_when_db_missing(self, client, tmp_path, monkeypatch):
        import config as _

        monkeypatch.setattr(
            _, "SQLITE_MEMORY_DATABASE", str(tmp_path / "missing.sqlite")
        )
        resp = client.delete("/threads/user-1/thread-1")
        assert resp.status_code == 404

    def test_returns_404_when_thread_not_found(
        self, client, tmp_path, monkeypatch
    ):
        import asyncio

        import aiosqlite

        import config as _

        db_path = str(tmp_path / "mem.sqlite")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE checkpoints (thread_id TEXT, metadata TEXT)"
                )
                await conn.commit()

        asyncio.run(_setup())
        resp = client.delete("/threads/user-1/ghost-thread")
        assert resp.status_code == 404

    def test_deletes_existing_thread_successfully(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db_path = self._db_with_checkpoint(tmp_path, "t-del", "owner")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)
        resp = client.delete("/threads/owner/t-del")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]

    def test_returns_404_when_no_checkpoints_table(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db = tmp_path / "empty.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", str(db))
        resp = client.delete("/threads/u/t")
        assert resp.status_code == 404

    def test_returns_500_on_unexpected_error(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db = tmp_path / "mem.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", str(db))
        with patch(
            "backend.routers.threads.aiosqlite.connect",
            side_effect=Exception("unexpected"),
        ):
            resp = client.delete("/threads/u/t")
        assert resp.status_code == 500


class TestGetThreadMessages:
    def _db_with_messages(self, tmp_path, thread_id, messages):
        """Build a real SQLite db with a serialized LangGraph checkpoint."""
        import asyncio

        import aiosqlite
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        db_path = str(tmp_path / "mem.sqlite")

        msg_objects = []
        for m in messages:
            if m["role"] == "user":
                msg_objects.append(HumanMessage(content=m["content"]))
            else:
                msg_objects.append(AIMessage(content=m["content"]))

        serde = JsonPlusSerializer()
        checkpoint = {
            "channel_values": {"messages": msg_objects},
            "channel_versions": {},
            "versions_seen": {},
        }
        cp_type, cp_data = serde.dumps_typed(checkpoint)

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE checkpoints "
                    "(thread_id TEXT, checkpoint BLOB, type TEXT, "
                    "checkpoint_id TEXT)"
                )
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?,?,?,?)",
                    (thread_id, cp_data, cp_type, "1"),
                )
                await conn.commit()

        asyncio.run(_setup())
        return db_path

    def test_returns_empty_when_checkpoint_not_bytes(
        self, client, tmp_path, monkeypatch
    ):
        import asyncio

        import aiosqlite

        import config as _

        db_path = str(tmp_path / "mem.sqlite")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE checkpoints "
                    "(thread_id TEXT, checkpoint TEXT, type TEXT, "
                    "checkpoint_id TEXT)"
                )
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?,?,?,?)",
                    ("t1", "not-bytes-string", "json", "1"),
                )
                await conn.commit()

        asyncio.run(_setup())
        resp = client.get("/threads/t1/messages")
        assert resp.json() == {"messages": []}

    def test_returns_messages_from_real_checkpoint(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db_path = self._db_with_messages(
            tmp_path,
            "t1",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        )
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)
        resp = client.get("/threads/t1/messages")
        msgs = resp.json()["messages"]
        assert any(
            m["role"] == "user" and "hello" in m["content"] for m in msgs
        )
        assert any(
            m["role"] == "assistant" and "world" in m["content"] for m in msgs
        )

    def test_skips_messages_without_type_or_content(
        self, client, tmp_path, monkeypatch
    ):
        """Exercises the hasattr guard for non-standard message objects."""
        import asyncio

        import aiosqlite
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        import config as _

        db_path = str(tmp_path / "mem.sqlite")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", db_path)

        serde = JsonPlusSerializer()
        checkpoint = {
            "channel_values": {"messages": ["bare-string"]},
            "channel_versions": {},
            "versions_seen": {},
        }
        cp_type, cp_data = serde.dumps_typed(checkpoint)

        async def _setup():
            async with aiosqlite.connect(db_path) as conn:
                await conn.execute(
                    "CREATE TABLE checkpoints "
                    "(thread_id TEXT, checkpoint BLOB, type TEXT, "
                    "checkpoint_id TEXT)"
                )
                await conn.execute(
                    "INSERT INTO checkpoints VALUES (?,?,?,?)",
                    ("t1", cp_data, cp_type, "1"),
                )
                await conn.commit()

        asyncio.run(_setup())
        resp = client.get("/threads/t1/messages")
        assert resp.json() == {"messages": []}

    def test_returns_empty_on_no_table_error(
        self, client, tmp_path, monkeypatch
    ):
        import config as _

        db = tmp_path / "empty.sqlite"
        db.write_bytes(b"")
        monkeypatch.setattr(_, "SQLITE_MEMORY_DATABASE", str(db))
        resp = client.get("/threads/t1/messages")
        assert resp.json() == {"messages": []}
