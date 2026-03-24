import logging
import os
import uuid

import aiosqlite
from fastapi import APIRouter, HTTPException
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

import config as _
from backend.models import (
    CreateThreadRequest,
    RenameThreadRequest,
    ThreadResponse,
)

router = APIRouter(prefix="/threads", tags=["threads"])


async def _ensure_thread_names_table(conn):
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
    )
    await conn.commit()


@router.post("/create", response_model=ThreadResponse)
async def create_thread(request: CreateThreadRequest):
    """
    Create a new thread for a user.
    Returns a unique thread_id that can be used in chat requests.
    """
    thread_id = str(uuid.uuid4())
    return ThreadResponse(thread_id=thread_id, user_id=request.user_id)


@router.get("/{user_id}")
async def get_user_threads(user_id: str):
    """
    Get all threads associated with a specific user.
    """
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        return {"threads": []}

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            await _ensure_thread_names_table(conn)
            query = """
            SELECT c.thread_id, tn.name
            FROM (
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            ) c
            LEFT JOIN thread_names tn ON c.thread_id = tn.thread_id
            """  # noqa: W291
            async with conn.execute(query, (user_id,)) as cur:
                rows = await cur.fetchall()

            return {
                "threads": [{"thread_id": r[0], "name": r[1]} for r in rows]
            }

    except Exception as e:
        if "no such table" in str(e).lower():
            return {"threads": []}
        logging.error(f"Database error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        ) from e


@router.patch("/{thread_id}/name")
async def rename_thread(thread_id: str, request: RenameThreadRequest):
    """Set or update the display name for a thread."""
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            await _ensure_thread_names_table(conn)
            await conn.execute(
                """
                INSERT INTO thread_names (thread_id, name)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET name = excluded.name
                """,
                (thread_id, request.name),
            )
            await conn.commit()
        return {"thread_id": thread_id, "name": request.name}
    except Exception as e:
        logging.error(f"Error renaming thread {thread_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        ) from e


@router.delete("/{user_id}/{thread_id}")
async def delete_thread(user_id: str, thread_id: str):
    """Delete a specific thread and its associated messages."""
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        raise HTTPException(status_code=404, detail="Database not found")

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            query = """
            SELECT 1 FROM checkpoints
            WHERE thread_id = ?
            AND json_extract(CAST(metadata AS TEXT), '$.user_id') = ?
            LIMIT 1
            """  # noqa W291
            async with conn.execute(query, (thread_id, user_id)) as cur:
                row = await cur.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail="Thread not found or access denied",
                )

            await conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
            )
            await conn.execute(
                "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
            )
            await conn.commit()
            return {"message": f"Thread {thread_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        if "no such table" in str(e).lower():
            raise HTTPException(
                status_code=404, detail="Thread not found or access denied"
            ) from e
        logging.error(f"CRITICAL DB ERROR: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error"
        ) from e


@router.get("/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    """
    Get the message history for a specific thread.
    Extracts messages from the LangGraph checkpointer.
    """
    if not os.path.exists(_.SQLITE_MEMORY_DATABASE):
        return {"messages": []}

    try:
        async with aiosqlite.connect(_.SQLITE_MEMORY_DATABASE) as conn:
            query = """
            SELECT checkpoint, type
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
            """  # noqa W291
            async with conn.execute(query, (thread_id,)) as cur:
                row = await cur.fetchone()

            if not row:
                return {"messages": []}

            checkpoint_data, checkpoint_type = row[0], row[1]

            if not isinstance(checkpoint_data, bytes):
                return {"messages": []}

            serde = JsonPlusSerializer()
            checkpoint = serde.loads_typed((checkpoint_type, checkpoint_data))

            messages = []
            channel_values = checkpoint.get("channel_values", {})
            msg_list = channel_values.get("messages", [])

            for msg in msg_list:
                if not (hasattr(msg, "type") and hasattr(msg, "content")):
                    continue
                msg_type = msg.type
                msg_content = msg.content
                if isinstance(msg_content, list):
                    continue
                if msg_type in ["human", "ai"] and msg_content:
                    role = "user" if msg_type == "human" else "assistant"
                    messages.append({"role": role, "content": msg_content})

            return {"messages": messages}

    except Exception as e:
        if "no such table" in str(e).lower():
            return {"messages": []}
        logging.error(f"Error loading messages for thread {thread_id}: {e}")
        return {"messages": []}
