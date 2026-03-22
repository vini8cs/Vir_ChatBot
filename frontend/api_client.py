import asyncio
import json
import mimetypes
import os
from typing import Any, Coroutine

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


async def gather_tasks(names: list[str], tasks: list[Coroutine[Any, Any, Any]]) -> dict:
    task_objs = {}
    async with asyncio.TaskGroup() as tg:
        for name, coro in zip(names, tasks, strict=False):
            task_objs[name] = tg.create_task(coro)
    return {name: task.result() for name, task in task_objs.items()}


async def get_user_threads_api(user_id: str) -> list:
    url = f"{API_BASE_URL}/threads/{user_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json().get("threads", [])
        except httpx.HTTPStatusError as e:
            st.error(f"API Error fetching threads: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            st.error(f"Connection Error fetching threads: {e}")
            return []


async def create_thread_api(user_id: str) -> dict:
    url = f"{API_BASE_URL}/threads/create"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json={"user_id": user_id})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"Error creating thread: {e.response.status_code}")
            return {}
        except httpx.RequestError as e:
            st.error(f"Connection error creating thread: {e}")
            return {}


async def delete_thread_api(user_id: str, thread_id: str) -> dict:
    url = f"{API_BASE_URL}/threads/{user_id}/{thread_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"Error deleting thread: {e.response.status_code}")
            return {}
        except httpx.RequestError as e:
            st.error(f"Connection error deleting thread: {e}")
            return {}


async def get_thread_messages_api(thread_id: str) -> list:
    url = f"{API_BASE_URL}/threads/{thread_id}/messages"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json().get("messages", [])
        except httpx.HTTPStatusError as e:
            st.error(f"Error loading messages: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            st.error(f"Connection error loading messages: {e}")
            return []


async def chat_stream_api(message: str, thread_id: str, user_id: str):
    url = f"{API_BASE_URL}/chat/stream"

    try:
        async with (
            httpx.AsyncClient(timeout=120.0) as client,
            client.stream(
                "POST",
                url,
                json={"message": message, "thread_id": thread_id, "user_id": user_id},
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                    if "content" in payload:
                        yield payload["content"]
                    else:
                        yield f"❌ Error: {payload.get('error', 'Unknown error')}"
                except json.JSONDecodeError:
                    continue

    except httpx.TimeoutException:
        yield "❌ Error: Timeout exceeded. The system may be busy."
    except httpx.HTTPStatusError as e:
        yield f"❌ API Error: {e.response.status_code}"
    except httpx.RequestError as e:
        yield f"❌ Connection error: {e}"


async def list_pdfs_api() -> list:
    url = f"{API_BASE_URL}/pdfs/list"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json().get("pdfs", [])
        except httpx.HTTPStatusError as e:
            st.error(f"Error fetching PDFs: {e.response.status_code}")
            return []
        except httpx.RequestError as e:
            st.error(f"Connection error fetching PDFs: {e}")
            return []


async def upload_pdfs_api(files) -> dict:
    url = f"{API_BASE_URL}/create-vectorstore-based-on-selected-pdfs/"

    files_to_upload = [
        (
            "files",
            (
                file.name,
                file.getvalue(),
                mimetypes.guess_type(file.name)[0] or "application/octet-stream",
            ),
        )
        for file in files
    ]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, files=files_to_upload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def delete_pdfs_api(filenames: list) -> dict:
    url = f"{API_BASE_URL}/delete-selected-pdfs/"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json={"filenames": filenames})
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def reload_vectorstore_api() -> dict:
    url = f"{API_BASE_URL}/vectorstore/reload"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def get_config_api() -> dict:
    url = f"{API_BASE_URL}/config"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def update_config_api(config_updates: dict) -> dict:
    url = f"{API_BASE_URL}/config"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, json=config_updates)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def reset_config_api(reload_vectorstore: bool = False) -> dict:
    endpoint = "/config/reset-and-reload" if reload_vectorstore else "/config/reset"
    url = f"{API_BASE_URL}{endpoint}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def get_task_status_api(task_id: str) -> dict:
    url = f"{API_BASE_URL}/tasks/{task_id}/status"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e), "status": "ERROR"}


async def cancel_task_api(task_id: str) -> dict:
    url = f"{API_BASE_URL}/tasks/{task_id}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}
