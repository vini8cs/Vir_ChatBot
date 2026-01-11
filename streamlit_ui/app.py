import asyncio
import json
import os
from datetime import datetime
from typing import Any, Coroutine

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

GEMINI_MODELS = [
    # "gemini-3-pro-preview",
    # "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


st.set_page_config(page_title="Vir ChatBot", page_icon="🧬", layout="wide")


async def get_user_threads_api(user_id: str) -> list:
    """Call the API to get all threads for a user."""
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
    """Call the API to create a new thread."""
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
    """Call the API to delete a thread for a specific user."""
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
    """Call the API to get message history for a thread."""
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
    """
    Call the chat stream API and yield responses asynchronously.
    """
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
                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                data = line[6:]

                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                    if "content" in payload:
                        yield payload["content"]
                    else:
                        error_msg = payload.get("error", "Unknown error")
                        yield f"❌ Error: {error_msg}"

                except json.JSONDecodeError:
                    continue

    except httpx.TimeoutException:
        yield "❌ Error: Timeout exceeded. The system may be busy."

    except httpx.HTTPStatusError as e:
        yield f"❌ API Error: {e.response.status_code}"

    except httpx.RequestError as e:
        yield f"❌ Connection error: {e}"


async def list_pdfs_api() -> list:
    """Get list of PDFs in the vectorstore cache."""
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
    """Upload PDFs to create/update vectorstore."""
    url = f"{API_BASE_URL}/create-vectorstore-based-on-selected-pdfs/"

    files_to_upload = [("files", (file.name, file.getvalue(), "application/pdf")) for file in files]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, files=files_to_upload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def create_vectorstore_from_folder_api() -> dict:
    """Create vectorstore from folder."""
    url = f"{API_BASE_URL}/create-vectorstore-from-folder/"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def delete_pdfs_api(filenames: list) -> dict:
    """Delete PDFs from vectorstore."""
    url = f"{API_BASE_URL}/delete-selected-pdfs/"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json={"filenames": filenames})
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def reload_vectorstore_api() -> dict:
    """Reload the VectorStore into memory after creation/update."""
    url = f"{API_BASE_URL}/vectorstore/reload"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def get_config_api() -> dict:
    """Get current runtime configuration."""
    url = f"{API_BASE_URL}/config"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def update_config_api(config_updates: dict) -> dict:
    """Update runtime configuration."""
    url = f"{API_BASE_URL}/config"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.put(url, json=config_updates)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e)}


async def reset_config_api(reload_vectorstore: bool = False) -> dict:
    """Reset configuration to defaults."""
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
    """Get the status and progress of a background task."""
    url = f"{API_BASE_URL}/tasks/{task_id}/status"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            return {"error": str(e), "status": "ERROR"}


async def gather_tasks(names: list[str], tasks: list[Coroutine[Any, Any, Any]]) -> dict:
    task_objs = {}
    async with asyncio.TaskGroup() as tg:
        for name, coro in zip(names, tasks, strict=False):
            task_objs[name] = tg.create_task(coro)
    return {name: task.result() for name, task in task_objs.items()}


async def start_session_data(user_id: str):
    """Initialize session state data with parallel fetching."""

    if "user_id" not in st.session_state:
        st.session_state.user_id = user_id
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "active_tasks" not in st.session_state:
        st.session_state.active_tasks = {}

    need_threads = "threads_id" not in st.session_state
    need_config = "runtime_config" not in st.session_state

    threads_result = None
    config_result = None

    tasks = []
    names = []

    if need_threads:
        tasks.append(get_user_threads_api(st.session_state.user_id))
        names.append("threads")

    if need_config:
        tasks.append(get_config_api())
        names.append("config")

    mapping = {}
    if tasks:
        mapping = await gather_tasks(names, tasks)

    threads_result = mapping.get("threads")
    config_result = mapping.get("config")

    if need_threads and threads_result is not None:
        st.session_state.threads_id = [t["thread_id"] for t in threads_result]

    if need_config and config_result is not None:
        st.session_state.runtime_config = config_result

    if "selected_thread" not in st.session_state:
        st.session_state.selected_thread = st.session_state.threads_id[-1] if st.session_state.threads_id else None

    if st.session_state.selected_thread and st.session_state.selected_thread not in st.session_state.messages:
        st.session_state.messages[st.session_state.selected_thread] = await get_thread_messages_api(
            st.session_state.selected_thread
        )

    if "active_tasks" not in st.session_state:
        st.session_state.active_tasks = {}

    if "runtime_config" not in st.session_state:
        st.session_state.runtime_config = await get_config_api()


async def create_new_thread(user_id: str):
    """Create a new thread and update session state."""
    thread = await create_thread_api(user_id)
    thread_id = thread["thread_id"]
    st.session_state.threads_id.append(thread_id)
    st.session_state.selected_thread = thread_id
    st.session_state.messages[thread_id] = []
    st.rerun()


async def delete_thread_and_update_state(thread_id: str):
    """Delete a thread and update the session state."""
    await delete_thread_api(st.session_state.user_id, thread_id)
    st.session_state.threads_id.remove(thread_id)

    if thread_id in st.session_state.messages:
        del st.session_state.messages[thread_id]

    if st.session_state.threads_id:
        st.session_state.selected_thread = st.session_state.threads_id[-1]
    else:
        st.session_state.selected_thread = None

    st.rerun()


async def select_thread(thread_id: str):
    """Select a thread and load its messages from the database."""
    st.session_state.selected_thread = thread_id
    if thread_id not in st.session_state.messages:
        st.session_state.messages[thread_id] = await get_thread_messages_api(thread_id)


def get_current_messages() -> list:
    """Get messages for the currently selected thread."""
    thread_id = st.session_state.selected_thread
    if thread_id and thread_id in st.session_state.messages:
        return st.session_state.messages[thread_id]
    return []


def add_message(role: str, content: str):
    """Add a message to the current thread."""
    thread_id = st.session_state.selected_thread
    if thread_id:
        if thread_id not in st.session_state.messages:
            st.session_state.messages[thread_id] = []
        st.session_state.messages[thread_id].append({"role": role, "content": content})


def add_active_task(task_id: str, task_name: str):
    """Add a task to the active tasks tracking."""
    st.session_state.active_tasks[task_id] = {
        "name": task_name,
        "started_at": datetime.now().isoformat(),
    }


def remove_active_task(task_id: str):
    """Remove a task from active tasks tracking."""
    if task_id in st.session_state.active_tasks:
        del st.session_state.active_tasks[task_id]


def create_pending_task():
    st.progress(0, text="⏳ Task queued...")
    return True


def create_progress_task(status):
    percent = status.get("percent", 0)
    step = status.get("step", "")
    details = status.get("details", "")
    current = status.get("current", 0)
    total = status.get("total", 0)

    st.progress(percent / 100, text=f"🔄 {step}: {details} ({current}/{total})")
    return True


async def create_success_task(status, task_name):
    result = status.get("result", {})
    message = result.get("message", "Finished successfully!")
    result_status = result.get("status", "")

    if result_status in ("Failure", "Error"):
        st.error(f"❌ ERROR: {result.get('error', 'Unknown Error')}")
        return
    st.success(f"✅ {message}")
    if "Upload" in task_name or "Create" in task_name:
        reload_result = await reload_vectorstore_api()
        if reload_result.get("status") == "success":
            st.info("🔄 VectorStore reloaded into memory.")


def create_failure_task(status):
    error = status.get("error", "Unknown error")
    st.error(f"❌ FAILURE: {error}")


async def render_task_progress():
    """Render progress bars for all active tasks."""
    if not st.session_state.active_tasks:
        return

    st.subheader("⏳ Tasks running...")

    tasks_to_remove = []
    has_running_tasks = False

    def _handle_pending(status, task_name, task_id):
        return create_pending_task(), False

    def _handle_progress(status, task_name, task_id):
        return create_progress_task(status), False

    async def _handle_success(status, task_name, task_id):
        await create_success_task(status, task_name)
        return False, True

    def _handle_failure(status, task_name, task_id):
        create_failure_task(status)
        return False, True

    def _handle_unknown(status, task_name, task_id):
        st.warning(f"⚠️ Status: {status.get('status', 'UNKNOWN')}")
        return True, False

    STATUS_HANDLERS = {
        "PENDING": _handle_pending,
        "PROGRESS": _handle_progress,
        "SUCCESS": _handle_success,
        "FAILURE": _handle_failure,
    }

    for task_id, task_info in list(st.session_state.active_tasks.items()):
        status = await get_task_status_api(task_id)

        if "error" in status and status.get("status") == "ERROR":
            st.error(f"❌ Error to verify task: {status['error']}")
            tasks_to_remove.append(task_id)
            continue

        task_status = status.get("status", "UNKNOWN")
        task_name = task_info["name"]

        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"**{task_name}**")

                handler = STATUS_HANDLERS.get(task_status, _handle_unknown)
                result = handler(status, task_name, task_id)
                if asyncio.iscoroutine(result):
                    has_running, remove_flag = await result
                else:
                    has_running, remove_flag = result
                has_running_tasks = has_running_tasks or has_running

                if remove_flag:
                    tasks_to_remove.append(task_id)

            with col2:
                if not remove_flag:
                    st.caption(f"ID: {task_id[:8]}...")

        st.divider()

    for task_id in tasks_to_remove:
        remove_active_task(task_id)

    if has_running_tasks:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("💡 Click 'Refresh' to see the latest progress")
        with col2:
            if st.button("🔄 Refresh", key="refresh_tasks", use_container_width=True):
                st.rerun()

        # NOTE: Removed st_autorefresh as it interrupts streaming chat responses
        # Users must manually click "Refresh" to refresh task progress


def model_selection(config=None, key="config_model"):
    current_model = config.get("gemini_model", "gemini-2.5-flash")
    model_index = GEMINI_MODELS.index(current_model) if current_model in GEMINI_MODELS else 0
    return st.selectbox(
        "🧬 Gemini Model",
        options=GEMINI_MODELS,
        index=model_index,
        key=key,
    )


def temperature_slider(config=None, key="config_temperature"):
    return st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(config.get("temperature", 0.1)),
        step=0.1,
        key=key,
        help="Higher = more creative, Lower = more focused",
    )


def max_tokens_input(config=None, key="config_max_tokens"):
    return st.number_input(
        "📝 Max Output Tokens",
        min_value=256,
        max_value=8192,
        value=int(config.get("max_output_tokens", 2048)),
        step=256,
        key=key,
    )


def retriever_limit_input(config=None, key="config_retriever_limit"):
    return st.number_input(
        "🔍 Retriever Limit (k)",
        min_value=1,
        max_value=20,
        value=int(config.get("retriever_limit", 5)),
        step=1,
        key=key,
        help="Number of documents to retrieve",
    )


def max_retries_input(config=None, key="config_max_retries"):
    return st.number_input(
        "🔄 Max Retries",
        min_value=1,
        max_value=10,
        value=int(config.get("max_retries", 3)),
        step=1,
        key=key,
    )


def summarize_toggle(config=None, key="config_summarize"):
    return st.toggle(
        "📋 Summarize Context",
        value=bool(config.get("summarize", False)),
        key=key,
    )


async def run_save_config(config_changed, updates, key="save_config"):
    if st.button(
        "💾 Save",
        use_container_width=True,
        type="primary" if config_changed else "secondary",
        disabled=not config_changed,
        key=key,
    ):
        result = await update_config_api(updates)
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("✅ Configuration saved!")
            st.session_state.runtime_config = result.get("config", await get_config_api())
            st.rerun()


def run_reset_config(key: str = "reset_config", keys_prefix: str = "config"):
    """Render a Reset button. The on-click schedules the async reset."""

    st.button(
        "🔄 Reset Defaults",
        use_container_width=True,
        on_click=make_reset_config_callback(keys_prefix),
        key=key,
    )

    error_key = f"{keys_prefix}_reset_error"
    success_key = f"{keys_prefix}_reset_success"

    if st.session_state.get(error_key):
        st.error(f"Error: {st.session_state[error_key]}")
        del st.session_state[error_key]
    if st.session_state.get(success_key):
        st.success("✅ Configuration reset!")
        del st.session_state[success_key]


def make_reset_config_callback(keys_prefix: str = "config"):
    """Return a synchronous callback that schedules the async reset.

    The returned function can be used with Streamlit's `on_click`. It will
    schedule the async `reset_config_api` flow using the running loop when
    available, otherwise it will run it to completion.
    """

    def _schedule_reset():
        async def _run():
            result = await reset_config_api(reload_vectorstore=False)
            if "error" in result:
                st.session_state[f"{keys_prefix}_reset_error"] = result["error"]
                return
            new_config = await get_config_api()
            st.session_state.runtime_config = new_config
            st.session_state[f"{keys_prefix}_model"] = new_config.get("gemini_model", "gemini-2.5-flash")
            st.session_state[f"{keys_prefix}_temperature"] = float(new_config.get("temperature", 0.1))
            st.session_state[f"{keys_prefix}_max_tokens"] = int(new_config.get("max_output_tokens", 2048))
            st.session_state[f"{keys_prefix}_retriever_limit"] = int(new_config.get("retriever_limit", 5))
            st.session_state[f"{keys_prefix}_max_retries"] = int(new_config.get("max_retries", 3))
            st.session_state[f"{keys_prefix}_summarize"] = bool(new_config.get("summarize", False))
            st.session_state[f"{keys_prefix}_reset_success"] = True

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run())
        except RuntimeError:
            asyncio.run(_run())

    return _schedule_reset


async def threads_management_sidebar():
    if not st.session_state.threads_id:
        st.info("No conversations yet. Click 'New Conversation' to start!")
        return

    for thread_id in st.session_state.threads_id:
        is_selected = thread_id == st.session_state.selected_thread

        col1, col2 = st.columns([5, 1])
        with col1:
            button_label = f"{'✅ ' if is_selected else '💬 '}{thread_id[:8]}..."
            if st.button(
                button_label,
                key=f"select_{thread_id}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                await select_thread(thread_id)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{thread_id}"):
                await delete_thread_and_update_state(thread_id)


async def first_setup():
    st.title("🧬 Vir ChatBot")
    st.divider()

    st.text_input(
        "Enter your User ID:",
        value=st.session_state.get("user_id", "default_user"),
        key="user_id",
    )

    if (
        "initialized_user" not in st.session_state
        or st.session_state.get("initialized_user") != st.session_state.user_id
    ):
        await start_session_data(st.session_state.user_id)
        st.session_state.initialized_user = st.session_state.user_id

    st.caption(f"👤 User: {st.session_state.user_id}")

    if st.button("➕ New Conversation", use_container_width=True):
        await create_new_thread(st.session_state.user_id)

    st.divider()
    st.subheader("💬 Conversations")

    await threads_management_sidebar()


async def run_reload_vectorstore():
    if st.button("🔄 Reload VectorStore", use_container_width=True, key="reload_vs"):
        with st.spinner("Reloading VectorStore..."):
            result = await reload_vectorstore_api()
            status = result.get("status")
            message = result.get("message", "Unknown error")

            if status == "success":
                st.success(f"✅ {message}")
            elif status == "warning":
                st.warning(f"⚠️ {message}")
            elif status == "error":
                st.error(f"❌ {message}")


async def run_upload_pdfs_vectorstore():
    uploaded_files = st.file_uploader(
        "Select PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files and st.button("📤 Upload and Create VectorStore", use_container_width=True):
        with st.spinner("Uploading PDFs..."):
            result = await upload_pdfs_api(uploaded_files)
            if "error" in result:
                return st.error(f"Error: {result['error']}")
            st.success(f"✅ {result.get('message', 'PDFs uploaded!')}")
            task_id = result.get("task_id")
            if task_id:
                add_active_task(task_id, f"Upload of {len(uploaded_files)} PDF(s)")
                st.rerun()
            return


async def run_create_from_folder_vectorstore():
    if st.button("📁 Create VectorStore from Folder", use_container_width=True):
        with st.spinner("Starting creation..."):
            result = await create_vectorstore_from_folder_api()
            if "error" in result:
                return st.error(f"Error: {result['error']}")
            st.success(f"✅ {result.get('message', 'Process started!')}")
            task_id = result.get("task_id")
            if task_id:
                add_active_task(task_id, "Create VectorStore from Folder")
                st.rerun()


async def run_vectorstore_settings():
    config = st.session_state.runtime_config
    if "error" in config:
        return st.error(f"Error loading config: {config['error']}")

    new_model = model_selection(config, key="vs_config_model")
    new_summarize = summarize_toggle(config, key="vs_config_summarize")
    new_max_tokens = max_tokens_input(config, key="vs_config_max_tokens")

    config_changed = (
        new_model != config.get("gemini_model")
        or new_summarize != config.get("summarize")
        or new_max_tokens != config.get("max_output_tokens")
    )

    updates = {
        "gemini_model": new_model,
        "summarize": new_summarize,
        "max_output_tokens": new_max_tokens,
    }

    col1, col2 = st.columns(2)

    with col1:
        await run_save_config(config_changed, updates, key="vs_save_config")

    with col2:
        run_reset_config(key="vs_reset_config", keys_prefix="vs_config")

    if config_changed:
        st.info("💡 You have unsaved changes")


async def check_selected_pdfs():
    if st.button("🔄 Refresh List", use_container_width=True, key="refresh_pdfs"):
        st.session_state.pdf_list = await list_pdfs_api()
        st.rerun()

    if "pdf_list" not in st.session_state:
        st.session_state.pdf_list = await list_pdfs_api()

    if "selected_pdfs" not in st.session_state:
        st.session_state.selected_pdfs = []

    pdf_list = st.session_state.pdf_list

    if not pdf_list:
        return st.info("No PDFs found in VectorStore.")

    st.markdown(f"**{len(pdf_list)} PDF(s) found**")

    search_term = st.text_input("🔍 Search PDF", key="pdf_search")

    filtered_pdfs = [pdf for pdf in pdf_list if search_term.lower() in pdf.lower()] if search_term else pdf_list

    selected = st.multiselect(
        "Select PDFs to delete:",
        options=filtered_pdfs,
        default=[],
        key="pdf_multiselect",
    )

    if not selected:
        return

    st.warning(f"⚠️ {len(selected)} PDF(s) selected for deletion")

    if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
        with st.spinner("Deleting PDFs..."):
            result = await delete_pdfs_api(selected)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"✅ {result.get('message', 'PDFs deleted!')}")
                task_id = result.get("task_id")
                if task_id:
                    add_active_task(task_id, f"Delete {len(selected)} PDF(s)")
                st.session_state.pdf_list = await list_pdfs_api()
                st.rerun()


async def run_llm_config():
    with st.expander("🔧 LLM Settings", expanded=False):
        config = st.session_state.runtime_config

        if "error" in config:
            st.error(f"Error loading config: {config['error']}")
        else:
            new_model = model_selection(config)
            new_temperature = temperature_slider(config)
            new_max_tokens = max_tokens_input(config)
            new_retriever_limit = retriever_limit_input(config)
            new_max_retries = max_retries_input(config)

            st.divider()

            config_changed = (
                new_model != config.get("gemini_model")
                or new_temperature != config.get("temperature")
                or new_max_tokens != config.get("max_output_tokens")
                or new_retriever_limit != config.get("retriever_limit")
                or new_max_retries != config.get("max_retries")
            )

            updates = {
                "gemini_model": new_model,
                "temperature": new_temperature,
                "max_output_tokens": new_max_tokens,
                "retriever_limit": new_retriever_limit,
                "max_retries": new_max_retries,
            }

            col1, col2 = st.columns(2)

            with col1:
                await run_save_config(config_changed, updates)

            with col2:
                run_reset_config()

            if config_changed:
                st.info("💡 You have unsaved changes")


async def run_chat():
    st.header(f"💬 Conversation: {st.session_state.selected_thread[:8]}...")
    chat_container = st.container()
    with chat_container:
        for message in get_current_messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        async for chunk in chat_stream_api(
            message=prompt,
            thread_id=st.session_state.selected_thread,
            user_id=st.session_state.user_id,
        ):
            full_response = chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    add_message("assistant", full_response)


async def run_chat_no_thread():
    st.title("🧬 Welcome to Vir ChatBot!")
    st.markdown(
        """
    ### How to use:
    1. Click **"New Conversation"** in the sidebar to create a new chat
    2. Type your message in the text box below
    3. The assistant will respond based on the available virology knowledge
    ---
    *Select or create a conversation to start!*
    """
    )

    if st.button("🚀 Start New Conversation", type="primary"):
        await create_new_thread(st.session_state.user_id)


async def main():
    with st.sidebar:
        await first_setup()

        st.divider()
        st.subheader("📚 Manage VectorStore")

        await run_reload_vectorstore()

        with st.expander("➕ Create VectorStore", expanded=False):
            st.markdown("**Option 1: Upload PDFs**")
            await run_upload_pdfs_vectorstore()

            st.markdown("---")
            st.markdown("**Option 2: Create from Folder**")
            st.caption("Creates the VectorStore from the configured PDFs folder.")
            await run_create_from_folder_vectorstore()

            st.markdown("---")
            st.markdown("**VectorStore Settings**")
            st.caption("Define if the summarization of context with AI should be enabled during VectorStore creation.")
            await run_vectorstore_settings()

        with st.expander("📋 PDFs in VectorStore", expanded=False):
            await check_selected_pdfs()

        st.divider()
        st.subheader("⚙️ Configuration")

        await run_llm_config()

        if st.session_state.active_tasks:
            st.divider()
            await render_task_progress()

    if st.session_state.selected_thread:
        await run_chat()
    else:
        await run_chat_no_thread()


if __name__ == "__main__":
    asyncio.run(main())
