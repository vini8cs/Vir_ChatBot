from datetime import datetime

import streamlit as st
from api_client import (
    create_thread_api,
    delete_thread_api,
    gather_tasks,
    get_config_api,
    get_thread_messages_api,
    get_user_threads_api,
)


async def start_session_data(user_id: str):
    """Initialize session state data with parallel fetching."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = user_id
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "active_tasks" not in st.session_state:
        st.session_state.active_tasks = {}
    if "thread_names" not in st.session_state:
        st.session_state.thread_names = {}

    need_threads = "threads_id" not in st.session_state
    need_config = "runtime_config" not in st.session_state

    tasks = []
    names = []

    if need_threads:
        tasks.append(get_user_threads_api(st.session_state.user_id))
        names.append("threads")

    if need_config:
        tasks.append(get_config_api())
        names.append("config")

    mapping = await gather_tasks(names, tasks) if tasks else {}

    threads_result = mapping.get("threads")
    config_result = mapping.get("config")

    if need_threads and threads_result is not None:
        st.session_state.threads_id = [t["thread_id"] for t in threads_result]
        st.session_state.thread_names = {
            t["thread_id"]: t.get("name") or f"{t['thread_id'][:8]}..."
            for t in threads_result
        }

    if need_config and config_result is not None:
        st.session_state.runtime_config = config_result

    if "selected_thread" not in st.session_state:
        st.session_state.selected_thread = (
            st.session_state.threads_id[-1]
            if st.session_state.threads_id
            else None
        )

    if (
        st.session_state.selected_thread
        and st.session_state.selected_thread not in st.session_state.messages
    ):
        st.session_state.messages[
            st.session_state.selected_thread
        ] = await get_thread_messages_api(st.session_state.selected_thread)

    if "runtime_config" not in st.session_state:
        st.session_state.runtime_config = await get_config_api()


async def create_new_thread(user_id: str):
    """Create a new thread and update session state."""
    thread = await create_thread_api(user_id)
    thread_id = thread["thread_id"]
    st.session_state.threads_id.append(thread_id)
    st.session_state.selected_thread = thread_id
    st.session_state.messages[thread_id] = []
    st.session_state.thread_names[thread_id] = f"{thread_id[:8]}..."
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
        st.session_state.messages[thread_id] = await get_thread_messages_api(
            thread_id
        )


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
        st.session_state.messages[thread_id].append(
            {"role": role, "content": content}
        )


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
