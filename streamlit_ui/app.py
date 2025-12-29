import json
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
GEMINI_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


st.set_page_config(page_title="Vir ChatBot", page_icon="🤖", layout="wide")


def get_user_threads_api(user_id: str) -> list:
    """Call the API to get all threads for a user."""
    try:
        response = requests.get(f"{API_BASE_URL}/threads/{user_id}")
        response.raise_for_status()
        data = response.json()
        return data.get("threads", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching threads: {e}")
        return []


def create_thread_api(user_id: str) -> dict:
    """Call the API to create a new thread."""
    response = requests.post(f"{API_BASE_URL}/threads/create", json={"user_id": user_id})
    response.raise_for_status()
    return response.json()


def delete_thread_api(user_id: str, thread_id: str):
    """Call the API to delete a thread for a specific user."""
    response = requests.delete(f"{API_BASE_URL}/threads/{user_id}/{thread_id}")
    response.raise_for_status()
    return response.json()


def get_thread_messages_api(thread_id: str) -> list:
    """Call the API to get message history for a thread."""
    try:
        response = requests.get(f"{API_BASE_URL}/threads/{thread_id}/messages")
        response.raise_for_status()
        data = response.json()
        return data.get("messages", [])
    except requests.exceptions.RequestException as e:
        print(f"Error loading messages: {e}")
        return []


def chat_stream_api(message: str, thread_id: str, user_id: str):
    """
    Call the chat stream API and yield responses.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json={"message": message, "thread_id": thread_id, "user_id": user_id},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue

            data = line_str[6:]
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
                if "content" in payload:
                    yield payload["content"]
                else:
                    yield f"❌ Error: {payload['error']}"
            except json.JSONDecodeError:
                continue
    except requests.exceptions.Timeout:
        yield "❌ Error: Timeout exceeded. The system may be busy."
    except requests.exceptions.RequestException as e:
        yield f"❌ Connection error: {e}"


def list_pdfs_api() -> list:
    """Get list of PDFs in the vectorstore cache."""
    try:
        response = requests.get(f"{API_BASE_URL}/pdfs/list")
        response.raise_for_status()
        data = response.json()
        return data.get("pdfs", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PDFs: {e}")
        return []


def upload_pdfs_api(files) -> dict:
    """Upload PDFs to create/update vectorstore."""
    try:
        files_to_upload = [("files", (file.name, file.getvalue(), "application/pdf")) for file in files]
        response = requests.post(
            f"{API_BASE_URL}/create-vectorstore-based-on-selected-pdfs/",
            files=files_to_upload,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def create_vectorstore_from_folder_api() -> dict:
    """Create vectorstore from folder."""
    try:
        response = requests.post(f"{API_BASE_URL}/create-vectorstore-from-folder/")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def delete_pdfs_api(filenames: list) -> dict:
    """Delete PDFs from vectorstore."""
    try:
        response = requests.post(f"{API_BASE_URL}/delete-selected-pdfs/", json={"filenames": filenames})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def reload_vectorstore_api() -> dict:
    """Reload the VectorStore into memory after creation/update."""
    try:
        response = requests.post(f"{API_BASE_URL}/vectorstore/reload")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_config_api() -> dict:
    """Get current runtime configuration."""
    try:
        response = requests.get(f"{API_BASE_URL}/config")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def update_config_api(config_updates: dict) -> dict:
    """Update runtime configuration."""
    try:
        response = requests.put(f"{API_BASE_URL}/config", json=config_updates)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def reset_config_api(reload_vectorstore: bool = False) -> dict:
    """Reset configuration to defaults."""
    try:
        endpoint = "/config/reset-and-reload" if reload_vectorstore else "/config/reset"
        response = requests.post(f"{API_BASE_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_task_status_api(task_id: str) -> dict:
    """Get the status and progress of a background task."""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status": "ERROR"}


def start_session_data(user_id: str):
    """Initialize session state data."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = user_id

    if "threads_id" not in st.session_state:
        st.session_state.threads_id = []
        threads = get_user_threads_api(st.session_state.user_id)
        for thread in threads:
            st.session_state.threads_id.append(thread["thread_id"])

    if "selected_thread" not in st.session_state:
        if st.session_state.threads_id:
            st.session_state.selected_thread = st.session_state.threads_id[-1]
        else:
            st.session_state.selected_thread = None

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    if st.session_state.selected_thread and st.session_state.selected_thread not in st.session_state.messages:
        st.session_state.messages[st.session_state.selected_thread] = get_thread_messages_api(
            st.session_state.selected_thread
        )

    if "active_tasks" not in st.session_state:
        st.session_state.active_tasks = {}


def create_new_thread(user_id: str):
    """Create a new thread and update session state."""
    thread = create_thread_api(user_id)
    thread_id = thread["thread_id"]
    st.session_state.threads_id.append(thread_id)
    st.session_state.selected_thread = thread_id
    st.session_state.messages[thread_id] = []
    st.rerun()


def delete_thread_and_update_state(thread_id: str):
    """Delete a thread and update the session state."""
    delete_thread_api(st.session_state.user_id, thread_id)
    st.session_state.threads_id.remove(thread_id)

    if thread_id in st.session_state.messages:
        del st.session_state.messages[thread_id]

    if st.session_state.threads_id:
        st.session_state.selected_thread = st.session_state.threads_id[-1]
    else:
        st.session_state.selected_thread = None

    st.rerun()


def select_thread(thread_id: str):
    """Select a thread and load its messages from the database."""
    st.session_state.selected_thread = thread_id
    if thread_id not in st.session_state.messages:
        st.session_state.messages[thread_id] = get_thread_messages_api(thread_id)


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
    from datetime import datetime

    st.session_state.active_tasks[task_id] = {
        "name": task_name,
        "started_at": datetime.now().isoformat(),
    }


def remove_active_task(task_id: str):
    """Remove a task from active tasks tracking."""
    if task_id in st.session_state.active_tasks:
        del st.session_state.active_tasks[task_id]


def render_task_progress():
    """Render progress bars for all active tasks."""
    if not st.session_state.active_tasks:
        return

    st.subheader("⏳ Tasks running...")

    tasks_to_remove = []
    has_running_tasks = False

    for task_id, task_info in list(st.session_state.active_tasks.items()):
        status = get_task_status_api(task_id)

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

                if task_status == "PENDING":
                    st.progress(0, text="⏳ Task queued...")
                    has_running_tasks = True

                elif task_status == "PROGRESS":
                    percent = status.get("percent", 0)
                    step = status.get("step", "")
                    details = status.get("details", "")
                    current = status.get("current", 0)
                    total = status.get("total", 0)

                    st.progress(percent / 100, text=f"🔄 {step}: {details} ({current}/{total})")
                    has_running_tasks = True

                elif task_status == "SUCCESS":
                    result = status.get("result", {})
                    message = result.get("message", "Finished successfully!")
                    result_status = result.get("status", "")

                    if result_status == "Failure":
                        st.error(f"❌ FAILURE: {result.get('error', 'Unknown Error')}")
                    else:
                        st.success(f"✅ {message}")
                        if "Upload" in task_name or "Create" in task_name:
                            reload_result = reload_vectorstore_api()
                            if reload_result.get("status") == "success":
                                st.info("🔄 VectorStore reloaded into memory.")
                    tasks_to_remove.append(task_id)

                elif task_status == "FAILURE":
                    error = status.get("error", "Unknown error")
                    st.error(f"❌ FAILURE: {error}")
                    tasks_to_remove.append(task_id)

                else:
                    st.warning(f"⚠️ Status: {task_status}")
                    has_running_tasks = True

            with col2:
                if task_status not in ["SUCCESS", "FAILURE"]:
                    st.caption(f"ID: {task_id[:8]}...")

        st.divider()

    # Remove completed tasks
    for task_id in tasks_to_remove:
        remove_active_task(task_id)

    # Auto-refresh button for running tasks
    if has_running_tasks:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("💡 Click 'Refresh' to see the latest progress")
        with col2:
            if st.button("🔄 Refresh", key="refresh_tasks", use_container_width=True):
                st.rerun()

        # NOTE: Removed st_autorefresh as it interrupts streaming chat responses
        # Users must manually click "Refresh" to refresh task progress


# ==================== MAIN APP ====================

start_session_data("default_user")

# Sidebar - Thread management
with st.sidebar:
    st.title("🤖 Vir ChatBot")
    st.divider()

    # User info
    st.caption(f"👤 User: {st.session_state.user_id}")

    # New thread button
    if st.button("➕ New Conversation", use_container_width=True):
        create_new_thread(st.session_state.user_id)

    st.divider()
    st.subheader("💬 Conversations")

    # List threads
    if st.session_state.threads_id:
        for thread_id in st.session_state.threads_id:
            is_selected = thread_id == st.session_state.selected_thread

            col1, col2 = st.columns([5, 1])
            with col1:
                # Thread selection button
                button_label = f"{'✅ ' if is_selected else '💬 '}{thread_id[:8]}..."
                if st.button(
                    button_label,
                    key=f"select_{thread_id}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                ):
                    select_thread(thread_id)
                    st.rerun()

            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{thread_id}"):
                    delete_thread_and_update_state(thread_id)
    else:
        st.info("No conversations yet. Click 'New Conversation' to start!")

    # ==================== VECTORSTORE MANAGEMENT ====================
    st.divider()
    st.subheader("📚 Manage VectorStore")

    # Reload VectorStore button
    if st.button("🔄 Reload VectorStore", use_container_width=True, key="reload_vs"):
        with st.spinner("Reloading VectorStore..."):
            result = reload_vectorstore_api()
            if result.get("status") == "success":
                st.success(f"✅ {result.get('message')}")
            elif result.get("status") == "warning":
                st.warning(f"⚠️ {result.get('message')}")
            else:
                st.error(f"❌ {result.get('message', 'Unknown error')}")

    # Expander for creating vectorstore
    with st.expander("➕ Create VectorStore", expanded=False):
        st.markdown("**Option 1: Upload PDFs**")
        uploaded_files = st.file_uploader(
            "Select PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files and st.button("📤 Upload and Create VectorStore", use_container_width=True):
            with st.spinner("Uploading PDFs..."):
                result = upload_pdfs_api(uploaded_files)
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"✅ {result.get('message', 'PDFs uploaded!')}")
                    task_id = result.get("task_id")
                    if task_id:
                        add_active_task(task_id, f"Upload of {len(uploaded_files)} PDF(s)")
                        st.rerun()

        st.markdown("---")
        st.markdown("**Option 2: Create from Folder**")
        st.caption("Creates the VectorStore from the configured PDFs folder.")

        if st.button("📁 Create VectorStore from Folder", use_container_width=True):
            with st.spinner("Starting creation..."):
                result = create_vectorstore_from_folder_api()
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"✅ {result.get('message', 'Process started!')}")
                    task_id = result.get("task_id")
                    if task_id:
                        add_active_task(task_id, "Create VectorStore from Folder")
                        st.rerun()

    # Expander for managing PDFs in vectorstore
    with st.expander("📋 PDFs in VectorStore", expanded=False):
        # Refresh button
        if st.button("🔄 Refresh List", use_container_width=True, key="refresh_pdfs"):
            st.session_state.pdf_list = list_pdfs_api()
            st.rerun()

        # Initialize pdf list in session state
        if "pdf_list" not in st.session_state:
            st.session_state.pdf_list = list_pdfs_api()

        if "selected_pdfs" not in st.session_state:
            st.session_state.selected_pdfs = []

        pdf_list = st.session_state.pdf_list

        if pdf_list:
            st.markdown(f"**{len(pdf_list)} PDF(s) found**")

            # Search filter
            search_term = st.text_input("🔍 Search PDF", key="pdf_search")

            # Filter PDFs based on search
            filtered_pdfs = [pdf for pdf in pdf_list if search_term.lower() in pdf.lower()] if search_term else pdf_list

            # Multiselect for PDFs
            selected = st.multiselect(
                "Select PDFs to delete:",
                options=filtered_pdfs,
                default=[],
                key="pdf_multiselect",
            )

            if selected:
                st.warning(f"⚠️ {len(selected)} PDF(s) selected for deletion")

                if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
                    with st.spinner("Deleting PDFs..."):
                        result = delete_pdfs_api(selected)
                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.success(f"✅ {result.get('message', 'PDFs deleted!')}")
                            task_id = result.get("task_id")
                            if task_id:
                                add_active_task(task_id, f"Delete {len(selected)} PDF(s)")
                            st.session_state.pdf_list = list_pdfs_api()
                            st.rerun()
        else:
            st.info("No PDFs found in VectorStore.")

    # ==================== CONFIGURATION MANAGEMENT ====================
    st.divider()
    st.subheader("⚙️ Configuration")

    with st.expander("🔧 LLM Settings", expanded=False):
        if "runtime_config" not in st.session_state:
            st.session_state.runtime_config = get_config_api()

        config = st.session_state.runtime_config

        if "error" in config:
            st.error(f"Error loading config: {config['error']}")
        else:
            current_model = config.get("gemini_model", "gemini-2.5-flash")
            model_index = GEMINI_MODELS.index(current_model) if current_model in GEMINI_MODELS else 0
            new_model = st.selectbox(
                "🤖 Gemini Model",
                options=GEMINI_MODELS,
                index=model_index,
                key="config_model",
            )

            new_temperature = st.slider(
                "🌡️ Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(config.get("temperature", 0.1)),
                step=0.1,
                key="config_temperature",
                help="Higher = more creative, Lower = more focused",
            )

            new_max_tokens = st.number_input(
                "📝 Max Output Tokens",
                min_value=256,
                max_value=8192,
                value=int(config.get("max_output_tokens", 2048)),
                step=256,
                key="config_max_tokens",
            )

            new_retriever_limit = st.number_input(
                "🔍 Retriever Limit (k)",
                min_value=1,
                max_value=20,
                value=int(config.get("retriever_limit", 5)),
                step=1,
                key="config_retriever_limit",
                help="Number of documents to retrieve",
            )

            new_max_retries = st.number_input(
                "🔄 Max Retries",
                min_value=1,
                max_value=10,
                value=int(config.get("max_retries", 3)),
                step=1,
                key="config_max_retries",
            )

            new_summarize = st.toggle(
                "📋 Summarize Context",
                value=bool(config.get("summarize", False)),
                key="config_summarize",
            )

            st.divider()

            config_changed = (
                new_model != config.get("gemini_model")
                or new_temperature != config.get("temperature")
                or new_max_tokens != config.get("max_output_tokens")
                or new_retriever_limit != config.get("retriever_limit")
                or new_max_retries != config.get("max_retries")
                or new_summarize != config.get("summarize")
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "💾 Save",
                    use_container_width=True,
                    type="primary" if config_changed else "secondary",
                    disabled=not config_changed,
                ):
                    updates = {
                        "gemini_model": new_model,
                        "temperature": new_temperature,
                        "max_output_tokens": new_max_tokens,
                        "retriever_limit": new_retriever_limit,
                        "max_retries": new_max_retries,
                        "summarize": new_summarize,
                    }
                    result = update_config_api(updates)
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("✅ Configuration saved!")
                        st.session_state.runtime_config = result.get("config", get_config_api())
                        st.rerun()

            with col2:
                if st.button("🔄 Reset Defaults", use_container_width=True):
                    result = reset_config_api(reload_vectorstore=False)
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("✅ Configuration reset!")
                        new_config = result.get("config", get_config_api())
                        st.session_state.runtime_config = new_config
                        # Clear widget keys to force refresh with new values
                        for key in [
                            "config_model",
                            "config_temperature",
                            "config_max_tokens",
                            "config_retriever_limit",
                            "config_max_retries",
                            "config_summarize",
                        ]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()

            if config_changed:
                st.info("💡 You have unsaved changes")

    if st.session_state.active_tasks:
        st.divider()
        render_task_progress()

if st.session_state.selected_thread:
    st.header(f"💬 Conversation: {st.session_state.selected_thread[:8]}...")

    chat_container = st.container()
    with chat_container:
        for message in get_current_messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Type your message..."):
        add_message("user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in chat_stream_api(
                message=prompt,
                thread_id=st.session_state.selected_thread,
                user_id=st.session_state.user_id,
            ):
                full_response = chunk  # The API sends the full accumulated content
                message_placeholder.markdown(full_response + "▌")

            # Final response without cursor
            message_placeholder.markdown(full_response)

        # Add assistant message to history
        add_message("assistant", full_response)

else:
    # No thread selected
    st.title("🤖 Welcome to Vir ChatBot!")
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

    # Quick start button
    if st.button("🚀 Start New Conversation", type="primary"):
        create_new_thread(st.session_state.user_id)
