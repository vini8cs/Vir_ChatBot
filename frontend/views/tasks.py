import asyncio

import streamlit as st
from api_client import (
    cancel_task_api,
    get_task_status_api,
    reload_vectorstore_api,
)
from session import remove_active_task


def _create_pending_task():
    st.progress(0, text="⏳ Task queued...")
    return True


def _create_progress_task(status):
    percent = status.get("percent", 0)
    step = status.get("step", "")
    details = status.get("details", "")
    current = status.get("current", 0)
    total = status.get("total", 0)

    st.progress(percent / 100, text=f"🔄 {step}: {details} ({current}/{total})")
    return True


async def _create_success_task(status, task_name):
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


def _create_failure_task(status):
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
        return _create_pending_task(), False

    def _handle_progress(status, task_name, task_id):
        return _create_progress_task(status), False

    async def _handle_success(status, task_name, task_id):
        await _create_success_task(status, task_name)
        return False, True

    def _handle_failure(status, task_name, task_id):
        _create_failure_task(status)
        return False, True

    def _handle_revoked(status, task_name, task_id):
        st.warning("🚫 Task cancelled.")
        return False, True

    def _handle_unknown(status, task_name, task_id):
        st.warning(f"⚠️ Status: {status.get('status', 'UNKNOWN')}")
        return True, False

    STATUS_HANDLERS = {
        "PENDING": _handle_pending,
        "PROGRESS": _handle_progress,
        "SUCCESS": _handle_success,
        "FAILURE": _handle_failure,
        "REVOKED": _handle_revoked,
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
                if not remove_flag and st.button(
                    "🚫 Cancel",
                    key=f"cancel_{task_id}",
                    use_container_width=True,
                ):
                    result = await cancel_task_api(task_id)
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        remove_active_task(task_id)
                        st.rerun()

        st.divider()

    for task_id in tasks_to_remove:
        remove_active_task(task_id)

    if has_running_tasks:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("💡 Click 'Refresh' to see the latest progress")
        with col2:
            if st.button(
                "🔄 Refresh", key="refresh_tasks", use_container_width=True
            ):
                st.rerun()

        # NOTE: Removed st_autorefresh as it interrupts streaming chat responses
        # Users must manually click "Refresh" to refresh task progress
