import streamlit as st
from session import (
    create_new_thread,
    delete_thread_and_update_state,
    select_thread,
    start_session_data,
)


async def threads_management_sidebar():
    if not st.session_state.threads_id:
        st.info("No conversations yet. Click 'New Conversation' to start!")
        return

    for thread_id in st.session_state.threads_id:
        is_selected = thread_id == st.session_state.selected_thread

        col1, col2 = st.columns([5, 1])
        with col1:
            button_label = (
                f"{'✅ ' if is_selected else '💬 '}{thread_id[:8]}..."
            )
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
        or st.session_state.get("initialized_user")
        != st.session_state.user_id
    ):
        await start_session_data(st.session_state.user_id)
        st.session_state.initialized_user = st.session_state.user_id

    st.caption(f"👤 User: {st.session_state.user_id}")

    if st.button("➕ New Conversation", use_container_width=True):
        await create_new_thread(st.session_state.user_id)

    st.divider()
    st.subheader("💬 Conversations")

    await threads_management_sidebar()
