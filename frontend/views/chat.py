import streamlit as st
from api_client import chat_stream_api, rename_thread_api
from session import add_message, create_new_thread, get_current_messages


async def run_chat():
    thread_id = st.session_state.selected_thread
    thread_name = st.session_state.thread_names.get(
        thread_id, f"{thread_id[:8]}..."
    )
    st.header(f"💬 {thread_name}")
    chat_container = st.container()
    with chat_container:
        for message in get_current_messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    is_first_message = len(get_current_messages()) == 0
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

    if is_first_message:
        title = prompt[:45].strip() + ("..." if len(prompt) > 45 else "")
        await rename_thread_api(thread_id, title)
        st.session_state.thread_names[thread_id] = title
        st.rerun()


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
