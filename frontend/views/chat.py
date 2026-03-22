import streamlit as st
from api_client import chat_stream_api
from session import add_message, create_new_thread, get_current_messages


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
