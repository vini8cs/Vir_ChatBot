import asyncio

import streamlit as st
from views.chat import run_chat, run_chat_no_thread
from views.config import run_llm_config, run_prompt_config
from views.sidebar import first_setup
from views.tasks import render_task_progress
from views.vectorstore import check_selected_pdfs, run_reload_vectorstore, run_upload_pdfs_vectorstore

st.set_page_config(page_title="Vir ChatBot", page_icon="🧬", layout="wide")


async def main():
    with st.sidebar:
        await first_setup()

        st.divider()
        st.subheader("📚 Manage VectorStore")

        await run_reload_vectorstore()

        with st.expander("➕ Create VectorStore", expanded=False):
            await run_upload_pdfs_vectorstore()

        with st.expander("📋 Documents in VectorStore", expanded=False):
            await check_selected_pdfs()

        st.divider()
        st.subheader("⚙️ Configuration")

        await run_llm_config()
        await run_prompt_config()

        if st.session_state.active_tasks:
            st.divider()
            await render_task_progress()

    if st.session_state.selected_thread:
        await run_chat()
    else:
        await run_chat_no_thread()


if __name__ == "__main__":
    asyncio.run(main())
