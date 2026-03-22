import streamlit as st
from api_client import (
    delete_pdfs_api,
    list_pdfs_api,
    reload_vectorstore_api,
    upload_pdfs_api,
)
from session import add_active_task


async def run_reload_vectorstore():
    if st.button(
        "🔄 Reload VectorStore", use_container_width=True, key="reload_vs"
    ):
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
        "Select files (PDF, DOCX, JPEG, PNG, TXT, TSV)",
        type=["pdf", "docx", "doc", "jpg", "jpeg", "png", "txt", "tsv"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files and st.button(
        "📤 Upload and Create VectorStore", use_container_width=True
    ):
        with st.spinner("Uploading files..."):
            result = await upload_pdfs_api(uploaded_files)
            if "error" in result:
                return st.error(f"Error: {result['error']}")
            st.success(f"✅ {result.get('message', 'Files uploaded!')}")
            task_id = result.get("task_id")
            if task_id:
                add_active_task(
                    task_id, f"Upload of {len(uploaded_files)} file(s)"
                )
                st.rerun()
            return


async def check_selected_pdfs():
    if st.button(
        "🔄 Refresh List", use_container_width=True, key="refresh_pdfs"
    ):
        st.session_state.pdf_list = await list_pdfs_api()
        st.rerun()

    if "pdf_list" not in st.session_state:
        st.session_state.pdf_list = await list_pdfs_api()

    if "selected_pdfs" not in st.session_state:
        st.session_state.selected_pdfs = []

    pdf_list = st.session_state.pdf_list

    if not pdf_list:
        return st.info("No documents found in VectorStore.")

    st.markdown(f"**{len(pdf_list)} document(s) found**")

    search_term = st.text_input("🔍 Search document", key="pdf_search")

    filtered_pdfs = (
        [pdf for pdf in pdf_list if search_term.lower() in pdf.lower()]
        if search_term
        else pdf_list
    )

    selected = st.multiselect(
        "Select documents to delete:",
        options=filtered_pdfs,
        default=[],
        key="pdf_multiselect",
    )

    if not selected:
        return

    st.warning(f"⚠️ {len(selected)} document(s) selected for deletion")

    if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
        with st.spinner("Deleting documents..."):
            result = await delete_pdfs_api(selected)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"✅ {result.get('message', 'Documents deleted!')}")
                task_id = result.get("task_id")
                if task_id:
                    add_active_task(
                        task_id, f"Delete {len(selected)} document(s)"
                    )
                st.session_state.pdf_list = await list_pdfs_api()
                st.rerun()
