import os

import pandas as pd
import streamlit as st

from runner import run_add, run_build, run_delete

# -----------------------
# PATHS
# -----------------------
PDF_FOLDER = "pdfs"
TEMP_FOLDER = "temp_folder"
CACHE_PATH = "vectorstore_cache.csv"
VECTORSTORE_PATH = "vectorstore"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

st.set_page_config(page_title="VirCharBot", layout="wide")

# -----------------------
# SESSION DEFAULTS
# -----------------------
if "config" not in st.session_state:
    st.session_state.config = {
        "vectorstore_path": VECTORSTORE_PATH,
        "pdf_folder": PDF_FOLDER,
        "temp_folder": TEMP_FOLDER,
        "cache": CACHE_PATH,
        "gemini_model": "gemini-2.5-flash",
        "embedding_model": "gemini-embedding-001",
        "temperature": 0.1,
        "max_output_tokens": 2048,
        "max_tokens_for_chunk": 2048,
        "languages": ["eng", "pt"],
        "tokenizer_model": "mistralai/Mistral-7B-v0.1",
        "threads": 4,
        "dont_summarize": False,
        "pdfs_to_delete": [],
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = {}

cfg = st.session_state.config

# =========================================================
# SIDEBAR (ENTIRE LEFT SIDE)
# =========================================================
with st.sidebar:
    st.title("Chats")

    # ---------------------------
    # Search chats + list
    # ---------------------------
    search_chat = st.text_input("Search chats")
    matches = [name for name in st.session_state.saved_chats if search_chat.lower() in name.lower()]

    for chat_name in matches:
        if st.button(chat_name):
            st.session_state.chat_history = st.session_state.saved_chats[chat_name]

    # ---------------------------
    # New chat button (separated)
    # ---------------------------
    st.markdown("---")
    if st.button("New Chat"):
        st.session_state.chat_history = []

    st.markdown("---")

    # =====================================================
    # CONFIGURATION PANEL
    # =====================================================
    with st.expander("Configuration", expanded=False):
        cfg["vectorstore_path"] = st.text_input("Vectorstore path", cfg["vectorstore_path"])
        cfg["pdf_folder"] = st.text_input("PDF folder", cfg["pdf_folder"])
        cfg["temp_folder"] = st.text_input("Temp folder", cfg["temp_folder"])
        cfg["cache"] = st.text_input("Cache CSV path", cfg["cache"])

        cfg["gemini_model"] = st.text_input("Gemini model", cfg["gemini_model"])
        cfg["embedding_model"] = st.text_input("Embedding model", cfg["embedding_model"])
        cfg["tokenizer_model"] = st.text_input("Tokenizer model", cfg["tokenizer_model"])

        cfg["temperature"] = st.number_input("Temperature", value=cfg["temperature"], format="%.2f")
        cfg["max_output_tokens"] = st.number_input("Max output tokens", value=cfg["max_output_tokens"])
        cfg["max_tokens_for_chunk"] = st.number_input("Max tokens per chunk", value=cfg["max_tokens_for_chunk"])
        cfg["threads"] = st.number_input("Threads", min_value=1, value=cfg["threads"])

        cfg["dont_summarize"] = st.checkbox("Skip summarization", value=cfg["dont_summarize"])
        cfg["languages"] = st.multiselect("Languages (OCR)", ["eng", "pt", "es", "fr"], default=cfg["languages"])

    # =====================================================
    # VECTORSTORE / PDF MANAGEMENT
    # =====================================================
    with st.expander("Vectorstore / PDF Management", expanded=False):

        st.header("Upload PDFs")
        uploaded = st.file_uploader("Insert PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                path = os.path.join(cfg["pdf_folder"], f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
            st.success("PDFs uploaded.")

        st.header("Vectorstore Actions")
        if st.button("Create vectorstore"):
            run_build(cfg, callback=st.write)
            st.success("Vectorstore created.")

        if st.button("Update vectorstore"):
            run_add(cfg, callback=st.write)
            st.success("Vectorstore updated.")

        st.header("Delete PDFs")
        if os.path.exists(cfg["cache"]):
            df = pd.read_csv(cfg["cache"])
        else:
            df = pd.DataFrame(columns=["filename"])

        query = st.text_input("Search PDFs")
        filtered = df[df["filename"].str.contains(query, case=False)]
        selection = st.multiselect("Select PDFs", filtered["filename"].tolist())

        if st.button("Delete selected PDFs"):
            temp_cfg = cfg.copy()
            temp_cfg["pdfs_to_delete"] = selection
            run_delete(temp_cfg)
            st.success("PDFs deleted.")

# =========================================================
# MAIN CHAT AREA
# =========================================================
st.title("Vir_ChatBot")

user_input = st.text_input("Message")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Example model output placeholder
    bot_response = "Bot: " + user_input[::-1]
    st.session_state.chat_history.append({"role": "bot", "content": bot_response})

st.subheader("History")
for msg in st.session_state.chat_history:
    st.write(f"**{msg['role'].upper()}**: {msg['content']}")
