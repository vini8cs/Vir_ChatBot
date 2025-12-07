import requests
import streamlit as st
from api import get_user_threads

import config as _


def start_session_data(user_id: str):
    if "user_id" not in st.session_state:
        st.session_state.user_id = user_id

    if "threads_id" not in st.session_state:
        st.session_state.threads_id = []
        threads = get_user_threads(st.session_state.user_id)
        for thread in threads:
            st.session_state.threads_id.append(thread["thread_id"])

    if "selected_thread" not in st.session_state:
        if st.session_state.threads_id:
            st.session_state.selected_thread = st.session_state.threads_id[-1]
        else:
            st.session_state.selected_thread = None

    if "thread_state" not in st.session_state:
        st.session_state.thread_state = {}


def create_thread_api(user_id: str) -> dict:
    response = requests.post(f"{_.API_BASE_URL}/threads/create", json={"user_id": user_id})
    response.raise_for_status()
    return response.json()


def create_new_thread(user_id: str):
    thread = create_thread_api(user_id)
    st.session_state.threads_id.append(thread["thread_id"])
    st.session_state.selected_thread = thread["thread_id"]
    st.session_state.thread_state = {}
    st.rerun()
