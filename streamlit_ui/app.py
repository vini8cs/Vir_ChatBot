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


def delete_thread_api(user_id: str, thread_id: str):
    """Call the API to delete a thread for a specific user."""
    response = requests.delete(f"{_.API_BASE_URL}/threads/{user_id}/{thread_id}")
    response.raise_for_status()
    return response.json()


def delete_thread_and_update_state(thread_id: str):
    """
    Delete a thread and update the session state to reflect the deleted thread.

    Args:
        thread_id (str): The thread ID
    """
    delete_thread_api(st.session_state.user_id, thread_id)
    st.session_state.threads_id.remove(thread_id)

    # Select another thread if available
    if st.session_state.threads_id:
        st.session_state.selected_thread = st.session_state.threads_id[-1]
    else:
        st.session_state.selected_thread = None

    # Clear thread state since we deleted the current thread
    st.session_state.thread_state = {}
    st.rerun()
