from concurrent.futures import ThreadPoolExecutor
import os
import time

import requests
import streamlit as st
from requests import exceptions as requests_exceptions

# 1. Configuration
BASE_URL = os.getenv("BACKEND_URL", "https://finsolve-api-278963574842.us-central1.run.app")  #http://localhost:8000
REQUEST_TIMEOUT_SECONDS = 30
st.set_page_config(page_title="FinSolve Secure Chat", layout="wide")

# 2. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None
if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)
if "inflight_future" not in st.session_state:
    st.session_state.inflight_future = None
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False


def bearer_headers(access_token):
    return {"Authorization": f"Bearer {access_token}"}


def send_chat_request(prompt, access_token):
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": prompt},
        headers=bearer_headers(access_token),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()

# 3. Sidebar for Authentication
with st.sidebar:
    st.title("FinSolve Login")
    if not st.session_state.access_token:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                resp = requests.post(
                    f"{BASE_URL}/login",
                    auth=(username, password),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                access_token = resp.json()["access_token"]

                profile_resp = requests.get(
                    f"{BASE_URL}/test",
                    headers=bearer_headers(access_token),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                profile_resp.raise_for_status()
                profile = profile_resp.json()
                authenticated_username = username
                authenticated_role = profile["role"]
            except (requests_exceptions.RequestException, KeyError, ValueError):
                st.error("Login failed. Check your credentials and try again.")
            else:
                st.session_state.access_token = access_token
                st.session_state.username = authenticated_username
                st.session_state.role = authenticated_role
                st.success(f"Logged in as {st.session_state.role}")
                st.rerun()
    else:
        st.write(f"Logged in as: **{st.session_state.username}**")
        st.write(f"Role: `{st.session_state.role}`")
        if st.button("Logout"):
            st.session_state.access_token = None
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()

# 4. Main Chat Interface
st.title("FinSolve AI Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("Ask a question about FinSolve..."):
    if not st.session_state.access_token:
        st.warning("Please log in first.")
    else:
        if st.session_state.inflight_future and not st.session_state.inflight_future.done():
            st.warning("A request is already in progress. Cancel it or wait.")
        else:
            st.session_state.cancel_requested = False
            st.session_state.inflight_future = st.session_state.executor.submit(
                send_chat_request,
                prompt,
                st.session_state.access_token,
            )

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

future = st.session_state.inflight_future
if future:
    if future.done():
        st.session_state.inflight_future = None
        if st.session_state.cancel_requested:
            st.info("Request canceled.")
            st.session_state.cancel_requested = False
        else:
            try:
                data = future.result()
            except requests_exceptions.Timeout:
                st.error("Request timed out. The server is still working or unavailable.")
            except requests_exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")
            else:
                answer = data["answer"]
                sources = data["sources"]

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in set(sources):
                                st.write(f"- {src}")

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.spinner("Retrieving authorized data..."):
            if st.button("Cancel request"):
                st.session_state.cancel_requested = True
                future.cancel()
                st.session_state.inflight_future = None
                st.info("Request canceled. The server may still finish in the background.")
                st.rerun()
            time.sleep(0.5)
            st.rerun()
