"""
Chat interface component.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from frontend.api_utils import get_api_response


def display_chat_interface():
    # ── Display chat history ───────────────────────────────────────────────
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Handle new user input ──────────────────────────────────────────────
    if prompt := st.chat_input("Ask anything about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        model = st.session_state.get("model", "gpt-4o-mini")
        session_id = st.session_state.get("session_id")

        with st.spinner("Thinking…"):
            response = get_api_response(prompt, session_id, model)

        if response:
            st.session_state.session_id = response.get("session_id")
            answer = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

            with st.expander("🔍 Response details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Model")
                    st.code(response.get("model", "-"))
                with col2:
                    st.caption("Session ID")
                    st.code(response.get("session_id", "-"))
        else:
            st.error("Failed to get a response. Check the API is running and your API key is set.")
