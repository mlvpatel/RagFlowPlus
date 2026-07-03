"""
RAGFlow Streamlit App entry point.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from frontend.chat_interface import display_chat_interface
from frontend.sidebar import display_sidebar

st.set_page_config(
    page_title="RAGFlow, AI Document Chat",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAGFlow, Chat with Your Documents")
st.caption("Powered by Gemini Embedding 2 · Retrieval-Augmented Generation")

# ── Initialise session state defaults ─────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("model", "gpt-4o-mini")
st.session_state.setdefault("documents", [])

display_sidebar()
display_chat_interface()
