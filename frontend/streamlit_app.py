"""
rag-advanced-2023 Streamlit App entry point.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from frontend.chat_interface import display_chat_interface
from frontend.sidebar import display_sidebar

st.set_page_config(
    page_title="rag-advanced-2023, AI Document Chat",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 rag-advanced-2023, Chat with Your Documents")
st.caption("Advanced RAG, 2023. Hybrid retrieval with cross encoder reranking.")

# ── Initialise session state defaults ─────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("model", "gpt-4o-mini")
st.session_state.setdefault("documents", [])

display_sidebar()
display_chat_interface()
