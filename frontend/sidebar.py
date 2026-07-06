"""
Sidebar component, document management panel.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from frontend.api_utils import delete_document, get_task_status, list_documents, upload_document


def display_sidebar():
    st.sidebar.title("⚙️ Settings")

    # ── Model selection ────────────────────────────────────────────────────
    model_options = [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-7-sonnet-20250219",
        "deepseek-r1",
        "llama3.2:3b",
        "llama3",
    ]
    st.sidebar.selectbox("🤖 Select Model", options=model_options, key="model")

    st.sidebar.divider()
    st.sidebar.header("📂 Documents")

    # ── Document upload ────────────────────────────────────────────────────
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "html", "txt", "md"],
        help="Supported: PDF, DOCX, HTML, TXT, Markdown",
    )
    if uploaded_file and st.sidebar.button("⬆️ Upload & Index"):
        with st.spinner("Uploading…"):
            task_info = upload_document(uploaded_file)
            if task_info:
                task_id = task_info.get("task_id")
                st.sidebar.info(
                    f"✅ Queued for indexing.\n\nTask ID: `{task_id}`\n"
                    "Refresh the document list in a moment when processing is complete."
                )
                st.session_state.documents = list_documents()

    # ── Document list ──────────────────────────────────────────────────────
    if st.sidebar.button("🔄 Refresh Documents"):
        st.session_state.documents = list_documents()

    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    docs = st.session_state.get("documents", [])
    if docs:
        with st.sidebar.expander(f"📄 {len(docs)} indexed document(s)", expanded=False):
            for doc in docs:
                st.text(f"[{doc['id']}] {doc['filename']}")

        selected_id = st.sidebar.selectbox(
            "Select document to delete",
            options=[doc["id"] for doc in docs],
            format_func=lambda fid: next((d["filename"] for d in docs if d["id"] == fid), str(fid)),
        )
        if st.sidebar.button("🗑️ Delete Selected", type="secondary"):
            with st.spinner("Deleting…"):
                result = delete_document(selected_id)
                if result:
                    st.sidebar.success("Document deleted.")
                    st.session_state.documents = list_documents()
    else:
        st.sidebar.caption("No documents indexed yet.")
