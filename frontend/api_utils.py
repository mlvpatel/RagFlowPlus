"""
Frontend API utilities, all calls go through the v1 router.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import os

import requests
import streamlit as st

# ── configuration ────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")


def _headers(json: bool = False) -> dict:
    """Build headers with optional API key and Content-Type."""
    h = {"X-API-Key": API_KEY} if API_KEY else {}
    if json:
        h["Content-Type"] = "application/json"
        h["accept"] = "application/json"
    return h


def get_api_response(question: str, session_id: str, model: str) -> dict | None:
    """Send a chat query and return the parsed JSON response, or None on error."""
    payload = {"question": question, "model": model}
    if session_id:
        payload["session_id"] = session_id
    try:
        resp = requests.post(
            f"{API_URL}/v1/chat",
            headers=_headers(json=True),
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Chat API error {resp.status_code}: {resp.text}")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model may be loading, please retry.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None


def upload_document(file) -> dict | None:
    """Upload a document to the API and return the task info, or None on error."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/upload-doc",
            headers=_headers(),
            files={"file": (file.name, file, file.type)},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Upload failed {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Upload error: {e}")
    return None


def list_documents() -> list:
    """Fetch the list of indexed documents."""
    try:
        resp = requests.get(
            f"{API_URL}/v1/list-docs",
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"List error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
    return []


def delete_document(file_id: int) -> dict | None:
    """Delete a document by id."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/delete-doc",
            headers=_headers(json=True),
            json={"file_id": file_id},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"Delete error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Error deleting document: {e}")
    return None


def get_task_status(task_id: str) -> dict | None:
    """Poll the status of an async indexing task."""
    try:
        resp = requests.get(
            f"{API_URL}/v1/task/{task_id}",
            headers=_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None
