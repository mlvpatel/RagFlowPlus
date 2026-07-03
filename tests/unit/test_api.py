"""
Unit tests for the FastAPI endpoints.
Uses TestClient (in-process, no server needed) with mocked dependencies.

Author: Malav Patel
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── test setup helpers ──────────────────────────────────────────────────────


def make_app():
    """
    Import and return the app with all external I/O mocked so tests
    don't hit real DBs, Chroma, or Celery.
    """
    import src.api.db_utils as db_mod
    import src.embeddings.chroma_utils as chroma_mod

    db_mod.create_application_logs = MagicMock()
    db_mod.create_document_store = MagicMock()

    with (
        patch("src.api.main.get_rag_chain"),
        patch("src.api.main.get_chat_history", return_value=[]),
        patch("src.api.main.insert_application_logs"),
        patch("src.api.main.get_all_documents", return_value=[]),
        patch("src.api.main.process_document") as mock_task,
        patch("src.api.main.delete_doc_from_chroma", return_value=True),
        patch("src.api.main.delete_document_record", return_value=True),
    ):

        from src.api.main import app

        return app, mock_task


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    """Return a fresh TestClient for each test."""
    with (
        patch("src.api.db_utils.create_application_logs"),
        patch("src.api.db_utils.create_document_store"),
        patch("src.embeddings.chroma_utils.vectorstore"),
        patch("src.embeddings.chroma_utils.query_embedding_function"),
        patch("src.core.langchain_utils._base_retriever"),
        patch("src.core.langchain_utils._get_cross_encoder"),
    ):
        from src.api.main import app

        yield TestClient(app)


# ── tests ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client):
        """Health endpoint must always be reachable, no auth required."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_contains_version(self, client):
        resp = client.get("/health")
        assert "version" in resp.json()


class TestChat:
    @patch("src.api.main.get_rag_chain")
    @patch("src.api.main.get_chat_history", return_value=[])
    @patch("src.api.main.insert_application_logs")
    def test_chat_returns_query_response(self, mock_log, mock_hist, mock_chain, client):
        """POST /v1/chat should call the RAG chain and return an answer."""
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {"answer": "42"}
        mock_chain.return_value = mock_rag

        resp = client.post(
            "/v1/chat",
            json={"question": "What is the meaning of life?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "42"
        assert "session_id" in data

    @patch("src.api.main.get_rag_chain")
    @patch("src.api.main.get_chat_history", return_value=[])
    @patch("src.api.main.insert_application_logs")
    def test_chat_uses_provided_session_id(self, mock_log, mock_hist, mock_chain, client):
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {"answer": "yes"}
        mock_chain.return_value = mock_rag

        session = "test-session-abc"
        resp = client.post(
            "/v1/chat",
            json={"question": "Hello?", "session_id": session},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session


class TestUploadDoc:
    def test_unsupported_extension_returns_400(self, client):
        """Uploading a .exe must be rejected with 400."""
        resp = client.post(
            "/v1/upload-doc",
            files={"file": ("malware.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    @patch("src.api.main.process_document")
    def test_valid_pdf_queues_task(self, mock_task, client):
        """A valid .pdf should be saved and a Celery task queued."""
        fake_result = MagicMock()
        fake_result.id = "task-uuid-001"
        mock_task.delay.return_value = fake_result

        import io

        resp = client.post(
            "/v1/upload-doc",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4 content"), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"
        assert "task_id" in data


class TestListDocs:
    @patch(
        "src.api.main.get_all_documents",
        return_value=[{"id": 1, "filename": "doc.pdf", "upload_timestamp": "2026-01-01T00:00:00"}],
    )
    def test_list_docs_returns_documents(self, mock_docs, client):
        resp = client.get("/v1/list-docs")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestDeleteDoc:
    @patch("src.api.main.delete_doc_from_chroma", return_value=True)
    @patch("src.api.main.delete_document_record", return_value=True)
    def test_delete_success(self, mock_db, mock_chroma, client):
        resp = client.post("/v1/delete-doc", json={"file_id": 1})
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"].lower()

    @patch("src.api.main.delete_doc_from_chroma", return_value=False)
    def test_delete_chroma_failure_returns_500(self, mock_chroma, client):
        resp = client.post("/v1/delete-doc", json={"file_id": 99})
        assert resp.status_code == 500
