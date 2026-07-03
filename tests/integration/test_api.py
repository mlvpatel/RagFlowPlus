"""
Integration tests for the FastAPI v1 endpoints.
These tests import the real app but mock all external services
(Chroma, DB, Celery) so they run in CI without infrastructure.

Author: Malav Patel
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Module-scoped client, patches applied for all tests in this file."""
    with (
        patch("src.api.db_utils.create_application_logs"),
        patch("src.api.db_utils.create_document_store"),
        patch("src.embeddings.chroma_utils.GoogleGenerativeAIEmbeddings"),
        patch("src.embeddings.chroma_utils.Chroma"),
        patch("src.core.langchain_utils._base_retriever"),
        patch("src.core.langchain_utils._get_cross_encoder"),
    ):
        from src.api.main import app

        yield TestClient(app)


@pytest.fixture
def mock_upload_file(tmp_path):
    """Create a temporary dummy PDF file."""
    file_path = tmp_path / "test_doc.pdf"
    file_path.write_bytes(b"%PDF-1.4 mock pdf content")
    return file_path


def test_health_check(client):
    """Test /health endpoint, no auth required."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch("src.api.main.get_chat_history", return_value=[])
@patch("src.api.main.get_rag_chain")
@patch("src.api.main.insert_application_logs")
def test_chat_endpoint(mock_logs, mock_chain, mock_history, client):
    """Test /v1/chat endpoint with mocked RAG chain."""
    mock_rag_response = {"answer": "This is a test answer"}
    mock_chain_instance = MagicMock()
    mock_chain_instance.invoke.return_value = mock_rag_response
    mock_chain.return_value = mock_chain_instance

    payload = {
        "question": "What is RAG?",
        "session_id": "test-session",
        "model": "gpt-4o-mini",
    }

    response = client.post("/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["answer"] == "This is a test answer"
    mock_chain_instance.invoke.assert_called_once()


@patch("src.api.main.process_document")
def test_upload_endpoint(mock_task_cls, mock_upload_file, client):
    """Test /v1/upload-doc endpoint (async Celery path)."""
    mock_task_result = MagicMock()
    mock_task_result.id = "test-task-id"
    mock_task_cls.delay.return_value = mock_task_result

    with open(mock_upload_file, "rb") as f:
        response = client.post(
            "/v1/upload-doc",
            files={"file": ("test_doc.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["task_id"] == "test-task-id"
    assert response.json()["status"] == "processing"
    mock_task_cls.delay.assert_called_once()
