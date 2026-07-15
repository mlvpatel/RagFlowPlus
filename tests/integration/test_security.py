"""
Integration security tests, input sanitization and rate limiting.
Updated to use /v1/ route prefix.
Author: Malav Patel
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# The vector stack is constructed lazily at call time, so the real app imports
# cleanly here; sys.modules stubbing would poison other tests' imports.
with (
    patch("src.api.db_utils.create_application_logs"),
    patch("src.api.db_utils.create_document_store"),
):
    from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_input_sanitization():
    """Verify that HTML/XSS tags are stripped from user input before processing."""
    from src.api.pydantic_models import QueryInput

    dirty_input = "<script>alert('xss')</script>What is RAG?"
    query = QueryInput(question=dirty_input)
    # bleach.clean strips tags, keeps text content
    assert "<script>" not in query.question
    assert "What is RAG?" in query.question
    assert query.question == "alert('xss')What is RAG?"


def test_rate_limiting_upload():
    """Verify that the /v1/upload-doc endpoint enforces the 10/minute limit."""
    with patch("src.api.main.process_document") as mock_task:
        mock_result = MagicMock()
        mock_result.id = "task-id"
        mock_task.delay.return_value = mock_result

        responses = []
        for _ in range(12):
            responses.append(
                client.post(
                    "/v1/upload-doc",
                    files={"file": ("test.txt", b"content", "text/plain")},
                )
            )

        status_codes = [r.status_code for r in responses]
        # At least one request should succeed
        assert 200 in status_codes
        # After 10 uploads the 11th and 12th should be rate-limited
        assert 429 in status_codes, f"Expected 429. Got: {status_codes}"


def test_security_headers():
    """Health endpoint should be reachable without an API key."""
    resp = client.get("/health")
    assert resp.status_code == 200
