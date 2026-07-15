from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB vector store."""
    mock = MagicMock()
    # default behavior for similarity search
    mock.similarity_search.return_value = [
        Document(page_content="Test doc 1", metadata={"id": 1}),
        Document(page_content="Test doc 2", metadata={"id": 2}),
    ]
    return mock


@pytest.fixture
def mock_documents():
    return [
        Document(page_content="Apple is a fruit", metadata={"category": "fruit"}),
        Document(page_content="Carrot is a vegetable", metadata={"category": "veg"}),
        Document(page_content="Banana is yellow", metadata={"category": "fruit"}),
    ]


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The limiter is Redis-backed, so its counters outlive a test (and even a
    run). Without this, any test that fires enough requests starves every
    later test of the same endpoint with 429s."""
    from src.core.security import limiter

    try:
        limiter.reset()
    except Exception:
        pass
    yield
