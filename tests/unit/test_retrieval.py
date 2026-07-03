from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.retrieval.chunking import SmartChunker
from src.retrieval.retrievers import VectorRetriever, reciprocal_rank_fusion


def test_smart_chunker_fallback():
    """Test SmartChunker falls back to recursive splitter when no embedding model provided."""
    chunker = SmartChunker()
    docs = [Document(page_content="A" * 2000)]
    splits = chunker.split_documents(docs)
    assert len(splits) > 1
    assert len(splits[0].page_content) <= 1000


def test_reciprocal_rank_fusion():
    """Test RRF math."""
    list1 = [Document(page_content="A"), Document(page_content="B")]
    list2 = [Document(page_content="B"), Document(page_content="A")]

    # A: rank 0 in list1, rank 1 in list2   to score = 1/61 + 1/62
    # B: rank 1 in list1, rank 0 in list2   to score = 1/62 + 1/61
    # Both equal, result order is non-deterministic
    fused = reciprocal_rank_fusion([list1, list2])
    assert len(fused) == 2
    assert fused[0].page_content in ["A", "B"]


def test_vector_retriever(mock_chroma):
    """Test VectorRetriever delegates to vectorstore.similarity_search correctly."""
    retriever = VectorRetriever(vectorstore=mock_chroma, k=2)
    results = retriever.invoke("test query")

    assert len(results) == 2
    assert results[0].page_content == "Test doc 1"
    mock_chroma.similarity_search.assert_called_once_with("test query", k=2)
