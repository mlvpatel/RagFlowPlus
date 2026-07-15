"""
Unit tests for chroma_utils: loading, chunking, indexing, deletion, and the
lazy embedder accessors. All vector store IO is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.embeddings.chroma_utils as cu


@pytest.fixture(autouse=True)
def _clear_lazy_caches():
    """The lazy accessors cache their singletons; tests must not leak them."""
    yield
    cu.get_embedding_function.cache_clear()
    cu.get_query_embedding_function.cache_clear()
    cu.get_vectorstore.cache_clear()


def test_load_and_split_txt(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("word " * 800)
    chunks = cu.load_and_split_document(str(f))
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_load_unsupported_extension_raises(tmp_path):
    f = tmp_path / "doc.xyz"
    f.write_text("data")
    with pytest.raises(ValueError):
        cu.load_and_split_document(str(f))


def test_index_document_stamps_metadata(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("some content to index")
    store = MagicMock()
    with patch.object(cu, "get_vectorstore", return_value=store):
        ok = cu.index_document_to_chroma(str(f), file_id=7)

    assert ok is True
    splits = store.add_documents.call_args[0][0]
    assert all(s.metadata["file_id"] == 7 for s in splits)
    assert all(s.metadata["filename"] == "doc.txt" for s in splits)
    assert all("indexed_at" in s.metadata for s in splits)


def test_index_document_returns_false_on_store_error(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("content")
    store = MagicMock()
    store.add_documents.side_effect = RuntimeError("chroma down")
    with patch.object(cu, "get_vectorstore", return_value=store):
        assert cu.index_document_to_chroma(str(f), file_id=1) is False


def test_delete_doc_removes_matching_ids():
    store = MagicMock()
    store.get.return_value = {"ids": ["a", "b"]}
    with patch.object(cu, "get_vectorstore", return_value=store):
        assert cu.delete_doc_from_chroma(3) is True
    store.get.assert_called_once_with(where={"file_id": 3})
    store.delete.assert_called_once_with(ids=["a", "b"])


def test_delete_doc_returns_false_on_error():
    store = MagicMock()
    store.get.side_effect = RuntimeError("boom")
    with patch.object(cu, "get_vectorstore", return_value=store):
        assert cu.delete_doc_from_chroma(3) is False


def test_ollama_embedders_and_caching(monkeypatch):
    """With the ollama provider the embedders build keylessly and are cached."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    e1 = cu.get_embedding_function()
    e2 = cu.get_embedding_function()
    q = cu.get_query_embedding_function()
    assert e1 is e2
    assert type(e1).__name__ == "OllamaEmbeddings"
    assert type(q).__name__ == "OllamaEmbeddings"


def test_vectorstore_local_persistence(monkeypatch, tmp_path):
    """Without CHROMA_HOST the store uses local persistence at the given dir."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("CHROMA_HOST", raising=False)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    store = cu.get_vectorstore()
    assert store is cu.get_vectorstore()
