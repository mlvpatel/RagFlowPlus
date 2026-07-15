from langchain_core.documents import Document

from src.retrieval.retrievers import VectorRetriever, reciprocal_rank_fusion


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


def test_reciprocal_rank_fusion_agreement_wins():
    """A document ranked well by BOTH lists must beat one ranked first by only one."""
    both = Document(page_content="both")
    dense_only = Document(page_content="dense only")
    sparse_only = Document(page_content="sparse only")
    fused = reciprocal_rank_fusion([[dense_only, both], [sparse_only, both]])
    # both: 1/62 + 1/62; dense_only and sparse_only: 1/61 each
    assert fused[0].page_content == "both"


def test_vector_retriever_dense_fallback(mock_chroma):
    """With no corpus for BM25, the retriever returns the dense results, and it
    over-fetches k*2 dense candidates so downstream fusion has recall to use."""
    mock_chroma.get.return_value = {"documents": [], "metadatas": []}
    retriever = VectorRetriever(vectorstore=mock_chroma, k=2)
    results = retriever.invoke("test query")

    assert len(results) == 2
    assert results[0].page_content == "Test doc 1"
    mock_chroma.similarity_search.assert_called_once_with("test query", k=4)


def test_vector_retriever_hybrid_rrf(mock_chroma):
    """A keyword-only match that dense search misses must surface via BM25+RRF."""
    dense = [
        Document(page_content="general text about fruit"),
        Document(page_content="more text about food"),
    ]
    corpus = {
        "documents": [
            "general text about fruit",
            "more text about food",
            "warranty voucher code QX7",
        ],
        "metadatas": [{}, {}, {}],
    }
    mock_chroma.similarity_search.return_value = dense
    mock_chroma.get.return_value = corpus

    retriever = VectorRetriever(vectorstore=mock_chroma, k=3)
    results = retriever.invoke("warranty voucher QX7")

    contents = [d.page_content for d in results]
    assert "warranty voucher code QX7" in contents
