"""
Retrieval components: VectorRetriever (with optional BM25), ReRankingRetriever, RRF fusion.
Author: Malav Patel
"""

import logging
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    results: List[List[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked document lists.
    Formula: score(d) = Σ 1 / (k + rank(d) + 1)
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for doc_list in results:
        for rank, doc in enumerate(doc_list):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    return [
        doc_map[key] for key, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]


def _build_bm25_index(documents: List[Document]):
    """Build a BM25Okapi index from a list of documents."""
    from rank_bm25 import BM25Okapi

    tokenised = [doc.page_content.lower().split() for doc in documents]
    return BM25Okapi(tokenised)


class VectorRetriever(BaseRetriever):
    """
    Retriever combining dense vector search (ChromaDB) with optional sparse
    BM25 keyword search, fused with Reciprocal Rank Fusion.

    When the vectorstore supports fetching all documents (small collections),
    a BM25 index is built on-the-fly and both result lists are merged via RRF.
    For large-scale BM25, replace with Elasticsearch/OpenSearch.
    """

    vectorstore: Any
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Dense retrieval
        vector_docs = self.vectorstore.similarity_search(query, k=self.k * 2)

        # 2. Sparse BM25 retrieval (over the same candidate set)
        try:
            all_docs_raw = self.vectorstore.get()
            if all_docs_raw and all_docs_raw.get("documents"):
                corpus_docs = [
                    Document(page_content=txt, metadata=meta)
                    for txt, meta in zip(
                        all_docs_raw["documents"],
                        all_docs_raw.get("metadatas", [{}] * len(all_docs_raw["documents"])),
                    )
                ]
                bm25 = _build_bm25_index(corpus_docs)
                tokens = query.lower().split()
                bm25_scores = bm25.get_scores(tokens)
                # Return top-k by BM25 score
                top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[: self.k * 2]
                bm25_docs = [corpus_docs[i] for i in top_indices if bm25_scores[i] > 0]

                if bm25_docs:
                    fused = reciprocal_rank_fusion([vector_docs, bm25_docs])
                    logger.debug(
                        f"Hybrid RRF: {len(vector_docs)} vector + {len(bm25_docs)} BM25 to {len(fused)} fused"
                    )
                    return fused[: self.k]
        except Exception as e:
            logger.warning(f"BM25 retrieval skipped (falling back to vector-only): {e}")

        return vector_docs[: self.k]


class ReRankingRetriever(BaseRetriever):
    """
    Two-stage retriever: high-recall base retrieval + Cross-Encoder reranking.

    Stage 1: Retrieve `top_n * 2` candidates via base_retriever (high recall).
    Stage 2: Score each (query, doc) pair with a Cross-Encoder; return top_n.
    """

    base_retriever: BaseRetriever
    cross_encoder_model: Optional[Any] = None
    top_n: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        initial_docs = self.base_retriever.invoke(query)

        if not self.cross_encoder_model or not initial_docs:
            return initial_docs[: self.top_n]

        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder_model.predict(pairs)

        sorted_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[: self.top_n]]
