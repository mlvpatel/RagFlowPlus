"""
Document chunking strategies.
Author: Malav Patel
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartChunker:
    """
    Document chunker using RecursiveCharacterTextSplitter with configurable
    chunk size and overlap.

    Planned enhancement: when an embedding_model is provided, use cosine
    similarity between sentence embeddings to find natural semantic boundaries
    before splitting. For now, the recursive strategy is production-safe and
    well-tested.

    Rename history: was SemanticChunker, renamed SmartChunker to avoid
    implying semantic boundaries are computed (they are not yet).
    """

    def __init__(
        self,
        embedding_model=None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        threshold: float = 0.8,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        When embedding_model is provided (future): merge sentences with
        cosine similarity > threshold into a single chunk.
        """
        # TODO: implement sentence-level semantic boundary detection
        # using self.embedding_model and self.threshold
        return self.base_splitter.split_documents(documents)
