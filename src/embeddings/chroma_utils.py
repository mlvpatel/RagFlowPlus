"""
Embedding utilities, ChromaDB + Google Gemini Embedding 2 (text-embedding-004).
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import datetime
import logging
import os
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ============================================
# Text splitter
# ============================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    length_function=len,
)


# ============================================
# Gemini Embedding 2, text-embedding-004
# Uses different task_types for index vs query for best accuracy.
# Indexing task_type is set here; the retriever uses "retrieval_query".
# ============================================
# Provider switch: Google Gemini by default, or local Ollama for a keyless run
# (set EMBEDDING_PROVIDER=ollama). The retrieval technique is unchanged, this
# only swaps where the vectors come from.
def _build_embeddings(task_type: str):
    provider = os.getenv("EMBEDDING_PROVIDER", "google").lower()
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY is not set. Google embedding calls will fail.")
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type=task_type,
    )


embedding_function = _build_embeddings("retrieval_document")
# Query-time embedding (different task type for better retrieval accuracy)
query_embedding_function = _build_embeddings("retrieval_query")

# ============================================
# ChromaDB, HTTP client-server (Docker) or local persistence (dev)
# ============================================
chroma_host = os.getenv("CHROMA_HOST")
chroma_port = os.getenv("CHROMA_PORT", "8000")

if chroma_host:
    import chromadb

    client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
    vectorstore = Chroma(
        client=client,
        collection_name="rag_naive_collection",
        embedding_function=embedding_function,
    )
    logger.info(f"ChromaDB connected to HTTP server: {chroma_host}:{chroma_port}")
else:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_function,
    )
    logger.info(f"ChromaDB using local persistence: {persist_dir}")


# ============================================
# Document loading helpers
# ============================================
def load_and_split_document(file_path: str) -> List[Document]:
    """Load and chunk a document by file type."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.endswith((".txt", ".md")):
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """Embed and store document chunks in Chroma with rich metadata."""
    try:
        splits = load_and_split_document(file_path)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        for split in splits:
            split.metadata["file_id"] = file_id
            split.metadata["filename"] = os.path.basename(file_path)
            split.metadata["indexed_at"] = timestamp

        vectorstore.add_documents(splits)
        logger.info(
            f"Indexed {len(splits)} chunks for file_id={file_id} ({os.path.basename(file_path)})"
        )
        return True
    except Exception as e:
        logger.error(f"Error indexing document {file_path}: {e}")
        return False


def delete_doc_from_chroma(file_id: int) -> bool:
    """Delete all Chroma chunks associated with a file_id using the public API."""
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        ids_to_delete = docs["ids"]
        logger.info(f"Found {len(ids_to_delete)} chunks for file_id {file_id}")

        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)  # ✅ stable public API
            logger.info(f"Deleted {len(ids_to_delete)} chunks for file_id {file_id}")

        return True
    except Exception as e:
        logger.error(f"Error deleting file_id {file_id} from Chroma: {e}")
        return False
