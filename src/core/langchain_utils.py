"""
LangChain RAG chain builder with lazy CrossEncoder loading.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import functools
import logging
import os
import sys

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Add src to path to allow relative imports when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.chroma_utils import query_embedding_function, vectorstore
from retrieval.retrievers import ReRankingRetriever, VectorRetriever

logger = logging.getLogger(__name__)

# ============================================
# Base retriever, vector similarity (k=5 for good recall)
# ============================================
_base_retriever = VectorRetriever(vectorstore=vectorstore, k=5)

output_parser = StrOutputParser()

# ============================================
# Prompt templates
# ============================================
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful AI assistant. Use the following retrieved context to answer "
                "the user's question accurately and concisely. If the context doesn't contain "
                "enough information, say so rather than guessing.\n\nContext:\n{context}"
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


# ============================================
# Lazy CrossEncoder, downloaded only on first real call
# ============================================
@functools.lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the CrossEncoder model once and cache it (lazy, ~90MB download)."""
    logger.info("Loading CrossEncoder model (first call only)...")
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("CrossEncoder model loaded.")
    return model


def _get_final_retriever() -> ReRankingRetriever:
    """Build the reranking retriever with lazily-loaded CrossEncoder."""
    return ReRankingRetriever(
        base_retriever=_base_retriever,
        cross_encoder_model=_get_cross_encoder(),
        top_n=5,
    )


# ============================================
# RAG chain factory
# ============================================
def get_rag_chain(model: str = "gpt-4o-mini"):
    """
    Build and return a history-aware RAG chain for the given model.

    Supports:
      - OpenAI (gpt-4o, gpt-4o-mini, etc.)
      - Anthropic Claude (claude-3-5-sonnet-*)
      - Local via Ollama (deepseek-r1, llama3, etc.)
    """
    if "claude" in model:
        llm = ChatAnthropic(model=model)
    elif "deepseek" in model or "llama" in model:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(model=model, base_url=ollama_url)
    else:
        llm = ChatOpenAI(model=model)

    final_retriever = _get_final_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, final_retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
