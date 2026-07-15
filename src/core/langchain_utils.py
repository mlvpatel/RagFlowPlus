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

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Add src to path to allow relative imports when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.chroma_utils import get_vectorstore
from retrieval.retrievers import ReRankingRetriever, VectorRetriever

logger = logging.getLogger(__name__)

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
    """Build the reranking retriever with lazily-loaded CrossEncoder.

    Base k=10 on purpose: the cross-encoder keeps top_n=5, and a reranker over
    exactly top_n candidates can only reorder, not rescue. Twice the final cut
    gives it real recall to work with.
    """
    return ReRankingRetriever(
        base_retriever=VectorRetriever(vectorstore=get_vectorstore(), k=10),
        cross_encoder_model=_get_cross_encoder(),
        top_n=5,
    )


# ============================================
# RAG chain factory
# ============================================
def _make_llm(model: str):
    if "claude" in model:
        return ChatAnthropic(model=model)
    if "deepseek" in model or "llama" in model:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=ollama_url)
    return ChatOpenAI(model=model)


def get_rag_chain(model: str = "gpt-4o-mini"):
    """
    Build a history aware RAG chain for the given model, using LCEL.

    Behaviour matches the classic retrieval chain: when chat history is present
    the question is first reformulated into a standalone query, then the
    reranking retriever fetches context and the model answers. Invoke with
    {"input": ..., "chat_history": [...]} and read the "answer" key.

    Supports OpenAI, Anthropic Claude, and local Ollama (deepseek, llama).
    """
    llm = _make_llm(model)
    final_retriever = _get_final_retriever()
    reformulate = contextualize_q_prompt | llm | StrOutputParser()
    answer_chain = qa_prompt | llm | StrOutputParser()

    def _run(inputs: dict) -> dict:
        history = inputs.get("chat_history") or []
        question = inputs["input"]
        search_query = (
            reformulate.invoke({"input": question, "chat_history": history})
            if history
            else question
        )
        docs = final_retriever.invoke(search_query)
        context = "\n\n".join(d.page_content for d in docs)
        answer = answer_chain.invoke(
            {"context": context, "chat_history": history, "input": question}
        )
        return {"answer": answer, "context": docs, "input": question}

    return RunnableLambda(_run)
