"""
Unit tests for the RAG chain builder: model routing and the history-aware
reformulate-then-retrieve flow. The LLM and retriever are fakes.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import src.core.langchain_utils as lu


def test_make_llm_routes_by_name():
    with (
        patch.object(lu, "ChatAnthropic") as anthropic,
        patch.object(lu, "ChatOllama") as ollama,
        patch.object(lu, "ChatOpenAI") as openai,
    ):
        lu._make_llm("claude-opus-4-8")
        anthropic.assert_called_once()
        lu._make_llm("llama3.2:3b")
        ollama.assert_called_once()
        lu._make_llm("deepseek-r1")
        assert ollama.call_count == 2
        lu._make_llm("gpt-4o-mini")
        openai.assert_called_once()


def _fake_llm(reply: str):
    """A Runnable standing in for a chat model; StrOutputParser passes str through."""
    return RunnableLambda(lambda _msgs: reply)


def test_chain_without_history_retrieves_raw_question():
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="ctx")]
    with (
        patch.object(lu, "_make_llm", return_value=_fake_llm("final answer")),
        patch.object(lu, "_get_final_retriever", return_value=retriever),
    ):
        chain = lu.get_rag_chain("gpt-4o-mini")
        out = chain.invoke({"input": "what is RRF?", "chat_history": []})

    assert out["answer"] == "final answer"
    retriever.invoke.assert_called_once_with("what is RRF?")


def test_chain_with_history_retrieves_reformulated_query():
    """With history present the retriever must see the standalone rewrite,
    not the raw follow-up."""
    retriever = MagicMock()
    retriever.invoke.return_value = [Document(page_content="ctx")]
    history = [{"role": "human", "content": "tell me about BM25"}]
    with (
        patch.object(lu, "_make_llm", return_value=_fake_llm("standalone rewrite")),
        patch.object(lu, "_get_final_retriever", return_value=retriever),
    ):
        chain = lu.get_rag_chain("gpt-4o-mini")
        out = chain.invoke({"input": "and its parameters?", "chat_history": history})

    retriever.invoke.assert_called_once_with("standalone rewrite")
    assert out["input"] == "and its parameters?"


def test_final_retriever_funnel_is_wider_than_cut():
    """The reranker keeps top_n=5, so the base retriever must fetch more than 5."""
    with (
        patch.object(lu, "get_vectorstore", return_value=MagicMock()),
        patch.object(lu, "_get_cross_encoder", return_value=MagicMock()),
    ):
        rr = lu._get_final_retriever()
    assert rr.top_n == 5
    assert rr.base_retriever.k == 2 * rr.top_n
