# rag-advanced-2023

**Advanced retrieval augmented generation for question answering over your own documents, with hybrid retrieval and cross encoder reranking.**

**Part of the RAG line, a series of reference enterprise RAG implementations. This repository is rag-advanced-2023, Advanced RAG.** See [the full line](#the-rag_naive-line) below.

rag-advanced-2023 lets you upload documents and ask questions about them. It retrieves the relevant passages with hybrid dense and keyword search, reranks them with a cross encoder, sends the best context to a language model, and returns a grounded answer while remembering the conversation. It runs on FastAPI with a Streamlit chat interface and packages the whole stack with Docker.

[![CI](https://github.com/mlvpatel/rag-advanced-2023/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/mlvpatel/rag-advanced-2023/actions/workflows/ci-cd.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-async-green) ![License](https://img.shields.io/badge/license-MIT-green)

![rag-advanced-2023 answering a document question live on a local model](assets/videos/rag_advanced-demo.gif)

The clip above is a live, unedited run on a local llama3.2 model with the bundled sample data, including a real SEC 10-K, indexed in Chroma. No paid keys were used. Full recording at [assets/videos/rag_advanced-demo.webm](assets/videos/rag_advanced-demo.webm), screenshot at [assets/screenshots/rag_advanced-ui.png](assets/screenshots/rag_advanced-ui.png).

## Try it with the bundled sample data

The repo ships with sample documents in [sample_data](sample_data), an HR handbook, a product FAQ, and a real SEC 10-K excerpt, so you can run and judge the system without your own files. Fully local and keyless with Ollama:

```bash
ollama serve &
ollama pull nomic-embed-text
ollama pull llama3.2:3b
EMBEDDING_PROVIDER=ollama python scripts/load_sample_data.py
```

Then start the API and UI and ask the questions in [sample_data/README.md](sample_data/README.md), including an honesty check where it should decline rather than guess.

## Features

| Area | Capability |
|---|---|
| Documents | PDF, DOCX, HTML, TXT, Markdown |
| Embeddings | Google text-embedding-004, or local Ollama nomic-embed-text |
| Retrieval | Dense ChromaDB search plus BM25, fused with Reciprocal Rank Fusion |
| Reranking | Cross encoder ms-marco-MiniLM-L-6-v2 on top of the fused results |
| Memory | Multi turn conversations stored in SQLite |
| Models | OpenAI, Anthropic, or local models via Ollama, chosen by model name |
| Async indexing | Celery worker backed by Redis, so uploads do not block |
| Security | API key auth, rate limiting, HTML input sanitization, CORS |
| Observability | Prometheus metrics at /metrics, structured logging, a health probe |
| Packaging | Docker Compose for the full stack, unit and integration tests, CI |

## Architecture

```mermaid
graph TD
    User[User in browser] --> UI[Streamlit chat UI]
    UI --> API[FastAPI backend]
    API --> Chain[RAG chain]
    API --> Memory[(SQLite conversation memory)]
    Chain --> Retriever[Hybrid retriever plus reranker]
    Retriever --> Chroma[(ChromaDB vector store)]
    Chain --> LLM[LLM provider OpenAI, Anthropic, or Ollama]
    API --> Worker[Celery worker via Redis]
    Worker --> Chroma
```

## How to use

### Docker Compose (recommended)

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY, one LLM key, and a value for API_KEY
docker compose -f docker/docker-compose.yml up --build -d
open http://localhost:8501      # the chat UI
# API docs at http://localhost:8000/docs
```

Then in the UI: upload a document in the sidebar, wait for it to index, and ask a question. The answer comes back grounded in your document, and follow up questions use the earlier conversation.

### Local development

```bash
make install                # install dependencies
cp .env.example .env         # fill in your keys
docker run -d -p 6379:6379 redis:7-alpine   # Redis for the worker
make dev                    # API at http://localhost:8000
make worker                 # Celery worker, in a second terminal
make frontend               # Streamlit at http://localhost:8501, in a third terminal
```

## Configuration

Settings come from environment variables (see `.env.example`).

| Variable | Required | Meaning |
|---|---|---|
| GOOGLE_API_KEY | yes | Google AI Studio key for embeddings |
| API_KEY | yes | Sent by clients in the X-API-Key header |
| OPENAI_API_KEY | optional | For GPT models |
| ANTHROPIC_API_KEY | optional | For Claude models |
| OLLAMA_BASE_URL | optional | For local models, default http://localhost:11434 |
| CHROMA_HOST | optional | Set to use a ChromaDB server, otherwise local persistence |

## API reference

| Method and path | Purpose |
|---|---|
| GET /health | Liveness probe, no auth |
| POST /v1/chat | Ask a question, grounded answer with memory |
| POST /v1/upload-doc | Upload and asynchronously index a document |
| GET /v1/list-docs | List indexed documents |
| POST /v1/delete-doc | Delete a document and its vectors |
| GET /v1/task/{task_id} | Status of an async indexing task |
| GET /metrics | Prometheus metrics |

## Tech stack

Python, FastAPI, Streamlit, LangChain, ChromaDB, SQLite, Celery, Redis, Prometheus, and Docker. Embeddings use Google text-embedding-004.

## The RAG line

rag-advanced-2023 is one implementation in the RAG line, a series demonstrating distinct enterprise RAG retrieval strategies.

| Year | Repository | Generation |
|---|---|---|
| 2022 | [rag-naive-2022](https://github.com/mlvpatel/rag-naive-2022) | Naive RAG, single dense retrieval |
| 2023 | rag-advanced-2023, this repo | Advanced RAG, hybrid retrieval and reranking |
| 2024 | [rag-modular-2023](https://github.com/mlvpatel/rag-modular-2023) | Modular production RAG, pgvector, streaming, evaluation |
| 2025 | [rag-agentic-2025](https://github.com/mlvpatel/rag-agentic-2025), rag-graph-2024 | Agentic RAG, knowledge graph with reasoning |
| 2026 | [rag-multiagent-2026](https://github.com/mlvpatel/rag-multiagent-2026), rag-multimodal-2026 | Multi agent enterprise, multimodal |

The next implementation up, [rag-modular-2023](https://github.com/mlvpatel/rag-modular-2023), replaces ChromaDB with pgvector on Postgres, moves memory to Postgres, computes hybrid retrieval in a single SQL query, streams answers, and adds a measurable evaluation harness.

## License

Released under the MIT License. See [LICENSE](LICENSE). MIT is the simplest and most permissive of the common licenses, so anyone can read, run, modify, and reuse the code freely.

## Author

Malav Patel. GitHub @mlvpatel.
