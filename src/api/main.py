"""
rag-advanced-2023 API Server, v1

Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import logging
import os
import shutil
import uuid

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.db_utils import (
    delete_document_record,
    get_all_documents,
    get_chat_history,
    insert_application_logs,
    insert_document_record,
)
from src.api.pydantic_models import DeleteFileRequest, DocumentInfo, QueryInput, QueryResponse
from src.core.langchain_utils import get_rag_chain
from src.core.logging_config import configure_logging, logger
from src.core.security import limiter, verify_api_key
from src.embeddings.chroma_utils import delete_doc_from_chroma, index_document_to_chroma
from src.worker.celery_app import celery_app
from src.worker.tasks import process_document

configure_logging()

# ============================================
# App & Middleware
# ============================================
app = FastAPI(
    title="rag-advanced-2023 API",
    description="Production-grade RAG System with Conversational Memory",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS, configure allowed origins via ALLOWED_ORIGINS env var
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ============================================
# Request logging middleware
# ============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time

    trace_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    start = time.time()
    response = await call_next(request)
    logger.info(
        "request_processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round((time.time() - start) * 1000, 2),
    )
    return response


# ============================================
# Unversioned routes (health / metrics)
# ============================================
@app.get("/health", tags=["System"])
def health_check():
    """Deep liveness probe, checks Chroma and Redis connectivity."""
    import redis as redis_lib

    from src.embeddings.chroma_utils import vectorstore

    checks: dict[str, str] = {}

    # ── Chroma ────────────────────────────────────────────────────────────
    try:
        vectorstore.get(limit=1)  # lightweight probe
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = f"error: {e}"

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    try:
        r = redis_lib.from_url(redis_url, socket_connect_timeout=2)
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    overall = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "version": "1.0.0", "dependencies": checks}


# ============================================
# v1 Router, all business routes
# ============================================
v1 = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])


@v1.post("/chat", response_model=QueryResponse, tags=["Chat"])
@limiter.limit("60/minute")
def chat(request: Request, query_input: QueryInput):
    """Chat with indexed documents using RAG + conversational memory."""
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info("chat_request", session_id=session_id, model=query_input.model.value)

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke(
        {
            "input": query_input.question,
            "chat_history": chat_history,
        }
    )["answer"]

    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logger.info("chat_response", session_id=session_id, answer_length=len(answer))
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)


@v1.post("/upload-doc", tags=["Documents"])
@limiter.limit("10/minute")
async def upload_and_index_document(request: Request, file: UploadFile = File(...)):
    """Upload and asynchronously index a document via Celery worker."""
    allowed_extensions = {".pdf", ".docx", ".html", ".txt", ".md"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_extension}'. Allowed: {sorted(allowed_extensions)}",
        )

    upload_dir = "/app/data/uploads" if os.getenv("CHROMA_HOST") else "temp_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task = process_document.delay(file_path, file.filename)
        logger.info("upload_queued", filename=file.filename, task_id=task.id)

        return {
            "task_id": task.id,
            "status": "processing",
            "message": f"File '{file.filename}' uploaded and queued for indexing.",
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@v1.get("/task/{task_id}", tags=["Documents"])
def get_task_status(task_id: str):
    """Poll the status of an async document indexing task."""
    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }


@v1.get("/list-docs", tags=["Documents"])
def list_documents():
    """List all indexed documents."""
    return get_all_documents()


@v1.post("/delete-doc", tags=["Documents"])
def delete_document(request: DeleteFileRequest):
    """Delete a document from both ChromaDB and the metadata database."""
    chroma_ok = delete_doc_from_chroma(request.file_id)
    if not chroma_ok:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete vectors for file_id {request.file_id}"
        )

    db_ok = delete_document_record(request.file_id)
    if not db_ok:
        raise HTTPException(
            status_code=500,
            detail="Vectors deleted from Chroma but database record removal failed.",
        )

    return {"message": f"Successfully deleted document {request.file_id}"}


# Mount the v1 router
app.include_router(v1)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
