# rag-advanced-2023 Dockerfile
# Author: Malav Patel
# Multi-stage build for production deployment

# ============================================
# Stage 1: Base
# ============================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

# ============================================
# Stage 2: Builder
# ============================================
FROM base AS builder

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip wheel setuptools \
    && pip install -r requirements.txt

# ============================================
# Stage 3: Production
# ============================================
FROM base AS production

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY --chown=app:app . .

RUN mkdir -p /app/data/chroma_db \
             /app/data/uploads \
             /app/logs \
    && chown -R app:app /app

USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
