"""
Security utilities: rate limiting, API key verification.
Author: Malav Patel
"""

import os

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

# ============================================
# Rate Limiter, Redis-backed for multi-replica safety
# ============================================
redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Wire storage_uri so all replicas share one counter (falls back gracefully to
# in-memory when Redis is unavailable, e.g. during unit tests)
try:
    limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
except Exception:
    limiter = Limiter(key_func=get_remote_address)

# ============================================
# API Key Authentication
# ============================================
API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
_API_KEY = os.getenv("API_KEY", "")


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency that enforces the X-API-Key header.

    If API_KEY env var is not set (e.g. local dev), auth is skipped entirely
    so developers don't need a key. Set API_KEY in production.
    """
    if not _API_KEY:
        # No API_KEY configured, open access (dev mode)
        return "dev"
    if api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid X-API-Key header.",
        )
    return api_key
