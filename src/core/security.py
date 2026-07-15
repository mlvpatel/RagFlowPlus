"""
Security utilities: rate limiting and input hygiene.
Author: Malav Patel

There is no API key auth on purpose. This is a reference service meant to run
on one machine: docker-compose binds every published port to the loopback
interface, so nothing is reachable from the network. A shipped default
credential would read as protection while being public knowledge. Put a real
gateway in front of the API before exposing it beyond localhost.
"""

import os

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
