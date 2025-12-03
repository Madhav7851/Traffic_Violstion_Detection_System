import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RedisService:
    """Minimal Redis-like service for duplicate detection and caching.

    This implementation uses a local dict with TTL to provide basic behavior
    without requiring a running Redis server. If you have `redis` installed
    and a server available, you can extend this class to use that client.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        # store -> key: (value, expiry_datetime)
        self._store = {}
        logger.info("RedisService scaffold initialized (in-memory TTL cache)")

    def _cleanup(self):
        now = datetime.utcnow()
        expired = [k for k, (_, exp) in self._store.items() if exp is not None and exp <= now]
        for k in expired:
            del self._store[k]

    def is_duplicate(self, key: str) -> bool:
        self._cleanup()
        return key in self._store

    def mark_seen(self, key: str, ttl: int = 300) -> bool:
        expiry = datetime.utcnow() + timedelta(seconds=ttl) if ttl is not None else None
        self._store[key] = ('1', expiry)
        return True

    def get_duplicate_ttl(self, key: str) -> int:
        self._cleanup()
        v = self._store.get(key)
        if not v:
            return 0
        _, exp = v
        if exp is None:
            return -1
        remaining = int((exp - datetime.utcnow()).total_seconds())
        return max(0, remaining)