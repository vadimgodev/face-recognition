"""Redis client wrapper with connection pooling and graceful fallback."""

import json
import hashlib
import logging
from typing import Optional, Any, Union
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from src.config.settings import settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis cache client with graceful fallback."""

    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._enabled = settings.redis_enabled

    async def initialize(self):
        """Initialize Redis connection pool."""
        if not self._enabled:
            logger.warning("Redis is disabled in settings")
            return

        try:
            self._pool = ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Operating without cache.")
            self._enabled = False
            self._client = None

    async def close(self):
        """Close Redis connection pool."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()

    def _is_available(self) -> bool:
        """Check if Redis is available."""
        return self._enabled and self._client is not None

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if not self._is_available():
            return None

        try:
            value = await self._client.get(key)
            return value
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None
    ) -> bool:
        """Set value in Redis with optional expiration."""
        if not self._is_available():
            return False

        try:
            await self._client.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set value with expiration time."""
        return await self.set(key, value, ex=seconds)

    async def delete(self, *keys: str) -> bool:
        """Delete one or more keys."""
        if not self._is_available():
            return False

        try:
            await self._client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error for keys {keys}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._is_available():
            return False

        try:
            result = await self._client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis."""
        value = await self.get(key)
        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for key {key}: {e}")
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        ex: Optional[int] = None
    ) -> bool:
        """Set JSON value in Redis."""
        try:
            json_str = json.dumps(value)
            return await self.set(key, json_str, ex=ex)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to encode JSON for key {key}: {e}")
            return False

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Generate SHA256 hash of bytes data."""
        return hashlib.sha256(data).hexdigest()

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self._is_available():
            return 0

        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self._client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} keys matching pattern: {pattern}")
                return len(keys)
            return 0
        except Exception as e:
            logger.error(f"Redis pattern invalidation error for {pattern}: {e}")
            return 0


# Global Redis client instance
_redis_cache: Optional[RedisCache] = None


def get_redis_client() -> RedisCache:
    """Get global Redis cache instance."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache
