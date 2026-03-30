"""Cache module for Redis-based caching."""

from .redis_client import RedisCache, get_redis_client

__all__ = ["get_redis_client", "RedisCache"]
