"""
cache_manager.py
Redis-based caching for query results and embeddings
"""
import redis
import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import timedelta
import structlog
import pickle

logger = structlog.get_logger()


class CacheManager:
    """Manages Redis caching for query results, embeddings, and frequently accessed data"""

    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,  # Use bytes for pickle compatibility
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("Redis cache initialized successfully", host=host, port=port)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("Redis not available, caching disabled", error=str(e))
            self.redis_client = None
            self.enabled = False

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from prefix and data"""
        # Create deterministic hash of the data
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return None

        try:
            value = self.redis_client.get(key)
            if value:
                logger.debug("Cache hit", key=key)
                return pickle.loads(value)
            logger.debug("Cache miss", key=key)
            return None
        except Exception as e:
            logger.error("Cache get failed", key=key, error=str(e))
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL (default 5 minutes)"""
        if not self.enabled:
            return False

        try:
            pickled_value = pickle.dumps(value)
            self.redis_client.setex(key, ttl, pickled_value)
            logger.debug("Cache set", key=key, ttl=ttl)
            return True
        except Exception as e:
            logger.error("Cache set failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.enabled:
            return False

        try:
            self.redis_client.delete(key)
            logger.debug("Cache deleted", key=key)
            return True
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.enabled:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info("Cache pattern cleared", pattern=pattern, count=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.error("Cache clear pattern failed", pattern=pattern, error=str(e))
            return 0

    # ============================================================================
    # Specialized Caching Methods
    # ============================================================================

    def get_query_result(self, query: str, query_type: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        key = self._generate_key(f"query:{query_type}", query)
        return self.get(key)

    def set_query_result(self, query: str, query_type: str, result: Dict[str, Any],
                        ttl: int = 300) -> bool:
        """Cache query result (default 5 min)"""
        key = self._generate_key(f"query:{query_type}", query)
        return self.set(key, result, ttl)

    def get_sql_result(self, sql_query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached SQL query result"""
        key = self._generate_key("sql", sql_query)
        return self.get(key)

    def set_sql_result(self, sql_query: str, result: List[Dict[str, Any]],
                      ttl: int = 600) -> bool:
        """Cache SQL result (default 10 min)"""
        key = self._generate_key("sql", sql_query)
        return self.set(key, result, ttl)

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding vector"""
        key = self._generate_key("embedding", text)
        return self.get(key)

    def set_embedding(self, text: str, embedding: List[float],
                     ttl: int = 86400) -> bool:
        """Cache embedding (default 24 hours)"""
        key = self._generate_key("embedding", text)
        return self.set(key, embedding, ttl)

    def get_vector_search(self, query: str, filters: Dict = None) -> Optional[List[Dict]]:
        """Get cached vector search result"""
        cache_data = {"query": query, "filters": filters or {}}
        key = self._generate_key("vector", cache_data)
        return self.get(key)

    def set_vector_search(self, query: str, filters: Dict, result: List[Dict],
                         ttl: int = 300) -> bool:
        """Cache vector search result (default 5 min)"""
        cache_data = {"query": query, "filters": filters or {}}
        key = self._generate_key("vector", cache_data)
        return self.set(key, result, ttl)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}

        try:
            info = self.redis_client.info('stats')
            return {
                "enabled": True,
                "total_keys": self.redis_client.dbsize(),
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "hit_rate": info.get('keyspace_hits', 0) /
                           (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1)),
                "memory_used": info.get('used_memory_human', 'N/A')
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"enabled": True, "error": str(e)}


# Global cache manager instance
cache_manager = CacheManager()
