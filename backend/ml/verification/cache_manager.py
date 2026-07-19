import os
import json
import time
import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger("verification_pipeline")

class CacheManager:
    """
    CacheManager handles storage and retrieval of external API query responses to avoid duplicate calls.
    Uses file-based JSON storage with configurable Time-To-Live (TTL).
    """
    def __init__(self, cache_dir: str = "ml/verification/cache", ttl_seconds: int = 3600) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, provider: str, query: str) -> str:
        # Create a unique filename based on provider name and hash of query
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{provider}_{query_hash}.json")

    def get(self, provider: str, query: str) -> Optional[Any]:
        """
        Retrieves cached response if it exists and has not expired.
        """
        cache_path = self._get_cache_path(provider, query)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            timestamp = data.get("timestamp", 0.0)
            if time.time() - timestamp > self.ttl_seconds:
                logger.debug(f"Cache expired for provider '{provider}' and query '{query}'")
                self.delete(provider, query)
                return None
                
            logger.info(f"Cache hit for provider '{provider}' and query '{query}'")
            return data.get("response")
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            return None

    def set(self, provider: str, query: str, response: Any) -> None:
        """
        Caches a response with a current timestamp.
        """
        cache_path = self._get_cache_path(provider, query)
        try:
            data = {
                "timestamp": time.time(),
                "provider": provider,
                "query": query,
                "response": response
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cached response written to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_path}: {e}")

    def delete(self, provider: str, query: str) -> None:
        """Deletes a specific cached file."""
        cache_path = self._get_cache_path(provider, query)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                logger.warning(f"Could not delete cache file {cache_path}: {e}")

    def clear(self) -> None:
        """Clears all cached files in the cache directory."""
        if not os.path.exists(self.cache_dir):
            return
        for file in os.listdir(self.cache_dir):
            if file.startswith("cache_") and file.endswith(".json"):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    pass
