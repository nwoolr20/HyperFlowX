"""Advanced caching layer for expensive computations in HyperFlowX.

This module provides sophisticated caching mechanisms including LRU cache,
persistent cache, and smart cache invalidation for computational results.
"""

import functools
import hashlib
import pickle
import os
import time
import threading
from typing import Dict, Any, Optional, Callable, Union, Tuple
from collections import OrderedDict
import json

from .monitoring import log_info, log_warning, get_logger


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache implementation."""

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize LRU cache.

        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.pop(key)
            elif len(self.cache) >= self.maxsize:
                # Remove least recently used
                self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "size": len(self.cache),
                "maxsize": self.maxsize,
            }


class PersistentCache:
    """Persistent cache that survives between program runs."""

    def __init__(
        self, cache_dir: str = ".hyperflowx_cache", max_age_hours: int = 24
    ) -> None:
        """Initialize persistent cache.

        Args:
            cache_dir: Directory to store cache files
            max_age_hours: Maximum age of cache entries in hours
        """
        self.cache_dir = cache_dir
        self.max_age_seconds = max_age_hours * 3600
        self.lock = threading.RLock()

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        self.logger = get_logger()

    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Use first 2 chars for subdirectory to avoid too many files in one dir
        subdir = key[:2]
        subdir_path = os.path.join(self.cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(subdir_path, f"{key}.cache")

    def _is_expired(self, filepath: str) -> bool:
        """Check if cache file is expired."""
        try:
            mtime = os.path.getmtime(filepath)
            return time.time() - mtime > self.max_age_seconds
        except OSError:
            return True

    def get(self, key: str) -> Optional[Any]:
        """Get item from persistent cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            cache_path = self._get_cache_path(key)

            if not os.path.exists(cache_path):
                return None

            if self._is_expired(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                return None

            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except (pickle.PickleError, OSError) as e:
                self.logger.warning(f"Failed to load cache entry {key}: {e}")
                return None

    def put(self, key: str, value: Any) -> None:
        """Put item in persistent cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            cache_path = self._get_cache_path(key)

            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)
            except (pickle.PickleError, OSError) as e:
                self.logger.warning(f"Failed to save cache entry {key}: {e}")

    def clear(self) -> None:
        """Clear all persistent cache entries."""
        with self.lock:
            try:
                import shutil

                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            except OSError as e:
                self.logger.warning(f"Failed to clear cache: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        removed = 0
        with self.lock:
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith(".cache"):
                        filepath = os.path.join(root, file)
                        if self._is_expired(filepath):
                            try:
                                os.remove(filepath)
                                removed += 1
                            except OSError:
                                pass

        self.logger.info(f"Cleaned up {removed} expired cache entries")
        return removed


class SmartCache:
    """Smart cache combining memory and persistent storage."""

    def __init__(
        self,
        memory_size: int = 64,
        persistent_cache_dir: str = ".hyperflowx_cache",
        max_age_hours: int = 24,
    ) -> None:
        """Initialize smart cache.

        Args:
            memory_size: Size of in-memory LRU cache
            persistent_cache_dir: Directory for persistent cache
            max_age_hours: Maximum age for persistent cache entries
        """
        self.memory_cache = LRUCache(memory_size)
        self.persistent_cache = PersistentCache(persistent_cache_dir, max_age_hours)
        self.logger = get_logger()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then persistent).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try persistent cache
        value = self.persistent_cache.get(key)
        if value is not None:
            # Store in memory cache for faster future access
            self.memory_cache.put(key, value)
            return value

        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in both memory and persistent cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.memory_cache.put(key, value)
        self.persistent_cache.put(key, value)

    def clear(self) -> None:
        """Clear both memory and persistent cache."""
        self.memory_cache.clear()
        self.persistent_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {"memory": self.memory_cache.stats(), "persistent_enabled": True}


# Global cache instance
_global_cache: Optional[SmartCache] = None


def get_cache() -> SmartCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache


def _make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Create a cache key from function arguments."""
    # Create a deterministic hash of arguments
    key_data = {"args": args, "kwargs": kwargs}

    # Handle numpy arrays specially
    processed_data = []
    for item in [args, kwargs]:
        if hasattr(item, "__dict__"):
            processed_data.append(str(item.__dict__))
        elif hasattr(item, "shape"):  # numpy array
            processed_data.append(f"array_shape_{item.shape}_dtype_{item.dtype}")
        else:
            processed_data.append(str(item))

    key_string = json.dumps(processed_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(
    cache_type: str = "smart",
    ttl_hours: Optional[int] = None,
    key_func: Optional[Callable] = None,
) -> Callable:
    """Decorator to cache function results.

    Args:
        cache_type: Type of cache ('memory', 'persistent', 'smart')
        ttl_hours: Time to live in hours (overrides cache default)
        key_func: Custom function to generate cache key

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        cache_instance = get_cache()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}_{_make_cache_key(*args, **kwargs)}"

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                log_info(f"Cache hit for {func.__name__}", cache_key=cache_key)
                return cached_result

            # Execute function and cache result
            log_info(
                f"Cache miss for {func.__name__}, computing...", cache_key=cache_key
            )
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)

            return result

        # Add cache management methods to function
        wrapper.cache_clear = lambda: cache_instance.clear()
        wrapper.cache_stats = lambda: cache_instance.stats()

        return wrapper

    return decorator


def clear_all_caches() -> None:
    """Clear all cached data."""
    cache = get_cache()
    cache.clear()
    log_info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches."""
    return get_cache().stats()


def cleanup_expired_cache() -> int:
    """Clean up expired cache entries."""
    cache = get_cache()
    return cache.persistent_cache.cleanup_expired()


# Convenience decorators for common use cases
def cache_expensive_computation(func: Callable) -> Callable:
    """Cache results of expensive computations."""
    return cached(cache_type="smart", ttl_hours=24)(func)


def cache_ml_model(func: Callable) -> Callable:
    """Cache machine learning model results."""
    return cached(cache_type="smart", ttl_hours=12)(func)


def cache_matrix_operation(func: Callable) -> Callable:
    """Cache matrix operation results."""
    return cached(cache_type="memory")(func)
