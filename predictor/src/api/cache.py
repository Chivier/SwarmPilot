"""Thread-safe LRU cache for loaded prediction models."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from src.utils.logging import get_logger

logger = get_logger()


class ModelCache:
    """Thread-safe LRU cache for loaded prediction models.

    Caches predictor instances to avoid reloading models from disk
    on every prediction. Uses OrderedDict for LRU eviction policy.

    Attributes:
        max_size: Maximum number of models to cache.
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize model cache.

        Args:
            max_size: Maximum number of models to cache.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, model_key: str) -> tuple[Any, str] | None:
        """Get cached predictor for model_key.

        Args:
            model_key: The model identifier.

        Returns:
            Tuple of (predictor, prediction_type) if cached, None otherwise.
        """
        with self._lock:
            if model_key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(model_key)
                self._hits += 1
                predictor, pred_type = self._cache[model_key]
                logger.debug(
                    f"Cache hit for model_key={model_key} "
                    f"(hits={self._hits}, misses={self._misses})"
                )
                return (predictor, pred_type)
            else:
                self._misses += 1
                logger.debug(
                    f"Cache miss for model_key={model_key} "
                    f"(hits={self._hits}, misses={self._misses})"
                )
                return None

    def put(self, model_key: str, predictor: Any, prediction_type: str) -> None:
        """Cache a predictor instance.

        Args:
            model_key: The model identifier.
            predictor: The predictor instance to cache.
            prediction_type: Type of prediction (expect_error or quantile).
        """
        with self._lock:
            # If already exists, move to end
            if model_key in self._cache:
                self._cache.move_to_end(model_key)

            self._cache[model_key] = (predictor, prediction_type)

            # Evict oldest if cache is full
            if len(self._cache) > self.max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                logger.debug(
                    f"Evicted model_key={evicted_key} from cache "
                    f"(size={len(self._cache)})"
                )

            logger.debug(
                f"Cached model_key={model_key}, cache size={len(self._cache)}"
            )

    def invalidate(self, model_key: str) -> None:
        """Remove a model from cache (e.g., after retraining).

        Args:
            model_key: The model identifier to remove.
        """
        with self._lock:
            if model_key in self._cache:
                del self._cache[model_key]
                logger.debug(f"Invalidated model_key={model_key} from cache")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cleared model cache")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including hit rate.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 2),
            }
