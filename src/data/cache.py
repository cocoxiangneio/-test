# -*- coding: utf-8 -*-
"""Cache management utilities."""

import os
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_dir: str = "cache", ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.ttl_days = ttl_days
        os.makedirs(cache_dir, exist_ok=True)

    def _make_key(self, prefix: str, **kwargs) -> str:
        raw = prefix + "_" + "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key = hashlib.md5(raw.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prefix}_{key}.pkl")

    def get(self, prefix: str, **kwargs) -> Optional[Any]:
        path = self._make_key(prefix, **kwargs)
        if not os.path.exists(path):
            return None
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
        if age_days > self.ttl_days:
            os.remove(path)
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None

    def set(self, value: Any, prefix: str, **kwargs) -> None:
        path = self._make_key(prefix, **kwargs)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def memoize(self, prefix: str, ttl_days: Optional[int] = None):
        def decorator(func: Callable) -> Callable:
            _ttl = ttl_days or self.ttl_days

            def wrapper(*args, **kwargs):
                cache_key = hashlib.md5(
                    str(args).encode() + str(kwargs).encode()
                ).hexdigest()
                path = os.path.join(self.cache_dir, f"{func.__name__}_{cache_key}.pkl")

                if os.path.exists(path):
                    age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
                    if age_days < _ttl:
                        with open(path, "rb") as f:
                            return pickle.load(f)

                result = func(*args, **kwargs)
                try:
                    with open(path, "wb") as f:
                        pickle.dump(result, f)
                except Exception as e:
                    logger.warning(f"memoize write failed: {e}")
                return result

            return wrapper

        return decorator

    def clear(self, prefix: Optional[str] = None):
        if prefix:
            for fname in os.listdir(self.cache_dir):
                if fname.startswith(prefix):
                    os.remove(os.path.join(self.cache_dir, fname))
        else:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, fname))
