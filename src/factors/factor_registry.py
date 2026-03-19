# -*- coding: utf-8 -*-
"""Factor registry - central lookup for all factors."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FactorRegistry:
    _instance: Optional["FactorRegistry"] = None

    def __init__(self):
        self._factors: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> "FactorRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        func: Callable,
        category: str = "technical",
        description: str = "",
        window_required: Optional[int] = None,
    ) -> None:
        self._factors[name] = func
        self._metadata[name] = {
            "category": category,
            "description": description,
            "window_required": window_required,
            "func": func,
        }
        logger.debug(f"Registered factor: {name} ({category})")

    def get(self, name: str) -> Optional[Callable]:
        return self._factors.get(name)

    def list_factors(self, category: Optional[str] = None) -> List[str]:
        if category:
            return [n for n, m in self._metadata.items() if m["category"] == category]
        return list(self._factors.keys())

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {k: v for k, v in meta.items() if k != "func"}
            for name, meta in self._metadata.items()
        }

    def calculate(self, name: str, df: pd.DataFrame, **kwargs) -> pd.Series:
        func = self.get(name)
        if func is None:
            raise ValueError(f"Factor '{name}' not found in registry")
        return func(df, **kwargs)

    def batch_calculate(
        self,
        names: List[str],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for name in names:
            try:
                result[name] = self.calculate(name, df)
            except Exception as e:
                logger.warning(f"Factor {name} calculation failed: {e}")
                result[name] = np.nan
        return result


_global_registry = FactorRegistry()


def register_factor(
    name: str,
    category: str = "technical",
    description: str = "",
    window_required: Optional[int] = None,
):
    def decorator(func: Callable) -> Callable:
        _global_registry.register(name, func, category, description, window_required)
        return func
    return decorator


def get_registry() -> FactorRegistry:
    return _global_registry
