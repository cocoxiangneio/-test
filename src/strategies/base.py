# -*- coding: utf-8 -*-
"""Base strategy class and interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    name: str = "BaseStrategy"

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {}

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            Series with same index as df:
              1.0  = long signal
             -1.0  = short signal
              0.0  = hold/neutral
        """
        pass

    def get_params(self) -> Dict:
        return self.params.copy()

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})" if params_str else self.name


class SignalTypes:
    LONG = 1.0
    SHORT = -1.0
    HOLD = 0.0
