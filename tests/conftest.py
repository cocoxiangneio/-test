# -*- coding: utf-8 -*-
"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="B")
    n = len(dates)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    open_p = close * (1 + np.random.randn(n) * 0.01)
    high = np.maximum(open_p, close) + np.random.rand(n) * 0.5
    low = np.minimum(open_p, close) - np.random.rand(n) * 0.5
    volume = np.random.randint(1000000, 10000000, n)

    df = pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    return df


@pytest.fixture
def sample_multi_ohlcv():
    dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="B")
    np.random.seed(42)
    result = {}
    for stock in ["000001.XSHE", "600036.XSHG", "600519.XSHG"]:
        close = 50 + np.cumsum(np.random.randn(len(dates)) * 1.5)
        result[stock] = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
    return result
