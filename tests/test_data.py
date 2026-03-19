# -*- coding: utf-8 -*-
"""Tests for data layer."""

import pytest
import pandas as pd
import numpy as np


def test_data_loader_summary(sample_ohlcv):
    from src.data.loader import DataLoader
    loader = DataLoader()
    summary = loader.data_summary(sample_ohlcv)
    assert "total_bars" in summary
    assert summary["total_bars"] == len(sample_ohlcv)
    assert "total_return_pct" in summary


def test_data_loader_get_returns(sample_ohlcv):
    from src.data.loader import DataLoader
    loader = DataLoader()
    returns = loader.get_returns(sample_ohlcv)
    assert len(returns) == len(sample_ohlcv)
    assert returns.name is None or returns.name == ""


def test_cache_manager():
    from src.data.cache import CacheManager
    cm = CacheManager(cache_dir="cache", ttl_days=7)
    cm.set({"test": 123}, prefix="unit_test")
    val = cm.get(prefix="unit_test")
    assert val is not None
    assert val["test"] == 123
    cm.clear("unit_test")
