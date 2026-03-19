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


def test_load_csv(sample_ohlcv):
    from src.data.loader import DataLoader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        sample_ohlcv.to_csv(csv_path, index=True)
        loader = DataLoader()
        df = loader.load_csv(csv_path)
        assert len(df) > 0
        assert "close" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns


def test_load_parquet(sample_ohlcv):
    from src.data.loader import DataLoader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = os.path.join(tmpdir, "test_data.parquet")
        sample_ohlcv.to_parquet(pq_path, index=True)
        loader = DataLoader()
        df = loader.load_parquet(pq_path)
        assert len(df) > 0
        assert "close" in df.columns
        assert "open" in df.columns


def test_load_csv_chinese_columns():
    from src.data.loader import DataLoader
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "cn_data.csv")
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "开盘": [100.0] * 10,
            "收盘": [105.0] * 10,
            "最高": [108.0] * 10,
            "最低": [98.0] * 10,
            "成交量": [1000000] * 10,
        })
        df.to_csv(csv_path, index=False, encoding="utf-8")
        loader = DataLoader()
        loaded = loader.load_csv(csv_path)
        assert "open" in loaded.columns
        assert "close" in loaded.columns
        assert len(loaded) == 10


def test_load_csv_missing_file():
    from src.data.loader import DataLoader
    loader = DataLoader()
    try:
        loader.load_csv("nonexistent_file.csv")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass


def test_load_parquet_missing_file():
    from src.data.loader import DataLoader
    loader = DataLoader()
    try:
        loader.load_parquet("nonexistent_file.parquet")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass
