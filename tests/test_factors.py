# -*- coding: utf-8 -*-
"""Tests for factor layer."""

import pytest
import pandas as pd
import numpy as np


def test_factor_registry():
    from src.factors.factor_registry import FactorRegistry, register_factor
    reg = FactorRegistry()
    reg.register("test_factor", lambda df: df["close"] * 2, "test", "test desc", 1)
    assert "test_factor" in reg.list_factors()
    result = reg.calculate("test_factor", pd.DataFrame({"close": [1, 2, 3]}))
    assert len(result) == 3


def test_sma_factor(sample_ohlcv):
    from src.factors.technical import sma_20
    result = sma_20(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)
    assert result.isna().sum() > 0


def test_rsi_factor(sample_ohlcv):
    from src.factors.technical import rsi_14
    result = rsi_14(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_macd_factor(sample_ohlcv):
    from src.factors.technical import macd, macd_signal
    m = macd(sample_ohlcv)
    s = macd_signal(sample_ohlcv)
    assert len(m) == len(sample_ohlcv)
    assert len(s) == len(sample_ohlcv)


def test_boll_factor(sample_ohlcv):
    from src.factors.technical import boll_upper, boll_mid, boll_lower
    u, mid, lo = boll_upper(sample_ohlcv), boll_mid(sample_ohlcv), boll_lower(sample_ohlcv)
    valid = u.dropna()
    assert (u[valid.index] >= mid[valid.index]).all()
    assert (mid[valid.index] >= lo[valid.index]).all()


def test_atr_factor(sample_ohlcv):
    from src.factors.technical import atr_14
    result = atr_14(sample_ohlcv)
    assert len(result) == len(sample_ohlcv)
    valid = result.dropna()
    assert (valid >= 0).all()


def test_cross_sectional_ic():
    from src.factors.cross_sectional import calculate_ic
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    factor_data = pd.DataFrame({"stock1": np.random.randn(100)}, index=dates)
    forward_ret = pd.Series(np.random.randn(100), index=dates)
    ic = calculate_ic(factor_data, forward_ret, method="spearman")
    assert len(ic) == len(dates)


def test_technical_factor_calculator(sample_ohlcv):
    from src.factors.technical import TechnicalFactorCalculator
    calc = TechnicalFactorCalculator()
    result = calc.calculate_selected(sample_ohlcv, ["sma_20", "rsi_14", "macd"])
    assert "sma_20" in result.columns
    assert "rsi_14" in result.columns
    assert len(result) == len(sample_ohlcv)
