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


def test_ic_validator_basic():
    from src.factors.cross_sectional import ICValidator
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    stock = "stock1"
    idx = pd.MultiIndex.from_product([dates, [stock]], names=["date", "code"])
    factor_vals = np.random.randn(60)
    price_vals = np.cumsum(np.random.randn(60)) + 100
    factor_panel = pd.DataFrame({"factor1": factor_vals}, index=idx)
    prices = pd.DataFrame({"stock1": price_vals}, index=dates)
    validator = ICValidator(ic_threshold=0.02, ir_threshold=0.5, forward_period=5)
    result = validator.validate(factor_panel, prices, method="spearman", period=5)
    assert "ic_mean" in result
    assert "monthly_ic_pass_rate" in result
    assert "ir_mean" in result
    assert "daily_ic" in result
    assert "monthly_ic" in result


def test_ic_validator_forward_returns():
    from src.factors.cross_sectional import ICValidator
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    stock = "stock1"
    idx = pd.MultiIndex.from_product([dates, [stock]], names=["date", "code"])
    returns = np.random.randn(40) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    factor_vals = returns + np.random.randn(40) * 0.005
    factor_panel = pd.DataFrame({"factor1": factor_vals}, index=idx)
    prices = pd.DataFrame({"stock1": price}, index=dates)
    validator = ICValidator(ic_threshold=0.02, ir_threshold=0.5, forward_period=5)
    fwd = validator.compute_forward_returns(prices, period=5)
    assert len(fwd) == len(prices)
    assert fwd.iloc[-5:].isna().all().all()
    ic_df = validator.compute_daily_ic(factor_panel, prices, method="spearman", period=5)
    assert "IC" in ic_df.columns
    assert "IR" in ic_df.columns


def test_ic_validator_is_valid():
    from src.factors.cross_sectional import ICValidator
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    stock = "stock1"
    idx = pd.MultiIndex.from_product([dates, [stock]], names=["date", "code"])
    returns = np.random.randn(80) * 0.02
    price = 100 * np.exp(np.cumsum(returns))
    factor_vals = returns + np.random.randn(80) * 0.003
    factor_panel = pd.DataFrame({"factor1": factor_vals}, index=idx)
    prices = pd.DataFrame({"stock1": price}, index=dates)
    validator = ICValidator(ic_threshold=0.02, ir_threshold=0.5, forward_period=5)
    result = validator.validate(factor_panel, prices)
    monthly_pass = result["monthly_ic_pass_rate"] >= 0.6
    is_valid = validator.is_valid(result)
    assert isinstance(is_valid, bool)


def test_ic_validator_monthly_ic():
    from src.factors.cross_sectional import ICValidator
    np.random.seed(99)
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    stocks = [f"stock{i}" for i in range(1, 16)]
    idx = pd.MultiIndex.from_product([dates, stocks], names=["date", "code"])
    returns = np.random.randn(120 * 15).reshape(120, 15) * 0.02
    price_dict = {}
    for i, stock in enumerate(stocks):
        price_dict[stock] = 100 * np.exp(np.cumsum(returns[:, i]))
    prices = pd.DataFrame(price_dict, index=dates)
    factor_vals = returns * 0.8 + np.random.randn(120 * 15).reshape(120, 15) * 0.004
    factor_panel = pd.DataFrame({"factor1": factor_vals.flatten()}, index=idx)
    validator = ICValidator(ic_threshold=0.02, ir_threshold=0.5, forward_period=5)
    monthly_ic = validator.compute_monthly_ic(factor_panel, prices, method="spearman", period=5)
    assert "IC" in monthly_ic.columns
    ic_values = monthly_ic["IC"].dropna()
    assert len(ic_values) > 0
