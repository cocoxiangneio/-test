# -*- coding: utf-8 -*-
"""Tests for portfolio optimization."""

import pytest
import numpy as np
import pandas as pd


def test_min_variance_optimizer(sample_multi_ohlcv):
    from src.portfolio.optimizer import MinVarianceOptimizer

    returns = {}
    for stock, df in sample_multi_ohlcv.items():
        returns[stock] = df["close"].pct_change().dropna()

    returns_df = pd.DataFrame(returns)
    opt = MinVarianceOptimizer()
    weights = opt.optimize(returns_df)

    assert len(weights) == len(sample_multi_ohlcv)
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(v >= 0 for v in weights.values())


def test_risk_parity_optimizer(sample_multi_ohlcv):
    from src.portfolio.optimizer import RiskParityOptimizer

    returns = {s: df["close"].pct_change().dropna() for s, df in sample_multi_ohlcv.items()}
    returns_df = pd.DataFrame(returns)
    opt = RiskParityOptimizer()
    weights = opt.optimize(returns_df)

    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_hrp_optimizer(sample_multi_ohlcv):
    from src.portfolio.optimizer import HRPOptimizer

    returns = {s: df["close"].pct_change().dropna() for s, df in sample_multi_ohlcv.items()}
    returns_df = pd.DataFrame(returns)
    opt = HRPOptimizer()
    weights = opt.optimize(returns_df)

    assert len(weights) == len(sample_multi_ohlcv)
    assert not any(np.isnan(v) for v in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_risk_manager():
    from src.portfolio.risk_manager import RiskManager, Rebalancer

    rm = RiskManager(max_position_pct=0.3)
    capped = rm.check_position_limits({"A": 0.5, "B": 0.3}, 100000)
    assert capped["A"] <= 0.3

    capped2 = rm.check_position_limits({"A": 0.6, "B": 0.5, "C": 0.5, "D": 0.5}, 100000)
    assert capped2["A"] <= 0.3
    assert abs(sum(capped2.values()) - 1.0) < 0.01

    rb = Rebalancer(rebalance_threshold=0.05)
    assert rb.should_rebalance({"A": 0.5}, {"A": 0.4}) == True
    assert rb.should_rebalance({"A": 0.5}, {"A": 0.49}) == False


def test_portfolio_optimizer_factory():
    from src.portfolio.optimizer import PortfolioOptimizerFactory
    methods = PortfolioOptimizerFactory.list_methods()
    assert "mean_variance" in methods
    assert "min_variance" in methods
    assert "risk_parity" in methods
    assert "hrp" in methods
