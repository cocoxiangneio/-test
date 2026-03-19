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


def test_weights_to_backtest_signals():
    from src.portfolio.optimizer import MeanVarianceOptimizer
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    returns_data = {f"s{i}": np.random.randn(60) * 0.01 for i in range(3)}
    returns_df = pd.DataFrame(returns_data, index=dates)
    opt = MeanVarianceOptimizer(risk_aversion=1.0)
    weights = opt.optimize(returns_df)
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01
    assert all(0 <= w <= 1 for w in weights.values())


def test_risk_manager_position_limits_in_backtest():
    from src.portfolio.risk_manager import RiskManager
    rm = RiskManager(max_position_pct=0.3)
    raw_weights = {"A": 0.5, "B": 0.4, "C": 0.1}
    capped = rm.check_position_limits(raw_weights, 100000)
    assert capped["A"] <= 0.3
    assert capped["B"] <= 0.3
    capped2 = rm.check_position_limits({"A": 0.6, "B": 0.5, "C": 0.5, "D": 0.5}, 100000)
    assert capped2["A"] <= 0.3
    assert abs(sum(capped2.values()) - 1.0) < 0.01


def test_optimizer_weights_backtest_integration():
    from src.backtest.engine import BacktestEngine
    from src.portfolio.optimizer import MeanVarianceOptimizer, RiskParityOptimizer
    from src.portfolio.risk_manager import RiskManager

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    stocks = ["A", "B", "C"]
    data = {}
    for s in stocks:
        close = 100 * np.exp(np.cumsum(np.random.randn(60) * 0.01))
        data[s] = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(60, 1000000),
        }, index=dates)

    returns_data = {s: data[s]["close"].pct_change().dropna() for s in stocks}
    returns_df = pd.DataFrame(returns_data)

    mv_opt = MeanVarianceOptimizer()
    mv_weights = mv_opt.optimize(returns_df)

    rp_opt = RiskParityOptimizer()
    rp_weights = rp_opt.optimize(returns_df)

    rm = RiskManager(max_position_pct=0.4)
    mv_capped = rm.check_position_limits(mv_weights, 100000)
    rp_capped = rm.check_position_limits(rp_weights, 100000)

    def make_strategy(target_weights):
        def strategy(data_dict, date):
            signals = {}
            for stock in data_dict.keys():
                if stock in target_weights and target_weights[stock] > 0.01:
                    signals[stock] = 1.0
            return signals
        return strategy

    for name, weights in [("MV", mv_capped), ("RP", rp_capped)]:
        engine = BacktestEngine(initial_cash=100000, commission_rate=0.0003)
        bt = engine.run(data, make_strategy(weights))
        assert bt.total_trades > 0
        assert bt.final_equity > 0
        buy_trades = [t for t in bt.trades if t.side == "buy"]
        assert len(buy_trades) >= len([w for w in weights.values() if w > 0.01])
        assert bt.final_equity > 0
