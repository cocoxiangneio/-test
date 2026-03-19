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
        engine = BacktestEngine(initial_cash=100000, commission_rate=0.0003, position_size=0.34)
        bt = engine.run(data, make_strategy(weights))
        assert bt.total_trades > 0
        assert bt.final_equity > 0
        buy_trades = [t for t in bt.trades if t.side == "buy"]
        assert len(buy_trades) >= len([w for w in weights.values() if w > 0.01])
        assert bt.final_equity > 0


def _build_controlled_ohlcv(dates, base_price=100.0, trend=0.0, vol=0.01, seed=42):
    np.random.seed(seed)
    close = base_price * np.exp(np.cumsum(np.random.randn(len(dates)) * vol + trend))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)


def test_factor_portfolio_build_quantile_weights_long_short():
    from src.portfolio.factor_portfolio import FactorPortfolioBuilder

    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    stocks = ["A", "B", "C", "D", "E"]
    panel_data = []
    for date in dates:
        for stock in stocks:
            panel_data.append((date, stock, float(ord(stock[0]) - ord("A"))))
    factor_panel = pd.DataFrame(panel_data, columns=["date", "stock", "momentum"])
    factor_panel = factor_panel.set_index(["date", "stock"])["momentum"]

    builder = FactorPortfolioBuilder(n_quantiles=3, long_short=True)
    weights = builder.build_quantile_weights(factor_panel, dates[5], None)

    long_weights = [v for v in weights.values() if v > 0]
    short_weights = [v for v in weights.values() if v < 0]

    assert len(weights) > 0, "Should have non-empty weights"
    assert len(long_weights) > 0, "Should have long positions"
    assert len(short_weights) > 0, "Should have short positions"
    assert abs(sum(weights.values())) < 1e-6, "Net long-short should be ~0"
    total_abs = sum(abs(v) for v in weights.values())
    assert abs(total_abs - 1.0) < 1e-6, "Total absolute weight should be 1"


def test_factor_portfolio_run_backtest_equity_curve():
    from src.portfolio.factor_portfolio import FactorPortfolioBuilder

    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    stocks = ["S1", "S2", "S3", "S4", "S5"]

    prices_dict = {}
    for i, s in enumerate(stocks):
        prices_dict[s] = _build_controlled_ohlcv(
            dates, base_price=50.0 + i * 10, trend=0.0002 * (i % 3 - 1), seed=42 + i
        )

    panel_data = []
    for date in dates:
        for s in stocks:
            s_idx = stocks.index(s)
            panel_data.append((date, s, float(s_idx)))
    factor_panel = pd.DataFrame(panel_data, columns=["date", "stock", "factor"])
    factor_panel = factor_panel.set_index(["date", "stock"])["factor"]

    builder = FactorPortfolioBuilder(n_quantiles=3, long_short=True)
    equity, metrics = builder.run_backtest(
        factor_panel, prices_dict, rebalance_freq="ME"
    )

    assert isinstance(equity, pd.Series), "Equity should be a Series"
    assert not equity.empty, "Equity should not be empty"
    assert metrics["sharpe"] is not None
    assert metrics["total_return"] is not None
    assert metrics["n_rebal_dates"] > 0, "Should have rebalancing dates"


def test_factor_portfolio_weights_normalized_to_one():
    from src.portfolio.factor_portfolio import FactorPortfolioBuilder

    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H"]
    panel_data = []
    for date in dates:
        for stock in stocks:
            panel_data.append((date, stock, float(ord(stock[0]))))
    factor_panel = pd.DataFrame(panel_data, columns=["date", "stock", "factor"])
    factor_panel = factor_panel.set_index(["date", "stock"])["factor"]

    builder = FactorPortfolioBuilder(n_quantiles=4, long_short=True)
    weights = builder.build_quantile_weights(factor_panel, dates[2], None)

    total_abs = sum(abs(v) for v in weights.values())
    assert abs(total_abs - 1.0) < 1e-6, f"Long-short weights should normalize to 1, got {total_abs}"

    long_pos = sum(v for v in weights.values() if v > 0)
    short_pos = sum(v for v in weights.values() if v < 0)
    assert abs(long_pos + short_pos) < 1e-6, "Net long-short should be zero"


def test_ica_aware_portfolio_builder():
    from src.portfolio.factor_portfolio import ICAwarePortfolioBuilder

    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    stocks = ["S1", "S2", "S3", "S4", "S5"]

    prices_dict = {}
    for i, s in enumerate(stocks):
        prices_dict[s] = _build_controlled_ohlcv(
            dates, base_price=50.0, trend=0.0001, seed=100 + i
        )

    panel_data = []
    for date in dates:
        for s in stocks:
            panel_data.append((date, s, float(stocks.index(s))))
    factor_panel = pd.DataFrame(panel_data, columns=["date", "stock", "factor"])
    factor_panel = factor_panel.set_index(["date", "stock"])["factor"]

    builder = ICAwarePortfolioBuilder(n_quantiles=3, ic_threshold=0.02)
    equity, metrics = builder.run_backtest(
        factor_panel, prices_dict, rebalance_freq="ME"
    )

    assert isinstance(equity, pd.Series)
    assert not equity.empty
    assert "sharpe" in metrics
