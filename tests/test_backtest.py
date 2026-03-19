# -*- coding: utf-8 -*-
"""Tests for backtest engine."""

import pytest
import pandas as pd
import numpy as np


def test_backtest_engine_basic(sample_ohlcv):
    from src.backtest.engine import BacktestEngine

    engine = BacktestEngine(initial_cash=100000, commission_rate=0.0003, slippage_pct=0.001)

    def dummy_strategy(data, date):
        return {stock: 0.0 for stock in data.keys()}

    result = engine.run({}, dummy_strategy)
    assert result.total_trades == 0
    assert result.final_equity == 100000.0


def test_backtest_engine_buy_hold(sample_multi_ohlcv):
    from src.backtest.engine import BacktestEngine

    engine = BacktestEngine(initial_cash=100000)

    def buy_hold(data, date):
        signals = {}
        for stock, df in data.items():
            if date in df.index:
                signals[stock] = 1.0
        return signals

    result = engine.run(sample_multi_ohlcv, buy_hold)
    assert result.total_trades > 0
    assert result.final_equity > 0


def test_commission_models():
    from src.backtest.commission import PercentCommission, FixedCommission, TieredCommission
    p = PercentCommission(0.0003)
    assert p.calc(10.0, 1000) == pytest.approx(3.0)
    f = FixedCommission(5.0)
    assert f.calc(10.0, 1000) == 5.0
    t = TieredCommission()
    assert t.calc(5000) >= 0


def test_slippage_models():
    from src.backtest.slippage import FixedSlippage, PercentSlippage, VolumeSlippage
    f = FixedSlippage(0.001)
    assert f.apply(100, "buy") == 100.1
    assert f.apply(100, "sell") == 99.9
    p = PercentSlippage(0.002)
    assert p.apply(100, "buy") == 100.2
    v = VolumeSlippage()
    assert v.apply(100, "buy", 100) > 100


def test_tca_single_trade():
    from src.backtest.tca import TransactionCostAnalyzer, TCAResult

    tca = TransactionCostAnalyzer(commission_rate=0.0003)
    result = tca.analyze(price=100.0, order_volume=1000, is_buy=True)

    assert isinstance(result, TCAResult)
    assert result.commission > 0
    assert result.total_cost > 0
    expected = result.commission + abs(result.slippage) + result.market_impact + result.liquidity_cost
    assert abs(result.total_cost - expected) < 1e-10


def test_tca_market_impact_positive_correlation():
    from src.backtest.tca import MarketImpactModel

    mi = MarketImpactModel(eta=0.1)
    small_order = mi.calculate(order_volume=100, avg_daily_volume=10000, price=100.0, volatility=0.02)
    large_order = mi.calculate(order_volume=1000, avg_daily_volume=10000, price=100.0, volatility=0.02)

    assert large_order > small_order, "Market impact should increase with order size"


def test_tca_liquidity_cost_with_adv():
    from src.backtest.tca import LiquidityCostModel

    lc = LiquidityCostModel(spread_bps=2.0)
    cost_no_adv = lc.calculate(order_volume=1000, price=100.0, is_buy=True, adv=None)
    cost_small_adv = lc.calculate(order_volume=1000, price=100.0, is_buy=True, adv=5000)

    assert cost_small_adv > cost_no_adv, f"Liquidity cost should increase with small ADV (got no_adv={cost_no_adv}, small_adv={cost_small_adv})"


def test_tca_batch_analysis():
    from src.backtest.tca import TransactionCostAnalyzer

    tca = TransactionCostAnalyzer(commission_rate=0.0003)
    trades = pd.DataFrame({
        "price": [100.0, 50.0],
        "volume": [100, 200],
        "is_buy": [True, False],
        "adv": [10000, 20000],
        "volatility": [0.02, 0.015],
    })

    results = tca.analyze_batch(trades)
    assert len(results) == 2
    assert "total_cost" in results.columns
    assert "net_proceeds" in results.columns
    assert all(results["total_cost"] > 0)
