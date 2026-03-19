# -*- coding: utf-8 -*-
"""Tests for backtest engine."""

import pytest
import pandas as pd


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
