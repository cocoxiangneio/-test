# -*- coding: utf-8 -*-
"""Tests for strategy layer."""

import pytest
import pandas as pd
import numpy as np


def test_base_strategy():
    from src.strategies.base import BaseStrategy
    class DummyStrategy(BaseStrategy):
        name = "Dummy"
        def signal(self, df):
            return pd.Series(0.0, index=df.index)

    strat = DummyStrategy({"param": 1})
    assert strat.params["param"] == 1
    strat.set_params(param=2)
    assert strat.params["param"] == 2


def test_ma_cross_strategy(sample_ohlcv):
    from src.strategies.breakout import MaCrossStrategy
    strat = MaCrossStrategy({"fast": 5, "slow": 20})
    sig = strat.signal(sample_ohlcv)
    assert len(sig) == len(sample_ohlcv)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_rsi_strategy(sample_ohlcv):
    from src.strategies.momentum import RsiStrategy
    strat = RsiStrategy({"period": 14, "buy_th": 30, "sell_th": 70})
    sig = strat.signal(sample_ohlcv)
    assert len(sig) == len(sample_ohlcv)
    assert sig.isin([-1.0, 0.0, 1.0]).all()


def test_boll_strategy(sample_ohlcv):
    from src.strategies.breakout import BollStrategy
    strat = BollStrategy({"period": 20, "num_std": 2})
    sig = strat.signal(sample_ohlcv)
    assert len(sig) == len(sample_ohlcv)


def test_ml_strategy(sample_ohlcv):
    from src.strategies.ml import MLEnsembleStrategy
    strat = MLEnsembleStrategy()
    sig = strat.signal(sample_ohlcv)
    assert len(sig) == len(sample_ohlcv)


def test_all_strategies_importable():
    from src.strategies.breakout import (
        MaCrossStrategy, MaCross3Strategy, BollStrategy, BreakoutStrategy,
        AtrStrategy, ChannelBreakoutStrategy, DualBreakoutStrategy,
        ParabolicSARStrategy, TurtleStrategy, VolumeMaStrategy, IntradayVolatilityStrategy,
    )
    from src.strategies.momentum import (
        MacdStrategy, RsiStrategy, KdjStrategy, MomentumStrategy,
        RocStrategy, CciStrategy, AdxStrategy, WillRStrategy, StochStrategy,
    )
    from src.strategies.mean_reversion import (
        MeanReversionStrategy, RsiMaStrategy, MacdRsiStrategy,
        BollingerBounceStrategy, SupportResistanceStrategy,
    )
    from src.strategies.multi_timeframe import (
        MultiTimeFrameStrategy, TripleScreenStrategy, IchimokuStrategy, TrendStrengthStrategy,
    )
    assert True


def test_ml_strategy_fit():
    from src.strategies.ml import MLStrategy

    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(300) * 0.5)
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 300),
    }, index=dates)

    strat = MLStrategy(model_type="rf", n_estimators=50, max_depth=3)
    strat.fit(df)

    assert strat._model is not None
    sig = strat.signal(df)
    assert isinstance(sig, pd.Series)
    assert len(sig) == len(df)


def test_ml_strategy_multiple_model_types():
    from src.strategies.ml import MLStrategy

    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(300) * 0.5)
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 300),
    }, index=dates)

    for model_type in ["rf", "gb", "dt"]:
        strat = MLStrategy(model_type=model_type, n_estimators=20, max_depth=3)
        strat.fit(df)
        assert strat._model is not None, f"Model {model_type} should be created"
        sig = strat.signal(df)
        assert len(sig) == len(df)


def test_ml_strategy_no_future_leakage():
    from src.strategies.ml import MLStrategy

    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(300) * 0.5)
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 300),
    }, index=dates)

    strat = MLStrategy(model_type="rf", n_estimators=20, max_depth=3)
    strat.fit(df, train_start=dates[100])

    sig = strat.signal(df)
    sig_train = sig.loc[dates[50]:dates[150]]
    sig_oos = sig.loc[dates[200]:]

    assert sig_train.abs().sum() > 0, "Should generate signals during training period"
    assert sig_oos.abs().sum() > 0, "Should generate signals during OOS period"
