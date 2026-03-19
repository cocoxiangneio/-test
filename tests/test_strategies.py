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
