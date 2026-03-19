# -*- coding: utf-8 -*-
"""Technical factors - 40+ factors across trend, momentum, volatility, volume, support/resistance."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from .factor_registry import register_factor

logger = logging.getLogger(__name__)


def _normalize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-10)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


@register_factor("sma_5", category="trend", description="5-day simple moving average", window_required=5)
def sma_5(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(5).mean()


@register_factor("sma_10", category="trend", description="10-day simple moving average", window_required=10)
def sma_10(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(10).mean()


@register_factor("sma_20", category="trend", description="20-day simple moving average", window_required=20)
def sma_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(20).mean()


@register_factor("sma_60", category="trend", description="60-day simple moving average", window_required=60)
def sma_60(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(60).mean()


@register_factor("ema_5", category="trend", description="5-day exponential moving average", window_required=5)
def ema_5(df: pd.DataFrame) -> pd.Series:
    return _ema(df["close"], 5)


@register_factor("ema_10", category="trend", description="10-day exponential moving average", window_required=10)
def ema_10(df: pd.DataFrame) -> pd.Series:
    return _ema(df["close"], 10)


@register_factor("ema_20", category="trend", description="20-day exponential moving average", window_required=20)
def ema_20(df: pd.DataFrame) -> pd.Series:
    return _ema(df["close"], 20)


@register_factor("ema_60", category="trend", description="60-day exponential moving average", window_required=60)
def ema_60(df: pd.DataFrame) -> pd.Series:
    return _ema(df["close"], 60)


@register_factor("ma_cross", category="trend", description="MA cross signal: 1=golden cross, -1=dead cross", window_required=60)
def ma_cross(df: pd.DataFrame) -> pd.Series:
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    signal = pd.Series(0, index=df.index)
    signal[ma5 > ma20] = 1
    signal[ma5 < ma20] = -1
    return signal


@register_factor("macd", category="momentum", description="MACD line (EMA12-EMA26)", window_required=26)
def macd(df: pd.DataFrame) -> pd.Series:
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    return ema12 - ema26


@register_factor("macd_signal", category="momentum", description="MACD signal line (9-day EMA of MACD)", window_required=35)
def macd_signal(df: pd.DataFrame) -> pd.Series:
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    macd = ema12 - ema26
    return _ema(macd, 9)


@register_factor("macd_hist", category="momentum", description="MACD histogram (MACD - signal)", window_required=35)
def macd_hist(df: pd.DataFrame) -> pd.Series:
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    return macd - signal


@register_factor("rsi_6", category="momentum", description="6-day RSI", window_required=7)
def rsi_6(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(6).mean()
    loss = (-delta.clip(upper=0)).rolling(6).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@register_factor("rsi_12", category="momentum", description="12-day RSI", window_required=13)
def rsi_12(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(12).mean()
    loss = (-delta.clip(upper=0)).rolling(12).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@register_factor("rsi_14", category="momentum", description="14-day RSI", window_required=15)


def rsi_14(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@register_factor("rsi_24", category="momentum", description="24-day RSI", window_required=25)
def rsi_24(df: pd.DataFrame) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(24).mean()
    loss = (-delta.clip(upper=0)).rolling(24).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


@register_factor("boll_upper", category="volatility", description="Bollinger upper band (20-day, 2std)", window_required=21)
def boll_upper(df: pd.DataFrame) -> pd.Series:
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    return sma20 + 2 * std20


@register_factor("boll_mid", category="volatility", description="Bollinger middle band (20-day SMA)", window_required=20)
def boll_mid(df: pd.DataFrame) -> pd.Series:
    return df["close"].rolling(20).mean()


@register_factor("boll_lower", category="volatility", description="Bollinger lower band (20-day, 2std)", window_required=21)
def boll_lower(df: pd.DataFrame) -> pd.Series:
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    return sma20 - 2 * std20


@register_factor("boll_width", category="volatility", description="Bollinger bandwidth", window_required=21)
def boll_width(df: pd.DataFrame) -> pd.Series:
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    return (2 * std20) / (sma20 + 1e-10)


@register_factor("atr_14", category="volatility", description="14-day Average True Range", window_required=15)
def atr_14(df: pd.DataFrame) -> pd.Series:
    return _atr(df["high"], df["low"], df["close"], 14)


@register_factor("atr_20", category="volatility", description="20-day Average True Range", window_required=21)
def atr_20(df: pd.DataFrame) -> pd.Series:
    return _atr(df["high"], df["low"], df["close"], 20)


@register_factor("volatility_20", category="volatility", description="20-day return volatility", window_required=21)
def volatility_20(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change()
    return returns.rolling(20).std() * np.sqrt(252)


@register_factor("volatility_60", category="volatility", description="60-day return volatility", window_required=61)
def volatility_60(df: pd.DataFrame) -> pd.Series:
    returns = df["close"].pct_change()
    return returns.rolling(60).std() * np.sqrt(252)


@register_factor("kdj_k", category="momentum", description="KDJ K value", window_required=10)
def kdj_k(df: pd.DataFrame) -> pd.Series:
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    return k


@register_factor("kdj_d", category="momentum", description="KDJ D value", window_required=10)
def kdj_d(df: pd.DataFrame) -> pd.Series:
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    return d


@register_factor("kdj_j", category="momentum", description="KDJ J value (3K-2D)", window_required=10)
def kdj_j(df: pd.DataFrame) -> pd.Series:
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    return 3 * k - 2 * d


@register_factor("cci_14", category="momentum", description="14-day Commodity Channel Index", window_required=15)
def cci_14(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(14).mean()
    mad = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad + 1e-10)


@register_factor("cci_20", category="momentum", description="20-day Commodity Channel Index", window_required=21)
def cci_20(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad + 1e-10)


@register_factor("willr_14", category="momentum", description="14-day Williams %R", window_required=15)
def willr_14(df: pd.DataFrame) -> pd.Series:
    high_n = df["high"].rolling(14).max()
    low_n = df["low"].rolling(14).min()
    return -100 * (high_n - df["close"]) / (high_n - low_n + 1e-10)


@register_factor("willr_20", category="momentum", description="20-day Williams %R", window_required=21)
def willr_20(df: pd.DataFrame) -> pd.Series:
    high_n = df["high"].rolling(20).max()
    low_n = df["low"].rolling(20).min()
    return -100 * (high_n - df["close"]) / (high_n - low_n + 1e-10)


@register_factor("adx_14", category="momentum", description="14-day Average Directional Index", window_required=15)
def adx_14(df: pd.DataFrame) -> pd.Series:
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr14 = _atr(df["high"], df["low"], df["close"], 14)
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(14).mean()


@register_factor("obv", category="volume", description="On-Balance Volume", window_required=1)
def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()) * df["volume"]).cumsum()


@register_factor("obv_ema", category="volume", description="EMA of OBV", window_required=10)
def obv_ema(df: pd.DataFrame) -> pd.Series:
    obv_val = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
    return _ema(obv_val, 10)


@register_factor("volume_ratio", category="volume", description="Volume / 20-day avg volume", window_required=21)
def volume_ratio(df: pd.DataFrame) -> pd.Series:
    return df["volume"] / df["volume"].rolling(20).mean()


@register_factor("volume_ma5", category="volume", description="5-day volume MA", window_required=5)
def volume_ma5(df: pd.DataFrame) -> pd.Series:
    return df["volume"].rolling(5).mean()


@register_factor("volume_ma20", category="volume", description="20-day volume MA", window_required=20)
def volume_ma20(df: pd.DataFrame) -> pd.Series:
    return df["volume"].rolling(20).mean()


@register_factor("money_flow_20", category="volume", description="20-day money flow (price*volume)", window_required=20)
def money_flow_20(df: pd.DataFrame) -> pd.Series:
    return (df["close"] * df["volume"]).rolling(20).mean()


@register_factor("momentum_5", category="momentum", description="5-day price momentum", window_required=6)
def momentum_5(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(5)


@register_factor("momentum_10", category="momentum", description="10-day price momentum", window_required=11)
def momentum_10(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(10)


@register_factor("momentum_20", category="momentum", description="20-day price momentum", window_required=21)
def momentum_20(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(20)


@register_factor("roc_10", category="momentum", description="10-day Rate of Change", window_required=11)
def roc_10(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(10)) / (df["close"].shift(10) + 1e-10) * 100


@register_factor("roc_20", category="momentum", description="20-day Rate of Change", window_required=21)
def roc_20(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["close"].shift(20)) / (df["close"].shift(20) + 1e-10) * 100


@register_factor("price_position", category="support_resistance", description="Price position within 20-day range", window_required=20)
def price_position(df: pd.DataFrame) -> pd.Series:
    high20 = df["high"].rolling(20).max()
    low20 = df["low"].rolling(20).min()
    return (df["close"] - low20) / (high20 - low20 + 1e-10)


@register_factor("support_level", category="support_resistance", description="20-day rolling min (support)", window_required=20)
def support_level(df: pd.DataFrame) -> pd.Series:
    return df["low"].rolling(20).min()


@register_factor("resistance_level", category="support_resistance", description="20-day rolling max (resistance)", window_required=20)
def resistance_level(df: pd.DataFrame) -> pd.Series:
    return df["high"].rolling(20).max()


@register_factor("avg_true_range_pct", category="volatility", description="ATR as % of close price", window_required=15)
def avg_true_range_pct(df: pd.DataFrame) -> pd.Series:
    return _atr(df["high"], df["low"], df["close"], 14) / df["close"] * 100


@register_factor("adl", category="volume", description="Accumulation/Distribution Line", window_required=1)
def adl(df: pd.DataFrame) -> pd.Series:
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-10)
    mfv = mfm * df["volume"]
    return mfv.cumsum()


@register_factor("stoch_k", category="momentum", description="Stochastic %K", window_required=15)
def stoch_k(df: pd.DataFrame) -> pd.Series:
    low_n = df["low"].rolling(14).min()
    high_n = df["high"].rolling(14).max()
    return 100 * (df["close"] - low_n) / (high_n - low_n + 1e-10)


@register_factor("stoch_d", category="momentum", description="Stochastic %D (3-day SMA of %K)", window_required=17)
def stoch_d(df: pd.DataFrame) -> pd.Series:
    low_n = df["low"].rolling(14).min()
    high_n = df["high"].rolling(14).max()
    k = 100 * (df["close"] - low_n) / (high_n - low_n + 1e-10)
    return k.rolling(3).mean()


class TechnicalFactorCalculator:
    def __init__(self, registry=None):
        self.registry = registry

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.registry is None:
            from .factor_registry import get_registry
            self.registry = get_registry()

        factors = self.registry.list_factors("technical")
        result, _ = self.registry.batch_calculate(factors, df)
        return result

    def calculate_by_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        if self.registry is None:
            from .factor_registry import get_registry
            self.registry = get_registry()

        factors = self.registry.list_factors(category)
        result, _ = self.registry.batch_calculate(factors, df)
        return result

    def calculate_selected(self, df: pd.DataFrame, factor_names: list) -> pd.DataFrame:
        if self.registry is None:
            from .factor_registry import get_registry
            self.registry = get_registry()

        return self.registry.batch_calculate(factor_names, df)[0]
