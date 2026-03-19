# -*- coding: utf-8 -*-
"""Multi-timeframe strategies."""

from __future__ import annotations

import pandas as pd
from ..base import BaseStrategy


class MultiTimeFrameStrategy(BaseStrategy):
    name = "MultiTimeFrameStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        short = self.params.get("short", 5)
        long = self.params.get("long", 20)
        ma_short = df["close"].rolling(short).mean()
        ma_long = df["close"].rolling(long).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[ma_short > ma_long] = 1.0
        sig[ma_short < ma_long] = -1.0
        return sig


class TripleScreenStrategy(BaseStrategy):
    name = "TripleScreenStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        ema_period = self.params.get("ema_period", 20)
        ema = df["close"].ewm(span=ema_period, adjust=False).mean()
        weekly_stoch_k = self.params.get("weekly_stoch_k", 20)
        weekly_stoch_d = self.params.get("weekly_stoch_d", 5)
        low_n = df["low"].rolling(weekly_stoch_k).min()
        high_n = df["high"].rolling(weekly_stoch_k).max()
        stoch = 100 * (df["close"] - low_n) / (high_n - low_n + 1e-10)
        stoch_ma = stoch.rolling(weekly_stoch_d).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[(df["close"] > ema) & (stoch < 20) & (stoch > stoch_ma)] = 1.0
        sig[(df["close"] < ema) & (stoch > 80) & (stoch < stoch_ma)] = -1.0
        return sig


class IchimokuStrategy(BaseStrategy):
    name = "IchimokuStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        conv = self.params.get("conversion", 9)
        base = self.params.get("base", 26)
        span_b = self.params.get("span_b", 52)
        lead = self.params.get("lead", 26)
        conv_line = (df["high"].rolling(conv).max() + df["low"].rolling(conv).min()) / 2
        base_line = (df["high"].rolling(base).max() + df["low"].rolling(base).min()) / 2
        span_a = ((conv_line + base_line) / 2).shift(lead)
        span_b_val = ((df["high"].rolling(span_b).max() + df["low"].rolling(span_b).min()) / 2).shift(lead)
        sig = pd.Series(0.0, index=df.index)
        sig[(conv_line > base_line) & (df["close"] > span_a) & (df["close"] > span_b_val)] = 1.0
        sig[(conv_line < base_line) & (df["close"] < span_a) & (df["close"] < span_b_val)] = -1.0
        return sig


class TrendStrengthStrategy(BaseStrategy):
    name = "TrendStrengthStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        adx_period = self.params.get("adx_period", 14)
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(adx_period).mean()
        plus_di = 100 * plus_dm.rolling(adx_period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(adx_period).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[adx > 25] = 1.0
        sig[adx < 15] = -1.0
        return sig
