# -*- coding: utf-8 -*-
"""Momentum strategies."""

from __future__ import annotations

import pandas as pd
from ..base import BaseStrategy


class MacdStrategy(BaseStrategy):
    name = "MACDStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        fast, slow, signal = self.params.get("fast", 12), self.params.get("slow", 26), self.params.get("signal", 9)
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        sig = pd.Series(0.0, index=df.index)
        sig[(macd_line > signal_line) & (hist > 0)] = 1.0
        sig[(macd_line < signal_line) & (hist < 0)] = -1.0
        return sig


class RsiStrategy(BaseStrategy):
    name = "RSIStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        buy_th, sell_th = self.params.get("buy_th", 30), self.params.get("sell_th", 70)
        sig = pd.Series(0.0, index=df.index)
        sig[rsi < buy_th] = 1.0
        sig[rsi > sell_th] = -1.0
        return sig


class KdjStrategy(BaseStrategy):
    name = "KDJStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 9)
        low_n = df["low"].rolling(period).min()
        high_n = df["high"].rolling(period).max()
        rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        sig = pd.Series(0.0, index=df.index)
        sig[(k > d) & (j < 80)] = 1.0
        sig[(k < d) & (j > 20)] = -1.0
        return sig


class MomentumStrategy(BaseStrategy):
    name = "MomentumStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        ret = df["close"].pct_change(period)
        threshold = self.params.get("threshold", 0.02)
        sig = pd.Series(0.0, index=df.index)
        sig[ret > threshold] = 1.0
        sig[ret < -threshold] = -1.0
        return sig


class RocStrategy(BaseStrategy):
    name = "ROCSstrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 12)
        roc = (df["close"] - df["close"].shift(period)) / (df["close"].shift(period) + 1e-10) * 100
        sig = pd.Series(0.0, index=df.index)
        sig[roc > 0] = 1.0
        sig[roc < 0] = -1.0
        return sig


class CciStrategy(BaseStrategy):
    name = "CCIStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
        sig = pd.Series(0.0, index=df.index)
        sig[cci < -100] = 1.0
        sig[cci > 100] = -1.0
        return sig


class AdxStrategy(BaseStrategy):
    name = "ADXStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(14).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[(plus_di > minus_di) & (adx > 25)] = 1.0
        sig[(plus_di < minus_di) & (adx > 25)] = -1.0
        return sig


class WillRStrategy(BaseStrategy):
    name = "WillRStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        high_n = df["high"].rolling(period).max()
        low_n = df["low"].rolling(period).min()
        wr = -100 * (high_n - df["close"]) / (high_n - low_n + 1e-10)
        sig = pd.Series(0.0, index=df.index)
        sig[wr < -80] = 1.0
        sig[wr > -20] = -1.0
        return sig


class StochStrategy(BaseStrategy):
    name = "StochasticStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        low_n = df["low"].rolling(period).min()
        high_n = df["high"].rolling(period).max()
        k = 100 * (df["close"] - low_n) / (high_n - low_n + 1e-10)
        d = k.rolling(3).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[(k > d) & (k < 20)] = 1.0
        sig[(k < d) & (k > 80)] = -1.0
        return sig
