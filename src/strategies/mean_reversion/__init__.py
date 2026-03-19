# -*- coding: utf-8 -*-
"""Mean reversion strategies."""

from __future__ import annotations

import pandas as pd
from ..base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "MeanReversionStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        z_threshold = self.params.get("z_threshold", 2.0)
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        z = (df["close"] - ma) / (std + 1e-10)
        sig = pd.Series(0.0, index=df.index)
        sig[z < -z_threshold] = 1.0
        sig[z > z_threshold] = -1.0
        sig[abs(z) < 0.5] = 0.0
        return sig


class RsiMaStrategy(BaseStrategy):
    name = "RSIMAStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        ma_period = self.params.get("ma_period", 20)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        ma = df["close"].rolling(ma_period).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[(df["close"] > ma) & (rsi < 40)] = 1.0
        sig[(df["close"] < ma) & (rsi > 60)] = -1.0
        return sig


class MacdRsiStrategy(BaseStrategy):
    name = "MACDRSIStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd = macd_line - signal_line
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        sig = pd.Series(0.0, index=df.index)
        sig[(macd > 0) & (rsi < 50)] = 1.0
        sig[(macd < 0) & (rsi > 50)] = -1.0
        return sig


class BollingerBounceStrategy(BaseStrategy):
    name = "BollingerBounceStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        num_std = self.params.get("num_std", 2)
        ma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        sig = pd.Series(0.0, index=df.index)
        sig[df["close"] < lower] = 1.0
        sig[df["close"] > upper] = -1.0
        return sig


class SupportResistanceStrategy(BaseStrategy):
    name = "SupportResistanceStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        lookback = self.params.get("lookback", 20)
        tol = self.params.get("tolerance", 0.01)
        rolling_min = df["low"].rolling(lookback).min()
        rolling_max = df["high"].rolling(lookback).max()
        sig = pd.Series(0.0, index=df.index)
        sig[df["close"] <= rolling_min * (1 + tol)] = 1.0
        sig[df["close"] >= rolling_max * (1 - tol)] = -1.0
        return sig
