# -*- coding: utf-8 -*-
"""Breakout strategies."""

from __future__ import annotations

import pandas as pd
from ..base import BaseStrategy


class MaCrossStrategy(BaseStrategy):
    name = "MACrossStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        fast = self.params.get("fast", 5)
        slow = self.params.get("slow", 20)
        ma_fast = df["close"].rolling(fast).mean()
        ma_slow = df["close"].rolling(slow).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[ma_fast > ma_slow] = 1.0
        sig[ma_fast < ma_slow] = -1.0
        return sig


class MaCross3Strategy(BaseStrategy):
    name = "MACross3Strategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        ma5 = df["close"].rolling(5).mean()
        ma20 = df["close"].rolling(20).mean()
        ma60 = df["close"].rolling(60).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[(ma5 > ma20) & (ma20 > ma60)] = 1.0
        sig[(ma5 < ma20) & (ma20 < ma60)] = -1.0
        return sig


class BollStrategy(BaseStrategy):
    name = "BollStrategy"

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
        sig[(df["close"] >= lower) & (df["close"] <= upper)] = 0.0
        return sig


class BreakoutStrategy(BaseStrategy):
    name = "BreakoutStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        rolling_max = df["high"].rolling(period).max()
        sig = pd.Series(0.0, index=df.index)
        sig[df["close"] > rolling_max.shift(1)] = 1.0
        sig[df["close"] < rolling_max.shift(1)] = -1.0
        return sig


class AtrStrategy(BaseStrategy):
    name = "ATRStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 14)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[atr > 0] = 1.0
        return sig


class ChannelBreakoutStrategy(BaseStrategy):
    name = "ChannelBreakoutStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        up_period = self.params.get("up_period", 20)
        dn_period = self.params.get("dn_period", 10)
        up_channel = df["high"].rolling(up_period).max()
        dn_channel = df["low"].rolling(dn_period).min()
        sig = pd.Series(0.0, index=df.index)
        sig[df["close"] > up_channel.shift(1)] = 1.0
        sig[df["close"] < dn_channel.shift(1)] = -1.0
        return sig


class DualBreakoutStrategy(BaseStrategy):
    name = "DualBreakoutStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        short_period = self.params.get("short_period", 20)
        long_period = self.params.get("long_period", 60)
        short_high = df["high"].rolling(short_period).max()
        long_high = df["high"].rolling(long_period).max()
        sig = pd.Series(0.0, index=df.index)
        sig[(df["close"] > short_high.shift(1)) & (df["close"] > long_high.shift(1))] = 1.0
        sig[(df["close"] < short_high.shift(1)) & (df["close"] < long_high.shift(1))] = -1.0
        return sig


class ParabolicSARStrategy(BaseStrategy):
    name = "ParabolicSARStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        af = self.params.get("af", 0.02)
        af_max = self.params.get("af_max", 0.2)
        high = df["high"].values
        low = df["low"].values
        n = len(df)
        psar = low[0] * 0.0
        ep = high[0]
        trend = 1
        sig = [0.0] * n
        for i in range(1, n):
            prev_psar = psar
            psar = prev_psar + af * (ep - prev_psar)
            if trend == 1:
                psar = min(psar, low[i-1], low[i-2]) if i > 1 else min(psar, low[i-1])
                if low[i] < psar:
                    trend = -1
                    psar = ep
                    ep = low[i]
                    af = self.params.get("af", 0.02)
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.params.get("af", 0.02), af_max)
                    sig[i] = 1.0
            else:
                psar = max(psar, high[i-1], high[i-2]) if i > 1 else max(psar, high[i-1])
                if high[i] > psar:
                    trend = 1
                    psar = ep
                    ep = high[i]
                    af = self.params.get("af", 0.02)
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.params.get("af", 0.02), af_max)
                    sig[i] = -1.0
        return pd.Series(sig, index=df.index)


class TurtleStrategy(BaseStrategy):
    name = "TurtleStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        entry = self.params.get("entry", 20)
        exit_period = self.params.get("exit", 10)
        entry_high = df["high"].rolling(entry).max()
        exit_low = df["low"].rolling(exit_period).min()
        sig = pd.Series(0.0, index=df.index)
        sig[df["close"] > entry_high.shift(1)] = 1.0
        sig[df["close"] < exit_low.shift(1)] = -1.0
        return sig


class VolumeMaStrategy(BaseStrategy):
    name = "VolumeMAStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        vol_period = self.params.get("vol_period", 20)
        vol_ma = df["volume"].rolling(vol_period).mean()
        price_ma = df["close"].rolling(vol_period).mean()
        sig = pd.Series(0.0, index=df.index)
        sig[(df["volume"] > vol_ma * 1.5) & (df["close"] > price_ma)] = 1.0
        sig[(df["volume"] > vol_ma * 1.5) & (df["close"] < price_ma)] = -1.0
        return sig


class IntradayVolatilityStrategy(BaseStrategy):
    name = "IntradayVolatilityStrategy"

    def signal(self, df: pd.DataFrame) -> pd.Series:
        period = self.params.get("period", 20)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        daily_range = df["high"] - df["low"]
        sig = pd.Series(0.0, index=df.index)
        sig[daily_range > atr * 1.5] = 1.0
        sig[daily_range < atr * 0.5] = -1.0
        return sig
