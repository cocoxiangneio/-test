# -*- coding: utf-8 -*-
"""ML-enhanced strategies."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np

from ..base import BaseStrategy

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    name = "MLStrategy"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self._model = None
        self._feature_cols: list = []

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        feats["return_1d"] = df["close"].pct_change(1)
        feats["return_5d"] = df["close"].pct_change(5)
        feats["return_10d"] = df["close"].pct_change(10)
        feats["vol_5d"] = df["close"].pct_change().rolling(5).std()
        feats["vol_10d"] = df["close"].pct_change().rolling(10).std()
        feats["rsi_14"] = self._calc_rsi(df["close"], 14)
        feats["ma5_ma20"] = df["close"].rolling(5).mean() / (df["close"].rolling(20).mean() + 1e-10)
        feats = feats.dropna()
        self._feature_cols = feats.columns.tolist()
        return feats

    def _calc_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def signal(self, df: pd.DataFrame) -> pd.Series:
        feats = self._build_features(df)
        if len(feats) < 30 or self._model is None:
            return pd.Series(0.0, index=df.index)
        try:
            preds = self._model.predict(feats.values)
            aligned_preds = pd.Series(np.nan, index=df.index)
            aligned_preds.loc[feats.index] = preds
            aligned_preds = aligned_preds.fillna(0.0)
            sig = pd.Series(0.0, index=df.index)
            sig[aligned_preds > 0] = 1.0
            sig[aligned_preds < 0] = -1.0
            return sig
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return pd.Series(0.0, index=df.index)


class MLEnsembleStrategy(BaseStrategy):
    name = "MLEnsembleStrategy"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self._models = []

    def signal(self, df: pd.DataFrame) -> pd.Series:
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_val = (macd - signal).fillna(0)

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + gain / (loss + 1e-10)))).fillna(50)

        sig = pd.Series(0.0, index=df.index)
        sig[(macd_val > 0) & (rsi < 50)] = 1.0
        sig[(macd_val < 0) & (rsi > 50)] = -1.0
        return sig
