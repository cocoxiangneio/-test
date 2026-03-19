# -*- coding: utf-8 -*-
"""Risk manager and rebalancer."""

from __future__ import annotations
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 0.2,
        max_total_leverage: float = 1.0,
        var_confidence: float = 0.95,
        var_horizon: int = 1,
    ):
        self.max_position_pct = max_position_pct
        self.max_total_leverage = max_total_leverage
        self.var_confidence = var_confidence
        self.var_horizon = var_horizon

    def check_position_limits(self, weights: Dict[str, float], total_value: float) -> Dict[str, float]:
        capped = {}
        for stock, weight in weights.items():
            capped[stock] = min(weight, self.max_position_pct)
        total_capped = sum(capped.values())
        if total_capped > 1.0:
            capped = {k: v / total_capped for k, v in capped.items()}
        return capped

    def calculate_var(self, returns: pd.Series, portfolio_value: float) -> float:
        if len(returns) == 0:
            return 0.0
        var = float(np.percentile(returns, (1 - self.var_confidence) * 100))
        return portfolio_value * var * np.sqrt(self.var_horizon)

    def check_var_risk(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,
        portfolio_value: float,
    ) -> bool:
        if len(returns) == 0:
            return True
        weighted_returns = pd.Series(0.0, index=returns.index)
        for stock, weight in positions.items():
            if stock in returns.columns:
                weighted_returns += returns[stock] * weight
        var = self.calculate_var(weighted_returns, portfolio_value)
        return abs(var) < portfolio_value * 0.05


class Rebalancer:
    def __init__(self, rebalance_threshold: float = 0.05, rebalance_freq: str = "monthly"):
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_freq = rebalance_freq

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        for stock, target in target_weights.items():
            current = current_weights.get(stock, 0.0)
            if abs(current - target) > self.rebalance_threshold:
                return True
        return False

    def get_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float,
    ) -> Dict[str, float]:
        current_value = {k: v * total_value for k, v in current_weights.items()}
        target_value = {k: v * total_value for k, v in target_weights.items()}
        trades = {}
        for stock in set(list(current_weights.keys()) + list(target_weights.keys())):
            cur = current_value.get(stock, 0.0)
            tgt = target_value.get(stock, 0.0)
            diff = tgt - cur
            if abs(diff) > total_value * 0.001:
                trades[stock] = diff
        return trades
