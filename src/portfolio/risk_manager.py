# -*- coding: utf-8 -*-
"""Risk manager and rebalancer."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

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


class AdvancedRiskMetrics:
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def calculate_cvar(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        if len(returns) == 0:
            return 0.0
        var_threshold = np.percentile(returns, (1 - self.confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) == 0:
            return 0.0
        cvar = float(tail_losses.mean())
        return portfolio_value * cvar

    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        if len(returns) == 0:
            return 0.0
        excess = returns - threshold
        gain = excess[excess > 0].sum()
        loss = abs(excess[excess < 0].sum())
        if loss == 0:
            return float("inf") if gain > 0 else 1.0
        return float(gain / loss)

    def calculate_tail_ratio(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        upper_tail = np.percentile(returns, 95)
        lower_tail = np.percentile(returns, 5)
        upper_mean = returns[returns >= upper_tail].mean() if len(returns[returns >= upper_tail]) > 0 else 0.0
        lower_mean = abs(returns[returns <= lower_tail].mean()) if len(returns[returns <= lower_tail]) > 0 else 1e-10
        return float(upper_mean / lower_mean)

    def calculate_max_consecutive_losses(self, returns: pd.Series) -> int:
        if len(returns) == 0:
            return 0
        is_loss = (returns < 0).astype(int)
        max_streak = 0
        current_streak = 0
        for loss in is_loss:
            if loss:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return int(max_streak)

    def calculate_var(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        if len(returns) == 0:
            return 0.0
        var = float(np.percentile(returns, (1 - self.confidence) * 100))
        return portfolio_value * var

    def calculate_all(self, returns: pd.Series, portfolio_value: float = 1.0) -> Dict:
        return {
            "cvar": self.calculate_cvar(returns, portfolio_value),
            "omega_ratio": self.calculate_omega_ratio(returns),
            "tail_ratio": self.calculate_tail_ratio(returns),
            "max_consecutive_losses": self.calculate_max_consecutive_losses(returns),
            "var_95": self.calculate_var(returns, portfolio_value),
        }


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
