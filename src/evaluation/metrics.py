# -*- coding: utf-8 -*-
"""Performance metrics calculation."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        logger.warning("Empty returns provided to sharpe_ratio")
        return 0.0
    std = returns.std()
    if std == 0:
        logger.warning("Zero-variance returns in sharpe_ratio: returns are constant or no variance")
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1)
    md = abs(max_drawdown(equity))
    return float(total_ret / md) if md != 0 else 0.0


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside.std())


def win_rate(trades: List) -> float:
    if not trades:
        return 0.0
    closed = [t for t in trades if hasattr(t, "pnl") and t.side in ("sell", "sell_short")]
    if not closed:
        return 0.0
    wins = sum(1 for t in closed if t.pnl > 0)
    return float(wins / len(closed))


def profit_loss_ratio(trades: List) -> float:
    closed = [t for t in trades if hasattr(t, "pnl") and t.pnl != 0]
    if not closed:
        return 0.0
    wins = [t.pnl for t in closed if t.pnl > 0]
    losses = [abs(t.pnl) for t in closed if t.pnl < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 1.0
    return avg_win / avg_loss if avg_loss > 0 else 0.0


def annual_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_ret) ** (1 / years) - 1)


class MetricsCalculator:
    def calculate(self, equity: pd.Series, returns: pd.Series = None, trades: List = None) -> Dict[str, float]:
        if returns is None:
            returns = equity.pct_change().dropna()
        return {
            "total_return": float((equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0),
            "annual_return": annual_return(equity),
            "sharpe_ratio": sharpe_ratio(returns),
            "sortino_ratio": sortino_ratio(returns),
            "max_drawdown": max_drawdown(equity),
            "calmar_ratio": calmar_ratio(equity),
            "volatility": float(returns.std() * np.sqrt(252)),
            "win_rate": win_rate(trades) if trades else 0.0,
            "profit_loss_ratio": profit_loss_ratio(trades) if trades else 0.0,
            "total_trades": len(trades) if trades else 0,
            "final_equity": float(equity.iloc[-1]) if len(equity) > 0 else 0.0,
        }

    def compare(self, results: Dict[str, pd.Series]) -> pd.DataFrame:
        calc = MetricsCalculator()
        rows = {}
        for name, equity in results.items():
            rows[name] = calc.calculate(equity)
        return pd.DataFrame(rows).T
