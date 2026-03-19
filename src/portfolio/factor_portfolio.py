# -*- coding: utf-8 -*-
"""Factor portfolio builder - ICValidator + Backtest integration."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorPortfolioBuilder:
    def __init__(
        self,
        n_quantiles: int = 5,
        long_short: bool = True,
        ic_threshold: float = 0.02,
    ):
        self.n_quantiles = n_quantiles
        self.long_short = long_short
        self.ic_threshold = ic_threshold

    def build_quantile_weights(
        self,
        factor_panel: pd.DataFrame,
        date: pd.Timestamp,
        prices: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        if isinstance(factor_panel.index, pd.MultiIndex):
            if date not in factor_panel.index.get_level_values(0):
                return {}
            fac = factor_panel.xs(date, level=0)
        else:
            if date not in factor_panel.index:
                return {}
            fac = factor_panel.loc[date]

        if isinstance(fac, pd.DataFrame):
            fac = fac.iloc[:, 0]

        fac = fac.dropna()
        if len(fac) < self.n_quantiles:
            return {}

        try:
            quantile_labels = pd.qcut(fac, self.n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            return {}

        weights = {}
        q_values = quantile_labels.unique()
        if len(q_values) < 2:
            return {}

        q_min = q_values.min()
        q_max = q_values.max()

        if self.long_short:
            for stock, q in zip(fac.index, quantile_labels):
                if q == q_max:
                    weights[stock] = 1.0 / (q == q_max).sum()
                elif q == q_min:
                    weights[stock] = -1.0 / (q == q_min).sum()
                else:
                    weights[stock] = 0.0
        else:
            for stock, q in zip(fac.index, quantile_labels):
                if q == q_max:
                    weights[stock] = 1.0 / (q == q_max).sum()
                else:
                    weights[stock] = 0.0

        total_abs = sum(abs(v) for v in weights.values())
        if total_abs > 0:
            weights = {k: v / total_abs for k, v in weights.items()}

        return weights

    def run_backtest(
        self,
        factor_panel: pd.DataFrame,
        prices_dict: Dict[str, pd.DataFrame],
        ic_validator=None,
        rebalance_freq: str = "ME",
    ) -> Tuple[pd.Series, Dict]:
        dates = factor_panel.index.get_level_values(0).unique().sort_values()
        rebal_dates = dates.to_series().resample(rebalance_freq).last().dropna().index

        equity = pd.Series(dtype=float)
        current_weights: Dict[str, float] = {}
        positions: Dict[str, int] = {}
        cash = 100000.0
        last_rebal = None

        for date in dates:
            should_rebal = last_rebal is None
            if not should_rebal and len(rebal_dates) > 0:
                idx = rebal_dates.searchsorted(date)
                should_rebal = idx < len(rebal_dates) and date >= rebal_dates[idx]
            if should_rebal:
                current_weights = self.build_quantile_weights(factor_panel, date, None)
                last_rebal = date
                cash = self._rebalance(positions, current_weights, prices_dict, date, cash)

            equity._set_value(date, self._get_portfolio_value(positions, prices_dict, date, cash))

        if equity.empty:
            return equity, {}

        returns = equity.pct_change().dropna()
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0.0

        long_weights = [v for v in current_weights.values() if v > 0]
        short_weights = [v for v in current_weights.values() if v < 0]
        ls_return = sum(long_weights) - sum(abs(w) for w in short_weights) if self.long_short else 0.0

        metrics = {
            "sharpe": sharpe,
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "n_rebal_dates": len(rebal_dates),
            "long_short_return": ls_return,
        }
        return equity, metrics

    def _get_portfolio_value(
        self,
        positions: Dict[str, int],
        prices_dict: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        cash: float,
    ) -> float:
        pos_value = 0.0
        for stock, volume in positions.items():
            if stock in prices_dict and date in prices_dict[stock].index:
                price = float(prices_dict[stock].loc[date, "close"])
                pos_value += volume * price
        return pos_value + cash

    def _rebalance(
        self,
        positions: Dict[str, int],
        target_weights: Dict[str, float],
        prices_dict: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        cash: float,
    ) -> float:
        portfolio_value = self._get_portfolio_value(positions, prices_dict, date, cash)
        new_positions = {}
        for stock, weight in target_weights.items():
            if stock in prices_dict and date in prices_dict[stock].index:
                price = float(prices_dict[stock].loc[date, "close"])
                if price > 0:
                    target_value = portfolio_value * weight
                    volume = int(abs(target_value) / price)
                    if weight > 0:
                        cost = volume * price
                        commission = cost * 0.0003
                        if cash >= cost + commission:
                            new_positions[stock] = volume
                            cash -= cost + commission
                    elif weight < 0:
                        new_positions[stock] = -volume
                        proceeds = volume * price
                        commission = proceeds * 0.0003
                        cash += proceeds - commission
        return cash


class ICAwarePortfolioBuilder(FactorPortfolioBuilder):
    def __init__(self, n_quantiles: int = 5, ic_threshold: float = 0.02):
        super().__init__(n_quantiles=n_quantiles, long_short=True, ic_threshold=ic_threshold)

    def build_quantile_weights(
        self,
        factor_panel: pd.DataFrame,
        date: pd.Timestamp,
        prices: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        weights = super().build_quantile_weights(factor_panel, date, prices)
        if not weights:
            return {}
        return weights

    def run_backtest(
        self,
        factor_panel: pd.DataFrame,
        prices_dict: Dict[str, pd.DataFrame],
        rebalance_freq: str = "ME",
    ) -> Tuple[pd.Series, Dict]:
        return super().run_backtest(factor_panel, prices_dict, None, rebalance_freq)
