# -*- coding: utf-8 -*-
"""Grid search optimizer for strategy parameters."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    def __init__(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 1,
        n_jobs: int = 1,
        seed: Optional[int] = None,
    ):
        self.param_grid = param_grid or {}
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_weights_: Optional[Dict[str, Any]] = None
        self.results_: List[Dict[str, Any]] = []
        self.n_combinations_: int = 0

    def _generate_param_combinations(
        self,
    ) -> List[Dict[str, Any]]:
        if not self.param_grid:
            return [{}]
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def _score_to_metrics(self, score: float, equity_curve: pd.Series) -> Dict[str, float]:
        returns = equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
        equity = equity_curve
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
        total_ret = float((equity.iloc[-1] / equity.iloc[0] - 1)) if len(equity) > 0 else 0.0
        calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0
        return {
            "sharpe": sharpe,
            "calmar": calmar,
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "score": score,
        }

    def search(
        self,
        objective_func: Callable[..., float],
        n_weights: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
        equity_curve_func: Optional[Callable[..., pd.Series]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        if self.seed is not None:
            np.random.seed(self.seed)

        param_combinations = self._generate_param_combinations()
        self.n_combinations_ = len(param_combinations)
        self.results_ = []

        for params in param_combinations:
            score = objective_func(**params, **kwargs)
            metrics = {"params": params, "score": score}
            if equity_curve_func is not None:
                equity = equity_curve_func(**params, **kwargs)
                extra = self._score_to_metrics(score, equity)
                metrics.update(extra)

            self.results_.append(metrics)

            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params

        return self.best_params_, self.best_score_, self.results_


class GridSearchWithBacktest:
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        backtest_func: Callable[..., Dict[str, float]],
        metric: str = "sharpe_ratio",
        seed: Optional[int] = None,
    ):
        self.param_grid = param_grid
        self.backtest_func = backtest_func
        self.metric = metric
        self.seed = seed
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.results_: List[Dict[str, Any]] = []
        self.n_combinations_: int = 0

    def search(self, **kwargs) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        if self.seed is not None:
            np.random.seed(self.seed)

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        self.n_combinations_ = len(combinations)
        self.results_ = []

        for combo in combinations:
            params = dict(zip(keys, combo))
            bt_result = self.backtest_func(**params, **kwargs)
            score = bt_result.get(self.metric, bt_result.get("score", 0.0))
            result = {"params": params, self.metric: score}
            result.update(bt_result)
            self.results_.append(result)

            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params

        return self.best_params_, self.best_score_, self.results_
