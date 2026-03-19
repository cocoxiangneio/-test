# -*- coding: utf-8 -*-
"""Bayesian optimizer for factor weight optimization using Optuna."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    def __init__(
        self,
        n_trials: int = 100,
        n_startup: int = 10,
        seed: Optional[int] = None,
    ):
        self.n_trials = n_trials
        self.n_startup = n_startup
        self.seed = seed
        self.best_weights_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.history_: List[float] = []
        self.n_weights_: int = 0

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        n_weights: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, float, List[float]]:
        self.n_weights_ = n_weights
        bounds = bounds or [(0.0, 1.0) for _ in range(n_weights)]

        def optuna_objective(trial: optuna.Trial) -> float:
            weights = np.array([trial.suggest_float(f"w{i}", bounds[i][0], bounds[i][1])
                                for i in range(n_weights)])
            wsum = weights.sum()
            if wsum > 0:
                weights /= wsum
            return objective_func(weights)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(optuna_objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        weights = np.array([best_params[f"w{i}"] for i in range(n_weights)])
        weights /= weights.sum()
        self.best_weights_ = weights
        self.best_fitness_ = float(study.best_value)

        for trial in study.trials:
            if trial.value is not None:
                self.history_.append(trial.value)

        return self.best_weights_, self.best_fitness_, self.history_

    def get_convergence(self) -> List[float]:
        return self.history_


class ObjectiveFunctions:
    @staticmethod
    def sharpe(returns: pd.DataFrame, weights: np.ndarray) -> float:
        weighted_ret = (returns * weights).sum(axis=1)
        if weighted_ret.std() == 0:
            return 0.0
        return float(np.sqrt(252) * weighted_ret.mean() / weighted_ret.std())

    @staticmethod
    def calmar(returns: pd.DataFrame, weights: np.ndarray) -> float:
        weighted_ret = (returns * weights).sum(axis=1)
        cummax = (1 + weighted_ret).cumprod().cummax()
        drawdown = (1 + weighted_ret).cumprod() / cummax - 1
        max_dd = abs(drawdown.min())
        ret = (1 + weighted_ret).prod() - 1
        return float(ret / max_dd) if max_dd > 0 else 0.0

    @staticmethod
    def return_drawdown(returns: pd.DataFrame, weights: np.ndarray) -> float:
        weighted_ret = (returns * weights).sum(axis=1)
        total_ret = (1 + weighted_ret).prod() - 1
        cummax = (1 + weighted_ret).cumprod().cummax()
        drawdown = (1 + weighted_ret).cumprod() / cummax - 1
        max_dd = abs(drawdown.min())
        return float(total_ret / max_dd) if max_dd > 0 else 0.0

    @staticmethod
    def composite(
        returns: pd.DataFrame,
        weights: np.ndarray,
        sharpe_w: float = 0.5,
        calmar_w: float = 0.3,
        ret_dd_w: float = 0.2,
    ) -> float:
        s = ObjectiveFunctions.sharpe(returns, weights)
        c = ObjectiveFunctions.calmar(returns, weights)
        rd = ObjectiveFunctions.return_drawdown(returns, weights)
        return sharpe_w * s + calmar_w * c + ret_dd_w * rd

