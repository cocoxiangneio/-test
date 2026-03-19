# -*- coding: utf-8 -*-
"""Portfolio optimizer - Mean-Variance, Min-Variance, Risk Parity, HRP."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    def __init__(self, risk_aversion: float = 1.0, allow_short: bool = False):
        self.risk_aversion = risk_aversion
        self.allow_short = allow_short

    def optimize(
        self,
        returns: pd.DataFrame,
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        cov = returns.cov()
        mu = returns.mean()
        n = len(mu)
        if n == 0:
            logger.warning("Empty returns provided to MeanVarianceOptimizer")
            return {}

        def objective(w):
            port_ret = float(np.dot(w, mu))
            port_var = float(np.dot(w, np.dot(cov.values, w)))
            return -(port_ret - self.risk_aversion * port_var)

        x0 = np.ones(n) / n
        bounds = None if self.allow_short else [(0, 1) for _ in range(n)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            weights = result.x
            weights = np.maximum(weights, 0) if not self.allow_short else weights
            weights /= weights.sum()
            return {returns.columns[i]: float(weights[i]) for i in range(n)}
        return {returns.columns[i]: 1.0 / n for i in range(n)}


class MinVarianceOptimizer:
    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        cov = returns.cov()
        n = returns.shape[1]
        if n == 0:
            logger.warning("Empty returns provided to MinVarianceOptimizer")
            return {}

        def objective(w):
            return float(np.dot(w, np.dot(cov.values, w)))

        x0 = np.ones(n) / n
        bounds = [(0, 1) for _ in range(n)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            weights = np.maximum(result.x, 0)
            weights /= weights.sum()
            return {returns.columns[i]: float(weights[i]) for i in range(n)}
        return {returns.columns[i]: 1.0 / n for i in range(n)}


class RiskParityOptimizer:
    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        cov = returns.cov()
        n = returns.shape[1]
        if n == 0:
            logger.warning("Empty returns provided to RiskParityOptimizer")
            return {}

        def objective(w):
            cov_matrix = cov.values
            port_var = np.dot(w, np.dot(cov_matrix, w))
            marginal_contrib = np.dot(cov_matrix, w)
            risk_contrib = w * marginal_contrib / np.sqrt(port_var + 1e-10)
            target_risk = np.sqrt(port_var + 1e-10) / n
            return float(np.sum((risk_contrib - target_risk) ** 2))

        x0 = np.ones(n) / n
        bounds = [(0.001, 1) for _ in range(n)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success:
            weights = np.maximum(result.x, 0.001)
            weights /= weights.sum()
            return {returns.columns[i]: float(weights[i]) for i in range(n)}
        return {returns.columns[i]: 1.0 / n for i in range(n)}


class HRPOptimizer:
    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        cov = returns.cov()
        corr = returns.corr()
        n = returns.shape[1]
        if n == 0:
            logger.warning("Empty returns provided to HRPOptimizer")
            return {}
        if n == 1:
            return {returns.columns[0]: 1.0}

        dist = np.sqrt(np.clip(0.5 * (1 - corr.values), 0, None))
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist)
        link = linkage(condensed, method="ward")

        def bisect(link_mat, labels):
            clusters = fcluster(link_mat, t=n, criterion="maxclust")
            return clusters

        n_clusters = bisect(link, returns.columns)
        sorted_idx = np.argsort(n_clusters)
        sorted_labels = returns.columns[sorted_idx]

        raw_weights = np.ones(n)
        for i in range(n):
            for j in range(i + 1, n):
                cov_ii = cov.values[sorted_idx[i], sorted_idx[i]]
                cov_jj = cov.values[sorted_idx[j], sorted_idx[j]]
                cov_ij = cov.values[sorted_idx[i], sorted_idx[j]]
                denom = cov_ii * cov_jj
                if denom > 0:
                    corr_val = cov_ij / np.sqrt(denom)
                    raw_weights[j] *= np.sqrt(max(0.0, 1.0 - corr_val))

        raw_weights = np.maximum(raw_weights, 1e-10)
        weights = raw_weights / raw_weights.sum()
        weights_dict = {sorted_labels[i]: float(weights[i]) for i in range(n)}
        return weights_dict


class PortfolioOptimizerFactory:
    _METHODS = {
        "mean_variance": MeanVarianceOptimizer,
        "min_variance": MinVarianceOptimizer,
        "risk_parity": RiskParityOptimizer,
        "hrp": HRPOptimizer,
    }

    @classmethod
    def create(cls, method: str = "mean_variance", **kwargs) -> MeanVarianceOptimizer:
        if method not in cls._METHODS:
            logger.warning(f"Unknown optimizer '{method}', using mean_variance")
            method = "mean_variance"
        return cls._METHODS[method](**kwargs)

    @classmethod
    def list_methods(cls) -> List[str]:
        return list(cls._METHODS.keys())
