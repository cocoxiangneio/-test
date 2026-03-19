# -*- coding: utf-8 -*-
"""Tests for optimization algorithms."""

import pytest
import numpy as np
import pandas as pd


def test_ga_optimizer():
    from src.optimization.ga_optimizer import GAOptimizer

    opt = GAOptimizer(n_generations=10, pop_size=20, seed=42)
    obj = lambda w: float(np.sum(w * np.array([0.1, 0.2, 0.3, 0.4])))
    best_w, best_f, history = opt.optimize(obj, 4)

    assert len(best_w) == 4
    assert best_f > 0
    assert len(history) == 11
    assert all(history[i] >= history[i-1] for i in range(1, len(history)))


def test_pso_optimizer():
    from src.optimization.pso_optimizer import PSOOptimizer

    opt = PSOOptimizer(n_particles=20, n_iterations=10, seed=42)
    obj = lambda w: float(np.sum(w * np.array([0.1, 0.2, 0.3])))
    best_w, best_f, history = opt.optimize(obj, 3)

    assert len(best_w) == 3
    assert len(history) == 11


def test_bayesian_optimizer():
    from src.optimization.bayesian_optimizer import BayesianOptimizer

    opt = BayesianOptimizer(n_trials=20, seed=42)
    obj = lambda w: float(np.sum(w * np.array([0.1, 0.2, 0.3])))
    best_w, best_f, history = opt.optimize(obj, 3)

    assert len(best_w) == 3
    assert len(history) > 0


def test_objective_functions():
    from src.optimization.bayesian_optimizer import ObjectiveFunctions
    import pandas as pd
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = pd.DataFrame({
        "s1": np.random.randn(100) * 0.02,
        "s2": np.random.randn(100) * 0.02,
    }, index=dates)
    weights = np.array([0.5, 0.5])

    sharpe = ObjectiveFunctions.sharpe(returns, weights)
    calmar = ObjectiveFunctions.calmar(returns, weights)
    comp = ObjectiveFunctions.composite(returns, weights)

    assert isinstance(sharpe, float)
    assert isinstance(calmar, float)
    assert isinstance(comp, float)
