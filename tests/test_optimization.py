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


def test_grid_search_basic():
    from src.optimization.grid_search import GridSearchOptimizer

    gs = GridSearchOptimizer(
        param_grid={"period": [5, 10, 20], "threshold": [0.01, 0.02]},
        seed=42,
    )
    obj = lambda period, threshold: float(period * threshold)
    best_p, best_s, results = gs.search(obj, n_weights=0)

    assert gs.n_combinations_ == 6
    assert best_p is not None
    assert "period" in best_p
    assert "threshold" in best_p
    assert len(results) == 6
    assert all("params" in r for r in results)
    assert all("score" in r for r in results)


def test_grid_search_with_metrics():
    from src.optimization.grid_search import GridSearchOptimizer
    import pandas as pd

    np.random.seed(99)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    equity = pd.Series(100 * np.exp(np.cumsum(np.random.randn(60) * 0.01)), index=dates)

    gs = GridSearchOptimizer(param_grid={"alpha": [0.1, 0.2]}, seed=99)
    obj = lambda alpha: alpha * 1.5
    equity_func = lambda alpha: equity * alpha
    best_p, best_s, results = gs.search(obj, n_weights=0, equity_curve_func=equity_func)

    assert best_p is not None
    assert len(results) == 2
    for r in results:
        assert "sharpe" in r
        assert "calmar" in r
        assert "total_return" in r


def test_grid_search_with_backtest():
    from src.optimization.grid_search import GridSearchWithBacktest

    def dummy_backtest(sma_period, rsi_period, stop_loss):
        return {
            "sharpe_ratio": float(sma_period * 0.1 - rsi_period * 0.05 + stop_loss),
            "total_return": float(sma_period * 0.01),
            "max_drawdown": -0.05,
        }

    gs = GridSearchWithBacktest(
        param_grid={
            "sma_period": [10, 20, 30],
            "rsi_period": [10, 14],
            "stop_loss": [0.03, 0.05],
        },
        backtest_func=dummy_backtest,
        metric="sharpe_ratio",
        seed=42,
    )
    best_p, best_s, results = gs.search()

    assert gs.n_combinations_ == 12
    assert best_p is not None
    assert "sma_period" in best_p
    assert "rsi_period" in best_p
    assert "stop_loss" in best_p
    assert len(results) == 12
    assert all("sharpe_ratio" in r for r in results)
    assert all("total_return" in r for r in results)
    max_sharpe = max(r["sharpe_ratio"] for r in results)
    assert best_s == max_sharpe


def test_grid_search_stability():
    from src.optimization.grid_search import GridSearchOptimizer

    gs1 = GridSearchOptimizer(param_grid={"p": [1, 2, 3]}, seed=0)
    obj = lambda p: float(p * 2)
    _, score1, _ = gs1.search(obj, n_weights=0)

    gs2 = GridSearchOptimizer(param_grid={"p": [1, 2, 3]}, seed=0)
    _, score2, _ = gs2.search(obj, n_weights=0)

    assert score1 == score2


def _schaffer(x):
    return [x[0] ** 2, (x[0] - 2) ** 2]


def test_nsga2_returns_pareto_front():
    from src.optimization.nsga_optimizer import NSGAII

    nsga = NSGAII(n_objectives=2, pop_size=100, n_generations=30, seed=42)
    bounds = [(-10.0, 10.0)]
    pareto = nsga.optimize([lambda x: x[0] ** 2, lambda x: (x[0] - 2) ** 2], bounds)

    assert len(pareto) >= 5, "Pareto front should have at least 5 solutions"
    objectives_only = [obj for _, obj in pareto]
    obj0_vals = sorted([o[0] for o in objectives_only])
    assert all(obj0_vals[i] <= obj0_vals[i + 1] for i in range(len(obj0_vals) - 1))


def test_nsga2_pareto_nondominated():
    from src.optimization.nsga_optimizer import NSGAII

    nsga = NSGAII(n_objectives=2, pop_size=80, n_generations=25, seed=42)
    bounds = [(-5.0, 5.0)]
    pareto = nsga.optimize([lambda x: x[0] ** 2, lambda x: (x[0] - 2) ** 2], bounds)

    for i, (g1, o1) in enumerate(pareto):
        for j, (g2, o2) in enumerate(pareto):
            if i != j:
                assert not (o1[0] <= o2[0] and o1[1] <= o2[1] and (o1[0] < o2[0] or o1[1] < o2[1])), \
                    f"Solution {i} should not dominate {j}"


def test_nsga2_multiobjective_tradeoff():
    from src.optimization.nsga_optimizer import NSGAII

    nsga = NSGAII(n_objectives=3, pop_size=100, n_generations=30, seed=42)
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def obj1(x): return x[0]
    def obj2(x): return x[1]
    def obj3(x): return 1.0 - x[0] - x[1]

    pareto = nsga.optimize([obj1, obj2, obj3], bounds)
    assert len(pareto) >= 3, "3-objective should yield diverse front"

    objectives = [obj for _, obj in pareto]
    obj0_vals = [o[0] for o in objectives]
    assert max(obj0_vals) - min(obj0_vals) > 0.1, "Objectives should vary across front"
