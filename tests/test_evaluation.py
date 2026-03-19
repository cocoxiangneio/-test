# -*- coding: utf-8 -*-
"""Tests for evaluation layer."""

import pytest
import numpy as np
import pandas as pd


def test_metrics_calculator():
    from src.evaluation.metrics import MetricsCalculator, sharpe_ratio, max_drawdown

    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    equity = pd.Series(100000 * (1 + np.random.randn(252) * 0.01).cumprod(), index=dates)
    returns = equity.pct_change().dropna()

    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)

    md = max_drawdown(equity)
    assert isinstance(md, float)
    assert md <= 0

    calc = MetricsCalculator()
    metrics = calc.calculate(equity, returns)
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "total_return" in metrics


def test_report_generator(tmp_path):
    from src.evaluation.report import ReportGenerator
    import os
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        gen = ReportGenerator(output_dir="test_results")
        os.makedirs("test_results", exist_ok=True)
        path = gen.save_json({"a": 1}, "test.json")
        assert os.path.exists(path)
        df = pd.DataFrame({"col": [1, 2, 3]})
        path2 = gen.save_csv(df, "test.csv")
        assert os.path.exists(path2)
    finally:
        os.chdir(orig_dir)
