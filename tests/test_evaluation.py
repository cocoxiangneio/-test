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


def test_pdf_report_generator(tmp_path):
    from src.evaluation.report import PDFReportGenerator
    import os
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        gen = PDFReportGenerator(output_dir="test_pdf_results")
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        np.random.seed(42)
        equity1 = pd.Series(100 * np.exp(np.cumsum(np.random.randn(60) * 0.01)), index=dates)
        equity2 = pd.Series(100 * np.exp(np.cumsum(np.random.randn(60) * 0.015)), index=dates)
        results = {
            "StrategyA": {"equity_curve": equity1},
            "StrategyB": {"equity_curve": equity2},
        }
        path = gen.save_comparison_pdf(results, "test_comparison.pdf")
        assert os.path.exists(path)
        assert path.endswith(".pdf")
        with open(path, "rb") as f:
            header = f.read(4)
        assert header == b"%PDF", "File should be a valid PDF"
    finally:
        os.chdir(orig_dir)


def test_pdf_report_with_metrics(tmp_path):
    from src.evaluation.report import PDFReportGenerator
    import os
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        gen = PDFReportGenerator(output_dir="test_pdf_results2")
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(99)
        results = {
            f"Strategy{i}": {"equity_curve": pd.Series(
                100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)),
                index=dates,
            )}
            for i in range(3)
        }
        path = gen.save_comparison_pdf(results, "multi_strategy.pdf")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.chdir(orig_dir)
