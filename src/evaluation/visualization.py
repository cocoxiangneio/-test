# -*- coding: utf-8 -*-
"""Visualization utilities."""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=1.5)
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label="Benchmark", alpha=0.7, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    ax.fill_between(equity.index, drawdown.values, 0, alpha=0.3, color="red")
    ax.plot(equity.index, drawdown.values, color="red", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(
    history: List[float],
    title: str = "Optimization Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Generation / Iteration")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_weights_heatmap(
    weights_dict: Dict[str, float],
    title: str = "Portfolio Weights",
    save_path: Optional[str] = None,
) -> plt.Figure:
    if not weights_dict:
        return plt.figure()
    stocks = list(weights_dict.keys())
    weights = list(weights_dict.values())
    fig, ax = plt.subplots(figsize=(max(4, len(stocks) * 0.5), 3))
    colors = ["green" if w > 0 else "red" for w in weights]
    ax.barh(stocks, weights, color=colors, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Weight")
    ax.grid(True, alpha=0.3, axis="x")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ic_analysis(
    ic_series: pd.Series,
    title: str = "IC Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ic_mean = ic_series.rolling(12).mean()
    ic_std = ic_series.rolling(12).std()
    ir = ic_mean / (ic_std + 1e-10)
    axes[0].plot(ic_series.index, ic_series.values, linewidth=1, alpha=0.7)
    axes[0].plot(ic_mean.index, ic_mean.values, linewidth=2, label="IC MA(12)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title(f"{title} - IC Time Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ir.index, ir.values, color="purple", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title(f"{title} - IR (IC Mean/Std)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
