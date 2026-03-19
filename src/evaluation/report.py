# -*- coding: utf-8 -*-
"""Report generation."""

import json
import os
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PDFReportGenerator:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _compute_metrics(self, equity: pd.Series) -> Dict[str, float]:
        if len(equity) == 0:
            return {}
        returns = equity.pct_change().dropna()
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] != 0 else 0.0
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())
        calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0
        wins = sum(1 for i in range(1, len(returns)) if returns.iloc[i] > 0)
        win_rate = wins / max(len(returns), 1)
        return {
            "Total Return": total_ret,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "Win Rate": win_rate,
        }

    def plot_equity_drawdown(
        self,
        equities: Dict[str, pd.Series],
        title: str = "Equity Curve & Drawdown",
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for name, equity in equities.items():
            axes[0].plot(equity.index, equity.values, label=name, linewidth=1.5)
            peak = equity.cummax()
            drawdown = (equity - peak) / peak
            axes[1].fill_between(equity.index, drawdown.values, 0, alpha=0.2)
            axes[1].plot(equity.index, drawdown.values, linewidth=0.8)
        axes[0].set_title(title)
        axes[0].set_ylabel("Equity")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def save_comparison_pdf(
        self,
        results: Dict[str, Dict[str, Any]],
        filename: str = "strategy_comparison.pdf",
    ) -> str:
        path = os.path.join(self.output_dir, filename)
        equities = {}
        metrics_table = []
        for name, result in results.items():
            equity = result.get("equity_curve")
            if equity is None:
                equity = result.get("equity", pd.Series())
            if not isinstance(equity, pd.Series) or len(equity) == 0:
                equity = pd.Series([1.0], index=[0])
            equities[name] = equity
            metrics = self._compute_metrics(equity)
            metrics_table.append({"Strategy": name, **metrics})
        df_metrics = pd.DataFrame(metrics_table).set_index("Strategy")
        with PdfPages(path) as pdf:
            fig = self.plot_equity_drawdown(equities)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig_table, ax = plt.subplots(figsize=(12, max(3, len(df_metrics) * 0.5 + 1)))
            ax.axis("off")
            table = ax.table(
                cellText=df_metrics.round(4).values,
                colLabels=df_metrics.columns,
                rowLabels=df_metrics.index,
                cellLoc="center",
                rowLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax.set_title("Strategy Comparison", fontsize=14, pad=20)
            pdf.savefig(fig_table, bbox_inches="tight")
            plt.close(fig_table)
        return path


class ReportGenerator:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self, data: Dict[str, Any], filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return path

    def save_csv(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=True, encoding="utf-8-sig")
        return path

    def save_metrics_report(
        self,
        metrics: Dict[str, float],
        strategy_name: str,
        stock: str,
    ) -> str:
        report = {
            "strategy": strategy_name,
            "stock": stock,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        filename = f"{stock}_{strategy_name}_metrics.json"
        return self.save_json(report, filename)

    def save_portfolio_summary(
        self,
        portfolio_results: Dict[str, Dict[str, Any]],
    ) -> tuple:
        records = []
        for stock, data in portfolio_results.items():
            row = {"stock": stock}
            row.update(data.get("metrics", {}))
            row["best_method"] = data.get("best_method", "N/A")
            row["score"] = data.get("score", 0.0)
            records.append(row)
        df = pd.DataFrame(records)
        csv_path = self.save_csv(df, "portfolio_summary.csv")
        json_path = self.save_json(portfolio_results, "portfolio_summary.json")
        return csv_path, json_path
