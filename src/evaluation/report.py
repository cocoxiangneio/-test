# -*- coding: utf-8 -*-
"""Report generation."""

import json
import os
from typing import Dict, Any

import pandas as pd


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
