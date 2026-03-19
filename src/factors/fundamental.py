# -*- coding: utf-8 -*-
"""Fundamental factors."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import numpy as np

from .factor_registry import register_factor


@register_factor("pe_ratio", category="fundamental", description="Price-to-Earnings ratio", window_required=1)
def pe_ratio(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("pe_ratio")
    return float(val) if val is not None else np.nan


@register_factor("pb_ratio", category="fundamental", description="Price-to-Book ratio", window_required=1)
def pb_ratio(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("pb_ratio")
    return float(val) if val is not None else np.nan


@register_factor("ps_ratio", category="fundamental", description="Price-to-Sales ratio", window_required=1)
def ps_ratio(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("ps_ratio")
    return float(val) if val is not None else np.nan


@register_factor("roe", category="fundamental", description="Return on Equity", window_required=1)
def roe(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("roe")
    return float(val) if val is not None else np.nan


@register_factor("roa", category="fundamental", description="Return on Assets", window_required=1)
def roa(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("roa")
    return float(val) if val is not None else np.nan


@register_factor("gross_margin", category="fundamental", description="Gross profit margin", window_required=1)
def gross_margin(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("gross_margin")
    return float(val) if val is not None else np.nan


@register_factor("debt_ratio", category="fundamental", description="Debt-to-Assets ratio", window_required=1)
def debt_ratio(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("debt_ratio")
    return float(val) if val is not None else np.nan


@register_factor("net_profit_growth", category="fundamental", description="Net profit growth rate", window_required=1)
def net_profit_growth(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("net_profit_growth")
    return float(val) if val is not None else np.nan


@register_factor("revenue_growth", category="fundamental", description="Revenue growth rate", window_required=1)
def revenue_growth(fundamentals: Dict[str, Any]) -> float:
    val = fundamentals.get("revenue_growth")
    return float(val) if val is not None else np.nan


class FundamentalFactorCalculator:
    def __init__(self):
        pass

    def calculate(self, fundamentals: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for name in ["pe_ratio", "pb_ratio", "ps_ratio", "roe", "roa",
                      "gross_margin", "debt_ratio", "net_profit_growth", "revenue_growth"]:
            try:
                result[name] = globals()[name](fundamentals)
            except Exception:
                result[name] = np.nan
        return result

    def batch_calculate(
        self,
        fundamentals_dict: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        records = {}
        for stock, fundamentals in fundamentals_dict.items():
            records[stock] = self.calculate(fundamentals)
        return pd.DataFrame(records).T
