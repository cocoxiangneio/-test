# -*- coding: utf-8 -*-
"""Cross-sectional factors and IC/IR analysis."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def normalize_factor(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-10)


def rank_factor(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def calculate_ic(
    factor_data: pd.DataFrame,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    dates = factor_data.index.get_level_values(0).unique()
    ic_series = []
    for date in dates:
        try:
            factor_vals = factor_data.xs(date, level=0) if isinstance(factor_data.index, pd.MultiIndex) else factor_data.loc[date]
            ret_vals = forward_returns.xs(date, level=0) if isinstance(forward_returns.index, pd.MultiIndex) else forward_returns
            common_idx = factor_vals.dropna().index.intersection(ret_vals.dropna().index)
            if len(common_idx) < 10:
                ic_series.append(np.nan)
                continue
            f = factor_vals.loc[common_idx]
            r = ret_vals.loc[common_idx]
            if method == "spearman":
                ic = f.rank().corr(r.rank())
            else:
                ic = f.corr(r)
            ic_series.append(ic)
        except Exception:
            ic_series.append(np.nan)
    return pd.Series(ic_series, index=dates)


def calculate_ir(
    ic_series: pd.Series,
    rolling_window: int = 12,
) -> pd.Series:
    ic_mean = ic_series.rolling(rolling_window).mean()
    ic_std = ic_series.rolling(rolling_window).std()
    return ic_mean / (ic_std + 1e-10)


class CrossSectionalAnalyzer:
    def __init__(self):
        pass

    def factor_ic_analysis(
        self,
        factor_panel: pd.DataFrame,
        return_panel: pd.DataFrame,
        method: str = "spearman",
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for col in factor_panel.columns:
            ic = calculate_ic(factor_panel[[col]], return_panel, method)
            ir = calculate_ir(ic)
            results[col] = pd.DataFrame({
                "IC": ic,
                "IC_mean": ic.rolling(12).mean(),
                "IC_std": ic.rolling(12).std(),
                "IR": ir,
            })
        return results

    def neutralize(
        self,
        factor: pd.Series,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
    ) -> pd.Series:
        factor_norm = normalize_factor(factor)
        return factor_norm

    def calculate_portfolio_zscore(
        self,
        factor_data: pd.DataFrame,
    ) -> pd.DataFrame:
        return factor_data.apply(normalize_factor)

    def quantile_returns(
        self,
        factor_data: pd.DataFrame,
        return_data: pd.DataFrame,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        results = []
        for date in factor_data.index:
            try:
                f = factor_data.loc[date]
                r = return_data.loc[date] if date in return_data.index else None
                if r is None:
                    continue
                quantiles = pd.qcut(f, n_quantiles, labels=False, duplicates="drop") + 1
                qret = {}
                for q in range(1, n_quantiles + 1):
                    mask = quantiles == q
                    if mask.sum() > 0:
                        qret[f"Q{q}"] = r[mask].mean()
                if qret:
                    qret["long_short"] = qret.get(f"Q{n_quantiles}", 0) - qret.get("Q1", 0)
                    qret["date"] = date
                    results.append(qret)
            except Exception as e:
                logger.warning(f"quantile_returns failed at {date}: {e}")
        return pd.DataFrame(results).set_index("date") if results else pd.DataFrame()
