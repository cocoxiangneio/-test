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


class ICValidator:
    def __init__(self, ic_threshold: float = 0.02, ir_threshold: float = 0.5, forward_period: int = 5):
        self.ic_threshold = ic_threshold
        self.ir_threshold = ir_threshold
        self.forward_period = forward_period

    def compute_forward_returns(
        self,
        prices: pd.DataFrame,
        period: Optional[int] = None,
    ) -> pd.DataFrame:
        period = period or self.forward_period
        return prices.shift(-period) / prices - 1

    def compute_daily_ic(
        self,
        factor_panel: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman",
        period: Optional[int] = None,
    ) -> pd.DataFrame:
        forward_ret = self.compute_forward_returns(prices, period)
        ic_dict = {}
        ir_dict = {}
        dates = factor_panel.index.get_level_values(0).unique()
        for date in dates:
            try:
                if isinstance(factor_panel.index, pd.MultiIndex):
                    fac = factor_panel.xs(date, level=0)
                else:
                    fac = factor_panel.loc[date]
                fwd = forward_ret.loc[date]
                if isinstance(fac, pd.DataFrame):
                    fac = fac.iloc[:, 0]
                if isinstance(fwd, pd.DataFrame):
                    fwd_series = fwd.iloc[0]
                    fwd_series.index = fac.index if isinstance(fac, pd.Series) else fac.iloc[:, 0].index
                    fwd = fwd_series
                fac_s = fac.squeeze() if hasattr(fac, "squeeze") else fac
                fwd_s = fwd.squeeze() if hasattr(fwd, "squeeze") else fwd
                fac_clean = fac_s.dropna()
                fwd_clean = fwd_s.dropna()
                common = fac_clean.index.intersection(fwd_clean.index)
                if len(common) < 10:
                    ic_dict[date] = np.nan
                    ir_dict[date] = np.nan
                    continue
                f_vals = fac_clean.loc[common]
                r_vals = fwd_clean.loc[common]
                if method == "spearman":
                    ic = f_vals.rank().corr(r_vals.rank())
                else:
                    ic = f_vals.corr(r_vals)
                ic_dict[date] = ic
            except Exception:
                ic_dict[date] = np.nan
            ir_dict[date] = np.nan

        ic_series = pd.Series(ic_dict).sort_index()
        ir_series = pd.Series(ir_dict)
        result = pd.DataFrame({"IC": ic_series, "IR": ir_series})
        result["IC_abs"] = result["IC"].abs()
        result["IC_mean_12"] = result["IC"].rolling(12).mean()
        result["IC_std_12"] = result["IC"].rolling(12).std()
        valid_ir = result["IC"].dropna()
        if len(valid_ir) >= 12:
            result.loc[valid_ir.index[11:], "IR"] = (
                result.loc[valid_ir.index[11:], "IC_mean_12"] /
                (result.loc[valid_ir.index[11:], "IC_std_12"] + 1e-10)
            )
        return result

    def compute_monthly_ic(
        self,
        factor_panel: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman",
        period: Optional[int] = None,
    ) -> pd.DataFrame:
        daily_ic = self.compute_daily_ic(factor_panel, prices, method, period)
        ic_ts = daily_ic["IC"].dropna()
        ir_ts = daily_ic["IR"].dropna()
        ic_df = pd.DataFrame({"IC": ic_ts.values}, index=pd.DatetimeIndex(ic_ts.index))
        ir_df = pd.DataFrame({"IR": ir_ts.values}, index=pd.DatetimeIndex(ir_ts.index))
        monthly_ic = ic_df.resample("ME").mean()
        monthly_ir = ir_df.resample("ME").mean()
        result = monthly_ic.join(monthly_ir, how="outer")
        return result

    def validate(
        self,
        factor_panel: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = "spearman",
        period: Optional[int] = None,
    ) -> Dict:
        daily_ic = self.compute_daily_ic(factor_panel, prices, method, period)
        monthly_ic_df = self.compute_monthly_ic(factor_panel, prices, method, period)
        ic_series = daily_ic["IC"].dropna()
        monthly_ic_series = monthly_ic_df["IC"].dropna()
        ir_series = daily_ic["IR"].dropna()

        ic_mean = ic_series.mean()
        ic_valid = ic_series[ic_series.abs() > self.ic_threshold]
        ic_pass_rate = len(ic_valid) / len(ic_series) if len(ic_series) > 0 else 0.0

        ir_mean = ir_series.mean() if len(ir_series) > 0 else np.nan
        ir_pass_rate = (ir_series > self.ir_threshold).sum() / len(ir_series) if len(ir_series) > 0 else 0.0

        monthly_pass_rate = (monthly_ic_series.abs() > self.ic_threshold).sum() / len(monthly_ic_series) if len(monthly_ic_series) > 0 else 0.0

        return {
            "ic_mean": ic_mean,
            "ic_pass_rate": ic_pass_rate,
            "ic_count": len(ic_series),
            "ir_mean": ir_mean,
            "ir_pass_rate": ir_pass_rate,
            "ir_count": len(ir_series),
            "monthly_ic_pass_rate": monthly_pass_rate,
            "monthly_ic_count": len(monthly_ic_series),
            "daily_ic": daily_ic["IC"],
            "monthly_ic": monthly_ic_df["IC"],
            "ir_series": ir_series,
            "daily_ic_df": daily_ic,
        }

    def is_valid(self, validation_result: Dict) -> bool:
        monthly_pass = validation_result["monthly_ic_pass_rate"] >= 0.6
        ir_valid = validation_result["ir_mean"] > self.ir_threshold if not np.isnan(validation_result["ir_mean"]) else False
        return monthly_pass and ir_valid

    def get_report(self, validation_result: Dict) -> str:
        lines = [
            "IC/IR Validation Report",
            "=" * 40,
            f"IC Mean: {validation_result['ic_mean']:.4f}",
            f"IC Pass Rate (>0.02): {validation_result['ic_pass_rate']:.1%}",
            f"Monthly IC Pass Rate: {validation_result['monthly_ic_pass_rate']:.1%}",
            f"IR Mean: {validation_result['ir_mean']:.4f}",
            f"IR Pass Rate (>0.5): {validation_result['ir_pass_rate']:.1%}",
            f"Is Valid: {self.is_valid(validation_result)}",
        ]
        return "\n".join(lines)

