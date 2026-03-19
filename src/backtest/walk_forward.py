# -*- coding: utf-8 -*-
"""Walk-forward analysis for strategy validation."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardAnalyzer:
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 21,
        mode: Literal["rolling", "expanding"] = "rolling",
        min_train_window: int = 126,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.mode = mode
        self.min_train_window = min_train_window
        self.results_: List[Dict] = []
        self.oot_sharpe_list: List[float] = []
        self.oot_return_list: List[float] = []

    def analyze(
        self,
        equity_curve: pd.Series,
        strategy_func: Optional[Callable] = None,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        **strategy_kwargs,
    ) -> Dict:
        if len(equity_curve) < self.train_window + self.test_window:
            logger.warning("Data too short for walk-forward analysis")
            return {}

        self.results_ = []
        self.oot_sharpe_list = []
        self.oot_return_list = []

        dates = equity_curve.index
        n = len(dates)
        start_idx = 0

        if self.mode == "rolling":
            train_start_idx = start_idx
            while train_start_idx + self.train_window + self.test_window <= n:
                train_end_idx = train_start_idx + self.train_window
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + self.test_window, n)

                train_data = equity_curve.iloc[train_start_idx:train_end_idx]
                test_data = equity_curve.iloc[test_start_idx:test_end_idx]

                train_sharpe = self._sharpe(train_data)
                test_sharpe = self._sharpe(test_data)
                test_return = float(test_data.iloc[-1] / test_data.iloc[0] - 1) if len(test_data) > 1 else 0.0

                result = {
                    "train_start": dates[train_start_idx],
                    "train_end": dates[train_end_idx - 1],
                    "test_start": dates[test_start_idx],
                    "test_end": dates[test_end_idx - 1],
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "test_return": test_return,
                    "train_length": len(train_data),
                    "test_length": len(test_data),
                    "is_oos_positive": test_sharpe > 0,
                    "is_robust": test_sharpe > 0 and train_sharpe > 0,
                }
                self.results_.append(result)
                self.oot_sharpe_list.append(test_sharpe)
                self.oot_return_list.append(test_return)

                train_start_idx += self.step

        elif self.mode == "expanding":
            train_end_idx = self.train_window
            while train_end_idx + self.test_window <= n:
                train_start_idx = 0
                train_data = equity_curve.iloc[train_start_idx:train_end_idx]
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + self.test_window, n)
                test_data = equity_curve.iloc[test_start_idx:test_end_idx]

                if len(train_data) < self.min_train_window:
                    train_end_idx += self.step
                    continue

                train_sharpe = self._sharpe(train_data)
                test_sharpe = self._sharpe(test_data)
                test_return = float(test_data.iloc[-1] / test_data.iloc[0] - 1) if len(test_data) > 1 else 0.0

                result = {
                    "train_start": dates[train_start_idx],
                    "train_end": dates[train_end_idx - 1],
                    "test_start": dates[test_start_idx],
                    "test_end": dates[test_end_idx - 1],
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                    "test_return": test_return,
                    "train_length": len(train_data),
                    "test_length": len(test_data),
                    "is_oos_positive": test_sharpe > 0,
                    "is_robust": test_sharpe > 0 and train_sharpe > 0,
                }
                self.results_.append(result)
                self.oot_sharpe_list.append(test_sharpe)
                self.oot_return_list.append(test_return)

                train_end_idx += self.step

        return self._summarize()

    def _sharpe(self, equity: pd.Series) -> float:
        returns = equity.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float(np.sqrt(252) * returns.mean() / returns.std())

    def _max_drawdown(self, equity: pd.Series) -> float:
        if len(equity) == 0:
            return 0.0
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return float(drawdown.min())

    def _summarize(self) -> Dict:
        if not self.results_:
            return {}

        df = pd.DataFrame(self.results_)
        oos_positive_count = sum(1 for r in self.results_ if r["is_oos_positive"])
        oos_robust_count = sum(1 for r in self.results_ if r["is_robust"])
        n_windows = len(self.results_)

        mean_oos_sharpe = float(np.nanmean(self.oot_sharpe_list))
        mean_oos_return = float(np.nanmean(self.oot_return_list))
        mean_train_sharpe = float(np.mean([r["train_sharpe"] for r in self.results_]))

        stability_ratio = mean_oos_sharpe / (abs(mean_oos_sharpe) + np.nanstd(self.oot_sharpe_list) + 1e-10)

        return {
            "n_windows": n_windows,
            "oos_positive_rate": oos_positive_count / n_windows,
            "oos_robust_rate": oos_robust_count / n_windows,
            "mean_oos_sharpe": mean_oos_sharpe,
            "mean_oos_return": mean_oos_return,
            "mean_train_sharpe": mean_train_sharpe,
            "stability_ratio": stability_ratio,
            "std_oos_sharpe": float(np.nanstd(self.oot_sharpe_list)),
            "results_df": df,
            "is_robust": (oos_positive_count / n_windows) >= 0.6,
            "ois_vs_is_ratio": mean_oos_sharpe / (abs(mean_train_sharpe) + 1e-10),
        }

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results_)


class WalkForwardOptimizer:
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        step: int = 21,
        mode: Literal["rolling", "expanding"] = "rolling",
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.mode = mode
        self.analyzer = WalkForwardAnalyzer(
            train_window=train_window,
            test_window=test_window,
            step=step,
            mode=mode,
        )

    def optimize(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        param_grid: Dict,
        metric: str = "sharpe",
    ) -> Tuple[Dict, float, pd.DataFrame]:
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(product(*values))

        best_params = None
        best_score = -np.inf
        all_results = []

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                all_oos_equities = []
                all_oos_sharpes = []

                start_idx = 0
                n_steps = 0
                while True:
                    train_end = start_idx + self.train_window
                    oos_end = train_end + self.test_window

                    equity = self._backtest_window(
                        data_dict, strategy_func, params,
                        start_idx=start_idx,
                        end_idx=oos_end,
                    )
                    if equity is None or len(equity) < 10:
                        start_idx += self.step
                        n_steps += 1
                        if n_steps > 100:
                            break
                        continue

                    equity.index = pd.RangeIndex(len(equity))
                    train_equity = equity.iloc[:self.train_window]
                    oos_equity = equity.iloc[self.train_window:]

                    if len(oos_equity) >= 5:
                        oos_returns = oos_equity.pct_change().dropna()
                        if len(oos_returns) > 0 and oos_returns.std() > 0:
                            oos_sharpe = float(np.sqrt(252) * oos_returns.mean() / oos_returns.std())
                            all_oos_sharpes.append(oos_sharpe)
                            all_oos_equities.append(oos_equity)

                    start_idx += self.step
                    n_steps += 1
                    if n_steps > 100:
                        break
                    if train_end + self.step > len(list(data_dict.values())[0]):
                        break

                if len(all_oos_sharpes) < 1:
                    continue

                mean_oos_sharpe = float(np.mean(all_oos_sharpes))
                n_windows = len(all_oos_sharpes)
                oos_positive_count = sum(1 for s in all_oos_sharpes if s > 0)

                combined_equity = pd.concat(all_oos_equities) if all_oos_equities else pd.Series()
                if len(combined_equity) > 0:
                    combined_returns = combined_equity.pct_change().dropna()
                    combined_sharpe = float(np.sqrt(252) * combined_returns.mean() / combined_returns.std()) if combined_returns.std() > 0 else 0.0
                else:
                    combined_sharpe = 0.0

                result = {
                    "mean_oos_sharpe": mean_oos_sharpe,
                    "n_windows": n_windows,
                    "oos_positive_count": oos_positive_count,
                    "combined_sharpe": combined_sharpe,
                    "oos_sharpes": all_oos_sharpes,
                }

                score = mean_oos_sharpe
                row = {"params": params, "score": score}
                row.update(result)
                all_results.append(row)

                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

        results_df = pd.DataFrame(all_results)
        return best_params, best_score, results_df

    def _backtest_window(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        params: Dict,
        start_idx: int,
        end_idx: int,
    ) -> Optional[pd.Series]:
        try:
            from src.backtest.engine import BacktestEngine

            window_data = {}
            for stock, df in data_dict.items():
                if start_idx < len(df) and end_idx <= len(df):
                    window_data[stock] = df.iloc[start_idx:end_idx].copy()

            if not window_data:
                return None

            def window_strategy(data, date):
                return strategy_func(data, date, **params)

            engine = BacktestEngine(initial_cash=100000, commission_rate=0.0003)
            result = engine.run(window_data, window_strategy)
            return result.equity_curve
        except Exception:
            return None
