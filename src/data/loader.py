# -*- coding: utf-8 -*-
"""Unified data loader - provides consistent interface for all data sources."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .fetcher import JQFetcher

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        fetcher: Optional[JQFetcher] = None,
        default_freq: str = "1d",
    ):
        self.fetcher = fetcher or JQFetcher()
        self.default_freq = default_freq
        self._kline_cache: Dict[str, pd.DataFrame] = {}
        self._fundamental_cache: Dict[str, Dict] = {}

    def load_kline(
        self,
        stock: str,
        start_date: str,
        end_date: str,
        freq: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        freq = freq or self.default_freq
        cache_key = f"{stock}_{freq}_{start_date}_{end_date}"

        if use_cache and cache_key in self._kline_cache:
            return self._kline_cache[cache_key]

        df = self.fetcher.get_kline(stock, start_date, end_date, freq, use_cache=use_cache)
        if len(df) > 0:
            self._kline_cache[cache_key] = df

        return df

    def load_multiple_kline(
        self,
        stocks: List[str],
        start_date: str,
        end_date: str,
        freq: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        freq = freq or self.default_freq
        results = self.fetcher.batch_get_kline(stocks, start_date, end_date, freq, use_cache=use_cache)
        return {s: df for s, df in results.items() if len(df) > 0}

    def load_fundamentals(
        self,
        stock: str,
        date: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cache_key = f"{stock}_{date or 'latest'}"
        if use_cache and cache_key in self._fundamental_cache:
            return self._fundamental_cache[cache_key]

        data = self.fetcher.get_fundamentals(stock, date, use_cache=use_cache)
        self._fundamental_cache[cache_key] = data
        return data

    def load_multiple_fundamentals(
        self,
        stocks: List[str],
        date: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        return self.fetcher.batch_get_fundamentals(stocks, date, use_cache=use_cache)

    def load_panel(
        self,
        stocks: List[str],
        start_date: str,
        end_date: str,
        freq: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        freq = freq or self.default_freq
        data_dict = self.load_multiple_kline(stocks, start_date, end_date, freq)

        if not data_dict:
            return pd.DataFrame()

        panel_rows = []
        for stock, df in data_dict.items():
            if len(df) == 0:
                continue
            temp = df.copy()
            temp["stock"] = stock
            panel_rows.append(temp)

        if not panel_rows:
            return pd.DataFrame()

        panel = pd.concat(panel_rows, ignore_index=False)
        panel = panel.sort_index()
        return panel

    def get_returns(
        self,
        df: pd.DataFrame,
        period: int = 1,
    ) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        result = df["close"].pct_change(period)
        result.name = None
        return result

    def get_forward_returns(
        self,
        df: pd.DataFrame,
        period: int = 1,
    ) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        return df["close"].shift(-period) / df["close"] - 1

    def clear_all_cache(self):
        self._kline_cache.clear()
        self._fundamental_cache.clear()
        self.fetcher.clear_cache()
        logger.info("All data loader caches cleared")

    def load_csv(
        self,
        file_path: str,
        date_column: str = "date",
        parse_dates: bool = True,
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")
        if parse_dates and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df = df.set_index(date_column)
            df = df.sort_index()
        column_mapping = {
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume",
            "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        df = pd.read_parquet(file_path)
        column_mapping = {
            "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume",
            "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def data_summary(self, df: pd.DataFrame) -> Dict:
        if len(df) == 0:
            return {}
        close = df["close"]
        returns = self.get_returns(df).dropna()
        return {
            "total_bars": len(df),
            "date_range": f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
            "trading_days": len(df),
            "total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            "mean_return_pct": float(returns.mean() * 100),
            "std_return_pct": float(returns.std() * 100),
            "max_return_pct": float(returns.max() * 100),
            "min_return_pct": float(returns.min() * 100),
        }
