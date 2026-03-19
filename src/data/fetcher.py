# -*- coding: utf-8 -*-
"""JoinQuant data fetcher with cache support."""

import os
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_FREQS = {
    "1d": "daily",
    "60m": "60minute",
    "30m": "30minute",
    "15m": "15minute",
}

JQ_VALUATION_FIELDS = [
    "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio",
    "market_cap", "float_market_cap", "turnover_ratio",
]
JQ_INCOME_FIELDS = [
    "operating_revenue", "operating_cost", "gross_profit_margin",
    "net_profit", "net_profit_growth", "revenue_growth",
]
JQ_BALANCE_FIELDS = [
    "total_assets", "total_liabilities", "equity",
    "debt_ratio", "current_ratio",
]
JQ_INDICATOR_FIELDS = [
    "roe", "roa", "net_profit_margin", "gross_profit_margin",
    "operating_margin", "eps", "bvps",
]


class JQFetcher:
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: str = "cache",
        cache_ttl_days: int = 7,
    ):
        self.username = username or os.environ.get("JQ_USER")
        self.password = password or os.environ.get("JQ_PASSWORD")
        self.cache_dir = cache_dir
        self.cache_ttl_days = cache_ttl_days
        self._jq = None
        self._authed = False
        os.makedirs(self.cache_dir, exist_ok=True)

    def auth(self) -> bool:
        if self._authed:
            return True
        try:
            import jqdatasdk as jq
            if not self.username or not self.password:
                raise ValueError("JQ_USER and JQ_PASSWORD environment variables are required")
            jq.auth(self.username, self.password)
            self._jq = jq
            self._authed = True
            logger.info("JoinQuant auth success")
            return True
        except ImportError:
            logger.error("jqdatasdk not installed: pip install jqdatasdk")
            raise
        except Exception as e:
            logger.error(f"JoinQuant auth failed: {e}")
            raise

    def _ensure_auth(self):
        if not self._authed:
            self.auth()

    def _cache_key(self, stock: str, data_type: str, freq: str = "") -> str:
        safe = stock.replace(".", "_").replace("-", "_")
        key = f"{safe}_{data_type}_{freq}" if freq else f"{safe}_{data_type}"
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _is_cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
        return age_days < self.cache_ttl_days

    def get_kline(
        self,
        stock: str,
        start_date: str,
        end_date: str,
        freq: str = "1d",
        fields: Optional[List[str]] = None,
        fq: str = "pre",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = self._cache_key(stock, "kline", freq)
        if use_cache and self._is_cache_valid(cache_path):
            df = pd.read_pickle(cache_path)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if len(df) > 0:
                logger.info(f"[CACHE HIT] {stock} {freq}: {len(df)} bars")
                return df

        self._ensure_auth()
        jq_freq = SUPPORTED_FREQS.get(freq, "daily")
        default_fields = ["open", "high", "low", "close", "volume", "money"]
        use_fields = fields or default_fields

        try:
            df = self._jq.get_price(
                security=stock,
                start_date=start_date,
                end_date=end_date,
                frequency=jq_freq,
                fields=use_fields,
                skip_paused=True,
                fq=fq,
            )
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df.columns = [c.lower() for c in df.columns]
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.to_pickle(cache_path)
            logger.info(f"[FETCH] {stock} {freq}: {len(df)} bars saved to cache")
            return df
        except Exception as e:
            logger.error(f"get_kline failed for {stock}: {e}")
            return pd.DataFrame()

    def get_fundamentals(
        self,
        stock: str,
        date: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cache_path = self._cache_key(stock, "fundamentals", date or "latest")
        if use_cache and self._is_cache_valid(cache_path):
            return pd.read_pickle(cache_path)

        self._ensure_auth()
        result: Dict[str, Any] = {}

        try:
            q = self._jq.query(
                self._jq.valuation,
                self._jq.income,
                self._jq.balance,
                self._jq.indicator,
            ).filter(self._jq.valuation.code == stock)

            df = self._jq.get_fundamentals(q, date=date)
            if df is not None and len(df) > 0:
                row = df.iloc[0]
                field_map = {
                    "pe_ratio": row.get("pe_ratio"),
                    "pb_ratio": row.get("pb_ratio"),
                    "ps_ratio": row.get("ps_ratio"),
                    "pcf_ratio": row.get("pcf_ratio"),
                    "roe": row.get("roe"),
                    "roa": row.get("roa"),
                    "net_profit_growth": row.get("net_profit_growth"),
                    "revenue_growth": row.get("revenue_growth"),
                    "gross_margin": row.get("gross_profit_margin"),
                    "debt_ratio": row.get("debt_to_assets"),
                    "net_profit_margin": row.get("net_profit_margin"),
                    "operating_margin": row.get("operating_margin"),
                    "eps": row.get("eps"),
                    "bvps": row.get("bvps"),
                }
                for k, v in field_map.items():
                    result[k] = float(v) if pd.notna(v) else None
            else:
                result = {k: None for k in [
                    "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio",
                    "roe", "roa", "net_profit_growth", "revenue_growth",
                    "gross_margin", "debt_ratio", "net_profit_margin",
                    "operating_margin", "eps", "bvps",
                ]}

            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            logger.info(f"[FETCH] fundamentals for {stock}: {result}")
        except Exception as e:
            logger.warning(f"get_fundamentals failed for {stock}: {e}")
            result = {k: None for k in [
                "pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio",
                "roe", "roa", "net_profit_growth", "revenue_growth",
                "gross_margin", "debt_ratio", "net_profit_margin",
                "operating_margin", "eps", "bvps",
            ]}

        return result

    def batch_get_kline(
        self,
        stocks: List[str],
        start_date: str,
        end_date: str,
        freq: str = "1d",
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for i, stock in enumerate(stocks):
            if show_progress:
                print(f"\r  [{i+1}/{len(stocks)}] {stock}", end="", flush=True)
            results[stock] = self.get_kline(stock, start_date, end_date, freq)
        if show_progress:
            print()
        return results

    def batch_get_fundamentals(
        self,
        stocks: List[str],
        date: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for i, stock in enumerate(stocks):
            if show_progress:
                print(f"\r  [{i+1}/{len(stocks)}] {stock}", end="", flush=True)
            results[stock] = self.get_fundamentals(stock, date)
        if show_progress:
            print()
        return results

    def clear_cache(self, stock: Optional[str] = None):
        if stock:
            for fname in os.listdir(self.cache_dir):
                if fname.startswith(stock.replace(".", "_").replace("-", "_")):
                    os.remove(os.path.join(self.cache_dir, fname))
                    logger.info(f"Cleared: {fname}")
        else:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, fname))
            logger.info("All cache cleared")
