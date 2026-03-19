# -*- coding: utf-8 -*-
"""Quantitative System v2 - Configuration"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    jq_username: Optional[str] = None
    jq_password: Optional[str] = None
    cache_dir: str = "cache"
    cache_ttl_days: int = 7
    default_freq: str = "1d"
    price_fields: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume", "money"]
    )


@dataclass
class StockConfig:
    stocks: List[str] = field(
        default_factory=lambda: [
            "000001.XSHE",
            "600036.XSHG",
            "601398.XSHG",
            "600519.XSHG",
            "000858.XSHE",
            "300750.XSHE",
            "601318.XSHG",
            "600900.XSHG",
            "601166.XSHG",
            "002594.XSHE",
        ]
    )
    stock_names: dict = field(
        default_factory=lambda: {
            "000001.XSHE": "平安银行",
            "600036.XSHG": "招商银行",
            "601398.XSHG": "工商银行",
            "600519.XSHG": "贵州茅台",
            "000858.XSHE": "五粮液",
            "300750.XSHE": "宁德时代",
            "601318.XSHG": "中国平安",
            "600900.XSHG": "长江电力",
            "601166.XSHG": "兴业银行",
            "002594.XSHE": "比亚迪",
        }
    )
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"


@dataclass
class BacktestConfig:
    initial_cash: float = 100000.0
    commission: float = 0.0003
    slippage: float = 0.001
    stop_loss: float = 0.05
    take_profit: float = 0.10
    position_size: float = 1.0


@dataclass
class OptimizeConfig:
    algorithm: str = "ga"
    n_generations: int = 50
    pop_size: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.1
    n_trials_bayesian: int = 100
    objective_weights: dict = field(
        default_factory=lambda: {
            "sharpe": 0.5,
            "calmar": 0.3,
            "return_drawdown": 0.2,
        }
    )


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    stocks: StockConfig = field(default_factory=StockConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    results_dir: str = "results"


_default_config = None


def get_config() -> SystemConfig:
    global _default_config
    if _default_config is None:
        _default_config = SystemConfig()
    return _default_config


def set_config(config: SystemConfig) -> None:
    global _default_config
    _default_config = config
