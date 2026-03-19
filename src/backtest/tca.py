# -*- coding: utf-8 -*-
"""Transaction Cost Analysis (TCA) module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class TCAResult:
    commission: float
    slippage: float
    market_impact: float
    liquidity_cost: float
    total_cost: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "commission": self.commission,
            "slippage": self.slippage,
            "market_impact": self.market_impact,
            "liquidity_cost": self.liquidity_cost,
            "total_cost": self.total_cost,
        }


class PartialFillModel:
    def __init__(self, participation_threshold: float = 0.05):
        self.participation_threshold = participation_threshold

    def calculate_filled_volume(
        self,
        requested_volume: int,
        adv: int,
        volatility: float = 0.02,
    ) -> int:
        if adv <= 0:
            return requested_volume
        participation = requested_volume / adv
        if participation <= self.participation_threshold:
            return requested_volume
        fill_rate = self.participation_threshold / participation
        return max(1, int(requested_volume * fill_rate))


class MarketImpactModel:
    def __init__(self, eta: float = 0.1, gamma: float = 0.1):
        self.eta = eta
        self.gamma = gamma

    def calculate(
        self,
        order_volume: int,
        avg_daily_volume: int,
        price: float,
        volatility: float = 0.02,
    ) -> float:
        if avg_daily_volume <= 0 or price <= 0:
            return 0.0
        participation_rate = order_volume / avg_daily_volume
        temporary = self.eta * volatility * participation_rate * price
        permanent = self.gamma * participation_rate * volatility * price
        return (temporary + permanent) * order_volume


class LiquidityCostModel:
    def __init__(self, spread_bps: float = 1.0):
        self.spread_bps = spread_bps

    def calculate(
        self,
        order_volume: int,
        price: float,
        is_buy: bool = True,
        adv: Optional[int] = None,
    ) -> float:
        if price <= 0 or order_volume <= 0:
            return 0.0
        half_spread = price * self.spread_bps / 10000 / 2
        if adv is not None and adv > 0:
            participation = order_volume / adv
            market_impact_cost = participation * price * 0.001
            return (half_spread + market_impact_cost) * order_volume
        return half_spread * 2 * order_volume


class SlippageModel:
    def __init__(self, vol_adjustment: bool = True, base_bps: float = 0.5):
        self.vol_adjustment = vol_adjustment
        self.base_bps = base_bps

    def calculate(
        self,
        price: float,
        order_volume: int,
        volatility: float = 0.02,
        is_buy: bool = True,
    ) -> float:
        if price <= 0:
            return 0.0
        base_slippage = price * self.base_bps / 10000
        if self.vol_adjustment:
            vol_scalar = volatility / 0.01
            base_slippage *= max(0.5, min(3.0, vol_scalar))
        direction = 1 if is_buy else -1
        random_component = 0.0
        return direction * base_slippage * order_volume + random_component


class TransactionCostAnalyzer:
    def __init__(
        self,
        commission_rate: float = 0.0003,
        spread_bps: float = 1.0,
        vol_adjustment: bool = True,
        market_impact_eta: float = 0.1,
    ):
        self.commission_rate = commission_rate
        self.slippage_model = SlippageModel(vol_adjustment=vol_adjustment)
        self.liquidity_model = LiquidityCostModel(spread_bps=spread_bps)
        self.market_impact_model = MarketImpactModel(eta=market_impact_eta)

    def analyze(
        self,
        price: float,
        order_volume: int,
        is_buy: bool = True,
        avg_daily_volume: int = 0,
        volatility: float = 0.02,
        adv: Optional[int] = None,
    ) -> TCAResult:
        if price <= 0 or order_volume <= 0:
            return TCAResult(0, 0, 0, 0, 0)

        commission = price * order_volume * self.commission_rate
        slippage = self.slippage_model.calculate(
            price, order_volume, volatility, is_buy
        )
        market_impact = self.market_impact_model.calculate(
            order_volume, max(avg_daily_volume, order_volume * 10),
            price, volatility
        )
        liquidity_cost = self.liquidity_model.calculate(
            order_volume, price, is_buy, adv
        )
        total_cost = commission + abs(slippage) + market_impact + liquidity_cost

        return TCAResult(
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            liquidity_cost=liquidity_cost,
            total_cost=total_cost,
        )

    def analyze_batch(
        self,
        trades: pd.DataFrame,
        price_col: str = "price",
        volume_col: str = "volume",
        is_buy_col: str = "is_buy",
    ) -> pd.DataFrame:
        results = []
        for _, row in trades.iterrows():
            result = self.analyze(
                price=row.get(price_col, 100.0),
                order_volume=int(row.get(volume_col, 0)),
                is_buy=row.get(is_buy_col, True),
                avg_daily_volume=int(row.get("adv", 0)),
                volatility=float(row.get("volatility", 0.02)),
            )
            d = result.to_dict()
            d["net_proceeds"] = (
                row.get(price_col, 100.0) * row.get(volume_col, 0)
                - result.total_cost * (1 if row.get(is_buy_col, True) else -1)
            )
            results.append(d)
        return pd.DataFrame(results)
