# -*- coding: utf-8 -*-
"""Commission models."""

from typing import Union


class PercentCommission:
    def __init__(self, rate: float = 0.0003):
        self.rate = rate

    def calc(self, price: float, volume: int) -> float:
        return price * volume * self.rate


class FixedCommission:
    def __init__(self, per_trade: float = 5.0):
        self.per_trade = per_trade

    def calc(self, price: float, volume: int) -> float:
        return self.per_trade


class TieredCommission:
    def __init__(self, tiers: list[tuple[float, float]] = None):
        self.tiers = tiers or [(0.0, 0.0003), (10000, 0.00025), (50000, 0.0002)]

    def calc(self, turnover: float) -> float:
        for threshold, rate in reversed(self.tiers):
            if turnover >= threshold:
                return turnover * rate
        return turnover * self.tiers[0][1]
