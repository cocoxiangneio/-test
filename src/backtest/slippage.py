# -*- coding: utf-8 -*-
"""Slippage models."""

import numpy as np
import pandas as pd


class FixedSlippage:
    def __init__(self, points: float = 0.001):
        self.points = points

    def apply(self, price: float, side: str) -> float:
        return price * (1 - self.points) if side == "sell" else price * (1 + self.points)


class PercentSlippage:
    def __init__(self, pct: float = 0.001):
        self.pct = pct

    def apply(self, price: float, side: str) -> float:
        return price * (1 - self.pct) if side == "sell" else price * (1 + self.pct)


class VolumeSlippage:
    def __init__(self, base_pct: float = 0.0005, volume_pct: float = 0.00001):
        self.base_pct = base_pct
        self.volume_pct = volume_pct

    def apply(self, price: float, side: str, volume: int = 100) -> float:
        slip = self.base_pct + self.volume_pct * (volume / 100)
        return price * (1 - slip) if side == "sell" else price * (1 + slip)
