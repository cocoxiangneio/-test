# -*- coding: utf-8 -*-
"""Event-driven backtest engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from .commission import PercentCommission
from .slippage import PercentSlippage

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    date: Any
    stock: str
    side: str
    price: float
    volume: int
    commission: float
    slippage: float
    signal: float
    reason: str = "signal"
    entry_price: float = 0.0
    pnl: float = 0.0


@dataclass
class Position:
    stock: str
    volume: int
    avg_price: float
    unrealized_pnl: float = 0.0
    side: str = "long"


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade]
    positions: Dict[str, Position]
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    total_trades: int
    metrics: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.0003,
        slippage_pct: float = 0.001,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        position_size: float = 1.0,
        allow_short: bool = False,
        short_margin: float = 0.5,
        allow_partial_fill: bool = False,
        participation_threshold: float = 0.05,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = PercentCommission(commission_rate)
        self.slippage = PercentSlippage(slippage_pct)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.allow_short = allow_short
        self.short_margin = short_margin
        self.allow_partial_fill = allow_partial_fill
        self.participation_threshold = participation_threshold
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[float] = []
        self.dates: List[Any] = []
        self._daily_volumes: Dict[str, int] = {}

    def reset(self):
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.dates = []

    def _get_equity(self) -> float:
        pos_value = sum(
            p.unrealized_pnl if p.side == "short" else p.volume * p.avg_price
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def _update_unrealized_pnl(self, prices: Dict[str, float]):
        for pos in self.positions.values():
            if pos.stock in prices:
                if pos.side == "long":
                    pos.unrealized_pnl = (prices[pos.stock] - pos.avg_price) * pos.volume
                else:
                    pos.unrealized_pnl = (pos.avg_price - prices[pos.stock]) * pos.volume

    def _apply_commission(self, price: float, volume: int) -> float:
        return self.commission.calc(price, volume)

    def _apply_slippage(self, price: float, side: str) -> float:
        return self.slippage.apply(price, side)

    def run(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy_func,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> BacktestResult:
        self.reset()
        sl = stop_loss if stop_loss is not None else self.stop_loss
        tp = take_profit if take_profit is not None else self.take_profit

        if not data_dict:
            return self._empty_result()

        common_dates = self._get_common_dates(data_dict)
        if len(common_dates) == 0:
            return self._empty_result()

        entry_prices: Dict[str, float] = {}
        stop_prices: Dict[str, float] = {}
        profit_prices: Dict[str, float] = {}

        prices: Dict[str, float] = {}
        for date in common_dates:
            prices = {s: float(df.loc[date, "close"]) for s, df in data_dict.items() if date in df.index}
            if not prices:
                continue
            if self.allow_partial_fill:
                for stock in prices.keys():
                    self._update_daily_volume(stock, date, data_dict)

            self._update_unrealized_pnl(prices)

            signals = strategy_func(data_dict, date)
            if signals is None:
                signals = {}

            for stock, signal in signals.items():
                if stock not in prices:
                    continue

                price = prices[stock]

                if stock in self.positions:
                    pos = self.positions[stock]

                    if sl > 0 and pos.avg_price > 0:
                        loss_pct = (pos.avg_price - price) / pos.avg_price
                        if loss_pct >= sl:
                            self._close_position(stock, date, price, signal, "stop_loss")
                            entry_prices.pop(stock, None)
                            stop_prices.pop(stock, None)
                            profit_prices.pop(stock, None)
                            continue

                    if tp > 0 and pos.avg_price > 0:
                        if pos.side == "long":
                            gain_pct = (price - pos.avg_price) / pos.avg_price
                        else:
                            gain_pct = (pos.avg_price - price) / pos.avg_price
                        if gain_pct >= tp:
                            self._close_position(stock, date, price, signal, "take_profit")
                            entry_prices.pop(stock, None)
                            stop_prices.pop(stock, None)
                            profit_prices.pop(stock, None)
                            continue

                    if sl > 0 and pos.avg_price > 0 and pos.side == "short":
                        loss_pct = (price - pos.avg_price) / pos.avg_price
                        if loss_pct >= sl:
                            self._close_position(stock, date, price, signal, "stop_loss")
                            entry_prices.pop(stock, None)
                            stop_prices.pop(stock, None)
                            profit_prices.pop(stock, None)
                            continue

                    if (signal == -1 and pos.side == "long") or (signal == 1 and pos.side == "short"):
                        self._close_position(stock, date, price, signal, "signal")
                        entry_prices.pop(stock, None)
                        stop_prices.pop(stock, None)
                        profit_prices.pop(stock, None)
                else:
                    if signal == 1:
                        vol = self._calculate_position_size(price)
                        if vol > 0:
                            self._open_position(stock, date, price, volume=vol)
                            entry_prices[stock] = price
                            stop_prices[stock] = price * (1 - sl)
                            profit_prices[stock] = price * (1 + tp)
                    elif signal == -1 and self.allow_short:
                        vol = self._calculate_position_size(price)
                        if vol > 0:
                            self._open_short_position(stock, date, price, volume=vol)
                            entry_prices[stock] = price
                            stop_prices[stock] = price * (1 + sl)
                            profit_prices[stock] = price * (1 - tp)

            self.dates.append(date)
            self.equity_history.append(self._get_equity())

        for stock in list(self.positions.keys()):
            if stock in prices:
                self._close_position(stock, common_dates[-1], prices[stock], 0, "eod_close")

        return self._build_result()

    def _calculate_position_size(self, price: float) -> int:
        if self.position_size <= 0:
            return 0
        max_cost = self.cash * self.position_size
        if max_cost <= 0:
            return 0
        volume = int(max_cost / price)
        return max(volume, 1)

    def _get_adv(self, stock: str) -> int:
        return self._daily_volumes.get(stock, 0)

    def _update_daily_volume(self, stock: str, date: Any, data_dict: Dict):
        if stock in data_dict and date in data_dict[stock].index:
            vol = data_dict[stock].loc[date, "volume"]
            self._daily_volumes[stock] = int(vol)

    def _open_position(self, stock: str, date: Any, price: float, volume: Optional[int] = None):
        exec_price = self._apply_slippage(price, "buy")
        if volume is None:
            volume = self._calculate_position_size(exec_price)
        if self.allow_partial_fill:
            adv = self._get_adv(stock)
            from .tca import PartialFillModel
            pf = PartialFillModel(participation_threshold=self.participation_threshold)
            volume = pf.calculate_filled_volume(volume, adv)
        total_cost = exec_price * volume
        if total_cost > self.cash:
            volume = int(self.cash / exec_price)
            if volume <= 0:
                return
        commission = self._apply_commission(exec_price, volume)

        self.positions[stock] = Position(
            stock=stock,
            volume=volume,
            avg_price=exec_price,
            side="long",
        )
        self.cash -= exec_price * volume + commission
        self.trades.append(
            Trade(date=date, stock=stock, side="buy", price=exec_price,
                  volume=volume, commission=commission, slippage=0.0, signal=1.0,
                  reason="open", entry_price=exec_price)
        )

    def _open_short_position(self, stock: str, date: Any, price: float, volume: Optional[int] = None):
        exec_price = self._apply_slippage(price, "sell")
        if volume is None:
            volume = self._calculate_position_size(exec_price)
        if self.allow_partial_fill:
            adv = self._get_adv(stock)
            from .tca import PartialFillModel
            pf = PartialFillModel(participation_threshold=self.participation_threshold)
            volume = pf.calculate_filled_volume(volume, adv)
        margin_required = exec_price * volume * self.short_margin
        if margin_required > self.cash:
            volume = int(self.cash / (exec_price * self.short_margin))
            if volume <= 0:
                return
            margin_required = exec_price * volume * self.short_margin
        commission = self._apply_commission(exec_price, volume)

        self.positions[stock] = Position(
            stock=stock,
            volume=volume,
            avg_price=exec_price,
            side="short",
        )
        proceeds = exec_price * volume - commission
        self.cash += proceeds
        self.trades.append(
            Trade(date=date, stock=stock, side="sell_short", price=exec_price,
                  volume=volume, commission=commission, slippage=0.0, signal=-1.0,
                  reason="open_short", entry_price=exec_price)
        )

    def _close_position(self, stock: str, date: Any, price: float, signal: float, reason: str = "signal"):
        pos = self.positions[stock]
        exec_price = self._apply_slippage(price, "sell" if pos.side == "long" else "buy")
        commission = self._apply_commission(exec_price, pos.volume)

        if pos.side == "long":
            pnl = (exec_price - pos.avg_price) * pos.volume - commission
            proceeds = exec_price * pos.volume - commission
        else:
            pnl = (pos.avg_price - exec_price) * pos.volume - commission
            proceeds = pos.avg_price * pos.volume - exec_price * pos.volume - commission

        self.trades.append(
            Trade(date=date, stock=stock, side="sell", price=exec_price,
                  volume=pos.volume, commission=commission, slippage=0.0, signal=signal,
                  reason=reason, entry_price=pos.avg_price, pnl=pnl)
        )
        self.cash += proceeds
        del self.positions[stock]

    def _get_common_dates(self, data_dict: Dict[str, pd.DataFrame]) -> List[Any]:
        date_sets = [set(df.index) for df in data_dict.values() if len(df) > 0]
        if not date_sets:
            return []
        common = date_sets[0]
        for s in date_sets[1:]:
            common = common.intersection(s)
        return sorted(common)

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            equity_curve=pd.Series(),
            trades=[],
            positions={},
            final_equity=self.initial_cash,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            total_trades=0,
        )

    def _build_result(self) -> BacktestResult:
        equity_curve = pd.Series(self.equity_history, index=self.dates)
        returns = equity_curve.pct_change().dropna()
        sharpe = self._sharpe_ratio(returns)
        max_dd = self._max_drawdown(equity_curve)
        total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0
        calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0
        closed_trades = [t for t in self.trades if t.side in ("sell", "sell_short") and t.pnl != 0]
        wins = sum(1 for t in closed_trades if t.pnl > 0)
        win_rate = wins / len(closed_trades) if closed_trades else 0.0
        self.win_rate = win_rate

        return BacktestResult(
            equity_curve=equity_curve,
            trades=self.trades,
            positions=self.positions,
            final_equity=self._get_equity(),
            total_return=total_ret,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            win_rate=win_rate,
            total_trades=len(self.trades),
            metrics={
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "sharpe_ratio": sharpe,
                "calmar_ratio": calmar,
                "win_rate": win_rate,
                "total_trades": len(self.trades),
                "final_equity": self._get_equity(),
            },
        )

    @staticmethod
    def _sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        if len(equity) == 0:
            return 0.0
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return float(drawdown.min())
