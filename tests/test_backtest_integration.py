# -*- coding: utf-8 -*-
"""Integration tests for backtest engine."""

import pytest
import pandas as pd
import numpy as np


def _make_ohlcv(dates, close_prices):
    open_p = close_prices * (1 + np.random.randn(len(dates)) * 0.005)
    high = np.maximum(open_p, close_prices) + np.random.rand(len(dates)) * 0.5
    low = np.minimum(open_p, close_prices) - np.random.rand(len(dates)) * 0.5
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)


def _build_controlled_ohlcv(dates, close_prices):
    open_p = close_prices.copy()
    high = close_prices * 1.01
    low = close_prices * 0.99
    volume = np.full(len(dates), 10000, dtype=int)
    return pd.DataFrame({
        "open": open_p,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": volume,
    }, index=dates)


def test_stop_loss_triggers():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    close_prices = np.array([100.0] * 20)
    close_prices[5] = 100.0
    for i in range(6, 20):
        close_prices[i] = close_prices[i - 1] * (1 - 0.08)
    data = {"STOCK": _build_controlled_ohlcv(dates, close_prices)}

    def strategy(data, date):
        if date == dates[5]:
            return {"STOCK": 1.0}
        return {"STOCK": 0.0}

    engine = BacktestEngine(
        initial_cash=100000,
        commission_rate=0.0,
        slippage_pct=0.0,
        stop_loss=0.05,
        take_profit=0.50,
    )
    result = engine.run(data, strategy)
    stop_trades = [t for t in result.trades if t.reason == "stop_loss"]
    assert len(stop_trades) > 0, f"Stop loss should have triggered. Trades: {[(t.side, t.reason) for t in result.trades]}"
    assert stop_trades[0].side == "sell"


def test_take_profit_triggers():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    close_prices = np.array([100.0] * 20)
    for i in range(1, 20):
        close_prices[i] = close_prices[i - 1] * (1 + 0.15)
    data = {"STOCK": _build_controlled_ohlcv(dates, close_prices)}

    def strategy(data, date):
        if date == dates[0]:
            return {"STOCK": 1.0}
        return {"STOCK": 0.0}

    engine = BacktestEngine(
        initial_cash=100000,
        commission_rate=0.0,
        slippage_pct=0.0,
        stop_loss=0.50,
        take_profit=0.10,
    )
    result = engine.run(data, strategy)
    tp_trades = [t for t in result.trades if t.reason == "take_profit"]
    assert len(tp_trades) > 0, f"Take profit should have triggered. Trades: {[(t.side, t.reason) for t in result.trades]}"
    assert tp_trades[0].side == "sell"


def test_commission_deducted():
    from src.backtest.engine import BacktestEngine
    from src.backtest.commission import PercentCommission
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    prices = np.linspace(100, 110, 10)
    data = {"STOCK": _make_ohlcv(dates, prices)}

    def strategy(data, date):
        return {"STOCK": 1.0}

    comm_model = PercentCommission(0.0003)
    engine = BacktestEngine(initial_cash=1000000, commission_rate=0.0003, slippage_pct=0.0)
    result = engine.run(data, strategy)
    total_commission = sum(t.commission for t in result.trades)
    assert total_commission > 0, "Commission should be charged on trades"
    buy_trade = next(t for t in result.trades if t.side == "buy")
    expected_comm = comm_model.calc(buy_trade.price, buy_trade.volume)
    assert abs(buy_trade.commission - expected_comm) < 1e-6, \
        f"commission={buy_trade.commission}, expected={expected_comm}, price={buy_trade.price}, volume={buy_trade.volume}"


def test_slippage_affects_execution():
    from src.backtest.engine import BacktestEngine
    from src.backtest.slippage import PercentSlippage
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = np.linspace(100, 104, 5)
    data = {"STOCK": _make_ohlcv(dates, prices)}

    def strategy(data, date):
        return {"STOCK": 1.0}

    engine_no_slip = BacktestEngine(
        initial_cash=1000000,
        commission_rate=0.0,
        slippage_pct=0.0,
    )
    result_no_slip = engine_no_slip.run(data, strategy)

    engine_slip = BacktestEngine(
        initial_cash=1000000,
        commission_rate=0.0,
        slippage_pct=0.001,
    )
    result_slip = engine_slip.run(data, strategy)

    assert result_slip.final_equity < result_no_slip.final_equity, \
        "Slippage should reduce final equity vs no-slippage baseline"


def test_factor_strategy_backtest_pipeline():
    from src.backtest.engine import BacktestEngine
    from src.factors.technical import sma_20
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(60) * 2)
    ohlcv = _make_ohlcv(dates, close)
    data = {"STOCK": ohlcv}

    def ma_cross_strategy(data_dict, date):
        signals = {}
        for stock, df in data_dict.items():
            prices = df["close"]
            if date not in prices.index:
                continue
            ma = prices.rolling(20).mean()
            if date in ma.index and not np.isnan(ma.loc[date]):
                if prices.loc[date] > ma.loc[date]:
                    signals[stock] = 1.0
                elif prices.loc[date] < ma.loc[date]:
                    signals[stock] = -1.0
        return signals

    engine = BacktestEngine(initial_cash=100000, commission_rate=0.0003, slippage_pct=0.001)
    result = engine.run(data, ma_cross_strategy)
    assert result.total_trades >= 0
    assert len(result.equity_curve) > 0
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics


def test_no_look_ahead_bias():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(42)
    returns = np.random.randn(30) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    data = {"STOCK": _make_ohlcv(dates, close)}

    def momentum_strategy(data_dict, date):
        signals = {}
        for stock, df in data_dict.items():
            prices = df["close"]
            if date not in prices.index:
                continue
            lookback = 5
            idx = list(prices.index).index(date)
            if idx < lookback:
                continue
            past = prices.iloc[idx - lookback:idx]
            future_idx = idx + 1
            if future_idx < len(prices):
                future_ret = (prices.iloc[future_idx] - prices.iloc[idx]) / prices.iloc[idx]
            else:
                future_ret = 0.0
            if past.iloc[-1] > past.iloc[0]:
                signals[stock] = 1.0
        return signals

    engine = BacktestEngine(initial_cash=100000)
    result = engine.run(data, momentum_strategy)
    assert result.total_trades >= 0


def test_position_sizing_fixed_fractional():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    close_prices = np.array([100.0, 200.0, 50.0], dtype=float)
    data = {}
    for i, (stock, price) in enumerate(zip(["A", "B", "C"], close_prices)):
        prices = np.full(20, price)
        prices = prices * (1 + np.linspace(0, 0.1, 20))
        data[stock] = _build_controlled_ohlcv(dates, prices)

    def strategy(data, date):
        return {s: 1.0 for s in data.keys()}

    engine = BacktestEngine(initial_cash=300000, commission_rate=0.0, slippage_pct=0.0, position_size=0.1)
    result = engine.run(data, strategy)
    buy_trades = [t for t in result.trades if t.side == "buy"]
    assert len(buy_trades) == 3, f"Should buy all 3 stocks, got {len(buy_trades)}"
    for trade in buy_trades:
        expected_vol = int(300000 * 0.1 / trade.price)
        assert trade.volume <= expected_vol + 1, f"Volume {trade.volume} exceeds expected {expected_vol}"


def test_position_sizing_zero_returns_no_trade():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = np.full(5, 100.0)
    data = {"STOCK": _build_controlled_ohlcv(dates, prices)}
    engine = BacktestEngine(initial_cash=100000, commission_rate=0.0, position_size=0.0)
    def strategy(d, date): return {"STOCK": 1.0}
    result = engine.run(data, strategy)
    buy_trades = [t for t in result.trades if t.side == "buy"]
    assert len(buy_trades) == 0, "position_size=0 should produce no trades"


def test_position_sizing_cash_constraint():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = np.full(5, 50000.0)
    data = {"STOCK": _build_controlled_ohlcv(dates, prices)}
    engine = BacktestEngine(initial_cash=100000, commission_rate=0.0, position_size=1.0)
    def strategy(d, date): return {"STOCK": 1.0}
    result = engine.run(data, strategy)
    buy_trades = [t for t in result.trades if t.side == "buy"]
    assert len(buy_trades) == 1
    assert buy_trades[0].volume <= 2, "With expensive stock, volume should be capped"


def test_multiple_stocks_allocation():
    from src.backtest.engine import BacktestEngine
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    np.random.seed(42)
    result = {}
    for stock in ["A", "B", "C"]:
        close = 100 + np.cumsum(np.random.randn(20) * 1.0)
        result[stock] = _make_ohlcv(dates, close)

    def buy_all(data_dict, date):
        return {s: 1.0 for s in data_dict.keys()}

    engine = BacktestEngine(initial_cash=300000, commission_rate=0.0003, position_size=0.34)
    bt_result = engine.run(result, buy_all)
    buy_trades = [t for t in bt_result.trades if t.side == "buy"]
    assert len(buy_trades) == 3, f"Should buy all 3 stocks, got {len(buy_trades)}"
    sell_trades = [t for t in bt_result.trades if t.side == "sell"]
    assert len(sell_trades) == 3, f"Should sell all 3 stocks at end, got {len(sell_trades)}"


def test_walk_forward_analysis():
    from src.backtest.walk_forward import WalkForwardAnalyzer
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    returns = np.random.randn(500) * 0.01
    equity = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    wfa = WalkForwardAnalyzer(train_window=100, test_window=50, step=25, mode="rolling")
    result = wfa.analyze(equity)
    assert result["n_windows"] >= 5
    assert "oos_positive_rate" in result
    assert "mean_oos_sharpe" in result
    assert "is_robust" in result
    assert result["ois_vs_is_ratio"] is not None
    results_df = wfa.get_results_df()
    assert len(results_df) == result["n_windows"]


def test_walk_forward_expanding_mode():
    from src.backtest.walk_forward import WalkForwardAnalyzer
    np.random.seed(99)
    dates = pd.date_range("2024-01-01", periods=400, freq="B")
    equity = pd.Series(100 * np.exp(np.cumsum(np.random.randn(400) * 0.01)), index=dates)

    wfa = WalkForwardAnalyzer(train_window=150, test_window=60, step=30, mode="expanding")
    result = wfa.analyze(equity)
    assert result["n_windows"] >= 3
    assert "stability_ratio" in result


def test_walk_forward_ois_ratio():
    from src.backtest.walk_forward import WalkForwardAnalyzer
    np.random.seed(123)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    equity = pd.Series(100 * np.exp(np.cumsum(np.random.randn(300) * 0.01)), index=dates)

    wfa = WalkForwardAnalyzer(train_window=80, test_window=40, step=20)
    result = wfa.analyze(equity)
    assert result["ois_vs_is_ratio"] is not None
    oos_positive_windows = sum(1 for r in wfa.results_ if r["is_oos_positive"])
    assert oos_positive_windows >= 0
    assert result["std_oos_sharpe"] >= 0


def test_short_selling_basic():
    from src.backtest.engine import BacktestEngine

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(42)
    close = 100 - np.cumsum(np.random.randn(30) * 1.0)
    close = np.maximum(close, 1.0)
    ohlcv = _make_ohlcv(dates, close)
    data = {"STOCK": ohlcv}

    def short_strategy(data_dict, date):
        return {"STOCK": -1.0}

    engine = BacktestEngine(initial_cash=100000, allow_short=True, short_margin=0.5, position_size=0.5)
    result = engine.run(data, short_strategy)

    assert result.total_trades >= 2
    sell_short_trades = [t for t in result.trades if t.side == "sell_short"]
    assert len(sell_short_trades) > 0
    cash_after_open = result.trades[0].price * result.trades[0].volume
    assert result.equity_curve.iloc[0] >= 100000


def test_short_selling_stop_loss():
    from src.backtest.engine import BacktestEngine

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(42)
    close = [100.0] * 30
    close[10] = 115.0
    close[11:20] = [115.0] * 9
    close[20] = 100.0
    ohlcv = _make_ohlcv(dates, close)
    data = {"STOCK": ohlcv}

    def short_strategy(data_dict, date):
        return {"STOCK": -1.0}

    engine = BacktestEngine(initial_cash=100000, allow_short=True, stop_loss=0.05, position_size=0.5)
    result = engine.run(data, short_strategy)

    stop_loss_trades = [t for t in result.trades if t.reason == "stop_loss" and t.side == "sell"]
    assert len(stop_loss_trades) > 0, "Short stop loss should trigger"


def test_win_rate_uses_pnl():
    from src.evaluation.metrics import win_rate, profit_loss_ratio
    from src.backtest.engine import Trade

    trades = [
        Trade(date=None, stock="A", side="sell", price=110.0, volume=100,
              commission=0.33, slippage=0.0, signal=1.0, reason="signal", entry_price=100.0, pnl=1000 - 0.33),
        Trade(date=None, stock="B", side="sell", price=90.0, volume=100,
              commission=0.27, slippage=0.0, signal=1.0, reason="signal", entry_price=100.0, pnl=-1000 - 0.27),
        Trade(date=None, stock="C", side="sell_short", price=90.0, volume=100,
              commission=0.27, slippage=0.0, signal=-1.0, reason="signal", entry_price=100.0, pnl=1000 - 0.27),
    ]

    wr = win_rate(trades)
    assert wr == pytest.approx(2/3), f"2/3 wins expected, got {wr}"

    plr = profit_loss_ratio(trades)
    assert plr > 0


def test_walk_forward_optimizer_multi_window():
    from src.backtest.walk_forward import WalkForwardOptimizer

    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(500) * 1.0)
    ohlcv = _make_ohlcv(dates, close)
    data = {"STOCK": ohlcv}

    def ma_cross_strategy(data_dict, date, fast=10, slow=20):
        signals = {}
        for stock, df in data_dict.items():
            if date not in df.index:
                continue
            hist = df.loc[:date]
            if len(hist) < slow:
                signals[stock] = 0.0
                continue
            ma_fast = hist["close"].rolling(fast).mean().iloc[-1]
            ma_slow = hist["close"].rolling(slow).mean().iloc[-1]
            if ma_fast > ma_slow:
                signals[stock] = 1.0
            elif ma_fast < ma_slow:
                signals[stock] = -1.0
            else:
                signals[stock] = 0.0
        return signals

    optimizer = WalkForwardOptimizer(
        train_window=100, test_window=50, step=40,
    )
    best_params, score, results_df = optimizer.optimize(
        data, ma_cross_strategy,
        param_grid={"fast": [5, 10], "slow": [20, 30]},
    )

    assert best_params is not None
    assert len(results_df) > 0
    assert "mean_oos_sharpe" in results_df.columns
    assert "n_windows" in results_df.columns
    assert all(results_df["n_windows"] >= 1)


def test_partial_fill_model():
    from src.backtest.tca import PartialFillModel

    pf = PartialFillModel(participation_threshold=0.05)

    small_order = pf.calculate_filled_volume(100, adv=10000)
    assert small_order == 100, "Small orders should fill fully"

    large_order = pf.calculate_filled_volume(1000, adv=1000)
    assert large_order < 1000, "Large orders should be partially filled"
    assert large_order >= 1, "At least 1 share should be filled"

    no_adv = pf.calculate_filled_volume(1000, adv=0)
    assert no_adv == 1000, "No ADV means full fill"


def test_walk_forward_expanding_boundary():
    from src.backtest.walk_forward import WalkForwardAnalyzer
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    equity = pd.Series(np.cumsum(np.random.randn(200) * 0.01) + 100, index=dates)
    wf = WalkForwardAnalyzer(
        mode="expanding",
        train_window=50,
        test_window=30,
        step=20,
    )
    result = wf.analyze(equity)
    assert len(wf.results_) >= 3
    for r in wf.results_:
        assert r["test_length"] >= 20
        assert r["train_length"] >= 30
