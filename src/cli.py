# -*- coding: utf-8 -*-
"""CLI entry point for quant_v2."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        logging.error("PyYAML is required for config files. Install with: pip install pyyaml")
        sys.exit(1)

    if not os.path.exists(config_path):
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Invalid YAML in {config_path}: {e}")
            sys.exit(1)

    return config if config is not None else {}


def merge_cli_with_yaml(args: argparse.Namespace, yaml_config: Optional[Dict[str, Any]]) -> argparse.Namespace:
    if not yaml_config:
        return args
    command = getattr(args, "command", None)
    if command == "run-backtest":
        sec = yaml_config.get("backtest", {})
        if hasattr(args, "stocks") and sec.get("stocks"):
            args.stocks = sec["stocks"]
        if hasattr(args, "start") and sec.get("start"):
            args.start = sec["start"]
        if hasattr(args, "end") and sec.get("end"):
            args.end = sec["end"]
        if hasattr(args, "cash") and sec.get("cash") is not None:
            args.cash = float(sec["cash"])
    elif command == "optimize":
        sec = yaml_config.get("optimization", {})
        if hasattr(args, "stocks") and sec.get("stocks"):
            args.stocks = sec["stocks"]
        if hasattr(args, "start") and sec.get("start"):
            args.start = sec["start"]
        if hasattr(args, "end") and sec.get("end"):
            args.end = sec["end"]
        if hasattr(args, "algorithm") and sec.get("algorithm"):
            args.algorithm = sec["algorithm"]
        if hasattr(args, "n_gen") and sec.get("n_gen") is not None:
            args.n_gen = int(sec["n_gen"])
        if hasattr(args, "pop_size") and sec.get("pop_size") is not None:
            args.pop_size = int(sec["pop_size"])
    return args


def run_backtest(args):
    from src.backtest.engine import BacktestEngine
    from src.data.loader import DataLoader
    from src.strategies.breakout import MaCrossStrategy
    from src.evaluation.metrics import MetricsCalculator
    from src.evaluation.visualization import plot_equity_curve
    from src.evaluation.report import ReportGenerator
    import pandas as pd

    logging.info(f"Loading data for {args.stocks} from {args.start} to {args.end}")
    loader = DataLoader()
    data_dict = loader.load_multiple_kline(args.stocks, args.start, args.end, use_cache=False)

    if not data_dict:
        logging.error("No data loaded. Check credentials and date range.")
        return

    stock = args.stocks[0]
    df = data_dict[stock]
    strategy = MaCrossStrategy({"fast": 5, "slow": 20})

    def strat_func(data, date):
        signals = {}
        for s, d in data.items():
            if date in d.index:
                sig = strategy.signal(d)
                if date in sig.index:
                    signals[s] = float(sig.loc[date])
        return signals

    engine = BacktestEngine(
        initial_cash=args.cash,
        commission_rate=0.0003,
        slippage_pct=0.001,
        stop_loss=0.05,
        take_profit=0.10,
    )
    result = engine.run(data_dict, strat_func)

    calc = MetricsCalculator()
    metrics = calc.calculate(result.equity_curve, trades=result.trades)
    logging.info(f"Results: Sharpe={metrics['sharpe_ratio']:.3f}, "
                  f"Return={metrics['total_return']*100:.2f}%, "
                  f"MaxDD={metrics['max_drawdown']*100:.2f}%")

    report_gen = ReportGenerator()
    report_gen.save_metrics_report(metrics, "MaCrossStrategy", stock)
    fig = plot_equity_curve(result.equity_curve, title=f"{stock} - MaCross Backtest")
    fig.savefig(f"results/{stock}_equity.png", dpi=150, bbox_inches="tight")
    logging.info(f"Report saved to results/")


def run_optimize(args):
    from src.optimization.ga_optimizer import GAOptimizer
    from src.optimization.pso_optimizer import PSOOptimizer
    from src.optimization.bayesian_optimizer import BayesianOptimizer
    from src.optimization.bayesian_optimizer import ObjectiveFunctions
    from src.data.loader import DataLoader
    from src.evaluation.visualization import plot_convergence
    import numpy as np

    logging.info(f"Optimizing {args.stocks} with {args.algorithm}")
    loader = DataLoader()
    data_dict = loader.load_multiple_kline(args.stocks, args.start, args.end, use_cache=False)

    if not data_dict:
        logging.error("No data loaded.")
        return

    stocks = list(data_dict.keys())
    returns_dict = {}
    for stock, df in data_dict.items():
        if "close" in df.columns:
            returns_dict[stock] = df["close"].pct_change().dropna()

    import pandas as pd
    returns_df = pd.DataFrame(returns_dict)

    n_weights = len(stocks)
    if args.algorithm == "ga":
        opt = GAOptimizer(n_generations=args.n_gen, pop_size=args.pop_size, seed=42)
        obj_func = lambda w: ObjectiveFunctions.composite(returns_df, w)
    elif args.algorithm == "pso":
        opt = PSOOptimizer(n_particles=args.pop_size, n_iterations=args.n_gen, seed=42)
        obj_func = lambda w: ObjectiveFunctions.composite(returns_df, w)
    else:
        opt = BayesianOptimizer(n_trials=args.n_trials or 100, seed=42)
        obj_func = lambda w: ObjectiveFunctions.composite(returns_df, w)

    best_weights, best_score, history = opt.optimize(obj_func, n_weights)
    weights_dict = {stocks[i]: float(best_weights[i]) for i in range(n_weights)}

    logging.info(f"Best score: {best_score:.4f}")
    logging.info(f"Weights: {weights_dict}")

    fig = plot_convergence(history, title=f"Convergence ({args.algorithm.upper()})")
    fig.savefig(f"results/optimization_convergence_{args.algorithm}.png", dpi=150, bbox_inches="tight")
    logging.info("Optimization complete.")


def main():
    parser = argparse.ArgumentParser(description="quant_v2 - Stock Quantitative Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    p_backtest = subparsers.add_parser("run-backtest", help="Run backtest")
    p_backtest.add_argument("--config", default=None, help="YAML config file path")
    p_backtest.add_argument("--stocks", nargs="+", default=["000001.XSHE"], help="Stock codes")
    p_backtest.add_argument("--start", default="2024-01-01", help="Start date")
    p_backtest.add_argument("--end", default="2025-01-01", help="End date")
    p_backtest.add_argument("--cash", type=float, default=100000.0, help="Initial cash")
    p_backtest.set_defaults(func=run_backtest)

    p_opt = subparsers.add_parser("optimize", help="Run optimization")
    p_opt.add_argument("--config", default=None, help="YAML config file path")
    p_opt.add_argument("--stocks", nargs="+", default=["000001.XSHE"], help="Stock codes")
    p_opt.add_argument("--start", default="2024-01-01", help="Start date")
    p_opt.add_argument("--end", default="2025-01-01", help="End date")
    p_opt.add_argument("--algorithm", choices=["ga", "pso", "bayesian"], default="ga", help="Algorithm")
    p_opt.add_argument("--n-gen", type=int, default=30, help="Generations/iterations")
    p_opt.add_argument("--pop-size", type=int, default=30, help="Population size")
    p_opt.add_argument("--n-trials", type=int, default=100, help="Bayesian trials")
    p_opt.set_defaults(func=run_optimize)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    yaml_config = None
    if hasattr(args, "config") and args.config:
        yaml_config = load_yaml_config(args.config)
        args = merge_cli_with_yaml(args, yaml_config)
    args.func(args)


if __name__ == "__main__":
    main()
