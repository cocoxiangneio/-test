# -*- coding: utf-8 -*-
"""Tests for CLI."""

import pytest
import sys
import os
import argparse


def test_cli_import():
    from src import cli
    assert hasattr(cli, "main")
    assert hasattr(cli, "run_backtest")
    assert hasattr(cli, "run_optimize")
    assert hasattr(cli, "load_yaml_config")
    assert hasattr(cli, "merge_cli_with_yaml")


def test_cli_run_backtest_help():
    from src import cli
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    try:
        sys.argv = ["cli", "run-backtest", "--help"]
        with redirect_stdout(f):
            cli.main()
    except SystemExit:
        pass
    output = f.getvalue()
    assert "usage" in output.lower()


def test_load_yaml_config(tmp_path):
    from src.cli import load_yaml_config
    import yaml
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "backtest": {"stocks": ["600036.XSHG"], "cash": 200000.0, "start": "2024-06-01"},
        "optimization": {"algorithm": "pso", "n_gen": 20},
        "data": {"cache_dir": "test_cache"},
    }
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")
    loaded = load_yaml_config(str(config_file))
    assert loaded["backtest"]["stocks"] == ["600036.XSHG"]
    assert loaded["backtest"]["cash"] == 200000.0
    assert loaded["optimization"]["algorithm"] == "pso"


def test_merge_cli_with_yaml():
    from src.cli import merge_cli_with_yaml
    args = argparse.Namespace(
        command="run-backtest",
        config=None,
        stocks=["000001.XSHE"],
        start="2024-01-01",
        end="2025-01-01",
        cash=100000.0,
    )
    yaml_config = {
        "backtest": {"stocks": ["600036.XSHG"], "cash": 500000.0, "start": "2024-06-01"},
        "optimization": {},
        "data": {},
    }
    merged = merge_cli_with_yaml(args, yaml_config)
    assert merged.stocks == ["600036.XSHG"]
    assert merged.cash == 500000.0
    assert merged.start == "2024-06-01"
    assert merged.end == "2025-01-01"


def test_merge_cli_optimize_yaml():
    from src.cli import merge_cli_with_yaml
    args = argparse.Namespace(
        command="optimize",
        config=None,
        stocks=["000001.XSHE"],
        start="2024-01-01",
        end="2025-01-01",
        algorithm="ga",
        n_gen=30,
        pop_size=30,
        n_trials=100,
    )
    yaml_config = {
        "backtest": {},
        "optimization": {"algorithm": "bayesian", "n_gen": 50, "pop_size": 50},
        "data": {},
    }
    merged = merge_cli_with_yaml(args, yaml_config)
    assert merged.algorithm == "bayesian"
    assert merged.n_gen == 50
    assert merged.pop_size == 50


def test_load_yaml_config_missing_file(tmp_path):
    from src.cli import load_yaml_config
    missing = tmp_path / "nonexistent.yaml"
    try:
        load_yaml_config(str(missing))
        assert False, "Should have exited"
    except SystemExit:
        pass


def test_load_yaml_config_invalid():
    import tempfile
    from src.cli import load_yaml_config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("invalid: yaml: content: [}")
        name = f.name
    try:
        load_yaml_config(name)
        assert False, "Should have exited"
    except SystemExit:
        pass
    finally:
        os.unlink(name)


def test_cli_optimize_help():
    from src import cli
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    try:
        sys.argv = ["cli", "optimize", "--help"]
        with redirect_stdout(f):
            cli.main()
    except SystemExit:
        pass
    output = f.getvalue()
    assert "usage" in output.lower()
    assert "--config" in output
