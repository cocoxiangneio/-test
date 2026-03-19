# Quantitative Analysis System v2

## 核心能力

| 模块 | 功能 | 数据源 |
|------|------|--------|
| 数据层 | JoinQuant API 封装、缓存、统一加载器 | 聚宽 (jqdatasdk) |
| 因子层 | 40+ 技术因子、8 基本面因子、截面因子 | 聚宽 |
| 策略层 | 28 种策略（动量/均值回归/突破/多周期/ML） | - |
| 回测层 | 事件驱动回测、佣金、滑点、止损止盈 | - |
| 组合层 | 均值方差/最小方差/风险平价/HRP 优化 | - |
| 优化层 | GA/PSO/Bayesian 因子权重优化 | - |
| 评估层 | Sharpe/Calmar/回撤/IC/IR 等指标 + 可视化 | - |

## 快速开始

```bash
pip install -e .
python -m src.cli run-backtest --stocks 000001.XSHE --start 2024-01-01 --end 2025-01-01
python -m src.cli optimize --stocks 000001.XSHE --algorithm ga
python -m src.cli analyze --stock 000001.XSHE
```

## 项目结构

```
src/
  data/       # 数据获取与加载
  factors/    # 因子计算与注册
  strategies/ # 策略实现
  backtest/   # 回测引擎
  portfolio/  # 组合优化与风控
  optimization/# 参数优化算法
  evaluation/ # 评估与可视化
```
