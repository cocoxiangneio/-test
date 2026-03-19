---
mode: plan
cwd: F:\github\quant_v2
task: 量化系统v2优化与补强
complexity: system
planning_method: builtin
created_at: 2026-03-19T14:00:00+08:00
---

# Plan: 量化系统v2优化与补强

🎯 任务概述

量化系统v2基础框架已完成（数据层/因子层/策略层/回测层/组合层/优化层/评估层），共31个策略、40+技术因子、3种优化算法。本轮在基础框架上增加系统可信度与可验证性，重点补强因子有效性验证、全链路集成测试、评估报告增强。

📋 执行计划

1. **因子IC时序验证框架**：在 factors/cross_sectional.py 基础上构建完整的IC/IR时序验证流程，支持多因子IC计算、月度IC衰减分析、IC>0.02阈值告警

2. **回测引擎集成测试**：构造端到端测试数据流（因子计算→策略信号→回测引擎→指标输出），验证无数据泄露、佣金滑点正确、止损止盈触发

3. **策略参数网格搜索工具**：在现有优化器基础上增加网格搜索（GridSearch），支持策略参数穷举回测，自动输出最优参数组合

4. **组合优化器与风险管理集成**：将portfolio optimizer与backtest engine打通，支持优化后的权重直接用于回测，验证风控约束生效

5. **评估报告PDF导出增强**：在report.py基础上增加PDF格式报告输出，支持多策略对比表格、权益曲线+回撤叠加图

6. **CLI配置YAML支持**：在config.py基础上增加YAML配置文件解析，支持以YAML文件而非命令行参数指定回测/优化参数

7. **数据层CSV/Parquet文件输入**：扩展DataLoader支持直接读取本地CSV/Parquet文件作为数据源，降低对JQData API的依赖

🧠 关键假设

- 因子IC验证使用模拟数据，真实数据验证需要manual确认
- PDF导出依赖reportlab/weasyprint等库，需在pyproject.toml中添加依赖
- YAML配置解析依赖pyyaml库，需在pyproject.toml中添加依赖

⚠️ 风险与注意事项

- **过拟合风险**：参数网格搜索可能在历史数据上过拟合，需在验收标准中要求IC验证
- **数据泄露风险**：集成测试必须验证无未来数据使用（.shift(-n)仅限forward_returns）
- **API限流风险**：真实JQData数据验证时需注意API调用频率

📎 参考

- `src/factors/cross_sectional.py:1`（现有IC/IR计算）
- `src/backtest/engine.py:1`（回测引擎）
- `src/evaluation/report.py:1`（报告生成）
- `src/cli.py:1`（CLI入口）
- `src/data/loader.py:1`（数据加载）
