---
description: Execute 循环结束后执行全量回归验证
argument-hint: "<issues CSV 文件路径（可选，默认为最后一次执行的 CSV）>"
---

你现在处于「Regression 回归验证模式」。

目标：在 `/prompts:issues_csv_execute` 循环结束后，执行**全量回归测试**，验证本次所有 issues 的改动未引入回归，并将回归结果输出到 `results/regression/` 目录。

> 核心原则：Regression 是每轮循环的"质量门禁"。0 failed 才能开启下一轮 Plan。

## 一、触发条件

**自动触发**：当 `/prompts:issues_csv_execute` 处理完 CSV 中所有 issues（全部达到四态完成或全部阻塞）后，自动进入 Regression 模式。

**手动触发**：用户可随时显式调用 `/prompts:regression <csv路径>` 进行回归验证。

## 二、回归内容（3 个层次）

### 层次 1：全量 pytest（必须执行）

```bash
python -m pytest tests/ -v --tb=short
```

覆盖所有 7 个测试模块：test_data / test_factors / test_strategies / test_backtest / test_portfolio / test_optimization / test_evaluation

**通过标准**：0 failed（允许 warning 但必须可解释）

### 层次 2：CLI 接口验证（必须执行）

```bash
python -m src.cli --help
python -m src.cli run-backtest --help
python -m src.cli optimize --help
```

验证 CLI 命令入口未被破坏。

### 层次 3：模块导入验证（必须执行）

```bash
python -c "from src.data import fetcher, loader, cache; from src.factors import technical, fundamental, cross_sectional; from src.strategies import base; from src.backtest import engine, commission, slippage; from src.portfolio import optimizer, risk_manager; from src.optimization import ga_optimizer, pso_optimizer, bayesian_optimizer; from src.evaluation import metrics, visualization, report; print('ALL OK')"
```

验证所有模块可正常导入（无 ImportError / SyntaxError）。

## 三、回归报告生成

回归完成后，必须生成报告文件并写入：

**文件路径**：`results/regression/YYYY-MM-DD_HH-mm-ss.md`

**报告结构**：

```markdown
---
mode: regression
cwd: F:\github\quant_v2
csv_source: <最后一次执行的 CSV 文件路径>
pytest_result: <passed/failed>
pytest_command: python -m pytest tests/ -v --tb=short
created_at: <ISO8601>
---

# Regression Report: <日期时间>

## 回归范围
- CSV 源：`issues/YYYY-MM-DD_*.csv`
- issues 数量：<N> 条
- 全部闭环完成：<M> 条
- 阻塞未完成：<K> 条

## pytest 全量结果
| 模块 | 结果 | 失败数 | 通过数 |
|-----|------|-------|-------|
| test_data.py | PASSED | 0 | 3 |
| test_factors.py | PASSED | 0 | 7 |
| ... | ... | ... | ... |

**总计**：<X> passed, <Y> failed

## CLI 接口验证
| 命令 | 结果 |
|-----|------|
| `python -m src.cli --help` | PASSED |
| `python -m src.cli run-backtest --help` | PASSED |
| `python -m src.cli optimize --help` | PASSED |

## 模块导入验证
**结果**：PASSED / FAILED
<如有失败，列出具体导入错误>

## 回归结论
- **通过**：0 failed，所有回归项正常
- **未通过**：<N> failed，需要修复后重新回归

## 下一步
- 通过 → 开启下一轮 `/prompts:plan`
- 未通过 → 进入修复流程 → 重新 `/prompts:regression`
```

## 四、回归失败处理

当 pytest 结果 > 0 failed 时：

1. **分析失败原因**：运行 `pytest tests/ -v` 查看具体失败项
2. **判断失败类型**：
   - **新引入的失败**：本次某条 issue 的改动导致 → 回到 issues_csv_execute 修复该条
   - **历史遗留的失败**：之前就存在的问题 → 记录到回归报告中，允许进入下一轮但需注明
3. **修复后重新回归**：修复完成后再次运行 `python -m pytest tests/ -v --tb=short`
4. **回归报告追加**：在报告中记录修复过程和结果

## 五、执行步骤

1. 确认 CSV 源文件路径（命令行参数或最后一次执行的 CSV）
2. 执行层次 1：全量 pytest，记录输出
3. 执行层次 2：CLI 接口验证
4. 执行层次 3：模块导入验证
5. 生成回归报告（UTF-8 编码）
6. 根据结果决定：
   - 0 failed → commit 回归报告 → 提示用户开启下一轮 Plan
   - > 0 failed → 分析原因 → 修复 → 重新回归

## 六、Git 提交

回归报告通过后，必须将报告 commit：

```bash
git add results/regression/YYYY-MM-DD_*.md
git commit -m "[regression] YYYY-MM-DD 全量回归通过"
```

如果回归失败：报告仍写入，但不 commit，等修复后再 commit 通过的版本。

## 七、对话内输出格式

回归完成后输出：

```markdown
🎯 Regression 回归结果

📊 回归范围：
- issues 完成：<M>/<N> 条
- 阻塞：<K> 条

✅ pytest 全量：<X> passed, <Y> failed
✅ CLI 接口：全部通过
✅ 模块导入：全部通过

📋 回归报告：
- 路径：`results/regression/YYYY-MM-DD_*.md`

🔄 下一步：
- 0 failed → `/prompts:plan <新任务>` 开启下一轮
- > 0 failed → 修复中 → 重新回归
```
