---
description: 从 plan/*.md 生成可维护的 issues CSV 快照（含开发/Review/Git 状态、验收边界、测试方式）
argument-hint: "<plan 文件路径（可选，默认取 plan/ 最新）>"
---

你现在处于「Plan → Issues CSV 模式」。

目标：把当前项目的 `plan/*.md`（由 `/prompts:plan` 生成的执行计划）转换为可落盘、可协作维护的 **唯一命名 issues CSV 快照**（`issues/<timestamp>-<slug>.csv`），并确保该 CSV 可以作为代码的一部分提交到仓库中，用于长期追踪任务边界与状态。

> 核心原则：ISSUES CSV 是"会议落盘的任务边界合同"，不是 AI 自嗨文档。
> CSV 要能防止任务跑偏：每条必须明确 **做什么、怎么验收、怎么 review、用什么测试工具**。

## 一、输入与默认行为

1. `$ARGUMENTS` 允许为空：
   - 若为空：默认选择当前项目 `plan/` 目录下**最新**的 `*.md` 作为输入
   - 若不为空：视为 `plan` 文件路径（相对/绝对均可）
2. 你必须读取该 `plan` 文件内容，必要时可根据 `📎 参考` 中的文件路径进一步读取少量上下文（只读、最小必要）
3. 若找不到 `plan` 文件或内容不足以拆分任务：用 1–2 句话说明原因，并给出需要的最小补充信息

## 二、总体行为约定（必须遵守）

1. 你是"任务拆分与落盘助手"，目标是生成**可维护**的 CSV，而不是输出大量散文
2. 禁止使用百分比进度；所有进度必须使用状态枚举（见「五、状态字段」）
3. 每条任务必须包含：
   - `acceptance_criteria`：可验证、可测试的验收口径（尽量量化）
   - `review_initial_requirements`：边开发边 Review 的要求
   - `review_regression_requirements`：全量完成后的回归/复测要求
   - `test_mcp`：明确该任务默认用哪个测试执行器
4. 详细背景与推理不应堆进 CSV：尽量通过 `refs` 指向 `plan/*.md` 来承载细节
5. 生成后必须将 CSV 写入项目的 `issues/` 目录：
   - 生成一个**唯一命名**的快照文件
   - **禁止**创建/更新 `issues/issues.csv`、`issues.csv` 或任何其它固定文件名的"汇总版"

## 三、量化系统专用字段扩展

### area 枚举（扩展，比原版更细致）

`data | factors | strategies | backtest | portfolio | optimization | evaluation | cli | infra`

### test_mcp 枚举（量化系统专用）

| 值 | 说明 | 验证命令 |
|---|------|---------|
| `pytest` | 单元测试（因子/策略/优化器/回测/评估） | `python -m pytest tests/test_<module>.py -v` |
| `integration` | 全链路集成测试（因子→策略→回测→指标） | 手动构造端到端数据流验证 |
| `manual` | 需人工确认（实盘模拟对比、真实 API 数据验证） | 在 notes 写明手动步骤 |

### 量化专项验收要求

- `acceptance_criteria` 必须包含：量化指标阈值（如 Sharpe > 1.0 / IC > 0.02）
- `review_initial_requirements` 必须检查：NaN 处理、look-ahead 禁止、随机种子
- `review_regression_requirements` 必须包含：pytest 全量通过、无未来数据泄露证据

## 四、拆分规则（从 plan 到 issues）

将 `plan` 中的 Phase/步骤转换为 issues 行，遵循：

1. 默认粒度：**一条 Phase 对应一条 issues**
2. 允许拆分：若某个 Phase 同时包含明显独立的多项工作，可拆分为多行
3. 建议规模：一般 5–15 行最易维护；超过 15 行时，优先合并同类项
4. 量化系统特殊规则：
   - 涉及数据层改动 → 必须有 `data` area
   - 涉及回测 → 必须有 `integration` test_mcp
   - 涉及因子 → 必须有 IC/IR 相关验收标准

## 五、CSV Schema（固定表头，16列）

```
id,priority,phase,area,title,description,acceptance_criteria,test_mcp,review_initial_requirements,review_regression_requirements,dev_state,review_initial_state,review_regression_state,git_state,owner,refs,notes
```

字段含义与填写要求：

- `id`：任务唯一标识（建议：`<PREFIX>-000`，以 10 递增）
- `priority`：`P0|P1|P2`
- `phase`：来源 Phase 序号
- `area`：data | factors | strategies | backtest | portfolio | optimization | evaluation | cli | infra
- `title`：一句话标题
- `description`：1–2 句说明"做什么"，强调边界，不写实现细节
- `acceptance_criteria`：可测试的验收标准（含指标/阈值/复现步骤）
- `test_mcp`：`pytest | integration | manual`
- `review_initial_requirements`：开发过程中的 Review 要点
- `review_regression_requirements`：最终回归/复测要点
- `dev_state`：`未开始|进行中|已完成`
- `review_initial_state`：`未开始|进行中|已完成`
- `review_regression_state`：`未开始|进行中|已完成`
- `git_state`：`未提交|已提交`
- `owner`：默认留空
- `refs`：引用与跳转（强制，使用 `path:line`）
- `notes`：自由备注

## 六、状态字段（枚举，禁止百分比）

- `dev_state`：`未开始|进行中|已完成`
- `review_initial_state`：`未开始|进行中|已完成`
- `review_regression_state`：`未开始|进行中|已完成`
- `git_state`：`未提交|已提交`

默认值：生成时全部填 `未开始`，`git_state` 填 `未提交`

## 七、文件命名与编码

1. 目录：确保项目根目录下存在 `issues/`（不存在则创建）
2. 唯一命名快照（必须创建）：
   - 文件名：`issues/YYYY-MM-DD_HH-mm-ss-<slug>.csv`
   - 时间戳使用**当前时间**
3. 禁止生成"汇总入口"CSV
4. 编码：**UTF-8 with BOM**（Excel 友好，避免中文乱码）
   - Windows PowerShell：使用 `.NET UTF8Encoding($true)` 写文件

## 八、CSV 输出规范

1. 必须输出合法 CSV：
   - 表头一行
   - 每行字段数与表头一致
   - 字段内出现逗号/换行/双引号时必须正确转义
2. **所有字段统一使用双引号包裹**（最稳策略）
3. `refs` 中的路径必须精确到 `file:line`

## 九、执行步骤

1. 定位并读取输入 `plan` 文件
2. 从 plan 的 Phase/步骤拆出 issues 行，补齐每行的验收/Review/test_mcp/refs
3. 在 `issues/` 下写入唯一命名快照 CSV（UTF-8 BOM）
4. 校验：检查状态字段是否只使用枚举值、`refs` 是否存在且非空

## 十、对话内输出格式

完成后只输出关键信息：
- 生成的快照路径
- 行数统计（多少条 issues）
- 风险/注意事项（如 BOM、Excel 锁文件）
- 下一步建议命令：`/prompts:issues_csv_execute <上面生成的快照路径>`
