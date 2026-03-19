---
description: 基于 issues CSV 执行闭环（开发→Review→自验收→提交）
argument-hint: "<issues CSV 文件路径>"
---

你现在处于「Issues CSV 执行模式（闭环）」。

目标：以 `issues/*.csv` 为任务边界与状态源，推进并交付 issue 的完整闭环：**实现 → Review → 自我验收 → Git 提交**（不 push）。

> 说明：本 prompt 只在用户显式调用 `/prompts:issues_csv_execute` 时生效，不影响普通对话。

## 一、总体行为约定（必须遵守）

1. **CSV 是边界与状态源**：只做 CSV 这一行描述的工作；任何需求变更先写回 CSV，再改代码。
2. **默认完成整个 CSV（顺序由你决定）**：自行决定执行顺序（优先处理高价值/高优先级/能解阻塞的任务）。目标是把 CSV 里的所有 issues 推到"闭环完成"。每完成一条都必须把 **代码 + 当前 CSV 文件** 一起提交（同 commit 演进）。
3. **闭环不可缺省**：实现 + pytest 验证 + Review + 自我验收 + Git commit 缺一不可。
4. **状态驱动**：仅使用枚举值更新状态字段：
   - `dev_state`：`未开始|进行中|已完成`
   - `review_initial_state`：`未开始|进行中|已完成`
   - `review_regression_state`：`未开始|进行中|已完成`
   - `git_state`：`未提交|已提交`
5. **执行类任务必须追踪进度**：多步任务（≥2 步）必须使用 `update_plan` 工具推进。
6. **KISS / YAGNI**：不做无关重构；不引入新架构；优先修根因；保持向后兼容性。
7. **不假想结果，但允许"受限验收"**：
   - 能跑测试就跑，优先用 pytest 作为证据。
   - 若因环境/权限/JQData API 限流导致测试无法运行：允许继续提交，但必须在该行 `notes` 写清：`validation_limited:<原因>`；`manual_test:<后续可执行的命令/步骤>`；`evidence:<已完成的替代验证>`；`risk:<low|medium|high> <说明>`。
   - 受限验收下禁止声称"测试通过"，交接输出必须明确"未运行哪些测试/为何未运行"。

**补充约定（工具/安全）**：

- **Shell 与文件系统**：读多写少；避免破坏性命令；大范围操作先小范围试验
- **安全与合规**：不泄露 API 凭证（JQData）；回测结果注明天赋偏差风险
- **唯一状态源**：你必须把"用户传入的这一个 CSV 文件"当作唯一状态源与提交对象
- **禁止擅自新建/同步**：不要创建/更新 `issues/issues.csv` 或任何其它"汇总/快照 CSV"

## 二、工作流程（执行版，来自 AGENTS.md）

每条 issue 的实现过程，必须按以下顺序推进：

1. **接收与现实检查**
   - 清晰重述该 issue 的目标与验收口径，确认问题真实存在且值得解决
   - 识别潜在破坏性变更（兼容性、数据迁移、接口变更）
   - 持久性原则：遇到不确定性时选择最合理假设继续

2. **上下文收集 `<context_gathering>`（最小必要）**
   - 方法：从广泛开始再聚焦；优先目标查询；优先从 `refs` 指向文件切入
   - 预算：首次上下文收集控制在 5–8 次工具调用内
   - 早停：能够命名"要修改哪些具体文件/函数"即可进入实现

3. **执行（实现 + 文档同步）**
   - 通过工具实际修改文件/运行命令，不假想结果
   - 失败要捕获 stdout/stderr 并分析再决定重试/回退

4. **量化专项验证（pytest 闭环）**
   - 能跑 pytest 就跑：`python -m pytest tests/ -v --tb=short`
   - 优先跑与改动最相关的测试模块：`python -m pytest tests/test_<module>.py -v`
   - pytest 通过后再推进状态

5. **验证与自我反思 `<self_reflection>`**
   - 最终化前自评：可维护性 / 测试覆盖 / 性能 / 安全性 / 代码风格 / 文档 / 向后兼容性
   - **量化专项检查**：
     - 因子计算 NaN 传播是否正确处理（warmup 期间合理 NaN）
     - 回测引擎无未来数据泄露（look-ahead 检查）
     - 随机数是否设置了 seed
     - 收益率计算是否正确（pct_change 或对数收益率）

6. **交接**
   - 简要结论（做了什么、当前状态）
   - 给出关键文件引用（`path:line`）
   - 显式列出风险与后续步骤

## 三、输入与选择 issue 规则

1. `$ARGUMENTS` 必须提供一个 issues CSV 路径（相对/绝对均可）
2. **"完成"判定**：仅当该行同时满足：`dev_state=已完成`、`review_initial_state=已完成`、`review_regression_state=已完成`、`git_state=已提交`，才视为"闭环完成"
3. **每轮选择一行的规则（顺序由你决定，但要可解释）**：
   - 先收敛半成品：若存在 `git_state=未提交` 且（`dev_state=进行中` 或 `dev_state=已完成`）的行，优先从这些行里选一条先完成提交
   - 再选可交付项：在其余"未闭环完成"的行中，自主决定下一条（建议顺序：P0 → P1 → P2）
   - 选中后需给出 1 句话理由（写入该行 `notes`，例如 `picked_reason:<...>`）
4. **阻塞策略**：
   - 单条 issue 若出现"硬阻塞"（需要用户决策/外部环境/权限/JQData API 限流）：按「五、失败/阻塞处理」落盘后，允许切到下一条继续
   - 当所有剩余未闭环完成的 issues 都处于阻塞状态：停止并汇总阻塞清单

## 四、执行闭环（12步，逐条闭环）

### 0. 接收与建立 update_plan

- 用 1–2 句话重述：本轮执行的 CSV 路径、当前要处理的 `id/title`、验收口径
- 用 `update_plan` 建立并追踪本轮执行计划（建议 3 步：读取/校验 CSV → 循环处理 issues → 汇总交接）

### 1. 读取 CSV + 校验表头

- 必须包含固定表头（16列）
- 快照策略：日常推进只维护用户本次传入的 CSV 文件
- **不要**自动新建 `issues/YYYY-...csv`

### 2. 锁定目标行并输出摘要

- 输出：`id/title/description/acceptance_criteria/test_mcp/refs`（简洁即可）

### 3. 补齐执行信息（如缺失）

- `acceptance_criteria` 必须可验证（最好给量化指标阈值）
- `review_initial_requirements` 与 `review_regression_requirements` 必须可执行
- `test_mcp` 必须明确（`pytest | integration | manual`）
- `refs` 至少 1 个 `path:line`
- 若需要变更这些字段：**先写入 CSV 再继续编码**

### 4. 启动状态并写回 CSV

- 将该行 `dev_state` 置为 `进行中`
- 将该行 `review_initial_state` 置为 `进行中`
- 保存 CSV（保持 **UTF-8 BOM**）

### 5. 上下文收集

- 优先从 `refs` 指向文件开始读
- 早停：能明确"要改哪些具体文件/函数"即可进入实现

### 6. 实现 + 量化专项验证

1. **实现前确认**：把 `acceptance_criteria` 拆成"可验证的最小变更集合"（优先 1–3 个可测点）
2. **最小变更设计**：复用项目既有模式；避免引入新架构/新依赖
3. **编码执行**：
   - 单一职责：函数只做一件事
   - NaN 处理：因子函数在返回前 dropna 或明确说明 NaN 位置
   - look-ahead 禁止：禁止在回测层使用 `.shift(-n)` 计算信号
   - 随机种子：涉及随机数的代码必须设置 seed
4. **pytest 验证（立即执行，不要等最后）**：
   - 修改完成后立即运行相关测试
   - `python -m pytest tests/test_<module>.py -v --tb=short`
   - pytest 失败先修再推进，不跳过

### 7. Review（两段式）

- 对照 `review_initial_requirements` 完成开发过程自查，并将 `review_initial_state` 置为 `已完成`
- 对照 `review_regression_requirements` 执行回归/复测，并将 `review_regression_state` 置为 `已完成`
- 若 pytest 不可执行：走"受限验收"，在 `notes` 记录 `validation_limited`

### 8. 自我验收（严格按 acceptance_criteria）

- 给出"通过/未通过"的证据
- 按 `test_mcp` 运行最相关的测试（`pytest` 命令明确）
- 若无法运行测试：按"受限验收"记录 `notes`

### 9. 完成状态并写回 CSV

- 将该行 `dev_state` 置为 `已完成`
- 将该行 `git_state` 置为 `已提交`
- `notes` 追加：`done_at:<date>`、验收要点/证据摘要

### 10. Git 提交（闭环关键步骤）

- `git status` / `git diff` 确认变更边界只覆盖该 issue
- `git add` 必须包含：**代码变更 + 当前 CSV 文件**
- 提交粒度：**一条 issue 一个 commit**
- commit message：`[quant-v2-<id>] <title>`
- 若 `git commit` 失败：必须将该行 `git_state` 回滚为 `未提交`，在 `notes` 记录 `blocked:git commit failed <原因>`

### 11. 对话交接输出

- 本次处理的 `id/title`
- 本次完成条数 / 剩余未完成条数 / 阻塞 id
- 关键变更点与文件引用（`path:line`）
- pytest 实际运行结果
- 若采用受限验收：列出未运行测试/原因/`manual_test`
- 本地 commit hash

### 12. 循环与停止条件

- 每完成并提交一条后，回到「三、选择 issue 规则」选择下一条继续
- 直到：
  - **所有 issues 均达到"闭环完成"** → 进入 `/prompts:regression`
  - **所有剩余 issues 均阻塞** → 停止并汇总阻塞清单

## 五、失败/阻塞处理（必须落盘）

出现以下任一情况，优先尝试自行消化；若确实无法解决，按本节处理：

- 验收口径不清
- `refs` 找不到/代码定位失败
- pytest 失败且无法在当前上下文修复
- JQData API 限流导致测试数据不可得
- 需要改动超出该行 `description` 边界

**处理方式**：

1. 在该行 `notes` 记录：`blocked:<原因>` + 已做过的排查/下一步建议
2. `dev_state/review_*_state` 保持"真实进度"，`git_state` 必须保持 `未提交`
3. 继续策略：
   - 若还有其他不依赖该阻塞项的 issues：允许继续下一条推进
   - 若剩余 issues 全部阻塞：停止并用 1–3 句话向用户汇报阻塞点

## 六、使用示例

```
/prompts:issues_csv_execute issues\2026-03-19_14-30-00-quant-v2-optimization.csv
```

## 七、提交前自检清单

**闭环必过**：

- [ ] 该 issue 的验收口径有"可复现证据"（pytest 输出/复现步骤）
- [ ] 若采用受限验收：该行 `notes` 已写 `validation_limited/manual_test/evidence/risk`
- [ ] `review_initial_state` 与 `review_regression_state` 已按要求推进
- [ ] CSV 与代码一起提交（`git add` 覆盖两者）
- [ ] commit message 以 `[quant-v2-<id>] <title>` 开头

**量化专项自检**：

- [ ] 因子计算 NaN 传播正确（warmup 期间合理）
- [ ] 回测引擎无未来数据泄露（代码审查）
- [ ] 随机数设置了 seed
- [ ] 收益率计算正确（pct_change 或对数收益率）

**流程自检**：

- [ ] 接触工具前已记录"接收与现实检查"
- [ ] 首次上下文收集在 5–8 次工具调用内
- [ ] 使用 `update_plan` 追踪 ≥2 步且实时更新
- [ ] 交接输出包含 `path:line`、风险与后续步骤
