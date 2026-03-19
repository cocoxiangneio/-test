---
description: 规定一轮循环结束后如何开启下一轮
argument-hint: "<可选：指定下一轮任务描述>"
---

你现在处于「Iteration 循环协议模式」。

目标：规定每轮小循环（Plan → CSV → Execute → Regression）结束后，如何开启下一轮，形成可持续的迭代闭环。

> 核心原则：小循环永续轮转，直到项目完成或用户主动终止。

## 一、循环状态机

```
┌──────────────────────────────────────────────────────────┐
│                     状态 0：待机                          │
│  项目初始化后 或 用户提出新需求                            │
│  → 等待用户输入或自动进入 状态 1                          │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│                    状态 1：Plan                           │
│  用户：/prompts:plan <任务描述>                           │
│  → 生成 plan/YYYY-MM-DD_*.md                             │
│  → 人工走读/修改 Plan                                     │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│                    状态 2：CSV 生成                       │
│  用户：/prompts:plan_to_issues_csv <plan文件>             │
│  → 生成 issues/YYYY-MM-DD_*.csv（唯一状态源）             │
│  → 人工检查 CSV 边界                                      │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│                   状态 3：Execute 闭环                     │
│  用户：新会话 /prompts:issues_csv_execute <csv>           │
│  → 逐条闭环（实现→pytest→Review→commit）                 │
│  → 直到所有 issues 四态完成 或 全部阻塞                    │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│                   状态 4：Regression                      │
│  自动进入 /prompts:regression                             │
│  → 全量 pytest + CLI + 导入验证                           │
│  → 0 failed → commit 回归报告                             │
│  → > 0 failed → 修复 → 再回归                            │
└──────────────────────────────────────────────────────────┘
                           ↓
               ┌─────────────────────────────┐
               │    状态 5：进入下一轮        │
               │    回到 状态 1               │
               └─────────────────────────────┘
```

## 二、开启新轮的触发条件

以下任一条件满足时，可开启新的一轮：

| 触发条件 | 说明 | 操作 |
|---------|------|------|
| Regression 通过 | 所有 issues 闭环完成且全量回归 0 failed | 自动提示用户开启新 Plan |
| 用户新需求 | 用户提出新的任务描述 | 直接 `/prompts:plan <新任务>` |
| Regression 发现系统性缺陷 | 回归通过但发现架构性问题 | 先生成 `Report.md` 分析，再 `/prompts:plan` |
| 阻塞回收完成 | 之前阻塞的 issues 现在可以处理 | 回到 CSV Execute 继续 |

## 三、新轮入口操作

### 入口 1：用户直接提需求（最常用）

```
用户：/prompts:plan 帮我实现XXX功能
→ 直接进入 状态 1
```

### 入口 2：基于 Regression 报告（系统性改进）

```
用户：基于上一轮 regression 报告，帮我制定优化计划
AI：先生成 results/regression/ 分析文档（可选）
→ /prompts:plan <基于分析的任务描述>
→ 进入 状态 1
```

### 入口 3：阻塞回收（继续未完成的轮次）

```
用户：/prompts:issues_csv_execute <原CSV路径>
→ 继续处理阻塞的 issues
→ 完成后 → 状态 4 Regression
```

## 四、元数据维护（issues/.meta.csv）

每轮循环完成后，必须更新 `issues/.meta.csv`：

```csv
round,date,csv_file,commit_hash,issues_total,issues_completed,issues_blocked,regression_status
1,2026-03-19,2026-03-19_initial.csv,a1b2c3d,7,7,0,passed
```

**说明**：
- `round`：轮次编号（从 1 开始递增）
- `date`：完成日期
- `csv_file`：该轮使用的 CSV 文件名
- `commit_hash`：Regression 通过后最后一次 commit 的 hash
- `issues_total`：总 issues 数
- `issues_completed`：完成数
- `issues_blocked`：阻塞数
- `regression_status`：passed / failed

## 五、轮次递增规则

1. **新轮判断**：当上一轮 `regression_status=passed` 且用户提出新需求时，round + 1
2. **继续判断**：当上一轮有阻塞 issues 且阻塞解除时，round 不变，继续处理原 CSV
3. **回退判断**：当 regression 发现系统性缺陷需要架构重构时，round + 1（但先进入 Plan 分析）

## 六、对话内输出格式

循环状态切换时，输出：

```markdown
🔄 循环状态更新

📊 当前轮次：Round <N>
📁 当前 CSV：<csv文件名>
📊 完成进度：<M>/<K> issues
🔀 状态：<当前状态>

🎯 下一步操作：
- 继续当前轮 → `/prompts:issues_csv_execute <csv路径>`
- 完成当前轮 → `/prompts:regression`
- 开启新轮 → `/prompts:plan <新任务>`
```

## 七、一轮完整示例

```
Round 1（初始构建）：

1. 用户：/prompts:plan 搭建量化系统v2基础框架
2. AI：生成 plan/2026-03-19_14-00-00-quant-v2-initial-plan.md
3. 用户：人工走读，确认 plan 合理
4. 用户：/prompts:plan_to_issues_csv plan/2026-03-19_14-00-00-quant-v2-initial-plan.md
5. AI：生成 issues/2026-03-19_14-30-00-quant-v2-initial.csv（7条）
6. 用户：检查 CSV 边界，确认无误
7. 用户：新会话 /prompts:issues_csv_execute issues/2026-03-19_14-30-00-quant-v2-initial.csv
8. AI：逐条闭环，7条全部完成并 commit
9. AI（自动）：/prompts:regression
10. AI：全量回归 0 failed
11. AI：commit 回归报告
12. AI：更新 issues/.meta.csv（Round 1 完成）

Round 2（下一轮优化）：

13. 用户：/prompts:plan 增加因子IC验证框架
14. ...（同上流程）
```
