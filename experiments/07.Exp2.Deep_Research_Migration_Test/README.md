# Deep Research Migration Test (Exp2)

## 快速开始 (5分钟上手)

### 1. 启动服务

```bash
cd experiments/07.Exp2.Deep_Research_Migration_Test

# 基线实验（无重部署）- 48个实例
./start_all_services_no_redeploy.sh 24 24

# 或者：带预测式重部署 - 48个实例（需要双倍用于迁移）
./start_all_services.sh 24 24
```

### 2. 运行实验

```bash
# 基线实验（无重部署，默认两阶段模式，QPS自动硬编码）
uv run python test_dynamic_workflow.py \
    --num-workflows 500 \
    --strategies probabilistic

# Oracle 预测式重部署实验（需要配合 start_all_services.sh）
uv run python test_dynamic_workflow.py \
    --num-workflows 500 \
    --strategies probabilistic \
    --oracle-deploy
```

> **注意**: 两阶段模式默认开启，QPS 使用硬编码值（Phase1=0.50, Phase2=0.29），忽略 `--qps` 参数。

### 3. 停止服务

```bash
./stop_all_services.sh
```

---

## 实验概述

本实验模拟 **Deep Research 工作流**，测试预测式重部署是否能提升性能。

### 工作流模式

```
A1 (boot) → n × B1 (query) → n × B2 (criteria) → A2 (summary) → 完成
   30s         3s each           2s each            25s
```

- **A1 任务**: 启动阶段，~30秒
- **B1/B2 任务**: 并行查询和评估，每个 B1 3秒，B2 2秒
- **A2 任务**: 合并摘要，~25秒
- **Fanout**: 每个工作流产生 n 个 B 任务（n 可变）

### 任务执行时间

| 任务类型 | 分布 | 平均时间 |
|---------|------|---------|
| A1 (boot) | N(30, 2²) | 30 秒 |
| A2 (summary) | N(25, 2²) | 25 秒 |
| B1 (query) | N(3, 0.3²) | 3 秒 |
| B2 (criteria) | N(2, 0.2²) | 2 秒 |

---

## 实验模式

### 模式 1: 基线（无重部署）

使用固定的实例分配，不进行动态迁移。

```bash
# 启动服务
./start_all_services_no_redeploy.sh 24 24

# 运行实验
uv run python test_dynamic_workflow.py \
    --num-workflows 500 \
    --qps 0.5 \
    --strategies probabilistic
```

### 模式 2: Oracle 预测式重部署

利用预先生成的工作流数据，在负载变化**之前**触发迁移。

```bash
# 启动服务（AUTO_OPTIMIZE_ENABLED=False）
./start_all_services.sh 24 24

# 运行实验（两阶段模式默认开启）
uv run python test_dynamic_workflow.py \
    --num-workflows 500 \
    --strategies probabilistic \
    --oracle-deploy
```

### 两阶段 Fanout 模式说明（默认开启）

测试 fanout 变化时的动态迁移效果：
- **Phase 1** (前20%): 小 fanout (mean=6)，实例分配 A=24, B=24，QPS=0.50 wf/s
- **Phase 2** (后80%): 大 fanout (mean=14)，实例分配 A=14, B=34，QPS=0.29 wf/s

迁移触发：在第5个大 fanout 任务提交后触发一次迁移。

如需禁用两阶段模式，使用 `--no-two-phase`：

```bash
uv run python test_dynamic_workflow.py \
    --num-workflows 500 \
    --qps 0.5 \
    --strategies probabilistic \
    --no-two-phase
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--num-workflows` | 100 | 工作流数量 |
| `--qps` | 8.0 | A 任务提交速率 (两阶段模式下被忽略，使用硬编码值) |
| `--strategies` | all | 调度策略: `probabilistic`, `random`, `round_robin`, `min_time`, `po2` |
| `--warmup` | 0.0 | 预热比例 (0.0-1.0) |
| `--metric-portion` | 0.5 | 统计比例，只统计前 50% 工作流 |
| `--timeout` | 20 | 超时时间（分钟） |
| `--oracle-deploy` | False | 启用 Oracle 预测式重部署 |
| `--two-phase` | **True** | 两阶段 fanout 模式（默认开启） |
| `--no-two-phase` | - | 禁用两阶段模式，使用均匀 fanout 分布 |
| `--continuous` | False | 连续提交模式 |
| `--gqps` | None | 全局 QPS 限制 |

---

## 服务端口配置

| 服务 | 端口 | 说明 |
|-----|------|------|
| Predictor | 8099 | 预测服务 |
| Scheduler A | 8100 | 模型 A 调度器 |
| Scheduler B | 8200 | 模型 B 调度器 |
| Planner | 8202 | 迁移规划服务 |
| Instance Group A | 8210-82xx | 模型 A 实例 |
| Instance Group B | 8300-83xx | 模型 B 实例 |

---

## 结果文件

实验结果保存在 `results/` 目录：

```
results/
├── results_workflow_b1b2_<timestamp>.json    # 详细结果
└── raw_results_*/                            # 原始数据
    ├── a1_task_results.jsonl
    ├── b1_task_results.jsonl
    ├── b2_task_results.jsonl
    └── merge_task_result.jsonl
```

### 结果示例

```
================================================================================
Results Summary: probabilistic
================================================================================

Workflows:
  Completed:  500
  Avg time:   65.23s
  Median:     63.87s
  P95:        89.45s
  P99:        102.23s

Strategy Comparison
================================================================================
Strategy        WF Avg (s)   WF P95 (s)   Completed
--------------------------------------------------------------------------------
probabilistic   65.23        89.45        500
================================================================================
```

---

## 架构

### 线程模型

```
Thread 1: A Task Submitter (Poisson)
    │
    └──> Scheduler A ──> Instance Group A
              │
              ▼ (WebSocket)
Thread 2: A Result Receiver + B1 Submitter
    │
    └──> Scheduler B ──> Instance Group B
              │
              ▼ (WebSocket)
Thread 3: B1 Result Receiver + B2 Submitter
    │
    └──> Scheduler B ──> Instance Group B
              │
              ▼ (WebSocket)
Thread 4: B2 Result Receiver
    │
    └──> merge_ready_queue
              │
              ▼
Thread 5: Merge Task Submitter
    │
    └──> Scheduler A ──> Instance Group A
              │
              ▼ (WebSocket)
Thread 6: Merge Task Receiver
    │
    └──> completion_queue
              │
              ▼
Thread 7: Workflow Monitor
    │
    └──> Statistics

Thread 8: Oracle Deployer (可选)
    │
    └──> Planner ──> 触发迁移
```

### Oracle 预测式重部署原理

1. **预计算迁移计划**: 基于预生成的工作流数据，计算每个时间窗口的最优实例分配
2. **提前触发迁移**: 在负载变化**之前** 3 秒触发迁移
3. **实时状态同步**: 每次迁移前从 Scheduler 获取真实实例状态

```python
# 两阶段模式硬编码值
Phase 1 (fanout=6):  A=24, B=24, QPS=0.50 wf/s
Phase 2 (fanout=14): A=14, B=34, QPS=0.29 wf/s
```

---

## 故障排除

### 服务无法启动

```bash
# 检查端口占用
lsof -i :8100
lsof -i :8200

# 强制停止所有服务
./stop_all_services.sh
pkill -f "python.*src.cli start"
```

### 工作流未完成

```bash
# 检查服务健康状态
curl http://localhost:8100/health
curl http://localhost:8200/health
curl http://localhost:8202/health

# 查看日志
tail -f logs/scheduler-a.log
tail -f logs/scheduler-b.log
```

### 超时问题

```bash
# 增加超时时间
uv run python test_dynamic_workflow.py --timeout 30

# 降低 QPS
uv run python test_dynamic_workflow.py --qps 0.3
```

---

## 文件结构

```
07.Exp2.Deep_Research_Migration_Test/
├── test_dynamic_workflow.py      # 主实验脚本
├── workload_generator.py         # 工作负载生成器
├── redeply.py                    # 重部署配置
├── start_all_services.sh         # 带重部署的服务启动
├── start_all_services_no_redeploy.sh  # 无重部署的服务启动
├── stop_all_services.sh          # 服务停止脚本
├── data/                         # 追踪数据模板
│   ├── dr_boot.json
│   ├── dr_summary_dict.json
│   ├── dr_query.json
│   └── dr_criteria.json
├── results/                      # 实验结果
├── logs/                         # 服务日志
└── README.md                     # 本文件
```

---

## 常用命令速查

```bash
# 基线实验（两阶段模式默认开启，QPS自动硬编码）
./start_all_services_no_redeploy.sh 24 24
uv run python test_dynamic_workflow.py --num-workflows 500 --strategies probabilistic
./stop_all_services.sh

# Oracle 预测式重部署（两阶段模式默认开启）
./start_all_services.sh 24 24
uv run python test_dynamic_workflow.py --num-workflows 500 --oracle-deploy --strategies probabilistic
./stop_all_services.sh

# 快速测试（10个工作流，两阶段模式）
uv run python test_dynamic_workflow.py --num-workflows 10 --strategies probabilistic

# 禁用两阶段模式（使用自定义QPS）
uv run python test_dynamic_workflow.py --num-workflows 100 --qps 1.0 --no-two-phase --strategies probabilistic

# 查看帮助
uv run python test_dynamic_workflow.py --help
```
