# Type2: Deep Research Workflow Benchmark

Deep Research 工作流基准测试，支持模拟模式和真实集群模式。

## 工作流模式

```
A (LLM Small) → n×B1 (LLM Large) → n×B2 (LLM Large) → Merge (LLM Small)
      │               │                   │                │
      │               │                   │                └── 聚合所有结果
      │               │                   └── n个并行深度分析
      │               └── n个并行研究任务（扇出，fanout）
      └── 初始查询/规划
```

## 快速开始

### 1. 进入工作目录

```bash
cd experiments/13.workflow_benchmark
```

### 2. 模拟模式（推荐先测试）

模拟模式使用 sleep_model 模拟真实模型的处理时间，无需真实模型服务。

```bash
# 最简单的运行方式（使用默认参数）
python tools/cli.py run-deep-research-sim --num-workflows 10

# 指定 QPS 和策略
python tools/cli.py run-deep-research-sim \
    --num-workflows 50 \
    --qps 2.0 \
    --strategies probabilistic,min_time

# ============================================================================
# 完整参数示例（带详细注释）
# ============================================================================
python tools/cli.py run-deep-research-sim \
    # -------------------------------------------------------------------------
    # 必需参数（实际上都有默认值，但通常需要根据实验调整）
    # -------------------------------------------------------------------------
    --num-workflows 100 \       # [可选] 工作流数量 (默认: 10)

    # -------------------------------------------------------------------------
    # 通用可选参数
    # -------------------------------------------------------------------------
    --qps 2.0 \                 # [可选] 每秒提交任务数 (默认: 2.0)
    --duration 300 \            # [可选] 最大实验时长，单位秒 (默认: 120)
    --strategies all \          # [可选] 调度策略，可选: default/all/具体策略名
                                #        default = probabilistic,min_time,round_robin,random,po2
                                #        all = probabilistic,min_time,round_robin,random,po2,serverless
                                #        (默认: default)
    --warmup 0.2 \              # [可选] 预热比例，0.0-1.0 (默认: 0.2)
                                #        实际预热数 = num_workflows * warmup
    --portion-stats 1.0 \       # [可选] 统计包含的非预热工作流比例 (默认: 1.0)
    --seed 42 \                 # [可选] 随机种子 (默认: 42)

    # -------------------------------------------------------------------------
    # Type2 Deep Research 特定参数
    # -------------------------------------------------------------------------
    --fanout 4 \                # [可选] 默认扇出数量，即B1/B2并行任务数 (默认: 4)
                                #        如果指定 --fanout-config 则忽略此参数
    --fanout-config type2_deep_research/configs/fanout_uniform.json \
                                # [可选] 扇出分布配置文件路径 (默认: None)
                                #        支持: static, uniform, two_peak, four_peak
    --fanout-seed 123 \         # [可选] 扇出分布采样的随机种子 (默认: None)
    --max-sleep-time 60.0       # [可选] 模拟模式最大睡眠时间，单位秒 (默认: 600.0)
                                #        睡眠时间在 [1, max_sleep_time] 范围内均匀分布
```

**注意**: 由于 bash 不支持行内注释，实际运行时请使用以下命令：

```bash
# 完整参数示例（可直接运行）
python tools/cli.py run-deep-research-sim \
    --num-workflows 100 \
    --qps 2.0 \
    --duration 300 \
    --strategies all \
    --warmup 0.2 \
    --portion-stats 1.0 \
    --seed 42 \
    --fanout 4 \
    --fanout-config type2_deep_research/configs/fanout_uniform.json \
    --fanout-seed 123 \
    --max-sleep-time 60.0
```

### 3. 真实集群模式

真实模式连接远程调度器，使用真实的 LLM 模型。

```bash
# 基本运行
python tools/cli.py run-deep-research-real --num-workflows 20

# 指定策略和参数
python tools/cli.py run-deep-research-real \
    --num-workflows 50 \
    --qps 1.0 \
    --strategies probabilistic \
    --duration 600

# 完整参数示例
python tools/cli.py run-deep-research-real \
    --num-workflows 100 \
    --qps 1.5 \
    --duration 600 \
    --strategies probabilistic,min_time \
    --warmup 0.2 \
    --portion-stats 0.8 \
    --seed 42 \
    --fanout 4 \
    --fanout-config type2_deep_research/configs/fanout_four_peak.json \
    --fanout-seed 456
```

## 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-workflows` | int | 10 | 工作流数量 |
| `--qps` | float | 2.0 | 每秒提交任务数 |
| `--duration` | int | 120 | 最大实验时长（秒） |
| `--strategies` | str | default | 调度策略（见下方） |
| `--warmup` | float | 0.2 | 预热比例（0.0-1.0） |
| `--portion-stats` | float | 1.0 | 统计包含的非预热工作流比例 |
| `--seed` | int | 42 | 随机种子 |

### Deep Research 特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fanout` | int | 4 | 扇出数量（B1/B2并行任务数） |
| `--fanout-config` | str | None | 扇出分布配置文件路径 |
| `--fanout-seed` | int | None | 扇出分布采样的随机种子 |
| `--max-sleep-time` | float | 600.0 | 模拟模式最大睡眠时间（秒） |

### 可用策略

```bash
# 使用默认策略集
--strategies default    # = probabilistic,min_time,round_robin,random,po2

# 使用所有策略
--strategies all        # = probabilistic,min_time,round_robin,random,po2,serverless

# 指定特定策略
--strategies probabilistic,min_time
```

## 使用示例

### 示例 1: 快速验证

```bash
# 10个工作流，单策略，60秒超时
python tools/cli.py run-deep-research-sim \
    --num-workflows 10 \
    --qps 1.0 \
    --duration 60 \
    --strategies probabilistic
```

### 示例 2: 策略对比实验

```bash
# 100个工作流，所有策略，5分钟
python tools/cli.py run-deep-research-sim \
    --num-workflows 100 \
    --qps 2.0 \
    --duration 300 \
    --strategies all \
    --warmup 0.2
```

### 示例 3: 使用扇出分布配置

```bash
# 使用四峰分布的扇出数
python tools/cli.py run-deep-research-sim \
    --num-workflows 50 \
    --fanout-config type2_deep_research/configs/fanout_four_peak.json \
    --fanout-seed 42
```

### 示例 4: 生产环境测试

```bash
# 真实集群，大规模测试
python tools/cli.py run-deep-research-real \
    --num-workflows 200 \
    --qps 1.5 \
    --duration 1800 \
    --strategies probabilistic,min_time \
    --warmup 0.1 \
    --portion-stats 0.8
```

## 输出说明

实验完成后，指标文件保存在 `output/` 目录：

```
output/
├── metrics_probabilistic.json   # probabilistic 策略的指标
├── metrics_min_time.json        # min_time 策略的指标
└── metrics_round_robin.json     # round_robin 策略的指标
```

运行结束时会输出策略对比表：

```
================================================================================
Strategy Comparison Results
================================================================================
Strategy          | A Avg  | A P90  | B1 Avg | B1 P90 | B2 Avg | B2 P90 | Merge  | WF Avg
------------------|--------|--------|--------|--------|--------|--------|--------|-------
probabilistic     | 1.23s  | 2.45s  | 3.18s  | 5.31s  | 3.12s  | 5.15s  | 1.05s  | 12.5s
min_time          | 1.45s  | 2.89s  | 3.32s  | 5.56s  | 3.28s  | 5.42s  | 1.12s  | 13.2s
================================================================================
```

## 前置要求

### 模拟模式

1. 启动本地调度器服务（端口 8100, 8200）
2. 注册 sleep_model_a（用于 A 和 Merge 任务）和 sleep_model_b（用于 B1/B2 任务）

### 真实模式

1. 远程调度器可访问
2. 模型已注册：
   - `llm_service_small_model`：用于 A 和 Merge 任务
   - `llm_service_large_model`：用于 B1/B2 任务

## 可用的分布配置文件

```
type2_deep_research/configs/
├── fanout_static.json       # 静态值（使用 --fanout 参数值）
├── fanout_uniform.json      # 均匀分布
├── fanout_two_peak.json     # 双峰分布
└── fanout_four_peak.json    # 四峰分布
```

## 目录结构

```
type2_deep_research/
├── simulation/
│   └── test_workflow_sim.py    # 模拟模式入口
├── real/
│   └── test_workflow_real.py   # 真实模式入口
├── configs/
│   ├── fanout_static.json
│   ├── fanout_uniform.json
│   ├── fanout_two_peak.json
│   └── fanout_four_peak.json
├── config.py                   # 配置类
├── submitters.py               # 任务提交器（A, B1, B2, Merge）
├── receivers.py                # 结果接收器
├── workflow_data.py            # 工作流数据结构
└── README.md                   # 本文件
```

## 与 Type1 的主要区别

| 特性 | Type1 (Text2Video) | Type2 (Deep Research) |
|------|-------------------|----------------------|
| 工作流模式 | A1→A2→B×N | A→n×B1→n×B2→Merge |
| 并行性 | B 任务串行循环 | B1/B2 任务并行执行 |
| 模型 | LLM + T2VID | LLM Small + LLM Large |
| 特有参数 | max_b_loops, frame_count | fanout |
| 同步点 | 无（流水线） | Merge（等待所有 B2 完成） |

## 常见问题

### Q: fanout 参数是什么意思？

`fanout` 定义了每个工作流中 B1 和 B2 任务的并行数量。例如 `fanout=4` 表示：
- A 任务完成后，触发 4 个并行的 B1 任务
- 每个 B1 完成后，触发对应的 B2 任务
- 所有 4 个 B2 完成后，触发 Merge 任务

### Q: 模拟模式和真实模式有什么区别？

模拟模式使用 `sleep_model` 模拟处理时间，真实模式调用实际的 LLM 服务。两者使用相同的工作流架构，只是请求体不同：

| 模式 | task_input 格式 |
|-----|----------------|
| 模拟 | `{"sleep_time": 1.23}` |
| 真实 | `{"query": "...", "max_tokens": 512}` |

### Q: warmup 和 portion_stats 有什么作用？

- `warmup=0.2`: 前 20% 的工作流用于预热，不计入统计
- `portion_stats=0.8`: 非预热工作流中，只统计前 80%

这样可以排除系统冷启动和尾部延迟的影响。
