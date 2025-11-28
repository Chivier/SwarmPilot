# Type1: Text2Video Workflow Benchmark

Text2Video 工作流基准测试，支持模拟模式和真实集群模式。

## 工作流模式

```
A1 (LLM) → A2 (LLM) → B (T2VID) × N loops
   │           │           │
   │           │           └── 视频生成（1-4次迭代）
   │           └── 生成 negative prompt
   └── 生成 positive prompt
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
python tools/cli.py run-text2video-sim --num-workflows 10

# 指定 QPS 和策略
python tools/cli.py run-text2video-sim \
    --num-workflows 50 \
    --qps 2.0 \
    --strategies probabilistic,min_time

# 完整参数示例
python tools/cli.py run-text2video-sim \
    --num-workflows 100 \
    --qps 2.0 \
    --duration 300 \
    --strategies all \
    --warmup 0.2 \
    --max-b-loops 4 \
    --frame-count 16 \
    --max-sleep-time 60.0
```

### 3. 真实集群模式

真实模式连接远程调度器，使用真实的 LLM 和 T2VID 模型。

```bash
# 基本运行
python tools/cli.py run-text2video-real --num-workflows 20

# 指定策略和参数
python tools/cli.py run-text2video-real \
    --num-workflows 50 \
    --qps 1.0 \
    --strategies probabilistic \
    --duration 600
```

## 参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--num-workflows` | 10 | 工作流数量 |
| `--qps` | 2.0 | 每秒提交任务数 |
| `--duration` | 120 | 最大实验时长（秒） |
| `--strategies` | default | 调度策略（见下方） |
| `--warmup` | 0.2 | 预热比例（0.0-1.0） |
| `--portion-stats` | 1.0 | 统计包含的非预热工作流比例 |
| `--seed` | 42 | 随机种子 |

### Text2Video 特定参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--max-b-loops` | 3 | B任务最大迭代次数 |
| `--frame-count` | 16 | 视频帧数 |
| `--max-sleep-time` | 600.0 | 模拟模式最大睡眠时间（秒） |
| `--frame-count-config` | - | 帧数分布配置文件路径 |
| `--max-b-loops-config` | - | B循环分布配置文件路径 |

### 可用策略

```bash
# 使用默认策略集
--strategies default    # = probabilistic,min_time,round_robin

# 使用所有策略
--strategies all        # = probabilistic,min_time,round_robin,random,po2

# 指定特定策略
--strategies probabilistic,min_time
```

## 使用示例

### 示例 1: 快速验证

```bash
# 10个工作流，单策略，60秒超时
python tools/cli.py run-text2video-sim \
    --num-workflows 10 \
    --qps 1.0 \
    --duration 60 \
    --strategies probabilistic
```

### 示例 2: 策略对比实验

```bash
# 100个工作流，所有策略，5分钟
python tools/cli.py run-text2video-sim \
    --num-workflows 100 \
    --qps 2.0 \
    --duration 300 \
    --strategies all \
    --warmup 0.2
```

### 示例 3: 使用分布配置

```bash
# 使用四峰分布的帧数和均匀分布的循环次数
python tools/cli.py run-text2video-sim \
    --num-workflows 50 \
    --frame-count-config type1_text2video/configs/frame_count_four_peak.json \
    --max-b-loops-config type1_text2video/configs/max_b_loops_uniform.json
```

### 示例 4: 生产环境测试

```bash
# 真实集群，大规模测试
python tools/cli.py run-text2video-real \
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
Strategy          | A1 Avg | A1 P90 | A2 Avg | A2 P90 | B Avg  | B P90  | WF Avg
------------------|--------|--------|--------|--------|--------|--------|-------
probabilistic     | 1.23s  | 2.45s  | 1.18s  | 2.31s  | 5.67s  | 8.92s  | 15.2s
min_time          | 1.45s  | 2.89s  | 1.32s  | 2.56s  | 6.12s  | 9.45s  | 16.8s
================================================================================
```

## 前置要求

### 模拟模式

1. 启动本地调度器服务（端口 8100, 8200）
2. 注册 sleep_model_a 和 sleep_model_b

### 真实模式

1. 远程调度器可访问
2. 模型已注册：`llm_service_small_model`, `t2vid`
3. `captions_10k.json` 数据集存在

## 目录结构

```
type1_text2video/
├── simulation/
│   └── test_workflow_sim.py    # 模拟模式入口
├── real/
│   └── test_workflow_real.py   # 真实模式入口
├── configs/
│   ├── frame_count_four_peak.json
│   └── max_b_loops_uniform.json
├── data/
│   ├── training_config.json    # 基准数据
│   └── captions_10k.jsonl
├── config.py                   # 配置类
├── submitters.py               # 任务提交器
├── receivers.py                # 结果接收器
├── workflow_data.py            # 工作流数据结构
└── README.md                   # 本文件
```

## 常见问题

### Q: 模拟模式和真实模式有什么区别？

模拟模式使用 `sleep_model` 模拟处理时间，真实模式调用实际的 LLM 和视频生成服务。两者使用相同的工作流架构，只是请求体不同：

| 模式 | task_input 格式 |
|-----|----------------|
| 模拟 | `{"sleep_time": 1.23}` |
| 真实 | `{"sentence": "...", "max_tokens": 512}` |

### Q: 如何调整任务处理时间的分布？

使用 `--max-sleep-time` 参数缩放睡眠时间：

```bash
# 快速测试（最大10秒）
python tools/cli.py run-text2video-sim --max-sleep-time 10.0

# 接近真实（最大600秒）
python tools/cli.py run-text2video-sim --max-sleep-time 600.0
```

### Q: warmup 和 portion_stats 有什么作用？

- `warmup=0.2`: 前 20% 的工作流用于预热，不计入统计
- `portion_stats=0.8`: 非预热工作流中，只统计前 80%

这样可以排除系统冷启动和尾部延迟的影响。
