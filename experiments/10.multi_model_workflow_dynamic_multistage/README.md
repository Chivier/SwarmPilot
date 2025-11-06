# Experiment 10: Multi-Stage Workflow with Intelligent Redeployment

## 项目概述

本实验基于实验 04 开发，实现了三阶段固定 fanout (8, 15, 1) 的工作流，并在阶段间进行智能的 instance 重新部署以优化性能。

实验对比了三种调度策略（min_time, round_robin, probabilistic）在有/无动态重新部署两种场景下的性能差异。

## 核心特性

### 三阶段工作流
- **阶段 1**: A→8B workflows（B 密集型工作负载）
- **阶段 2**: A→15B workflows（非常 B 密集型）
- **阶段 3**: A→1B workflows（平衡型工作负载）

### 智能重新部署
- 在阶段边界计算下一阶段的最优 instance 配比
- 只重新部署需要变更的 instances（最小化变更）
- 基于任务时间分布和 fanout 进行优化

## 项目结构

```
experiments/10.multi_model_workflow_dynamic_multistage/
├── workload_generator.py               # ✅ 工作负载生成器（支持多阶段）
├── optimal_ratio_calculator.py         # ✅ 最优配比计算器
├── redeployment_manager.py            # 🔄 重新部署管理器（待实现）
├── test_multistage_workflow.py        # 🔄 多阶段测试脚本（待实现）
├── start_all_services.sh              # 🔄 智能服务启动脚本（待实现）
├── stop_all_services.sh               # 服务停止脚本
├── requirements.txt                   # Python 依赖
├── results/                           # 实验结果目录
├── logs/                              # 日志目录
└── README.md                          # 本文档
```

## 已完成的组件

### 1. 多阶段工作负载生成器

`workload_generator.py` 现在支持三个固定 fanout 的阶段：

```python
from workload_generator import generate_multistage_fanout_distribution

stages = generate_multistage_fanout_distribution(
    num_workflows_per_stage=[100, 100, 100],
    fanouts=[8, 15, 1],
    seed=42
)

# 输出：
# Stage 1: 100 workflows, fanout=8 (800 total B tasks)
# Stage 2: 100 workflows, fanout=15 (1500 total B tasks)
# Stage 3: 100 workflows, fanout=1 (100 total B tasks)
```

**测试命令**:
```bash
.venv/bin/python3 workload_generator.py --test-multistage
```

### 2. 最优配比计算器

`optimal_ratio_calculator.py` 根据 fanout 和任务统计信息计算最优的 A/B instance 分配：

```python
from optimal_ratio_calculator import OptimalRatioCalculator, SchedulingStrategy

calc = OptimalRatioCalculator(safety_factor=1.2)

# 计算阶段 1 (fanout=8) 的最优配比
num_a, num_b = calc.calculate_optimal_ratio(
    fanout=8,
    task_stats={'avg_a_time': 5.0, 'avg_b_time': 5.0},
    total_instances=30,
    strategy=SchedulingStrategy.MIN_TIME
)

# 结果: num_a=2, num_b=28 (1:14 ratio)
```

**关键功能**:
- 支持三种调度策略的不同效率特性
- 包含安全因子防止队列积压
- 提供详细的计算过程说明

**测试命令**:
```bash
.venv/bin/python3 optimal_ratio_calculator.py
```

### 计算示例

| 阶段 | Fanout | 总 Instances | 最优配比 (min_time) |
|------|--------|--------------|---------------------|
| 阶段 1 | 8 | 30 | 2A + 28B (1:14) |
| 阶段 2 | 15 | 30 | 1A + 29B (1:29) |
| 阶段 3 | 1 | 30 | 13A + 17B (1:1.3) |

## 待实现的组件

根据 TaskMaster 任务列表，还需要完成以下组件：

### 3. 重新部署管理器 (Task 4)

创建 `redeployment_manager.py` 来编排智能的 instance 重新部署：
- 查询当前 instance 分配情况
- 生成最小变更的重新部署计划
- 执行重新部署并监控状态
- 使用 instance `/model/restart` API

### 4. 多阶段测试编排器 (Task 5)

扩展 `test_dynamic_workflow.py` 为 `test_multistage_workflow.py`：
- 三阶段顺序执行
- 阶段间等待完成
- 可选的重新部署功能
- 指标收集和保存

### 5. 智能服务启动脚本 (Task 6)

修改 `start_all_services.sh`：
- 接受总 instance 数量参数
- 自动计算阶段 1 的初始配比
- 启动 Scheduler A, B 和配置好的 instances

### 6. 单元测试 (Task 7)

编写全面的单元测试：
- `test_redeployment.py`: 重新部署管理器测试
- `test_ratio_calculation.py`: 配比计算器测试
- `test_stage_transition.py`: 阶段转换测试

### 7. 文档套件 (Task 8)

完善文档：
- README.md (架构、使用说明)
- QUICK_REFERENCE.md (快速命令参考)
- REDEPLOYMENT_STRATEGY.md (重新部署策略详解)
- OPTIMAL_RATIO.md (配比计算方法)

### 8. 完整实验和分析 (Task 9)

运行全部 6 个实验并分析结果：
- 实验 1-3: 无重新部署（baseline）
- 实验 4-6: 有重新部署
- 生成性能对比报告和可视化

## 快速开始

### 环境设置

```bash
# 创建虚拟环境并安装依赖
uv venv
uv pip install -r requirements.txt

# 或者使用 pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 测试已完成的组件

```bash
# 测试多阶段工作负载生成器
.venv/bin/python3 workload_generator.py --test-multistage

# 测试最优配比计算器
.venv/bin/python3 optimal_ratio_calculator.py
```

## TaskMaster 集成

本项目使用 TaskMaster 进行任务追踪。查看任务状态：

```bash
# 查看所有任务
task-master list

# 查看下一个任务
task-master next

# 查看特定任务详情
task-master show <task-id>

# 标记任务完成
task-master set-status --id=<task-id> --status=done
```

当前任务完成情况：
- ✅ Task 1: 初始化项目结构
- ✅ Task 2: 实现多阶段工作负载生成器
- ✅ Task 3: 开发最优配比计算器
- 🔄 Task 4-9: 待完成

## 实验设计

### 六个实验场景

**实验组 A (静态配置 - Baseline)**:
1. min_time 策略，固定配置
2. round_robin 策略，固定配置
3. probabilistic 策略，固定配置

**实验组 B (动态重新部署)**:
4. min_time 策略，阶段间重新部署
5. round_robin 策略，阶段间重新部署
6. probabilistic 策略，阶段间重新部署

### 优化目标

最小化端到端 workflow 延迟，同时保持重新部署开销 < 5%。

### 预期结果

- 阶段 1→2 转换: 小幅改善（1 instance 变更）
- 阶段 2→3 转换: 显著改善（12+ instances 变更）
- 总体性能提升: 10-20%

## 开发指南

### 继续开发

要继续实现剩余的任务，请按以下顺序进行：

1. **实现重新部署管理器** (Task 4)
   - 参考 TaskMaster 生成的详细代码
   - 测试 API 调用和状态轮询

2. **实现多阶段测试编排器** (Task 5)
   - 集成工作负载生成器和配比计算器
   - 实现阶段转换逻辑

3. **创建启动脚本** (Task 6)
   - 使用配比计算器自动计算初始配置
   - 添加健康检查

4. **编写测试** (Task 7)
   - 使用 pytest 和 unittest.mock
   - 确保高覆盖率

5. **完善文档** (Task 8)
   - 详细的使用说明
   - API 文档和示例

6. **运行实验** (Task 9)
   - 自动化实验执行
   - 结果分析和可视化

### 代码风格

- 遵循 PEP 8
- 使用类型提示（Type Hints）
- 编写文档字符串（Docstrings）
- 添加适当的日志记录

## 参考资料

- [Experiment 04 README](../04.multi_model_workflow_dynamic/README.md)
- [Instance API Reference](../../instance/docs/5.API_REFERENCE.md)
- [Instance Restart API Documentation](../../instance/RESTART_API_IMPLEMENTATION.md)
- [Scheduler WebSocket API](../../scheduler/docs/7.WEBSOCKET_API.md)

## 许可证

与主项目相同。

## 贡献者

- TaskMaster AI: PRD 解析和任务生成
- Claude Code: 核心组件实现
