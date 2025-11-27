# Type4 OCR+LLM Workflow

OCR 文字识别 + LLM 文本处理的两阶段工作流。

```
[图片] → [A: OCR] → [提取文本] → [B: LLM] → [处理结果]
```

---

## 🚀 五分钟快速上手

### 模拟模式（推荐新手）

**只需 3 条命令即可运行实验：**

```bash
# 1. 进入实验目录
cd experiments/13.workflow_benchmark

# 2. 启动模拟集群（约 30 秒）
./scripts/start_all_services.sh

# 3. 运行实验
python tools/cli.py run-ocr-llm-sim --num-workflows 50 --qps 2.0 --strategies all
```

**实验结束后停止服务：**
```bash
./scripts/stop_all_services.sh
```

**查看结果：**
```bash
cat output/metrics_probabilistic.json | python -m json.tool
```

---

## 📋 详细使用说明

### 模拟模式

模拟模式使用 `sleep_model` 模拟 OCR 和 LLM 处理时间，无需真实模型。

#### 启动集群

```bash
# 默认配置：4 个 OCR 实例 + 2 个 LLM 实例
./scripts/start_all_services.sh

# 自定义实例数量
N1=8 N2=4 ./scripts/start_all_services.sh
```

#### 运行实验

```bash
# 基础用法
python tools/cli.py run-ocr-llm-sim \
    --num-workflows 50 \
    --qps 2.0 \
    --strategies probabilistic

# 测试所有调度策略
python tools/cli.py run-ocr-llm-sim \
    --num-workflows 100 \
    --qps 3.0 \
    --strategies all

# 自定义睡眠时间分布（正态分布）
python tools/cli.py run-ocr-llm-sim \
    --num-workflows 100 \
    --qps 2.0 \
    --strategies probabilistic \
    --sleep-time-a-config '{"type": "normal", "mean": 0.8, "std": 0.2, "min": 0.1}' \
    --sleep-time-b-config '{"type": "normal", "mean": 5.0, "std": 0.5, "min": 0.5}'
```

#### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-workflows` | 10 | 工作流数量 |
| `--qps` | 2.0 | 每秒请求数 |
| `--strategies` | probabilistic | 调度策略 (`all` 测试所有) |
| `--warmup` | 0.2 | 预热比例 |
| `--portion-stats` | 1.0 | 统计采样比例 (0.0-1.0) |

#### 统计采样 (`--portion-stats`)

`--portion-stats` 用于控制统计结果中包含多少非预热工作流。适用于需要早停或仅统计部分结果的场景。

**计算公式：**
```
提交总数 = num_workflows × (1 + warmup)
统计样本数 = num_workflows × portion_stats
```

**示例：**
```bash
# 提交 120 个工作流（100 + 20 预热），仅统计前 50 个非预热结果
python tools/cli.py run-ocr-llm-sim \
    --num-workflows 100 \
    --warmup 0.2 \
    --portion-stats 0.5 \
    --strategies probabilistic
```

| num_workflows | warmup | portion_stats | 提交总数 | 统计样本 |
|---------------|--------|---------------|----------|----------|
| 100 | 0.2 | 1.0 | 120 | 100 |
| 100 | 0.2 | 0.5 | 120 | 50 |
| 100 | 0.0 | 0.8 | 100 | 80 |

---

### 真实模式

真实模式使用实际的 OCR（EasyOCR）和 LLM 模型。

#### 0. 准备测试数据集

真实模式需要图片数据。推荐使用 OmniDocBench 数据集：

```bash
# 下载 OmniDocBench 数据集（约 1.25GB）
python type4_ocr_llm/scripts/prepare_dataset.py

# 数据将保存到 type4_ocr_llm/data/omnidocbench/images/
```

**脚本参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `data/omnidocbench` | 输出目录 |
| `--cache-dir` | HF 默认缓存 | HuggingFace 缓存目录 |

**数据集信息：**

| 属性 | 值 |
|------|------|
| 来源 | [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) |
| 格式 | PNG 图片（文档页面） |
| 数量 | ~1,355 张图片 |
| 大小 | ~1.25GB |
| 内容 | 学术论文、财务报告、报纸、教科书、手写笔记等 9 种文档类型 |

**自定义数据集：**

如需使用自己的图片，只需将图片放入目录即可：
```bash
# 支持的格式：jpg, jpeg, png, gif, bmp, webp
mkdir -p my_images
cp /path/to/your/images/*.png my_images/

# 使用自定义目录运行
python tools/cli.py run-ocr-llm-real --image-dir my_images ...
```

#### 1. 构建并启动服务

```bash
# 构建 OCR Docker 镜像
cd instance/dockers/ocr_model && docker build -t ocr_model:latest .

# 启动服务（16 OCR + 8 LLM 实例）
cd experiments/13.workflow_benchmark/type4_ocr_llm/scripts
./start_real_ocr_service.sh
```

#### 2. 运行实验

```bash
# 使用下载的 OmniDocBench 数据集
python tools/cli.py run-ocr-llm-real \
    --num-workflows 50 \
    --qps 2.0 \
    --image-dir type4_ocr_llm/data/omnidocbench/images \
    --strategies probabilistic

# 或使用自定义图片目录
python tools/cli.py run-ocr-llm-real \
    --num-workflows 50 \
    --qps 2.0 \
    --image-dir ./your_images \
    --strategies probabilistic
```

#### 3. 停止服务

```bash
./stop_real_ocr_service.sh
```

#### 真实模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image-dir` | - | 图片目录 (jpg, png) |
| `--image-json` | - | Base64 编码的图片 JSON |
| `--ocr-languages` | en | OCR 语言 |
| `--max-tokens` | 512 | LLM 最大 token 数 |
| `--portion-stats` | 1.0 | 统计采样比例 (见模拟模式说明) |

---

## 📊 调度策略

| 策略 | 说明 |
|------|------|
| `probabilistic` | 基于运行时估计的概率调度（推荐） |
| `min_time` | 最小化总执行时间 |
| `round_robin` | 轮询调度 |
| `random` | 随机选择 |
| `po2` | Power of 2 choices |

使用 `--strategies all` 自动测试所有策略。

---

## 📁 输出结果

结果保存在 `output/` 目录：

```
output/
├── metrics_probabilistic.json
├── metrics_min_time.json
├── metrics_round_robin.json
├── metrics_random.json
└── metrics_po2.json
```

每个文件包含：
- 工作流完成时间
- 各阶段延迟 (A/B)
- 吞吐量统计
- 百分位分布

---

## 🔧 高级配置

### 睡眠时间分布

模拟模式支持多种分布类型来模拟任务执行时间：

| 类型 | 说明 | 示例 |
|------|------|------|
| `normal` | 正态分布（推荐） | `{"type": "normal", "mean": 0.8, "std": 0.2, "min": 0.1}` |
| `uniform` | 均匀分布 | `{"type": "uniform", "min": 1, "max": 5}` |
| `static` | 固定值 | `{"type": "static", "value": 2.0}` |

**参数说明：**
- `mean`: 正态分布均值（秒）
- `std`: 标准差
- `min`/`max`: 值域裁剪（防止负值或极端值）

### 服务端口

| 服务 | 端口 |
|------|------|
| Predictor | 8101 |
| Planner | 8202 |
| Scheduler A (OCR) | 8100 |
| Scheduler B (LLM) | 8200 |
| OCR 实例 | 8210+ |
| LLM 实例 | 8300+ |

### 真实模式部署

| 模型 | 实例数 | 端口 | 资源 |
|------|--------|------|------|
| OCR (EasyOCR) | 16 | 9000-9015 | CPU |
| LLM | 8 | 9100-9107 | GPU |

---

## 📂 目录结构

```
type4_ocr_llm/
├── config.py              # 配置
├── workflow_data.py       # 数据处理（图片加载、base64 编码）
├── submitters.py          # 任务提交
├── receivers.py           # 结果接收
├── simulation/            # 模拟模式
├── real/                  # 真实模式
├── data/                  # 数据目录
│   └── omnidocbench/      # OmniDocBench 数据集（需下载）
│       └── images/        # 图片文件
└── scripts/
    ├── prepare_dataset.py        # 数据集下载脚本
    ├── start_real_ocr_service.sh # 启动真实 OCR 服务
    └── stop_real_ocr_service.sh  # 停止真实 OCR 服务
```

---

## ❓ 常见问题

**Q: 集群启动失败？**
```bash
# 检查端口占用
lsof -i :8100
# 清理残留进程
./scripts/stop_all_services.sh
```

**Q: 如何查看日志？**
```bash
ls experiments/13.workflow_benchmark/logs/
tail -f logs/scheduler-a.log
```

**Q: 如何修改实例数量？**
```bash
N1=16 N2=8 ./scripts/start_all_services.sh
```

**Q: 数据集下载失败或中断？**
```bash
# 重新运行脚本会自动跳过已下载的文件
python type4_ocr_llm/scripts/prepare_dataset.py

# 如需完全重新下载，删除缓存目录后重试
rm -rf type4_ocr_llm/data/omnidocbench
python type4_ocr_llm/scripts/prepare_dataset.py
```

**Q: 如何使用自己的图片？**
```bash
# 将图片放入目录（支持 jpg, png, gif, bmp, webp）
python tools/cli.py run-ocr-llm-real \
    --image-dir /path/to/your/images \
    --num-workflows 50 --qps 2.0
```

**Q: 磁盘空间不足？**
```bash
# OmniDocBench 需要约 2GB 磁盘空间
# 检查可用空间
df -h .

# 可指定其他目录
python type4_ocr_llm/scripts/prepare_dataset.py --output-dir /other/path/data
```
