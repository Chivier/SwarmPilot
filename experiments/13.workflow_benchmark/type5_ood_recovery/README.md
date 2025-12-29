# Type 5: OOD Recovery Experiment

This experiment demonstrates the system's ability to recover from Out-of-Distribution (OOD) workloads by dynamically correcting runtime predictions.

---

## Experiment Results

All experiment results are organized in the `results/` directory. See [results/RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md) for detailed analysis.

### Quick Links

| Result | Improvement | Location |
|--------|-------------|----------|
| **Best Real-Time Throughput** | +18.2% | [results/best_config/realtime_diff/](results/best_config/realtime_diff/) |
| **High QPS Visual Difference** | +17.2% | [results/best_config/high_qps/](results/best_config/high_qps/) |
| **Full-Process Experiments** | v1-v5 | [results/fullprocess_experiments/](results/fullprocess_experiments/) |
| **Grid Search (375 configs)** | Various | [results/grid_search/](results/grid_search/) |

### Best Configuration (One-Click Reproduction)

```bash
# +18.2% Real-Time Throughput improvement
cd experiments/13.workflow_benchmark
uv run python type5_ood_recovery/standalone_sim.py \
    --num-instances 512 --num-tasks 10000 \
    --phase1-count 500 --phase1-qps 200.0 --phase23-qps 180.0 \
    --phase23-distribution weighted_bimodal --runtime-scale 0.05 \
    --phase23-bimodal-scale 3.0 --phase23-small-peak-ratio 0.1 \
    --phase2-transition-count 300 --seed 42 \
    --output-dir output_test/recovery

# Add --no-recovery for baseline comparison
```

---

## Quick Start (5 minutes)

### All-in-One Experiment Runner (Recommended)

Run both Recovery and Baseline simulations, then automatically perform SLO analysis:

```bash
cd experiments/13.workflow_benchmark

# Run with optimal configuration
python type5_ood_recovery/run_ood_experiment.py \
    --num-instances 512 \
    --num-tasks 10000 \
    --phase1-count 500 \
    --phase1-qps 200.0 \
    --phase23-qps 180.0 \
    --phase23-distribution weighted_bimodal \
    --runtime-scale 0.05 \
    --phase23-bimodal-scale 3.0 \
    --phase23-small-peak-ratio 0.1 \
    --phase2-transition-count 300 \
    --seed 42 \
    --output-dir output_experiment
```

This will:
1. Run **Recovery** simulation → `output_experiment/recovery/`
2. Run **Baseline** simulation → `output_experiment/baseline/`
3. Generate **SLO analysis** plots → `output_experiment/slo_analysis/`

#### Additional Options

```bash
# Skip SLO analysis
python type5_ood_recovery/run_ood_experiment.py --skip-slo-analysis

# Custom SLO thresholds (1.0 to 5.0, step 0.2)
python type5_ood_recovery/run_ood_experiment.py \
    --slo-min-threshold 1.0 \
    --slo-max-threshold 5.0 \
    --slo-step 0.2
```

---

### 1. Start Services (Optional)
If not using user-started instances:
```bash
# Start services (48 instances)
cd experiments/13.workflow_benchmark/scripts
./start_type5_services.sh 48
```

### 2. Run Experiment (Manual)

**Option A: Run Both Modes (Recommended)**
```bash
cd experiments/13.workflow_benchmark
python -m scripts.run_type5_ood_experiment \
    --num-tasks 500 \
    --qps 2.83 \
    --phase1-count 100 \
    --mode both
```

**Option B: Run Single Mode**
```bash
# Recovery mode
python -m type5_ood_recovery.simulation.test_ood_sim \
    --num-tasks 500 --qps 2.83

# Baseline mode (no recovery)
python -m type5_ood_recovery.simulation.test_ood_sim \
    --num-tasks 500 --qps 2.83 --no-recovery
```

### 3. View Results
Results are saved to `output_ood/`:
- `metrics_recovery.json` / `metrics_baseline.json`
- `throughput_*.png` - Throughput over time
- `gantt_*.png` - Task scheduling visualization

---

## Optimal Configurations (48 Instances)

Based on grid search results (`parallel_grid_search.py`):

| Scale | Load | Phase2/3 QPS | Recovery | Baseline | Improvement |
|-------|------|--------------|----------|----------|-------------|
| 0.2 | 1.15 | 14.81 | 11.66 | 10.38 | **12.37%** |
| 0.5 | 1.15 | 5.93 | 5.01 | 4.36 | **14.97%** |
| 1.0 | 1.10 | 2.83 | 2.56 | 2.19 | **17.17%** |
| **1.0** | **1.15** | **2.96** | **2.60** | **2.12** | **22.56%** ⭐ |
| 2.0 | 1.10 | 1.42 | 1.31 | 1.08 | **20.50%** |

> **Grid Search Parameters**: `--num-tasks 2000`, `--phase1-count 200`, `--phase1-qps 10.0`, `--phase23-distribution four_peak`

### 🚀 Copy-Paste Ready: Best Configuration (48 Instances)

```bash
# Navigate to experiment directory
cd experiments/13.workflow_benchmark

# Run with optimal parameters (22.56% improvement)
python -m scripts.run_type5_ood_experiment \
    --num-tasks 2000 \
    --phase1-count 200 \
    --phase1-qps 10.0 \
    --phase23-qps 2.96 \
    --runtime-scale 1.0 \
    --phase23-distribution four_peak \
    --mode both
```

### Alternative Configurations
```bash
# 17.17% improvement (slightly lower load)
python -m scripts.run_type5_ood_experiment \
    --num-tasks 2000 --phase1-count 200 --phase1-qps 10.0 \
    --phase23-qps 2.83 --mode both

# 20.50% improvement (2x runtime scale)
python -m scripts.run_type5_ood_experiment \
    --num-tasks 2000 --phase1-count 200 --phase1-qps 10.0 \
    --phase23-qps 1.42 --runtime-scale 2.0 --mode both
```

### 128 Instances
| Phase2/3 QPS | Improvement |
|--------------|-------------|
| 7.90 | ~38% |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-tasks` | 100 | Total tasks to submit |
| `--qps` | 1.0 | Default task submission rate |
| `--phase1-qps` | (=qps) | Phase 1 QPS (warmup phase) |
| `--phase23-qps` | (=qps) | Phase 2/3 QPS (OOD phase) |
| `--runtime-scale` | 1.0 | Global scaling factor for task runtime |
| `--phase1-count` | 100 | Warmup tasks (correct predictions) |
| `--phase2-transition-count` | 10 | Trigger recovery after N Phase 2 completions |
| `--phase23-distribution` | four_peak | OOD distribution (`normal`, `uniform`, `four_peak`) |
| `--no-recovery` | false | Baseline mode (disable recovery) |
| `--scheduler-url` | http://127.0.0.1:8100 | Scheduler endpoint |
| `--predictor-url` | http://127.0.0.1:8000 | Predictor endpoint |
| `--skip-service-check` | false | Skip health checks (for user-started instances) |

---

## Using User-Started Instances

```bash
# Start your own services first, then:
python -m scripts.run_type5_ood_experiment \
    --scheduler-url http://your-scheduler:8100 \
    --predictor-url http://your-predictor:8000 \
    --model-id your_model \
    --skip-service-check \
    --num-tasks 500 --qps 2.83
```

---

## Standalone Simulation (No Services Required)

For pure simulation without external services:
```bash
uv run type5_ood_recovery/standalone_sim.py \
    --num-instances 48 \
    --num-tasks 500 \
    --qps 2.83 \
    --phase23-distribution four_peak \
    --output-dir output_sim
```

---

## Best Configuration: High QPS Visual Difference

This configuration produces the most visually dramatic difference between Recovery and Baseline modes, with a **15.6 tasks/s QPS difference**.

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num-instances` | 512 | Number of simulated instances |
| `--num-tasks` | 10000 | Total tasks to process |
| `--phase1-count` | 500 | Warmup tasks with correct predictions |
| `--phase1-qps` | 200.0 | High QPS for fast warmup |
| `--phase23-qps` | 150.0 | Overload QPS (~3x service capacity) |
| `--phase23-distribution` | weighted_bimodal | 80/20 → 20/80 distribution flip |
| `--runtime-scale` | 0.05 | Scale factor for fast execution |
| `--phase23-bimodal-scale` | 2.0 | Scale factor for Phase 2/3 bimodal samples |
| `--phase23-small-peak-ratio` | 0.2 | Ratio of small peak in Phase 2/3 |
| `--phase2-transition-count` | 200 | Trigger recovery after 200 Phase 2 completions |

### Expected Results

| Metric | Recovery | Baseline |
|--------|----------|----------|
| Duration | ~94s | ~115s |
| Avg Throughput | **106.6 tasks/s** | **91.0 tasks/s** |
| Improvement | +17.2% | - |
| QPS Difference | **+15.6 tasks/s** | - |

### Reproduction Commands

```bash
# Navigate to experiment directory
cd experiments/13.workflow_benchmark

# Run Recovery mode
uv run python type5_ood_recovery/standalone_sim.py \
    --num-instances 512 \
    --num-tasks 10000 \
    --phase1-count 500 \
    --phase1-qps 200.0 \
    --phase23-qps 150.0 \
    --phase23-distribution weighted_bimodal \
    --runtime-scale 0.05 \
    --phase23-bimodal-scale 2.0 \
    --phase23-small-peak-ratio 0.2 \
    --phase2-transition-count 200 \
    --seed 42 \
    --output-dir output_high_qps/recovery

# Run Baseline mode
uv run python type5_ood_recovery/standalone_sim.py \
    --num-instances 512 \
    --num-tasks 10000 \
    --phase1-count 500 \
    --phase1-qps 200.0 \
    --phase23-qps 150.0 \
    --phase23-distribution weighted_bimodal \
    --runtime-scale 0.05 \
    --phase23-bimodal-scale 2.0 \
    --phase23-small-peak-ratio 0.2 \
    --phase2-transition-count 200 \
    --seed 42 \
    --no-recovery \
    --output-dir output_high_qps/baseline
```

### Output Files

After running both modes, the following files are generated:
- `output_high_qps/recovery/recovery/metrics.json` - Recovery metrics
- `output_high_qps/recovery/recovery/throughput.png` - Recovery throughput plot
- `output_high_qps/recovery/recovery/gantt.png` - Recovery Gantt chart
- `output_high_qps/baseline/baseline/metrics.json` - Baseline metrics
- `output_high_qps/baseline/baseline/throughput.png` - Baseline throughput plot

### Pre-generated Charts

Reference charts are available in this directory:
- `high_qps_comparison.png` - Side-by-side comparison chart
- `high_qps_recovery_throughput.png` - Recovery throughput details
- `high_qps_baseline_throughput.png` - Baseline throughput details

### Key Insights

1. **Queue Backlog is Essential**: Recovery advantage requires QPS > steady-state service capacity
2. **Weighted Bimodal Distribution**: Phase 1 uses 80% small peak / 20% large peak; Phase 2/3 flips to 20% small / 80% large with 2x scale
3. **Prediction Mismatch**: Phase 2 tasks use predictions sampled from Phase 1 distribution, causing ~46x average underestimation
4. **Runtime Scale Effect**: Reducing `runtime-scale` proportionally increases QPS values while maintaining the same improvement ratio

---

## Best Configuration: Real-Time Throughput Difference >15%

This configuration maximizes the **Real-Time Throughput** difference between Predictor Re-Train and Baseline modes, achieving **+18.2% improvement**.

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--num-instances` | 512 | Number of simulated instances |
| `--num-tasks` | 10000 | Total tasks to process |
| `--phase1-count` | 500 | Warmup tasks with correct predictions |
| `--phase1-qps` | 200.0 | High QPS for fast warmup |
| `--phase23-qps` | 180.0 | Higher overload for queue buildup |
| `--phase23-distribution` | weighted_bimodal | 80/20 → 10/90 distribution flip |
| `--runtime-scale` | 0.05 | Scale factor for fast execution |
| `--phase23-bimodal-scale` | 3.0 | **Increased** scale for Phase 2/3 (原2.0) |
| `--phase23-small-peak-ratio` | 0.1 | **Reduced** small peak ratio (原0.2) |
| `--phase2-transition-count` | 300 | **Increased** for more queue buildup (原200) |

### Expected Results

| Metric | Predictor Re-Train | Baseline |
|--------|----------|----------|
| Duration | ~135s | ~165s |
| **Real-Time Throughput Avg** | **74.3 tasks/s** | **62.9 tasks/s** |
| **Improvement** | **+18.2%** | - |
| QPS Difference | **+11.5 tasks/s** | - |

### Reproduction Commands

```bash
# Navigate to experiment directory
cd experiments/13.workflow_benchmark

# Run Predictor Re-Train mode
uv run python type5_ood_recovery/standalone_sim.py \
    --num-instances 512 \
    --num-tasks 10000 \
    --phase1-count 500 \
    --phase1-qps 200.0 \
    --phase23-qps 180.0 \
    --phase23-distribution weighted_bimodal \
    --runtime-scale 0.05 \
    --phase23-bimodal-scale 3.0 \
    --phase23-small-peak-ratio 0.1 \
    --phase2-transition-count 300 \
    --seed 42 \
    --output-dir output_realtime_diff/recovery

# Run Baseline mode
uv run python type5_ood_recovery/standalone_sim.py \
    --num-instances 512 \
    --num-tasks 10000 \
    --phase1-count 500 \
    --phase1-qps 200.0 \
    --phase23-qps 180.0 \
    --phase23-distribution weighted_bimodal \
    --runtime-scale 0.05 \
    --phase23-bimodal-scale 3.0 \
    --phase23-small-peak-ratio 0.1 \
    --phase2-transition-count 300 \
    --seed 42 \
    --no-recovery \
    --output-dir output_realtime_diff/baseline
```

### Pre-generated Chart

- `realtime_diff_comparison_annotated.png` - Annotated comparison showing +18.2% Real-Time improvement

### Key Differences from High QPS Configuration

| Parameter | High QPS Config | Real-Time Diff Config |
|-----------|-----------------|----------------------|
| `phase23-bimodal-scale` | 2.0 | **3.0** |
| `phase23-small-peak-ratio` | 0.2 | **0.1** |
| `phase2-transition-count` | 200 | **300** |
| `phase23-qps` | 150.0 | **180.0** |
| Real-Time Improvement | +17.2% | **+18.2%** |

The increased `phase23-bimodal-scale` and reduced `phase23-small-peak-ratio` create larger runtime variance, amplifying the prediction error and making the Re-Train correction more impactful.

---

## Three-Phase Pattern

1. **Phase 1** (Warmup): Correct predictions, system stable
2. **Phase 2** (OOD): Wrong predictions, throughput collapses
3. **Phase 3** (Recovery): Predictions corrected, throughput restored

The experiment measures how quickly and effectively the system recovers from Phase 2 to Phase 3.

---

## SLO Violation Rate Analysis

Analyze SLO (Service Level Objective) violations by comparing task latency to execution time across different ratio thresholds.

### SLO Definition

- **Execution Time** = `complete_time - exec_start_time` (actual task processing time)
- **Latency** = `complete_time - submit_time` (total time from submission to completion)
- **SLO Ratio** = `latency / execution_time`
- **SLO Violation**: A task violates SLO when `ratio > threshold`

### Usage

```bash
# Navigate to experiment directory
cd experiments/13.workflow_benchmark

# Run with default settings (thresholds 1.0 to 10.0, step 0.5)
/path/to/.venv/bin/python3 type5_ood_recovery/plot_slo_violation.py

# Or use uv from project root
cd /path/to/swarmpilot-refresh
uv run python experiments/13.workflow_benchmark/type5_ood_recovery/plot_slo_violation.py \
    --baseline experiments/13.workflow_benchmark/output_realtime_diff/baseline/metrics.json \
    --recovery experiments/13.workflow_benchmark/output_realtime_diff/recovery/metrics.json \
    --output-dir experiments/13.workflow_benchmark/type5_ood_recovery/results/slo_analysis
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--baseline` | `output_realtime_diff/baseline/metrics.json` | Path to baseline metrics |
| `--recovery` | `output_realtime_diff/recovery/metrics.json` | Path to recovery metrics |
| `--output-dir` | `type5_ood_recovery/results/slo_analysis` | Output directory for plots |
| `--min-threshold` | 1.0 | Minimum SLO ratio threshold |
| `--max-threshold` | 10.0 | Maximum SLO ratio threshold |
| `--step` | 0.5 | Step size for threshold sweep |

### Examples

```bash
# Fine-grained search (1.0 to 5.0, step 0.2)
python3 type5_ood_recovery/plot_slo_violation.py \
    --min-threshold 1.0 \
    --max-threshold 5.0 \
    --step 0.2

# Coarse search (1.0 to 20.0, step 1.0)
python3 type5_ood_recovery/plot_slo_violation.py \
    --min-threshold 1.0 \
    --max-threshold 20.0 \
    --step 1.0
```

### Output

Generates one plot per threshold value in the output directory:
- `slo_ratio_1.0.png`
- `slo_ratio_1.5.png`
- `slo_ratio_2.0.png`
- ... (19 plots for default settings)
- `slo_ratio_10.0.png`

Each plot shows:
- **X-axis**: Phase 2 (OOD), Phase 3 (Post-Recovery), All Tasks (P2+P3)
- **Y-axis**: SLO Violation Rate (%)
- **Bars**: Baseline (red) vs Recovery (blue)
- **Info box**: Task counts per phase

> **Note**: Phase 1 (warmup) data is excluded from the analysis to focus on the OOD and Recovery phases where the comparison is meaningful.

### Interpretation

- **Lower threshold** (e.g., 1.5): Strict SLO - latency should be close to pure execution time
- **Higher threshold** (e.g., 5.0): Relaxed SLO - allows more queuing/scheduling overhead
- **Recovery advantage**: Lower violation rates in Phase 3 compared to Baseline indicate successful recovery
