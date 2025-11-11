# Out-of-Distribution Experiments (Experiment 12)

## Overview

This experiment suite implements out-of-distribution testing for the SwarmPilot scheduler to validate predictor re-training necessity. It contains two main experiments testing different scenarios where model B1 receives different parameter distributions.

## Experiments

### Experiment 1: B1 Samples Sleep Time from A1 Distribution
- **Baseline**: B1 sleep_time from A1 (dr_boot), exp_runtime from standard B1 (dr_query)
- **Comparison**: B1 sleep_time from A1, exp_runtime = sleep_time

### Experiment 2: B1 Scaling
- **Baseline**: B1 sleep_time scaled by [0.2, 0.5, 0.8]x, exp_runtime unchanged
- **Comparison**: Both sleep_time and exp_runtime scaled by [0.2, 0.5, 0.8]x

## Components

### 1. A1 Data Sampler (`a1_data_sampler.py`)
Module for sampling sleep_time values from A1 task distribution.

**Features:**
- Load and parse A1 (dr_boot) data
- Random sampling with proper distribution maintenance
- Statistical validation of sampled values
- Thread-safe sampling operations

**Usage:**
```python
from a1_data_sampler import A1DataSampler

# Create sampler
sampler = A1DataSampler(seed=42)

# Sample single value
sleep_time = sampler.sample()

# Sample batch
sleep_times = sampler.sample_batch(count=100)

# Get statistics
sampler.print_statistics()
```

### 2. Workload Generator Enhancement (`workload_generator.py`)
Extended workload generator with A1 sampling support.

**New Function:**
```python
from workload_generator import generate_workflow_with_a1_b1_sampling

# Generate workflow with B1 sleep_time from A1
workflow, config = generate_workflow_with_a1_b1_sampling(
    num_workflows=100,
    seed=42,
    use_a1_for_exp_runtime=False  # True for exp_runtime = sleep_time
)
```

### 3. Experiment Configuration (`ood_config.py`)
Comprehensive configuration system for all experiment variants.

**Configuration Classes:**
- `OODExperimentConfig`: Main configuration class
- `ExperimentType`: Enum for experiment types
- `B1DataSource`: Enum for B1 data sources

**Predefined Configurations:**
```python
from ood_config import (
    get_exp1_baseline_config,
    get_exp1_comparison_config,
    get_exp2_baseline_configs,
    get_exp2_comparison_configs,
    get_all_experiment_configs
)

# Get all 8 experiment configurations
configs = get_all_experiment_configs(num_workflows=100, seed=42)
```

### 4. Metrics Collector (`metrics_collector.py`)
Comprehensive metrics collection and analysis system.

**Features:**
- Workflow latency tracking (mean, median, P95, P99)
- Makespan calculation
- Statistical comparison between baseline and comparison
- JSON export for further analysis

**Usage:**
```python
from metrics_collector import MetricsCollector, MetricsComparator

# Create collector
collector = MetricsCollector("exp1_baseline", "Baseline experiment")

# Collect metrics
collector.start_collection()
# ... run workflows ...
collector.record_workflow_completion(latency_seconds)
collector.end_collection()

# Generate report
metrics = collector.generate_metrics_report(
    num_workflows=100,
    strategy="probabilistic"
)

# Save to file
collector.save_metrics(metrics, output_dir)

# Compare experiments
comparison = MetricsComparator.compare_experiments(baseline, comparison)
MetricsComparator.print_comparison(comparison)
```

### 5. Test Scripts

#### `test_a1_sampling.py`
Tests A1 sampling integration and validates that B1 tasks correctly use A1-derived values.

```bash
python test_a1_sampling.py
```

### 6. Orchestration Script (`run_ood_experiments.py`)
Main automation script for running all experiments.

**Usage:**
```bash
# Run all experiments
python run_ood_experiments.py --experiment all --num-workflows 100

# Run only Experiment 1
python run_ood_experiments.py --experiment exp1 --num-workflows 100

# Run only Experiment 2
python run_ood_experiments.py --experiment exp2 --num-workflows 100

# Specify output directory
python run_ood_experiments.py --experiment all --output-dir ./custom_results
```

## Results

Results are saved to the `ood_results/` directory with the following structure:

```
ood_results/
├── exp1_baseline_YYYYMMDD_HHMMSS.json
├── exp1_comparison_YYYYMMDD_HHMMSS.json
├── exp2_baseline_scale_0.2_YYYYMMDD_HHMMSS.json
├── exp2_baseline_scale_0.5_YYYYMMDD_HHMMSS.json
├── exp2_baseline_scale_0.8_YYYYMMDD_HHMMSS.json
├── exp2_comparison_scale_0.2_YYYYMMDD_HHMMSS.json
├── exp2_comparison_scale_0.5_YYYYMMDD_HHMMSS.json
└── exp2_comparison_scale_0.8_YYYYMMDD_HHMMSS.json
```

Each JSON file contains:
- Experiment metadata and configuration
- Workflow latency statistics (mean, median, P95, P99)
- Makespan timing
- Raw workflow completion times
- Timestamps

## Key Features

### 1. Probabilistic Strategy Enforcement
All experiments automatically enforce the use of probabilistic scheduling strategy as specified in the requirements.

### 2. Statistical Validation
- Distribution matching validation for A1 sampling
- Percentile calculations (P50, P95, P99)
- Baseline vs comparison statistical analysis

### 3. Scaling Support
- Independent scaling of sleep_time and exp_runtime
- Multiple scaling factors [0.2, 0.5, 0.8]
- Validation of scaled values

### 4. Comprehensive Logging
- Parameter source tracking
- Scaling operation logs
- Experiment execution progress

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
uv sync
```

### 2. Run Single Test
```python
# Test A1 sampling
python test_a1_sampling.py

# Test metrics collection
python metrics_collector.py

# Test configuration
python ood_config.py
```

### 3. Run Full Experiment Suite
```bash
# Run all experiments with 100 workflows each
python run_ood_experiments.py --experiment all --num-workflows 100
```

### 4. Analyze Results
```python
import json
from pathlib import Path

# Load results
results_dir = Path("ood_results")
for result_file in results_dir.glob("*.json"):
    with open(result_file) as f:
        data = json.load(f)
        print(f"{data['experiment_name']}: {data['workflow_latencies']['mean']:.3f}s mean latency")
```

## Implementation Notes

### A1 Sampling Accuracy
The A1 sampler maintains distribution characteristics with <15% deviation in mean and std (validated with 1000+ samples).

### Scaling Precision
Scaling operations apply exact multiplicative factors to all selected parameters with full precision.

### Metrics Accuracy
All percentile calculations use NumPy's standard algorithms with proper handling of edge cases.

## Testing

All components include built-in testing capabilities:

```bash
# Component tests
python a1_data_sampler.py        # Test A1 sampling
python ood_config.py             # Test configurations
python metrics_collector.py       # Test metrics collection
python test_a1_sampling.py        # Integration test

# Full suite
python run_ood_experiments.py --experiment exp1 --num-workflows 20
```

## File Structure

```
experiments/12.out_of_distribution/
├── OOD_README.md                    # This file
├── a1_data_sampler.py               # A1 sampling module
├── ood_config.py                    # Configuration system
├── metrics_collector.py             # Metrics collection
├── workload_generator.py            # Enhanced workload generator
├── test_a1_sampling.py              # Integration test
├── run_ood_experiments.py           # Main orchestration script
├── common.py                        # Common utilities (from exp07)
├── requirements.txt                 # Python dependencies
├── data/                            # Trace data
│   ├── dr_boot.json
│   ├── dr_query.json
│   ├── dr_criteria.json
│   └── dr_summary_dict.json
└── ood_results/                     # Experiment results (generated)
```

## Dependencies

- Python 3.11+
- numpy >= 1.24.0
- scipy >= 1.11.0
- requests >= 2.31.0
- websockets >= 12.0

## Related Experiments

This experiment extends:
- **Experiment 07**: Multi-model workflow with dynamic merge

## Authors

Generated via Task Master AI workflow automation.

## License

Part of the SwarmPilot project.
