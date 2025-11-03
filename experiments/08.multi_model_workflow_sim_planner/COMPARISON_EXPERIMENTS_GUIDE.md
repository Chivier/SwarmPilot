# Comparison Experiments Guide

This guide explains how to run and analyze comparison experiments for Experiment 08.

## Overview

The comparison experiments test **6 different configurations**:

| # | Strategy       | Migration Mode | Description                                    |
|---|----------------|----------------|------------------------------------------------|
| 1 | min_time       | Enabled        | Minimum expected time + dynamic migration      |
| 2 | min_time       | Disabled       | Minimum expected time + static distribution    |
| 3 | probabilistic  | Enabled        | Probabilistic sampling + dynamic migration     |
| 4 | probabilistic  | Disabled       | Probabilistic sampling + static distribution   |
| 5 | round_robin    | Enabled        | Round robin + dynamic migration                |
| 6 | round_robin    | Disabled       | Round robin + static distribution              |

## Quick Start

### 1. Run All 6 Configurations (Automated)

The easiest way to run all experiments:

```bash
# From the experiment directory
cd experiments/08.multi_model_workflow_sim_planner

# Run all 6 configurations with 100 workflows per phase
uv run python3 run_comparison_experiments.py --num-workflows-per-phase 100
```

This will:
- Automatically run all 6 configurations sequentially
- Stop/start services between each experiment
- Save individual results to `results/exp08_*.json`
- Generate a summary in `results/comparison_run_*.json`

**Estimated time:** ~15-30 minutes per configuration (total: 1.5-3 hours for all 6)

### 2. Analyze Results

After all experiments complete:

```bash
# Analyze all results
uv run python3 analyze_comparison_results.py --results-dir results

# Or analyze specific results
uv run python3 analyze_comparison_results.py --pattern "exp08_*_20250103_*.json"
```

This generates:
- **JSON report:** `results/comparison_analysis_*.json` (structured data)
- **Markdown report:** `results/comparison_analysis_*.md` (human-readable tables)

### 3. View Results

Open the Markdown report:

```bash
# View in terminal
cat results/comparison_analysis_*.md

# Or open in editor
code results/comparison_analysis_*.md
```

## Manual Experiment Execution

If you prefer to run experiments manually or test individual configurations:

### Run Single Configuration

```bash
# With migration (default)
uv run python3 test_dynamic_migration.py \
    --strategy min_time \
    --num-workflows-per-phase 100 \
    --enable-migration

# Without migration (static distribution)
uv run python3 test_dynamic_migration.py \
    --strategy probabilistic \
    --num-workflows-per-phase 100 \
    --disable-migration
```

### Before Each Manual Run

```bash
# Stop all services
./stop_all_services.sh

# Clean logs
rm -f logs/*.log logs/*.pid

# Start all services
./start_all_services.sh
```

## Configuration Options

### run_comparison_experiments.py

```bash
uv run python3 run_comparison_experiments.py \
    --num-workflows-per-phase 100 \    # Workflows per phase (default: 100)
    --qps 2.0 \                        # QPS for task submission (default: 2.0)
    --output-dir results               # Output directory (default: ./results)
```

### test_dynamic_migration.py

```bash
uv run python3 test_dynamic_migration.py \
    --strategy min_time \              # Strategy: min_time, probabilistic, round_robin
    --num-workflows-per-phase 100 \    # Workflows per phase
    --qps 2.0 \                        # QPS for A task submission
    --enable-migration                 # Enable dynamic migration (default)
    # OR --disable-migration           # Use static distribution
    --total-instances 16 \             # Total instances (default: 16)
    --instance-start-port 8210         # Starting port (default: 8210)
```

### analyze_comparison_results.py

```bash
uv run python3 analyze_comparison_results.py \
    --results-dir results \            # Results directory (default: ./results)
    --pattern "exp08_*.json" \         # File pattern (default: exp08_*.json)
    --output-prefix comparison_analysis # Output prefix (default: comparison_analysis)
```

## Understanding Results

### Result Files

Each experiment generates:

```
results/
├── exp08_migration_min_time_20250103_120000.json      # Individual experiment results
├── exp08_static_min_time_20250103_120530.json
├── exp08_migration_probabilistic_20250103_121100.json
├── ...
├── comparison_run_20250103_123000.json                # Batch run summary
├── comparison_analysis_20250103_123500.json           # Analysis (JSON)
└── comparison_analysis_20250103_123500.md             # Analysis (Markdown)
```

### Key Metrics in Analysis Report

**1. Workflow Latency Comparison**
- Average, P50, P90, P99 latencies for each configuration
- Comparison across all three phases
- Standard deviation and coefficient of variation

**2. Migration Overhead**
- Time spent migrating instances between phases
- Only applicable to migration-enabled experiments
- Broken down by migration event (Phase 1→2, Phase 2→3)

**3. Throughput Comparison**
- Completion rate (% of workflows completed)
- Actual QPS achieved
- Phase duration

**4. Load Balancing Effectiveness**
- Coefficient of variation (CV) - lower is better
- Indicates how evenly distributed the workload is
- Compares different strategies

**5. Migration Impact**
- Direct comparison of migration vs static for each strategy
- Shows improvement percentage (positive = migration is better)
- Helps evaluate if migration overhead is worth it

## Troubleshooting

### Services Won't Start

```bash
# Check if ports are in use
netstat -tuln | grep -E '8100|8200|821[0-5]'

# Force kill all processes
pkill -f "scheduler|instance|predictor"
pkill -f "sleep_model"
docker-compose down

# Try again
./start_all_services.sh
```

### Experiment Hangs or Fails

Check logs:

```bash
# Scheduler logs
tail -f logs/scheduler_a.log
tail -f logs/scheduler_b.log

# Instance logs
tail -f logs/instance_000.log

# Experiment logs
tail -f logs/experiment_*.log
```

### Missing Results

Verify result files exist:

```bash
ls -lh results/exp08_*.json
```

If some configurations are missing, re-run just those:

```bash
# Re-run specific configuration
./stop_all_services.sh
./start_all_services.sh
uv run python3 test_dynamic_migration.py \
    --strategy round_robin \
    --disable-migration \
    --num-workflows-per-phase 100
```

## Example Workflow

Complete example from start to finish:

```bash
# 1. Navigate to experiment directory
cd experiments/08.multi_model_workflow_sim_planner

# 2. Run all 6 configurations (grab a coffee ☕)
uv run python3 run_comparison_experiments.py --num-workflows-per-phase 100

# 3. Analyze results
uv run python3 analyze_comparison_results.py

# 4. View report
cat results/comparison_analysis_*.md

# 5. Optional: View individual experiment details
cat results/exp08_migration_min_time_*.json | python3 -m json.tool | less
```

## Advanced Usage

### Custom Workflow Patterns

To test with different workflow patterns, modify `workload_generator.py`:

```python
# Edit generate_bimodal_distribution() or other distribution functions
# Then re-run experiments
```

### Different Phase Configurations

To test different phase configurations, modify `PHASE_CONFIGS` in `test_dynamic_migration.py`:

```python
PHASE_CONFIGS = [
    PhaseConfig(phase_id=1, fanout=5, scheduler_a_instances=4, scheduler_b_instances=12, num_workflows=0),
    PhaseConfig(phase_id=2, fanout=10, scheduler_a_instances=2, scheduler_b_instances=14, num_workflows=0),
    PhaseConfig(phase_id=3, fanout=2, scheduler_a_instances=8, scheduler_b_instances=8, num_workflows=0),
]
```

### Extending Analysis

To add custom metrics, edit `analyze_comparison_results.py`:

```python
class ComparisonAnalyzer:
    def analyze_custom_metric(self) -> Dict:
        # Add your custom analysis here
        pass
```

## Notes

- Each configuration runs 3 phases with different fanout ratios (3, 8, 1)
- Migration-enabled configs redistribute instances between phases
- Static configs keep the initial distribution throughout
- Results include both workflow-level and phase-level metrics
- Analysis automatically handles missing data (e.g., no migration stats for static experiments)

## Questions?

Check the existing documentation:
- `README.md` - General experiment overview
- `ARCHITECTURE_FIX.md` - System architecture details
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `QUICK_REFERENCE.md` - Quick command reference
