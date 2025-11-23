# Quick Start Guide (5 Minutes)

Get started with the Unified Workflow Benchmark Framework in under 5 minutes.

## Prerequisites

- Python 3.8+
- `uv` package manager (recommended) or `pip`

## Installation (1 minute)

```bash
# Navigate to the benchmark directory
cd experiments/13.workflow_benchmark

# Install dependencies with uv (recommended)
uv sync

# OR install with pip
pip install -r requirements.txt
```

If dependencies file doesn't exist, the core framework has minimal dependencies:
```bash
pip install requests websockets numpy
```

## Method 1: Using the CLI Tool (Easiest - 2 minutes)

The CLI tool provides the simplest way to run experiments:

### Text2Video Simulation

```bash
# Run a quick 1-minute test
python tools/cli.py run-text2video-sim --duration 60 --num-workflows 120

# Run full 5-minute experiment
python tools/cli.py run-text2video-sim --qps 2.0 --duration 300 --num-workflows 600
```

### Deep Research Simulation

```bash
# Run a quick 1-minute test
python tools/cli.py run-deep-research-sim --duration 60 --num-workflows 60

# Run full 10-minute experiment
python tools/cli.py run-deep-research-sim --qps 1.0 --duration 600 --num-workflows 600
```

### View Results

```bash
# Metrics are saved to output/metrics.json
cat output/metrics.json | python -m json.tool
```

## Method 2: Direct Python Script (2 minutes)

### Text2Video Simulation

```bash
# Set configuration
export QPS=2.0
export DURATION=60
export NUM_WORKFLOWS=120

# Run experiment
python type1_text2video/simulation/test_workflow_sim.py
```

**Output**: Metrics saved to `output/metrics.json`

### Deep Research Simulation

```bash
# Set configuration
export QPS=1.0
export DURATION=60
export NUM_WORKFLOWS=60
export FANOUT_COUNT=3

# Run experiment
python type2_deep_research/simulation/test_workflow_sim.py
```

**Output**: Metrics saved to `output/metrics.json`

## Method 3: Real Cluster Mode (Advanced)

### Prerequisites for Real Mode

1. Running scheduler services (see [Service Setup](#service-setup))
2. Deployed models
3. Caption data file (for Text2Video)

### Text2Video Real Mode

```bash
# Configure cluster endpoints
export MODE=real
export SCHEDULER_A_URL=http://localhost:8100
export SCHEDULER_B_URL=http://localhost:8101
export QPS=2.0
export DURATION=300
export NUM_WORKFLOWS=100

# Run experiment
python type1_text2video/real/test_workflow_real.py
```

### Deep Research Real Mode

```bash
# Configure cluster endpoints
export MODE=real
export SCHEDULER_A_URL=http://localhost:8100
export SCHEDULER_B_URL=http://localhost:8101
export QPS=1.0
export DURATION=600
export NUM_WORKFLOWS=100
export FANOUT_COUNT=3

# Run experiment
python type2_deep_research/real/test_workflow_real.py
```

## Understanding the Output

After running an experiment, you'll see:

1. **Console Output**: Real-time progress and summary statistics
2. **Metrics File**: Detailed metrics in `output/metrics.json`

### Example Metrics Report

```
==================== Workflow Benchmark Metrics ====================

Task Metrics:
  Total tasks: 3600
  Task types: A1, A2, B

  A1 tasks:
    Count: 600
    Success rate: 100.0%
    Latency p50: 7.5s
    Latency p95: 12.3s
    Latency p99: 14.8s

  A2 tasks:
    Count: 600
    Success rate: 100.0%
    Latency p50: 7.2s
    Latency p95: 11.9s

  B tasks:
    Count: 2400
    Success rate: 100.0%
    Latency p50: 8.1s
    Latency p95: 13.2s

Workflow Metrics:
  Completed: 600/600 (100.0%)
  Total latency p50: 85.2s
  Total latency p95: 142.8s

Throughput:
  Achieved QPS: 2.01
  Duration: 298.5s

====================================================================
```

## Configuration Options

### Common Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `QPS` | Query per second rate | 2.0 (T2V), 1.0 (DR) | `export QPS=3.0` |
| `DURATION` | Experiment duration (seconds) | 300 (T2V), 600 (DR) | `export DURATION=600` |
| `NUM_WORKFLOWS` | Total workflows to execute | 600 | `export NUM_WORKFLOWS=1000` |
| `OUTPUT_DIR` | Output directory | `output` | `export OUTPUT_DIR=results` |
| `MODE` | Simulation or real | `simulation` | `export MODE=real` |

### Text2Video Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_B_LOOPS` | B task iterations per workflow | 4 |
| `CAPTION_FILE` | Path to captions JSON | `captions_10k.json` |

### Deep Research Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `FANOUT_COUNT` | B1/B2 tasks per workflow | 3 |

## Troubleshooting

### Import Errors

```bash
# If you see "ModuleNotFoundError: No module named 'websockets'"
pip install websockets requests numpy
```

### Permission Errors

```bash
# Make scripts executable
chmod +x type1_text2video/simulation/test_workflow_sim.py
chmod +x type2_deep_research/simulation/test_workflow_sim.py
chmod +x tools/cli.py
```

### No Output Directory

```bash
# Create output directory manually
mkdir -p output
```

## Next Steps

- **Advanced Configuration**: See [Configuration Guide](CONFIGURATION.md)
- **Real Cluster Mode**: See [Real Mode Setup](REAL_MODE_SETUP.md)
- **API Reference**: See [API Documentation](API.md)
- **Custom Workflows**: See [Development Guide](DEVELOPMENT.md)

## Quick Examples by Use Case

### 1. Quick Smoke Test (30 seconds)

```bash
# Fast sanity check
python tools/cli.py run-text2video-sim --duration 30 --num-workflows 60
```

### 2. Performance Benchmarking (5 minutes)

```bash
# Full benchmark with metrics
python tools/cli.py run-text2video-sim --qps 2.0 --duration 300 --num-workflows 600
```

### 3. Stress Testing (10+ minutes)

```bash
# High load test
python tools/cli.py run-text2video-sim --qps 5.0 --duration 600 --num-workflows 3000
```

### 4. Comparison Testing

```bash
# Run multiple experiments with different QPS
for qps in 1.0 2.0 3.0 4.0; do
  python tools/cli.py run-text2video-sim --qps $qps --duration 300 --output-dir output_qps_$qps
done
```

## Service Setup

For real cluster mode, you need running services. See the original experiment setups:

- **Text2Video**: `experiments/03.Exp4.Text2Video/start_all_services.sh`
- **Deep Research**: `experiments/07.Exp2.Deep_Research_Migration_Test/start_all_services.sh`

Or use the service management tools:

```python
from service_management import ServiceLauncher, DeploymentManager

# Start services
launcher = ServiceLauncher()
launcher.start_all_services(mode="simulation")

# Deploy models
deployer = DeploymentManager()
deployer.deploy_to_scheduler(models, scheduler_url)
```

See [Service Management Guide](SERVICE_MANAGEMENT.md) for details.
