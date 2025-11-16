# Multi-Instance Parallel Execution Guide

## Overview

The `collect_training_data.py` script now supports parallel execution across multiple LLM inference instances. This feature enables:

- **Parallel data collection** from multiple instances simultaneously
- **Load balancing** using round-robin distribution
- **Concurrency control** to prevent overloading instances
- **Unified configuration** through a single JSON file

## Configuration File Format

Create a JSON configuration file (e.g., `config.json`) with the following structure:

```json
{
  "dataset": "data/dataset.jsonl",
  "model_id": "llama-7b",
  "prediction_types": ["expect_error", "quantile"],
  "max_samples": null,

  "instances": [
    {
      "url": "http://localhost:8001",
      "hardware_name": "NVIDIA H20",
      "software_name": "sglang",
      "software_version": "1.0.0"
    },
    {
      "url": "http://localhost:8002",
      "hardware_name": "NVIDIA H20",
      "software_name": "sglang",
      "software_version": "1.0.0"
    }
  ],

  "predictor": {
    "url": "http://localhost:9000"
  },

  "training_config": {
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    "epochs": 500,
    "learning_rate": 0.01
  },

  "execution": {
    "timeout": 300.0,
    "max_concurrent_requests": 10
  }
}
```

## Configuration Fields

### Required Fields

#### `dataset` (string)
Path to the JSONL dataset file containing boot, summary, and query tasks.

**Example**: `"data/dataset.jsonl"`

#### `model_id` (string)
Model identifier for the LLM being used across all instances.

**Example**: `"llama-7b"`, `"mistral-8x7b"`

#### `instances` (array of objects)
List of LLM inference instances to use for parallel execution. Each instance must specify:

- `url` (string): HTTP endpoint of the instance (e.g., `"http://localhost:8001"`)
- `hardware_name` (string): GPU hardware name (e.g., `"NVIDIA H20"`, `"NVIDIA A100"`)
- `software_name` (string): Inference software name (e.g., `"sglang"`, `"vllm"`)
- `software_version` (string): Software version (e.g., `"1.0.0"`)

**Important**: All instances should use the same hardware/software configuration for consistent training data.

#### `predictor` (object)
Predictor service configuration:

- `url` (string): HTTP endpoint of the predictor service (e.g., `"http://localhost:9000"`)

### Optional Fields

#### `prediction_types` (array of strings)
Types of predictions to train. Defaults to `["expect_error", "quantile"]`.

**Options**:
- `"expect_error"` - Expected value prediction
- `"quantile"` - Quantile regression (requires `training_config.quantiles`)

#### `max_samples` (integer or null)
Maximum number of samples to collect. Use `null` for all tasks in dataset.

**Example**: `100` (collect only first 100 tasks), `null` (collect all tasks)

#### `training_config` (object)
Training hyperparameters:

- `quantiles` (array of floats): Quantile values for quantile regression (0.0 to 1.0)
  - **Default**: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]`
  - Provides fine-grained percentile predictions
- `epochs` (integer): Number of training epochs
  - **Default**: `500`
- `learning_rate` (float): Learning rate for training
  - **Default**: `0.01`

#### `execution` (object)
Execution control parameters:

- `timeout` (float): Timeout in seconds for each LLM request
  - **Default**: `300.0` (5 minutes)
- `max_concurrent_requests` (integer): Maximum number of concurrent requests across all instances
  - **Default**: `10`
  - Controls concurrency to prevent overloading

## Usage

### Basic Usage

```bash
cd experiments/07.multi_model_workflow_dynamic_merge_real

# Run with configuration file
uv run python3 collect_training_data.py --config config.json
```

### Example Workflow

#### 1. Prepare Dataset

Ensure your dataset file exists and contains entries in the expected format:

```bash
ls -lh data/dataset.jsonl
# Should show file size (e.g., 35.9 MB)
```

#### 2. Start LLM Instances

Start multiple LLM inference instances (example with sglang):

```bash
# Terminal 1: Instance 1
python -m sglang.launch_server --model llama-7b --port 8001

# Terminal 2: Instance 2
python -m sglang.launch_server --model llama-7b --port 8002

# Terminal 3: Instance 3
python -m sglang.launch_server --model llama-7b --port 8003
```

#### 3. Start Predictor Service

```bash
# Terminal 4: Predictor
cd predictor
uv run python3 -m uvicorn src.main:app --host 0.0.0.0 --port 9000
```

#### 4. Create Configuration File

Copy the example and customize:

```bash
cp config.example.json my_config.json
# Edit my_config.json to match your instance URLs and settings
```

#### 5. Run Data Collection and Training

```bash
uv run python3 collect_training_data.py --config my_config.json
```

## How It Works

### Load Balancing

The script uses **round-robin load balancing** to distribute tasks across instances:

1. Task 1 → Instance 0
2. Task 2 → Instance 1
3. Task 3 → Instance 2
4. Task 4 → Instance 0 (cycles back)
5. ...and so on

This ensures even distribution of work across all instances.

### Concurrency Control

The `max_concurrent_requests` parameter controls how many tasks run simultaneously:

```
max_concurrent_requests = 10
instances = 3

Best case: ~3-4 concurrent requests per instance
Worst case (uneven timing): Some instances may handle more temporarily
```

**Recommendation**: Set `max_concurrent_requests` to `2-5 × number_of_instances` for optimal throughput.

### Parallel Execution Flow

```
1. Load dataset → Extract 1,471 tasks
2. Initialize 3 instances (round-robin pool)
3. Create semaphore with max_concurrent=10
4. For each task:
   a. Acquire semaphore slot
   b. Get next instance (round-robin)
   c. Execute task on instance (async)
   d. Extract hardware specs from instance config
   e. Build training sample with features
   f. Release semaphore slot
5. Collect all results
6. Submit to predictor for training
```

## Performance Tuning

### Optimal Concurrency Settings

#### Single Instance
```json
{
  "instances": [{"url": "http://localhost:8001", ...}],
  "execution": {
    "max_concurrent_requests": 5
  }
}
```

#### Two Instances
```json
{
  "instances": [
    {"url": "http://localhost:8001", ...},
    {"url": "http://localhost:8002", ...}
  ],
  "execution": {
    "max_concurrent_requests": 10
  }
}
```

#### Four Instances
```json
{
  "instances": [
    {"url": "http://localhost:8001", ...},
    {"url": "http://localhost:8002", ...},
    {"url": "http://localhost:8003", ...},
    {"url": "http://localhost:8004", ...}
  ],
  "execution": {
    "max_concurrent_requests": 16
  }
}
```

### Timeout Considerations

Set timeout based on expected task duration:

- **Short tasks** (< 10s): `timeout: 30.0`
- **Medium tasks** (10-60s): `timeout: 120.0`
- **Long tasks** (1-5 min): `timeout: 300.0`
- **Very long tasks** (> 5 min): `timeout: 600.0`

## Expected Output

### Successful Execution

```
2025-11-16 12:29:41,862 - INFO - Loaded 99 entries from data/dataset.jsonl
2025-11-16 12:29:41,863 - INFO - Extracted 1471 LLM tasks from 99 dataset entries
2025-11-16 12:29:41,863 - INFO -   - Boot tasks: 99
2025-11-16 12:29:41,863 - INFO -   - Summary tasks: 99
2025-11-16 12:29:41,863 - INFO -   - Query tasks: 1273
2025-11-16 12:29:41,864 - INFO -   - max_tokens distribution: {128: 13, 256: 683, 512: 566, 1024: 207, 2048: 2}

2025-11-16 12:29:42,000 - INFO - Initialized 3 LLM instances for parallel execution
2025-11-16 12:29:42,000 - INFO -   - instance-0@http://localhost:8001
2025-11-16 12:29:42,000 - INFO -   - instance-1@http://localhost:8002
2025-11-16 12:29:42,000 - INFO -   - instance-2@http://localhost:8003

2025-11-16 12:29:42,100 - INFO - Starting parallel data collection (max concurrent: 10)
Collecting training data: 100%|████████████████| 1471/1471 [02:35<00:00, 9.47tasks/s]

2025-11-16 12:32:17,500 - INFO - ✓ Successfully collected 1471 samples

2025-11-16 12:32:17,600 - INFO - Training predictor for expect_error...
2025-11-16 12:32:22,800 - INFO - ✓ Training submitted successfully for expect_error

2025-11-16 12:32:22,900 - INFO - Training predictor for quantile...
2025-11-16 12:32:28,100 - INFO - ✓ Training submitted successfully for quantile

2025-11-16 12:32:28,200 - INFO - ✓ All training jobs completed successfully
```

### Performance Metrics

With 3 instances and `max_concurrent_requests=10`:

- **Total tasks**: 1,471
- **Expected throughput**: ~9-10 tasks/second
- **Total time**: ~2.5 minutes
- **Average per task**: ~100ms (distributed across instances)

## Troubleshooting

### Issue: "Connection refused" errors

**Symptom**:
```
Error: Cannot connect to instance http://localhost:8001
```

**Solution**:
1. Verify all instances are running: `curl http://localhost:8001/v1/models`
2. Check firewall settings
3. Ensure ports match configuration

### Issue: Slow execution despite multiple instances

**Symptom**: Tasks execute slowly even with multiple instances

**Possible causes**:
1. `max_concurrent_requests` too low → Increase to `2-5 × num_instances`
2. Instances overloaded → Reduce `max_concurrent_requests`
3. Timeout too high → Lower timeout for faster failure detection
4. Network latency → Use local instances or improve network

### Issue: Inconsistent hardware specs extraction

**Symptom**:
```
WARNING: Could not extract GPU specs for 'NVIDIA RTX 3090'
```

**Solution**:
1. Check `hardware_name` matches supported GPUs (V100, A100, H100, H20, etc.)
2. Use exact names: `"NVIDIA H20"` not `"H20 GPU"`
3. If unsupported GPU, specs will be empty but training will still work

### Issue: Out of memory during data collection

**Symptom**:
```
MemoryError: Unable to allocate array
```

**Solution**:
1. Use `max_samples` to limit dataset size: `"max_samples": 1000`
2. Reduce `max_concurrent_requests` to limit memory usage
3. Process dataset in batches (run script multiple times with different samples)

## Advanced Configuration

### Multiple Hardware Types

For heterogeneous deployments (different GPUs):

```json
{
  "instances": [
    {
      "url": "http://gpu1:8001",
      "hardware_name": "NVIDIA A100-80GB",
      "software_name": "sglang",
      "software_version": "1.0.0"
    },
    {
      "url": "http://gpu2:8002",
      "hardware_name": "NVIDIA H100",
      "software_name": "sglang",
      "software_version": "1.0.0"
    }
  ]
}
```

**Note**: Training will use the first instance's platform info. Mixing hardware types may affect prediction accuracy.

### Custom Quantiles

For specific scheduling strategies:

```json
{
  "training_config": {
    "quantiles": [0.25, 0.5, 0.75, 0.95, 0.99],
    "epochs": 1000,
    "learning_rate": 0.005
  }
}
```

**Use cases**:
- `[0.5, 0.9, 0.95, 0.99]` - Focus on tail latency
- `[0.1, 0.3, 0.5, 0.7, 0.9]` - Balanced distribution
- `[0.5]` - Median only (fastest training)

### Testing with Small Samples

For quick validation:

```json
{
  "dataset": "data/dataset.jsonl",
  "max_samples": 10,
  "execution": {
    "max_concurrent_requests": 5
  }
}
```

Run time: ~10-20 seconds for 10 samples

## Migration from Old CLI Arguments

### Old Command (deprecated)

```bash
python collect_training_data.py \
  --dataset data/dataset.jsonl \
  --instance-url http://localhost:8001 \
  --predictor-url http://localhost:9000 \
  --model-id llama-7b \
  --max-samples 100
```

### New Command

1. Create `config.json`:

```json
{
  "dataset": "data/dataset.jsonl",
  "model_id": "llama-7b",
  "max_samples": 100,
  "instances": [
    {
      "url": "http://localhost:8001",
      "hardware_name": "NVIDIA H20",
      "software_name": "sglang",
      "software_version": "1.0.0"
    }
  ],
  "predictor": {
    "url": "http://localhost:9000"
  }
}
```

2. Run:

```bash
python collect_training_data.py --config config.json
```

## Related Documentation

- [FIX_SUMMARY.md](FIX_SUMMARY.md) - Complete changelog of all fixes
- [QUANTILES_CONFIG.md](QUANTILES_CONFIG.md) - Quantile configuration details
- [COLLECT_TRAINING_DATA_README.md](COLLECT_TRAINING_DATA_README.md) - Original README
- [config.example.json](config.example.json) - Example configuration file
- [../../predictor/README_FOR_LLM.md](../../predictor/README_FOR_LLM.md) - Predictor service documentation

---

**Last Updated**: 2025-11-16
**Version**: 2.0
**Author**: SwarmPilot Team
