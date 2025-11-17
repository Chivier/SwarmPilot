# Quick Start Guide - Multi-Instance Training Data Collection

## TL;DR

```bash
# 1. Start your LLM instances
python -m sglang.launch_server --model llama-7b --port 8001 &
python -m sglang.launch_server --model llama-7b --port 8002 &

# 2. Start predictor service
cd predictor && uv run python3 -m uvicorn src.main:app --port 9000 &

# 3. Configure instances
cp config.example.json config.json
# Edit config.json with your instance URLs

# 4. Run data collection and training
cd experiments/07.multi_model_workflow_dynamic_merge_real
uv run python3 collect_training_data.py --config config.json
```

## Minimal Configuration

Create `config.json`:

```json
{
  "dataset": "data/dataset.jsonl",
  "model_id": "llama-7b",
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

## Multi-Instance Configuration

For parallel execution with 3 instances:

```json
{
  "dataset": "data/dataset.jsonl",
  "model_id": "llama-7b",
  "instances": [
    {"url": "http://localhost:8001", "hardware_name": "NVIDIA H20", "software_name": "sglang", "software_version": "1.0.0"},
    {"url": "http://localhost:8002", "hardware_name": "NVIDIA H20", "software_name": "sglang", "software_version": "1.0.0"},
    {"url": "http://localhost:8003", "hardware_name": "NVIDIA H20", "software_name": "sglang", "software_version": "1.0.0"}
  ],
  "predictor": {"url": "http://localhost:9000"},
  "execution": {
    "max_concurrent_requests": 15
  }
}
```

**Expected speedup**: ~3x faster (2.5 minutes vs 8 minutes for 1,471 tasks)

## What It Does

1. **Loads dataset**: Reads `dataset.jsonl` (99 entries → 1,471 tasks)
2. **Extracts tasks**: Boot (99) + Summary (99) + Queries (1,273)
3. **Parallel execution**: Distributes tasks across instances using round-robin
4. **Collects samples**: Executes each task, measures runtime, builds training samples
5. **Trains predictor**: Submits samples for `expect_error` and `quantile` prediction
6. **Validates models**: Automatically tests predictions and calculates metrics (pinball loss, MAPE)

## Expected Output

```
INFO - Loaded 99 entries from data/dataset.jsonl
INFO - Extracted 1471 LLM tasks from 99 dataset entries
INFO -   - Boot tasks: 99
INFO -   - Summary tasks: 99
INFO -   - Query tasks: 1273
INFO -   - max_tokens distribution: {128: 13, 256: 683, 512: 566, 1024: 207, 2048: 2}

INFO - Initialized 3 LLM instances for parallel execution

Collecting training data: 100%|████████████| 1471/1471 [02:35<00:00, 9.47tasks/s]

INFO - ✓ Successfully collected 1471 samples
INFO - Saving training data to training_data.json
INFO - ✓ Training data saved to training_data.json

INFO - ✓ Training submitted successfully for expect_error
INFO - Validating expect_error model...
============================================================
EXPECT_ERROR Model Validation Results:
============================================================
  Samples validated: 1471/1471
  MAPE: 8.52%
  MAE: 15.67 ms
  Mean Error Margin: 12.34 ms
  Coverage (within error margin): 85.42%
============================================================

INFO - ✓ Training submitted successfully for quantile
INFO - Validating quantile model...
============================================================
QUANTILE Model Validation Results:
============================================================
  Samples validated: 1471

Per-Quantile Metrics:
  Quantile 0.1:
    Pinball Loss: 12.34
    MAPE: 7.85%
    MAE: 14.23 ms
  Quantile 0.5:
    Pinball Loss: 18.45
    MAPE: 9.23%
    MAE: 18.67 ms
  ...

Average Metrics (across all quantiles):
  Avg Pinball Loss: 15.67
  Avg MAPE: 9.15%
  Avg MAE: 17.42 ms
============================================================

INFO - ✓ All training and validation completed successfully
```

## Troubleshooting

### Problem: "Connection refused"
**Solution**: Ensure instances are running
```bash
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
```

### Problem: Slow execution
**Solution**: Increase concurrency
```json
{"execution": {"max_concurrent_requests": 20}}
```

### Problem: Out of memory
**Solution**: Reduce concurrency or limit samples
```json
{
  "max_samples": 100,
  "execution": {"max_concurrent_requests": 5}
}
```

## Training Data Output

The script automatically saves collected training data to `training_data.json` (configurable via `output_file` in config):

```json
{
  "model_id": "llama-7b",
  "platform_info": {
    "software_name": "sglang",
    "software_version": "1.0.0",
    "hardware_name": "NVIDIA H20"
  },
  "samples": [
    {
      "token_length": 125.0,
      "max_tokens": 512.0,
      "cuda_cores": 17920.0,
      "tensor_cores": 560.0,
      "fp32_tflops": 63.0,
      "memory_gb": 96.0,
      "runtime_ms": 245.67
    },
    ...
  ],
  "num_samples": 1471,
  "config": {
    "prediction_types": ["expect_error", "quantile"],
    "training_config": {...}
  }
}
```

This file can be used for:
- Debugging and analysis
- Retraining models with different configurations
- Sharing datasets across experiments
- Offline model development

## Next Steps

After successful training:

```bash
# Query the trained predictor
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llama-7b",
    "hardware_name": "NVIDIA H20",
    "software_name": "sglang",
    "software_version": "1.0.0",
    "sentence": "Your prompt here",
    "max_tokens": 512,
    "prediction_type": "quantile"
  }'
```

## Full Documentation

- **[MULTI_INSTANCE_GUIDE.md](MULTI_INSTANCE_GUIDE.md)** - Comprehensive guide
- **[FIX_SUMMARY.md](FIX_SUMMARY.md)** - Detailed changelog
- **[QUANTILES_CONFIG.md](QUANTILES_CONFIG.md)** - Quantile configuration
- **[config.example.json](config.example.json)** - Full configuration template

---

**Quick tip**: Test with `"max_samples": 10` first to verify everything works before running full dataset.
