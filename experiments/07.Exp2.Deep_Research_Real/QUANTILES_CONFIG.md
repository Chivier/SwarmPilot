# Quantile Configuration for Predictor Training

## Overview

The `collect_training_data.py` script now includes enhanced quantile configuration for more accurate runtime distribution modeling.

## Default Quantiles

**Configured Values:** `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]`

This configuration provides **10 percentile points** across the distribution:

| Quantile | Percentile | Meaning |
|----------|------------|---------|
| 0.1 | 10th | 10% of tasks finish faster |
| 0.2 | 20th | 20% of tasks finish faster |
| 0.3 | 30th | 30% of tasks finish faster |
| 0.4 | 40th | 40% of tasks finish faster |
| 0.5 | 50th (median) | 50% of tasks finish faster |
| 0.6 | 60th | 60% of tasks finish faster |
| 0.7 | 70th | 70% of tasks finish faster |
| 0.8 | 80th | 80% of tasks finish faster |
| 0.9 | 90th | 90% of tasks finish faster |
| 0.99 | 99th | 99% of tasks finish faster |

## Advantages Over Default (4 quantiles)

### Predictor Default
- **Quantiles:** `[0.5, 0.9, 0.95, 0.99]` (4 points)
- **Coverage:** Focuses on median and tail latencies
- **Use Case:** General purpose, tail latency monitoring

### This Configuration
- **Quantiles:** `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]` (10 points)
- **Coverage:** Fine-grained across entire distribution
- **Use Case:** Detailed performance profiling, scheduling decisions

**Benefits:**
1. **More accurate distribution modeling** - 2.5x more data points
2. **Better scheduler decisions** - Can choose appropriate quantiles for different scenarios
3. **Improved understanding** - See how runtime varies across the full distribution
4. **Flexibility** - Scheduler can query any percentile and get accurate interpolated values

## Usage

### Default Behavior (Uses Enhanced Quantiles)

```bash
# Automatically uses [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
uv run python3 collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://localhost:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b
```

### Custom Configuration (Override with JSON File)

Create a custom config file `custom_config.json`:

```json
{
  "epochs": 500,
  "learning_rate": 0.01,
  "quantiles": [0.25, 0.5, 0.75, 0.95, 0.99]
}
```

Use it:

```bash
uv run python3 collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://localhost:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --training-config custom_config.json
```

## Prediction Example

After training with 10 quantiles, the predictor will return:

```json
{
  "result": {
    "quantiles": {
      "0.1": 85.2,    // 10% finish in ≤85.2ms
      "0.2": 92.1,    // 20% finish in ≤92.1ms
      "0.3": 97.8,    // 30% finish in ≤97.8ms
      "0.4": 102.5,   // 40% finish in ≤102.5ms
      "0.5": 108.0,   // 50% finish in ≤108.0ms (median)
      "0.6": 114.2,   // 60% finish in ≤114.2ms
      "0.7": 121.5,   // 70% finish in ≤121.5ms
      "0.8": 130.8,   // 80% finish in ≤130.8ms
      "0.9": 145.3,   // 90% finish in ≤145.3ms
      "0.99": 198.7   // 99% finish in ≤198.7ms
    }
  }
}
```

## Scheduler Integration

The scheduler can use these quantiles for various strategies:

### Conservative (P90)
```python
# Use 90th percentile for conservative estimates
runtime_estimate = predictions["quantiles"]["0.9"]  # 145.3ms
# Ensures 90% of tasks will finish within estimate
```

### Balanced (P50)
```python
# Use median for balanced approach
runtime_estimate = predictions["quantiles"]["0.5"]  # 108.0ms
# Half finish faster, half slower
```

### Optimistic (P30)
```python
# Use 30th percentile for aggressive scheduling
runtime_estimate = predictions["quantiles"]["0.3"]  # 97.8ms
# 70% will take longer, but maximizes throughput
```

### Adaptive
```python
# Choose based on system load
if system_load < 0.5:
    # Low load: optimistic
    runtime_estimate = predictions["quantiles"]["0.3"]
elif system_load < 0.8:
    # Medium load: balanced
    runtime_estimate = predictions["quantiles"]["0.5"]
else:
    # High load: conservative
    runtime_estimate = predictions["quantiles"]["0.9"]
```

## Implementation Details

### Code Location

File: `collect_training_data.py`, lines 912-923

```python
# Load training config if provided, or use defaults
training_config = None
if args.training_config:
    with open(args.training_config, 'r') as f:
        training_config = json.load(f)
else:
    # Use default training config with custom quantiles
    # Default quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    # This provides fine-grained percentile predictions across the distribution
    training_config = {
        'quantiles': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    }
```

### Training Behavior

- **Quantile Type Only**: These quantiles only apply when training with `--prediction-types quantile`
- **Expect Error Type**: Ignores quantiles (trains single expectation value)
- **Both Types**: Each type gets trained separately with appropriate configuration

### Model Storage

The trained model stores the quantile configuration:

```python
model_key = f"{model_id}__{software_name}-{software_version}__{hardware_name}__quantile"
# Model metadata includes: quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
```

## Verification

After training, verify quantiles are configured correctly:

```bash
# Check predictor models endpoint
curl http://localhost:8002/models | jq '.models[] | select(.prediction_type == "quantile")'
```

Expected output includes:
```json
{
  "model_id": "llama-7b",
  "prediction_type": "quantile",
  "samples_count": 1471,
  "metadata": {
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
  }
}
```

## Performance Considerations

**Training Time:**
- 10 quantiles vs 4 default: ~25% more training time
- Still completes in < 10 seconds for 1,471 samples

**Memory:**
- Neural network output layer: 10 neurons vs 4
- Negligible memory increase (~2KB)

**Inference:**
- Prediction time: Same (single forward pass)
- Response size: ~2x larger (more quantiles in JSON)

## Recommended Use Cases

✅ **Use Enhanced Quantiles (10 points) When:**
- Building scheduling systems that need fine-grained percentile control
- Analyzing runtime distribution in detail
- Need flexible percentile selection (e.g., adaptive strategies)
- Modeling complex workloads with varying latency requirements

✅ **Use Default Quantiles (4 points) When:**
- Quick prototyping
- Primary concern is tail latency (P90, P95, P99)
- Limited training data (< 50 samples)
- Simple monitoring applications

## Related Documentation

- [Predictor README](../../predictor/README_FOR_LLM.md) - Quantile regression details
- [COLLECT_TRAINING_DATA_README.md](COLLECT_TRAINING_DATA_README.md) - Training data collection guide
- [FIX_SUMMARY.md](FIX_SUMMARY.md) - Complete fix documentation

---

**Last Updated:** 2025-11-16
**Version:** 1.0
**Author:** SwarmPilot Team
