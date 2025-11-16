# Training Data Collection Script

This script collects training data for the predictor model by executing LLM tasks and measuring execution times.

## Overview

The script:
1. Reads LLM execution tasks from `dataset.jsonl`
2. Executes each task on the LLM service (via instance API) and collects execution times
3. Transforms metadata according to scheduler's transformation mechanism:
   - Extracts `token_length` and `max_tokens` from tasks (as submitted to scheduler)
   - Appends hardware specifications (as extracted by predictor from `platform_info`)
4. Submits training data to the predictor service `/train` endpoint

## Prerequisites

1. **Instance service** running with LLM model loaded
2. **Predictor service** running and accessible
3. **Dataset file** (`dataset.jsonl`) with LLM task data

## Installation

Install required dependencies:

```bash
pip install httpx tqdm
```

## Usage

Basic usage:

```bash
python collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://localhost:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --hardware-name "NVIDIA H20"
```

### Parameters

- `--dataset`: Path to dataset.jsonl file (required)
- `--instance-url`: URL of the instance service (LLM service), default: `http://localhost:8001`
- `--predictor-url`: URL of the predictor service, default: `http://localhost:8002`
- `--model-id`: Model identifier for training (required)
- `--prediction-types`: Types of prediction models to train (`expect_error` and/or `quantile`), default: both `["expect_error", "quantile"]`. Can specify one or both.
- `--hardware-name`: Hardware name (e.g., `NVIDIA H20`). If not provided, will be auto-detected from system or instance service
- `--software-name`: Software name (e.g., `sglang`), default: `sglang`
- `--software-version`: Software version, default: `1.0.0`
- `--batch-size`: Number of samples to collect before submitting, default: `100`
- `--max-samples`: Maximum number of tasks to process (None for all), default: `None`
- `--training-config`: Path to training configuration JSON file (optional)

### Example: Training with Auto-Detected GPU

```bash
# Train both quantile and expect_error models (default)
python collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://192.168.1.10:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --batch-size 50 \
    --max-samples 500

# Train only quantile model
python collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://192.168.1.10:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --prediction-types quantile \
    --batch-size 50

# Train only expect_error model
python collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://192.168.1.10:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --prediction-types expect_error \
    --batch-size 50
```

### Example: Training with Explicit GPU Name

```bash
# Explicitly specify GPU name and train both model types
python collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://192.168.1.10:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --hardware-name "NVIDIA H20" \
    --prediction-types expect_error quantile \
    --batch-size 50
```

## Data Transformation

The script correctly transforms metadata as follows:

### User Metadata (as submitted to scheduler)
- `token_length`: Estimated token count from input sentence
- `max_tokens`: Maximum tokens to generate

### Hardware Specifications (automatically added)
Extracted from `platform_info.extract_gpu_specs()`, including:
- `cuda_cores`: Number of CUDA cores
- `tensor_cores`: Number of tensor cores
- `fp32_tflops`: FP32 TFLOPS
- `fp16_tflops`: FP16 TFLOPS
- `tensor_tflops`: Tensor core TFLOPS (FP16)
- `fp8_tensor_tflops`: FP8 tensor core TFLOPS
- `memory_gb`: GPU memory in GB
- `memory_bandwidth_gb_s`: Memory bandwidth in GB/s

### Training Sample Format

Each training sample contains:
```json
{
  "token_length": 256.0,
  "max_tokens": 512.0,
  "cuda_cores": 17920.0,
  "tensor_cores": 560.0,
  "fp32_tflops": 63.0,
  "fp16_tflops": 252.0,
  "tensor_tflops": 1230.0,
  "fp8_tensor_tflops": 2460.0,
  "memory_gb": 96.0,
  "memory_bandwidth_gb_s": 4000.0,
  "runtime_ms": 1234.5
}
```

## Dataset Format

The script expects `dataset.jsonl` with entries like:

```json
{
  "id": "entry-001",
  "boot": {
    "input": "Generate a query...",
    "max_tokens": 512
  },
  "queries": [
    {
      "input": "Query 1...",
      "max_tokens": 256
    }
  ],
  "summary": {
    "input": "Summarize...",
    "max_tokens": 512
  }
}
```

## Notes

- The script processes tasks in batches and submits training data when enough samples are collected (default: 100)
- At least 10 samples are required for training (predictor requirement)
- **Multiple Model Types**: By default, the script trains both `quantile` and `expect_error` models using the same collected samples. You can specify which model types to train using the `--prediction-types` parameter
- The same training samples are used for all requested model types, ensuring consistency
- Failed tasks are skipped and logged
- **GPU Detection**: The script automatically detects GPU name from:
  1. Instance service (if `--instance-url` is provided and service exposes GPU info)
  2. System via `nvidia-smi` command (if available)
  3. System via `pynvml` library (if installed)
  4. Falls back to "CPU" if no GPU is detected
- Hardware specifications are automatically extracted from the detected GPU name

## Troubleshooting

### Error: "Could not extract hardware specs"
- Ensure the hardware name matches a recognized GPU model (e.g., "NVIDIA H20", "NVIDIA H100")
- Check that `predictor/src/utils/hardware_perf_info.py` contains the GPU specifications

### Error: "Training submission failed"
- Verify predictor service is running and accessible
- Check that at least 10 samples have been collected
- Ensure feature names match the trained model's expected features

### Error: "Task execution failed"
- Verify instance service is running and model is loaded
- Check instance URL is correct
- Ensure LLM service `/inference` endpoint is accessible

