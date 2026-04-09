# MLP Predictor: Best Parameters & Minimum Training Samples

## Models Evaluated

| Model | Type | Runtime Data |
|-------|------|-------------|
| Qwen3-VL-8B-Instruct | Vision-language | `runtime_qwen8b.json` |
| Qwen3-Next-80B-A3B-Instruct | MoE | `runtime_qwen80b.json` |

## Best MLP Architecture

| Model | Hidden Layers | Train Samples | MAE (ms) | MAPE (%) | Q90 Coverage | Q95 Coverage |
|-------|--------------|---------------|----------|----------|-------------|-------------|
| **8B** | `[256, 128, 64]` | 560 | 2296.9 | 109.3 | 97% | 97% |
| **80B** | `[32]` | 550 | 692.4 | 35.8 | 88% | 99% |

The 80B model is far easier to predict — a single-layer `[32]` MLP suffices.
The 8B vision-language model requires a deeper 3-layer network due to
higher input-to-latency variance from multimodal processing.

## Minimum Samples Required

Using a 10%-above-best-MAE threshold to define "sufficient" accuracy:

| Model | Architecture | Min Samples | MAE at Threshold (ms) |
|-------|-------------|-------------|----------------------|
| **8B** | `[64, 32]` | **25** | ≤2526.6 |
| **8B** | `[128]` | 30 | ≤2526.6 |
| **8B** | `[256, 128, 64]` (best) | 75 | ≤2526.6 |
| **80B** | `[32]` (best) | **60** | ≤761.7 |
| **80B** | `[64]` | 100 | ≤761.7 |
| **80B** | `[64, 32]` | 145 | ≤761.7 |

**Takeaway:** 25-60 profiling samples are enough to train a usable predictor.
The best architecture for each model also reaches threshold quickly (75 / 60).

## Practical Recommendation

- **8B-class models:** use `[64, 32]` with ≥200 samples for stable accuracy.
  Collect 25+ samples for a quick bootstrap, 200+ for production.
- **80B-class models:** use `[32]` with ≥60 samples.
  Larger architectures offer <3% MAE improvement but need 3-5x more data.
- **Training config:** 200 epochs, lr=0.001, quantiles `[0.5, 0.9, 0.95, 0.99]`.
