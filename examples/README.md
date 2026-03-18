# SwarmPilot Examples

## Deployment Examples

| Example | Scenario | Schedulers | Planner | Models |
|---------|----------|-----------|---------|--------|
| [single_model/](single_model/) | Single model, multi-replica | 1 | No | Qwen3-8B-VL |
| [multi_model_direct/](multi_model_direct/) | Multi-model, manual schedulers | N (one per model) | No | Qwen3-8B-VL, Llama-3.1-8B |
| [multi_model_planner/](multi_model_planner/) | Multi-model, planner-managed | N (one per model) | Yes | Qwen3-8B-VL, Llama-3.1-8B |

## Other Examples

| Example | Topic |
|---------|-------|
| [predictor/](predictor/) | ML-based runtime prediction (library + HTTP API) |
| [planner/](planner/) | Optimization plan generation |

## Key Architecture Rule

> Each Scheduler process serves exactly one model. For multi-model deployments, run one Scheduler per model.
> See [Architecture docs](../docs/ARCHITECTURE.md) for details.
