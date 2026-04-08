# SwarmPilot Examples

| # | Example | Topic | Difficulty |
|---|---------|-------|------------|
| 1 | [1.single_model/](1.single_model/) | Single model, multi-replica deployment | Beginner |
| 2 | [2.predictor/](2.predictor/) | ML-based runtime prediction (library + HTTP API) | Beginner → Intermediate |
| 3 | [3.multi_model_direct/](3.multi_model_direct/) | Multi-model, manual schedulers (no Planner) | Intermediate |
| 4 | [4.planner/](4.planner/) | Optimization plan generation | Intermediate |
| 5 | [5.multi_model_planner/](5.multi_model_planner/) | Multi-model, Planner-managed deployment | Intermediate / Advanced |
| 6 | [6.predictor_training/](6.predictor_training/) | Real-world data collection & predictor training pipeline | Advanced |

## Benchmark Gallery

| Gallery | Dataset | Approach |
|---------|---------|----------|
| [gallery/mcp_atlas/](gallery/mcp_atlas/) | MCP-Atlas | Latency replay (dual-model) |
| [gallery/dataclaw/](gallery/dataclaw/) | Dataclaw | Latency replay (dual-model) |
| [gallery/swebench/](gallery/swebench/) | SWE-bench Pro | Multi-turn agent evaluation |

See [gallery/README.md](gallery/README.md) for details.

## Key Architecture Rule

> Each Scheduler process serves exactly one model. For multi-model deployments, run one Scheduler per model.
> See [Architecture docs](../docs/ARCHITECTURE.md) for details.
