# Gallery: Real-Model Benchmark Examples

End-to-end examples of using SwarmPilot as the inference backend for LLM benchmark workloads. Each gallery is **fully self-contained** — it includes its own code, scripts, and configuration with no cross-dependencies.

## Galleries

| Gallery | Dataset | Approach | Key Metrics |
|---------|---------|----------|-------------|
| [mcp_atlas/](mcp_atlas/) | [ScaleAI/MCP-Atlas](https://huggingface.co/datasets/ScaleAI/MCP-Atlas) | Latency replay (dual-model) | p50/p90/p99 latency |
| [dataclaw/](dataclaw/) | [peteromallet/dataclaw-peteromallet](https://huggingface.co/datasets/peteromallet/dataclaw-peteromallet) | Latency replay (dual-model) | p50/p90/p99 latency |
| [swebench/](swebench/) | [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) | Multi-turn agent evaluation | pass/fail score |
| [replay/](replay/) | All of the above | Multi-dataset replay framework + real-model benchmark | p50/p90/p99 latency |

## Getting Started

Each gallery directory contains its own README with full deployment instructions. `cd` into any gallery and follow its README.
