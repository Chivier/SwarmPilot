# SWE-bench Agent Benchmark

Run multi-turn agent evaluations on [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) through a SwarmPilot cluster using the swarmbench framework.

Unlike the replay benchmarks (MCP-Atlas, Dataclaw) which measure scheduling latency, swarmbench evaluates **end-to-end agent performance**: the LLM reasons, calls tools, and iterates until the task is solved.

## Architecture

```
HuggingFace Dataset
    |
    v
[TaskLoader] --> Task (prompt + tools + ground_truth)
    |
    v
[AgentLoop]  (multi-turn: LLM → tool → LLM → ... → done)
    |
    +---> [OpenAI SDK] ---> SwarmPilot Scheduler :8000/v1/chat/completions
    |                              |
    |                              v
    |                         vLLM / Model Backend
    |
    +---> [ToolProvider] ---> local execution (bash, file ops, git)
    |
    v
[Evaluator] --> EvalResult (score, pass/fail)
    |
    v
[Reporter] --> JSON report + console summary
```

## Deployment

A single scheduler is sufficient (one model):

```bash
# Start a single scheduler
SCHEDULER_MODEL_ID=Qwen/Qwen3-Next-80B-A3B-Instruct \
sscheduler start --port 8000
```

## Usage

All commands run from within this directory (`gallery/swebench/`).

```bash

# Run SWE-bench Pro (live mode with real workspace)
python -m swarmbench.cli run \
  --dataset swe-bench-pro \
  --mode live \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --workspace /tmp/swebench-workspace \
  --limit 10 \
  --output ./output \
  --report ./report.json

# Run MCP-Atlas (real mode with MCP server)
python -m swarmbench.cli run \
  --dataset mcp-atlas \
  --mode real \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy

# Run Dataclaw (live mode with coding tools)
python -m swarmbench.cli run \
  --dataset dataclaw \
  --mode live \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --workspace /tmp/dataclaw-workspace

# Evaluate saved logs
python -m swarmbench.cli evaluate \
  --dataset swe-bench-pro \
  --output ./output \
  --report ./eval-report.json
```

## Datasets and Modes

| Dataset | Mode | Tools | Evaluation |
|---------|------|-------|------------|
| `swe-bench-pro` | `live` | bash, str_replace_editor, submit | Binary pass/fail (test suite) |
| `swe-bench-pro` | `dry-run` | Same, but mocked submission | Same |
| `mcp-atlas` | `real` | HTTP POST to MCP server | Per-claim LLM judging |
| `mcp-atlas` | `mock` | Replays gold trajectory | Same |
| `dataclaw` | `live` | bash, file read/write/edit, grep | Trajectory + response similarity |
| `dataclaw` | `trajectory` | No execution (comparison only) | Same |

## Key Difference from Replay Benchmarks

| Aspect | Replay (mcp_atlas/dataclaw) | Swarmbench (swebench) |
|--------|----------------------------|-----------------------|
| Purpose | Scheduling latency measurement | Agent capability evaluation |
| Client | Async, Poisson arrival | Sync, sequential turns |
| Request content | Real prompts from dataset | Model-generated (iterative) |
| Tool execution | None (prompt replay only) | Real (bash, files, git) |
| Metrics | p50/p90/p99 latency | pass/fail score |

## Dependencies

```
pydantic>=2.0
openai>=1.0
httpx
datasets
```
