# Unified Workflow Benchmark Framework

A unified, reusable framework for multi-model workflow experiments supporting Text2Video and Deep Research patterns.

## Architecture Overview

```
experiments/13.workflow_benchmark/
├── common/                           # Shared infrastructure (✅ Complete)
│   ├── base_classes.py              # BaseTaskSubmitter, BaseTaskReceiver
│   ├── rate_limiter.py              # Token bucket rate limiting
│   ├── data_structures.py           # WorkflowState, enums
│   ├── distribution.py              # Generic distribution classes
│   ├── metrics_collector.py         # Metrics tracking (✅ Complete)
│   └── utils.py                     # JSON, logging, HTTP utilities
│
├── type1_text2video/                # Text2Video workflow (A1→A2→B)
│   ├── config.py                    # Configuration (✅ Complete)
│   ├── workflow_data.py             # Data structures
│   ├── submitters.py                # A1, A2, B task submitters
│   ├── receivers.py                 # A1, A2, B task receivers
│   ├── configs/                     # Parameter distribution configs
│   │   ├── frame_count_*.json       # Frame count distribution configs
│   │   └── max_b_loops_*.json       # Max B loops distribution configs
│   ├── simulation/
│   │   └── test_workflow_sim.py     # Simulation mode runner
│   └── real/
│       ├── test_workflow_real.py    # Real cluster mode
│       ├── start_real_service.sh    # Service launcher
│       └── manual_deploy_planner.sh # Model deployment
│
├── type2_deep_research/             # Deep Research (A→n×B1→n×B2→Merge)
│   ├── config.py
│   ├── workflow_data.py
│   ├── fanout_distribution.py       # Fanout distribution classes
│   ├── submitters.py                # A, B1, B2, Merge submitters
│   ├── receivers.py                 # A, B1, B2, Merge receivers
│   ├── configs/                     # Fanout distribution configs
│   │   ├── fanout_static.json
│   │   ├── fanout_uniform.json
│   │   ├── fanout_two_peak.json
│   │   └── fanout_four_peak.json
│   ├── simulation/
│   │   └── test_workflow_sim.py
│   └── real/
│       ├── test_workflow_real.py
│       └── deployment scripts
│
├── service_management/              # Service orchestration
│   ├── service_launcher.py          # Start/stop services
│   ├── deployment_manager.py        # Model deployment
│   ├── health_checker.py            # Health monitoring
│   └── resource_binder.py           # CPU/GPU binding
│
├── tools/                           # Testing & validation
│   ├── experiment_runner.py         # Unified experiment runner
│   ├── metrics_validator.py         # Metrics comparison
│   ├── workload_generator.py        # Traffic pattern generation
│   └── cli.py                       # Command-line interface
│
├── configs/                         # Configuration templates
│   ├── text2video_sim_config.yaml
│   ├── text2video_real_config.yaml
│   ├── deep_research_sim_config.yaml
│   └── deep_research_real_config.yaml
│
├── docs/                            # Documentation
│   ├── API.md                       # API reference
│   ├── MIGRATION.md                 # Migration guide
│   ├── ARCHITECTURE.md              # System architecture
│   └── TROUBLESHOOTING.md           # Common issues
│
└── tests/                           # Test suite
    ├── test_common_infrastructure.py (✅ Complete)
    ├── test_text2video.py
    ├── test_deep_research.py
    └── test_integration.py
```

## Completed Components (Tasks 11-12)

### ✅ Task 11: Core Infrastructure
- **BaseTaskSubmitter**: Abstract base for task submission with rate limiting
- **BaseTaskReceiver**: Abstract base for WebSocket result receiving
- **RateLimiter**: Token bucket algorithm with Poisson distribution
- **WorkflowState**: Unified state tracking for both workflow types
- **Utilities**: JSON, logging, HTTP, timestamp helpers
- **Tests**: 20+ comprehensive unit tests with >90% coverage

### ✅ Task 12: Metrics Collection
- **MetricsCollector**: Thread-safe metrics tracking
- **TaskMetrics**: Task-level performance metrics
- **WorkflowMetrics**: Workflow-level statistics
- **Export formats**: JSON, CSV, text reports
- **Statistics**: p50/p95/p99 latencies, QPS, success rates

## Implementation Status

| Task | Component | Status | Files |
|------|-----------|--------|-------|
| 11 | Core Infrastructure | ✅ Complete | common/{base_classes,rate_limiter,data_structures,utils}.py |
| 12 | Metrics Collection | ✅ Complete | common/metrics_collector.py |
| 13 | Text2Video Simulation | ✅ Complete | type1_text2video/{config,workflow_data,submitters,receivers,simulation}.py |
| 14 | Deep Research Simulation | ✅ Complete | type2_deep_research/{config,workflow_data,submitters,receivers,simulation}.py |
| 15 | Service Management | ✅ Complete | service_management/{service_launcher,deployment_manager,health_checker,resource_binder}.py |
| 16 | Real Cluster Mode | ✅ Complete | */real/*, updated configs and submitters for both workflows |
| 17 | Testing Tools | ✅ Complete | tools/experiment_runner.py, tools/cli.py |
| 18 | Documentation | ✅ Complete | docs/{QUICKSTART,API,MIGRATION,TROUBLESHOOTING}.md, configs/*.yaml |

## Quick Start

### 5-Minute Quick Start

See [QUICKSTART.md](docs/QUICKSTART.md) for comprehensive getting started guide with:
- **Method 1**: Using the CLI tool (easiest)
- **Method 2**: Direct Python scripts
- **Method 3**: Real cluster mode

**Quick Example**:
```bash
cd experiments/13.workflow_benchmark

# Run Text2Video simulation (1 minute test)
python tools/cli.py run-text2video-sim --duration 60 --num-workflows 120

# Run Deep Research simulation (1 minute test)
python tools/cli.py run-deep-research-sim --duration 60 --num-workflows 60

# View results
cat output/metrics.json | python -m json.tool
```

### Running Text2Video Simulation

```bash
# Method 1: CLI (recommended)
python tools/cli.py run-text2video-sim --qps 2.0 --duration 300 --num-workflows 600

# Method 2: Direct Python with environment variables
export QPS=2.0 DURATION=300 NUM_WORKFLOWS=600
python type1_text2video/simulation/test_workflow_sim.py

# Method 3: With custom frame_count and max_b_loops distributions
python -m type1_text2video.simulation.test_workflow_sim \
    --num-workflows 100 \
    --frame-count-config type1_text2video/configs/frame_count_two_peak.json \
    --max-b-loops-config type1_text2video/configs/max_b_loops_uniform.json \
    --frame-count-seed 42 --max-b-loops-seed 42
```

### Running Deep Research Simulation

```bash
# Method 1: CLI (recommended)
python tools/cli.py run-deep-research-sim --qps 1.0 --duration 600 --num-workflows 600

# Method 2: Direct Python with environment variables
export QPS=1.0 DURATION=600 NUM_WORKFLOWS=600 FANOUT_COUNT=3
python type2_deep_research/simulation/test_workflow_sim.py

# Method 3: With custom fanout distribution (see Fanout Distribution section)
python tools/cli.py run-deep-research-sim --qps 1.0 --num-workflows 100 \
    --fanout-config type2_deep_research/configs/fanout_two_peak.json \
    --fanout-seed 42
```

## Deep Research Fanout Distribution

The Deep Research workflow supports configurable fanout distributions for B1/B2 task parallelism. This allows simulating realistic workload patterns where different workflows may have varying degrees of parallelism.

### Supported Distribution Types

| Type | Description | Use Case |
|------|-------------|----------|
| `static` | Fixed fanout value for all workflows | Baseline testing, consistent load |
| `uniform` | Uniform distribution between min/max | Random variation testing |
| `two_peak` | Bimodal distribution with two Gaussian peaks | Simulating "small" vs "large" queries |
| `four_peak` | Four Gaussian peaks | Complex workload patterns |

### Configuration File Format

Configuration files are JSON with the following schemas:

**Static Distribution** (`fanout_static.json`):
```json
{
    "type": "static",
    "value": 4
}
```

**Uniform Distribution** (`fanout_uniform.json`):
```json
{
    "type": "uniform",
    "min": 2,
    "max": 8
}
```

**Two-Peak Distribution** (`fanout_two_peak.json`):
```json
{
    "type": "two_peak",
    "peaks": [
        {"mean": 3, "std": 0.5, "weight": 1.0},
        {"mean": 8, "std": 1.0, "weight": 1.0}
    ],
    "min": 1,
    "max": 12
}
```

**Four-Peak Distribution** (`fanout_four_peak.json`):
```json
{
    "type": "four_peak",
    "peaks": [
        {"mean": 2, "std": 0.3, "weight": 1.0},
        {"mean": 5, "std": 0.5, "weight": 1.5},
        {"mean": 8, "std": 0.5, "weight": 1.5},
        {"mean": 12, "std": 1.0, "weight": 1.0}
    ],
    "min": 1,
    "max": 15
}
```

### Peak Configuration

For `two_peak` and `four_peak` distributions, each peak is a Gaussian (normal) distribution with:

| Field | Type | Description |
|-------|------|-------------|
| `mean` | float | Center of the peak (expected fanout value) |
| `std` | float | Standard deviation (spread of the peak) |
| `weight` | float | Relative probability of selecting this peak |

The sampling process:
1. Select a peak based on normalized weights
2. Sample from that peak's Gaussian distribution
3. Clamp to `[min, max]` range and round to integer

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--fanout` | Static fanout value (ignored if `--fanout-config` is specified) | 4 |
| `--fanout-config` | Path to JSON config file for fanout distribution | None |
| `--fanout-seed` | Random seed for reproducible fanout sampling | None |

### Example Usage

```bash
# Static fanout (default behavior)
python -m type2_deep_research.simulation.test_workflow_sim --fanout 4

# Uniform distribution (2-8 parallel tasks)
python -m type2_deep_research.simulation.test_workflow_sim \
    --fanout-config type2_deep_research/configs/fanout_uniform.json

# Two-peak distribution with reproducible sampling
python -m type2_deep_research.simulation.test_workflow_sim \
    --fanout-config type2_deep_research/configs/fanout_two_peak.json \
    --fanout-seed 42

# Using CLI tool
python tools/cli.py run-deep-research-sim --num-workflows 100 \
    --fanout-config type2_deep_research/configs/fanout_four_peak.json
```

### Example Config Files

Pre-built example configurations are available in `type2_deep_research/configs/`:

```
type2_deep_research/configs/
├── fanout_static.json      # Fixed value: 4
├── fanout_uniform.json     # Uniform: [2, 8]
├── fanout_two_peak.json    # Bimodal: peaks at 3 and 8
└── fanout_four_peak.json   # Four peaks: 2, 5, 8, 12
```

## Text2Video Parameter Distribution

The Text2Video workflow supports configurable distributions for `frame_count` (video frame count) and `max_b_loops` (B task iterations per workflow). This allows simulating realistic workload patterns with varying video complexity and iteration counts.

### Supported Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `frame_count` | Number of frames for video generation | 16 |
| `max_b_loops` | Maximum B task iterations per workflow | 3 |

Both parameters support the same four distribution types as Deep Research fanout:
- `static`: Fixed value for all workflows
- `uniform`: Uniform distribution between min/max
- `two_peak`: Bimodal distribution with two Gaussian peaks
- `four_peak`: Four Gaussian peaks

### Configuration File Format

The configuration files use the same JSON schema as fanout distributions.

**Frame Count - Static** (`frame_count_static.json`):
```json
{
    "type": "static",
    "value": 16
}
```

**Frame Count - Two-Peak** (`frame_count_two_peak.json`):
```json
{
    "type": "two_peak",
    "peaks": [
        {"mean": 8, "std": 1.0, "weight": 1.0},
        {"mean": 24, "std": 2.0, "weight": 1.0}
    ],
    "min": 4,
    "max": 32
}
```

**Max B Loops - Uniform** (`max_b_loops_uniform.json`):
```json
{
    "type": "uniform",
    "min": 1,
    "max": 5
}
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--frame-count` | Static frame count (ignored if `--frame-count-config` is specified) | 16 |
| `--frame-count-config` | Path to JSON config file for frame_count distribution | None |
| `--frame-count-seed` | Random seed for reproducible frame_count sampling | None |
| `--max-b-loops` | Static max B loops (ignored if `--max-b-loops-config` is specified) | 3 |
| `--max-b-loops-config` | Path to JSON config file for max_b_loops distribution | None |
| `--max-b-loops-seed` | Random seed for reproducible max_b_loops sampling | None |

### Example Usage

```bash
# Static values (default behavior)
python -m type1_text2video.simulation.test_workflow_sim --frame-count 16 --max-b-loops 3

# Uniform max_b_loops distribution (1-5 iterations)
python -m type1_text2video.simulation.test_workflow_sim \
    --max-b-loops-config type1_text2video/configs/max_b_loops_uniform.json

# Two-peak frame count with reproducible sampling
python -m type1_text2video.simulation.test_workflow_sim \
    --frame-count-config type1_text2video/configs/frame_count_two_peak.json \
    --frame-count-seed 42

# Combined distributions for both parameters
python -m type1_text2video.simulation.test_workflow_sim \
    --num-workflows 100 \
    --frame-count-config type1_text2video/configs/frame_count_two_peak.json \
    --max-b-loops-config type1_text2video/configs/max_b_loops_uniform.json \
    --frame-count-seed 42 --max-b-loops-seed 42
```

### Example Config Files

Pre-built example configurations are available in `type1_text2video/configs/`:

```
type1_text2video/configs/
├── frame_count_static.json      # Fixed value: 16
├── frame_count_uniform.json     # Uniform: [8, 32]
├── frame_count_two_peak.json    # Bimodal: peaks at 8 and 24
├── frame_count_four_peak.json   # Four peaks: 8, 16, 24, 32
├── max_b_loops_static.json      # Fixed value: 3
├── max_b_loops_uniform.json     # Uniform: [1, 5]
└── max_b_loops_two_peak.json    # Bimodal: peaks at 2 and 4
```

## Workflow Submission Order

The Text2Video workflow supports configurable submission order for workflows. This is useful when you want to control the order in which workflows with different `max_b_loops` values are submitted.

### Supported Submission Orders

| Order | Description | Requirements |
|-------|-------------|--------------|
| `sequential` | Workflows submitted in generation order (0, 1, 2, ...) | Default, works with any distribution |
| `alternating-peaks` | Odd-indexed peaks forward, then even-indexed peaks backward | Requires `two_peak` or `four_peak` distribution |

### Alternating Peaks Mode

When using `alternating-peaks` submission order with a multi-peak distribution, workflows are reordered based on which peak their `max_b_loops` value was sampled from:

1. **Odd-indexed peaks** (Peak 1, Peak 3, ...) are submitted first in forward order
2. **Even-indexed peaks** (Peak N, Peak N-2, ...) are submitted in reverse order

**Example for 4 peaks** (with means 30, 60, 120, 200):
- Peak 1 (mean=30) → submitted first (forward)
- Peak 3 (mean=120) → submitted second (forward)
- Peak 4 (mean=200) → submitted third (backward)
- Peak 2 (mean=60) → submitted last (backward)

Result order: **Peak1 → Peak3 → Peak4 → Peak2**

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--submission-order` | Workflow submission order: `sequential` or `alternating-peaks` | `sequential` |

### Example Usage

```bash
# Default sequential order (works with any distribution)
python -m type1_text2video.simulation.test_workflow_sim \
    --max-b-loops-config type1_text2video/configs/max_b_loops_two_peak.json \
    --num-workflows 100

# Alternating peaks order (requires multi-peak distribution)
python -m type1_text2video.simulation.test_workflow_sim \
    --max-b-loops-config type1_text2video/configs/max_b_loops_four_peak.json \
    --num-workflows 100 \
    --submission-order alternating-peaks
```

### Validation

When `--submission-order alternating-peaks` is specified, the system validates that:
- A `max_b_loops_config` is specified (not using static `--max-b-loops`)
- The distribution type is `two_peak` or `four_peak`

If validation fails, a clear error message is displayed:
```
ValueError: alternating-peaks submission order requires a multi-peak distribution
(two_peak or four_peak), but got 'uniform'.
Please use --max-b-loops-config with a two_peak or four_peak distribution.
```

### Use Case

The alternating peaks submission order is useful for experiments where you want to:
- Separate workflows with different computational costs
- Test scheduler behavior under varying load patterns
- Simulate burst patterns where similar workflows arrive together

## Implementation Guide

### Task 13: Text2Video Workflow

**Pattern**: Linear A1→A2→B with B-loop support (1-4 iterations)

**Key Files to Implement**:

1. `workflow_data.py`:
```python
@dataclass
class Text2VideoWorkflowData:
    workflow_id: str
    caption: str
    a1_result: Optional[str] = None  # Positive prompt
    a2_result: Optional[str] = None  # Negative prompt
    b_loop_count: int = 0
    max_b_loops: int = 4

def load_captions(filepath: str) -> List[str]:
    with open(filepath) as f:
        return json.load(f)
```

2. `submitters.py`:
- `A1TaskSubmitter(BaseTaskSubmitter)`: Submits caption-based A1 tasks
- `A2TaskSubmitter(BaseTaskSubmitter)`: Submits A2 using A1 result
- `BTaskSubmitter(BaseTaskSubmitter)`: Submits B with loop control

3. `receivers.py`:
- `A1TaskReceiver(BaseTaskReceiver)`: Extracts positive prompt → triggers A2
- `A2TaskReceiver(BaseTaskReceiver)`: Extracts negative prompt → triggers B
- `BTaskReceiver(BaseTaskReceiver)`: Loop logic → re-submit or complete

### Task 14: Deep Research Workflow

**Pattern**: Fan-out/fan-in A→n×B1→n×B2→Merge

**Key Components**:
- A receiver: Fans out to n B1 tasks
- B1 receiver: 1:1 triggers B2 tasks
- B2 receiver: Synchronizes → triggers Merge when all complete
- Merge receiver: Aggregates results

### Task 15: Service Management

**Components**:
- `ServiceLauncher`: Start/stop services with CPU/GPU binding
- `DeploymentManager`: Parallel model deployment with retry
- `HealthChecker`: Service health monitoring
- `ResourceBinder`: CPU core and GPU allocation

### Task 16: Real Cluster Mode

**Extensions**:
- Use real model IDs (llm_service_small_model, t2vid, etc.)
- Add platform metadata and token estimation
- Configure actual scheduler/predictor/planner endpoints
- Implement deployment scripts for distributed execution

### Task 17: Testing & Validation

**Tools**:
- `ExperimentRunner`: Unified runner for all experiment types
- `MetricsValidator`: Compare against original implementations
- `WorkloadGenerator`: Various traffic patterns (constant, Poisson, bursty, ramp)
- `CLI`: Command-line interface for automation

### Task 18: Documentation

**Deliverables**:
- API documentation for all public classes
- Migration guide from original scripts
- Configuration examples (YAML templates)
- Architecture diagrams
- Troubleshooting guide

## Design Principles

1. **Reusability**: Base classes eliminate code duplication
2. **Type Safety**: Unified WorkflowState supports both patterns
3. **Thread Safety**: Locks protect all shared state
4. **Observability**: Comprehensive metrics collection
5. **Testability**: Mock-friendly abstractions
6. **Flexibility**: Config-driven with environment overrides

## Performance Targets

- **QPS Accuracy**: Within ±5% of target
- **Thread Safety**: Support 10+ concurrent threads
- **Memory**: <100MB for 10K workflows
- **CPU Overhead**: <1% for metrics collection
- **Test Coverage**: >85% for experiment code

## Reference Implementations

- **Text2Video**: `experiments/03.Exp4.Text2Video/test_dynamic_workflow_sim.py`
- **Deep Research**: `experiments/07.Exp2.Deep_Research_Migration_Test/test_dynamic_workflow.py`

## Service Management

### Quick Start Scripts

The `scripts/` directory contains adapted service management scripts that work from the current experiment directory:

```bash
# Start all services (simulation mode)
./scripts/start_all_services.sh

# Start with custom configuration
./scripts/start_all_services.sh 10 6 llm_service_small_model t2vid

# Stop all services
./scripts/stop_all_services.sh
```

### What Gets Started

- **Predictor** (port 8101): Performance prediction service
- **Planner** (port 8202): Auto-optimization planner
- **Scheduler A** (port 8100): Scheduler for Group A models
- **Scheduler B** (port 8200): Scheduler for Group B models
- **Instances**: N1 + N2 model instances (configurable)

### Service Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `start_all_services.sh` | Start all services | `./scripts/start_all_services.sh [N1] [N2] [MODEL_A] [MODEL_B]` |
| `stop_all_services.sh` | Stop all services | `./scripts/stop_all_services.sh` |
| `deploy_models_local.sh` | Deploy models to instances | `./scripts/deploy_models_local.sh --model-id-a MODEL_A --n1 N1` |

See [scripts/README.md](scripts/README.md) for detailed documentation.

### Typical Workflow

```bash
# 1. Start services
cd experiments/13.workflow_benchmark
./scripts/start_all_services.sh 4 2 sleep_model_a sleep_model_b

# 2. Run experiment
python tools/cli.py run-text2video-sim --duration 300 --num-workflows 600

# 3. View results
cat output/metrics.json | python -m json.tool

# 4. Stop services
./scripts/stop_all_services.sh
```

### Logs and Debugging

- **Log Directory**: `logs/` (created automatically)
- **Service Logs**: `logs/predictor.log`, `logs/scheduler-a.log`, etc.
- **PID Files**: `logs/*.pid` for process tracking

**View logs**:
```bash
tail -f logs/predictor.log
tail -f logs/scheduler-a.log
```

**Check service health**:
```bash
curl http://localhost:8101/health  # Predictor
curl http://localhost:8100/health  # Scheduler A
curl http://localhost:8200/health  # Scheduler B
```

## Next Steps

1. Complete Task 13 subtasks (workflow_data.py, submitters.py, receivers.py, main script)
2. Implement Task 14 (Deep Research) following same pattern
3. Add service management (Task 15)
4. Extend to real cluster mode (Task 16)
5. Build testing tools (Task 17)
6. Write comprehensive documentation (Task 18)

## Contributing

When implementing components:
1. Follow the existing patterns in common/
2. Add comprehensive docstrings
3. Write unit tests for new functionality
4. Update this README with progress
5. Reference original implementations for behavioral consistency
