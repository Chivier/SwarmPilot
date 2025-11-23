# Unified Workflow Benchmark Framework

A unified, reusable framework for multi-model workflow experiments supporting Text2Video and Deep Research patterns.

## Architecture Overview

```
experiments/13.workflow_benchmark/
├── common/                           # Shared infrastructure (✅ Complete)
│   ├── base_classes.py              # BaseTaskSubmitter, BaseTaskReceiver
│   ├── rate_limiter.py              # Token bucket rate limiting
│   ├── data_structures.py           # WorkflowState, enums
│   ├── metrics_collector.py         # Metrics tracking (✅ Complete)
│   └── utils.py                     # JSON, logging, HTTP utilities
│
├── type1_text2video/                # Text2Video workflow (A1→A2→B)
│   ├── config.py                    # Configuration (✅ Complete)
│   ├── workflow_data.py             # Data structures
│   ├── submitters.py                # A1, A2, B task submitters
│   ├── receivers.py                 # A1, A2, B task receivers
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
│   ├── submitters.py                # A, B1, B2, Merge submitters
│   ├── receivers.py                 # A, B1, B2, Merge receivers
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
```

### Running Deep Research Simulation

```bash
# Method 1: CLI (recommended)
python tools/cli.py run-deep-research-sim --qps 1.0 --duration 600 --num-workflows 600

# Method 2: Direct Python with environment variables
export QPS=1.0 DURATION=600 NUM_WORKFLOWS=600 FANOUT_COUNT=3
python type2_deep_research/simulation/test_workflow_sim.py
```

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
