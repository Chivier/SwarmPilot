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
| 15 | Service Management | 📋 Next | service_management/* |
| 16 | Real Cluster Mode | 📋 Planned | */real/* |
| 17 | Testing Tools | 📋 Planned | tools/* |
| 18 | Documentation | 📋 Planned | docs/*, configs/* |

## Quick Start

### Running Text2Video Simulation

```bash
cd experiments/13.workflow_benchmark

# Configure experiment
export QPS=2.0
export DURATION=300
export NUM_WORKFLOWS=600

# Run simulation
python -m type1_text2video.simulation.test_workflow_sim
```

### Running Deep Research Simulation

```bash
# Configure experiment
export FANOUT_COUNT=3
export QPS=1.0

# Run simulation
python -m type2_deep_research.simulation.test_workflow_sim
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
