"""E2E Multi-Scheduler Experiment Framework.

This package provides end-to-end testing for the multi-scheduler architecture
(PYLET-024). It validates that:
1. Each model gets its own scheduler, registered with planner
2. Instances are deployed via planner -> PyLet
3. Requests are routed to correct scheduler based on model_id
4. Multiple schedulers handle concurrent requests

It also provides scheduler isolation testing (PYLET-026):
- Single scheduler without planner
- Direct subprocess instance launch (no PyLet)
- Tests scheduler behavior in isolation

Usage:
    # Multi-scheduler experiment
    python -m tests.integration.e2e_multi_scheduler.run_experiment \
        --experiment sleep_model --workers 10 --qps 5 --duration 30

    # Scheduler isolation experiment
    python -m tests.integration.e2e_multi_scheduler.run_scheduler_only_experiment \
        --instances 4 --qps 2 --duration 60
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    PlannerConfig,
    PyLetClusterConfig,
)
from .orchestrator import MultiSchedulerOrchestrator
from .workload_generator import (
    MultiSchedulerWorkloadConfig,
    MultiSchedulerWorkloadGenerator,
    MultiSchedulerWorkloadResult,
)
from .experiments.scheduler_only_config import (
    SchedulerOnlyConfig,
    create_scheduler_only_config,
)
from .scheduler_only_orchestrator import SchedulerOnlyOrchestrator

__all__ = [
    # Multi-scheduler components
    "ExperimentConfig",
    "ModelConfig",
    "PlannerConfig",
    "PyLetClusterConfig",
    "MultiSchedulerOrchestrator",
    "MultiSchedulerWorkloadConfig",
    "MultiSchedulerWorkloadGenerator",
    "MultiSchedulerWorkloadResult",
    # Scheduler-only components
    "SchedulerOnlyConfig",
    "create_scheduler_only_config",
    "SchedulerOnlyOrchestrator",
]
