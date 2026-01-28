"""E2E Multi-Scheduler Experiment Framework.

This package provides end-to-end testing for the multi-scheduler architecture
(PYLET-024). It validates that:
1. Each model gets its own scheduler, registered with planner
2. Instances are deployed via planner -> PyLet
3. Requests are routed to correct scheduler based on model_id
4. Multiple schedulers handle concurrent requests

Usage:
    python -m tests.integration.e2e_multi_scheduler.run_experiment \
        --experiment sleep_model --workers 10 --qps 5 --duration 30
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

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "PlannerConfig",
    "PyLetClusterConfig",
    "MultiSchedulerOrchestrator",
    "MultiSchedulerWorkloadConfig",
    "MultiSchedulerWorkloadGenerator",
    "MultiSchedulerWorkloadResult",
]
