"""
Type5 OOD Recovery Experiment

This experiment validates that when task runtime distribution changes (OOD),
the system can recover when the predictor successfully re-trains.

Three-phase pattern:
- Phase 1: Correct prediction (warmup)
- Phase 2: OOD - wrong prediction (distribution shifted)
- Phase 3: Recovery - predictor re-trained, correct prediction
"""

from .config import OODRecoveryConfig
from .task_data import OODTaskData, load_base_sleep_times, pre_generate_tasks
from .submitter import OODTaskSubmitter
from .receiver import OODTaskReceiver

__all__ = [
    "OODRecoveryConfig",
    "OODTaskData",
    "load_base_sleep_times",
    "pre_generate_tasks",
    "OODTaskSubmitter",
    "OODTaskReceiver",
]
