"""Deep Research workflow data structures and utilities."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DeepResearchWorkflowData:
    """Workflow-level data for Deep Research pattern (A→n×B1→n×B2→Merge)."""

    workflow_id: str
    fanout_count: int  # Number of B1/B2 tasks to spawn

    # Results from A task (populated by receiver)
    a_result: Optional[str] = None

    # B1 and B2 task tracking
    b1_task_ids: List[str] = field(default_factory=list)  # List of B1 task IDs
    b2_task_ids: List[str] = field(default_factory=list)  # List of B2 task IDs

    # Completion tracking using dicts (atomic operations)
    # Key = task_id, Value = completion timestamp
    b1_complete_times: Dict[str, float] = field(default_factory=dict)
    b2_complete_times: Dict[str, float] = field(default_factory=dict)

    # Merge task tracking
    merge_task_id: Optional[str] = None
    merge_complete_time: Optional[float] = None

    # Timing tracking (compatible with Exp07)
    a_submit_time: Optional[float] = None
    a_complete_time: Optional[float] = None
    b1_submit_times: List[float] = field(default_factory=list)
    b2_submit_times: List[float] = field(default_factory=list)
    merge_submit_time: Optional[float] = None
    workflow_complete_time: Optional[float] = None

    # Additional fields for simulation mode
    a_sleep_time: Optional[float] = None    # A task sleep time
    b1_sleep_times: List[float] = field(default_factory=list)  # B1 task sleep times
    b2_sleep_times: List[float] = field(default_factory=list)  # B2 task sleep times
    merge_sleep_time: Optional[float] = None  # Merge task sleep time

    # Additional fields for real mode
    topic: str = ""  # Research topic for real mode
    max_tokens: int = 512  # Max tokens for LLM tasks
    sentences: List[str] = field(default_factory=list)  # Sentences for each task

    # Additional tracking fields
    strategy: str = "probabilistic"  # Scheduling strategy name
    is_warmup: bool = False    # Whether this is a warmup workflow

    def all_b1_complete(self) -> bool:
        """Check if all B1 tasks are complete."""
        return len(self.b1_complete_times) >= self.fanout_count

    def all_b2_complete(self) -> bool:
        """Check if all B2 tasks are complete."""
        return len(self.b2_complete_times) >= self.fanout_count

    def is_complete(self) -> bool:
        """Check if entire workflow is complete (Merge done)."""
        return self.merge_complete_time is not None

    def get_b2_for_b1(self, b1_task_id: str) -> Optional[str]:
        """
        Get the corresponding B2 task ID for a B1 task.

        Uses 1:1 mapping based on list position.

        Args:
            b1_task_id: B1 task ID

        Returns:
            Corresponding B2 task ID, or None if not found
        """
        try:
            index = self.b1_task_ids.index(b1_task_id)
            if index < len(self.b2_task_ids):
                return self.b2_task_ids[index]
        except ValueError:
            pass
        return None


def pre_generate_workflows(
    config,
    seed: int = 42
) -> List["DeepResearchWorkflowData"]:
    """Pre-generate all workflow data before strategy testing.

    This ensures all strategies use identical workflow data (same sleep times,
    fanout counts, etc.) for fair comparison.

    Args:
        config: DeepResearchConfig instance
        seed: Random seed for reproducibility

    Returns:
        List of pre-generated DeepResearchWorkflowData instances
    """
    import random

    # Set random seed for reproducibility
    random.seed(seed)

    # Create fanout sampler for distribution-based fanout
    fanout_sampler = config.create_fanout_sampler()

    workflows = []
    for i in range(config.num_workflows):
        # Sample fanout from distribution (or use static value)
        # Convert to int since fanout must be an integer
        fanout_count = int(fanout_sampler.sample())

        workflow = DeepResearchWorkflowData(
            workflow_id=f"workflow-{i:04d}",
            fanout_count=fanout_count,
            strategy="pending",  # Will be set per-strategy run
            is_warmup=(i < getattr(config, 'num_warmup', 0))
        )

        # Pre-generate sleep times for simulation mode
        # Use uniform distribution [1, max_sleep_time_seconds] consistent with type1
        if config.mode == "simulation":
            min_sleep = 1.0  # Minimum 1 second (consistent with type1's MIN_SLEEP_TIME_MS)
            max_sleep = config.max_sleep_time_seconds
            workflow.a_sleep_time = random.uniform(min_sleep, max_sleep)
            workflow.b1_sleep_times = [
                random.uniform(min_sleep, max_sleep)
                for _ in range(fanout_count)
            ]
            workflow.b2_sleep_times = [
                random.uniform(min_sleep, max_sleep)
                for _ in range(fanout_count)
            ]
            workflow.merge_sleep_time = random.uniform(min_sleep, max_sleep)

        # Pre-generate topic for real mode
        if config.mode == "real":
            workflow.topic = f"Deep research topic {i+1}: Advanced computing architectures"

        workflows.append(workflow)

    return workflows
