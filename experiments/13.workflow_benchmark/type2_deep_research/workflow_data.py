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
