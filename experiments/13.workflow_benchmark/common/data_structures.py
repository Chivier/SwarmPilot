"""
Data structures and enums for workflow state management.

This module provides unified data structures supporting both Text2Video (Type 1)
and Deep Research (Type 2) workflow patterns with thread-safe state tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class WorkflowType(Enum):
    """Workflow type classification."""
    TEXT2VIDEO = "text2video"  # Linear workflow: A1→A2→B (with B loops)
    DEEP_RESEARCH = "deep_research"  # Fan-out/fan-in: A→n×B1→n×B2→Merge


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowState:
    """
    Unified workflow state tracker supporting both workflow types.

    Thread Safety:
    - Uses dict length operations for completion tracking (atomic)
    - Individual field updates should be protected by external locks
    - Helper methods are read-only and thread-safe

    Text2Video Pattern (A1→A2→B with loops):
    - Uses: a1_result, a2_result, b_loop_count, max_b_loops
    - B task can iterate 1-4 times before completion

    Deep Research Pattern (A→n×B1→n×B2→Merge):
    - Uses: fanout_count, b1_task_ids, b2_task_ids, b1/b2_complete_times
    - Tracks parallel B1/B2 tasks and synchronizes merge
    """
    workflow_id: str
    workflow_type: WorkflowType

    # Core timestamps
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_update: Optional[float] = None

    # ========================================================================
    # Text2Video Workflow Fields (A1→A2→B)
    # ========================================================================

    # A1 task (caption → positive prompt)
    a1_task_id: Optional[str] = None
    a1_submit_time: Optional[float] = None
    a1_complete_time: Optional[float] = None
    a1_result: Optional[str] = None  # Positive prompt

    # A2 task (positive prompt → negative prompt)
    a2_task_id: Optional[str] = None
    a2_submit_time: Optional[float] = None
    a2_complete_time: Optional[float] = None
    a2_result: Optional[str] = None  # Negative prompt

    # B task loop support (1-4 iterations)
    b_task_ids: List[str] = field(default_factory=list)
    b_submit_times: List[float] = field(default_factory=list)
    b_complete_times: List[float] = field(default_factory=list)
    b_loop_count: int = 0  # Current iteration count
    max_b_loops: int = 1  # Maximum allowed iterations (1-4)

    # ========================================================================
    # Deep Research Workflow Fields (A→n×B1→n×B2→Merge)
    # ========================================================================

    # A task (initial query → research directions)
    a_task_id: Optional[str] = None
    a_submit_time: Optional[float] = None
    a_complete_time: Optional[float] = None
    a_result: Optional[str] = None

    # Fanout configuration
    fanout_count: int = 0  # Number of parallel B1/B2 tasks

    # B1 tasks (parallel research tasks)
    b1_task_ids: List[str] = field(default_factory=list)
    b1_submit_times: Dict[str, float] = field(default_factory=dict)
    b1_complete_times: Dict[str, float] = field(default_factory=dict)
    b1_results: Dict[str, str] = field(default_factory=dict)
    all_b1_complete_time: Optional[float] = None

    # B2 tasks (refinement tasks, 1:1 with B1)
    b2_task_ids: List[str] = field(default_factory=list)
    b2_submit_times: Dict[str, float] = field(default_factory=dict)
    b2_complete_times: Dict[str, float] = field(default_factory=dict)
    b2_results: Dict[str, str] = field(default_factory=dict)
    all_b2_complete_time: Optional[float] = None

    # Merge task (aggregates all B1+B2 results)
    merge_task_id: Optional[str] = None
    merge_submit_time: Optional[float] = None
    merge_complete_time: Optional[float] = None
    merge_initiated: bool = False

    # ========================================================================
    # Common Fields
    # ========================================================================

    is_warmup: bool = False
    is_target_for_stats: bool = True  # For continuous mode statistics

    # ========================================================================
    # Text2Video Helper Methods
    # ========================================================================

    def should_continue_b_loop(self) -> bool:
        """
        Check if B loop should continue (Text2Video).

        Returns:
            True if current iteration count < max iterations
        """
        return self.b_loop_count < self.max_b_loops

    def mark_b_iteration_complete(self, complete_time: float):
        """
        Mark current B iteration as complete (Text2Video).

        Args:
            complete_time: Completion timestamp
        """
        self.b_complete_times.append(complete_time)

    def prepare_next_b_iteration(self) -> bool:
        """
        Prepare for next B iteration (Text2Video).

        Returns:
            True if should continue, False if max iterations reached
        """
        if self.should_continue_b_loop():
            self.b_loop_count += 1
            return True
        return False

    # ========================================================================
    # Deep Research Helper Methods
    # ========================================================================

    def are_all_b1_tasks_complete(self) -> bool:
        """
        Check if all B1 tasks are complete (Deep Research).

        Uses atomic dict length operation for thread safety.

        Returns:
            True if all B1 tasks have completed
        """
        return len(self.b1_complete_times) >= self.fanout_count

    def are_all_b2_tasks_complete(self) -> bool:
        """
        Check if all B2 tasks are complete (Deep Research).

        Uses atomic dict length operation for thread safety.

        Returns:
            True if all B2 tasks have completed
        """
        return len(self.b2_complete_times) >= self.fanout_count

    def mark_b1_task_complete(self, b1_task_id: str, complete_time: float):
        """
        Mark a B1 task as complete (Deep Research).

        Thread-safe: Uses dict insertion which is more atomic than counter increment.

        Args:
            b1_task_id: ID of the B1 task
            complete_time: Completion timestamp
        """
        if b1_task_id not in self.b1_complete_times:
            self.b1_complete_times[b1_task_id] = complete_time

            # Update all_b1_complete_time if this is the last B1 task
            if self.are_all_b1_tasks_complete():
                self.all_b1_complete_time = max(self.b1_complete_times.values())

    def mark_b2_task_complete(self, b2_task_id: str, complete_time: float):
        """
        Mark a B2 task as complete (Deep Research).

        Thread-safe: Uses dict insertion which is more atomic than counter increment.

        Args:
            b2_task_id: ID of the B2 task
            complete_time: Completion timestamp
        """
        if b2_task_id not in self.b2_complete_times:
            self.b2_complete_times[b2_task_id] = complete_time

            # Update all_b2_complete_time if this is the last B2 task
            if self.are_all_b2_tasks_complete():
                self.all_b2_complete_time = max(self.b2_complete_times.values())

    # ========================================================================
    # Unified Helper Methods
    # ========================================================================

    def is_complete(self) -> bool:
        """
        Check if workflow is complete (works for both types).

        Returns:
            True if workflow has completed based on its type
        """
        if self.workflow_type == WorkflowType.TEXT2VIDEO:
            # Text2Video: complete when B loops are done
            return (
                len(self.b_complete_times) > 0 and
                len(self.b_complete_times) >= self.max_b_loops
            )
        elif self.workflow_type == WorkflowType.DEEP_RESEARCH:
            # Deep Research: complete when merge task is done
            return self.merge_complete_time is not None
        else:
            return False

    def get_total_duration(self) -> Optional[float]:
        """
        Calculate total workflow duration in seconds.

        Returns:
            Duration from start to end, or None if not yet complete
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def get_status_summary(self) -> Dict[str, any]:
        """
        Get a summary of the current workflow state.

        Returns:
            Dictionary with status information
        """
        summary = {
            'workflow_id': self.workflow_id,
            'workflow_type': self.workflow_type.value,
            'is_complete': self.is_complete(),
            'duration': self.get_total_duration(),
        }

        if self.workflow_type == WorkflowType.TEXT2VIDEO:
            summary.update({
                'b_loop_count': self.b_loop_count,
                'max_b_loops': self.max_b_loops,
                'b_iterations_complete': len(self.b_complete_times),
            })
        elif self.workflow_type == WorkflowType.DEEP_RESEARCH:
            summary.update({
                'fanout_count': self.fanout_count,
                'b1_complete': len(self.b1_complete_times),
                'b2_complete': len(self.b2_complete_times),
                'all_b1_done': self.are_all_b1_tasks_complete(),
                'all_b2_done': self.are_all_b2_tasks_complete(),
                'merge_initiated': self.merge_initiated,
            })

        return summary
