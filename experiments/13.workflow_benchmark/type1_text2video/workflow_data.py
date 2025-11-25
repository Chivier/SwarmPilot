"""Text2Video workflow data structures and utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Text2VideoWorkflowData:
    """Workflow-level data for Text2Video pattern (A1→A2→B with loops)."""

    workflow_id: str
    caption: str
    max_b_loops: int = 4  # Maximum B task iterations (1-4)

    # Results from each stage (populated by receivers)
    a1_result: Optional[str] = None  # Positive prompt from A1
    a2_result: Optional[str] = None  # Negative prompt from A2

    # Timing tracking
    a1_submit_time: Optional[float] = None
    a1_complete_time: Optional[float] = None
    a2_submit_time: Optional[float] = None
    a2_complete_time: Optional[float] = None

    # B-loop tracking
    b_loop_count: int = 0  # Current iteration count (increments on each B submission)
    b_submit_times: List[float] = field(default_factory=list)  # Submission timestamps
    b_complete_times: List[float] = field(default_factory=list)  # Completion timestamps
    workflow_complete_time: Optional[float] = None

    # Additional fields for simulation mode
    a1_sleep_time: Optional[float] = None  # A1 task sleep time (simulation)
    a2_sleep_time: Optional[float] = None  # A2 task sleep time (simulation)
    b_sleep_time: Optional[float] = None   # B task sleep time (simulation)

    # Additional fields for real mode
    frame_count: int = 16  # Video frame count
    max_tokens: int = 512  # Max tokens for LLM tasks

    # Additional fields for tracking
    strategy: str = "probabilistic"  # Scheduling strategy name
    is_warmup: bool = False    # Whether this is a warmup workflow

    def should_continue_b_loop(self) -> bool:
        """Check if we should submit another B task."""
        return self.b_loop_count < self.max_b_loops

    def is_complete(self) -> bool:
        """Check if workflow is complete (all B iterations done)."""
        return len(self.b_complete_times) >= self.max_b_loops

    def get_workflow_time(self) -> Optional[float]:
        """Get total workflow time from A1 submit to last B complete."""
        if self.a1_submit_time and self.workflow_complete_time:
            return self.workflow_complete_time - self.a1_submit_time
        return None


def load_captions(filepath: str) -> List[str]:
    """
    Load captions from JSON file.

    The JSON file should contain either:
    - A list of strings: ["caption1", "caption2", ...]
    - An object with 'captions' key: {"captions": ["caption1", "caption2", ...]}

    Args:
        filepath: Path to JSON file containing captions

    Returns:
        List of caption strings

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If file format is invalid
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Caption file not found: {filepath}")

    with open(path) as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, list):
        captions = data
    elif isinstance(data, dict) and 'captions' in data:
        captions = data['captions']
    else:
        raise ValueError(
            f"Invalid caption file format. Expected list or dict with 'captions' key, "
            f"got {type(data).__name__}"
        )

    if not captions:
        raise ValueError("Caption file is empty")

    return captions
