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
    peak_index: Optional[int] = None  # Which peak this workflow's max_b_loops was sampled from (0-based)

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


def pre_generate_workflows(
    config,
    captions: List[str],
    seed: int = 42,
    run_prefix: str = "",
    submission_order: str = "sequential"
) -> List["Text2VideoWorkflowData"]:
    """Pre-generate all workflow data before strategy testing.

    This ensures all strategies use identical workflow data (same sleep times,
    frame counts, max_b_loops, etc.) for fair comparison.

    Frame count source priority:
    1. If --frame-count-config is specified: use the configured distribution
    2. Otherwise: sample from benchmark dataset (captions_10k.jsonl)

    Args:
        config: Text2VideoConfig instance
        captions: List of captions to use
        seed: Random seed for reproducibility
        run_prefix: Optional prefix for workflow IDs
        submission_order: "sequential" or "alternating-peaks".
                         If "alternating-peaks", workflows are reordered so
                         odd-indexed peaks are submitted forward and
                         even-indexed peaks are submitted backward.

    Returns:
        List of pre-generated Text2VideoWorkflowData instances
    """
    import random

    # Set random seed for reproducibility
    random.seed(seed)

    # Determine frame_count source:
    # - If frame_count_config is specified, use config.sample_frame_count() (from distribution sampler)
    # - Otherwise, use data_loader.sample_frame_count() (from benchmark dataset)
    use_config_frame_count = config.frame_count_config is not None

    # Validate submission_order and check distribution compatibility
    if submission_order == "alternating-peaks":
        # Ensure sampler is initialized
        if config._max_b_loops_sampler is None:
            config.sample_max_b_loops()  # This initializes the sampler

        dist = config._max_b_loops_sampler.distribution
        dist_type = dist.to_dict().get("type", "unknown")

        # Only two_peak and four_peak distributions are supported
        valid_types = ("two_peak", "four_peak")
        if dist_type not in valid_types:
            raise ValueError(
                f"alternating-peaks submission order requires a multi-peak distribution "
                f"(two_peak or four_peak), but got '{dist_type}'. "
                f"Please use --max-b-loops-config with a two_peak or four_peak distribution."
            )
        is_multi_peak = True
    else:
        is_multi_peak = False
        # Check if distribution supports peak tracking (for future use)
        if config._max_b_loops_sampler is None:
            config.sample_max_b_loops()  # This initializes the sampler
        dist = config._max_b_loops_sampler.distribution
        if hasattr(dist, 'sample_with_peak'):
            is_multi_peak = True

    workflows = []
    for i in range(config.num_workflows):
        # Sample max_b_loops with or without peak tracking
        if is_multi_peak and submission_order == "alternating-peaks":
            sampled_value, peak_index = dist.sample_with_peak()
            sampled_max_b_loops = int(sampled_value)
        else:
            sampled_max_b_loops = config.sample_max_b_loops()
            peak_index = None

        # Sample frame_count based on configuration
        if use_config_frame_count:
            # User specified --frame-count-config, use the configured distribution
            sampled_frame_count = config.sample_frame_count()
        else:
            # Default: sample from benchmark dataset for realistic distribution
            sampled_frame_count = config.data_loader.sample_frame_count()

        # Generate workflow_id with optional run_prefix for ID collision avoidance
        workflow_id = f"workflow-{run_prefix}-{i:04d}" if run_prefix else f"workflow-{i:04d}"

        workflow = Text2VideoWorkflowData(
            workflow_id=workflow_id,
            caption=captions[i % len(captions)],
            max_b_loops=sampled_max_b_loops,
            strategy="pending",  # Will be set per-strategy run
            frame_count=sampled_frame_count,
            max_tokens=getattr(config, 'max_tokens', 512),
            is_warmup=(i < getattr(config, 'num_warmup', 0)),
            peak_index=peak_index  # Track which peak this workflow's max_b_loops came from
        )

        # Pre-generate sleep times for simulation mode only
        # In simulation mode, data_loader provides runtime estimates
        if config.mode == "simulation":
            # Sample from real benchmark data (data_loader must exist)
            # A1 and A2: independent samples from LLM benchmark runtimes
            workflow.a1_sleep_time = config.data_loader.sample_llm_runtime_ms() / 1000.0
            workflow.a2_sleep_time = config.data_loader.sample_llm_runtime_ms() / 1000.0

            # B: use the already-sampled frame_count to get runtime via regression
            workflow.b_sleep_time = config.data_loader.get_t2vid_runtime_ms(sampled_frame_count) / 1000.0

        workflows.append(workflow)

    # Reorder workflows if alternating-peaks mode is requested
    if submission_order == "alternating-peaks" and is_multi_peak:
        workflows = _reorder_alternating_peaks(workflows)

    return workflows


def _reorder_alternating_peaks(workflows: List["Text2VideoWorkflowData"]) -> List["Text2VideoWorkflowData"]:
    """Reorder workflows: odd peaks forward, even peaks backward.

    For N peaks (0-indexed internally, 1-indexed for user-facing naming):
    - Peak 0 ("Peak 1") and Peak 2 ("Peak 3") are "odd" peaks - submitted forward
    - Peak 1 ("Peak 2") and Peak 3 ("Peak 4") are "even" peaks - submitted backward

    Result order for 4 peaks: Peak1 → Peak3 → Peak4 → Peak2

    Args:
        workflows: List of workflows with peak_index set

    Returns:
        Reordered list of workflows
    """
    from collections import defaultdict

    # Group workflows by peak index
    by_peak = defaultdict(list)
    for w in workflows:
        if w.peak_index is not None:
            by_peak[w.peak_index].append(w)

    if not by_peak:
        return workflows  # No peak info, return unchanged

    num_peaks = max(by_peak.keys()) + 1

    # peak_index=0 -> "Peak 1" (odd), peak_index=1 -> "Peak 2" (even)
    # peak_index=2 -> "Peak 3" (odd), peak_index=3 -> "Peak 4" (even)
    odd_peaks = [i for i in range(num_peaks) if (i + 1) % 2 == 1]   # 0, 2 -> Peak 1, 3
    even_peaks = [i for i in range(num_peaks) if (i + 1) % 2 == 0]  # 1, 3 -> Peak 2, 4

    reordered = []

    # Odd peaks forward (Peak 1, Peak 3, ...)
    for peak_idx in sorted(odd_peaks):
        reordered.extend(by_peak.get(peak_idx, []))

    # Even peaks backward (Peak 4, Peak 2, ...)
    for peak_idx in sorted(even_peaks, reverse=True):
        reordered.extend(by_peak.get(peak_idx, []))

    return reordered


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
