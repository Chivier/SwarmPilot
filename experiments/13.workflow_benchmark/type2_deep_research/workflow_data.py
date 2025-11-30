"""Deep Research workflow data structures and utilities."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Dataset Loading for Real Mode (aligned with Exp07)
# =============================================================================

# Path to dataset.jsonl (copied from experiments/07.Exp2.Deep_Research_Real/data/)
DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.jsonl"


def load_dataset() -> List[dict]:
    """
    Load all entries from dataset.jsonl.

    Returns:
        List of dataset entries, each containing:
        - boot: str - A task prompt
        - queries: List[dict] - B1/B2 task data (each has 'input' key)
        - summary: str - Merge task prompt
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_FILE}\n"
            f"Please copy from experiments/07.Exp2.Deep_Research_Real/data/dataset.jsonl"
        )

    dataset = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                dataset.append(entry)
    return dataset


def get_query_inputs(queries: List[dict]) -> List[str]:
    """
    Extract query input strings from queries list.

    Args:
        queries: List of query dicts from dataset entry

    Returns:
        List of query input strings
    """
    inputs = []
    for q in queries:
        if isinstance(q, dict):
            inputs.append(q.get("input", str(q)))
        else:
            inputs.append(str(q))
    return inputs


# =============================================================================
# Exp07-aligned Configuration (hardcoded, not user-configurable)
# =============================================================================

# Fanout distribution: random choice from 5-15 (matching dr_summary_dict keys in Exp07)
# This is NOT user-configurable to ensure reproducibility with Exp07
AVAILABLE_FANOUTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Sleep time distribution parameters (aligned with experiments/07.Exp2)
# These define normal distributions for each task type:
#   - Task A (boot): mean=30s, std=2s
#   - Task Merge (summary): mean=25s, std=2s
#   - Task B1 (query): mean=3s, std=0.3s
#   - Task B2 (criteria): mean=2s, std=0.2s
SLEEP_TIME_DISTRIBUTIONS = {
    "A": {"mean": 30.0, "std": 2.0},
    "merge": {"mean": 25.0, "std": 2.0},
    "B1": {"mean": 3.0, "std": 0.3},
    "B2": {"mean": 2.0, "std": 0.2},
}


def generate_sleep_times(
    num_workflows: int,
    fanout_counts: List[int],
    seed: int = 42
) -> Tuple[List[float], List[List[float]], List[List[float]], List[float]]:
    """
    Generate sleep times from normal distributions (aligned with experiment 07).

    The distributions are:
        - Task A: N(30, 2) seconds
        - Task Merge: N(25, 2) seconds
        - Task B1: N(3, 0.3) seconds
        - Task B2: N(2, 0.2) seconds

    Args:
        num_workflows: Number of workflows to generate
        fanout_counts: List of fanout counts for each workflow
        seed: Random seed for reproducibility

    Returns:
        Tuple of (a_times, b1_times_list, b2_times_list, merge_times)
        - a_times: List of A task sleep times
        - b1_times_list: List of lists, each inner list contains B1 sleep times for one workflow
        - b2_times_list: List of lists, each inner list contains B2 sleep times for one workflow
        - merge_times: List of Merge task sleep times
    """
    rng = np.random.default_rng(seed)

    # Generate A task times
    a_params = SLEEP_TIME_DISTRIBUTIONS["A"]
    a_times = rng.normal(a_params["mean"], a_params["std"], size=num_workflows).tolist()
    # Ensure positive values (clip to minimum 0.1 seconds)
    a_times = [max(0.1, t) for t in a_times]

    # Generate Merge task times
    merge_params = SLEEP_TIME_DISTRIBUTIONS["merge"]
    merge_times = rng.normal(merge_params["mean"], merge_params["std"], size=num_workflows).tolist()
    merge_times = [max(0.1, t) for t in merge_times]

    # Generate B1 and B2 task times for each workflow
    b1_params = SLEEP_TIME_DISTRIBUTIONS["B1"]
    b2_params = SLEEP_TIME_DISTRIBUTIONS["B2"]

    b1_times_list = []
    b2_times_list = []

    for fanout in fanout_counts:
        # Generate B1 times for this workflow
        b1_times = rng.normal(b1_params["mean"], b1_params["std"], size=fanout).tolist()
        b1_times = [max(0.1, t) for t in b1_times]
        b1_times_list.append(b1_times)

        # Generate B2 times for this workflow
        b2_times = rng.normal(b2_params["mean"], b2_params["std"], size=fanout).tolist()
        b2_times = [max(0.1, t) for t in b2_times]
        b2_times_list.append(b2_times)

    return a_times, b1_times_list, b2_times_list, merge_times


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

    # Additional fields for real mode (aligned with Exp07's dataset.jsonl)
    topic: str = ""  # Research topic for real mode (legacy, not used in aligned mode)
    max_tokens: int = 512  # Max tokens for LLM tasks (legacy, not used in aligned mode)
    sentences: List[str] = field(default_factory=list)  # Sentences for each task (legacy)

    # Real mode fields aligned with Exp07 dataset.jsonl
    boot_sentence: Optional[str] = None  # A task sentence (from dataset["boot"])
    query_inputs: List[str] = field(default_factory=list)  # B1/B2 task sentences (from dataset["queries"])
    summary_sentence: Optional[str] = None  # Merge task sentence (from dataset["summary"])

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

    For simulation mode:
        - Fanout: random choice from [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        - Task A: N(30, 2) seconds
        - Task Merge: N(25, 2) seconds
        - Task B1: N(3, 0.3) seconds
        - Task B2: N(2, 0.2) seconds

    For real mode (aligned with Exp07):
        - Data loaded from dataset.jsonl
        - Fanout determined by len(queries) in each dataset entry
        - boot/queries/summary text used directly

    Args:
        config: DeepResearchConfig instance
        seed: Random seed for reproducibility

    Returns:
        List of pre-generated DeepResearchWorkflowData instances
    """
    import random

    # Set random seed for reproducibility
    random.seed(seed)

    workflows = []

    if config.mode == "real":
        # Real mode: load from dataset.jsonl (aligned with Exp07)
        dataset = load_dataset()

        # Sample dataset entries WITH REPLACEMENT
        sampled_entries = random.choices(dataset, k=config.num_workflows)

        for i, entry in enumerate(sampled_entries):
            # Extract data from dataset entry (aligned with Exp07)
            boot_sentence = entry.get("boot", "")
            queries = entry.get("queries", [])
            summary_sentence = entry.get("summary", "")

            # Fanout is determined by number of queries in the entry
            fanout_count = len(queries)

            # Extract query input strings
            query_inputs = get_query_inputs(queries)

            workflow = DeepResearchWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                fanout_count=fanout_count,
                strategy="pending",  # Will be set per-strategy run
                is_warmup=(i < getattr(config, 'num_warmup', 0)),
                # Real mode fields (aligned with Exp07)
                boot_sentence=boot_sentence,
                query_inputs=query_inputs,
                summary_sentence=summary_sentence,
            )

            workflows.append(workflow)

    else:
        # Simulation mode: use normal distributions
        # First pass: determine fanout counts for all workflows
        # NOTE: Using hardcoded AVAILABLE_FANOUTS (5-15) to match Exp07
        fanout_counts = []
        for i in range(config.num_workflows):
            fanout_count = random.choice(AVAILABLE_FANOUTS)
            fanout_counts.append(fanout_count)

        # Pre-generate all sleep times using normal distributions
        a_times, b1_times_list, b2_times_list, merge_times = generate_sleep_times(
            num_workflows=config.num_workflows,
            fanout_counts=fanout_counts,
            seed=seed
        )

        # Second pass: create workflow data with pre-generated values
        for i in range(config.num_workflows):
            fanout_count = fanout_counts[i]

            workflow = DeepResearchWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                fanout_count=fanout_count,
                strategy="pending",  # Will be set per-strategy run
                is_warmup=(i < getattr(config, 'num_warmup', 0)),
                # Simulation mode fields
                a_sleep_time=a_times[i],
                b1_sleep_times=b1_times_list[i],
                b2_sleep_times=b2_times_list[i],
                merge_sleep_time=merge_times[i],
            )

            workflows.append(workflow)

    return workflows
