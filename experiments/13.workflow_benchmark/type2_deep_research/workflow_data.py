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

# Timing data files for simulation mode (real execution times from Exp07)
TIMING_BOOT_FILE = DATA_DIR / "dr_boot.json"      # A task times
TIMING_QUERY_FILE = DATA_DIR / "dr_query.json"    # B1 task times (query, full inference)
TIMING_CRITERIA_FILE = DATA_DIR / "dr_criteria.json"  # B2 task times (criteria, max_tokens=1)
TIMING_SUMMARY_FILE = DATA_DIR / "dr_summary_dict.json"  # Merge task times (by fanout)


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
# Timing Data Loading for Simulation Mode (real execution times from Exp07)
# =============================================================================

# Cached timing data (loaded once)
_TIMING_DATA_CACHE: Dict[str, any] = {}


def load_timing_data() -> Dict[str, any]:
    """
    Load real execution timing data from Exp07 for simulation mode.

    Returns:
        Dict with keys:
        - 'boot_times': List[float] - A task execution times (seconds)
        - 'query_times': List[float] - B1 task execution times (seconds, full inference)
        - 'criteria_times': List[float] - B2 task execution times (seconds, max_tokens=1)
        - 'summary_times': Dict[str, List[float]] - Merge task times by fanout count
    """
    global _TIMING_DATA_CACHE

    if _TIMING_DATA_CACHE:
        return _TIMING_DATA_CACHE

    # Load A task (boot) times
    if not TIMING_BOOT_FILE.exists():
        raise FileNotFoundError(
            f"Timing data file not found: {TIMING_BOOT_FILE}\n"
            f"Please copy from experiments/07.Exp2.Deep_Research_Real/data/dr_boot.json"
        )
    with open(TIMING_BOOT_FILE, 'r') as f:
        boot_times = json.load(f)

    # Load B1 task (query) times - full inference
    if not TIMING_QUERY_FILE.exists():
        raise FileNotFoundError(
            f"Timing data file not found: {TIMING_QUERY_FILE}\n"
            f"Please copy from experiments/07.Exp2.Deep_Research_Real/data/dr_query.json"
        )
    with open(TIMING_QUERY_FILE, 'r') as f:
        query_times = json.load(f)

    # Load B2 task (criteria) times - max_tokens=1, receives B1 output
    if not TIMING_CRITERIA_FILE.exists():
        raise FileNotFoundError(
            f"Timing data file not found: {TIMING_CRITERIA_FILE}\n"
            f"Please copy from experiments/07.Exp2.Deep_Research_Real/data/dr_criteria.json"
        )
    with open(TIMING_CRITERIA_FILE, 'r') as f:
        criteria_times = json.load(f)

    # Load Merge task (summary) times - keyed by fanout count
    if not TIMING_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Timing data file not found: {TIMING_SUMMARY_FILE}\n"
            f"Please copy from experiments/07.Exp2.Deep_Research_Real/data/dr_summary_dict.json"
        )
    with open(TIMING_SUMMARY_FILE, 'r') as f:
        summary_times = json.load(f)

    _TIMING_DATA_CACHE = {
        'boot_times': boot_times,
        'query_times': query_times,
        'criteria_times': criteria_times,
        'summary_times': summary_times,
    }

    return _TIMING_DATA_CACHE


def sample_timing_from_real_data(
    task_type: str,
    count: int = 1,
    fanout: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> List[float]:
    """
    Sample execution times from real timing data.

    Args:
        task_type: One of 'A', 'B1', 'B2', 'merge'
        count: Number of samples to draw
        fanout: Fanout count (required for 'merge' task type)
        rng: NumPy random generator for reproducibility

    Returns:
        List of sampled execution times (seconds)

    Notes:
        - B1 (query): Full LLM inference, avg ~2.7s
        - B2 (criteria): Receives B1 output with max_tokens=1, avg ~0.048s
    """
    timing_data = load_timing_data()

    if rng is None:
        rng = np.random.default_rng()

    if task_type == 'A':
        pool = timing_data['boot_times']
    elif task_type == 'B1':
        # B1 uses query times (full inference)
        pool = timing_data['query_times']
    elif task_type == 'B2':
        # B2 uses criteria times (max_tokens=1, much shorter)
        pool = timing_data['criteria_times']
    elif task_type == 'merge':
        if fanout is None:
            raise ValueError("fanout is required for merge task type")
        # summary_times is keyed by fanout count as string
        fanout_key = str(fanout)
        if fanout_key in timing_data['summary_times']:
            pool = timing_data['summary_times'][fanout_key]
        else:
            # Fallback: use closest available fanout
            available_fanouts = [int(k) for k in timing_data['summary_times'].keys()]
            closest = min(available_fanouts, key=lambda x: abs(x - fanout))
            pool = timing_data['summary_times'][str(closest)]
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Sample with replacement
    indices = rng.integers(0, len(pool), size=count)
    return [pool[i] for i in indices]


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

    # =========================================================================
    # Loop tracking (workflow-level loops: A→B1→B2→Merge repeated max_loops times)
    # Similar to Type1's b_loop_count pattern
    # =========================================================================
    loop_count: int = 0  # Current loop iteration (1, 2, 3...), 0 = not started yet
    max_loops: int = 1   # Total loop iterations for this workflow (sampled from distribution)

    # Per-loop timing tracking (lists, one element per loop iteration)
    loop_a_submit_times: List[float] = field(default_factory=list)
    loop_a_complete_times: List[float] = field(default_factory=list)
    loop_merge_complete_times: List[float] = field(default_factory=list)

    # Pre-generated data for each loop iteration
    loop_fanouts: List[int] = field(default_factory=list)  # Fanout count per loop

    # Simulation mode: pre-generated sleep times per loop
    loop_a_sleep_times: List[float] = field(default_factory=list)
    loop_b1_sleep_times: List[List[float]] = field(default_factory=list)
    loop_b2_sleep_times: List[List[float]] = field(default_factory=list)
    loop_merge_sleep_times: List[float] = field(default_factory=list)

    # Real mode: pre-generated input data per loop
    loop_boot_sentences: List[str] = field(default_factory=list)
    loop_query_inputs: List[List[str]] = field(default_factory=list)
    loop_summary_sentences: List[str] = field(default_factory=list)

    def should_continue_loop(self) -> bool:
        """Check if more loop iterations are needed (called after Merge completes)."""
        return self.loop_count < self.max_loops

    def is_all_loops_complete(self) -> bool:
        """Check if all loop iterations are complete."""
        return len(self.loop_merge_complete_times) >= self.max_loops

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

    def prepare_for_next_loop(self) -> bool:
        """Prepare workflow data for the next loop iteration.

        Called after Merge completes when should_continue_loop() returns True.
        This method:
        1. Increments loop_count
        2. Copies pre-generated data for the new loop to current fields
        3. Resets B1/B2 task tracking for the new loop

        Returns:
            True if successfully prepared for next loop, False if no more loops
        """
        if not self.should_continue_loop():
            return False

        # Increment loop count (1-indexed)
        self.loop_count += 1
        loop_idx = self.loop_count - 1  # 0-indexed for array access

        # Update fanout_count for this loop
        if loop_idx < len(self.loop_fanouts):
            self.fanout_count = self.loop_fanouts[loop_idx]

        # Update simulation mode fields (sleep times)
        if loop_idx < len(self.loop_a_sleep_times):
            self.a_sleep_time = self.loop_a_sleep_times[loop_idx]
        if loop_idx < len(self.loop_b1_sleep_times):
            self.b1_sleep_times = self.loop_b1_sleep_times[loop_idx]
        if loop_idx < len(self.loop_b2_sleep_times):
            self.b2_sleep_times = self.loop_b2_sleep_times[loop_idx]
        if loop_idx < len(self.loop_merge_sleep_times):
            self.merge_sleep_time = self.loop_merge_sleep_times[loop_idx]

        # Update real mode fields (input data)
        if loop_idx < len(self.loop_boot_sentences):
            self.boot_sentence = self.loop_boot_sentences[loop_idx]
        if loop_idx < len(self.loop_query_inputs):
            self.query_inputs = self.loop_query_inputs[loop_idx]
        if loop_idx < len(self.loop_summary_sentences):
            self.summary_sentence = self.loop_summary_sentences[loop_idx]

        # Reset task tracking for the new loop
        self.a_result = None
        self.b1_task_ids = []
        self.b2_task_ids = []
        self.b1_complete_times = {}
        self.b2_complete_times = {}
        self.merge_task_id = None
        self.merge_complete_time = None

        # Reset timing for this loop iteration
        self.a_submit_time = None
        self.a_complete_time = None
        self.b1_submit_times = []
        self.b2_submit_times = []
        self.merge_submit_time = None

        return True


def _generate_loop_sleep_times(
    num_loops: int,
    loop_fanouts: List[int],
    rng: np.random.Generator
) -> Tuple[List[float], List[List[float]], List[List[float]], List[float]]:
    """Generate sleep times for all loop iterations of a single workflow.

    Sleep times are sampled from real execution timing data collected from Exp07,
    ensuring simulation reflects actual LLM task execution time distributions.

    Args:
        num_loops: Number of loop iterations
        loop_fanouts: Fanout count for each loop iteration
        rng: NumPy random generator (for reproducibility)

    Returns:
        Tuple of (a_times, b1_times_list, b2_times_list, merge_times)
        - a_times: List of A task sleep times (one per loop)
        - b1_times_list: List of lists of B1 sleep times (one list per loop)
        - b2_times_list: List of lists of B2 sleep times (one list per loop)
        - merge_times: List of Merge task sleep times (one per loop)
    """
    # Sample A task times from real data
    a_times = sample_timing_from_real_data('A', count=num_loops, rng=rng)

    # Sample Merge task times from real data (need fanout for each loop)
    merge_times = []
    for fanout in loop_fanouts:
        merge_time = sample_timing_from_real_data('merge', count=1, fanout=fanout, rng=rng)[0]
        merge_times.append(merge_time)

    # Sample B1 and B2 task times from real data for each loop
    b1_times_list = []
    b2_times_list = []

    for fanout in loop_fanouts:
        # Sample B1 times for this loop
        b1_times = sample_timing_from_real_data('B1', count=fanout, rng=rng)
        b1_times_list.append(b1_times)

        # Sample B2 times for this loop
        b2_times = sample_timing_from_real_data('B2', count=fanout, rng=rng)
        b2_times_list.append(b2_times)

    return a_times, b1_times_list, b2_times_list, merge_times


def pre_generate_workflows(
    config,
    seed: int = 42
) -> List["DeepResearchWorkflowData"]:
    """Pre-generate all workflow data before strategy testing.

    This ensures all strategies use identical workflow data (same sleep times,
    fanout counts, etc.) for fair comparison.

    For simulation mode:
        - Fanout: random choice from [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        - Sleep times sampled from real execution data (dr_boot.json, dr_query.json, dr_summary_dict.json)
        - Task A: sampled from 5000 real boot task times (avg ~3.6s)
        - Task B1/B2: sampled from 42941 real query task times (avg ~2.7s)
        - Task Merge: sampled from real summary task times by fanout (avg ~28s)

    For real mode (aligned with Exp07):
        - Data loaded from dataset.jsonl
        - Fanout determined by len(queries) in each dataset entry
        - boot/queries/summary text used directly

    Loop support (when config.max_loops_count > 1):
        - max_loops sampled uniformly from [1, config.max_loops_count]
        - Each loop iteration has independent fanout and data
        - Pre-generated for reproducibility across strategies

    Args:
        config: DeepResearchConfig instance
        seed: Random seed for reproducibility

    Returns:
        List of pre-generated DeepResearchWorkflowData instances
    """
    import random

    # Set random seed for reproducibility
    random.seed(seed)

    # Create NumPy random generator for sleep time generation
    rng = np.random.default_rng(seed)

    workflows = []

    if config.mode == "real":
        # Real mode: load from dataset.jsonl (aligned with Exp07)
        dataset = load_dataset()

        for i in range(config.num_workflows):
            # Determine max_loops for this workflow
            if config.max_loops_count > 1:
                # Sample loop count uniformly from [1, max_loops_count]
                max_loops = random.randint(1, config.max_loops_count)
            else:
                max_loops = 1

            # Sample dataset entries for each loop iteration (WITH REPLACEMENT)
            sampled_entries = random.choices(dataset, k=max_loops)

            # Pre-generate data for each loop iteration
            loop_fanouts = []
            loop_boot_sentences = []
            loop_query_inputs = []
            loop_summary_sentences = []

            for entry in sampled_entries:
                boot_sentence = entry.get("boot", "")
                queries = entry.get("queries", [])
                summary_sentence = entry.get("summary", "")

                fanout_count = len(queries)
                query_inputs = get_query_inputs(queries)

                loop_fanouts.append(fanout_count)
                loop_boot_sentences.append(boot_sentence)
                loop_query_inputs.append(query_inputs)
                loop_summary_sentences.append(summary_sentence)

            # First loop's data is used as the initial values (backward compatible)
            workflow = DeepResearchWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                fanout_count=loop_fanouts[0],
                strategy="pending",  # Will be set per-strategy run
                is_warmup=(i < getattr(config, 'num_warmup', 0)),
                # Real mode fields (first loop - backward compatible)
                boot_sentence=loop_boot_sentences[0],
                query_inputs=loop_query_inputs[0],
                summary_sentence=loop_summary_sentences[0],
                # Loop configuration
                max_loops=max_loops,
                loop_fanouts=loop_fanouts,
                loop_boot_sentences=loop_boot_sentences,
                loop_query_inputs=loop_query_inputs,
                loop_summary_sentences=loop_summary_sentences,
            )

            workflows.append(workflow)

    else:
        # Simulation mode: use normal distributions
        for i in range(config.num_workflows):
            # Determine max_loops for this workflow
            if config.max_loops_count > 1:
                # Sample loop count uniformly from [1, max_loops_count]
                max_loops = random.randint(1, config.max_loops_count)
            else:
                max_loops = 1

            # Pre-generate fanout counts for all loop iterations
            loop_fanouts = [random.choice(AVAILABLE_FANOUTS) for _ in range(max_loops)]

            # Pre-generate sleep times for all loop iterations
            loop_a_sleep_times, loop_b1_sleep_times, loop_b2_sleep_times, loop_merge_sleep_times = \
                _generate_loop_sleep_times(max_loops, loop_fanouts, rng)

            # First loop's data is used as the initial values (backward compatible)
            workflow = DeepResearchWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                fanout_count=loop_fanouts[0],
                strategy="pending",  # Will be set per-strategy run
                is_warmup=(i < getattr(config, 'num_warmup', 0)),
                # Simulation mode fields (first loop - backward compatible)
                a_sleep_time=loop_a_sleep_times[0],
                b1_sleep_times=loop_b1_sleep_times[0],
                b2_sleep_times=loop_b2_sleep_times[0],
                merge_sleep_time=loop_merge_sleep_times[0],
                # Loop configuration
                max_loops=max_loops,
                loop_fanouts=loop_fanouts,
                loop_a_sleep_times=loop_a_sleep_times,
                loop_b1_sleep_times=loop_b1_sleep_times,
                loop_b2_sleep_times=loop_b2_sleep_times,
                loop_merge_sleep_times=loop_merge_sleep_times,
            )

            workflows.append(workflow)

    return workflows
