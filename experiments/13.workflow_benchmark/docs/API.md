# API Reference

Complete API documentation for the Unified Workflow Benchmark Framework.

## Table of Contents

- [Core Infrastructure (common/)](#core-infrastructure)
  - [BaseTaskSubmitter](#basetasksubmitter)
  - [BaseTaskReceiver](#basetaskreceiver)
  - [RateLimiter](#ratelimiter)
  - [MetricsCollector](#metricscollector)
  - [WorkflowState](#workflowstate)
- [Text2Video Workflow (type1_text2video/)](#text2video-workflow)
  - [Text2VideoConfig](#text2videoconfig)
  - [Text2VideoWorkflowData](#text2videoworkflowdata)
  - [A1TaskSubmitter](#a1tasksubmitter)
  - [A2TaskSubmitter](#a2tasksubmitter)
  - [BTaskSubmitter](#btasksubmitter)
- [Deep Research Workflow (type2_deep_research/)](#deep-research-workflow)
  - [DeepResearchConfig](#deepresearchconfig)
  - [DeepResearchWorkflowData](#deepresearchworkflowdata)
  - [ATaskSubmitter](#atasksubmitter)
  - [B1TaskSubmitter](#b1tasksubmitter)
  - [B2TaskSubmitter](#b2tasksubmitter)
  - [MergeTaskSubmitter](#mergetasksubmitter)
- [Testing Tools (tools/)](#testing-tools)
  - [ExperimentRunner](#experimentrunner)
  - [CLI](#command-line-interface)

---

## Core Infrastructure

### BaseTaskSubmitter

Abstract base class for task submission with rate limiting.

**Location**: `common/base_classes.py`

#### Constructor

```python
BaseTaskSubmitter(
    name: str,
    scheduler_url: str,
    rate_limiter: Optional[RateLimiter] = None,
    logger: Optional[logging.Logger] = None
)
```

**Parameters**:
- `name` (str): Thread name for identification
- `scheduler_url` (str): Scheduler endpoint URL
- `rate_limiter` (Optional[RateLimiter]): Rate limiter instance for QPS control
- `logger` (Optional[logging.Logger]): Logger instance

#### Methods

##### start()

Start the submission thread.

```python
def start(self) -> None:
    """Start the task submission thread."""
```

**Example**:
```python
submitter = A1TaskSubmitter(
    name="A1Submitter",
    scheduler_url="http://localhost:8100",
    rate_limiter=RateLimiter(rate=2.0)
)
submitter.start()
```

##### stop()

Stop the submission thread gracefully.

```python
def stop(self) -> None:
    """Stop the submission thread."""
```

##### add_task()

Add a task to the submission queue (for queue-based submitters).

```python
def add_task(self, *args, **kwargs) -> None:
    """Add task to submission queue."""
```

#### Abstract Methods (Must Implement)

##### _get_next_task_data()

Get the next task data to submit.

```python
def _get_next_task_data(self) -> Optional[Any]:
    """
    Get next task data to submit.

    Returns:
        Task data object, or None if no more tasks
    """
```

##### _prepare_task_payload()

Prepare the task payload for submission.

```python
def _prepare_task_payload(self, task_data: Any) -> Dict:
    """
    Prepare task payload from task data.

    Args:
        task_data: Task data object

    Returns:
        Task payload dict with task_id, model_id, task_input, metadata
    """
```

---

### BaseTaskReceiver

Abstract base class for WebSocket result receiving.

**Location**: `common/base_classes.py`

#### Constructor

```python
BaseTaskReceiver(
    name: str,
    scheduler_url: str,
    model_id: str,
    logger: Optional[logging.Logger] = None
)
```

**Parameters**:
- `name` (str): Thread name for identification
- `scheduler_url` (str): Scheduler WebSocket endpoint URL
- `model_id` (str): Model ID to subscribe to
- `logger` (Optional[logging.Logger]): Logger instance

#### Methods

##### start()

Start the receiver thread.

```python
def start(self) -> None:
    """Start the WebSocket receiver thread."""
```

##### stop()

Stop the receiver thread gracefully.

```python
def stop(self) -> None:
    """Stop the receiver thread."""
```

#### Abstract Methods (Must Implement)

##### _get_subscription_payload()

Get the WebSocket subscription payload.

```python
def _get_subscription_payload(self) -> Dict:
    """
    Get subscription payload for WebSocket.

    Returns:
        Subscription payload dict
    """
```

##### _process_result()

Process a received result.

```python
async def _process_result(self, data: Dict) -> None:
    """
    Process received result data.

    Args:
        data: Result data dict from WebSocket
    """
```

---

### RateLimiter

Token bucket rate limiter with Poisson distribution.

**Location**: `common/rate_limiter.py`

#### Constructor

```python
RateLimiter(rate: float, burst_size: Optional[int] = None)
```

**Parameters**:
- `rate` (float): Target rate in requests per second (QPS)
- `burst_size` (Optional[int]): Maximum burst size (default: rate * 2)

#### Methods

##### wait()

Wait until a token is available (blocking).

```python
def wait(self) -> None:
    """Block until a token is available."""
```

**Example**:
```python
rate_limiter = RateLimiter(rate=2.0)

# In submission loop
rate_limiter.wait()  # Blocks until token available
submit_task(...)
```

##### try_acquire()

Try to acquire a token without blocking.

```python
def try_acquire(self) -> bool:
    """
    Try to acquire a token.

    Returns:
        True if token acquired, False otherwise
    """
```

---

### MetricsCollector

Thread-safe metrics collection and reporting.

**Location**: `common/metrics_collector.py`

#### Constructor

```python
MetricsCollector(logger: Optional[logging.Logger] = None)
```

#### Methods

##### record_task_submitted()

Record a task submission.

```python
def record_task_submitted(
    self,
    task_id: str,
    task_type: str,
    workflow_id: str,
    timestamp: Optional[float] = None
) -> None:
    """Record task submission."""
```

##### record_task_completed()

Record a task completion.

```python
def record_task_completed(
    self,
    task_id: str,
    success: bool,
    timestamp: Optional[float] = None
) -> None:
    """Record task completion."""
```

##### record_workflow_completed()

Record a workflow completion.

```python
def record_workflow_completed(
    self,
    workflow_id: str,
    success: bool,
    total_latency: float,
    timestamp: Optional[float] = None
) -> None:
    """Record workflow completion."""
```

##### export_to_json()

Export metrics to JSON file.

```python
def export_to_json(self, filepath: Union[str, Path]) -> None:
    """
    Export metrics to JSON file.

    Args:
        filepath: Output file path
    """
```

**Example**:
```python
metrics = MetricsCollector()

# Record events
metrics.record_task_submitted("task-1-A1", "A1", "workflow-1")
metrics.record_task_completed("task-1-A1", success=True)
metrics.record_workflow_completed("workflow-1", success=True, total_latency=85.2)

# Export
metrics.export_to_json("output/metrics.json")
```

##### generate_text_report()

Generate human-readable text report.

```python
def generate_text_report(self) -> str:
    """
    Generate text report of metrics.

    Returns:
        Formatted text report string
    """
```

---

### WorkflowState

Unified workflow state tracking.

**Location**: `common/data_structures.py`

#### Constructor

```python
WorkflowState(
    workflow_id: str,
    workflow_type: WorkflowType,
    metadata: Optional[Dict] = None
)
```

**Parameters**:
- `workflow_id` (str): Unique workflow identifier
- `workflow_type` (WorkflowType): Type of workflow (TEXT2VIDEO or DEEP_RESEARCH)
- `metadata` (Optional[Dict]): Additional metadata

#### Attributes

```python
workflow_id: str
workflow_type: WorkflowType
start_time: float
end_time: Optional[float]
status: WorkflowStatus  # PENDING, IN_PROGRESS, COMPLETED, FAILED

# Text2Video specific
a1_result: Optional[str]
a2_result: Optional[str]
b_loop_count: int
b_results: List[str]

# Deep Research specific
a_result: Optional[str]
b1_task_ids: List[str]
b1_results: Dict[str, str]
b2_task_ids: List[str]
b2_results: Dict[str, str]
merge_result: Optional[str]

# Common
metadata: Dict
```

---

## Text2Video Workflow

### Text2VideoConfig

Configuration for Text2Video experiments.

**Location**: `type1_text2video/config.py`

#### Factory Method

```python
@classmethod
def from_env(cls) -> "Text2VideoConfig":
    """
    Create config from environment variables.

    Environment Variables:
        MODE: simulation or real (default: simulation)
        QPS: Query per second (default: 2.0)
        DURATION: Experiment duration in seconds (default: 300)
        NUM_WORKFLOWS: Total workflows (default: 600)
        MAX_B_LOOPS: B task iterations (default: 4)
        SCHEDULER_A_URL: Scheduler A endpoint
        SCHEDULER_B_URL: Scheduler B endpoint
        OUTPUT_DIR: Output directory (default: output)

    Returns:
        Text2VideoConfig instance
    """
```

**Example**:
```python
import os
os.environ["QPS"] = "3.0"
os.environ["DURATION"] = "600"

config = Text2VideoConfig.from_env()
print(f"QPS: {config.qps}, Duration: {config.duration}")
```

#### Attributes

```python
# Experiment mode
mode: str  # "simulation" or "real"

# QPS and duration
qps: float
duration: int
num_workflows: int

# Workflow parameters
max_b_loops: int

# Service endpoints
scheduler_a_url: str
scheduler_b_url: str

# Model IDs
model_a_id: str  # A1/A2 model
model_b_id: str  # B model

# Task parameters
sleep_time_min: float  # Simulation mode
sleep_time_max: float  # Simulation mode
max_tokens: int  # Real mode (default: 512)

# Output
output_dir: Path
metrics_file: str
log_level: str

# Caption data
caption_file: str
```

---

### Text2VideoWorkflowData

Workflow data structure for Text2Video.

**Location**: `type1_text2video/workflow_data.py`

#### Constructor

```python
@dataclass
class Text2VideoWorkflowData:
    workflow_id: str
    caption: str
    a1_result: Optional[str] = None
    a2_result: Optional[str] = None
    b_loop_count: int = 0
    max_b_loops: int = 4
```

---

### A1TaskSubmitter

Submitter for A1 tasks (caption processing).

**Location**: `type1_text2video/submitters.py`

#### Constructor

```python
A1TaskSubmitter(
    name: str,
    captions: List[str],
    config: Text2VideoConfig,
    scheduler_url: str,
    rate_limiter: Optional[RateLimiter] = None,
    logger: Optional[logging.Logger] = None
)
```

**Parameters**:
- `captions` (List[str]): List of captions for workflows
- `config` (Text2VideoConfig): Configuration instance
- Other parameters same as BaseTaskSubmitter

**Prompt Template (Real Mode)**:
```
"Generate a detailed image generation prompt based on this caption: {caption}"
```

---

## Deep Research Workflow

### DeepResearchConfig

Configuration for Deep Research experiments.

**Location**: `type2_deep_research/config.py`

#### Factory Method

```python
@classmethod
def from_env(cls) -> "DeepResearchConfig":
    """
    Create config from environment variables.

    Environment Variables:
        MODE: simulation or real (default: simulation)
        QPS: Query per second (default: 1.0)
        DURATION: Experiment duration in seconds (default: 600)
        NUM_WORKFLOWS: Total workflows (default: 600)
        FANOUT_COUNT: Number of B1/B2 tasks per workflow (default: 3)
        SCHEDULER_A_URL: Scheduler A endpoint
        SCHEDULER_B_URL: Scheduler B endpoint
        OUTPUT_DIR: Output directory (default: output)

    Returns:
        DeepResearchConfig instance
    """
```

#### Attributes

```python
# Experiment mode
mode: str  # "simulation" or "real"

# QPS and duration
qps: float
duration: int
num_workflows: int

# Workflow parameters
fanout_count: int  # Number of B1/B2 tasks per workflow

# Service endpoints
scheduler_a_url: str
scheduler_b_url: str

# Model IDs
model_a_id: str    # A model
model_b1_id: str   # B1 model
model_b2_id: str   # B2 model
model_merge_id: str  # Merge model

# Task parameters
sleep_time_min: float  # Simulation mode
sleep_time_max: float  # Simulation mode
max_tokens: int  # Real mode (default: 512)

# Output
output_dir: Path
metrics_file: str
log_level: str
```

---

### DeepResearchWorkflowData

Workflow data structure for Deep Research.

**Location**: `type2_deep_research/workflow_data.py`

#### Constructor

```python
@dataclass
class DeepResearchWorkflowData:
    workflow_id: str
    topic: str
    a_result: Optional[str] = None
    b1_count: int = 0
    b2_count: int = 0
    expected_b1_count: int = 3
    expected_b2_count: int = 3
    b1_results: Dict[str, str] = field(default_factory=dict)
    b2_results: Dict[str, str] = field(default_factory=dict)
    merge_triggered: bool = False
```

---

## Testing Tools

### ExperimentRunner

Unified runner for all experiment types.

**Location**: `tools/experiment_runner.py`

#### Constructor

```python
ExperimentRunner(
    workspace_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
)
```

**Parameters**:
- `workspace_dir` (Optional[Path]): Root directory for experiments (default: current directory)
- `logger` (Optional[Logger]): Logger instance

#### Methods

##### run_text2video_simulation()

Run Text2Video workflow in simulation mode.

```python
def run_text2video_simulation(
    self,
    qps: float = 2.0,
    duration: int = 300,
    num_workflows: int = 600,
    max_b_loops: int = 4,
    output_dir: str = "output",
    **kwargs
) -> Dict:
    """
    Run Text2Video workflow in simulation mode.

    Args:
        qps: Query per second rate
        duration: Experiment duration in seconds
        num_workflows: Total number of workflows to execute
        max_b_loops: Maximum B task iterations per workflow
        output_dir: Output directory for metrics
        **kwargs: Additional configuration parameters

    Returns:
        Dict with experiment results:
            - success (bool): Whether experiment succeeded
            - elapsed_time (float): Total execution time
            - metrics_path (str): Path to metrics file
            - stdout (str): Standard output
            - stderr (str): Standard error

    Raises:
        RuntimeError: If experiment fails
    """
```

**Example**:
```python
from tools.experiment_runner import ExperimentRunner

runner = ExperimentRunner()

result = runner.run_text2video_simulation(
    qps=2.0,
    duration=60,
    num_workflows=120
)

print(f"Success: {result['success']}")
print(f"Elapsed: {result['elapsed_time']:.2f}s")
print(f"Metrics: {result['metrics_path']}")
```

##### run_text2video_real()

Run Text2Video workflow in real cluster mode.

```python
def run_text2video_real(
    self,
    qps: float = 2.0,
    duration: int = 300,
    num_workflows: int = 100,
    max_b_loops: int = 4,
    output_dir: str = "output",
    **kwargs
) -> Dict:
    """
    Run Text2Video workflow in real cluster mode.

    Additional kwargs:
        SCHEDULER_A_URL: Scheduler A endpoint
        SCHEDULER_B_URL: Scheduler B endpoint
    """
```

##### run_deep_research_simulation()

Run Deep Research workflow in simulation mode.

```python
def run_deep_research_simulation(
    self,
    qps: float = 1.0,
    duration: int = 600,
    num_workflows: int = 600,
    fanout_count: int = 3,
    output_dir: str = "output",
    **kwargs
) -> Dict:
    """
    Run Deep Research workflow in simulation mode.

    Args:
        fanout_count: Number of B1/B2 tasks per workflow
        Other args same as run_text2video_simulation
    """
```

##### run_deep_research_real()

Run Deep Research workflow in real cluster mode.

```python
def run_deep_research_real(
    self,
    qps: float = 1.0,
    duration: int = 600,
    num_workflows: int = 100,
    fanout_count: int = 3,
    output_dir: str = "output",
    **kwargs
) -> Dict:
    """Run Deep Research workflow in real cluster mode."""
```

---

### Command-Line Interface

Command-line tool for running experiments.

**Location**: `tools/cli.py`

#### Commands

##### run-text2video-sim

Run Text2Video simulation.

```bash
python tools/cli.py run-text2video-sim [OPTIONS]

Options:
  --qps FLOAT              Query per second (default: 2.0)
  --duration INT           Duration in seconds (default: 300)
  --num-workflows INT      Number of workflows (default: 600)
  --max-b-loops INT        Max B iterations (default: 4)
  --output-dir STR         Output directory (default: output)
```

##### run-text2video-real

Run Text2Video real cluster mode.

```bash
python tools/cli.py run-text2video-real [OPTIONS]

Options:
  --qps FLOAT              Query per second (default: 2.0)
  --duration INT           Duration in seconds (default: 300)
  --num-workflows INT      Number of workflows (default: 100)
  --max-b-loops INT        Max B iterations (default: 4)
  --output-dir STR         Output directory (default: output)
  --scheduler-a-url STR    Scheduler A URL
  --scheduler-b-url STR    Scheduler B URL
```

##### run-deep-research-sim

Run Deep Research simulation.

```bash
python tools/cli.py run-deep-research-sim [OPTIONS]

Options:
  --qps FLOAT              Query per second (default: 1.0)
  --duration INT           Duration in seconds (default: 600)
  --num-workflows INT      Number of workflows (default: 600)
  --fanout-count INT       Fanout count (default: 3)
  --output-dir STR         Output directory (default: output)
```

##### run-deep-research-real

Run Deep Research real cluster mode.

```bash
python tools/cli.py run-deep-research-real [OPTIONS]

Options:
  --qps FLOAT              Query per second (default: 1.0)
  --duration INT           Duration in seconds (default: 600)
  --num-workflows INT      Number of workflows (default: 100)
  --fanout-count INT       Fanout count (default: 3)
  --output-dir STR         Output directory (default: output)
  --scheduler-a-url STR    Scheduler A URL
  --scheduler-b-url STR    Scheduler B URL
```

---

## Common Patterns

### Creating a Custom Submitter

```python
from common import BaseTaskSubmitter
import random

class CustomTaskSubmitter(BaseTaskSubmitter):
    def __init__(self, tasks, config, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.config = config
        self.index = 0

    def _get_next_task_data(self):
        if self.index < len(self.tasks):
            task = self.tasks[self.index]
            self.index += 1
            return task
        return None

    def _prepare_task_payload(self, task_data):
        return {
            "task_id": f"task-{task_data['id']}",
            "model_id": self.config.model_id,
            "task_input": {
                "sentence": task_data['prompt'],
                "max_tokens": 512
            },
            "metadata": {
                "custom_field": task_data.get('metadata')
            }
        }
```

### Creating a Custom Receiver

```python
from common import BaseTaskReceiver

class CustomTaskReceiver(BaseTaskReceiver):
    def __init__(self, next_submitter, **kwargs):
        super().__init__(**kwargs)
        self.next_submitter = next_submitter

    def _get_subscription_payload(self):
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data):
        task_id = data.get("task_id")
        result = data.get("result", {}).get("output", "")

        # Trigger next stage
        self.next_submitter.add_task(task_id, result)
```

### Running Experiments Programmatically

```python
from tools.experiment_runner import ExperimentRunner
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create runner
runner = ExperimentRunner()

# Run multiple experiments
configs = [
    {"qps": 1.0, "duration": 300},
    {"qps": 2.0, "duration": 300},
    {"qps": 3.0, "duration": 300},
]

results = []
for config in configs:
    result = runner.run_text2video_simulation(**config)
    results.append(result)
    print(f"QPS {config['qps']}: {result['elapsed_time']:.2f}s")
```

---

## Type Definitions

### Enums

```python
from enum import Enum

class WorkflowType(Enum):
    TEXT2VIDEO = "text2video"
    DEEP_RESEARCH = "deep_research"

class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Task Metrics

```python
@dataclass
class TaskMetrics:
    task_type: str
    count: int = 0
    success_count: int = 0
    latencies: List[float] = field(default_factory=list)
```

### Workflow Metrics

```python
@dataclass
class WorkflowMetrics:
    total: int = 0
    completed: int = 0
    failed: int = 0
    latencies: List[float] = field(default_factory=list)
```

---

## Error Handling

All methods may raise:

- `ConnectionError`: Network connection issues
- `TimeoutError`: Operation timeout
- `ValueError`: Invalid parameter values
- `RuntimeError`: Runtime execution errors

**Example**:
```python
try:
    result = runner.run_text2video_simulation(qps=2.0, duration=300)
except RuntimeError as e:
    logger.error(f"Experiment failed: {e}")
except Exception as e:
    logger.exception("Unexpected error")
```

---

## Performance Considerations

- **Thread Safety**: All metrics operations are thread-safe
- **Memory**: Metrics stored in memory; export regularly for long experiments
- **Rate Limiting**: Token bucket allows bursts up to 2x the configured rate
- **WebSocket**: Automatic reconnection with exponential backoff
- **Logging**: Configure log level via environment or config

---

## Version Compatibility

- Python 3.8+
- Dependencies: `requests`, `websockets`, `numpy` (optional)

---

For examples and use cases, see:
- [Quick Start Guide](QUICKSTART.md)
- [Migration Guide](MIGRATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
