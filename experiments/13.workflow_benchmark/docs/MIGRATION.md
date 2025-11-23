# Migration Guide

Step-by-step guide for migrating from original experiment scripts to the Unified Workflow Benchmark Framework.

## Table of Contents

- [Overview](#overview)
- [Migration Benefits](#migration-benefits)
- [Prerequisites](#prerequisites)
- [Migration Paths](#migration-paths)
  - [Text2Video Migration](#text2video-migration)
  - [Deep Research Migration](#deep-research-migration)
- [Configuration Migration](#configuration-migration)
- [Code Structure Comparison](#code-structure-comparison)
- [Feature Mapping](#feature-mapping)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Unified Workflow Benchmark Framework consolidates multiple experiment implementations into a reusable, maintainable architecture with:

- **Common infrastructure**: Shared base classes, rate limiting, metrics
- **Dual-mode support**: Simulation and real cluster modes
- **Simplified configuration**: Environment variables or YAML configs
- **Unified API**: Consistent interfaces across workflow types
- **Testing tools**: CLI, experiment runner, validation utilities

---

## Migration Benefits

### Before (Original Scripts)

```
experiments/03.Exp4.Text2Video/
├── test_dynamic_workflow_sim.py          # 600+ lines, mixed concerns
├── test_workflow_sim.py                  # 500+ lines, duplicated logic
├── start_all_services.sh                 # Manual service management
└── manual_deploy_planner.sh              # Manual deployment

experiments/07.Exp2.Deep_Research_Migration_Test/
├── test_dynamic_workflow.py              # 700+ lines, complex synchronization
├── test_workflow_sim.py                  # Similar patterns, duplicated code
└── deployment scripts                    # Scattered configuration
```

**Issues**:
- Code duplication across experiments
- Mixed concerns (submission, receiving, metrics, orchestration)
- Hard-coded configuration
- Difficult to test components in isolation
- No reusable abstractions

### After (Unified Framework)

```
experiments/13.workflow_benchmark/
├── common/                               # Reusable infrastructure
│   ├── base_classes.py                  # BaseTaskSubmitter, BaseTaskReceiver
│   ├── rate_limiter.py                  # Token bucket rate limiting
│   ├── metrics_collector.py             # Thread-safe metrics
│   └── utils.py                         # Common utilities
│
├── type1_text2video/                    # Clean workflow implementation
│   ├── config.py                        # Typed configuration
│   ├── submitters.py                    # 3 focused submitters (~50 lines each)
│   ├── receivers.py                     # 3 focused receivers (~50 lines each)
│   └── simulation/test_workflow_sim.py  # 200 lines, pure orchestration
│
├── type2_deep_research/                 # Same clean structure
└── tools/                               # Testing & automation
    ├── cli.py                           # Command-line interface
    └── experiment_runner.py             # Unified API
```

**Benefits**:
- **85% less code** through reuse
- **Clear separation of concerns**
- **Type-safe configuration**
- **Easy to test** (mock-friendly abstractions)
- **Consistent patterns** across workflow types

---

## Prerequisites

1. **Python 3.8+**
2. **uv** (recommended) or pip:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Original experiment data**:
   - Caption file (Text2Video): `captions_10k.json`
   - Research topics (Deep Research): Optional, auto-generated if missing

---

## Migration Paths

### Text2Video Migration

#### Step 1: Understand Current Implementation

**Original**: `experiments/03.Exp4.Text2Video/test_dynamic_workflow_sim.py`

Key patterns to identify:
```python
# A1 submission (caption-based)
for i in range(num_workflows):
    caption = captions[i % len(captions)]
    task = {
        "task_id": f"workflow-{i:04d}-A1",
        "model_id": "llm_service_small_model",
        "task_input": {"sleep_time": random.uniform(5, 15)},
        "metadata": {"workflow_id": f"workflow-{i:04d}", "caption": caption}
    }
    submit_task(task)

# A1 result → trigger A2
def handle_a1_result(data):
    positive_prompt = data["result"]["output"]
    submit_a2_task(positive_prompt)

# A2 result → trigger B
def handle_a2_result(data):
    negative_prompt = data["result"]["output"]
    submit_b_task(positive_prompt, negative_prompt)

# B result → loop or complete
def handle_b_result(data):
    workflow_data.b_loop_count += 1
    if workflow_data.b_loop_count < max_b_loops:
        submit_b_task_again()
```

#### Step 2: Map to New Framework

**New**: `experiments/13.workflow_benchmark/type1_text2video/`

| Original Concept | New Framework |
|-----------------|---------------|
| A1 submission logic | `A1TaskSubmitter._prepare_task_payload()` |
| A1 result handling | `A1TaskReceiver._process_result()` |
| A2 submission logic | `A2TaskSubmitter._prepare_task_payload()` |
| B loop control | `BTaskReceiver._process_result()` + `workflow_data.b_loop_count` |
| Metrics tracking | `MetricsCollector.record_*()` |
| Rate limiting | `RateLimiter.wait()` |

#### Step 3: Create Configuration

**Option A: Environment Variables**
```bash
export MODE=simulation
export QPS=2.0
export DURATION=300
export NUM_WORKFLOWS=600
export MAX_B_LOOPS=4
export SCHEDULER_A_URL=http://localhost:8100
export SCHEDULER_B_URL=http://localhost:8101
export CAPTION_FILE=captions_10k.json
export OUTPUT_DIR=output
```

**Option B: YAML Configuration**
```yaml
# Use configs/text2video_sim_config.yaml
experiment_type: text2video
mode: simulation
qos:
  qps: 2.0
  duration: 300
  num_workflows: 600
workflow:
  max_b_loops: 4
# ... (see full config)
```

#### Step 4: Run Migration

```bash
cd experiments/13.workflow_benchmark

# Method 1: CLI (easiest)
python tools/cli.py run-text2video-sim --duration 300 --num-workflows 600

# Method 2: Direct Python
export QPS=2.0
python type1_text2video/simulation/test_workflow_sim.py

# Method 3: Programmatic
from tools.experiment_runner import ExperimentRunner
runner = ExperimentRunner()
result = runner.run_text2video_simulation(qps=2.0, duration=300, num_workflows=600)
```

#### Step 5: Validate Results

```bash
# Compare metrics
diff <(jq -S . experiments/03.Exp4.Text2Video/output/metrics.json) \
     <(jq -S . experiments/13.workflow_benchmark/output/metrics.json)

# Check key metrics:
# - Total workflows completed
# - Task counts (A1, A2, B)
# - Success rates
# - Latency distributions (p50, p95, p99)
```

---

### Deep Research Migration

#### Step 1: Understand Current Implementation

**Original**: `experiments/07.Exp2.Deep_Research_Migration_Test/test_dynamic_workflow.py`

Key patterns to identify:
```python
# A submission
def submit_a_task():
    task = {
        "task_id": f"workflow-{i:04d}-A",
        "model_id": "llm_service_small_model",
        "task_input": {"sleep_time": random.uniform(5, 15)},
        "metadata": {"workflow_id": f"workflow-{i:04d}", "topic": topic}
    }
    submit_task(task)

# A result → fan out to n B1 tasks
def handle_a_result(data):
    for j in range(fanout_count):
        submit_b1_task(j, a_result)

# B1 result → trigger B2 (1:1 mapping)
def handle_b1_result(data):
    submit_b2_task(b1_result)

# B2 result → check if all complete → trigger merge
def handle_b2_result(data):
    workflow_data.b2_count += 1
    if workflow_data.b2_count == expected_b2_count:
        submit_merge_task()
```

#### Step 2: Map to New Framework

| Original Concept | New Framework |
|-----------------|---------------|
| A submission | `ATaskSubmitter` |
| A result → fanout | `ATaskReceiver._process_result()` (loops to create B1 tasks) |
| B1 submission | `B1TaskSubmitter` |
| B1 → B2 (1:1) | `B1TaskReceiver._process_result()` |
| B2 synchronization | `B2TaskReceiver._process_result()` (atomic counter check) |
| Merge submission | `MergeTaskSubmitter` |
| Merge result | `MergeTaskReceiver._process_result()` |

#### Step 3: Create Configuration

**Environment Variables**:
```bash
export MODE=simulation
export QPS=1.0
export DURATION=600
export NUM_WORKFLOWS=600
export FANOUT_COUNT=3
export SCHEDULER_A_URL=http://localhost:8100
export SCHEDULER_B_URL=http://localhost:8101
export OUTPUT_DIR=output
```

**YAML Configuration**: Use `configs/deep_research_sim_config.yaml`

#### Step 4: Run Migration

```bash
# CLI
python tools/cli.py run-deep-research-sim --duration 600 --num-workflows 600 --fanout-count 3

# Direct
export QPS=1.0 FANOUT_COUNT=3
python type2_deep_research/simulation/test_workflow_sim.py

# Programmatic
from tools.experiment_runner import ExperimentRunner
runner = ExperimentRunner()
result = runner.run_deep_research_simulation(qps=1.0, duration=600, num_workflows=600, fanout_count=3)
```

#### Step 5: Validate Results

Key metrics to check:
- Total workflows completed
- Task counts: A, B1, B2, Merge (should be 1:n:n:1 ratio)
- All B2 tasks complete before Merge triggered
- Success rates
- Latency distributions

---

## Configuration Migration

### Hard-coded Values → Environment Variables

**Before**:
```python
# Hard-coded in script
QPS = 2.0
DURATION = 300
NUM_WORKFLOWS = 600
SCHEDULER_A_URL = "http://10.28.1.16:8100"
```

**After**:
```python
# Type-safe config
from type1_text2video.config import Text2VideoConfig

config = Text2VideoConfig.from_env()  # Reads from environment
print(f"QPS: {config.qps}, Duration: {config.duration}")
```

### Script Arguments → YAML Configuration

**Before**:
```bash
python test_workflow_sim.py \
    --qps 2.0 \
    --duration 300 \
    --num-workflows 600 \
    --scheduler-a http://localhost:8100 \
    --scheduler-b http://localhost:8101
```

**After**:
```yaml
# text2video_sim_config.yaml
qos:
  qps: 2.0
  duration: 300
  num_workflows: 600
schedulers:
  scheduler_a:
    url: "http://localhost:8100"
  scheduler_b:
    url: "http://localhost:8101"
```

```bash
# Load from YAML (implement YAML loader if needed)
python tools/cli.py run-text2video-sim --config configs/text2video_sim_config.yaml
```

---

## Code Structure Comparison

### Original Text2Video (Single File)

```python
# test_dynamic_workflow_sim.py (600+ lines)

# Global state
workflows = {}
state_lock = threading.Lock()

# Submission logic (inline)
def a1_submission_thread():
    while True:
        # Rate limiting logic
        # Caption loading
        # Task creation
        # HTTP submission
        # Error handling

# Result handling (inline)
async def a1_receiver_thread():
    async with websockets.connect(...) as ws:
        # WebSocket setup
        # Subscription
        # Result processing
        # State updates
        # Next task triggering

# Metrics (inline)
metrics = {}
def record_metric(...):
    # Manual dict updates
    # No thread safety

# Main
if __name__ == "__main__":
    # Parse args
    # Setup threads
    # Start threads
    # Wait
    # Stop threads
    # Export metrics
```

### New Framework (Modular)

```python
# common/base_classes.py (~150 lines)
class BaseTaskSubmitter(threading.Thread):
    """Reusable submission logic."""
    def _get_next_task_data(self): ...      # Abstract
    def _prepare_task_payload(self, data): ... # Abstract

class BaseTaskReceiver(threading.Thread):
    """Reusable receiving logic."""
    async def _process_result(self, data): ... # Abstract

# type1_text2video/submitters.py (~150 lines total)
class A1TaskSubmitter(BaseTaskSubmitter):
    def _get_next_task_data(self):
        """Get next workflow from list."""
        ...
    def _prepare_task_payload(self, workflow_data):
        """Create A1 task from caption."""
        ...

# type1_text2video/receivers.py (~150 lines total)
class A1TaskReceiver(BaseTaskReceiver):
    async def _process_result(self, data):
        """Extract positive prompt, trigger A2."""
        ...

# type1_text2video/simulation/test_workflow_sim.py (~200 lines)
def main():
    # Load config
    config = Text2VideoConfig.from_env()

    # Initialize components
    metrics = MetricsCollector()
    rate_limiter = RateLimiter(rate=config.qps)
    captions = load_captions(config.caption_file)

    # Create threads
    a1_submitter = A1TaskSubmitter(captions=captions, config=config, ...)
    a1_receiver = A1TaskReceiver(a2_submitter=a2_submitter, ...)
    # ...

    # Start all
    a1_submitter.start()
    a1_receiver.start()
    # ...

    # Wait and export
    time.sleep(config.duration)
    metrics.export_to_json(config.output_dir / config.metrics_file)
```

**Key Improvements**:
- **Separation of Concerns**: Each class has single responsibility
- **Reusability**: Base classes used by multiple workflows
- **Testability**: Can mock individual components
- **Type Safety**: Dataclasses for configuration and state
- **Maintainability**: Changes localized to specific files

---

## Feature Mapping

### Rate Limiting

**Before**:
```python
# Manual token bucket implementation
tokens = qps
last_refill = time.time()

def wait_for_token():
    nonlocal tokens, last_refill
    now = time.time()
    tokens += (now - last_refill) * qps
    tokens = min(tokens, qps * 2)
    last_refill = now

    if tokens >= 1:
        tokens -= 1
        return
    time.sleep((1 - tokens) / qps)
```

**After**:
```python
from common import RateLimiter

rate_limiter = RateLimiter(rate=2.0)
rate_limiter.wait()  # Handles everything
```

### Metrics Collection

**Before**:
```python
# Manual dict management
metrics = {
    "tasks": {},
    "workflows": {}
}
metrics_lock = threading.Lock()

def record_task(task_id, task_type):
    with metrics_lock:
        if task_type not in metrics["tasks"]:
            metrics["tasks"][task_type] = []
        metrics["tasks"][task_type].append({"id": task_id, "time": time.time()})
```

**After**:
```python
from common import MetricsCollector

metrics = MetricsCollector()
metrics.record_task_submitted("task-1-A1", "A1", "workflow-1")
metrics.record_task_completed("task-1-A1", success=True)
metrics.export_to_json("output/metrics.json")
```

### WebSocket Connection

**Before**:
```python
# Manual connection management
async def receiver():
    retries = 0
    while retries < 3:
        try:
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"type": "subscribe", ...}))
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    # Process result
        except Exception as e:
            retries += 1
            await asyncio.sleep(2 ** retries)
```

**After**:
```python
# Handled by BaseTaskReceiver
class A1TaskReceiver(BaseTaskReceiver):
    async def _process_result(self, data):
        # Just handle the result, connection management automatic
        ...
```

---

## Validation

### Metrics Validation

Compare key metrics between original and new implementations:

```python
# tools/metrics_validator.py (example usage)
from tools.metrics_validator import MetricsValidator

validator = MetricsValidator()
original_metrics = validator.load_metrics("experiments/03.Exp4.Text2Video/output/metrics.json")
new_metrics = validator.load_metrics("experiments/13.workflow_benchmark/output/metrics.json")

# Compare
comparison = validator.compare_metrics(original_metrics, new_metrics, tolerance=0.05)
print(comparison.summary())

# Expected:
# ✓ Total workflows: 600 vs 600 (0% diff)
# ✓ A1 tasks: 600 vs 600 (0% diff)
# ✓ Success rate: 100% vs 100% (0% diff)
# ✓ p50 latency: 85.2s vs 84.8s (0.5% diff, within 5% tolerance)
```

### Functional Validation

```bash
# Run both versions
cd experiments/03.Exp4.Text2Video
./start_all_services.sh
python test_dynamic_workflow_sim.py
# Output: output/metrics.json

cd ../../13.workflow_benchmark
python tools/cli.py run-text2video-sim --duration 300 --num-workflows 600
# Output: output/metrics.json

# Compare outputs
python tools/metrics_validator.py \
    --original ../03.Exp4.Text2Video/output/metrics.json \
    --new output/metrics.json \
    --tolerance 0.05
```

---

## Troubleshooting

### Issue: Metrics Don't Match

**Symptom**: New framework shows different task counts or latencies

**Diagnosis**:
```python
# Check configuration
config = Text2VideoConfig.from_env()
print(f"QPS: {config.qps}, Duration: {config.duration}")
print(f"Workflows: {config.num_workflows}, Max B Loops: {config.max_b_loops}")

# Verify caption loading
captions = load_captions(config.caption_file)
print(f"Loaded {len(captions)} captions")
```

**Solution**:
- Ensure environment variables match original config
- Verify caption file is identical
- Check that QPS and duration are consistent

### Issue: Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'common'`

**Solution**:
```bash
# Ensure Python path includes parent directory
cd experiments/13.workflow_benchmark
export PYTHONPATH=$PYTHONPATH:$(pwd)
python type1_text2video/simulation/test_workflow_sim.py

# Or use relative imports in scripts
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### Issue: WebSocket Connection Failures

**Symptom**: `ConnectionRefusedError: [Errno 111] Connection refused`

**Diagnosis**:
```bash
# Check if schedulers are running
curl http://localhost:8100/health
curl http://localhost:8101/health
```

**Solution**:
```bash
# Start services
cd experiments/03.Exp4.Text2Video
./start_all_services.sh

# Or use new service management
from service_management import ServiceLauncher
launcher = ServiceLauncher()
launcher.start_all_services(mode="simulation")
```

### Issue: Performance Degradation

**Symptom**: New framework shows higher latencies or lower throughput

**Diagnosis**:
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check rate limiter
rate_limiter = RateLimiter(rate=2.0)
start = time.time()
for i in range(10):
    rate_limiter.wait()
elapsed = time.time() - start
print(f"10 tokens in {elapsed:.2f}s (expected ~5s)")
```

**Solution**:
- Verify rate limiter configuration
- Check for resource contention (CPU, memory)
- Profile with `cProfile` if needed

---

## Best Practices

1. **Migrate Incrementally**:
   - Start with simulation mode
   - Validate metrics match original
   - Then migrate to real cluster mode

2. **Version Control**:
   - Keep original scripts for reference
   - Use git branches for migration work
   - Tag releases: `v1.0-migration-complete`

3. **Documentation**:
   - Document configuration changes
   - Update README with new commands
   - Add comments for non-obvious mappings

4. **Testing**:
   - Run both versions side-by-side
   - Compare metrics with tolerance
   - Test edge cases (failures, timeouts)

5. **Performance**:
   - Profile before and after
   - Monitor resource usage
   - Optimize hot paths if needed

---

## Next Steps

After successful migration:

1. **Extend with New Workflows**:
   - Create `type3_custom/` for new workflow patterns
   - Reuse common infrastructure

2. **Add Advanced Features**:
   - Custom metrics exporters
   - Real-time dashboards
   - Automated CI/CD pipelines

3. **Optimize**:
   - Profile and optimize bottlenecks
   - Implement connection pooling
   - Add caching layers

4. **Documentation**:
   - Create architecture diagrams
   - Document design decisions
   - Write runbooks for operations

---

## Support

- **Documentation**: See [API.md](API.md), [QUICKSTART.md](QUICKSTART.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: File issues in project repository
- **Questions**: Contact maintainers or open discussions

---

**Migration Checklist**:
- [ ] Understand original implementation
- [ ] Map to new framework components
- [ ] Create configuration (env vars or YAML)
- [ ] Run new implementation
- [ ] Validate metrics match
- [ ] Update documentation
- [ ] Test edge cases
- [ ] Deploy to production (if applicable)
