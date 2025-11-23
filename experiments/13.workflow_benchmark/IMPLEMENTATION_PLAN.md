# Implementation Plan for Tasks 13-18

## Overview
This document provides detailed implementation guidance for completing the unified workflow benchmark framework.

## Task 13: Text2Video Workflow - Simulation Mode ⏳

### Files to Create

#### 1. `type1_text2video/workflow_data.py`
```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Text2VideoWorkflowData:
    workflow_id: str
    caption: str
    a1_result: Optional[str] = None
    a2_result: Optional[str] = None
    b_loop_count: int = 0
    max_b_loops: int = 4

def load_captions(filepath: str) -> List[str]:
    """Load captions from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('captions', [])
```

#### 2. `type1_text2video/submitters.py`
```python
import random
from common import BaseTaskSubmitter
from .workflow_data import Text2VideoWorkflowData

class A1TaskSubmitter(BaseTaskSubmitter):
    def __init__(self, captions, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.workflows = [
            Text2VideoWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                caption=captions[i % len(captions)],
                max_b_loops=config.max_b_loops
            )
            for i in range(config.num_workflows)
        ]
        self.index = 0

    def _prepare_task_payload(self, workflow_data):
        sleep_time = random.uniform(
            self.config.sleep_time_min,
            self.config.sleep_time_max
        )
        return {
            "task_id": f"{workflow_data.workflow_id}-A1",
            "model_id": self.config.model_a_id,
            "task_input": {"sleep_time": sleep_time},
            "metadata": {
                "workflow_id": workflow_data.workflow_id,
                "caption": workflow_data.caption
            }
        }

    def _get_next_task_data(self):
        if self.index < len(self.workflows):
            workflow = self.workflows[self.index]
            self.index += 1
            return workflow
        return None

# Similar pattern for A2TaskSubmitter and BTaskSubmitter
```

#### 3. `type1_text2video/receivers.py`
```python
from common import BaseTaskReceiver

class A1TaskReceiver(BaseTaskReceiver):
    def __init__(self, a2_submitter, workflow_states, state_lock, **kwargs):
        super().__init__(**kwargs)
        self.a2_submitter = a2_submitter
        self.workflow_states = workflow_states
        self.state_lock = state_lock

    def _get_subscription_payload(self):
        return {
            "type": "subscribe",
            "model_id": self.model_id
        }

    async def _process_result(self, data):
        workflow_id = data.get("metadata", {}).get("workflow_id")
        result = data.get("result", {}).get("output", "")

        # Update workflow state
        with self.state_lock:
            state = self.workflow_states.get(workflow_id)
            if state:
                state.a1_result = result

        # Trigger A2 task
        self.a2_submitter.add_task(workflow_id, result)

# Similar pattern for A2TaskReceiver and BTaskReceiver
```

#### 4. `type1_text2video/simulation/test_workflow_sim.py`
```python
import sys
import threading
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import (
    configure_logging,
    MetricsCollector,
    RateLimiter,
    WorkflowState,
    WorkflowType
)
from type1_text2video.config import Text2VideoConfig
from type1_text2video.workflow_data import load_captions
from type1_text2video.submitters import A1TaskSubmitter
from type1_text2video.receivers import A1TaskReceiver, A2TaskReceiver, BTaskReceiver

def main():
    # Load config
    config = Text2VideoConfig.from_env()

    # Setup logging
    logger = configure_logging(level="INFO")

    # Load captions
    captions = load_captions(config.caption_file)

    # Initialize components
    metrics = MetricsCollector(logger)
    rate_limiter = RateLimiter(rate=config.qps)
    workflow_states = {}
    state_lock = threading.Lock()

    # Create submitters and receivers
    a1_submitter = A1TaskSubmitter(
        name="A1Submitter",
        captions=captions,
        config=config,
        scheduler_url=config.scheduler_a_url,
        rate_limiter=rate_limiter
    )

    # ... create other submitters and receivers

    # Start all threads
    logger.info("Starting experiment...")
    a1_submitter.start()
    # ... start all receivers

    # Wait for duration
    time.sleep(config.duration)

    # Stop all threads
    logger.info("Stopping experiment...")
    a1_submitter.stop()
    # ... stop all threads

    # Export metrics
    metrics.export_to_json(config.output_dir / config.metrics_file)
    print(metrics.generate_text_report())

if __name__ == "__main__":
    main()
```

## Task 14: Deep Research Workflow - Simulation Mode

### Pattern Differences
- **Fanout**: A receiver creates n B1 tasks (n = fanout_count)
- **1:1 Mapping**: Each B1 triggers exactly one B2
- **Synchronization**: Merge triggers only when all B2 complete
- **6 Threads**: A submitter, A receiver, B1 receiver, B2 receiver, Merge submitter, Merge receiver

### Key Implementation Points
- Use atomic dict operations for B1/B2 completion tracking
- Implement merge triggering with conditional check
- Track task IDs in lists (b1_task_ids, b2_task_ids)

## Task 15: Service Management

### `service_management/service_launcher.py`
```python
import subprocess
import socket
from pathlib import Path

class ServiceLauncher:
    def get_local_ip(self):
        """Extract IP from bond1 interface."""
        # ip addr show bond1 | grep inet | awk ...
        pass

    def start_service(self, service_type, host, port, cpu_cores, gpu_id, env_vars):
        """Start service with resource binding."""
        cmd = ["taskset", "-c", cpu_cores]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.update(env_vars)

        process = subprocess.Popen(cmd, env=env, stdout=log_file)
        return process.pid

    def start_all_services(self, mode="simulation"):
        """Start all required services."""
        if mode == "simulation":
            # Start local services
            pass
        elif mode == "real":
            # Start distributed services based on IP
            pass
```

### `service_management/deployment_manager.py`
```python
from concurrent.futures import ThreadPoolExecutor
import requests

class DeploymentManager:
    def deploy_to_scheduler(self, models, scheduler_url, max_workers=10):
        """Deploy models to scheduler in parallel."""
        def deploy_one(model):
            for attempt in range(3):  # Retry logic
                try:
                    response = requests.post(
                        f"{scheduler_url}/model/start",
                        json=model,
                        timeout=30
                    )
                    response.raise_for_status()
                    return True
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(deploy_one, m) for m in models]
            return all(f.result() for f in futures)
```

## Task 16: Real Cluster Mode

### Extensions to Text2Video
```python
# In submitters.py, update _prepare_task_payload for real mode
def _prepare_task_payload(self, workflow_data):
    if self.mode == "simulation":
        task_input = {"sleep_time": random.uniform(5, 15)}
    else:  # real mode
        task_input = {
            "sentence": f"Generate video from: {workflow_data.caption}",
            "max_tokens": 512,
            "platform": "linux"
        }
        task_input["token_length"] = estimate_token_length(workflow_data.caption)

    return {
        "task_id": f"{workflow_data.workflow_id}-A1",
        "model_id": self.model_a_id,
        "task_input": task_input,
        "metadata": {...}
    }
```

## Task 17: Testing & Validation Tools

### `tools/experiment_runner.py`
```python
class ExperimentRunner:
    def run_experiment(self, experiment_type, mode, config):
        """Run a single experiment."""
        # 1. Launch services via ServiceLauncher
        # 2. Deploy models via DeploymentManager
        # 3. Start experiment script
        # 4. Monitor progress
        # 5. Collect metrics
        # 6. Stop services
        pass

    def run_comparison(self, experiment_type, configs):
        """Run multiple configs and compare."""
        results = []
        for config in configs:
            result = self.run_experiment(experiment_type, "simulation", config)
            results.append(result)
        return self.compare_results(results)
```

### `tools/workload_generator.py`
```python
import numpy as np

class WorkloadGenerator:
    def generate_constant_qps(self, qps, duration):
        """Generate constant rate arrivals."""
        interval = 1.0 / qps
        return [i * interval for i in range(int(qps * duration))]

    def generate_poisson(self, lambda_qps, duration):
        """Generate Poisson arrivals."""
        arrivals = []
        t = 0
        while t < duration:
            t += np.random.exponential(1.0 / lambda_qps)
            if t < duration:
                arrivals.append(t)
        return arrivals

    def generate_bursty(self, base_qps, burst_factor, burst_duration):
        """Generate bursty traffic pattern."""
        pass
```

## Task 18: Documentation

### Files to Create
1. **docs/API.md**: Complete API reference with examples
2. **docs/MIGRATION.md**: Step-by-step migration from original scripts
3. **docs/ARCHITECTURE.md**: System diagrams and design decisions
4. **docs/TROUBLESHOOTING.md**: Common issues and solutions
5. **configs/*.yaml**: Configuration templates for all scenarios

### Configuration Template Example
```yaml
# configs/text2video_sim_config.yaml
experiment_type: text2video
mode: simulation

qps: 2.0
duration: 300
num_workflows: 600
max_b_loops: 4

scheduler_a:
  host: 127.0.0.1
  port: 8100

scheduler_b:
  host: 127.0.0.1
  port: 8101

models:
  model_a_id: llm_service_small_model
  model_b_id: t2vid

output:
  dir: output
  metrics_file: metrics.json
  log_level: INFO
```

## Testing Strategy

### Unit Tests
```python
# tests/test_text2video.py
def test_a1_submitter_caption_loading():
    captions = load_captions("test_captions.json")
    assert len(captions) > 0

def test_a1_submitter_payload_format():
    submitter = A1TaskSubmitter(...)
    payload = submitter._prepare_task_payload(workflow_data)
    assert "task_id" in payload
    assert "model_id" in payload
```

### Integration Tests
```python
# tests/test_integration.py
def test_text2video_end_to_end():
    # Start mock services
    # Run experiment for 10 seconds
    # Verify all workflows complete
    # Check metrics accuracy
    pass
```

## Implementation Checklist

- [x] Task 11: Core Infrastructure
- [x] Task 12: Metrics Collection
- [x] Task 13: Text2Video Simulation
  - [x] Config
  - [x] workflow_data.py
  - [x] submitters.py
  - [x] receivers.py
  - [x] main simulation script
  - [x] comprehensive tests (17 tests, all passing)
- [x] Task 14: Deep Research Simulation
  - [x] Config
  - [x] workflow_data.py
  - [x] submitters.py (A, B1, B2, Merge)
  - [x] receivers.py (A, B1, B2, Merge)
  - [x] main simulation script (8-thread architecture)
  - [x] comprehensive tests (15 tests, all passing)
- [x] Task 15: Service Management
  - [x] service_launcher.py (CPU/GPU binding, process management)
  - [x] deployment_manager.py (parallel deployment, retry logic)
  - [x] health_checker.py (health monitoring)
  - [x] resource_binder.py (CPU/GPU allocation)
  - [x] comprehensive tests (19 tests, all passing)
- [x] Task 16: Real Cluster Mode
  - [x] Added mode parameter to configs (Text2Video and Deep Research)
  - [x] Added estimate_token_length utility (already present in common/utils.py)
  - [x] Updated Text2Video submitters (A1, A2, B) for real mode with prompt templates
  - [x] Updated Deep Research submitters (A, B1, B2, Merge) for real mode with prompt templates
  - [x] Created type1_text2video/real/test_workflow_real.py
  - [x] Created type2_deep_research/real/test_workflow_real.py
  - [x] Real mode uses sentence/max_tokens instead of sleep_time
  - [x] Token estimation in metadata for performance prediction
- [x] Task 17: Testing Tools
  - [x] tools/experiment_runner.py - Unified runner for all experiment types
  - [x] tools/cli.py - Command-line interface with 4 commands
  - [x] ExperimentRunner class with run_text2video_simulation/real and run_deep_research_simulation/real methods
  - [x] CLI with argparse for all 4 experiment modes
  - [x] Subprocess execution with environment variable passing
- [x] Task 18: Documentation
  - [x] docs/QUICKSTART.md - 5-minute quick start guide (3 methods: CLI, direct Python, real cluster)
  - [x] docs/API.md - Complete API reference with examples for all classes
  - [x] docs/MIGRATION.md - Step-by-step migration guide from original scripts
  - [x] docs/TROUBLESHOOTING.md - Common issues and solutions
  - [x] configs/text2video_sim_config.yaml - Text2Video simulation template
  - [x] configs/text2video_real_config.yaml - Text2Video real mode template
  - [x] configs/deep_research_sim_config.yaml - Deep Research simulation template
  - [x] configs/deep_research_real_config.yaml - Deep Research real mode template

## Development Workflow

1. Implement one workflow type completely (Task 13)
2. Test with simulation mode
3. Implement second workflow type (Task 14) using same patterns
4. Add service management (Task 15)
5. Extend to real mode (Task 16)
6. Build validation tools (Task 17)
7. Complete documentation (Task 18)

## Notes

- All implementations should reference original code for behavioral consistency
- Use type hints throughout
- Add comprehensive docstrings
- Write tests alongside implementation
- Update README as components are completed
