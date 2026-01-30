# PyLet User Guide

> **Summary**: PyLet is a lightweight distributed task execution system for GPU servers. It provides job orchestration with resource-based scheduling, service discovery, and virtual environment support. It does NOT provide load balancing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Python API Reference](#5-python-api-reference)
6. [CLI Reference](#6-cli-reference)
7. [Configuration Files (TOML)](#7-configuration-files-toml)
8. [Environment Variables](#8-environment-variables)
9. [Features and Limitations](#9-features-and-limitations)
10. [Examples](#10-examples)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

### 1.1 What is PyLet

```
Type: Distributed task execution system
Target: GPU servers and ML workloads
Architecture: Head-Worker model
Persistence: SQLite with WAL mode
API Style: Sync-first (async available in pylet.aio)
License: Apache 2.0
Requirements: Python 3.9+, Linux
```

### 1.2 Key Capabilities

| Capability | Supported | Description |
|------------|-----------|-------------|
| Job Scheduling | Yes | Assign jobs to workers based on resource availability |
| GPU Allocation | Yes | Count-based or index-based GPU assignment |
| Service Discovery | Yes | PORT env var + endpoint retrieval |
| Virtual Environments | Yes | Run jobs in pre-existing venvs |
| TOML Configuration | Yes | Define jobs in config files |
| Graceful Shutdown | Yes | SIGTERM with configurable grace period |
| Log Management | Yes | Capture, store, retrieve stdout/stderr |
| Load Balancing | No | Not supported (see Section 9.2) |
| Job Dependencies | No | Instances are independent |
| Automatic Retries | No | Application must resubmit |

### 1.3 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HEAD NODE                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  REST API   │  │  Scheduler  │  │  SQLite Database    │  │
│  │  (FastAPI)  │  │  (Resource) │  │  (State Persistence)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
              Long-poll heartbeat protocol
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   WORKER 0   │    │   WORKER 1   │    │   WORKER 2   │
│  GPU: 0,1,2,3│    │  GPU: 0,1    │    │  GPU: 0,1,2,3│
│  CPU: 32     │    │  CPU: 16     │    │  CPU: 64     │
│  MEM: 128GB  │    │  MEM: 64GB   │    │  MEM: 256GB  │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 2. Installation

### 2.1 Using pip

```bash
pip install pylet
```

### 2.2 Using uv

```bash
uv add pylet
```

---

## 3. Quick Start

### 3.1 Start Head Node

```bash
# Terminal 1
pylet start
# Output: Head node running on http://0.0.0.0:8000
```

### 3.2 Start Worker Node

```bash
# Terminal 2
pylet start --head localhost:8000 --gpu-units 4
# Output: Worker registered with 4 GPUs
```

### 3.3 Submit a Job

```bash
# Terminal 3
pylet submit "nvidia-smi" --gpu-units 1 --name test-job
```

### 3.4 Python API

```python
import pylet

# Step 1: Initialize (required before any other calls)
pylet.init(address="http://localhost:8000")

# Step 2: Submit instance
instance = pylet.submit(
    command="python train.py",
    gpu=1,
    cpu=4,
    memory=8192,
    name="my-training"
)

# Step 3: Wait and get results
instance.wait()
print(f"Status: {instance.status}")
print(f"Exit code: {instance.exit_code}")
print(f"Logs: {instance.logs()}")
```

---

## 4. Core Concepts

### 4.1 Instance

An **instance** is a unit of work that runs a shell command on a worker node.

#### Instance Lifecycle States

```
PENDING ──► ASSIGNED ──► RUNNING ──► COMPLETED (exit_code = 0)
                │            │
                │            ├──► FAILED (exit_code != 0)
                │            │
                │            └──► CANCELLED (user requested)
                │
                └──► UNKNOWN (worker went offline)
```

| State | Description | Terminal |
|-------|-------------|----------|
| `PENDING` | Waiting to be scheduled | No |
| `ASSIGNED` | Scheduled to worker, not yet started | No |
| `RUNNING` | Actively executing | No |
| `COMPLETED` | Finished with exit code 0 | Yes |
| `FAILED` | Finished with non-zero exit code | Yes |
| `CANCELLED` | Terminated by user | Yes |
| `UNKNOWN` | Worker offline while running | No |

### 4.2 Worker

A **worker** is a node that executes instances. Workers register with the head node and report their resources.

#### Worker Health States

| State | Condition | Description |
|-------|-----------|-------------|
| `ONLINE` | Heartbeat received within 30s | Healthy |
| `SUSPECT` | No heartbeat for 30-90s | May be unhealthy |
| `OFFLINE` | No heartbeat for 90s+ | Assumed dead |

### 4.3 Resources

Three resource types are tracked:

| Resource | Unit | Parameter (Python) | Parameter (CLI) | Default |
|----------|------|-------------------|-----------------|---------|
| GPU | count | `gpu=N` | `--gpu-units N` | 0 |
| CPU | cores | `cpu=N` | `--cpu-cores N` | 1 |
| Memory | MB | `memory=N` | `--memory-mb N` | 512 |

### 4.4 Service Discovery

Each instance receives:
- `PORT` environment variable (range: 15600-15700)
- Accessible via `instance.endpoint` as `"host:port"`

---

## 5. Python API Reference

### 5.1 Module Functions

#### `pylet.init(address)`

Initialize the PyLet client. **Must be called before any other operations.**

```python
# Signature
def init(address: str = "http://localhost:8000") -> None

# Example
pylet.init(address="http://head-node:8000")
```

#### `pylet.submit(command, **kwargs)`

Submit a new instance.

```python
# Signature
def submit(
    command: str | list[str],
    *,
    name: str | None = None,
    gpu: int = 0,
    cpu: int = 1,
    memory: int = 512,
    # SLLM features
    target_worker: str | None = None,
    gpu_indices: list[int] | None = None,
    exclusive: bool = True,
    labels: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    # Venv support
    venv: str | None = None,
) -> Instance

# Example: Basic
instance = pylet.submit("echo hello")

# Example: With resources
instance = pylet.submit(
    command="python train.py --epochs 10",
    gpu=2,
    cpu=8,
    memory=16384,
    name="training-job"
)

# Example: With SLLM features
instance = pylet.submit(
    command="python inference.py",
    gpu_indices=[0, 1],
    target_worker="gpu-node-0",
    exclusive=True,
    labels={"model": "llama"},
    env={"HF_TOKEN": "xxx"}
)

# Example: With venv
instance = pylet.submit(
    command="python script.py",
    venv="/shared/venvs/ml-env"
)
```

**Parameter Details:**

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `command` | `str \| list[str]` | Shell command to execute | Required |
| `name` | `str \| None` | Instance name for lookup | Optional, must be unique |
| `gpu` | `int` | Number of GPUs | >= 0, mutually exclusive with `gpu_indices` |
| `cpu` | `int` | Number of CPU cores | >= 1 |
| `memory` | `int` | Memory in MB | >= 1 |
| `target_worker` | `str \| None` | Pin to specific worker hostname | Worker must exist |
| `gpu_indices` | `list[int] \| None` | Specific GPU indices | Mutually exclusive with `gpu` |
| `exclusive` | `bool` | GPU exclusivity mode | Default: True |
| `labels` | `dict[str, str] \| None` | Custom metadata | Key-value pairs |
| `env` | `dict[str, str] \| None` | Environment variables | Key-value pairs |
| `venv` | `str \| None` | Virtual environment path | Must be absolute path |

#### `pylet.get(name, *, id)`

Retrieve an instance by name or ID.

```python
# Signature
def get(name: str | None = None, *, id: str | None = None) -> Instance

# Example: By name
instance = pylet.get(name="my-job")

# Example: By ID
instance = pylet.get(id="550e8400-e29b-41d4-a716-446655440000")
```

#### `pylet.instances(status)`

List all instances, optionally filtered by status.

```python
# Signature
def instances(status: str | None = None) -> list[Instance]

# Example: All instances
all_instances = pylet.instances()

# Example: Only running
running = pylet.instances(status="RUNNING")
```

#### `pylet.workers()`

List all registered workers.

```python
# Signature
def workers() -> list[WorkerInfo]

# Example
for worker in pylet.workers():
    print(f"{worker.host}: {worker.gpu_available}/{worker.gpu} GPUs")
```

### 5.2 Instance Object

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique UUID |
| `name` | `str \| None` | User-provided name |
| `status` | `str` | Current lifecycle state |
| `display_status` | `str` | User-friendly status (shows "CANCELLING") |
| `endpoint` | `str \| None` | "host:port" when running |
| `exit_code` | `int \| None` | Exit code when terminal |
| `gpu_indices` | `list[int] \| None` | Allocated GPU indices |

#### Methods

```python
# Wait until RUNNING state
instance.wait_running(timeout: float = 300) -> None
# Raises: TimeoutError if not running within timeout

# Wait until terminal state
instance.wait(timeout: float | None = None) -> None
# Raises: TimeoutError, InstanceFailedError

# Request cancellation
instance.cancel() -> None

# Get logs
instance.logs(tail: int | None = None) -> str
# tail=None: full logs
# tail=1000: last 1000 bytes

# Refresh state from server
instance.refresh() -> None
```

### 5.3 WorkerInfo Object

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Worker UUID |
| `host` | `str` | IP address or hostname |
| `status` | `str` | ONLINE, SUSPECT, or OFFLINE |
| `gpu` | `int` | Total GPU count |
| `gpu_available` | `int` | Available GPUs |
| `cpu` | `int` | Total CPU cores |
| `cpu_available` | `int` | Available CPU cores |
| `memory` | `int` | Total memory (MB) |
| `memory_available` | `int` | Available memory (MB) |
| `gpu_indices_available` | `list[int]` | Available GPU indices |

### 5.4 Exceptions

```python
from pylet import (
    PyletError,              # Base exception
    NotInitializedError,     # pylet.init() not called
    NotFoundError,           # Instance/worker not found
    TimeoutError,            # Operation timed out
    InstanceFailedError,     # Instance FAILED/CANCELLED
    InstanceTerminatedError, # Invalid op on terminal instance
)
```

| Exception | Cause | Recovery |
|-----------|-------|----------|
| `NotInitializedError` | `pylet.init()` not called | Call `pylet.init(address)` |
| `NotFoundError` | Instance/worker doesn't exist | Check name/ID |
| `TimeoutError` | Operation exceeded timeout | Increase timeout or cancel |
| `InstanceFailedError` | Instance entered FAILED/CANCELLED | Check `e.instance.logs()` |
| `InstanceTerminatedError` | Operation on terminal instance | Submit new instance |

### 5.5 Async API

All functions are available in `pylet.aio` with async/await:

```python
import asyncio
import pylet.aio as pylet

async def main():
    await pylet.init(address="http://localhost:8000")

    instance = await pylet.submit("python train.py", gpu=1)
    await instance.wait_running()
    print(f"Endpoint: {instance.endpoint}")

    await instance.wait()
    print(f"Exit code: {instance.exit_code}")

asyncio.run(main())
```

---

## 6. CLI Reference

### 6.1 Cluster Management

#### Start Head Node

```bash
pylet start [--port PORT]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 8000 | HTTP server port |

#### Start Worker Node

```bash
pylet start --head HOST:PORT --gpu-units N [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--head` | Required | Head node address |
| `--gpu-units` | Required | Number of GPUs |
| `--cpu-cores` | Auto | Number of CPU cores |
| `--memory-mb` | Auto | Memory in MB |

### 6.2 Instance Management

#### Submit Instance

```bash
pylet submit COMMAND [OPTIONS]
pylet submit --config FILE [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `COMMAND` | Shell command (or use --config) |
| `--config`, `-c` | TOML config file path |
| `--name` | Instance name |
| `--gpu-units` | GPU count |
| `--cpu-cores` | CPU cores |
| `--memory-mb` | Memory in MB |
| `--target-worker` | Pin to worker |
| `--gpu-indices` | Specific GPUs (comma-separated) |
| `--exclusive/--no-exclusive` | GPU exclusivity |
| `--label` | Custom label (key=value, repeatable) |
| `--env` | Environment var (key=value, repeatable) |
| `--venv` | Virtual environment path |

#### Get Instance

```bash
pylet get-instance --name NAME
pylet get-instance --instance-id ID
```

#### Get Endpoint

```bash
pylet get-endpoint --name NAME
# Output: 192.168.1.5:15600
```

#### List Workers

```bash
pylet list-workers
```

#### Get Logs

```bash
pylet logs INSTANCE_ID [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--tail N` | Last N bytes |
| `--follow` | Real-time tail |

#### Cancel Instance

```bash
pylet cancel INSTANCE_ID
```

#### Get Result

```bash
pylet get-result INSTANCE_ID
```

---

## 7. Configuration Files (TOML)

### 7.1 Schema

```toml
# job.toml

# Required
command = "python train.py"              # String or array

# Optional
name = "my-job"                          # Defaults to filename

[resources]
gpus = 2                                 # GPU count (mutually exclusive with gpu_indices)
cpus = 8                                 # CPU cores
memory = "16Gi"                          # Memory with units
gpu_indices = [0, 1]                     # Specific GPUs (mutually exclusive with gpus)
exclusive = true                         # GPU exclusivity
target_worker = "gpu-node-0"             # Pin to worker

[env]
KEY = "value"                            # Static value
HF_TOKEN = "${HF_TOKEN}"                 # Shell env interpolation

[labels]
key = "value"                            # Custom metadata
```

### 7.2 Memory Format

| Format | Meaning | Example |
|--------|---------|---------|
| `Gi` or `G` | Gibibytes (×1024 MB) | `"16Gi"` → 16384 MB |
| `Mi` or `M` | Mebibytes | `"512Mi"` → 512 MB |
| `Ki` or `K` | Kibibytes (÷1024 MB) | `"1024Ki"` → 1 MB |
| `Ti` or `T` | Tebibytes | `"1Ti"` → 1048576 MB |
| Number | Megabytes | `"8192"` → 8192 MB |

### 7.3 Environment Variable Interpolation

```toml
[env]
# Full variable
TOKEN = "${HF_TOKEN}"

# Alternative syntax
TOKEN = "$HF_TOKEN"

# Partial interpolation
PATH = "/data/${USER}/output"
```

### 7.4 Precedence

```
CLI arguments > Config file > Defaults
```

### 7.5 Example Configurations

#### Minimal

```toml
command = "echo hello"
```

#### Training Job

```toml
name = "llama-finetune"
command = "torchrun --nproc_per_node=4 train.py"

[resources]
gpus = 4
cpus = 32
memory = "128Gi"

[env]
HF_TOKEN = "${HF_TOKEN}"
WANDB_API_KEY = "${WANDB_API_KEY}"

[labels]
project = "llama"
stage = "training"
```

#### Inference Service

```toml
name = "vllm-service"
command = "vllm serve meta-llama/Llama-3.1-8B --port $PORT"

[resources]
gpus = 1
memory = "32Gi"

[env]
HF_TOKEN = "${HF_TOKEN}"

[labels]
type = "inference"
```

#### Non-Exclusive GPU (SLLM Storage)

```toml
name = "sllm-store"
command = "sllm-store start --port $PORT"

[resources]
gpu_indices = [0, 1, 2, 3]
exclusive = false
target_worker = "storage-node"

[labels]
type = "storage"
```

---

## 8. Environment Variables

### 8.1 Path Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PYLET_DATA_DIR` | `~/.pylet` | Base data directory |
| `PYLET_DB_PATH` | `$DATA_DIR/pylet.db` | SQLite database |
| `PYLET_RUN_DIR` | `$DATA_DIR/run` | Process state |
| `PYLET_LOG_DIR` | `$DATA_DIR/logs` | Log storage |

### 8.2 Worker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PYLET_WORKER_PORT_MIN` | 15600 | Port range start |
| `PYLET_WORKER_PORT_MAX` | 15700 | Port range end |
| `PYLET_WORKER_HTTP_PORT` | 15599 | Worker HTTP server |

### 8.3 Controller Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PYLET_SUSPECT_THRESHOLD_SECONDS` | 30 | Worker suspect threshold |
| `PYLET_OFFLINE_THRESHOLD_SECONDS` | 90 | Worker offline threshold |
| `PYLET_LIVENESS_CHECK_INTERVAL` | 5 | Health check frequency |
| `PYLET_SCHEDULER_INTERVAL` | 2 | Scheduling loop frequency |
| `PYLET_HEARTBEAT_POLL_TIMEOUT` | 30.0 | Heartbeat timeout |

### 8.4 Graceful Shutdown

| Variable | Default | Description |
|----------|---------|-------------|
| `PYLET_DEFAULT_GRACE_PERIOD_SECONDS` | 30 | SIGTERM grace period |
| `PYLET_MAX_GRACE_PERIOD_SECONDS` | 300 | Maximum grace period |

### 8.5 Log Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PYLET_LOG_CHUNK_SIZE` | 10485760 | Log file size (10MB) |
| `PYLET_LOG_MAX_FILES` | 5 | Log file count |
| `PYLET_LOG_MAX_RESPONSE_SIZE` | 10485760 | Max response size |

---

## 9. Features and Limitations

### 9.1 Feature Matrix

| Feature | Status | Details |
|---------|--------|---------|
| GPU Scheduling | Supported | Count-based or index-based |
| CPU Scheduling | Supported | Core-based allocation |
| Memory Scheduling | Supported | MB-based allocation |
| Service Discovery | Supported | PORT env + endpoint API |
| Virtual Environments | Supported | Pre-existing venvs only |
| TOML Configuration | Supported | Single job per file |
| Graceful Shutdown | Supported | SIGTERM → SIGKILL |
| Log Capture | Supported | 50MB max per instance |
| Persistence | Supported | SQLite with crash recovery |
| Load Balancing | Not Supported | See Section 9.2 |
| Job Dependencies | Not Supported | Use application logic |
| Automatic Retries | Not Supported | Application resubmits |
| Multi-Worker Jobs | Not Supported | One worker per instance |
| Fractional GPU | Not Supported | Integer allocation only |
| Authentication | Not Supported | No auth between nodes |

### 9.2 Load Balancing (Not Supported)

**PyLet does NOT include load balancing.**

#### Scheduler vs Load Balancer

| Concept | Function | PyLet |
|---------|----------|-------|
| **Scheduler** | Assigns jobs to workers | Yes |
| **Load Balancer** | Distributes requests to services | No |

#### What's Missing

- Request distribution across replicas
- Health-based routing
- Round-robin / least-connections algorithms
- Automatic failover

#### Workarounds

**Option 1: External Load Balancer**

```python
import pylet

pylet.init()

# Deploy replicas
endpoints = []
for i in range(3):
    inst = pylet.submit(f"vllm serve model --port $PORT", gpu=1, name=f"vllm-{i}")
    inst.wait_running()
    endpoints.append(inst.endpoint)

# Use nginx/HAProxy to balance across endpoints
print(f"Configure load balancer with: {endpoints}")
```

**Option 2: Application-Level**

```python
import random

class RoundRobinBalancer:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.index = 0

    def next(self):
        ep = self.endpoints[self.index]
        self.index = (self.index + 1) % len(self.endpoints)
        return ep

balancer = RoundRobinBalancer(endpoints)
endpoint = balancer.next()
```

**Option 3: nginx Configuration**

```nginx
upstream vllm {
    least_conn;
    server 192.168.1.10:15600;
    server 192.168.1.11:15601;
    server 192.168.1.12:15602;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm;
    }
}
```

### 9.3 Resource Limitations

| Limitation | Value | Workaround |
|------------|-------|------------|
| Port range | 101 per worker | Deploy fewer services |
| Log retention | 50MB per instance | External log storage |
| SQLite scale | ~10K instances | Archive completed |
| Single head | No redundancy | Run on reliable node |

---

## 10. Examples

### 10.1 Basic Task Execution

```python
import pylet

pylet.init()

instance = pylet.submit("echo 'Hello, World!'")
instance.wait()

print(f"Output: {instance.logs()}")
# Output: Hello, World!
```

### 10.2 GPU Training Job

```python
import pylet

pylet.init()

instance = pylet.submit(
    command="python train.py --model bert --epochs 10",
    gpu=2,
    cpu=8,
    memory=32768,
    name="bert-training"
)

instance.wait()

if instance.status == "COMPLETED":
    print("Training succeeded")
else:
    print(f"Training failed: exit_code={instance.exit_code}")
    print(instance.logs())
```

### 10.3 Deploy Inference Service

```python
import pylet
import httpx

pylet.init()

# Deploy
instance = pylet.submit(
    command="vllm serve meta-llama/Llama-3.1-8B-Instruct --port $PORT",
    gpu=1,
    memory=32768,
    name="llama-service"
)

# Wait for service
instance.wait_running(timeout=300)
endpoint = instance.endpoint
print(f"Service at: {endpoint}")

# Use service
response = httpx.post(
    f"http://{endpoint}/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    },
    timeout=60
)
print(response.json())

# Cleanup
instance.cancel()
instance.wait()
```

### 10.4 Multi-GPU Tensor Parallelism

```python
import pylet

pylet.init()

instance = pylet.submit(
    command="vllm serve meta-llama/Llama-3.1-70B --port $PORT --tensor-parallel-size 4",
    gpu=4,
    memory=131072,
    name="llama-70b"
)

instance.wait_running(timeout=600)
print(f"70B model at: {instance.endpoint}")
```

### 10.5 Specific GPU Indices

```python
import pylet

pylet.init()

# Use specific GPUs
instance = pylet.submit(
    command="python train.py",
    gpu_indices=[0, 2],  # Use GPU 0 and GPU 2 specifically
    name="specific-gpu-job"
)

instance.wait()
```

### 10.6 Non-Exclusive GPU Sharing

```python
import pylet

pylet.init()

# Storage daemon shares GPUs
storage = pylet.submit(
    command="sllm-store start --port $PORT",
    gpu_indices=[0, 1, 2, 3],
    exclusive=False,  # Allow sharing
    target_worker="storage-node",
    name="storage-daemon"
)

storage.wait_running()

# Inference can share GPU 0
inference = pylet.submit(
    command="python inference.py --port $PORT",
    gpu_indices=[0],  # Shares with storage daemon
    name="inference"
)

inference.wait_running()
```

### 10.7 Using Virtual Environment

```python
import pylet

pylet.init()

instance = pylet.submit(
    command="python -c 'import torch; print(torch.__version__)'",
    venv="/shared/venvs/torch-2.0",
    name="torch-version"
)

instance.wait()
print(instance.logs())
```

### 10.8 Using TOML Configuration

**File: `job.toml`**

```toml
name = "config-job"
command = "python train.py"

[resources]
gpus = 2
memory = "16Gi"

[env]
HF_TOKEN = "${HF_TOKEN}"
```

**Submit:**

```bash
# Use config
pylet submit --config job.toml

# Override GPU count
pylet submit --config job.toml --gpu-units 4
```

### 10.9 Batch Job Submission

```python
import pylet

pylet.init()

configs = [
    {"lr": 0.001, "batch": 32},
    {"lr": 0.0001, "batch": 64},
    {"lr": 0.00001, "batch": 128},
]

instances = []
for i, cfg in enumerate(configs):
    inst = pylet.submit(
        f"python train.py --lr {cfg['lr']} --batch {cfg['batch']}",
        gpu=1,
        name=f"hparam-{i}",
        labels={"lr": str(cfg['lr']), "batch": str(cfg['batch'])}
    )
    instances.append(inst)

# Wait for all
for inst in instances:
    inst.wait()
    print(f"{inst.name}: {inst.status}")
```

### 10.10 Error Handling

```python
import pylet
from pylet import NotInitializedError, TimeoutError, InstanceFailedError, NotFoundError

# Handle initialization
try:
    pylet.submit("echo test")
except NotInitializedError:
    pylet.init()

# Handle timeout
try:
    instance = pylet.submit("python long_job.py", gpu=1)
    instance.wait(timeout=60)
except TimeoutError:
    print("Timed out, cancelling...")
    instance.cancel()
    instance.wait()

# Handle failure
try:
    instance = pylet.submit("python might_fail.py", gpu=1)
    instance.wait()
except InstanceFailedError as e:
    print(f"Failed: exit_code={e.instance.exit_code}")
    print(e.instance.logs())

# Handle not found
try:
    inst = pylet.get(name="nonexistent")
except NotFoundError:
    print("Instance not found")
```

### 10.11 Local Cluster for Testing

```python
import pylet

with pylet.local_cluster(workers=2, gpu_per_worker=1) as cluster:
    # pylet auto-initialized

    inst1 = pylet.submit("nvidia-smi", gpu=1)
    inst2 = pylet.submit("nvidia-smi", gpu=1)

    inst1.wait()
    inst2.wait()

    print(inst1.logs())
    print(inst2.logs())
# Cluster auto-cleaned
```

### 10.12 Async API

```python
import asyncio
import pylet.aio as pylet

async def main():
    await pylet.init()

    # Submit multiple concurrently
    instances = await asyncio.gather(
        pylet.submit("task1.py", gpu=1),
        pylet.submit("task2.py", gpu=1),
        pylet.submit("task3.py", gpu=1),
    )

    # Wait for all
    await asyncio.gather(*[inst.wait() for inst in instances])

    for inst in instances:
        print(f"{inst.name}: {inst.status}")

asyncio.run(main())
```

### 10.13 Monitor Running Instance

```python
import pylet
import time

pylet.init()

instance = pylet.submit("python long_training.py", gpu=1, name="monitor-demo")
instance.wait_running()

# Poll status and logs
while instance.status == "RUNNING":
    instance.refresh()
    print(f"Status: {instance.status}")
    print(f"Recent logs:\n{instance.logs(tail=500)}")
    time.sleep(30)

print(f"Final: {instance.status}, exit={instance.exit_code}")
```

---

## 11. Troubleshooting

### 11.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `NotInitializedError` | `pylet.init()` not called | Call `pylet.init(address)` first |
| `NotFoundError` | Wrong name/ID | Verify instance exists with `pylet.instances()` |
| `TimeoutError` | Operation too slow | Increase timeout or check resources |
| Instance stuck `PENDING` | No worker has resources | Check `pylet.workers()` for availability |
| Instance `UNKNOWN` | Worker went offline | Check worker health, may recover |

### 11.2 Debugging Commands

```bash
# Check worker status
pylet list-workers

# Check all instances
pylet get-instance --name NAME

# Get logs
pylet logs INSTANCE_ID --tail 1000

# Follow logs real-time
pylet logs INSTANCE_ID --follow
```

### 11.3 Resource Issues

**Instance stuck in PENDING:**

```python
# Check available resources
workers = pylet.workers()
for w in workers:
    print(f"{w.host}: GPU {w.gpu_available}/{w.gpu}, CPU {w.cpu_available}/{w.cpu}")
```

**GPU not available:**

```python
# Check specific GPU indices
for w in pylet.workers():
    print(f"{w.host}: Available GPUs: {w.gpu_indices_available}")
```

---

## See Also

- [API Reference](api_reference.md) - Detailed API documentation
- [CLI Reference](cli_reference.md) - Complete CLI command reference
