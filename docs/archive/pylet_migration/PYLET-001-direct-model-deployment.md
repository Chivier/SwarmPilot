# PYLET-001: Direct Model Deployment

## Objective

Deploy model services (vLLM, sglang) directly via PyLet without a wrapper layer. Models expose their native HTTP API on PyLet-assigned ports.

## Prerequisites

- PyLet development environment set up
- Model services (vLLM, sglang) installed
- Understanding of PyLet submit API

## Background

Instead of wrapping model services, we deploy them directly via PyLet:
- PyLet manages process lifecycle (start, stop, monitor)
- Model exposes its native HTTP API on `$PORT`
- Scheduler routes requests directly to model endpoints

This simplifies the architecture by removing the wrapper layer.

## Implementation Steps

### Step 1: Define Model Launch Commands

Create launch command templates for each model type:

```python
# Model launch commands
MODEL_COMMANDS = {
    "vllm": "vllm serve {model_id} --port $PORT --host 0.0.0.0",
    "sglang": "python -m sglang.launch_server --model {model_id} --port $PORT --host 0.0.0.0",
}
```

### Step 2: Deploy via PyLet

```python
import pylet

def deploy_model(model_id: str, backend: str = "vllm", gpu_count: int = 1) -> pylet.Instance:
    """Deploy a model directly via PyLet.

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen3-0.6B")
        backend: Model backend ("vllm" or "sglang")
        gpu_count: Number of GPUs to allocate

    Returns:
        PyLet instance handle
    """
    command_template = MODEL_COMMANDS[backend]
    command = command_template.format(model_id=model_id)

    instance = pylet.submit(
        command,
        gpu=gpu_count,
        name=f"{model_id.replace('/', '-')}-{backend}",
        labels={
            "model_id": model_id,
            "backend": backend,
            "managed_by": "swarmpilot",
        },
    )

    return instance
```

### Step 3: Wait for Model Ready

```python
async def wait_model_ready(instance: pylet.Instance, timeout: float = 300.0) -> str:
    """Wait for model to be ready and return endpoint.

    Args:
        instance: PyLet instance handle
        timeout: Maximum wait time in seconds

    Returns:
        Model endpoint (host:port)
    """
    import httpx
    import asyncio

    # Wait for instance to be running
    instance.wait_running(timeout=timeout)
    endpoint = instance.endpoint

    # Wait for model health check
    async with httpx.AsyncClient(timeout=10.0) as client:
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                response = await client.get(f"http://{endpoint}/health")
                if response.status_code == 200:
                    return endpoint
            except httpx.RequestError:
                pass
            await asyncio.sleep(2.0)

    raise TimeoutError(f"Model at {endpoint} did not become healthy")
```

### Step 4: Test Deployment

```python
# Test script
import pylet

pylet.init("http://localhost:8000")

# Deploy model
instance = deploy_model("Qwen/Qwen3-0.6B", backend="vllm", gpu_count=1)
print(f"Instance submitted: {instance.id}")

# Wait for ready
endpoint = wait_model_ready(instance)
print(f"Model ready at: {endpoint}")

# Test inference
import httpx
response = httpx.post(
    f"http://{endpoint}/v1/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "Hello, world!",
        "max_tokens": 10,
    }
)
print(f"Response: {response.json()}")

# Cleanup
instance.cancel()
```

## Test Strategy

### Unit Tests

```python
def test_deploy_model_command():
    """Test command generation."""
    command = MODEL_COMMANDS["vllm"].format(model_id="test/model")
    assert "--port $PORT" in command
    assert "test/model" in command

def test_deploy_model_labels():
    """Test instance labels."""
    # Mock pylet.submit
    instance = deploy_model("test/model", backend="vllm")
    assert instance.labels["model_id"] == "test/model"
    assert instance.labels["backend"] == "vllm"
```

### Integration Tests

```bash
# Start PyLet cluster
pylet start &
pylet start --head localhost:8000 &

# Run deployment test
python test_direct_deployment.py
```

## Acceptance Criteria

- [ ] Model deploys via PyLet submit
- [ ] Model accessible on PyLet-assigned port
- [ ] Health check endpoint works
- [ ] Labels correctly applied
- [ ] Graceful shutdown on cancel

## Next Steps

Proceed to [PYLET-002](PYLET-002-model-registration.md) for model registration with scheduler.

## Code References

- PyLet submit API: [pylet/_sync_api.py](/home/yanweiye/Projects/pylet/pylet/_sync_api.py)
