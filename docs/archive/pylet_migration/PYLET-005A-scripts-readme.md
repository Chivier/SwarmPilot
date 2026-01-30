# PYLET-005A: Scripts Module README & Quick Start

## Objective

Create comprehensive README documentation for the `scripts/` module that enables users to run model deployment scripts with minimal configuration. This provides a standalone entry point for Phase 1 functionality.

## Prerequisites

- PYLET-001 through PYLET-004 completed
- Model startup scripts implemented
- Registration and signal handling working

## Background

After Phase 1 implementation, users should be able to:
1. Deploy a model directly via PyLet using provided scripts
2. Understand the scripts' purpose and configuration options
3. Run a minimal example without deep knowledge of the system

## Files to Create/Modify

```
scripts/
├── README.md                    # NEW: Module documentation
├── start_model.sh               # EXISTS: Model startup script
├── register_with_scheduler.py   # EXISTS: Registration script
└── examples/
    └── quickstart.sh            # NEW: Minimal example
```

## Implementation Steps

### Step 1: Create Scripts README

Create `scripts/README.md`:

```markdown
# SwarmPilot Model Scripts

Scripts for deploying and managing model services via PyLet.

## Quick Start

### Prerequisites

- PyLet cluster running (head + worker nodes)
- Model weights available (e.g., `Qwen/Qwen3-0.6B`)
- Scheduler service running (for registration)

### Minimal Deployment

```bash
# Set required environment variables
export MODEL_ID="Qwen/Qwen3-0.6B"
export MODEL_BACKEND="vllm"
export SCHEDULER_URL="http://localhost:8000"
export PYLET_HEAD="http://localhost:8000"

# Deploy via PyLet
pylet submit "bash scripts/start_model.sh" \
    --gpu 1 \
    --env MODEL_ID="$MODEL_ID" \
    --env MODEL_BACKEND="$MODEL_BACKEND" \
    --env SCHEDULER_URL="$SCHEDULER_URL"
```

### Verify Deployment

```bash
# Check instance status
pylet list

# Test model health (replace with actual endpoint)
curl http://<instance-endpoint>/health

# Test inference
curl -X POST http://<instance-endpoint>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Hello", "max_tokens": 10}'
```

## Scripts Reference

### start_model.sh

Main entry point for model deployment. Starts the model service and handles registration.

**Environment Variables:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_ID` | Yes | - | Model identifier (e.g., `Qwen/Qwen3-0.6B`) |
| `MODEL_BACKEND` | No | `vllm` | Backend engine (`vllm` or `sglang`) |
| `SCHEDULER_URL` | No | `http://localhost:8000` | Scheduler URL for registration |
| `PORT` | Yes (auto) | - | Port allocated by PyLet (set automatically) |
| `HOSTNAME` | No | `localhost` | Hostname for registration |

**Signal Handling:**
- `SIGTERM`: Graceful shutdown, deregisters from scheduler
- `SIGINT`: Same as SIGTERM

### register_with_scheduler.py

Handles model registration with the scheduler after startup.

**Usage:**
```bash
# Usually called by start_model.sh, but can be run manually
python scripts/register_with_scheduler.py
```

**Environment Variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `SCHEDULER_URL` | Yes | Scheduler endpoint |
| `MODEL_ID` | Yes | Model identifier |
| `PORT` | Yes | Model service port |
| `HOSTNAME` | No | Override hostname (default: system hostname) |

## Configuration Examples

### vLLM Backend

```bash
export MODEL_ID="meta-llama/Llama-2-7b-hf"
export MODEL_BACKEND="vllm"
export VLLM_EXTRA_ARGS="--tensor-parallel-size 2"

pylet submit "bash scripts/start_model.sh" --gpu 2
```

### SGLang Backend

```bash
export MODEL_ID="meta-llama/Llama-2-7b-hf"
export MODEL_BACKEND="sglang"

pylet submit "bash scripts/start_model.sh" --gpu 1
```

### Custom Port Range

PyLet automatically allocates ports, but you can configure the range:

```bash
# On PyLet worker startup
pylet start --port-range 8000-9000
```

## Troubleshooting

### Model Fails to Start

1. Check PyLet logs: `pylet logs <instance-id>`
2. Verify GPU availability: `nvidia-smi`
3. Check model weights exist

### Registration Fails

1. Verify scheduler is running: `curl $SCHEDULER_URL/health`
2. Check network connectivity between worker and scheduler
3. Review registration script logs

### Health Check Fails

1. Wait for model to fully load (can take several minutes for large models)
2. Check model process is running: `ps aux | grep vllm`
3. Verify port is listening: `netstat -tlnp | grep $PORT`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PyLet Worker                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  start_model.sh                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │  │
│  │  │   Model     │    │ Registration│    │  Signal   │  │  │
│  │  │  (vLLM)     │◀──▶│   Script    │    │  Handler  │  │  │
│  │  │  :$PORT     │    │             │    │           │  │  │
│  │  └──────┬──────┘    └──────┬──────┘    └─────┬─────┘  │  │
│  │         │                  │                  │        │  │
│  └─────────┼──────────────────┼──────────────────┼────────┘  │
│            │                  │                  │           │
└────────────┼──────────────────┼──────────────────┼───────────┘
             │                  │                  │
             ▼                  ▼                  ▼
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │   Clients    │   │  Scheduler   │   │    PyLet     │
      │  (requests)  │   │ (routing)    │   │   (cancel)   │
      └──────────────┘   └──────────────┘   └──────────────┘
```

## See Also

- [PyLet Documentation](/home/yanweiye/Projects/pylet/README.md)
- [Migration Guide](../docs/pylet_migration.md)
- [Phase 1 Tasks](../docs/pylet_migration/)
```

### Step 2: Create Quick Start Example

Create `scripts/examples/quickstart.sh`:

```bash
#!/bin/bash
# Quick Start: Deploy a model via PyLet
#
# Prerequisites:
#   - PyLet cluster running
#   - Scheduler running at SCHEDULER_URL
#
# Usage:
#   ./scripts/examples/quickstart.sh [MODEL_ID]

set -e

# Configuration (override with environment variables)
MODEL_ID="${1:-Qwen/Qwen3-0.6B}"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
SCHEDULER_URL="${SCHEDULER_URL:-http://localhost:8000}"
PYLET_HEAD="${PYLET_HEAD:-http://localhost:8000}"
GPU_COUNT="${GPU_COUNT:-1}"

echo "=== SwarmPilot Quick Start ==="
echo "Model: $MODEL_ID"
echo "Backend: $MODEL_BACKEND"
echo "Scheduler: $SCHEDULER_URL"
echo "PyLet Head: $PYLET_HEAD"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v pylet &> /dev/null; then
    echo "ERROR: pylet command not found"
    echo "Install: pip install pylet"
    exit 1
fi

if ! curl -s "$SCHEDULER_URL/health" > /dev/null 2>&1; then
    echo "WARNING: Scheduler not responding at $SCHEDULER_URL"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Initialize PyLet client
echo "Connecting to PyLet..."
pylet init "$PYLET_HEAD"

# Submit model instance
echo "Submitting model instance..."
INSTANCE_ID=$(pylet submit "bash scripts/start_model.sh" \
    --gpu "$GPU_COUNT" \
    --name "${MODEL_ID//\//-}-quickstart" \
    --env MODEL_ID="$MODEL_ID" \
    --env MODEL_BACKEND="$MODEL_BACKEND" \
    --env SCHEDULER_URL="$SCHEDULER_URL" \
    --json | jq -r '.id')

echo "Instance submitted: $INSTANCE_ID"

# Wait for running
echo "Waiting for instance to start..."
pylet wait "$INSTANCE_ID" --timeout 300

# Get endpoint
ENDPOINT=$(pylet show "$INSTANCE_ID" --json | jq -r '.endpoint')
echo "Instance running at: $ENDPOINT"

# Wait for health
echo "Waiting for model to be healthy..."
for i in {1..60}; do
    if curl -s "http://$ENDPOINT/health" > /dev/null 2>&1; then
        echo "Model is healthy!"
        break
    fi
    echo "  Waiting... ($i/60)"
    sleep 5
done

# Test inference
echo ""
echo "=== Testing Inference ==="
curl -s -X POST "http://$ENDPOINT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_ID\", \"prompt\": \"Hello, world!\", \"max_tokens\": 20}" \
    | jq .

echo ""
echo "=== Quick Start Complete ==="
echo "Instance ID: $INSTANCE_ID"
echo "Endpoint: http://$ENDPOINT"
echo ""
echo "To stop: pylet cancel $INSTANCE_ID"
```

### Step 3: Update start_model.sh Header

Add documentation header to `scripts/start_model.sh`:

```bash
#!/bin/bash
# =============================================================================
# start_model.sh - Start model service via PyLet
# =============================================================================
#
# This script is the main entry point for deploying model services on PyLet
# workers. It handles:
#   1. Starting the model service (vLLM or SGLang)
#   2. Registering with the scheduler after model is healthy
#   3. Graceful shutdown on SIGTERM/SIGINT
#
# ENVIRONMENT VARIABLES:
#   MODEL_ID       (required) - Model identifier (e.g., "Qwen/Qwen3-0.6B")
#   MODEL_BACKEND  (optional) - Backend engine: "vllm" or "sglang" (default: vllm)
#   SCHEDULER_URL  (optional) - Scheduler URL (default: http://localhost:8000)
#   PORT           (required) - Auto-set by PyLet
#
# USAGE:
#   # Via PyLet
#   pylet submit "bash scripts/start_model.sh" --gpu 1 --env MODEL_ID=...
#
#   # Direct (for testing)
#   PORT=8001 MODEL_ID=test ./scripts/start_model.sh
#
# SEE ALSO:
#   scripts/README.md - Full documentation
#   docs/pylet_migration/PYLET-001-direct-model-deployment.md
#
# =============================================================================

set -e
# ... rest of script
```

## Test Strategy

### Documentation Tests

```bash
# Verify README renders correctly
cat scripts/README.md | head -50

# Check all links are valid
grep -E '\[.*\]\(.*\)' scripts/README.md | while read link; do
    # Extract path and verify exists
    path=$(echo "$link" | sed 's/.*(\(.*\))/\1/')
    if [[ ! -f "$path" && ! -d "$path" ]]; then
        echo "WARNING: Broken link: $path"
    fi
done
```

### Quick Start Test

```bash
# Run quick start in dry-run mode
DRYRUN=1 ./scripts/examples/quickstart.sh Qwen/Qwen3-0.6B
```

## Acceptance Criteria

- [ ] scripts/README.md created with:
  - [ ] Quick Start section
  - [ ] Environment variables reference
  - [ ] Troubleshooting guide
  - [ ] Architecture diagram
- [ ] scripts/examples/quickstart.sh created and executable
- [ ] start_model.sh has documentation header
- [ ] All documentation links are valid
- [ ] Quick start example works end-to-end

## Next Steps

Proceed to [PYLET-005](PYLET-005-phase1-integration-tests.md) for integration tests.

## Code References

- Model startup: [scripts/start_model.sh](../../scripts/start_model.sh)
- Registration: [scripts/register_with_scheduler.py](../../scripts/register_with_scheduler.py)
