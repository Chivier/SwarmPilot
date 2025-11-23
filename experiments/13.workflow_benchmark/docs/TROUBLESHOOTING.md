# Troubleshooting Guide

Common issues and solutions for the Unified Workflow Benchmark Framework.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Network & Connection Issues](#network--connection-issues)
- [Performance Issues](#performance-issues)
- [Metrics & Output Issues](#metrics--output-issues)
- [Service Management Issues](#service-management-issues)
- [Workflow-Specific Issues](#workflow-specific-issues)
- [Debugging Techniques](#debugging-techniques)

---

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'common'"

**Symptom**:
```bash
$ python type1_text2video/simulation/test_workflow_sim.py
ModuleNotFoundError: No module named 'common'
```

**Cause**: Python path doesn't include the parent directory.

**Solution**:
```bash
# Option 1: Run from correct directory
cd experiments/13.workflow_benchmark
python type1_text2video/simulation/test_workflow_sim.py

# Option 2: Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/experiments/13.workflow_benchmark
python type1_text2video/simulation/test_workflow_sim.py

# Option 3: Use absolute imports (already in scripts)
# Scripts include this at the top:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

---

### Issue: "ModuleNotFoundError: No module named 'websockets'"

**Symptom**:
```bash
ModuleNotFoundError: No module named 'websockets'
```

**Cause**: Missing dependencies.

**Solution**:
```bash
# Install dependencies
uv sync

# Or with pip
pip install websockets requests numpy
```

---

### Issue: "uv: command not found"

**Symptom**:
```bash
bash: uv: command not found
```

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version
```

---

## Configuration Issues

### Issue: Environment Variables Not Loaded

**Symptom**:
- Using default values instead of configured values
- `QPS=2.0` when expecting `QPS=3.0`

**Diagnosis**:
```python
import os
print(f"QPS: {os.getenv('QPS', 'NOT SET')}")
print(f"DURATION: {os.getenv('DURATION', 'NOT SET')}")
```

**Solution**:
```bash
# Verify variables are exported
export QPS=3.0
export DURATION=600
echo $QPS  # Should print 3.0

# Or source a config file
cat > experiment.env <<EOF
export QPS=3.0
export DURATION=600
export NUM_WORKFLOWS=1000
EOF
source experiment.env
```

---

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'captions_10k.json'"

**Symptom**:
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'captions_10k.json'
```

**Cause**: Caption file not found at expected path.

**Solution**:
```bash
# Check if file exists
ls -la captions_10k.json

# Option 1: Use absolute path
export CAPTION_FILE=/absolute/path/to/captions_10k.json

# Option 2: Copy to correct location
cp /path/to/captions_10k.json experiments/13.workflow_benchmark/

# Option 3: Create symlink
ln -s /path/to/captions_10k.json experiments/13.workflow_benchmark/captions_10k.json
```

---

### Issue: Invalid Configuration Values

**Symptom**:
```bash
ValueError: QPS must be positive
```

**Diagnosis**:
```python
from type1_text2video.config import Text2VideoConfig

config = Text2VideoConfig.from_env()
print(f"QPS: {config.qps}")
print(f"Duration: {config.duration}")
print(f"Num Workflows: {config.num_workflows}")
```

**Solution**:
```bash
# Ensure valid values
export QPS=2.0           # Must be > 0
export DURATION=300      # Must be > 0
export NUM_WORKFLOWS=600 # Must be > 0
export MAX_B_LOOPS=4     # Must be 1-4 (Text2Video)
export FANOUT_COUNT=3    # Must be > 0 (Deep Research)
```

---

## Network & Connection Issues

### Issue: "ConnectionRefusedError: [Errno 111] Connection refused"

**Symptom**:
```bash
ConnectionRefusedError: [Errno 111] Connection refused
```

**Cause**: Scheduler services not running or wrong URL.

**Diagnosis**:
```bash
# Check if services are listening
netstat -tuln | grep 8100
netstat -tuln | grep 8101

# Test connection
curl http://localhost:8100/health
curl http://localhost:8101/health
```

**Solution**:
```bash
# Option 1: Start services
cd experiments/03.Exp4.Text2Video
./start_all_services.sh

# Option 2: Check configuration
export SCHEDULER_A_URL=http://localhost:8100
export SCHEDULER_B_URL=http://localhost:8101

# Option 3: Verify services are running
ps aux | grep scheduler
docker ps  # If using Docker
```

---

### Issue: "WebSocket connection failed: HTTP 404 Not Found"

**Symptom**:
```bash
WebSocket connection failed: HTTP 404
```

**Cause**: Incorrect WebSocket endpoint.

**Diagnosis**:
```python
# Check WebSocket URL construction
scheduler_url = "http://localhost:8100"
ws_url = scheduler_url.replace("http://", "ws://").replace("https://", "wss://")
print(f"WebSocket URL: {ws_url}/results")
```

**Solution**:
```bash
# Ensure correct URL format
# HTTP: http://localhost:8100
# WebSocket: ws://localhost:8100/results

# Test with wscat (if available)
npm install -g wscat
wscat -c ws://localhost:8100/results
```

---

### Issue: "WebSocket connection timeout"

**Symptom**:
```bash
asyncio.TimeoutError: WebSocket connection timeout
```

**Cause**: Network latency or service overload.

**Solution**:
```python
# Increase timeout in config
# type1_text2video/config.py
@dataclass
class Text2VideoConfig:
    websocket_timeout: int = 600  # Increase from 300 to 600

# Or via environment
export WEBSOCKET_TIMEOUT=600
```

---

### Issue: "Too many open files"

**Symptom**:
```bash
OSError: [Errno 24] Too many open files
```

**Cause**: File descriptor limit reached.

**Diagnosis**:
```bash
# Check current limit
ulimit -n

# Check process file descriptors
lsof -p <pid> | wc -l
```

**Solution**:
```bash
# Increase limit temporarily
ulimit -n 4096

# Increase limit permanently
echo "* soft nofile 4096" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 8192" | sudo tee -a /etc/security/limits.conf

# Logout and login again
```

---

## Performance Issues

### Issue: Lower Than Expected Throughput

**Symptom**: Achieved QPS is significantly lower than configured.

**Diagnosis**:
```bash
# Check metrics
cat output/metrics.json | python -m json.tool | grep -A5 "throughput"

# Expected output:
# "throughput": {
#   "achieved_qps": 1.95,  # Should be ~2.0
#   "target_qps": 2.0,
#   "duration": 298.5
# }
```

**Possible Causes & Solutions**:

1. **Rate Limiter Issue**:
```python
# Check rate limiter configuration
rate_limiter = RateLimiter(rate=2.0)
# Burst size should be rate * 2

# Test rate limiter
import time
start = time.time()
for i in range(10):
    rate_limiter.wait()
elapsed = time.time() - start
print(f"10 tokens in {elapsed:.2f}s (expected ~{10/2.0:.2f}s)")
```

2. **CPU Contention**:
```bash
# Check CPU usage
top -p $(pgrep -f test_workflow)

# Reduce concurrent processes
export NUM_WORKFLOWS=300  # Reduce from 600
```

3. **Network Latency**:
```bash
# Measure latency to scheduler
ping -c 10 localhost
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8100/health
```

---

### Issue: High Memory Usage

**Symptom**: Process memory grows unbounded.

**Diagnosis**:
```bash
# Monitor memory
top -p $(pgrep -f test_workflow)
# Or
ps aux | grep test_workflow

# Check Python memory profile
pip install memory_profiler
python -m memory_profiler type1_text2video/simulation/test_workflow_sim.py
```

**Solution**:
```python
# Reduce workflow count
export NUM_WORKFLOWS=300  # Instead of 600

# Clear completed workflows periodically
# In receivers.py
if len(completed_workflows) > 100:
    completed_workflows.clear()

# Use generators instead of lists
# In submitters.py
def _get_next_task_data(self):
    # Instead of storing all workflows in list
    return generate_next_workflow()  # Generator
```

---

### Issue: Thread Deadlock

**Symptom**: Experiment hangs indefinitely.

**Diagnosis**:
```bash
# Get thread dump
kill -3 <pid>  # Sends SIGQUIT
# Or use py-spy
pip install py-spy
sudo py-spy dump --pid <pid>
```

**Solution**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add timeout to locks
# In receivers.py
with self.state_lock.acquire(timeout=10):
    if not acquired:
        logger.error("Failed to acquire lock")
        return
```

---

## Metrics & Output Issues

### Issue: "No such file or directory: 'output/metrics.json'"

**Symptom**:
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'output/metrics.json'
```

**Cause**: Output directory doesn't exist.

**Solution**:
```bash
# Create output directory
mkdir -p output

# Or set OUTPUT_DIR
export OUTPUT_DIR=./results
mkdir -p $OUTPUT_DIR
```

---

### Issue: Empty or Incomplete Metrics

**Symptom**: `metrics.json` exists but is empty or missing data.

**Diagnosis**:
```bash
# Check file size
ls -lh output/metrics.json

# View contents
cat output/metrics.json | python -m json.tool
```

**Possible Causes & Solutions**:

1. **Experiment Duration Too Short**:
```bash
# Increase duration
export DURATION=60  # At least 60 seconds for meaningful results
```

2. **Metrics Not Exported**:
```python
# Ensure export is called at end
# In test_workflow_sim.py
metrics.export_to_json(config.output_dir / config.metrics_file)

# Check for errors during export
try:
    metrics.export_to_json(filepath)
except Exception as e:
    logger.error(f"Failed to export metrics: {e}")
```

3. **Permission Issues**:
```bash
# Check write permissions
ls -la output/
chmod 755 output
```

---

### Issue: Metrics Don't Match Original Implementation

**Symptom**: Task counts or latencies differ from original scripts.

**Diagnosis**:
```bash
# Compare side by side
diff <(jq -S . original/output/metrics.json) \
     <(jq -S . new/output/metrics.json)

# Check specific fields
jq '.tasks.A1.count' original/output/metrics.json
jq '.tasks.A1.count' new/output/metrics.json
```

**Solution**:
```python
# Use validation tool
from tools.metrics_validator import MetricsValidator

validator = MetricsValidator()
comparison = validator.compare_metrics(
    "original/output/metrics.json",
    "new/output/metrics.json",
    tolerance=0.05  # Allow 5% difference
)
print(comparison.summary())
```

---

## Service Management Issues

### Issue: "Port already in use"

**Symptom**:
```bash
OSError: [Errno 98] Address already in use
```

**Diagnosis**:
```bash
# Check what's using the port
lsof -i :8100
netstat -tuln | grep 8100
```

**Solution**:
```bash
# Option 1: Kill existing process
lsof -ti :8100 | xargs kill -9

# Option 2: Use different port
export SCHEDULER_A_PORT=8200
export SCHEDULER_A_URL=http://localhost:8200

# Option 3: Wait for port to be released
# Ports may take 60s to be released after process termination
```

---

### Issue: Services Start But Don't Respond

**Symptom**: Services appear to be running but health checks fail.

**Diagnosis**:
```bash
# Check process status
ps aux | grep scheduler

# Check logs
tail -f /tmp/scheduler_a.log
tail -f /tmp/scheduler_b.log

# Test health endpoint
curl -v http://localhost:8100/health
```

**Solution**:
```bash
# Restart services with verbose logging
export LOG_LEVEL=DEBUG
./start_all_services.sh

# Check for errors in logs
grep -i error /tmp/scheduler_*.log

# Verify GPU availability (if using real models)
nvidia-smi
```

---

## Workflow-Specific Issues

### Text2Video: B Tasks Not Looping

**Symptom**: Only one B task executed per workflow instead of 1-4.

**Diagnosis**:
```python
# Check workflow state
print(f"B loop count: {workflow_data.b_loop_count}")
print(f"Max B loops: {workflow_data.max_b_loops}")

# Check B receiver logic
# In type1_text2video/receivers.py:BTaskReceiver._process_result
logger.debug(f"B loop {workflow_data.b_loop_count}/{workflow_data.max_b_loops}")
```

**Solution**:
```python
# Verify B receiver logic
async def _process_result(self, data):
    workflow_data.b_loop_count += 1

    if workflow_data.b_loop_count < workflow_data.max_b_loops:
        # Re-submit B task
        self.b_submitter.add_task(...)
    else:
        # Complete workflow
        metrics.record_workflow_completed(...)
```

---

### Deep Research: Merge Not Triggered

**Symptom**: B1 and B2 tasks complete but Merge never starts.

**Diagnosis**:
```python
# Check synchronization counter
print(f"B2 count: {workflow_data.b2_count}")
print(f"Expected: {workflow_data.expected_b2_count}")

# Check merge trigger logic
# In type2_deep_research/receivers.py:B2TaskReceiver._process_result
logger.debug(f"B2 completed: {workflow_data.b2_count}/{workflow_data.expected_b2_count}")
```

**Solution**:
```python
# Fix synchronization logic
async def _process_result(self, data):
    with self.state_lock:
        workflow_data.b2_count += 1

        # Trigger merge when all B2 complete
        if workflow_data.b2_count == workflow_data.expected_b2_count:
            if not workflow_data.merge_triggered:  # Prevent duplicate
                workflow_data.merge_triggered = True
                self.merge_submitter.add_task(...)
```

---

### Deep Research: Duplicate Merge Tasks

**Symptom**: Multiple merge tasks submitted for same workflow.

**Cause**: Race condition in synchronization check.

**Solution**:
```python
# Add flag to prevent duplicates
@dataclass
class DeepResearchWorkflowData:
    merge_triggered: bool = False

# Check flag before triggering
async def _process_result(self, data):
    with self.state_lock:
        workflow_data.b2_count += 1

        if (workflow_data.b2_count == workflow_data.expected_b2_count and
            not workflow_data.merge_triggered):
            workflow_data.merge_triggered = True
            self.merge_submitter.add_task(...)
```

---

## Debugging Techniques

### Enable Debug Logging

```python
import logging

# Set logging level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or via environment
export LOG_LEVEL=DEBUG
```

### Use Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use ipdb for better interface
pip install ipdb
import ipdb; ipdb.set_trace()

# Python 3.7+
breakpoint()
```

### Profile Performance

```python
# CPU profiling
python -m cProfile -o profile.stats type1_text2video/simulation/test_workflow_sim.py

# View results
python -m pstats profile.stats
> sort cumtime
> stats 20

# Or use py-spy for live profiling
pip install py-spy
sudo py-spy top --pid <pid>
sudo py-spy record -o profile.svg --pid <pid>
```

### Memory Profiling

```python
# Install memory_profiler
pip install memory_profiler

# Add decorator to functions
from memory_profiler import profile

@profile
def my_function():
    ...

# Run with profiler
python -m memory_profiler script.py
```

### Network Debugging

```bash
# Capture WebSocket traffic
tcpdump -i lo -A 'tcp port 8100'

# Or use Wireshark
wireshark -i lo -f 'tcp port 8100'

# Test HTTP endpoints
curl -v http://localhost:8100/health
curl -X POST -H "Content-Type: application/json" \
     -d '{"task_id": "test-1", ...}' \
     http://localhost:8100/task/submit
```

### Thread Debugging

```python
# List all threads
import threading
print(f"Active threads: {threading.active_count()}")
for thread in threading.enumerate():
    print(f"  {thread.name}: {thread.is_alive()}")

# Get stack trace
import traceback
import sys
for thread_id, frame in sys._current_frames().items():
    print(f"Thread {thread_id}:")
    traceback.print_stack(frame)
```

---

## Getting Help

### Collect Diagnostic Information

Before requesting help, collect:

1. **Environment**:
```bash
python --version
uv --version
cat .env | grep -v API_KEY
```

2. **Configuration**:
```bash
printenv | grep -E "(QPS|DURATION|NUM_WORKFLOWS|SCHEDULER|MODEL)"
```

3. **Logs**:
```bash
# Console output
python script.py 2>&1 | tee experiment.log

# Service logs
cat /tmp/scheduler_*.log
```

4. **Metrics**:
```bash
cat output/metrics.json | python -m json.tool
```

5. **System Info**:
```bash
uname -a
free -h
df -h
netstat -tuln | grep -E "(8100|8101)"
```

### Contact Channels

- **Documentation**: See [API.md](API.md), [QUICKSTART.md](QUICKSTART.md), [MIGRATION.md](MIGRATION.md)
- **Issues**: File issues in project repository with diagnostic info
- **Discussions**: Open discussions for questions and feature requests

---

## Quick Reference

### Common Commands

```bash
# Check service status
curl http://localhost:8100/health

# View metrics
cat output/metrics.json | python -m json.tool

# Check logs
tail -f /tmp/scheduler_*.log

# Monitor process
top -p $(pgrep -f test_workflow)

# Debug network
netstat -tuln | grep -E "(8100|8101)"

# Check file descriptors
lsof -p <pid> | wc -l
```

### Emergency Fixes

```bash
# Kill all experiment processes
pkill -f test_workflow

# Clean up services
./stop_all_services.sh

# Reset environment
unset QPS DURATION NUM_WORKFLOWS

# Clean output
rm -rf output/*
mkdir -p output
```

---

**Remember**: When troubleshooting, always:
1. Check logs first
2. Verify configuration
3. Test components in isolation
4. Use debug logging
5. Compare with working baseline
