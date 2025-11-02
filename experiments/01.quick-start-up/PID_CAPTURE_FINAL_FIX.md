# PID Capture - Final Fix Summary

## Problem Root Causes

### 1. Shell PID vs Python PID
```bash
# Original code captured shell PID
eval "$command" & pid=$!  # ❌ Wrong PID
```

### 2. Entry Point Names Don't Appear in Process
```bash
# Attempted fix V1 - Failed
pgrep -f "spredictor start"  # ❌ Not in command line
```

### 3. Environment Variables Not Passed to uv run
```bash
# Instance startup issue
INSTANCE_PORT=8200 uv run python -m src.cli start  # ❌ Env var lost
```

## Final Solution

### Use `--port` Parameter for All Services

All services now explicitly use `--port` parameter, making PID lookup consistent and reliable.

#### Predictor
```bash
start_service "predictor" \
    "cd $PROJECT_ROOT/predictor && \
     PREDICTOR_PORT=$PREDICTOR_PORT \
     PREDICTOR_LOG_DIR=$SCRIPT_DIR/logs/predictor \
     uv run python -m src.cli start --port $PREDICTOR_PORT" \
    "$PREDICTOR_PORT"
```

**Process:** `python3 -m src.cli start --port 8101`
**Pattern:** `python3.*src\.cli start.*--port 8101`

#### Scheduler
```bash
start_service "scheduler" \
    "cd $PROJECT_ROOT/scheduler && \
     PREDICTOR_URL=http://localhost:$PREDICTOR_PORT \
     SCHEDULER_PORT=$SCHEDULER_PORT \
     SCHEDULER_LOG_DIR=$SCRIPT_DIR/logs/scheduler \
     uv run python -m src.cli start --port $SCHEDULER_PORT" \
    "$SCHEDULER_PORT"
```

**Process:** `python3 -m src.cli start --port 8100`
**Pattern:** `python3.*src\.cli start.*--port 8100`

#### Instance
```bash
start_service "instance-000" \
    "cd $PROJECT_ROOT/instance && \
     SCHEDULER_URL=http://localhost:$SCHEDULER_PORT \
     INSTANCE_ID=$instance_id \
     INSTANCE_PORT=$instance_port \
     INSTANCE_LOG_DIR=$SCRIPT_DIR/logs/instance_$instance_port \
     uv run python -m src.cli start --port $instance_port" \
    "$instance_port"
```

**Process:** `python3 -m src.cli start --port 8200`
**Pattern:** `python3.*src\.cli start.*--port 8200`

**Note:** INSTANCE_PORT environment variable is still set because Config reads it on startup for model_port calculation.

## Updated start_service() Function

```bash
start_service() {
    local name=$1
    local command=$2
    local port=$3  # Port number for PID lookup

    # Start service in background
    nohup bash -c "$command" > "$log_file" 2>&1 &

    # Wait for startup
    sleep 2

    # Find Python process by --port parameter
    local actual_pid=$(pgrep -f "python3.*src\.cli start.*--port $port" | head -1)

    # Retry up to 5 times
    local retry=0
    while [ -z "$actual_pid" ] && [ $retry -lt 5 ]; do
        sleep 1
        actual_pid=$(pgrep -f "python3.*src\.cli start.*--port $port" | head -1)
        retry=$((retry + 1))
    done

    # Save PID
    if [ -n "$actual_pid" ]; then
        echo $actual_pid > "$pid_file"
        echo "Started $name (PID: $actual_pid, Port: $port)"
    else
        echo "Failed to find PID for $name (port: $port)"
        return 1
    fi
}
```

## Key Features

### 1. Unified Pattern
All services use the same pattern:
```bash
python3.*src\.cli start.*--port <PORT>
```

### 2. Port-Based Identification
Each service has a unique port:
- Predictor: 8101
- Scheduler: 8100
- Instance-000: 8200
- Instance-001: 8201
- ...
- Instance-015: 8215

### 3. Reliable PID Capture
- **Matches actual Python process** (not shell or uv wrapper)
- **Unique per service** (port numbers don't overlap)
- **Retry mechanism** (up to 5 attempts with 1s delay)

### 4. Graceful Shutdown
stop_all_services.sh updated to:
1. Try SIGTERM first (graceful)
2. Wait up to 10 seconds
3. Use SIGKILL only if needed

## Testing

### Test PID Capture
```bash
./test_pattern_matching.sh
```

### Expected Output
```
Testing Pattern Matching for PID Capture
==========================================

Test 1: Predictor (Port 8101)
  Pattern: python3.*src\.cli start.*--port 8101
  ✅ Found PID: 25561
    25561 python3 -m src.cli start --port 8101

Test 2: Scheduler (Port 8100)
  Pattern: python3.*src\.cli start.*--port 8100
  ✅ Found PID: 12347
    12347 python3 -m src.cli start --port 8100

Test 3: Instance-000 (Port 8200)
  Pattern: python3.*src\.cli start.*--port 8200
  ✅ Found PID: 12348
    12348 python3 -m src.cli start --port 8200

==========================================
Summary
==========================================
✅ Predictor (Port 8101, PID 25561)
✅ Scheduler (Port 8100, PID 12347)
✅ Instance-000 (Port 8200, PID 12348)

Success: 3/3
✅ All patterns working correctly!
```

## Manual Verification

```bash
# Start all services
./start_all_services.sh

# Check PIDs are correct
for service_pid in logs/*.pid; do
    service=$(basename "$service_pid" .pid)
    pid=$(cat "$service_pid")
    if ps -p "$pid" > /dev/null 2>&1; then
        port=$(ps -p "$pid" -o args= | grep -oP -- '--port \K\d+')
        echo "✅ $service (PID: $pid, Port: $port)"
    else
        echo "❌ $service (PID: $pid) - Not running"
    fi
done

# Stop all services
./stop_all_services.sh

# Verify all stopped
pgrep -f "python3.*src.cli start" || echo "✅ All services stopped"
```

## Problem Resolution Summary

| Issue | Cause | Solution |
|-------|-------|----------|
| **Wrong PID captured** | `$!` returns shell PID | Use `pgrep` with pattern matching |
| **Pattern not matching** | Entry point names not in command line | Use `--port` parameter in pattern |
| **Instance startup fails** | Environment variables lost with `uv run` | Add explicit `--port` parameter |
| **Multiple PIDs match** | Pattern too generic | Filter by `python3` + specific port |
| **Services can't stop** | Wrong PID in .pid file | Correct PID capture fixes this |

## Benefits

✅ **Reliable**: PIDs always correctly captured
✅ **Consistent**: All services use same pattern
✅ **Unique**: Port-based identification prevents conflicts
✅ **Testable**: Easy to verify with test scripts
✅ **Maintainable**: Clear and understandable logic
✅ **Graceful**: Proper shutdown with SIGTERM → SIGKILL

## Files Modified

1. **start_all_services.sh**
   - Updated `start_service()` function
   - All service commands now use `--port` parameter
   - Simplified PID lookup (removed environment variable fallback)

2. **stop_all_services.sh**
   - Added graceful shutdown (SIGTERM before SIGKILL)
   - Better error handling and messaging

3. **Test Scripts**
   - `test_pattern_matching.sh`: Verify patterns work
   - `test_pid_capture.sh`: Verify saved PIDs are correct

4. **Documentation**
   - `PID_CAPTURE_FIX.md`: V1 solution attempt
   - `PID_CAPTURE_FIX_V2.md`: Port-based solution
   - `PID_CAPTURE_FINAL_FIX.md`: This document
   - `QUICK_REFERENCE.md`: Quick reference guide

All services now start and stop reliably with correct PID capture!
