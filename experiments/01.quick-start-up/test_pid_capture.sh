#!/bin/bash

# Test script to verify PID capture works correctly

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "Testing PID capture mechanism"
echo "=============================="
echo ""

# Test function
test_pid_capture() {
    local service_name=$1
    local pattern=$2

    echo "Testing: $service_name"
    echo "  Pattern: $pattern"

    # Find PIDs matching the pattern
    local pids=$(pgrep -f "$pattern")

    if [ -z "$pids" ]; then
        echo "  Status: ❌ No processes found"
        return 1
    else
        echo "  Status: ✅ Found process(es)"
        echo "  PIDs: $pids"

        # Check if PID file exists and matches
        local pid_file="$LOG_DIR/${service_name}.pid"
        if [ -f "$pid_file" ]; then
            local saved_pid=$(cat "$pid_file")
            echo "  Saved PID: $saved_pid"

            # Check if saved PID is in the list of found PIDs
            if echo "$pids" | grep -q "$saved_pid"; then
                echo "  Match: ✅ Saved PID matches running process"

                # Verify process is actually running
                if ps -p "$saved_pid" > /dev/null 2>&1; then
                    echo "  Verify: ✅ Process is running"

                    # Show process details
                    echo "  Details:"
                    ps -p "$saved_pid" -o pid,ppid,cmd | sed 's/^/    /'
                    return 0
                else
                    echo "  Verify: ❌ Process not running"
                    return 1
                fi
            else
                echo "  Match: ❌ Saved PID doesn't match running process"
                echo "  Running PIDs: $pids"
                echo "  Saved PID: $saved_pid"
                return 1
            fi
        else
            echo "  PID file: ⚠️  Not found ($pid_file)"
            return 1
        fi
    fi
}

echo "Checking running services:"
echo ""

# Test predictor
test_pid_capture "predictor" "spredictor start"
echo ""

# Test scheduler
test_pid_capture "scheduler" "sscheduler start"
echo ""

# Test first instance
test_pid_capture "instance-000" "sinstance start.*instance-000"
echo ""

# Summary
echo "=============================="
echo "Summary:"
echo ""

total=0
success=0

for service in predictor scheduler instance-000 instance-001 instance-002; do
    total=$((total + 1))
    pid_file="$LOG_DIR/${service}.pid"

    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ $service (PID: $pid) - Running"
            success=$((success + 1))
        else
            echo "❌ $service (PID: $pid) - Not running"
        fi
    else
        echo "⚠️  $service - No PID file"
    fi
done

echo ""
echo "Success rate: $success/$total"

if [ $success -eq $total ]; then
    echo "✅ All PIDs captured correctly!"
    exit 0
else
    echo "❌ Some PIDs are incorrect"
    exit 1
fi
