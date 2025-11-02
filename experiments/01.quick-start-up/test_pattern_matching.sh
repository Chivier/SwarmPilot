#!/bin/bash

# Test script to verify pattern matching works correctly

echo "Testing Pattern Matching for PID Capture"
echo "=========================================="
echo ""

# Test predictor pattern
echo "Test 1: Predictor (Port 8101)"
echo "  Pattern: python3.*src\.cli start.*--port 8101"
predictor_pid=$(pgrep -f "python3.*src\.cli start.*--port 8101" | head -1)
if [ -n "$predictor_pid" ]; then
    echo "  ✅ Found PID: $predictor_pid"
    ps -p $predictor_pid -o pid,args | tail -1 | sed 's/^/  /'
else
    echo "  ❌ No process found"
fi
echo ""

# Test scheduler pattern
echo "Test 2: Scheduler (Port 8100)"
echo "  Pattern: python3.*src\.cli start.*--port 8100"
scheduler_pid=$(pgrep -f "python3.*src\.cli start.*--port 8100" | head -1)
if [ -n "$scheduler_pid" ]; then
    echo "  ✅ Found PID: $scheduler_pid"
    ps -p $scheduler_pid -o pid,args | tail -1 | sed 's/^/  /'
else
    echo "  ❌ No process found"
fi
echo ""

# Test instance pattern (port 8200)
echo "Test 3: Instance-000 (Port 8200)"
echo "  Pattern: python3.*src\.cli start.*INSTANCE_PORT=8200"
instance_pid=$(pgrep -f "python3.*src\.cli start" | while read pid; do
    if ps -p $pid -o args= | grep -q "INSTANCE_PORT=8200"; then
        echo $pid
        break
    fi
done | head -1)

# Alternative: check if instance uses --port
if [ -z "$instance_pid" ]; then
    instance_pid=$(pgrep -f "python3.*src\.cli start.*--port 8200" | head -1)
fi

if [ -n "$instance_pid" ]; then
    echo "  ✅ Found PID: $instance_pid"
    ps -p $instance_pid -o pid,args | tail -1 | sed 's/^/  /'
else
    echo "  ❌ No process found"
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="

services=(
    "Predictor:8101"
    "Scheduler:8100"
    "Instance-000:8200"
)

success=0
total=0

for service_info in "${services[@]}"; do
    IFS=':' read -r service port <<< "$service_info"
    total=$((total + 1))

    pid=$(pgrep -f "python3.*src\.cli start.*--port $port" | head -1)

    # For instances without --port, try environment variable
    if [ -z "$pid" ] && [[ "$service" == Instance* ]]; then
        pid=$(pgrep -f "python3.*src\.cli start" | while read p; do
            if ps -p $p -o args= | grep -q "INSTANCE_PORT=$port"; then
                echo $p
                break
            fi
        done | head -1)
    fi

    if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
        echo "✅ $service (Port $port, PID $pid)"
        success=$((success + 1))
    else
        echo "❌ $service (Port $port) - Not found"
    fi
done

echo ""
echo "Success: $success/$total"

if [ $success -eq $total ]; then
    echo "✅ All patterns working correctly!"
    exit 0
else
    echo "⚠️  Some patterns failed"
    exit 1
fi
