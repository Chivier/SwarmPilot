#!/bin/bash
# Multi-strategy test script for Type1 Text2Video workflow
# Tests probabilistic and min_time strategies

echo "=========================================="
echo "Multi-Strategy Test: probabilistic, min_time"
echo "=========================================="

# Set test parameters (small values for quick testing)
export MODE="simulation"
export STRATEGIES="probabilistic,min_time"
export QPS=2.0
export DURATION=60  # 1 minute test
export NUM_WORKFLOWS=20  # Small number for quick test
export MAX_B_LOOPS=2  # Reduce B loops
export SLEEP_TIME_MIN=1.0
export SLEEP_TIME_MAX=3.0

# Output directory
export OUTPUT_DIR="output/multi_strategy_test"

echo ""
echo "Test Configuration:"
echo "  Strategies: $STRATEGIES"
echo "  QPS: $QPS"
echo "  Duration: $DURATION seconds"
echo "  Workflows: $NUM_WORKFLOWS"
echo "  Max B loops: $MAX_B_LOOPS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if schedulers are running
echo "Checking scheduler availability..."
SCHEDULER_A_URL="${SCHEDULER_A_URL:-http://127.0.0.1:8100}"
SCHEDULER_B_URL="${SCHEDULER_B_URL:-http://127.0.0.1:8101}"

# Test scheduler A
if curl -s -f -m 2 "$SCHEDULER_A_URL/health" > /dev/null 2>&1; then
    echo "  ✓ Scheduler A is available at $SCHEDULER_A_URL"
else
    echo "  ✗ Scheduler A is NOT available at $SCHEDULER_A_URL"
    echo "  Please start the scheduler first"
    exit 1
fi

# Test scheduler B
if curl -s -f -m 2 "$SCHEDULER_B_URL/health" > /dev/null 2>&1; then
    echo "  ✓ Scheduler B is available at $SCHEDULER_B_URL"
else
    echo "  ✗ Scheduler B is NOT available at $SCHEDULER_B_URL"
    echo "  Please start the scheduler first"
    exit 1
fi

echo ""

# Run the test
echo "Starting test..."
echo ""

cd /chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/13.workflow_benchmark

uv run python type1_text2video/simulation/test_workflow_sim.py

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Test completed successfully!"
    echo ""
    echo "Results:"
    echo "  Check logs for strategy execution details"
    echo "  Output directory: $OUTPUT_DIR"

    # Check if output files exist
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
    fi
else
    echo "Test failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
