#!/bin/bash

# ============================================================================
# Experiment 13: Workflow Benchmark Runner
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="simulation"
TYPE=""
QPS=1.0
DURATION=60
NUM_WORKFLOWS=10
FANOUT_COUNT=3
MAX_B_LOOPS=2

# Function to display usage
usage() {
    echo -e "${GREEN}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE          Workflow type: type1 or type2 (required)"
    echo "  -m, --mode MODE          Mode: simulation or real (default: simulation)"
    echo "  -q, --qps QPS            Queries per second (default: 1.0)"
    echo "  -d, --duration DURATION  Duration in seconds (default: 60)"
    echo "  -n, --num NUM            Number of workflows (default: 10)"
    echo "  -f, --fanout FANOUT      Fanout count for Type2 (default: 3)"
    echo "  -l, --loops LOOPS        Max B loops for Type1 (default: 2)"
    echo "  -v, --verify             Run verification tests only"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run Type1 simulation"
    echo "  $0 -t type1 -m simulation"
    echo ""
    echo "  # Run Type2 real mode with custom parameters"
    echo "  $0 -t type2 -m real -q 2.0 -d 300 -n 100"
    echo ""
    echo "  # Verify Type1 alignment"
    echo "  $0 -t type1 -v"
    exit 0
}

# Function to verify alignment
verify_alignment() {
    local workflow_type=$1

    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Verifying ${workflow_type} Alignment${NC}"
    echo -e "${BLUE}============================================================${NC}"

    if [ "$workflow_type" = "type1" ]; then
        cd type1_text2video
        echo -e "${YELLOW}Running Type1 Text2Video verification...${NC}"
        uv run python3 simple_payload_test.py
    elif [ "$workflow_type" = "type2" ]; then
        cd type2_deep_research
        echo -e "${YELLOW}Running Type2 Deep Research verification...${NC}"
        if command -v uv &> /dev/null; then
            uv run python test_alignment_verification.py
        else
            python3 test_alignment_verification.py
        fi
    fi

    echo -e "${GREEN}✅ Verification complete!${NC}"
}

# Function to run experiment
run_experiment() {
    local workflow_type=$1
    local mode=$2

    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Running ${workflow_type} - ${mode} mode${NC}"
    echo -e "${BLUE}============================================================${NC}"

    # Set environment variables
    export MODE=$mode
    export QPS=$QPS
    export DURATION=$DURATION
    export NUM_WORKFLOWS=$NUM_WORKFLOWS

    if [ "$workflow_type" = "type1" ]; then
        export MAX_B_LOOPS=$MAX_B_LOOPS
        cd type1_text2video

        echo -e "${YELLOW}Configuration:${NC}"
        echo "  Mode: $MODE"
        echo "  QPS: $QPS"
        echo "  Duration: ${DURATION}s"
        echo "  Workflows: $NUM_WORKFLOWS"
        echo "  Max B Loops: $MAX_B_LOOPS"
        echo ""

        if [ "$mode" = "simulation" ]; then
            echo -e "${YELLOW}Starting Type1 Text2Video simulation...${NC}"
            uv run python3 -m simulation.test_workflow_sim
        else
            echo -e "${YELLOW}Starting Type1 Text2Video real mode...${NC}"
            uv run python3 -m real.test_workflow_real
        fi

    elif [ "$workflow_type" = "type2" ]; then
        export FANOUT_COUNT=$FANOUT_COUNT
        cd type2_deep_research

        echo -e "${YELLOW}Configuration:${NC}"
        echo "  Mode: $MODE"
        echo "  QPS: $QPS"
        echo "  Duration: ${DURATION}s"
        echo "  Workflows: $NUM_WORKFLOWS"
        echo "  Fanout Count: $FANOUT_COUNT"
        echo ""

        if [ "$mode" = "simulation" ]; then
            echo -e "${YELLOW}Starting Type2 Deep Research simulation...${NC}"
            uv run python3 -m simulation.test_workflow_sim
        else
            echo -e "${YELLOW}Starting Type2 Deep Research real mode...${NC}"
            uv run python3 -m real.test_workflow_real
        fi
    fi

    echo -e "${GREEN}✅ Experiment complete!${NC}"
}

# Parse command line arguments
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -q|--qps)
            QPS="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -n|--num)
            NUM_WORKFLOWS="$2"
            shift 2
            ;;
        -f|--fanout)
            FANOUT_COUNT="$2"
            shift 2
            ;;
        -l|--loops)
            MAX_B_LOOPS="$2"
            shift 2
            ;;
        -v|--verify)
            VERIFY_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$TYPE" ]; then
    echo -e "${RED}Error: Workflow type is required (-t type1 or -t type2)${NC}"
    usage
fi

if [ "$TYPE" != "type1" ] && [ "$TYPE" != "type2" ]; then
    echo -e "${RED}Error: Invalid workflow type. Must be 'type1' or 'type2'${NC}"
    usage
fi

if [ "$MODE" != "simulation" ] && [ "$MODE" != "real" ]; then
    echo -e "${RED}Error: Invalid mode. Must be 'simulation' or 'real'${NC}"
    usage
fi

# Main execution
if [ "$VERIFY_ONLY" = true ]; then
    verify_alignment $TYPE
else
    run_experiment $TYPE $MODE
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}All operations completed successfully!${NC}"
echo -e "${BLUE}============================================================${NC}"