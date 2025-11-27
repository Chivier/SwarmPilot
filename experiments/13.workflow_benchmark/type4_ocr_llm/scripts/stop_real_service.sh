#!/bin/bash

# ============================================
# Stop Distributed OCR+LLM Services (Type4)
# ============================================
# Stops all OCR and LLM instance processes
# launched by start_real_service.sh
#
# Usage:
#   bash stop_real_service.sh

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stopping OCR+LLM Services${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# -----------------------------
# 1. Stop OCR Instance Processes
# -----------------------------
echo -e "${YELLOW}Stopping OCR instance processes...${NC}"

# Kill OCR instance processes (instance/src/cli with ocr in name)
if pgrep -f "INSTANCE_ID=ocr_model" >/dev/null 2>&1; then
    echo "Found OCR instance processes, killing..."
    pkill -f "INSTANCE_ID=ocr_model" 2>/dev/null
    echo -e "${GREEN}OCR instance processes killed${NC}"
else
    echo "No OCR instance processes found"
fi

# -----------------------------
# 2. Stop LLM Instance Processes
# -----------------------------
echo ""
echo -e "${YELLOW}Stopping LLM instance processes...${NC}"

# Kill LLM instance processes
if pgrep -f "INSTANCE_ID=llm_model" >/dev/null 2>&1; then
    echo "Found LLM instance processes, killing..."
    pkill -f "INSTANCE_ID=llm_model" 2>/dev/null
    echo -e "${GREEN}LLM instance processes killed${NC}"
else
    echo "No LLM instance processes found"
fi

# Kill sglang launch server processes
if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "Found sglang processes, killing..."
    pkill -f "sglang.launch_server" 2>/dev/null
    echo -e "${GREEN}sglang processes killed${NC}"
fi

# Kill all instance processes (broader match)
if pgrep -f "instance.*src.cli" >/dev/null 2>&1; then
    echo "Found instance processes, killing..."
    pkill -f "instance.*src.cli" 2>/dev/null
    echo -e "${GREEN}Instance processes killed${NC}"
fi

# -----------------------------
# 3. Stop Core Services
# -----------------------------
echo ""
echo -e "${YELLOW}Stopping core services (scheduler, predictor, planner)...${NC}"

# Stop Scheduler processes
if pgrep -f "scheduler.*src.cli" >/dev/null 2>&1; then
    echo "Found scheduler processes, killing..."
    pkill -f "scheduler.*src.cli" 2>/dev/null
    echo -e "${GREEN}Scheduler processes killed${NC}"
fi

# Stop Predictor processes
if pgrep -f "predictor.*src.cli" >/dev/null 2>&1; then
    echo "Found predictor processes, killing..."
    pkill -f "predictor.*src.cli" 2>/dev/null
    echo -e "${GREEN}Predictor processes killed${NC}"
fi

# Stop Planner processes
if pgrep -f "planner.*src.cli" >/dev/null 2>&1; then
    echo "Found planner processes, killing..."
    pkill -f "planner.*src.cli" 2>/dev/null
    echo -e "${GREEN}Planner processes killed${NC}"
fi

# -----------------------------
# 4. Verify Cleanup
# -----------------------------
echo ""
echo -e "${BLUE}Verifying cleanup...${NC}"

sleep 2

# Check for remaining OCR processes
REMAINING_OCR=$(pgrep -f "INSTANCE_ID=ocr_model" 2>/dev/null || true)
if [[ -n "$REMAINING_OCR" ]]; then
    echo -e "${RED}Warning: Some OCR processes still running${NC}"
    ps aux | grep "INSTANCE_ID=ocr_model" | grep -v grep
else
    echo -e "${GREEN}All OCR processes stopped${NC}"
fi

# Check for remaining LLM processes
REMAINING_LLM=$(pgrep -f "INSTANCE_ID=llm_model" 2>/dev/null || true)
if [[ -n "$REMAINING_LLM" ]]; then
    echo -e "${RED}Warning: Some LLM processes still running${NC}"
    ps aux | grep "INSTANCE_ID=llm_model" | grep -v grep
else
    echo -e "${GREEN}All LLM processes stopped${NC}"
fi

# Check for remaining sglang processes
REMAINING_SGLANG=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)
if [[ -n "$REMAINING_SGLANG" ]]; then
    echo -e "${RED}Warning: Some sglang processes still running${NC}"
    ps aux | grep "sglang.launch_server" | grep -v grep
else
    echo -e "${GREEN}All sglang processes stopped${NC}"
fi

# Check for remaining instance processes
REMAINING_INSTANCE=$(pgrep -f "instance.*src.cli" 2>/dev/null || true)
if [[ -n "$REMAINING_INSTANCE" ]]; then
    echo -e "${RED}Warning: Some instance processes still running${NC}"
    ps aux | grep "instance.*src.cli" | grep -v grep
else
    echo -e "${GREEN}All instance processes stopped${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Service Stop Sequence Completed${NC}"
echo -e "${GREEN}========================================${NC}"
