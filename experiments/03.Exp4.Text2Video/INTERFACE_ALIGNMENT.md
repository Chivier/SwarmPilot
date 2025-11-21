# Interface Alignment with Experiment 07

This document confirms that Experiment 03's service management scripts now match Experiment 07's interface and usage patterns exactly.

## Service Scripts Interface

### 1. start_all_services.sh

**Interface (matching exp07):**
```bash
# Positional arguments
./start_all_services.sh [N1] [N2] [MODEL_ID_A] [MODEL_ID_B]

# Environment variables
N1=<groupA> N2=<groupB> MODEL_ID_A=<idA> MODEL_ID_B=<idB> ./start_all_services.sh

# Help
./start_all_services.sh --help
```

**Examples:**
```bash
# Use defaults (N1=4, N2=2, sleep models)
./start_all_services.sh

# Positional arguments (8 Group A, 4 Group B)
./start_all_services.sh 8 4

# Specify models with positional arguments
./start_all_services.sh 10 6 llm_service_small_model t2vid

# Environment variables (simulation)
N1=4 N2=2 ./start_all_services.sh

# Environment variables (real models)
N1=10 N2=6 MODEL_ID_A=llm_service_small_model MODEL_ID_B=t2vid ./start_all_services.sh

# Mixed (positional + environment)
MODEL_ID_A=llm_service_small_model MODEL_ID_B=t2vid ./start_all_services.sh 8 4
```

**Differences from exp07:**
- Exp07 uses single `MODEL_ID` for both groups (always `sleep_model`)
- Exp03 supports separate `MODEL_ID_A` and `MODEL_ID_B` for two different models
- Exp03 includes planner by default (exp07 doesn't have planner in basic script)

### 2. stop_all_services.sh

**Interface (identical to exp07):**
```bash
./stop_all_services.sh
```

**No arguments needed** - automatically detects and stops all services using PID files.

**Features (matching exp07):**
- Parallel instance shutdown
- Graceful shutdown with SIGTERM (10 second timeout)
- Force kill with SIGKILL if needed
- Cleanup of Docker containers
- Color-coded output

### 3. manual_deploy_planner.sh

**Interface:**
```bash
./manual_deploy_planner.sh [options]

Options:
  --scheduler-a-url URL   Scheduler A URL
  --scheduler-b-url URL   Scheduler B URL
  --planner-url URL       Planner URL
  --model-id-a ID         Model for Group A
  --model-id-b ID         Model for Group B
  --n1 N                  Group A instance count
  --n2 N                  Group B instance count
  --port-a-start P        Group A start port
  --port-b-start P        Group B start port
```

**Called automatically** by `start_all_services.sh` after instance startup.

## Usage Patterns

### Pattern 1: Quick Start (Simulation)
```bash
# Exp07 style
./start_all_services.sh 10 6

# Exp03 equivalent
./start_all_services.sh 10 6
```

### Pattern 2: Environment Variables
```bash
# Exp07 style
N1=10 N2=6 ./start_all_services.sh

# Exp03 equivalent
N1=10 N2=6 ./start_all_services.sh
# Or with specific models:
N1=10 N2=6 MODEL_ID_A=llm_service_small_model MODEL_ID_B=t2vid ./start_all_services.sh
```

### Pattern 3: Stop Services
```bash
# Identical for both
./stop_all_services.sh
```

## Key Alignment Points

### ✅ Command-Line Argument Parsing
- Both support positional arguments: `[N1] [N2]`
- Both support environment variables: `N1=X N2=Y`
- Both support `--help` flag
- Both validate integer inputs

### ✅ Output Formatting
- Identical color scheme (GREEN, RED, YELLOW, BLUE)
- Same health check waiting pattern with dots
- Same "Starting..." and "OK" messages
- Same summary format at the end

### ✅ Service Startup Order
1. Predictor
2. Planner (exp03 only)
3. Scheduler A
4. Scheduler B
5. Group A Instances (parallel)
6. Group B Instances (parallel)
7. Model deployment

### ✅ PID Management
- Same PID file naming: `logs/{service-name}.pid`
- Same PID discovery using `pgrep -f "python.*--port"`
- Same retry logic (5 attempts)

### ✅ Health Checks
- Same pattern: `curl -s -f "$url/health"`
- Same timeout: 30 seconds (30 attempts × 1 second)
- Same output format

### ✅ Logging
- Same log directory: `./logs/`
- Same log file naming: `logs/{service-name}.log`
- Same nohup redirection: `> "$log_file" 2>&1`

### ✅ Shutdown Process
- Same graceful shutdown: SIGTERM with 10 second timeout
- Same force kill: SIGKILL if SIGTERM fails
- Same parallel instance shutdown
- Same PID file cleanup
- Same Docker container cleanup

## Comparison Table

| Feature | Exp07 | Exp03 | Compatible? |
|---------|-------|-------|-------------|
| Positional args `[N1] [N2]` | ✅ | ✅ | ✅ Yes |
| Environment vars | ✅ | ✅ | ✅ Yes |
| `--help` flag | ✅ | ✅ | ✅ Yes |
| Color output | ✅ | ✅ | ✅ Yes |
| Health checks | ✅ | ✅ | ✅ Yes |
| PID management | ✅ | ✅ | ✅ Yes |
| Parallel startup | ✅ | ✅ | ✅ Yes |
| Graceful shutdown | ✅ | ✅ | ✅ Yes |
| Docker cleanup | ✅ | ✅ | ✅ Yes |
| Planner support | ❌ | ✅ | ⚠️ Extra feature |
| Separate models | ❌ | ✅ | ⚠️ Extra feature |

**Note:** ⚠️ = Exp03 has additional features but maintains full backward compatibility

## Migration from Old Interface

**Old Exp03 interface (now deprecated):**
```bash
N1=<groupA> N2=<groupB> MODEL_ID_A=<idA> MODEL_ID_B=<idB> ./start_all_services.sh
```

**New Exp03 interface (exp07-compatible):**
```bash
# All of these work now:
./start_all_services.sh 10 6                              # Positional
N1=10 N2=6 ./start_all_services.sh                        # Environment
./start_all_services.sh 10 6 llm_service t2vid            # Positional with models
N1=10 N2=6 MODEL_ID_A=llm_service MODEL_ID_B=t2vid ./start_all_services.sh  # Environment
```

## Testing the Alignment

To verify the interface works as expected:

```bash
# Test 1: Help message
./start_all_services.sh --help

# Test 2: Positional arguments
./start_all_services.sh 2 2

# Test 3: Environment variables
N1=2 N2=2 ./start_all_services.sh

# Test 4: Stop services
./stop_all_services.sh

# Test 5: With real models
./start_all_services.sh 4 2 llm_service_small_model t2vid
./stop_all_services.sh
```

## Summary

Experiment 03's service management scripts now provide:
- ✅ **100% interface compatibility** with Experiment 07
- ✅ **Same command-line argument patterns**
- ✅ **Same environment variable support**
- ✅ **Same output formatting and colors**
- ✅ **Same PID and health check mechanisms**
- ✅ **Additional features** (planner, separate models) without breaking compatibility

Users familiar with Experiment 07 can use Experiment 03 with identical commands and workflows.
