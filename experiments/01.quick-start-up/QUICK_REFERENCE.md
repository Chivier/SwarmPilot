# Quick Reference Guide

## Service Management

### Start All Services
```bash
./start_all_services.sh
```

### Stop All Services
```bash
./stop_all_services.sh
```

### Check Service Status
```bash
# Quick check
curl http://localhost:8101/health  # Predictor
curl http://localhost:8100/health  # Scheduler
curl http://localhost:8200/health  # Instance-000

# Detailed check
./test_pid_capture.sh
```

### View Logs
```bash
# All logs
ls -lh logs/

# Specific service
tail -f logs/predictor.log
tail -f logs/scheduler.log
tail -f logs/instance-000.log
```

## Testing

### Original Sequential Mode
```bash
python test_scheduling.py
```

### Poisson Process Mode (Recommended)
```bash
# Quick test (3 tasks, 1 QPS, single strategy)
uv run python test_scheduling_poisson.py --qps 1 --num-tasks 3 --strategies round_robin

# Default (100 tasks, 10 QPS, all strategies)
uv run python test_scheduling_poisson.py

# High throughput (200 tasks, 20 QPS)
uv run python test_scheduling_poisson.py --qps 20 --num-tasks 200

# Compare strategies
uv run python test_scheduling_poisson.py --strategies min_time probabilistic
```

## Troubleshooting

### Services won't start
```bash
# Check if ports are in use
netstat -tulpn | grep -E '8100|8101|8200'

# Kill stuck processes
pkill -f spredictor
pkill -f sscheduler
pkill -f sinstance

# Clean up and retry
rm -f logs/*.pid
./start_all_services.sh
```

### PIDs are incorrect
```bash
# Test PID capture
./test_pid_capture.sh

# View actual PIDs
pgrep -f "spredictor start" -a
pgrep -f "sscheduler start" -a
pgrep -f "sinstance start" -a
```

### Can't stop services
```bash
# Force cleanup
./stop_all_services.sh

# If that fails, manual cleanup
pkill -9 -f spredictor
pkill -9 -f sscheduler
pkill -9 -f sinstance
docker stop $(docker ps -q --filter "ancestor=sleep_model")
rm -f logs/*.pid
```

### WebSocket connection failed
```bash
# Check scheduler is running
curl http://localhost:8100/health

# Test WebSocket manually
wscat -c ws://localhost:8100/task/get_result

# Check logs
tail -f logs/scheduler.log
```

## Port Reference

| Service | Port | Health Check |
|---------|------|--------------|
| Predictor | 8101 | http://localhost:8101/health |
| Scheduler | 8100 | http://localhost:8100/health |
| Instance-000 | 8200 | http://localhost:8200/health |
| Instance-001 | 8201 | http://localhost:8201/health |
| ... | ... | ... |
| Instance-015 | 8215 | http://localhost:8215/health |

## File Locations

| File/Directory | Purpose |
|----------------|---------|
| `logs/` | Service logs and PID files |
| `results/` | Test results (JSON) |
| `test_scheduling.py` | Sequential test mode |
| `test_scheduling_poisson.py` | Poisson process test mode |
| `start_all_services.sh` | Start all services |
| `stop_all_services.sh` | Stop all services |
| `test_pid_capture.sh` | Verify PID capture |
| `requirements.txt` | Python dependencies |

## Common Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# List running instances
pgrep -f "sinstance start" -a

# Count running instances
pgrep -f "sinstance start" | wc -l

# Check scheduler strategy
curl http://localhost:8100/strategy/get

# Clear all tasks
curl -X POST http://localhost:8100/task/clear

# List all instances
curl http://localhost:8100/instance/list

# View task statistics
curl http://localhost:8100/task/list?status=completed
```

## Test Results

Results are saved in JSON format:

```bash
# Sequential mode
ls -lh results/results_*.json

# Poisson mode
ls -lh results/results_poisson_*.json

# View latest result
cat results/results_poisson_*.json | jq '.'
```

## Environment Variables

Key environment variables (set in start script):

```bash
PREDICTOR_PORT=8101
SCHEDULER_PORT=8100
INSTANCE_START_PORT=8200
NUM_INSTANCES=16
MODEL_ID="sleep_model"
```

## Docker Commands

```bash
# List sleep_model containers
docker ps --filter "ancestor=sleep_model"

# Stop all containers
docker stop $(docker ps -q --filter "ancestor=sleep_model")

# Remove stopped containers
docker rm $(docker ps -aq --filter "ancestor=sleep_model")

# Check image
docker images | grep sleep_model
```
