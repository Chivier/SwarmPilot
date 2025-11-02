# Quick Reference Guide - Experiment 02

## Quick Start

### 1. Start Services (Default: 10+6 instances)
```bash
./start_all_services.sh
```

### 2. Run Experiment
```bash
uv run python test_dual_scheduler.py
```

### 3. Stop Services
```bash
./stop_all_services.sh
```

## Common Usage Patterns

### Small Test Run (Fast)
```bash
# Start with fewer instances
./start_all_services.sh 4 2

# Run with fewer tasks
uv run python test_dual_scheduler.py --n1 4 --n2 2 --num-tasks 50

# Cleanup
./stop_all_services.sh
```

### Balanced Configuration
```bash
# Equal instance groups
./start_all_services.sh 8 8

# Equal QPS
uv run python test_dual_scheduler.py --n1 8 --n2 8 --qps1 10.0 --qps2 10.0

./stop_all_services.sh
```

### High Load Test
```bash
# More instances
./start_all_services.sh 12 8

# Higher QPS and more tasks
uv run python test_dual_scheduler.py \
    --n1 12 --n2 8 \
    --qps1 15.0 --qps2 12.0 \
    --num-tasks 200

./stop_all_services.sh
```

### Test Single Strategy
```bash
# Only test probabilistic strategy
uv run python test_dual_scheduler.py --strategies probabilistic

# Test two strategies
uv run python test_dual_scheduler.py --strategies round_robin min_time
```

## Command Cheat Sheet

### Service Management
```bash
# Start with custom config
./start_all_services.sh [N1] [N2]

# Check service health
curl http://localhost:8100/health  # Scheduler A
curl http://localhost:8200/health  # Scheduler B
curl http://localhost:8101/health  # Predictor

# View logs
tail -f logs/scheduler-a.log
tail -f logs/scheduler-b.log
tail -f logs/predictor.log

# Stop everything
./stop_all_services.sh
```

### Testing
```bash
# Full test (all strategies, default config)
uv run python test_dual_scheduler.py

# Custom instance counts
uv run python test_dual_scheduler.py --n1 10 --n2 6

# Custom QPS
uv run python test_dual_scheduler.py --qps1 8.0 --qps2 5.0

# More/fewer tasks
uv run python test_dual_scheduler.py --num-tasks 150

# Specific strategies
uv run python test_dual_scheduler.py --strategies round_robin

# Combined options
uv run python test_dual_scheduler.py \
    --n1 8 --n2 8 \
    --qps1 10.0 --qps2 10.0 \
    --num-tasks 100 \
    --strategies round_robin probabilistic
```

### Workload Generator Testing
```bash
# Test workload distributions
uv run python workload_generator.py --num-tasks 100

# Different random seed
uv run python workload_generator.py --num-tasks 100 --seed 12345
```

## Port Allocation

| Service | Port(s) | Description |
|---------|---------|-------------|
| Predictor | 8101 | Shared prediction service |
| Scheduler A | 8100 | Group A scheduler (bimodal workload) |
| Scheduler B | 8200 | Group B scheduler (Pareto workload) |
| Group A Instances | 8210-82xx | N1 instances for Scheduler A |
| Group B Instances | 8300-83xx | N2 instances for Scheduler B |

## Default Configuration

```
N1 = 10              # Group A instances
N2 = 6               # Group B instances
QPS1 = 8.0           # Scheduler A target QPS
QPS2 = 5.0           # Scheduler B target QPS
NUM_TASKS = 100      # Tasks per scheduler per strategy
```

## Results Location

```
results/results_dual_YYYYMMDD_HHMMSS.json
```

## Quick Troubleshooting

### Port Already in Use
```bash
./stop_all_services.sh
# Wait a few seconds
./start_all_services.sh
```

### Docker Issues
```bash
# Check Docker
docker ps

# Clean containers
docker ps -a | grep sleep_model | awk '{print $1}' | xargs docker rm -f

# Rebuild image
cd ../../instance && ./build_sleep_model.sh
```

### Check Running Processes
```bash
# Find Python services
pgrep -f "python3.*src.cli start" -a

# Check ports
netstat -tulpn | grep -E "810|820|830"
```

### View Errors
```bash
# Search all logs for errors
grep -i error logs/*.log

# Recent errors only
grep -i error logs/*.log | tail -20
```

## Performance Expectations

### Small Config (N1=4, N2=2, 50 tasks each)
- Startup: ~30 seconds
- Test run: ~2-3 minutes per strategy
- Total: ~10 minutes for all strategies

### Default Config (N1=10, N2=6, 100 tasks each)
- Startup: ~60 seconds
- Test run: ~3-5 minutes per strategy
- Total: ~15-20 minutes for all strategies

### Large Config (N1=12, N2=8, 200 tasks each)
- Startup: ~90 seconds
- Test run: ~6-8 minutes per strategy
- Total: ~25-30 minutes for all strategies

## Experiment Workflow

```
1. Start services      → ./start_all_services.sh [N1] [N2]
2. Wait for readiness  → Check logs/health endpoints
3. Run experiment      → uv run python test_dual_scheduler.py [options]
4. Analyze results     → results/results_dual_*.json
5. Stop services       → ./stop_all_services.sh
```

## File Structure

```
02.multi_model_no_dep/
├── logs/                      # Service logs (created at runtime)
├── results/                   # Test results (JSON files)
├── requirements.txt           # Python dependencies
├── workload_generator.py      # Workload distribution generator
├── test_dual_scheduler.py     # Main test script
├── start_all_services.sh      # Service startup script
├── stop_all_services.sh       # Service shutdown script
├── README.md                  # Full documentation
└── QUICK_REFERENCE.md         # This file
```

## Key Differences from Experiment 01

| Aspect | Experiment 01 | Experiment 02 |
|--------|--------------|--------------|
| Schedulers | 1 scheduler | 2 independent schedulers |
| Workloads | Bimodal only | Bimodal + Pareto |
| Instances | Fixed 16 | Configurable N1+N2 |
| QPS | Single QPS | Independent QPS per scheduler |
| Ports | 8100, 8200-8215 | 8100/8200, 8210-82xx, 8300-83xx |

## Tips

1. **Start Small**: Test with N1=4, N2=2 first to verify setup
2. **Monitor Logs**: Keep logs open in separate terminals during testing
3. **Check Health**: Always verify health endpoints before running tests
4. **Clean Shutdown**: Always use stop script to avoid orphaned processes
5. **Compare Results**: Compare Scheduler A vs B for same strategy to see workload impact

## Common Issues

### Issue: "Scheduler not responding"
**Solution**: Check if services started successfully, view logs

### Issue: "Port already in use"
**Solution**: Run `./stop_all_services.sh` and wait before restarting

### Issue: "No tasks completed"
**Solution**: Check Docker containers are running, check instance logs

### Issue: "Connection timeout"
**Solution**: Increase wait times in test script or reduce QPS

## Help

For detailed information, see:
- Full documentation: `README.md`
- Experiment 01 reference: `../01.quick-start-up/README.md`
- GitHub issues: Report problems at the project repository
