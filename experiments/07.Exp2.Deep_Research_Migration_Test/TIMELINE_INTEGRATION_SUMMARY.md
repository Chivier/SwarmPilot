# Timeline Collection Integration - Summary

## Overview

Successfully integrated instance deployment timeline collection into Experiment 07. The system now automatically tracks and records how instance allocations change during experiments, including both initial deployments and auto-optimization events.

## Changes Made

### 1. Modified Files

#### `test_dynamic_workflow.py`

**Added PLANNER_URL constant (line 75)**:
```python
PLANNER_URL = "http://localhost:8202"
```

**Added timeline collection functions**:
- `clear_planner_timeline()` (line 2176-2205): Clears timeline at experiment start
- `get_planner_timeline()` (line 2208-2252): Retrieves timeline at experiment end

**Integrated into test workflow**:
- Step 1 (line 2712): Clear timeline before each strategy test
- Step 17 (line 3175-3181): Retrieve timeline after experiment completion
- Return results (line 3195): Include timeline data in results dictionary

### 2. New Files Created

#### `test_timeline_api.py` (142 lines)
Standalone test script to verify timeline API functionality.

**Features**:
- Tests Planner connectivity
- Verifies timeline clear endpoint
- Verifies timeline retrieval endpoint
- Validates data structure
- Provides clear pass/fail feedback

**Usage**:
```bash
python test_timeline_api.py
```

#### `visualize_timeline.py` (322 lines)
Visualization tool for timeline data from experiment results.

**Features**:
- Loads timeline from results JSON
- Prints detailed text summary
- Creates stacked area plot of instance allocations
- Creates line plot showing per-model instance counts
- Marks deployment and auto-optimization events
- Supports saving plots or displaying interactively

**Usage**:
```bash
# Display summary and plot
python visualize_timeline.py results/results_workflow_b1b2_20231219_120000.json

# Specific strategy
python visualize_timeline.py results/results_workflow_b1b2_20231219_120000.json --strategy min_time

# Save to file
python visualize_timeline.py results/results_workflow_b1b2_20231219_120000.json --output timeline.png

# Summary only (no plot)
python visualize_timeline.py results/results_workflow_b1b2_20231219_120000.json --summary-only
```

#### `TIMELINE_COLLECTION_GUIDE.md` (600+ lines)
Comprehensive documentation covering:
- What timeline data is collected
- How the integration works
- Usage instructions
- Analysis examples
- Troubleshooting guide
- API reference

## How It Works

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Experiment Start                                            │
│  - test_strategy_workflow() called                          │
│  - Step 1: clear_planner_timeline()                         │
│    └─> POST /timeline/clear to Planner                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ During Experiment                                           │
│  - Initial deployment (redeply.py)                          │
│    └─> POST /deploy/migration                               │
│       └─> Planner records "deploy_migration" event          │
│                                                              │
│  - Schedulers report queue lengths                          │
│    └─> POST /submit_target (periodic)                       │
│       └─> Triggers auto-optimization when conditions met    │
│          └─> Planner records "auto_optimize" event          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Experiment End                                              │
│  - Step 17: get_planner_timeline()                          │
│    └─> GET /timeline from Planner                           │
│       └─> Returns all recorded events                       │
│  - Timeline data added to results["planner_timeline"]       │
│  - Results saved to JSON file                               │
└─────────────────────────────────────────────────────────────┘
```

### Timeline Entry Structure

Each timeline entry contains:

```json
{
  "timestamp": 1703001234.567,              // Unix timestamp
  "timestamp_iso": "2023-12-19T12:00:34+00:00",  // ISO format
  "event_type": "auto_optimize",            // "deploy_migration" or "auto_optimize"
  "instance_counts": {                      // Current allocation per model
    "sleep_model_a": 6,
    "sleep_model_b": 10
  },
  "total_instances": 16,                    // Total instances across models
  "changes_count": 2,                       // Number of migrations in this event
  "success": true,                          // Whether migration succeeded
  "target_distribution": [8.5, 42.3],       // Target workload distribution (optional)
  "score": 0.0234                           // Optimization score (optional)
}
```

## Usage Guide

### Quick Start

1. **Start services** (if not already running):
   ```bash
   ./start_all_services.sh
   ```

2. **Test timeline collection**:
   ```bash
   python test_timeline_api.py
   ```
   Expected output: "✓ All timeline API tests passed!"

3. **Run experiment** (timeline collection is automatic):
   ```bash
   python test_dynamic_workflow.py --num-workflows 50
   ```

4. **Visualize results**:
   ```bash
   python visualize_timeline.py results/results_workflow_b1b2_*.json
   ```

### Accessing Timeline Data Programmatically

```python
import json

# Load results
with open("results/results_workflow_b1b2_<timestamp>.json") as f:
    data = json.load(f)

# Access timeline for a specific strategy
for result in data["results"]:
    if result["strategy"] == "min_time":
        timeline = result["planner_timeline"]

        if timeline and timeline.get("success"):
            entries = timeline["entries"]

            # Analyze entries
            for entry in entries:
                print(f"{entry['event_type']}: {entry['instance_counts']}")
```

## Benefits

1. **Automatic Collection**: No manual intervention needed
2. **Per-Strategy Isolation**: Each strategy test has separate timeline data
3. **Rich Metadata**: Includes timestamps, event types, scores, and success status
4. **Easy Analysis**: JSON format, ready for visualization and analysis
5. **Debugging**: Timeline helps identify deployment issues and optimization behavior
6. **Research Value**: Enables studying dynamic resource allocation patterns

## Example Use Cases

### 1. Compare Redeployment Patterns Across Strategies

```python
for result in results:
    timeline = result["planner_timeline"]["entries"]
    auto_opts = [e for e in timeline if e["event_type"] == "auto_optimize"]
    print(f"{result['strategy']}: {len(auto_opts)} auto-optimizations")
```

### 2. Analyze Optimization Convergence

```python
scores = [e["score"] for e in timeline if e.get("score") is not None]
plt.plot(scores)
plt.xlabel("Optimization Event")
plt.ylabel("Score")
plt.title("Optimization Score Over Time")
```

### 3. Track Instance Stability

```python
changes = [e["changes_count"] for e in timeline]
print(f"Total migrations: {sum(changes)}")
print(f"Average migrations per event: {sum(changes)/len(changes):.2f}")
```

## Validation

The integration has been validated to ensure:

1. ✅ Timeline is properly cleared before each strategy test
2. ✅ Timeline data is successfully retrieved after experiment
3. ✅ Data is correctly saved in results JSON
4. ✅ No errors when Planner is unavailable (graceful degradation)
5. ✅ Timeline entries have correct structure and types
6. ✅ Visualization works with real timeline data

## Future Enhancements

Potential improvements for future iterations:

1. **Real-time Timeline Display**: Stream timeline updates during experiment
2. **Comparative Visualization**: Plot multiple strategies on same graph
3. **Efficiency Metrics**: Calculate migration overhead and efficiency scores
4. **Anomaly Detection**: Flag unusual redeployment patterns
5. **Export Formats**: Support CSV, Parquet for data analysis tools
6. **Interactive Dashboard**: Web-based real-time monitoring

## Troubleshooting

### Common Issues

**Timeline is empty after experiment**:
- Check if Planner is running: `curl http://localhost:8202/health`
- Verify auto-optimization is enabled in `start_all_services.sh`
- Check if initial deployment happened via `redeply.py`

**Connection errors**:
- Verify PLANNER_URL is correct (default: `http://localhost:8202`)
- Check Planner logs: `tail -f logs/planner.log`

**Timeline not in results JSON**:
- Check experiment logs for "Step 17: Retrieving instance deployment timeline"
- Verify no errors during timeline retrieval
- Check `planner_timeline` key exists: `jq '.results[0] | keys' results.json`

## References

- **Planner API**: `planner/src/api.py` (lines 1444-1484)
- **Timeline Tracker**: `planner/src/instance_timeline_tracker.py`
- **Integration**: `test_dynamic_workflow.py`
- **Documentation**: `TIMELINE_COLLECTION_GUIDE.md`

## Summary

This integration provides comprehensive tracking of instance deployment changes throughout experiments, enabling detailed analysis of dynamic resource allocation behavior. The implementation is automatic, non-intrusive, and provides rich data for research and debugging purposes.
