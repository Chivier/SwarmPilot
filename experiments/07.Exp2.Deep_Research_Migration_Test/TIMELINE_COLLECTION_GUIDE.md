# Instance Deployment Timeline Collection Guide

## Overview

This experiment now automatically collects instance deployment timeline data from the Planner service. The timeline tracks how instance allocations change over time during the experiment, including both initial deployment and auto-optimization events.

## What is Collected

The timeline includes the following events:

1. **Initial Deployment** (`deploy_migration`): When instances are first deployed or manually redeployed
2. **Auto-Optimization** (`auto_optimize`): When the Planner automatically adjusts instance allocation based on workload

Each timeline entry contains:
- `timestamp`: Unix timestamp of the event
- `timestamp_iso`: ISO 8601 formatted timestamp
- `event_type`: Either `"deploy_migration"` or `"auto_optimize"`
- `instance_counts`: Dictionary mapping model_id to instance count (e.g., `{"sleep_model_a": 5, "sleep_model_b": 11}`)
- `total_instances`: Total number of instances across all models
- `changes_count`: Number of instance model changes in this event
- `success`: Whether the migration/optimization was successful
- `target_distribution`: Target distribution used for optimization (if applicable)
- `score`: Optimization score achieved (if applicable)

## How It Works

### Automatic Integration

The timeline collection is automatically integrated into the experiment workflow:

1. **Before each strategy test** (`test_strategy_workflow`):
   - Timeline is cleared via `POST /timeline/clear` to the Planner
   - Ensures clean data for each strategy run

2. **During the experiment**:
   - Planner automatically records timeline events when:
     - Initial deployment happens (`/deploy/migration` endpoint)
     - Auto-optimization triggers (based on scheduler reports)

3. **After each strategy test**:
   - Timeline is retrieved via `GET /timeline` from the Planner
   - Data is included in the results dictionary under `"planner_timeline"` key
   - Results are saved to JSON file with all other metrics

### Timeline Data Location

Timeline data is saved in two locations:

1. **Planner logs directory**: `./logs/instance_count_timeline.json`
   - Persistent storage maintained by Planner
   - Updated in real-time as events occur
   - Shared across all experiments until cleared

2. **Experiment results file**: `results/results_workflow_b1b2_<timestamp>.json`
   - Copied at the end of each strategy test
   - Stored under `results[i]["planner_timeline"]` for strategy `i`
   - Allows comparison of deployment patterns across strategies

## Usage

### Running the Experiment

Simply run the experiment as usual:

```bash
# Run with default parameters
python test_dynamic_workflow.py

# Run with custom parameters
python test_dynamic_workflow.py --num-workflows 50 --qps-a 5.0
```

The timeline will be automatically collected and included in the results.

### Testing Timeline Collection

Before running the full experiment, you can test if timeline collection is working:

```bash
# Make sure services are running first
./start_all_services.sh

# Test timeline API
python test_timeline_api.py
```

This will verify that:
- Planner service is running and accessible
- Timeline clear endpoint works
- Timeline retrieval endpoint works
- Data structure is correct

### Accessing Timeline Data

After the experiment completes, timeline data is available in the results JSON:

```python
import json

# Load results
with open("results/results_workflow_b1b2_<timestamp>.json", "r") as f:
    data = json.load(f)

# Access timeline for each strategy
for result in data["results"]:
    strategy = result["strategy"]
    timeline = result["planner_timeline"]

    if timeline and timeline.get("success"):
        entries = timeline["entries"]
        print(f"\n{strategy}: {len(entries)} timeline events")

        for entry in entries:
            print(f"  {entry['timestamp_iso']}: {entry['event_type']}")
            print(f"    Instance counts: {entry['instance_counts']}")
            print(f"    Changes: {entry['changes_count']}, Success: {entry['success']}")
```

## Analyzing Timeline Data

### Visualization Example

You can visualize how instance allocation changes over time:

```python
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load results
with open("results/results_workflow_b1b2_<timestamp>.json", "r") as f:
    data = json.load(f)

# Extract timeline for a specific strategy
timeline = data["results"][0]["planner_timeline"]["entries"]

# Parse data
timestamps = [datetime.fromisoformat(e["timestamp_iso"]) for e in timeline]
model_a_counts = [e["instance_counts"].get("sleep_model_a", 0) for e in timeline]
model_b_counts = [e["instance_counts"].get("sleep_model_b", 0) for e in timeline]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps, model_a_counts, 'o-', label='Model A instances', linewidth=2)
plt.plot(timestamps, model_b_counts, 's-', label='Model B instances', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Instance Count')
plt.title('Instance Allocation Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('instance_timeline.png')
```

### Key Metrics to Analyze

1. **Redeployment Frequency**:
   - Count of `auto_optimize` events
   - Time intervals between redeployments

2. **Instance Allocation Patterns**:
   - How instance counts change for each model
   - Correlation with workload changes

3. **Optimization Effectiveness**:
   - Scores over time
   - Success rate of migrations

4. **System Stability**:
   - Frequency of changes
   - Convergence to optimal allocation

## Configuration

### Planner Auto-Optimization

The Planner's auto-optimization behavior is configured in `start_all_services.sh`:

```bash
AUTO_OPTIMIZE_ENABLED=True AUTO_OPTIMIZE_INTERVAL=150
```

- `AUTO_OPTIMIZE_ENABLED`: Enable/disable auto-optimization
- `AUTO_OPTIMIZE_INTERVAL`: Seconds between optimization checks

### Timeline Storage Location

Default location is `./logs/instance_count_timeline.json` relative to the Planner service directory. This can be changed in `planner/src/instance_timeline_tracker.py` if needed.

## Troubleshooting

### Timeline is Empty

If the timeline has no entries after running an experiment:

1. **Check if Planner is running**:
   ```bash
   curl http://localhost:8202/health
   ```

2. **Check if auto-optimization is enabled**:
   - Look in `start_all_services.sh` for `AUTO_OPTIMIZE_ENABLED=True`
   - Check Planner logs: `logs/planner.log`

3. **Verify initial deployment happened**:
   - Check if `./redeply.py` was executed
   - Look for "deploy_migration" events in Planner logs

### Timeline API Connection Errors

If you see connection errors when retrieving timeline:

1. **Verify PLANNER_URL**:
   - Default is `http://localhost:8202`
   - Check `test_dynamic_workflow.py` line 75

2. **Check Planner service logs**:
   ```bash
   tail -f logs/planner.log
   ```

3. **Test API manually**:
   ```bash
   # Clear timeline
   curl -X POST http://localhost:8202/timeline/clear

   # Get timeline
   curl http://localhost:8202/timeline | jq
   ```

### Timeline Data Missing from Results

If timeline data is not in the results JSON:

1. **Check for errors in experiment logs**:
   - Look for "Step 17: Retrieving instance deployment timeline"
   - Check for warning messages about timeline retrieval

2. **Verify results structure**:
   ```bash
   # Check if planner_timeline key exists
   cat results/results_workflow_b1b2_<timestamp>.json | jq '.results[0] | keys'
   ```

## Integration Details

### Code Changes

The following changes were made to integrate timeline collection:

1. **`test_dynamic_workflow.py`**:
   - Added `PLANNER_URL` constant (line 75)
   - Added `clear_planner_timeline()` function (line 2176)
   - Added `get_planner_timeline()` function (line 2208)
   - Integrated timeline clearing in Step 1 (line 2712)
   - Integrated timeline retrieval in Step 17 (line 3175)
   - Added timeline data to return results (line 3195)

2. **New files**:
   - `test_timeline_api.py`: Standalone test script for timeline API
   - `TIMELINE_COLLECTION_GUIDE.md`: This documentation

### API Endpoints Used

- `POST /timeline/clear`: Clear timeline for new experiment
  - Response: `{"success": bool, "message": str}`

- `GET /timeline`: Retrieve timeline data
  - Response: `{"success": bool, "entry_count": int, "entries": List[Dict]}`

## Example Timeline Entry

```json
{
  "timestamp": 1703001234.567,
  "timestamp_iso": "2023-12-19T12:00:34.567000+00:00",
  "event_type": "auto_optimize",
  "instance_counts": {
    "sleep_model_a": 6,
    "sleep_model_b": 10
  },
  "total_instances": 16,
  "changes_count": 2,
  "success": true,
  "target_distribution": [8.5, 42.3],
  "score": 0.0234
}
```

## References

- Planner timeline tracker implementation: `planner/src/instance_timeline_tracker.py`
- Planner API timeline endpoints: `planner/src/api.py` (lines 1444-1484)
- Experiment integration: `test_dynamic_workflow.py`
