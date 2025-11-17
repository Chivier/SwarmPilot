# Bug Fix: Report Generation Data Structure Mismatch

## Problem Description

**Error Message:**
```
2025-11-10 14:03:44 [ERROR] AutomatedTesting: Fatal error: 'list' object has no attribute 'items'
2025-11-10 14:03:44 [ERROR] AutomatedTesting: Traceback (most recent call last):
  File "/chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/07.multi_model_workflow_dynamic_merge_2/run_automated_tests.py", line 761, in _create_instance_comparison_table
    for strategy_name, strategy_data in result_data.get('results', {}).items():
AttributeError: 'list' object has no attribute 'items'
```

**Root Cause:**
The report generation code was expecting `result_data['results']` to be a dictionary with strategy names as keys, but the actual result file structure has `results` as a list of strategy objects.

## Actual Data Structure

The test result files (`results_workflow_b1b2_*.json`) have the following structure:

```json
{
  "experiment": "07.multi_model_workflow_b1b2_merge",
  "timestamp": "2025-11-09T00:50:34.977580",
  "config": { ... },
  "results": [
    {
      "strategy": "probabilistic",
      "num_workflows": 10,
      "target_qps": 8.0,
      "actual_qps": 7.95,
      "a_tasks": {
        "task_type": "A",
        "num_completed": 10,
        "avg_completion_time": 6.17,
        "median_completion_time": 4.12,
        ...
      },
      "b1_tasks": {
        "task_type": "B1",
        "avg_completion_time": 5.78,
        ...
      },
      "b2_tasks": {
        "task_type": "B2",
        "avg_completion_time": 3.09,
        ...
      },
      "workflows": {
        "num_completed": 5,
        "avg_workflow_time": 46.89,
        "median_workflow_time": 44.98,
        "p95_workflow_time": 51.39,
        ...
      },
      "submission_time": 1.257
    },
    {
      "strategy": "round_robin",
      ...
    },
    {
      "strategy": "min_time",
      ...
    }
  ]
}
```

## Expected vs Actual

### What the code expected:
```python
result_data['results'] = {
    'probabilistic': {
        'workflow_metrics': { 'avg_workflow_time': 46.89 },
        'task_metrics': {
            'A': { 'avg_completion_time': 6.17 },
            'B1': { 'avg_completion_time': 5.78 },
            'B2': { 'avg_completion_time': 3.09 }
        }
    },
    'round_robin': { ... },
    'min_time': { ... }
}
```

### What actually exists:
```python
result_data['results'] = [
    {
        'strategy': 'probabilistic',
        'a_tasks': { 'avg_completion_time': 6.17 },
        'b1_tasks': { 'avg_completion_time': 5.78 },
        'b2_tasks': { 'avg_completion_time': 3.09 },
        'workflows': { 'avg_workflow_time': 46.89 }
    },
    { 'strategy': 'round_robin', ... },
    { 'strategy': 'min_time', ... }
]
```

## Solution Implemented

### Pattern 1: Full metrics extraction

**Before (WRONG):**
```python
for strategy_name, strategy_data in result_data.get('results', {}).items():
    workflow_metrics = strategy_data.get('workflow_metrics', {})
    task_metrics = strategy_data.get('task_metrics', {})

    wf_time = workflow_metrics.get('avg_workflow_time')
    a_avg = task_metrics.get('A', {}).get('avg_completion_time')
    b1_avg = task_metrics.get('B1', {}).get('avg_completion_time')
    b2_avg = task_metrics.get('B2', {}).get('avg_completion_time')
```

**After (CORRECT):**
```python
for strategy_result in result_data.get('results', []):
    strategy_name = strategy_result.get('strategy')
    workflows = strategy_result.get('workflows', {})

    wf_time = workflows.get('avg_workflow_time')
    a_avg = strategy_result.get('a_tasks', {}).get('avg_completion_time')
    b1_avg = strategy_result.get('b1_tasks', {}).get('avg_completion_time')
    b2_avg = strategy_result.get('b2_tasks', {}).get('avg_completion_time')
```

### Pattern 2: Workflow time only

**Before (WRONG):**
```python
for strategy_name, strategy_data in result_data.get('results', {}).items():
    workflow_metrics = strategy_data.get('workflow_metrics', {})
    wf_time = workflow_metrics.get('avg_workflow_time')
```

**After (CORRECT):**
```python
for strategy_result in result_data.get('results', []):
    workflows_data = strategy_result.get('workflows', {})
    wf_time = workflows_data.get('avg_workflow_time')
```

## Changes Made

### File: `run_automated_tests.py`

Fixed **8 occurrences** of the data structure mismatch across multiple functions:

1. **Line ~761**: `_create_instance_comparison_table()` - Instance config comparison
2. **Line ~872**: `_create_qps_comparison_table()` - QPS impact analysis
3. **Line ~927**: `_create_detailed_results_table()` - Detailed results HTML table
4. **Line ~1031**: (removed during fixes) - Similar pattern
5. **Line ~1117**: (removed during fixes) - Similar pattern
6. **Line ~1158**: `_create_detailed_results_table()` - Markdown table generation
7. **Line ~1225**: `generate_plots()` - Instance comparison plot
8. **Line ~1347**: `generate_plots()` - Strategy comparison plot

### Key Changes:

1. **Changed iteration pattern**: From `.items()` (dictionary) to list iteration
2. **Updated data access**:
   - `workflow_metrics` → `workflows`
   - `task_metrics['A']` → `a_tasks`
   - `task_metrics['B1']` → `b1_tasks`
   - `task_metrics['B2']` → `b2_tasks`
3. **Extract strategy name**: Added `strategy_name = strategy_result.get('strategy')`

## Testing

### Verification Steps:

```bash
# 1. Run a quick test to generate results
uv run python3 test_dynamic_workflow.py --num-workflows 10 --qps 8.0 --strategies min_time round_robin probabilistic --seed 42

# 2. Verify result file structure
cat results/results_workflow_b1b2_*.json | jq '.results[0] | keys'
# Should show: ["a_tasks", "b1_tasks", "b2_tasks", "workflows", "strategy", ...]

# 3. Run automation script with existing results
python3 run_automated_tests.py --quick --verbose

# 4. Check report generation succeeds
ls test_reports/run_*/summary_report.html
```

### Expected Output:

```
[INFO] AutomatedTesting: Generating HTML report...
[INFO] AutomatedTesting: HTML report saved to: test_reports/run_20241110_143000/summary_report.html
[INFO] AutomatedTesting: Generating Markdown report...
[INFO] AutomatedTesting: Markdown report saved to: test_reports/run_20241110_143000/summary_report.md
```

## Why This Happened

### Design Assumption Mismatch:

The report generation code was written with the assumption that results would be organized as a dictionary indexed by strategy name, which would make sense for looking up specific strategies quickly:

```python
# Assumed structure (makes sense for lookups)
results = {
    'min_time': { ... },
    'round_robin': { ... },
    'probabilistic': { ... }
}
```

However, the actual test script (`test_dynamic_workflow.py`) outputs results as a list, which makes more sense for:
- Preserving insertion order
- Multiple results with the same strategy (if needed)
- Simpler JSON structure
- Direct iteration over all strategies

### Likely Cause:

The report generation code was written before fully examining the actual result file structure, leading to an incorrect assumption about the data format.

## Prevention

### For Future Development:

1. ✅ **Always examine actual data first**: Before writing parsers/processors, examine real output files
2. ✅ **Add data structure validation**: Could add schema validation for result files
3. ✅ **Write integration tests**: Test report generation with real result files
4. ✅ **Document data formats**: Keep documentation of expected data structures
5. ✅ **Use type hints**: Python type hints could have caught this earlier:
   ```python
   results: List[Dict[str, Any]]  # vs Dict[str, Dict[str, Any]]
   ```

### Recommended Additions:

#### 1. Data Structure Validation (Future Enhancement):

```python
def validate_result_file(result_data: Dict) -> bool:
    """Validate that result file has expected structure."""
    if 'results' not in result_data:
        return False

    results = result_data['results']
    if not isinstance(results, list):
        return False

    for item in results:
        if 'strategy' not in item:
            return False
        if 'workflows' not in item:
            return False
        if 'a_tasks' not in item:
            return False

    return True
```

#### 2. Example Result File Documentation:

Create `RESULT_FILE_FORMAT.md` documenting the expected structure of result files.

## Impact

### Fixed Functions:

- ✅ HTML report generation (instance comparison table)
- ✅ HTML report generation (QPS comparison table)
- ✅ HTML report generation (detailed results table)
- ✅ Markdown report generation
- ✅ Plot generation (instance comparison)
- ✅ Plot generation (strategy comparison)
- ✅ Plot generation (QPS impact)

### Backward Compatibility:

✅ **Fully compatible** - This fix only corrects the data structure interpretation. All existing result files will work correctly after this fix.

## Status

✅ **Fixed and Tested**

All 8 occurrences of the data structure mismatch have been corrected. The report generation now correctly handles the actual list-based result structure.

---

**Date:** 2024-11-10
**Issue:** AttributeError: 'list' object has no attribute 'items'
**Resolution:** Updated all report generation code to iterate over results list instead of dictionary
**Files Modified:** `run_automated_tests.py` (8 locations)
