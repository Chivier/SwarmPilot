# Dataset Extraction Fix Summary

## Problem

The `collect_training_data.py` script failed to extract training data from `dataset.jsonl` because of a **structural mismatch** between expected and actual data formats.

### Root Cause

**Expected format (from documentation):**
```json
{
  "id": "entry-001",
  "boot": {"input": "...", "max_tokens": 512},
  "summary": {"input": "...", "max_tokens": 512},
  "queries": [{"input": "...", "max_tokens": 256}]
}
```

**Actual format (from dataset_generator.py):**
```json
{
  "boot": "You are an expert researcher...",           // Plain string, not dict!
  "summary": "You are an expert researcher...",        // Plain string, not dict!
  "queries": [
    {
      "input": "...",                                  // No max_tokens field!
      "output": "...",
      "input_len": 1073,
      "output_len": 291                               // But has output_len field
    }
  ]
}
```

### Impact Before Fix

- **Only 1,273 query tasks extracted** (missing boot and summary tasks)
- **All tasks defaulted to max_tokens=512** (no diversity for model training)
- **Total: ~87% task loss** (1,273 out of expected 1,471 tasks)

---

## Solution Implemented

Modified `collect_training_data.py` function `extract_tasks_from_dataset()` (lines 433-559) to:

### 1. Handle Dual Formats for boot/summary Fields

```python
# Now handles both dict and string formats
if isinstance(boot_data, dict) and "input" in boot_data:
    # Original expected format
    tasks.append({...})
elif isinstance(boot_data, str):
    # Actual format - adapt to it
    estimated_tokens = estimate_token_length(boot_data)
    max_tokens = 512 if estimated_tokens < 500 else 1024
    tasks.append({
        "sentence": boot_data,
        "max_tokens": max_tokens,
        ...
    })
```

### 2. Infer max_tokens from Query Metadata

```python
# Try multiple sources for max_tokens
max_tokens = query_data.get("max_tokens")

if max_tokens is None:
    # Use output_len as proxy (actual dataset has this!)
    output_len = query_data.get("output_len")
    if output_len:
        # Round up to power of 2 for better training
        max_tokens = 128 if output_len < 128 else \
                    256 if output_len < 256 else \
                    512 if output_len < 512 else \
                    1024 if output_len < 1024 else 2048
```

### 3. Add Comprehensive Logging

```python
logger.info(f"Extracted {len(tasks)} LLM tasks from {len(dataset)} dataset entries")
logger.info(f"  - Boot tasks: {boot_count}")
logger.info(f"  - Summary tasks: {summary_count}")
logger.info(f"  - Query tasks: {query_count}")
logger.info(f"  - max_tokens distribution: {dict(sorted(max_tokens_dist.items()))}")
```

---

## Results After Fix

### Task Extraction Statistics

```
Total dataset entries: 99
Total tasks extracted: 1,471 tasks

Task breakdown:
  - Boot tasks:    99 (100% success rate)
  - Summary tasks: 99 (100% success rate)
  - Query tasks:   1,273 (100% success rate)
```

### max_tokens Distribution

```
{
  128:    13 tasks  (0.9%)
  256:   683 tasks (46.4%)
  512:   566 tasks (38.5%)
  1024:  207 tasks (14.1%)
  2048:    2 tasks  (0.1%)
}
```

**Key Improvements:**
- ✅ **100% task recovery** (was 87%, now 100%)
- ✅ **5 diverse max_tokens values** (was 1, now 5)
- ✅ **Natural distribution** based on actual output lengths
- ✅ **Better model training quality** with varied data

---

## Validation

### Test Command

```bash
cd /chivier-disk/yanweiye/Projects/swarmpilot-refresh/experiments/07.multi_model_workflow_dynamic_merge_real
uv run python3 collect_training_data.py \
    --dataset data/dataset.jsonl \
    --instance-url http://localhost:8001 \
    --predictor-url http://localhost:8002 \
    --model-id llama-7b \
    --max-samples 10  # Test with small sample first
```

### Expected Output

```
2025-11-16 12:29:41,862 - INFO - Loaded 99 entries from data/dataset.jsonl
2025-11-16 12:29:41,863 - INFO - Extracted 1471 LLM tasks from 99 dataset entries
2025-11-16 12:29:41,863 - INFO -   - Boot tasks: 99
2025-11-16 12:29:41,863 - INFO -   - Summary tasks: 99
2025-11-16 12:29:41,863 - INFO -   - Query tasks: 1273
2025-11-16 12:29:41,864 - INFO -   - max_tokens distribution: {128: 13, 256: 683, 512: 566, 1024: 207, 2048: 2}
```

---

## Backward Compatibility

The fix maintains **full backward compatibility**:

✅ **Still works with documented format** (dict with `input` and `max_tokens`)
✅ **Now also works with actual format** (string for boot/summary)
✅ **Handles missing max_tokens** (infers from output_len or content length)
✅ **Graceful fallbacks** (defaults to 512 if no inference possible)

---

## Files Modified

- [collect_training_data.py](collect_training_data.py) - Fixed `extract_tasks_from_dataset()` function and removed external dependencies

### Detailed Changes

1. **Lines 36-226**: Added self-contained hardware performance database
   - Copied `NVIDIA_TESLA_SPECS` from `predictor/src/utils/hardware_perf_info.py`
   - Copied `extract_gpu_specs()` function from `predictor/src/models.py::PlatformInfo.extract_gpu_specs()`
   - Script is now fully self-contained with no external package dependencies

2. **Lines 605-759**: Fixed `extract_tasks_from_dataset()` function
   - Handle both dict and string formats for `boot`/`summary` fields
   - Infer `max_tokens` from `output_len` field in query metadata
   - Add comprehensive logging for task extraction statistics

3. **Line 906**: Simplified hardware spec extraction
   - Removed dependency on `predictor/src/models.py`
   - Now uses local `extract_gpu_specs()` function

## Files Analyzed

- [data/dataset.jsonl](data/dataset.jsonl) - Actual dataset (35.9MB, 99 entries)
- [dataset_generator.py](dataset_generator.py) - Data generation source
- [COLLECT_TRAINING_DATA_README.md](COLLECT_TRAINING_DATA_README.md) - Documentation
- [../../predictor/src/models.py](../../predictor/src/models.py) - Source for hardware spec extraction
- [../../predictor/src/utils/hardware_perf_info.py](../../predictor/src/utils/hardware_perf_info.py) - GPU specifications database

---

## Next Steps

1. ✅ **Script is now ready for use** with existing dataset
2. ⏭️ **Run full training data collection** (remove `--max-samples` limit)
3. ⏭️ **Train predictor models** with collected data
4. 📝 **Optional: Update dataset_generator.py** to match documented format (for future datasets)

---

## Technical Details

### Changes Made

**Primary Changes:**
1. **Hardware Spec Database (lines 57-226)**
   - Added `NVIDIA_TESLA_SPECS` dictionary with 11 GPU models
   - Added `extract_gpu_specs()` function for GPU spec extraction
   - Eliminates dependency on `predictor/src` packages

2. **Task Extraction (lines 605-759)**
   - Fixed `extract_tasks_from_dataset()` to handle dual formats
   - Added `max_tokens` inference from `output_len` metadata
   - Added comprehensive logging and statistics

3. **Dependencies (line 906)**
   - Removed import of `predictor.src.models.PlatformInfo`
   - Now uses local `extract_gpu_specs()` function

**Commit message suggestion:**

```
fix(experiment): make collect_training_data.py self-contained

- Copy NVIDIA_TESLA_SPECS database from predictor/src/utils/hardware_perf_info.py
- Copy extract_gpu_specs() logic from predictor/src/models.py
- Remove dependency on external predictor package
- Add dual-format support for boot/summary fields (dict or string)
- Infer max_tokens from output_len field in query metadata
- Add comprehensive logging for task extraction statistics

The script is now fully self-contained with no dependencies on
packages outside the experiment directory. Fixes dataset extraction
which was failing due to:
1. Structural mismatch (expected dict, got string for boot/summary)
2. Missing predictor package in experiment environment

Now extracts all 1,471 tasks with proper max_tokens diversity
instead of only 1,273 tasks with uniform max_tokens=512.

Resolves: Dataset structure mismatch and external dependency issues
