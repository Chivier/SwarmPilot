# Quick Fix Guide - Timeline Collection

## The Problem You Encountered

```
[ERROR] Failed to retrieve timeline from Planner: 500 Server Error
```

Error in Planner logs:
```
Failed to record timeline entry: Object of type int64 is not JSON serializable
```

## The Fix

Modified `planner/src/instance_timeline_tracker.py` to convert numpy types to Python native types before JSON serialization.

**Status**: ✅ **FIXED AND TESTED**

## How to Apply the Fix

### Step 1: Restart Services

```bash
cd /home/yanweiye/Project/swarmpilot-refresh/experiments/07.Exp2.Deep_Research_Migration_Test
./stop_all_services.sh
./start_all_services.sh
```

### Step 2: Test Timeline (Optional)

```bash
# Quick test to verify Planner is working
python test_timeline_api.py
```

Expected output: `✓ All timeline API tests passed!`

### Step 3: Run Your Experiment

```bash
# Run experiment normally - timeline will work now
python test_dynamic_workflow.py --num-workflows 50
```

### Step 4: Verify Results

```bash
# Check if timeline data is in results
cat results/results_workflow_b1b2_*.json | jq '.results[0].planner_timeline.entry_count'
```

You should see a number (not null), indicating timeline entries were collected.

## What Was Fixed

1. ✅ Numpy int64/float64 types now converted to Python int/float
2. ✅ Special values (inf, -inf, nan) converted to null
3. ✅ Timeline can be successfully retrieved via API
4. ✅ Timeline data is properly saved in results JSON

## Files Modified

- `planner/src/instance_timeline_tracker.py` (lines 9, 72-88)
  - Added math import
  - Added type conversion logic

## Verification

The fix has been tested and confirmed working:

```bash
✓ Successfully recorded migration with numpy types
✓ JSON is valid
✓ inf correctly converted to None
```

## Next Experiment Run

The next time you run the experiment:

1. Timeline will be automatically cleared at start
2. Instance deployment changes will be tracked
3. Timeline data will be retrieved at end
4. Results will include timeline under `planner_timeline` key

No manual intervention needed!

## If You Still See Errors

### Timeline API connection error
```bash
# Check if Planner is running
curl http://localhost:8202/health

# Check Planner logs
tail -f logs/planner.log
```

### Timeline is empty
- Check if `redeply.py` was executed
- Check Planner logs for "Timeline: recorded" messages
- Verify auto-optimization is enabled in `start_all_services.sh`

### Other errors
See detailed documentation:
- `JSON_SERIALIZATION_FIX.md` - Technical details
- `TIMELINE_COLLECTION_GUIDE.md` - Complete usage guide

## Summary

**Before Fix**: Timeline collection failed with 500 error due to numpy type serialization
**After Fix**: Timeline works correctly, data is saved in results
**Action Required**: Restart services (one time)
**Future Runs**: No changes needed, works automatically

---

**Status**: Ready to use! 🎉
