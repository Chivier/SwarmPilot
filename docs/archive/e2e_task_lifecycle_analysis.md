# E2E Task Lifecycle Analysis

**Test Date:** 2026-01-07 17:29:50 - 17:31:12
**Test Parameters:** 3 QPS, 15 seconds, 45 tasks
**Scheduling Strategy:** ProbabilisticSchedulingStrategy

---

## 1. Instance Deployment

### 1.1 Instance Registration Timeline

| Time | Instance ID | Model | Port | Status |
|------|-------------|-------|------|--------|
| 17:29:56.844 | sleep_model_a-000 | sleep_model_a | 8100 | ACTIVE |
| 17:29:57.897 | sleep_model_a-001 | sleep_model_a | 8101 | ACTIVE |
| 17:29:58.954 | sleep_model_a-002 | sleep_model_a | 8102 | ACTIVE |
| 17:30:00.017 | sleep_model_a-003 | sleep_model_a | 8103 | ACTIVE |
| 17:30:01.069 | sleep_model_b-000 | sleep_model_b | 8104 | ACTIVE |
| 17:30:02.142 | sleep_model_b-001 | sleep_model_b | 8105 | ACTIVE |
| 17:30:03.195 | sleep_model_b-002 | sleep_model_b | 8106 | ACTIVE |
| 17:30:04.248 | sleep_model_c-000 | sleep_model_c | 8107 | ACTIVE |
| 17:30:05.302 | sleep_model_c-001 | sleep_model_c | 8108 | ACTIVE |
| 17:30:06.357 | sleep_model_c-002 | sleep_model_c | 8109 | ACTIVE |

### 1.2 Instance Distribution Summary

| Model | Instance Count | Port Range |
|-------|---------------|------------|
| sleep_model_a | 4 | 8100-8103 |
| sleep_model_b | 3 | 8104-8106 |
| sleep_model_c | 3 | 8107-8109 |
| **Total** | **10** | 8100-8109 |

### 1.3 Registration Lifecycle

Each instance goes through:
1. **INITIALIZING** - Instance registered with scheduler
2. **Work Stealing Check** - Look for tasks to steal from existing instances
3. **ACTIVE** - Ready to receive tasks

Example log sequence:
```
Registered instance sleep_model_a-000 for model sleep_model_a on cpu (status=INITIALIZING)
[WorkStealing] Started for new instance sleep_model_a-000 (model=sleep_model_a)
[WorkStealing] No donor candidates - no other active instances
[WorkStealing] Instance sleep_model_a-000 status -> ACTIVE (work stealing complete)
```

---

## 2. Task Routing Decisions

### 2.1 First 20 Tasks Routing Table

| # | Task ID | Model | Selected Instance | Queue Status at Selection | Predicted Time |
|---|---------|-------|-------------------|--------------------------|----------------|
| 0 | bench-000000-4a6015 | sleep_model_a | sleep_model_a-000 | All queues empty (0.0ms) | 278.17ms |
| 1 | bench-000001-d4bc25 | sleep_model_b | sleep_model_b-000 | All queues empty (0.0ms) | 944.42ms |
| 2 | bench-000002-9eda81 | sleep_model_c | sleep_model_c-000 | All queues empty (0.0ms) | 586.11ms |
| 3 | bench-000003-fd96dd | sleep_model_a | sleep_model_a-000 | Task 0 completed, all empty | 825.31ms |
| 4 | bench-000004-56c908 | sleep_model_b | sleep_model_b-001 | b-000: 945.77ms queued | 947.58ms |
| 5 | bench-000005-a7fecb | sleep_model_c | sleep_model_c-000 | Task 2 completed, empty | 436.13ms |
| 6 | bench-000006-b0c37a | sleep_model_a | sleep_model_a-000 | a-000 nearly empty (0.34ms p50) | 835.92ms |
| 7 | bench-000007-d50dcf | sleep_model_b | sleep_model_b-002 | b-000: 1.35ms, b-001: 947.58ms | 447.10ms |
| 8 | bench-000008-a6c6b5 | sleep_model_c | sleep_model_c-000 | c-000 nearly empty | 570.96ms |
| 9 | bench-000009-ad0e91 | sleep_model_a | sleep_model_a-001 | a-000: 0.34ms, others empty | 456.86ms |
| 10 | bench-000010-c726a0 | sleep_model_b | sleep_model_b-000 | b-000: 1.35ms, b-001: 0.06ms, b-002: 2.53ms | 723.63ms |
| 11 | bench-000011-d19a94 | sleep_model_c | sleep_model_c-001 | c-000: 0.73ms queued | 539.39ms |
| 12 | bench-000012-08fae0 | sleep_model_a | sleep_model_a-002 | a-000: 0.34ms, a-001: 0.67ms, a-002/3: empty | 415.83ms |
| 13 | bench-000013-588409 | sleep_model_b | sleep_model_b-001 | b-001 has lowest effective queue | 481.72ms |
| 14 | bench-000014-7632b2 | sleep_model_c | sleep_model_c-002 | c-002 empty (lowest queue) | 156.91ms |
| 15 | bench-000015-68e23c | sleep_model_a | sleep_model_a-002 | a-002 has small residual | 649.71ms |
| 16 | bench-000016-b59471 | sleep_model_b | sleep_model_b-001 | Probabilistic selection | 607.58ms |
| 17 | bench-000017-1f3481 | sleep_model_c | sleep_model_c-001 | c-001 has lowest effective | 100.31ms |

### 2.2 Instance Queue Status Examples

**Task 4 (bench-000004-56c908) - sleep_model_b:**
```
Queue Status:
  sleep_model_b-000: p50=945.77ms, p90=1036.29ms, p95=1089.21ms, p99=1123.79ms  <-- BUSY
  sleep_model_b-001: p50=0.00ms, p90=0.00ms, p95=0.00ms, p99=0.00ms              <-- SELECTED
  sleep_model_b-002: p50=0.00ms, p90=0.00ms, p95=0.00ms, p99=0.00ms

Decision: Selected sleep_model_b-001 (empty queue vs b-000's ~946ms queue)
```

**Task 7 (bench-000007-d50dcf) - sleep_model_b:**
```
Queue Status:
  sleep_model_b-000: p50=1.35ms, p90=1.35ms, p95=1.35ms, p99=2.34ms
  sleep_model_b-001: p50=947.58ms, p90=1053.31ms, p95=1100.81ms, p99=1127.62ms  <-- BUSY
  sleep_model_b-002: p50=0.00ms, p90=0.00ms, p95=0.00ms, p99=0.00ms              <-- SELECTED

Decision: Selected sleep_model_b-002 (empty queue)
```

**Task 9 (bench-000009-ad0e91) - sleep_model_a:**
```
Queue Status:
  sleep_model_a-000: p50=0.34ms, p90=5.48ms, p95=6.09ms, p99=6.59ms  <-- Small residual
  sleep_model_a-001: p50=0.00ms, p90=0.00ms, p95=0.00ms, p99=0.00ms  <-- SELECTED (empty)
  sleep_model_a-002: p50=0.00ms, p90=0.00ms, p95=0.00ms, p99=0.00ms
  sleep_model_a-003: p50=0.00ms, p90=0.00ms, p95=0.00ms, p99=0.00ms

Decision: Selected sleep_model_a-001 (completely empty vs a-000's residual)
```

---

## 3. Task Lifecycle Flow

### 3.1 Complete Lifecycle Example (Task 0)

```
17:30:06.405 [TASK_SUBMIT] task_id=bench-000000-4a6015 model_id=sleep_model_a
             metadata_keys=['sleep_time'] input_keys=['sleep_time'] available_instances=4

17:30:06.405 [QUEUE_ENQUEUE] task_id=bench-000000-4a6015 model_id=sleep_model_a
             position=1 queue_size=1

17:30:06.405 [SCHEDULE_INPUT] model_id=sleep_model_a strategy=ProbabilisticSchedulingStrategy
             available_instances=['sleep_model_a-000', 'sleep_model_a-001', 'sleep_model_a-002', 'sleep_model_a-003']

17:30:06.458 [SCHEDULE_PREDICTIONS] model_id=sleep_model_a
             predictions=[sleep_model_a-000:278.17ms, sleep_model_a-001:278.17ms,
                         sleep_model_a-002:278.17ms, sleep_model_a-003:278.17ms]

17:30:06.458 [SCHEDULE_QUEUE_INFO] model_id=sleep_model_a
             queue_info={
               sleep_model_a-000: values=[0.0, 0.0, 0.0, 0.0]
               sleep_model_a-001: values=[0.0, 0.0, 0.0, 0.0]
               sleep_model_a-002: values=[0.0, 0.0, 0.0, 0.0]
               sleep_model_a-003: values=[0.0, 0.0, 0.0, 0.0]
             }

17:30:06.477 [SCHEDULE_RESULT] model_id=sleep_model_a strategy=ProbabilisticSchedulingStrategy
             selected_instance=sleep_model_a-000 predicted_time_ms=278.17

17:30:06.477 [QUEUE_DISPATCH] task_id=bench-000000-4a6015 model_id=sleep_model_a
             selected_instance=sleep_model_a-000 wait_time_ms=72.14

17:30:06.477 [DISPATCH_SEND] task_id=bench-000000-4a6015 instance_id=sleep_model_a-000
             endpoint=http://localhost:8100 model_id=sleep_model_a

17:30:06.484 [DISPATCH_SENT] task_id=bench-000000-4a6015 instance_id=sleep_model_a-000
             response_status=200

17:30:06.846 [CALLBACK_RECEIVED] task_id=bench-000000-4a6015 status=completed
             execution_time_ms=279.14 instance_id=sleep_model_a-000

17:30:06.846 [CALLBACK_QUEUE_UPDATE] task_id=bench-000000-4a6015 instance_id=sleep_model_a-000
             predicted_ms=278.17 actual_ms=279.14
```

### 3.2 Lifecycle Stages Summary

| Stage | Event | Key Information |
|-------|-------|-----------------|
| 1. Submit | TASK_SUBMIT | Task ID, Model ID, Available Instances |
| 2. Enqueue | QUEUE_ENQUEUE | Queue Position, Queue Size |
| 3. Predict | SCHEDULE_PREDICTIONS | Per-instance predicted latency |
| 4. Queue Check | SCHEDULE_QUEUE_INFO | Per-instance queue quantiles |
| 5. Select | SCHEDULE_RESULT | Selected instance, Final prediction |
| 6. Dispatch | QUEUE_DISPATCH | Wait time in queue |
| 7. Send | DISPATCH_SEND/SENT | HTTP status, Endpoint |
| 8. Complete | CALLBACK_RECEIVED | Actual execution time |
| 9. Update | CALLBACK_QUEUE_UPDATE | Prediction accuracy |

---

## 4. Routing Decision Analysis

### 4.1 Queue-Aware Load Balancing

The ProbabilisticSchedulingStrategy considers:
1. **Predicted execution time** - From predictor service
2. **Current queue depth** - Quantile-based (p50, p90, p95, p99)
3. **Probabilistic selection** - Monte Carlo sampling

**Key Observation:** When all instances have empty queues, the scheduler tends to select the first instance. As queues fill up, it distributes load to less busy instances.

### 4.2 Instance Utilization Pattern

Based on the first 18 tasks:

| Instance | Tasks Received | Selection Reason |
|----------|---------------|------------------|
| sleep_model_a-000 | 4 | First available, often empty/near-empty |
| sleep_model_a-001 | 1 | Empty when a-000 had residual |
| sleep_model_a-002 | 2 | Empty when others queued |
| sleep_model_a-003 | 0 | Never selected (others always available) |
| sleep_model_b-000 | 2 | First available, returned to empty |
| sleep_model_b-001 | 4 | Probabilistic selection, good timing |
| sleep_model_b-002 | 1 | Empty when others busy |
| sleep_model_c-000 | 3 | First available |
| sleep_model_c-001 | 2 | Empty when c-000 had residual |
| sleep_model_c-002 | 1 | Empty queue selection |

### 4.3 Prediction Accuracy

Sample comparison (predicted vs actual):

| Task | Predicted | Actual | Error |
|------|-----------|--------|-------|
| bench-000000 | 278.17ms | 279.14ms | +0.35% |
| bench-000001 | 944.42ms | 944.44ms | +0.00% |
| bench-000002 | 586.11ms | 586.81ms | +0.12% |
| bench-000003 | 825.31ms | 826.21ms | +0.11% |
| bench-000004 | 947.58ms | 948.30ms | +0.08% |

**Observation:** Predictions are highly accurate (< 1% error), indicating the predictor service is well-calibrated for this workload.

---

## 5. Summary Statistics

### 5.1 Test Results

| Metric | Value |
|--------|-------|
| Tasks Submitted | 45 |
| Tasks Completed | 45 |
| Tasks Failed | 0 |
| Success Rate | 100% |
| Actual QPS | 3.07 |
| Submission p50 | 5.22ms |
| Submission p99 | 34.73ms |

### 5.2 Log Event Counts

| Event Type | Count | Description |
|------------|-------|-------------|
| TASK_SUBMIT | 45 | Task submissions |
| QUEUE_ENQUEUE | 45 | Queue insertions |
| SCHEDULE_RESULT | 45 | Routing decisions |
| QUEUE_DISPATCH | 45 | Task dispatches |
| DISPATCH_SEND | 45 | HTTP sends |
| CALLBACK_RECEIVED | 45 | Completions |

---

## 6. Key Insights

1. **Load Balancing Works**: The scheduler effectively distributes load across instances using queue-aware probabilistic selection.

2. **Empty Queue Preference**: When queues are empty, the scheduler prefers the first instance in the list, but quickly diversifies as queues fill.

3. **Accurate Predictions**: The predictor service provides highly accurate latency estimates, enabling effective scheduling decisions.

4. **Fast Dispatch**: Queue wait times are typically < 10ms, indicating efficient task processing.

5. **Work Stealing Ready**: The system checks for work stealing opportunities during instance registration, though none were needed in this test (no tasks during registration phase).
