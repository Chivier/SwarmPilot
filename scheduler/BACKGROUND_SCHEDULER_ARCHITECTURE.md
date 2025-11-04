# Background Scheduler Architecture

## 概述

将CPU密集型任务调度操作移至后台处理，使API端点立即返回，从根本上解决高负载下的WebSocket超时问题。

---

## 问题分析

### 当前架构（同步调度）

```
客户端请求 POST /task/submit
    ↓
[1] 验证任务不存在 (1ms)
    ↓
[2] 获取可用实例 (5ms)
    ↓
[3] 调用调度策略 schedule_task()
    ├─ 调用predictor服务获取预测 (10-50ms 网络I/O)
    ├─ 执行蒙特卡洛采样更新队列 (1-5ms CPU密集)
    └─ 选择最优实例 (1ms)
    ↓
[4] 创建任务记录 (2ms)
    ↓
[5] 更新实例统计 (1ms)
    ↓
[6] 分发任务到实例 (fire-and-forget)
    ↓
[7] 返回响应
```

**总耗时：20-70ms per task**

**在500 workflow (3500 tasks)场景下：**
- 如果串行：3500 × 50ms = **175秒**
- 即使并发50：3500 ÷ 50 × 50ms = **3.5秒**
- **问题**：在步骤3期间，事件循环被占用：
  - predictor_client有Lock，强制串行
  - 蒙特卡洛采样是CPU密集操作
  - 无法及时响应WebSocket keepalive ping

---

### 新架构（后台调度）

```
客户端请求 POST /task/submit
    ↓
[1] 验证任务不存在 (1ms)
    ↓
[2] 创建任务记录 (状态=PENDING, instance=null) (2ms)
    ↓
[3] 提交到后台调度器 (非阻塞，<1ms)
    ↓
[4] 立即返回响应 ✅
    ↓
总耗时：~5ms per task

────────────────────────────────────────
后台异步处理（不阻塞API响应）：

BackgroundScheduler.schedule_task_background()
    ↓
[后台任务开始]
    ↓
[1] 获取可用实例 (5ms)
    ↓
[2] 调用调度策略 schedule_task()
    ├─ 调用predictor服务获取预测 (10-50ms)
    ├─ 执行蒙特卡洛采样 (1-5ms)
    └─ 选择最优实例 (1ms)
    ↓
[3] 更新任务记录（添加instance和预测信息）
    ↓
[4] 更新实例统计
    ↓
[5] 分发任务到实例
    ↓
[后台任务结束]
```

**API响应时间：~5ms per task**
**后台处理：仍需20-70ms，但不阻塞API**

---

## 核心实现

### 1. BackgroundScheduler类

```python
class BackgroundScheduler:
    def __init__(
        self,
        scheduling_strategy,
        task_registry,
        instance_registry,
        task_dispatcher,
        max_concurrent_scheduling: int = 50,  # 关键参数
    ):
        # 限制并发调度操作，防止事件循环过载
        self._semaphore = asyncio.Semaphore(max_concurrent_scheduling)

        # 追踪活跃的后台任务
        self._active_tasks: Dict[str, asyncio.Task] = {}
```

**关键特性：**

#### A. 并发控制
```python
async with self._semaphore:
    # 最多50个任务同时进行调度
    # 第51个任务会等待，直到有槽位释放
    await self.scheduling_strategy.schedule_task(...)
```

**为什么是50？**
- 500 workflows = 3500 tasks
- 如果50个并发调度，每个50ms完成
- 总时间：3500 ÷ 50 × 50ms = **3.5秒**完成所有调度
- 避免过度并发（100+）导致事件循环仍然过载

#### B. 非阻塞提交
```python
def schedule_task_background(self, task_id, ...):
    # 创建后台任务
    task = asyncio.create_task(self._schedule_task_async(...))

    # 立即返回，不等待完成
    self._active_tasks[task_id] = task

    # 自动清理
    task.add_done_callback(lambda t: self._active_tasks.pop(task_id, None))
```

#### C. 错误处理
```python
async def _schedule_task_async(self, task_id, ...):
    try:
        # 调度逻辑
        ...
    except Exception as e:
        # 自动标记任务为FAILED
        await self.task_registry.update_status(task_id, TaskStatus.FAILED)
        await self.task_registry.set_error(task_id, str(e))
```

### 2. API端点修改

**修改前（/task/submit）：**
```python
async def submit_task(request):
    # 1. 验证
    if await task_registry.get(request.task_id):
        raise HTTPException(400, "Task exists")

    # 2. 获取实例
    instances = await instance_registry.list_active(...)

    # 3. 🔴 阻塞：调度（20-70ms）
    schedule_result = await scheduling_strategy.schedule_task(...)

    # 4. 创建任务（带实例信息）
    task = await task_registry.create_task(
        ...,
        assigned_instance=schedule_result.selected_instance_id,
        predicted_time_ms=...,
    )

    # 5. 分发
    task_dispatcher.dispatch_task_async(task_id)

    # 6. 返回（总耗时：20-70ms）
    return TaskSubmitResponse(task=...)
```

**修改后（/task/submit）：**
```python
async def submit_task(request):
    # 1. 验证
    if await task_registry.get(request.task_id):
        raise HTTPException(400, "Task exists")

    # 2. 立即创建任务（状态=PENDING，无实例）
    task = await task_registry.create_task(
        ...,
        assigned_instance="",  # 🔑 空值，后台填充
        predicted_time_ms=None,  # 🔑 后台填充
    )

    # 3. ✅ 非阻塞：提交到后台调度器
    background_scheduler.schedule_task_background(
        task_id=request.task_id,
        model_id=request.model_id,
        ...
    )

    # 4. 立即返回（总耗时：~5ms）
    return TaskSubmitResponse(
        task=TaskInfo(
            status="PENDING",
            assigned_instance=None,  # 还未分配
        )
    )
```

---

## 性能影响分析

### API响应时间

| 场景 | 当前架构 | 新架构 | 改进 |
|------|---------|--------|------|
| 单个任务 | 20-70ms | ~5ms | **4-14x faster** |
| 10个任务（串行） | 200-700ms | 50ms | **4-14x faster** |
| 100个任务 | 2-7秒 | 500ms | **4-14x faster** |
| 500 workflows (3500任务) | 70-245秒 | 17.5秒 | **4-14x faster** |

### 事件循环响应性

| 指标 | 当前架构 | 新架构 |
|------|---------|--------|
| WebSocket ping响应 | ❌ 阻塞，可能超时 | ✅ <10ms，不阻塞 |
| 并发请求处理 | ❌ 受限于调度速度 | ✅ 立即响应 |
| CPU密集操作影响 | ❌ 直接阻塞事件循环 | ✅ 后台处理，不影响 |

### 资源使用

| 资源 | 当前架构 | 新架构 |
|------|---------|--------|
| 内存 | 低 | 略高（追踪50个后台任务） |
| CPU | 高峰期100% | 平滑分布，峰值降低 |
| 网络连接 | 串行predictor调用 | 最多50个并发predictor调用 |
| 事件循环负载 | 高（阻塞式） | 低（非阻塞） |

---

## 权衡与考虑

### ✅ 优势

1. **API响应极快**
   - 5ms vs 20-70ms，用户体验显著提升
   - 吞吐量提升4-14倍

2. **解决WebSocket超时**
   - 事件循环不再被长时间占用
   - 可以及时响应keepalive ping
   - 500 workflow场景不再超时

3. **更好的并发控制**
   - Semaphore限制避免过度并发
   - 资源使用更可预测

4. **优雅的错误处理**
   - 调度失败自动标记任务FAILED
   - 不影响其他任务

### ⚠️ 权衡/挑战

1. **任务状态延迟**
   - **问题**：客户端收到响应时，任务还是PENDING，没有assigned_instance
   - **影响**：客户端需要轮询或使用WebSocket监听任务状态更新
   - **解决方案**：
     ```python
     # 客户端代码
     response = submit_task(task_id)
     # response.task.status = "PENDING"
     # response.task.assigned_instance = None

     # 方案1：轮询
     while True:
         task = get_task(task_id)
         if task.assigned_instance:
             break
         await asyncio.sleep(0.1)

     # 方案2：WebSocket订阅（推荐）
     ws.subscribe([task_id])
     # 等待任务状态更新
     ```

2. **调度失败的可见性**
   - **问题**：API返回成功，但后台调度可能失败
   - **影响**：客户端认为任务已提交，但实际上失败了
   - **解决方案**：
     - 后台失败自动标记任务为FAILED
     - 客户端通过WebSocket获得失败通知
     - 或定期检查任务状态

3. **调试复杂性增加**
   - **问题**：错误发生在后台，不在API请求上下文中
   - **影响**：日志关联变复杂
   - **解决方案**：
     - 详细的后台任务日志（已实现）
     - 添加trace_id关联（可选）

4. **内存占用增加**
   - **问题**：_active_tasks字典追踪后台任务
   - **影响**：3500个任务 × ~1KB = 约3.5MB
   - **评估**：可接受，现代服务器内存充足

5. **Predictor服务负载**
   - **问题**：50个并发predictor调用 vs 当前的串行（有Lock）
   - **影响**：predictor服务QPS增加50倍
   - **解决方案**：
     - 确保predictor服务能处理50 QPS
     - 必要时降低max_concurrent_scheduling
     - 或在predictor服务添加缓存

### 🔧 潜在问题与解决方案

#### 问题1：实例统计不准确的竞态条件

**场景**：
```python
# Task A和B都选择了同一个instance
# 因为在选择时看到的queue状态相同
Task A: 选择 inst-1 (queue=0)
Task B: 选择 inst-1 (queue=0)  # 同时发生
# 结果：inst-1被分配2个任务，但选择时以为只有0个
```

**当前设计的保护**：
- Semaphore限制并发为50，减少竞态
- 实例统计使用asyncio.Lock保护更新
- Queue更新是原子操作

**是否需要额外保护**：
- 短期：不需要，调度策略本身是最优化的尽力而为
- 长期：可以添加实例级锁（但会降低性能）

#### 问题2：任务提交速率超过调度速率

**场景**：
- 客户端以100 QPS提交任务
- 后台只能处理50个并发调度
- 积压队列增长

**当前设计的保护**：
- Semaphore自动排队：第51个任务等待槽位释放
- asyncio.Task自动管理，不会OOM
- 优雅降级：任务排队但最终会被处理

**监控指标**（需要添加）：
```python
stats = await background_scheduler.get_stats()
# {
#   "active_scheduling_tasks": 50,  # 当前进行中
#   "available_slots": 0,  # 可用槽位
# }
# 如果 available_slots = 0 持续超过10秒，说明积压
```

---

## 实现细节

### 任务状态转换

**当前架构：**
```
submit_task() →  [PENDING + assigned_instance] → dispatch → [RUNNING]
```

**新架构：**
```
submit_task() → [PENDING + NO instance]
                    ↓
    (后台) schedule_task_background()
                    ↓
            [PENDING + assigned_instance]
                    ↓
            dispatch → [RUNNING]
```

### 数据库/内存状态

**TaskRecord字段变化：**
- `assigned_instance`: 初始为空字符串`""`，后台填充
- `predicted_time_ms`: 初始为`None`，后台填充
- `predicted_quantiles`: 初始为`None`，后台填充

**客户端兼容性：**
- 旧客户端：可能期望立即获得assigned_instance，需要更新
- 新客户端：知道PENDING状态时instance可能为null

### 错误处理流程

```
submit_task()
    ↓
    错误：任务已存在 → 返回400（同步检测）
    ↓
    创建任务成功 → 返回200 + PENDING
    ↓
background_scheduler
    ↓
    错误：无可用实例 → 标记任务FAILED，WebSocket通知客户端
    ↓
    错误：predictor超时 → 标记任务FAILED，WebSocket通知客户端
    ↓
    错误：内部异常 → 标记任务FAILED，记录详细日志
    ↓
    成功：分配实例 → 更新任务，dispatch到实例
```

---

## 迁移策略

### 阶段1：准备（已完成）
- ✅ 实现BackgroundScheduler类
- ✅ 集成到API初始化

### 阶段2：API修改（进行中）
- ⏳ 修改/task/submit端点
- ⏳ 更新返回值和文档
- ⏳ 添加shutdown处理

### 阶段3：测试
- 单元测试：BackgroundScheduler
- 集成测试：修改后的/task/submit
- 性能测试：500 workflow场景
- 压力测试：3500任务并发提交

### 阶段4：监控
- 添加/stats端点显示后台调度统计
- 添加日志和指标
- 监控积压队列长度

### 阶段5：优化（可选）
- 根据predictor服务能力调整max_concurrent_scheduling
- 添加自适应并发控制
- 实现任务优先级队列

---

## 配置参数

```python
# src/api.py
background_scheduler = BackgroundScheduler(
    scheduling_strategy=scheduling_strategy,
    task_registry=task_registry,
    instance_registry=instance_registry,
    task_dispatcher=task_dispatcher,
    max_concurrent_scheduling=50,  # 🔧 可调整
)
```

**max_concurrent_scheduling调优指南：**

| 值 | 场景 | 优点 | 缺点 |
|----|------|------|------|
| 10 | Predictor服务弱 | Predictor负载低 | 调度慢（350任务需175秒） |
| 25 | 保守配置 | 平衡 | 中等速度（350任务需70秒） |
| 50 | **推荐（默认）** | API快速响应 | Predictor需处理50 QPS |
| 100 | Predictor服务强 | 调度极快 | 可能过载事件循环 |

**如何选择：**
1. 查看predictor服务能力（QPS）
2. 监控事件循环负载（CPU使用率）
3. 测量调度完成时间
4. 调整到最佳平衡点

---

## 测试计划

### 单元测试
```python
async def test_background_scheduler_basic():
    """测试基本调度功能"""
    scheduler = BackgroundScheduler(...)
    scheduler.schedule_task_background(task_id="test-1", ...)

    # 等待后台完成
    await asyncio.sleep(0.1)

    # 验证任务已分配实例
    task = await task_registry.get("test-1")
    assert task.assigned_instance != ""

async def test_background_scheduler_concurrency_limit():
    """测试并发限制"""
    scheduler = BackgroundScheduler(max_concurrent_scheduling=2)

    # 提交10个任务
    for i in range(10):
        scheduler.schedule_task_background(f"task-{i}", ...)

    # 验证只有2个在处理
    stats = await scheduler.get_stats()
    assert stats["active_scheduling_tasks"] <= 2

async def test_background_scheduler_error_handling():
    """测试错误处理"""
    scheduler = BackgroundScheduler(...)

    # 提交任务（无可用实例）
    scheduler.schedule_task_background("test-fail", model_id="nonexistent", ...)

    await asyncio.sleep(0.1)

    # 验证任务标记为FAILED
    task = await task_registry.get("test-fail")
    assert task.status == TaskStatus.FAILED
    assert "No available instance" in task.error
```

### 性能测试
```python
async def test_500_workflow_scenario():
    """模拟500 workflow (3500 tasks)场景"""
    start = time.time()

    # 提交3500个任务
    for i in range(3500):
        response = await client.post("/task/submit", json={
            "task_id": f"task-{i}",
            "model_id": "test",
            ...
        })
        assert response.status_code == 200
        assert response.json()["task"]["status"] == "PENDING"

    api_time = time.time() - start

    # API应该在<20秒内完成（3500 × 5ms = 17.5秒）
    assert api_time < 20

    # 等待所有后台调度完成
    while True:
        stats = await background_scheduler.get_stats()
        if stats["active_scheduling_tasks"] == 0:
            break
        await asyncio.sleep(1)

    total_time = time.time() - start

    # 总时间应该显著低于当前架构（175秒）
    # 预期：~70秒（3500 ÷ 50 × 1秒）
    assert total_time < 120

    # 验证所有任务都分配了实例
    tasks = await task_registry.list_all()
    assigned = [t for t in tasks if t.assigned_instance]
    assert len(assigned) == 3500
```

---

## 回滚计划

如果新架构出现问题，快速回滚：

```python
# 方案1：特性开关
USE_BACKGROUND_SCHEDULING = os.getenv("USE_BACKGROUND_SCHEDULING", "true") == "true"

if USE_BACKGROUND_SCHEDULING:
    # 新架构：后台调度
    background_scheduler.schedule_task_background(...)
else:
    # 旧架构：同步调度
    schedule_result = await scheduling_strategy.schedule_task(...)
    task = await task_registry.create_task(..., assigned_instance=...)
```

```bash
# 回滚命令
export USE_BACKGROUND_SCHEDULING=false
systemctl restart scheduler
```

---

## 监控指标

### API级别
- `task_submit_latency_ms`: P50/P90/P99延迟
- `task_submit_qps`: 每秒任务提交数

### BackgroundScheduler级别
- `background_scheduling_active`: 当前活跃调度任务数
- `background_scheduling_queue_depth`: 等待槽位的任务数
- `background_scheduling_success_total`: 成功调度的任务总数
- `background_scheduling_failure_total`: 失败调度的任务总数
- `background_scheduling_duration_ms`: P50/P90/P99调度耗时

### 告警规则
```yaml
# 调度积压过多
- alert: BackgroundSchedulingBacklog
  expr: background_scheduling_queue_depth > 100
  for: 30s

# 调度失败率过高
- alert: HighSchedulingFailureRate
  expr: rate(background_scheduling_failure_total[5m]) > 0.1

# 调度耗时过长
- alert: SlowScheduling
  expr: background_scheduling_duration_ms_p90 > 5000  # 5秒
  for: 1m
```

---

## 总结

### 核心变更
1. **API立即返回**：5ms vs 20-70ms（**4-14x faster**）
2. **后台调度**：不阻塞事件循环，解决WebSocket超时
3. **并发控制**：Semaphore限制50个并发，可调整

### 主要收益
- ✅ WebSocket超时问题根本解决
- ✅ API响应速度提升4-14倍
- ✅ 更好的并发性和吞吐量
- ✅ 优雅的错误处理

### 需要注意
- ⚠️ 客户端需要适配（轮询或WebSocket订阅）
- ⚠️ 调度失败在后台发生，需要监控
- ⚠️ Predictor服务负载增加（50 QPS）

### 下一步
1. 完成API端点修改
2. 添加shutdown处理
3. 添加监控端点
4. 测试500 workflow场景
5. 部署到生产环境
