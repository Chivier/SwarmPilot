# WebSocket Keepalive修复文档

## 问题描述

客户端报告WebSocket连接意外断开错误：
```
Error receiving message: sent 1011 (internal error) keepalive ping timeout; no close frame received
```

**根本原因：**
- WebSocket连接在长时间无消息交互时被视为死连接
- 调度暂停功能可能导致任务长时间处于PENDING状态
- 缺少keepalive机制维持连接活性

## 解决方案

### 1. 添加Ping/Pong消息类型

在 `scheduler/src/model.py` 中添加：

```python
class WSMessageType(str, Enum):
    """Enumeration of WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    RESULT = "result"
    ERROR = "error"
    ACK = "ack"
    PING = "ping"      # 新增
    PONG = "pong"      # 新增
```

```python
class WSPingMessage(BaseModel):
    """WebSocket ping message for keepalive."""
    type: WSMessageType = WSMessageType.PING
    timestamp: Optional[float] = None


class WSPongMessage(BaseModel):
    """WebSocket pong message for keepalive response."""
    type: WSMessageType = WSMessageType.PONG
    timestamp: Optional[float] = None
```

### 2. 实现服务器端Keepalive机制

**修改位置：** `scheduler/src/api.py:1055` - WebSocket endpoint

**关键改动：**

#### a) 创建后台Keepalive任务
```python
PING_INTERVAL = 20  # 每20秒发送一次ping

async def send_keepalive():
    """Send periodic ping messages to keep connection alive."""
    try:
        while True:
            await asyncio.sleep(PING_INTERVAL)
            ping_msg = WSPingMessage(timestamp=asyncio.get_event_loop().time())
            await websocket.send_json(ping_msg.model_dump())
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"Keepalive task error: {e}")

# 启动keepalive任务
keepalive_task = asyncio.create_task(send_keepalive())
```

#### b) 处理PING/PONG消息
```python
if message_type == WSMessageType.PONG:
    # 客户端响应ping
    logger.debug("Received pong from WebSocket client")
    continue

elif message_type == WSMessageType.PING:
    # 客户端发送ping，回复pong
    pong_msg = WSPongMessage(timestamp=asyncio.get_event_loop().time())
    await websocket.send_json(pong_msg.model_dump())
    continue
```

#### c) 清理Keepalive任务
```python
finally:
    # 取消keepalive任务
    keepalive_task.cancel()
    try:
        await keepalive_task
    except asyncio.CancelledError:
        pass
```

## 工作原理

### 消息流程

```
服务器                                客户端
  |                                     |
  |--- PING (每20秒) ----------------->|
  |                                     |
  |<-- PONG (可选响应) -----------------|
  |                                     |
  |<-- PING (客户端可主动发送) ---------|
  |                                     |
  |--- PONG (服务器响应) -------------->|
  |                                     |
```

### 双向Keepalive支持

1. **服务器主动Keepalive**
   - 每20秒自动发送PING消息
   - 保持连接活性，防止超时

2. **客户端主动Keepalive（可选）**
   - 客户端可发送PING消息
   - 服务器自动响应PONG

3. **无强制PONG要求**
   - 客户端无需响应服务器的PING
   - 服务器持续发送PING即可维持连接

## 配置参数

### PING_INTERVAL
- **默认值**: 20秒
- **说明**: 服务器发送PING消息的间隔
- **建议**:
  - 生产环境：20-30秒
  - 开发环境：可调整为10秒以便调试

### 为什么选择20秒？

常见WebSocket实现的超时时间：
- **websockets库**: 默认20秒ping超时
- **大多数浏览器**: 30-60秒空闲超时
- **云负载均衡器**: 60-120秒空闲超时

选择20秒确保在各种环境下都能维持连接。

## 客户端兼容性

### 自动兼容
客户端无需任何修改即可受益：
- 服务器的PING消息会自动维持连接活性
- 客户端可以忽略PING消息（不响应PONG也没问题）

### 推荐实现
客户端最好实现PING/PONG处理：

```python
# Python客户端示例
async def handle_message(message):
    msg_type = message.get("type")

    if msg_type == "ping":
        # 响应pong（可选）
        await websocket.send(json.dumps({
            "type": "pong",
            "timestamp": time.time()
        }))

    elif msg_type == "pong":
        # 收到服务器的pong响应
        pass

    # ... 其他消息处理
```

```javascript
// JavaScript客户端示例
websocket.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'ping') {
        // 响应pong（可选）
        websocket.send(JSON.stringify({
            type: 'pong',
            timestamp: Date.now() / 1000
        }));
    } else if (message.type === 'pong') {
        // 收到服务器的pong响应
        console.debug('Received pong from server');
    }

    // ... 其他消息处理
};
```

## 测试验证

### 1. 单元测试
所有现有WebSocket测试继续通过：
```bash
cd scheduler
uv run pytest tests/test_websocket_manager.py -v
# Result: 24 passed
```

### 2. 手动测试场景

#### 测试1：长时间空闲连接
```python
# 客户端保持连接60秒不发送任何消息
# 预期：连接保持活跃，不断开
```

#### 测试2：调度暂停期间
```python
# 1. 订阅任务
# 2. 触发调度暂停（提交大量任务）
# 3. 等待任务完成（可能需要数分钟）
# 预期：连接保持活跃，最终收到任务结果
```

#### 测试3：网络抖动
```python
# 模拟短暂网络延迟（1-2秒）
# 预期：keepalive帮助快速检测连接状态
```

## 日志输出

启用DEBUG日志可以看到keepalive活动：

```
[DEBUG] Sent keepalive ping to WebSocket client
[DEBUG] Received pong from WebSocket client
[DEBUG] Sent pong response to WebSocket client
[DEBUG] WebSocket keepalive task cancelled
```

生产环境建议使用INFO级别，避免日志过多。

## 兼容性说明

### 向后兼容
- ✅ 完全向后兼容现有客户端
- ✅ 不影响现有消息类型处理
- ✅ 不破坏现有API行为

### 性能影响
- **CPU**: 忽略不计（每20秒一次简单JSON发送）
- **内存**: 每个连接增加约1KB（keepalive任务）
- **网络**: 每连接每20秒约100字节

### 适用场景
所有WebSocket连接场景：
- ✅ 短期订阅（几秒钟）
- ✅ 长期订阅（几分钟到几小时）
- ✅ 调度暂停期间
- ✅ 网络不稳定环境

## 故障排查

### 问题1：仍然出现timeout
**可能原因：**
- 中间代理/负载均衡器超时设置太短
- 客户端WebSocket库超时设置太短

**解决方案：**
- 减小PING_INTERVAL（改为10秒）
- 检查代理/负载均衡器配置
- 升级客户端WebSocket库

### 问题2：Keepalive任务未启动
**检查点：**
```python
# 在startup_event中添加日志
logger.info("WebSocket endpoint registered with keepalive support")
```

### 问题3：大量PING消息
**正常现象：**
- 每个WebSocket连接每20秒发送一次PING
- 100个并发连接 = 5 PING/秒
- 这是正常且轻量的开销

## 未来改进

### 可配置Interval
将PING_INTERVAL移到配置文件：
```python
config.websocket.ping_interval = 20
```

### 连接健康监控
添加指标：
- 活跃WebSocket连接数
- Keepalive失败次数
- 平均连接时长

### 自适应Interval
根据网络条件动态调整ping间隔。

## 参考资料

- [RFC 6455 - WebSocket Protocol](https://tools.ietf.org/html/rfc6455)
- [FastAPI WebSocket文档](https://fastapi.tiangolo.com/advanced/websockets/)
- [websockets库文档](https://websockets.readthedocs.io/)
