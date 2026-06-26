# Qwen3 Engine Socket Daemon 设计文档

> 目标：把 `qwen3-cli` 从一个“每次启动都要重新加载模型”的命令行工具，演进为一个**常驻、可多 session 复用、可通过 socket 被 Agent 调用**的推理守护进程。
> 
> 与 [ds3-kv-cache-service-plan.md](../ds3-kv-cache-service-plan.md) 的关系：Socket Daemon 负责**模型加载、tokenize、generate、session 生命周期**；KV Cache Service 负责**跨 session / 跨进程的 KV Cache 存储与前缀共享**。二者通过 UDS 协同工作。

---

## 1. 背景与动机

当前两种调用方式都有明显瓶颈：

| 方式 | 优点 | 缺点 |
|------|------|------|
| `SubprocessBackend`（每轮 spawn） | 简单、无状态 | 每轮都要加载 GGUF + 分配 KV Cache，首 token 延迟高 |
| `InteractiveBackend`（`-i`） | 只加载一次模型 | 仅支持单 session、 prompts 受 4096 字节行缓冲限制、输出靠超时解析 |

Socket Daemon 要解决的核心问题：

1. **模型加载一次，长期驻留内存**。
2. **真正的多 session**：每个 session 有独立的 KV Cache 上下文。
3. **任意长度 prompt**：不再受 CLI 行缓冲限制。
4. **结构化协议**：JSON / protobuf 请求响应，告别 stdout 解析。
5. **Agent 原生集成**：Go 客户端通过 socket 调用，采用 `SocketClient` + `SocketExecutor` 架构与 daemon 增量式交互。

---

## 2. 架构总览

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent (Go)                                   │
│                                                                      │
│   ┌─────────────────┐    ┌────────────────────────────────┐         │
│   │ SubprocessBackend│    │ SocketClient + SocketExecutor  │  ← 增量式 │
│   │ (fallback)       │    │ (recommended)                  │         │
│   └────────┬────────┘    └────────┬────────────────────────┘         │
│            │                      │                                  │
└────────────┼──────────────────────┼──────────────────────────────────┘
             │ spawn                │ Unix Domain Socket / TCP
             ▼                      ▼
┌─────────────────────┐   ┌──────────────────────────────────────────┐
│   qwen3-cli (each)  │   │     qwen3-engine-daemon ( resident )     │
│   load model every  │   │                                          │
│   time              │   │  ┌──────────────┐  ┌──────────────────┐  │
│                     │   │  │ Session      │  │ Generation       │  │
│                     │   │  │ Manager      │──│ Engine (C+Metal) │  │
│                     │   │  └──────────────┘  └──────────────────┘  │
└─────────────────────┘   │                                          │
                          │  ┌──────────────┐  ┌──────────────────┐  │
                          │  │ Tokenizer    │  │ KV Cache Client  │  │
                          │  │ (BPE)        │  │ (UDS to Rust)    │  │
                          │  └──────────────┘  └──────────────────┘  │
                          └────────────────────┬─────────────────────┘
                                               │
                          ┌────────────────────┼─────────────────────┐
                          │                    │                     │
                          ▼                    ▼                     ▼
                ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
                │ KV Cache Service│  │  Model GGUF     │  │ 持久化存储      │
                │ (Rust)          │  │  (mmap)         │  │ (session状态)   │
                └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 2.1 进程模型

- **单进程单模型**：一个 daemon 只加载一个 GGUF。要跑多个模型可启动多个 daemon，监听不同 socket。
- **多 session**：每个 session 持有独立的 `seq_len`、conversation tokens、采样参数。
- **主线程 + 线程池**：
  - 主线程：accept socket，派发请求。
  - 工作线程池：执行 generate（CPU/GPU 阻塞调用）。
  - 可选：后台线程负责 KV Cache 的异步 flush / 压缩。

---

## 3. 传输层与协议

### 3.1 默认传输

- **Unix Domain Socket (SOCK_STREAM)**，路径可配置。
  - 默认：`/tmp/qwen3-engine.sock`
  - 配置项：`--socket-path /path/to.sock`
- **可选 TCP**：`--listen 127.0.0.1:9333`，方便远程调试。
- 长连接：一个 Agent session 可复用一条 socket 连接，也可每次请求新建连接。

### 3.2 协议格式：NDJSON

采用 **Newline-Delimited JSON**，简单、可流式、易调试。

每条请求/响应都是一个 JSON object，以 `\n` 结尾。

```text
{ "id": "req-1", "method": "generate", "params": { ... } }\n
{ "id": "req-1", "result": { ... } }\n
{ "id": "req-2", "error": { "code": -32602, "message": "unknown session" } }\n
```

后续可升级为：

- **protobuf + length-prefix**：更高吞吐、更小体积。
- **HTTP/2**：如果需要跨语言、跨网络直接暴露。

### 3.3 基础 JSON-RPC 2.0 风格字段

```json
{
  "id": "uuid-or-sequence",
  "method": "generate",
  "params": { ... }
}
```

```json
{
  "id": "uuid-or-sequence",
  "result": { ... }
}
```

```json
{
  "id": "uuid-or-sequence",
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": { "detail": "..." }
  }
}
```

### 3.4 流式协议草图（Phase 2）

> Phase 1 不实现流式，仅保留协议草图供 Phase 2 参考。

`generate` 设置 `"stream": true` 时，daemon 在生成过程中多次发送 `partial`，最后用一条 `result` 收尾。所有消息共享同一个 `id`。

```json
{ "id": "2", "partial": { "text": "我会先", "tokens_generated": 3 } }
{ "id": "2", "partial": { "text": "读取相关", "tokens_generated": 6 } }
{ "id": "2", "result": { "text": "我会先读取相关文件...", "tokens_generated": 42, "current_tokens": 170, "max_tokens": 4096 } }
```

---

## 4. API 设计

### 4.1 `load_model`

启动时自动调用，也可在运行中切换模型（会清空所有 session）。

**Request**

```json
{
  "id": "0",
  "method": "load_model",
  "params": {
    "model_path": "/Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q4_K_M.gguf",
    "n_ctx": 4096,
    "gpu_layers": -1
  }
}
```

**Response**

```json
{
  "id": "0",
  "result": {
    "loaded": true,
    "model_name": "Qwen3-30B-A3B",
    "n_ctx": 4096,
    "vocab_size": 151936
  }
}
```

### 4.2 `create_session`

为一次新的用户对话分配 session。**system_prompt 和 tools 在这里一次性传入**，后续 `generate` 不再重复携带。

**Request**

```json
{
  "id": "1",
  "method": "create_session",
  "params": {
    "session_id": "sess-uuid-1234",
    "system_prompt": "You are qw3-agent, a local AI coding assistant powered by Qwen3.\n\nAvailable tools: ...",
    "temperature": 0.7,
    "max_tokens": 2048
  }
}
```

**Response**

```json
{
  "id": "1",
  "result": {
    "session_id": "sess-uuid-1234",
    "created_at": "2026-06-18T11:40:46Z"
  }
}
```

### 4.3 `generate`

核心接口。Agent 发送**当前轮用户输入**；daemon 自己维护该 session 的完整对话历史与 KV Cache，按与 Agent 完全一致的 chat template 编码 system + 多轮历史 + 当前 user 后生成。

daemon 内部不再调用 `ds3_engine_chat_format` 把整段对话包成单条 user 消息，而是直接构造标准 Qwen3 多轮 prompt：

```
<|im_start|>system
{system_prompt}[\n\n{context}]<|im_end|>
<|im_start|>user
{user1}<|im_end|>
<|im_start|>assistant
{assistant1}[\n<tool_call>...]</tool_call>]<|im_end|>
<|im_start|>tool\n[{id}] {name}: {result}<|im_end|>
...
<|im_start|>user
{current_user}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

然后 `ds3_engine_tokenize()` → `ds3_engine_generate_ex()`。多轮历史部分与 `agent.go:writeMessage()` 逐字节一致；最后的空 `<think>` 块用于引导模型输出最终答案。

- `user_prompt` 可为空字符串，表示“工具结果已追加，请继续生成”。
- `stream` 默认 `false`。

**Request**

```json
{
  "id": "2",
  "method": "generate",
  "params": {
    "session_id": "sess-uuid-1234",
    "user_prompt": "请帮我看看这个 bug",
    "context": "Relevant context from previous conversations:\n- ...",
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": false
  }
}
```

- `context`（可选）：本回合临时注入的上下文（如 memory 检索结果），daemon 会把它拼到 system prompt 后、user prompt 前，**但不写入 conversation 历史**。

**Response（非流式）**

```json
{
  "id": "2",
  "result": {
    "session_id": "sess-uuid-1234",
    "text": "我会先读取相关文件...",
    "tokens_generated": 42,
    "tokens_prompt": 128,
    "current_tokens": 170,
    "max_tokens": 4096,
    "duration_ms": 1250,
    "cached_prefix_len": 128,
    "new_cached_prefix_len": 170,
    "tool_calls": []
  }
}
```

> `current_tokens` / `max_tokens` 供 Agent 做 compact 决策，取代 Agent 本地的字符数估算。`current_tokens` **包含本轮 `context` 注入的 token 数**。

> **Thinking 块处理**：daemon 在 generation prompt 末尾追加官方 Qwen3 `enable_thinking=false` 空 `<think>` 块（`<|im_start|>assistant\n<think>\n\n</think>\n\n`），引导模型把推理放进 `<think>...</think>` 并在同一次生成中输出最终答案。`ds3_engine` 不再把 `</think>` 当作硬停止符，因此答案不会被截断；daemon 返回前会剥离 `<think>...</think>` 块，只把最终答案写入 `text` 和 conversation 历史。
>
> **Phase 1 KV Cache 限制**：`ds3_engine_generate_ex()` 每次调用都会重置 `seq_len` 并重新 prefill 完整 prompt，因此 daemon 当前主要省去的是每轮重新加载模型的时间，**并未实现跨轮 KV Cache 复用**。真正的前缀缓存需要 Phase 3 的 KV Cache Service。

> **TODO（Phase 1 可选）**：当前 Agent 从 `text` 中正则提取 `<tool_call>`。未来可在 `result` 中直接返回结构化 `tool_calls` 数组，减少文本解析。

### 4.4 `append_turn`（必需）

Agent 的工具调用循环必须用它把 assistant / tool_result / user 等角色消息追加进 session，**不触发生成**。这是维持 daemon 端 KV Cache 与 Agent 端历史一致的关键。

支持的角色：`assistant`、`tool_result`、`user`。

- `tool_result`：工具执行结果回填，最常见。
- `user`：用户历史消息重放（resume / replace_history 时）。
- `assistant`：**仅用于历史重放**（如从 JSONL 恢复 assistant 消息）。正常生成时，daemon 会自动把 assistant 输出追加到 conversation，无需手动调用 `append_turn(role=assistant)`。

```json
{
  "id": "3",
  "method": "append_turn",
  "params": {
    "session_id": "sess-uuid-1234",
    "role": "tool_result",
    "content": "read 结果：..."
  }
}
```

典型工具调用循环：

```text
generate(user_input)           → assistant 输出含 <tool_call>
Agent 解析并执行工具
append_turn(role=tool_result)  → 回填工具结果
generate(user_prompt="")       → daemon 继续生成，无新 user 输入
```

### 4.5 `replace_history`

用于 `Compact()` 场景：Agent 把旧历史替换为摘要 + 最近消息后，调用此方法让 daemon 用新的历史重新编码，可选择是否保留现有 KV Cache。

**模式 A：KV 失效重建（默认）**

```json
{
  "id": "4",
  "method": "replace_history",
  "params": {
    "session_id": "sess-uuid-1234",
    "preserve_kv_cache": false,
    "messages": [
      { "role": "user", "content": "[摘要] 之前我们在讨论..." },
      { "role": "assistant", "content": "明白了。" },
      { "role": "user", "content": "现在的问题是..." },
      {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          { "id": "tc_1", "name": "read", "arguments": "{\"path\": \"main.go\"}" }
        ]
      },
      {
        "role": "tool_result",
        "content": "package main\n...",
        "tool_call_id": "tc_1",
        "tool_name": "read"
      }
    ]
  }
}
```

`messages` 中每条消息支持以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `role` | string | `user` / `assistant` / `tool_result` |
| `content` | string | 文本内容 |
| `tool_calls` | array | assistant 消息中的 tool call 列表（可选） |
| `tool_call_id` | string | tool_result 对应的 tool call ID（可选） |
| `tool_name` | string | tool_result 对应的 tool 名称（可选） |

**模式 B：保留 KV Cache（实验性）**

如果摘要长度与旧历史前段 token 完全一致，可以设置 `preserve_kv_cache: true`，daemon 只截断 conversation tokens 到摘要结尾。该模式需要 Agent 保证摘要 token 对齐，初期可作为优化项。

### 4.6 `reset_session`

清空某个 session 的 KV Cache 和历史，保留 system prompt。

```json
{
  "id": "5",
  "method": "reset_session",
  "params": { "session_id": "sess-uuid-1234" }
}
```

### 4.7 `close_session`

释放 session 资源。

```json
{
  "id": "6",
  "method": "close_session",
  "params": { "session_id": "sess-uuid-1234" }
}
```

### 4.8 `health` / `stats`

```json
{
  "id": "7",
  "method": "health",
  "params": {}
}
```

```json
{
  "id": "7",
  "result": {
    "status": "ok",
    "model_loaded": true,
    "active_sessions": 3,
    "total_tokens_generated": 10240,
    "uptime_seconds": 3600
  }
}
```

---

## 5. Session 与 KV Cache 管理

### 5.1 Session 数据结构（daemon 内部）

```c
typedef struct {
    char session_id[64];
    ds3_engine_t *engine;          // 指向全局 engine
    ds3_kv_cache_t kv_cache;       // 该 session 当前 KV cache 状态
    int *conversation_tokens;      // 已编码的完整对话 token
    int n_conversation;
    int n_ctx;                     // 该 session 允许的最大上下文
    float temperature;
    int max_tokens;
    char *system_prompt;
    uint64_t last_active_at;
} ds3_session_t;
```

### 5.2 对话历史维护（Agent-Daemon 契约）

这是整个设计的核心边界：

- **Daemon 是“历史 + KV Cache”的权威来源**；Agent 只保留一份用于 UI / 持久化 / memory 注入的副本。
- **Agent 不再调用 `buildPrompt()` 生成完整 chat template**。在 SocketExecutor 模式下，`buildPrompt` 被禁用或重构为仅用于 memory 摘要注入。
- Agent 通过增量消息与 daemon 同步：
  - 用户新输入 → `generate(user_prompt)`
  - assistant 输出 → daemon 返回，Agent 追加到 `state.Messages`
  - tool result → `append_turn(role=tool_result, content=...)`
  - compact 后 → `replace_history(messages)`

daemon 内部流程：

1. `create_session` 时缓存 `system_prompt`、temperature、max_tokens。
2. `generate(user_prompt)`：按 `agent.go:writeMessage()` 格式构造完整多轮 prompt，并在 assistant prefix 后追加空 `<think>` 块，然后 `ds3_engine_tokenize()` → `ds3_engine_generate_ex()`。
3. `append_turn(role, content)`：仅把消息追加到 daemon 内存历史，不触发生成。
4. `generate("")`：在已追加 tool result 的基础上继续生成。
5. assistant 输出 tokens 解码为 text 并剥离 `<think>` 块后，作为 assistant turn 追加到 conversation。
6. **`generate("")` 的边界**：daemon 在已有 tool_result 后生成时，遇到 EOS 或下一个 `<tool_call>` 起始标记即停止；`<think>`/`</think>`  reasoning 块不会被当作终止符；输出文本作为 assistant turn 追加后，下一轮 `generate("")` 不会重复编码它，因为该 assistant turn 已经是 conversation 历史的一部分。

> **Phase 1 实现说明**：由于 `ds3_engine_generate_ex()` 每次都会重置 KV state，第 2 步实际上每次都会重新编码完整 prompt。Session 隔离和模型常驻已生效，但**增量 KV Cache 追加与前缀缓存尚未实现**。

### 5.3 Context Window 管理

daemon 在每次 `generate` 响应中返回：

```json
{
  "current_tokens": 170,
  "max_tokens": 4096
}
```

Agent 用这两个值做 compact 决策，**不再使用本地的 `autoCompactChars` 字符数估算**。决策策略示例：

- `current_tokens > max_tokens * 0.75` → 触发 `Compact()`
- `current_tokens >= max_tokens` → daemon 返回 `-32003 Context Overflow`，Agent 必须 compact 或 reset

daemon 也可以在响应中附加 warning：

```json
{
  "warning": "context_window_almost_full"
}
```

### 5.4 Compact 流程与 `replace_history` 的时机

Agent 的 `Compact()` 需要通过 LLM 生成历史摘要。在 SocketExecutor 模式下，**推荐使用独立的 fallback backend 执行摘要生成**（方案 B），避免污染当前 session 的 KV Cache 和历史。

**推荐方案 B：fallback SubprocessBackend / 独立 daemon session**

```text
Agent 检测到 current_tokens > threshold
  ↓
用 SubprocessBackend 调用 summarizeMessages(history)
  ↓
拿到摘要后，Agent 更新 state.Messages
  ↓
调用 replace_history(session_id, preserve_kv_cache=false, messages=compact后消息)
  ↓
daemon 用新历史重建 KV Cache
```

优点：compact 过程与当前 session 完全隔离，实现简单，风险小。

**备选方案 A：复用当前 daemon session**

把 `summarizeMessages` 当作一次普通 `generate`，但**不将这次请求追加到 conversation 历史**。这需要在 daemon 中增加一个内部标记或单独的 `summarize` 方法，复杂度较高，Phase 1 不建议采用。

### 5.5 与 KV Cache Service 的交互

daemon 内部可预留 hook：

- `generate` 前：向 KV Cache Service `lookup(prefix_hash)`，命中则把 KV 拷贝到 session cache，跳过 prefill。
- `generate` 后：向 KV Cache Service `store(prefix_hash, kv_ptr, seq_len)`，供后续 session 共享。

在 KV Cache Service 完成前，daemon 先**本地管理每个 session 的 KV cache**。

---

## 6. 并发、内存与性能

### 6.1 Phase 1 并发策略

**所有 generate 串行执行**（无论是否同 session）。这样：

- 避免 Metal command queue 竞争。
- 简化 KV Cache 状态管理。
- 先验证正确性，再用 benchmark 决定是否需要多 worker。

```text
[ accept ] → [ single worker queue ] → [ generate ]
```

### 6.2 Phase 2 并发策略（待 benchmark 验证后）

- **同一 session 的 generate 串行**。
- **不同 session 可并行**，但 worker 数不建议超过性能核心数。
- 线程池大小默认 `1`，可配置 `--workers`。

### 6.3 内存预算

每个 session 的 KV Cache 占用可估算：

```text
KV(bytes) = n_ctx × n_layer × n_head_kv × head_dim × 2(K+V) × 2(fp16)
```

以 Qwen3-30B-A3B、4K context 为例：

```text
4096 × 48 × 4 × 128 × 4 ≈ 384 MB / session
```

默认限制（可配）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--max-sessions` | 4 | 同时在线 session 上限 |
| `--session-idle-timeout` | 300s | 空闲 session 多久后回收 |
| `--max-context` | 4096 | 每个 session 最大 context |

超出 `max-sessions` 时，按 LRU 淘汰并关闭最老的 idle session。

### 6.4 超时

- 连接空闲超时：300 秒（可配）。
- 单次 generate 超时：10 分钟（可配）。
- prompt encode 超时：30 秒。

---

## 7. Go 客户端（SocketClient + SocketExecutor）

实际实现没有采用 `llm.SocketBackend`（prompt-based 的 `llm.Backend` 接口与 daemon 的增量式协议不匹配），而是拆成两层：

- `llm.SocketClient`：负责 NDJSON 编码、socket 连接、重连恢复、各 RPC 方法封装。
- `agent.SocketExecutor`：实现 `turnExecutor` 接口，对 Agent 核心循环隐藏 daemon 的增量 API。

```go
// llm/socket.go
type SocketClient struct {
    socketPath string
    // ...
}

func (c *SocketClient) CreateSession(...) error
func (c *SocketClient) Generate(...) (*SocketResult, error)
func (c *SocketClient) AppendTurn(...) error
func (c *SocketClient) ReplaceHistory(...) error
func (c *SocketClient) CloseSession(...) error
```

```go
// agent/socket_executor.go
type SocketExecutor struct {
    client  *llm.SocketClient
    created map[string]bool
    // ...
}

func (e *SocketExecutor) Generate(ctx, sessionID, userPrompt, systemPrompt string, tools []Tool, opts GenerateOpts) (GenerateResponse, error)
```

> **架构演进说明**：设计初期曾设想 `llm.SocketBackend` 实现 `llm.Backend` 接口，但 `llm.Backend.Generate(ctx, prompt, opts)` 需要调用方拼接完整 prompt，这与 daemon 的增量式 `generate(user_prompt)` 相矛盾。最终采用 `SocketClient` + `SocketExecutor` 的组合，`SocketExecutor` 内部维护 `created` 标记并增量同步消息，Agent 侧无需再拼接完整 prompt。

Agent 侧实际集成（`pie/main.go`）：

```go
if cfg.SocketPath != "" {
    if err := ensureDaemonRunning(ctx, cfg); err != nil {
        return fmt.Errorf("failed to start daemon: %w", err)
    }
    client := llm.NewSocketClient(cfg.SocketPath)
    defer client.Close()

    fallback := llm.NewSubprocessBackend(cfg.QW3CLIPath, cfg.ModelPath)
    executor := agent.NewSocketExecutor(client, fallback)
    ag = agent.NewWithExecutor(executor, fallback, registry, state).WithMemory(memStore)
} else {
    ag = agent.New(backend, registry, state).WithMemory(memStore)
}
```

### 7.1 与 qw3-agent 的集成路径

| 现有组件 | SocketDaemon 模式下的变化 |
|---|---|
| `SubprocessBackend` | 保留作为 fallback |
| `InteractiveBackend` | 可废弃（被 SocketDaemon 取代） |
| `MetricsBackend` | 可包裹 `SubprocessBackend`；socket 模式下 metrics 建议在 `SocketExecutor` 层统计 |
| `buildPrompt()` | **重构**：不再构建完整 prompt；改为向 daemon 发送增量 `user_prompt` + `append_turn` |
| `formatMemoryNotes()` | **改为 `context` 字段**：每轮 memory 检索结果作为 `generate` 的临时上下文注入，不进入 conversation 历史 |
| `auto-compact` | **适配**：用 daemon 返回的 `current_tokens` / `max_tokens` 做决策，不再用字符数估算 |
| `Compact()` | **联动**：compact 后调用 `replace_history`，默认重建 KV Cache；摘要 LLM 建议用 fallback backend |
| `session.Store` | 继续用——Agent 侧的 JSONL 持久化不变 |
| `config.Config` | 新增 `SocketPath` / `QW3DaemonPath` 字段 |

### 7.2 Resume 冷启动恢复

Agent 使用 `--resume` 从 JSONL 恢复 `state.Messages` 后，必须让 daemon 侧历史与之一致：

```text
Agent 启动
  ↓
create_session(session_id, system_prompt)
  ↓
replace_history(messages=state.Messages)  // 重放 JSONL 恢复的历史
  ↓
daemon 重建 KV Cache，Agent 与 daemon 状态对齐
  ↓
进入正常 generate / append_turn 循环
```

- `session_id` 应从恢复的 `state.SessionID` 中取得，保持一致。
- 如果 daemon 中已存在同名 session（如 daemon 未重启），应先 `close_session` 再重新 `create_session` + `replace_history`。

### 7.3 连接生命周期与恢复

`SocketClient` 默认维护一条长连接。需要处理以下情况：

1. **daemon 重启**：连接断开，所有 session 丢失。
2. **网络/UDS 异常**：`write` / `read` 返回 error。
3. **版本不兼容**：连接后握手发现协议版本不匹配。

恢复策略：

```text
检测到连接断开
  ↓
尝试重新连接（指数退避，最多 3 次）
  ↓
重新 create_session(session_id, system_prompt)
  ↓
replace_history(messages=state.Messages)  // 用 Agent 本地历史重建
  ↓
重试失败的 generate / append_turn 请求
```

设计要点：

- `SocketClient` 每次写前检查连接活性；失败时自动重连。
- 重连后必须能根据 `opts.SessionID` 重建 session。
- 对于 `generate` 这类非幂等请求，重试需要 Agent 层配合（如重新触发一次 generate）。

### 7.4 建议：抽象 `turnExecutor` 接口

`RunOnceWithLimit` 的改动量最大。建议在 `agent/` 下抽象一个接口：

```go
type turnExecutor interface {
    Generate(ctx context.Context, userInput string, opts llm.GenerateOpts) (*TurnResult, error)
    AppendAssistant(ctx context.Context, content string, toolCalls []ToolCall) error
    AppendToolResult(ctx context.Context, result ToolResult) error
    Compact(ctx context.Context, keep int) (bool, error)
}
```

- `promptBasedExecutor`：当前实现，自己构建完整 prompt 调用 `llm.Backend`。
- `socketExecutor`：新实现，通过增量 API 与 daemon 交互。

`Agent` 持有 `turnExecutor`，两种模式各自实现，避免 `RunOnceWithLimit` 里堆满 `if socketMode` 分支。

---

## 8. 部署与启动

### 8.1 命令行

```bash
./qwen3-engine-daemon \
  --model /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --socket /tmp/qwen3-engine.sock \
  --n-ctx 4096 \
  --workers 1 \
  --quiet
```

### 8.2 与 Agent 一起启动

推荐由 Agent 在启动时检查 daemon 是否存在，不存在则自动拉起：

```go
socketPath := cfg.SocketPath
if !daemonRunning(socketPath) {
    logFile, _ := os.OpenFile(filepath.Join(cfg.DataDir, "daemon.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
    cmd := exec.Command(cfg.QW3DaemonPath, "--model", cfg.ModelPath, "--socket", socketPath)
    cmd.Stdout = logFile
    cmd.Stderr = logFile
    cmd.Start()
    waitForSocket(socketPath, 30*time.Second)
}
```

### 8.3 权限

- UDS 文件权限设为 `0600`，只允许同一用户访问。
- TCP 模式建议仅监听 `127.0.0.1`。

---

## 9. 错误码

| 错误码 | 含义 | 场景 |
|--------|------|------|
| 0 | OK | 成功 |
| -32600 | Invalid Request | JSON 格式错误 |
| -32601 | Method Not Found | 未知方法 |
| -32602 | Invalid Params | 缺少 session_id 等 |
| -32001 | Model Not Loaded | daemon 未加载模型 |
| -32002 | Session Not Found | session_id 不存在 |
| -32003 | Context Overflow | 超出 n_ctx |
| -32004 | Generation Timeout | 生成超时 |
| -32005 | Engine Error | C engine 内部错误 |
| -32006 | Request Too Large | 单条 NDJSON 请求超过 `--max-request-size` |

---

## 10. 与 `-i` 模式的对比

| 能力 | `qwen3-cli -i` | Socket Daemon |
|------|----------------|---------------|
| 模型只加载一次 | ✅ | ✅ |
| 多 session | ❌ | ✅ |
| 任意长度 prompt | ❌（4096 行缓冲） | ✅ |
| 结构化响应 | ❌ | ✅ |
| KV Cache 共享 | ❌ | ✅（配合 KV Service） |
| 并发请求 | ❌ | ✅（不同 session） |
| 可远程调用 | ❌ | ✅（TCP） |
| 实现/维护成本 | 低 | 中 |

---

## 11. 实现计划

### Phase 1：最小可运行 daemon（核心已跑通，仍有收尾缺口）

- [x] 在 `tools/` 下新建 `qwen3-engine-daemon.c`。
- [x] 实现 UDS server（**单线程串行处理所有请求**，先保证正确性）。
- [x] 实现 `load_model`、`create_session`、`generate`（支持空 `user_prompt` 续生成）、`append_turn`、`replace_history`、`close_session`。
- [x] daemon 端维护 session conversation tokens 与 KV Cache。
- [x] `generate` 响应返回 `tokens_prompt` / `tokens_generated` / `current_tokens` / `max_tokens`。
- [x] Go 端实现 `llm.SocketClient` + `agent.SocketExecutor`（原设计的 `llm.SocketBackend` 因与增量式协议不匹配，由 Executor 架构取代）。
- [x] Agent 端重构：socket 模式下禁用完整 prompt 构建，改为增量消息同步；`Compact()` 联动 `replace_history`。
- [x] 与 Agent 集成，支持 `--socket-path` / `--qw3-daemon-path` 启动。

> 收尾缺口见 `docs/qwen3-engine-daemon-backlog.md`：历史同步边界、Context Overflow 错误码、断线恢复、超时控制、缓存命中字段透传等。

### Phase 2：多 session 与并发（Phase 1 稳定后）

- [ ] per-session 请求队列，避免同 session 竞态。
- [ ] 线程池与超时控制。
- [ ] session 空闲回收（LRU）。
- [ ] health / stats API。

### Phase 3：KV Cache Service 对接（✅ 已完成）

> 对应实现见 [ds3-kv-cache-service-plan.md](../ds3-kv-cache-service-plan.md)。

- [x] daemon 通过 UDS protobuf 调用 Rust KV Cache Service 的 `Config` / `Heartbeat` / `Lookup` / `Allocate` / `Store` / `Stats`。
- [x] prefix caching：相同 system prompt / 文件内容在多个 session 间自动命中；跨 daemon 时提升为 Global 前缀共享。
- [x] 跨进程 zero-copy：每个 daemon 拥有独立 L1 SHM arena，其它 daemon 按 `Lookup` 返回的 `shm_name` + offset 以 `O_RDONLY` / `PROT_READ` 只读映射 owner daemon 的 arena。
- [x] daemon 通过 `--daemon-id <id>` 与 service 交互；service 用 `--max-daemons` 限制并发 arena 数量。
- [x] service 在 daemon lease 超时/断连后自动迁移其 Global block 到活跃 daemon，并清理 stale SHM arena。

### Phase 4：工程化（Phase 2 稳定后可启动）

#### 4.1 工程化基础（推荐先做）

- [ ] TCP 监听可选（方便远程调试）。
- [ ] 日志、指标、graceful shutdown。
- [ ] systemd / launchd plist 示例。

#### 4.2 高级特性（按需延后）

- [ ] protobuf / gRPC 协议可选（替代 NDJSON，提升吞吐）。
- [ ] 多模型并发：一个 daemon 管理多个 model，或模型热切换。
- [ ] TLS / 鉴权（TCP 模式下才需要）。

---

## 12. 风险与注意事项

1. **Agent-Daemon 历史同步**：最大风险。必须严格遵循增量契约（`generate` + `append_turn` + `replace_history`），任何一方漏步都会导致 KV Cache 与 Agent 历史 diverge。
   - **chat template 格式化一致性（Phase 1 已修复）**：daemon 直接构造与 `agent.go:writeMessage()` 逐字节一致的多轮 prompt，再调用 `ds3_engine_tokenize()`，因此 `replace_history` 重放的历史与 `generate` 自身编码的历史完全等价。
2. **Compact 后 KV 失效**：`replace_history` 默认重建 KV Cache，会暂时失去前缀缓存优势。`preserve_kv_cache: true` 模式需谨慎验证。
3. **Metal 资源竞争**：多个 session 同时生成会共享 GPU，Phase 1 串行可避免；Phase 2 需 benchmark 确定 worker 数。
4. **内存占用**：service 按 `--max-daemons` 均分总内存预算，每个 daemon 的 SHM arena 大小固定；多 session 共享同一 system prompt 时内存占用接近“一份共享前缀 + 每 session 增量”。
5. **Session 状态持久化**：daemon 崩溃后 session 丢失；重要对话需要 Agent 自己保存/恢复。service 侧 prefix index 通过 WAL + snapshot 持久化，重启后同一 `daemon-id` 重连可继续命中历史 Global 前缀。
6. **与 KV Cache Service 的耦合**：daemon 已不再本地管理跨 session KV，全部走 Rust service；断连后 service 心跳超时（默认 30s，可用 `--heartbeat-timeout-secs` 调整）会清理孤儿 session 并迁移 Global block。
7. **跨平台**：UDS 在 Linux/macOS 可用；Windows 需要 TCP fallback。

---

## 13. 使用示例

启动 KV Cache service（tiered 模式，1 GiB 总预算，最多 2 个 daemon）：

```bash
./ds3-kv-cache-svc --tiered \
    -s /tmp/ds3-kv-cache.sock \
    -p /tmp/ds3-kv-cache-dir \
    -m 1073741824 \
    --max-daemons 2
```

启动 daemon 并接入 service：

```bash
./qwen3-engine-daemon \
    -m /path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
    -s /tmp/qwen3-engine.sock \
    -c 2048 \
    -k service \
    -K /tmp/ds3-kv-cache.sock \
    -i daemon1
```

跨 daemon 共享前缀时，第二个 daemon 使用 `-i daemon2`，其 `lookup` 会返回 daemon1 的 SHM arena 名，daemon2 以只读方式映射该 arena。

## 14. 下一步建议

1. 跑通 `tests/test_two_daemon_cross_read.py` 与 `tests/test_milestone3_acceptance.py` 验证跨 daemon 共享与生命周期。
2. 使用 `tests/test_performance_baseline.py` 采集 10 session / 128 token 输出场景下的延迟与内存基线。
3. 根据基线结果调整 `--max-daemons`、内存预算与 Global promotion 阈值。

