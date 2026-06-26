# qwen3-engine-daemon 剩余工作清单

> 生成时间：2026-06-26
> 依据文档：`docs/qwen3-engine-socket-daemon-design.md`
> 当前状态：Phase 3（KV Cache Service 对接）已完成；Phase 1 主体已跑通，仍有收尾缺口；Phase 2 / Phase 4 基本未开始。

---

## 说明

本文档按**优先级**组织，不再严格区分 Phase。每个条目仍标注其来源 Phase，方便回溯设计文档。

---

## 🔴 P0 — 必须优先完成（3 项）

| # | 工作项 | 来源 Phase | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|------------|----------|-------------|----------|
| 1.1 | 修复 Agent ↔ daemon 历史同步边界 | Phase 1 | 已完成 | `pie/agent/agent.go`、`pie/agent/socket_executor.go`、`pie/agent/prompt_executor.go` | 移除了泄漏实现细节的 `Executor.AppendUserMessage()` 接口：PromptExecutor 自己在 Generate 内追加/回滚 user message；SocketExecutor 在 daemon 成功后追加。Agent 不再预追加，parse-error retry 也不会让 socket 模式提前把 fake user 写进 state，避免失败时 diverge。 |
| 1.2 | 实现 `-32003 Context Overflow` 错误码 | Phase 1 | 已完成 | `tools/qwen3-engine-daemon.c:generate_for_session`、`handle_generate`、`pie/agent/agent.go`、`pie/llm/socket.go` | daemon 侧已返回 `-32003`；Go 侧新增 `IsContextOverflow`/`NewContextOverflowError`，Agent `RunOnceWithLimit` 识别到该错误后自动触发一次 `Compact()` 并重试，否则才返回错误。 |
| 1.3 | `SocketClient` 断线后自动恢复 session | Phase 1 | 已完成 | `pie/llm/socket.go`、`pie/agent/socket_executor.go` | `SocketClient` 新增 `ErrTransport`、`IsSessionNotFound`；`SocketExecutor` 对幂等操作（create_session/append_turn/replace_history）自动重连并重放历史，对非幂等的 `generate` 不再自动重试，而是把连接/session 状态重置后返回 `ErrTransport`，由调用方决定是否重新发起。`SocketExecutor` 内部状态已加 `sync.Mutex` 保护。 |

---

## 🟡 P1 — 高优先级（10 项）

### Phase 1 完善

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 1.4 | daemon 端 generate / prefill 超时 | 缺失 | `tools/qwen3-engine-daemon.c`、`pie/llm/socket.go` | 设计 §6.4 要求：连接空闲超时 300s、单次 generate 超时 10min、prompt encode 超时 30s。当前 Go 端只有默认 10min socket deadline，daemon 侧无任何超时。 |
| 1.5 | 实现 `stats` API | 缺失 | `tools/qwen3-engine-daemon.c:dispatch_request` | 设计文档 §4.8 列出 `health` 和 `stats`，当前只实现了 `health`，且字段很少（缺 `model_name`、`vocab_size`、`gpu_layers` 等）。 |
| 1.6 | 限制 NDJSON 单条请求大小 | 缺失 | `tools/qwen3-engine-daemon.c:read_line` | `read_line` 未限制单行长度，异常长请求可能导致无界内存增长，存在 DoS 风险。 |
| 1.7 | `generate` 响应透传缓存命中字段 | 缺失 | `tools/qwen3-engine-daemon.c:handle_generate` | 当前响应只返回 `tokens_prompt/tokens_generated/current_tokens/max_tokens/tool_calls`，`cached_prefix_len` / `new_cached_prefix_len` 始终为 0。KV provider 内部能拿到缓存长度，需要透传到 JSON-RPC 响应。 observability 属性，不阻断 correctness，因此放在 P1。 |
| 1.8 | 集成测试：Go 与真实 daemon 端到端 | 缺失 | `pie/llm/socket_test.go`、`pie/agent/socket_executor_test.go` | 当前测试使用 mock daemon；缺少启动真实 `qwen3-engine-daemon` 的集成测试，覆盖 create_session、generate、append_turn、replace_history、close_session 全链路。应在 P0 修完后立刻补充，否则 P0 的修复缺乏自动化验证。 |
| 1.9 | 结构化 `tool_calls` 返回 | 缺失 | `tools/qwen3-engine-daemon.c:handle_generate` | 当前固定返回 `"tool_calls":[]`，Agent 仍靠正则从 `text` 提取 `<tool_call>`。设计 §4.3 TODO 建议直接返回结构化数组。当前正则可用，优先级不高。 |
| 1.10 | `replace_history` 的 `preserve_kv_cache: true` 优化路径 | 缺失 | `tools/qwen3-engine-daemon.c:session_replace_history` | 当前直接忽略 `preserve_kv` 并总是重建 KV。文档 §4.5 模式 B 标记为实验性，但代码里未实现优化路径。 |
| 1.11 | resume 时显式 close + create session | 部分 | `pie/main.go:268-283` | 当前直接 `CreateSession`，daemon 内部会关闭旧 session，但 Go 端没有显式处理；如果 create 失败会留下不一致状态。 |

### Phase 4 工程化（高优先级）

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 4.1 | graceful shutdown | 缺失 | `tools/qwen3-engine-daemon.c:signal_handler`、`main` | 收到 SIGINT/SIGTERM 后直接退出，不等待 in-flight generate 完成。launchd KeepAlive 重启是常态，每次重启都会触发，影响可用性。 |
| 4.2 | launchd / systemd 示例 | 缺失 | 仓库根目录 / `docs/` | 没有 service 或 plist 文件示例。单独做 graceful shutdown 意义有限：如果没有 plist/service 拉起，就没有机会触发 graceful shutdown。建议两项配对完成。 |

---

## 🟢 P2 — 中低优先级 / 细节

### Phase 1 细节

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 1.12 | `load_model` 运行时参数完善 | 部分 | `tools/qwen3-engine-daemon.c:handle_load_model` | `load_model` RPC 存在，但 CLI 没有 `--gpu_layers` 参数；响应中的 `vocab_size` 使用宏 `DS3_N_VOCAB`，不是运行时读取。 |

### Phase 2：多 session 与并发

当前 daemon 为**全局单线程串行**：`accept()` 后直接同步 `handle_connection()`，一个长 `generate` 会阻塞所有其他 session。

> ⚠️ **Phase 2 并发收益约束**：Metal command queue 在当前实现下是全局共享的。即使有线程池，多个 session 的 `generate` 也无法在 GPU 上真正并行——GPU 端仍是串行执行。Phase 2 的主要收益是：
> - 一个 session 在 `generate` 时，另一个 session 的 tokenize/encode 可以并行；
> - 非生成请求（`create_session`、`append_turn`、`replace_history` 等）不被阻塞。

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 2.1 | per-session 请求队列 | ❌ | `tools/qwen3-engine-daemon.c` | 避免同 session 内多个请求竞态，确保同一 session 的 generate 串行。 |
| 2.2 | 线程池 + `--workers` 参数 | ❌ | `tools/qwen3-engine-daemon.c` | 文档 §6.2 / §8.1 提到可按性能核心数配置 worker 数量。 |
| 2.3 | 同 session 串行、跨 session 并行 | ❌ | `tools/qwen3-engine-daemon.c` | 当前所有请求全局串行，无法满足设计目标。 |
| 2.4 | session 空闲超时回收 + `--max-sessions` | ❌ | `tools/qwen3-engine-daemon.c:session_t` | 已记录 `last_active`，但没有定时检查或 LRU 淘汰；文档 §6.3 默认上限 4、空闲超时 300s。 |
| 2.5 | 流式 generate（`stream: true`） | ❌ | `tools/qwen3-engine-daemon.c`、协议 §3.4 | 文档 §3.4 列为 Phase 2，当前固定非流式。 |
| 2.6 | 连接空闲超时控制 | ❌ | `tools/qwen3-engine-daemon.c` | 与 Phase 1 超时控制相关，长连接无请求时应自动关闭。 |

### Phase 4 工程化（剩余）

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 4.3 | TCP 监听可选（`--listen`） | ❌ | `tools/qwen3-engine-daemon.c` | 当前只有 UDS；远程调试需要 SSH forward，不方便。 |
| 4.4 | 结构化日志 / 日志级别 | ❌ | 全局 | 当前只有 `fprintf(stderr, ...)` 和 `--quiet` 开关，缺少 DEBUG/INFO/WARN/ERROR 分级。 |
| 4.5 | 指标输出（Prometheus / stats） | ❌ | `tools/qwen3-engine-daemon.c` | 没有 `/metrics` 或 `stats` RPC；无法监控延迟、吞吐量、session 数。 |

### Phase 4 高级特性（按需延后）

| # | 工作项 | 当前状态 | 说明 |
|---|--------|----------|------|
| 4.6 | protobuf / gRPC 可选协议 | ❌ | 当前只有 NDJSON。 |
| 4.7 | 多模型并发 / 热切换 | ❌ | 一个 daemon 只加载一个 GGUF。 |
| 4.8 | TLS / 鉴权 | ❌ | TCP 未实现，更无 TLS。 |
| 4.9 | 配置热重载 | ❌ | 非文档明确要求，但属工程化常见需求。 |

### 已取代项

| # | 工作项 | 当前状态 | 说明 |
|---|--------|----------|------|
| ~~D1~~ | ~~实现 `llm.SocketBackend`~~ | ~~已取代~~ | 设计文档 §7 原本设想 `SocketBackend` 实现 `llm.Backend` 接口，但实际架构演变为 `Agent → Executor → SocketExecutor → SocketClient → daemon`，更贴合 daemon 的增量式协议。`llm.Backend.Generate(prompt, opts)` 是 prompt-based 的，强行包装需要在 Go 侧维护完整对话历史来拼接 prompt，等于绕回老路。因此**不再实现 `SocketBackend`**；如需 metrics，可在 Executor 层包装。设计文档对应位置已标注为“由 Executor 接口取代”。 |

---

## ✅ 已确认实现 / by design

| 工作项 | 状态 | 说明 |
|--------|------|------|
| `generate` 的 `context` 字段（memory 注入） | 已实现 | `pie/llm/socket.go:Generate` 接收 `context` 参数并下发；daemon `build_chat_prompt` 将其拼接到 system prompt 之后、user prompt 之前；`SocketExecutor.Generate` 把 `memoryNotes` 作为 `context` 传入。该字段不进入 conversation 历史。 |
| `<think>` 块剥离 | 已实现 | `tools/qwen3-engine-daemon.c:strip_think_blocks` 在返回前剥离 `<think>...</think>`；未闭合的 think 块从 `<think>` 开始丢弃。 |
| `create_session` 的 `max_tokens` 生效 | 已实现 | daemon `session_create` 保存 `max_tokens`；`generate_for_session` 中 `max_gen = max_tokens > 0 ? max_tokens : s->max_tokens`，限制单次生成长度。 |
| Socket 文件清理（防止 `EADDRINUSE`） | 已实现 | daemon 启动时 `unlink(path)` 旧 socket，退出时再次 `unlink(socket_path)`。 |
| Socket 文件权限 0600 | 已实现 | `create_unix_socket` 中 `umask(0077)` + `chmod(path, 0600)`。 |
| Session 持久化 | by design | daemon 崩溃后 session 丢失；重要对话由 Agent 自己保存/恢复。service 侧 prefix index 通过 WAL + snapshot 持久化，重启后同一 `daemon-id` 重连可继续命中历史 Global 前缀。 |
| KV provider 运行期切换 | by design | CLI `-k` 在启动时选择 provider（`local` / `service` / `fallback` / `none`），运行期不切换。 |

---

## 测试覆盖缺口

- Go 侧缺少 `SocketClient` 与真实 daemon 的集成测试。
- 缺少 daemon 断线重连 / session 恢复测试。
- 缺少 socket 模式下 Agent `RunOnce` 的端到端测试。
- 缺少 `Context Overflow`、超时、大请求边界测试。

---

## 建议推进顺序

1. **P0（3 项）**：历史同步、Context Overflow、断线恢复。把 Phase 1 correctness 真正闭环。
2. **P1（10 项）**：
   - Phase 1 完善：超时、stats、请求大小限制、缓存命中字段透传、集成测试、结构化 tool_calls、`preserve_kv_cache`、resume 显式 close/create。
   - Phase 4 工程化：graceful shutdown + launchd/systemd 示例（配对完成）。
3. **P2**：`load_model` 参数完善、Phase 2 并发、剩余 Phase 4 工程化、高级特性按需延后。

---

## 关键文件清单

| 文件 | 作用 |
|------|------|
| `docs/qwen3-engine-socket-daemon-design.md` | 设计文档 |
| `docs/qwen3-engine-daemon-backlog.md` | 本清单 |
| `tools/qwen3-engine-daemon.c` | daemon 主程序 |
| `src/ds3_engine.c` | 推理引擎，含 KV provider 集成 |
| `src/ds3.h` | 公共 API 与模型常量 |
| `src/ds3_kv_cache.h` | KV Cache provider 接口 |
| `pie/llm/socket.go` | Go SocketClient |
| `pie/agent/socket_executor.go` | SocketExecutor |
| `pie/agent/executor.go` | Executor 接口 + `MessageToSocket` |
| `pie/agent/agent.go` | Agent 主循环、`shouldCompact` |
| `pie/main.go` | daemon 启动集成、resume 历史同步 |
| `pie/config/config.go` | `SocketPath` / `QW3DaemonPath` 配置 |
