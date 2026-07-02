# qwen3-engine-daemon 剩余工作清单

> 生成时间：2026-06-26
> 依据文档：`docs/qwen3-engine-socket-daemon-design.md`
> 当前状态：P0 与 P1 全部完成；Phase 3（KV Cache Service 对接）已完成；Phase 2 核心并发与 2.5 流式 generate 已完成；Phase 4 剩余项待推进。

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
| 1.4 | daemon 端 generate / prefill 超时 | 已完成 | `tools/qwen3-engine-daemon.c`、`src/ds3_engine.c`、`pie/llm/socket.go` | 新增 CLI 参数 `-t`/`-T`/`-R`（generate/idle/request timeout，默认 600/300/60 秒）。daemon 对 accept 后的 socket 设置 `SO_RCVTIMEO`/`SO_SNDTIMEO`；`ds3_engine_generate_ex` 内通过 `clock_gettime(CLOCK_MONOTONIC)` 检查 deadline，超时返回 `-2`，daemon 映射为 JSON-RPC `-32004 Generation Timeout`。Go 侧新增 `IsGenerationTimeout`/`NewGenerationTimeoutError`。 |
| 1.5 | 实现 `stats` API | 已完成 | `tools/qwen3-engine-daemon.c:handle_stats`、`pie/llm/socket.go` | 新增 `stats` RPC，返回 `status/model_name/model_loaded/n_ctx/vocab_size/gpu_layers/active_sessions/total_tokens_generated/total_tokens_prompt/uptime_seconds`。`handle_generate` 累加 prompt tokens 到 `total_prompt_tokens`。Go 侧新增 `SocketClient.Stats` 与 `SocketStats`。 |
| 1.6 | 限制 NDJSON 单条请求大小 | 已完成 | `tools/qwen3-engine-daemon.c:read_line` | 新增 CLI 参数 `-r BYTES`（默认 1MB）。`read_line` 超过限制返回 `-2`，`handle_connection` 发送 `-32006 Request Too Large` 并关闭连接。 |
| 1.7 | `generate` 响应透传缓存命中字段 | 已完成 | `tools/qwen3-engine-daemon.c:handle_generate`、`src/ds3_engine.c`、`pie/llm/socket.go` | `ds3_engine_t` 新增 `last_cached_prefix_len`/`last_new_cached_prefix_len`；`generate_ex` 在 KV lookup 后和新 KV write 后记录；daemon 响应包含 `cached_prefix_len`/`new_cached_prefix_len`。`SocketResult` 增加对应字段。 |
| 1.8 | 集成测试：Go 与真实 daemon 端到端 | 已完成 | `pie/llm/socket_integration_test.go`、`tools/qwen3-engine-daemon.c:create_unix_socket` | 新增 `//go:build integration` 集成测试，覆盖 daemon 生命周期（health/stats/create/generate/append/continue/replace/close）、请求大小限制、graceful shutdown。修复了 macOS 上无法运行的问题：为 daemon 子进程设置 `cmd.Dir`（Metal 需要相对路径读 `.metal` 源码）、改用 `/tmp` 下的短 socket 路径以符合 `sun_path` 104 字节限制、daemon 在 socket 路径过长时返回 `ENAMETOOLONG` 而非静默截断。运行需设置 `QW3_MODEL`（和可选 `QW3_DAEMON`）。 |
| 1.9 | 结构化 `tool_calls` 返回 | 已完成 | `tools/qwen3-engine-daemon.c:handle_generate` | daemon 现在解析 `<tool_call>` 内的 JSON object/array 并作为结构化数组返回；非 JSON（如简化格式）仍返回 `[]`，由 Agent 侧正则兜底。`SocketResult.ToolCalls` 字段保留原始 JSON。 |
| 1.10 | `replace_history` 的 `preserve_kv_cache: true` 优化路径 | 已完成 | `tools/qwen3-engine-daemon.c:handle_replace_history` | `handle_replace_history` 已读取 `preserve_kv_cache`：为 `false` 时调用 `ds3_kv_cache_reset_session` 重建 KV；为 `true` 时只替换消息历史，保留现有 KV 前缀缓存。 |
| 1.11 | resume 时显式 close + create session | 已完成 | `pie/main.go` | `--resume` 路径现在先 `CloseSession`（失败忽略），再 `CreateSession` + `ReplaceHistory`，避免 daemon 内部隐式关闭带来的不一致。 |

### Phase 4 工程化（高优先级）

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 4.1 | graceful shutdown | 已完成 | `tools/qwen3-engine-daemon.c` | 收到 SIGINT/SIGTERM 后停止 accept，广播 shutdown 信号，join 所有 worker 线程和 reaper 线程。in-flight generate 由 engine 内部 `generate_timeout` deadline 兜底返回；daemon 侧不再做额外的 force-cleanup，避免损坏 KV cache。 |
| 4.2 | launchd / systemd 示例 | 已完成 | `examples/launchd/`、`examples/systemd/`、`docs/deployment.md` | 新增 macOS plist、Linux systemd service 和部署文档。 |

---

## 🟢 P2 — 中低优先级 / 细节

### Phase 1 细节

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 1.12 | `load_model` 运行时参数完善 | 部分 | `tools/qwen3-engine-daemon.c:handle_load_model` | `load_model` RPC 存在，但 CLI 没有 `--gpu_layers` 参数；响应中的 `vocab_size` 使用宏 `DS3_N_VOCAB`，不是运行时读取。 |

### Phase 2：多 session 与并发

当前 daemon 已改造为**多线程**：主线程只负责 `accept()`，连接派发到 worker 线程池；同 session 请求通过 per-session mutex 串行，所有 `generate` 通过全局 `engine_mutex` 串行。

> ⚠️ **Phase 2 并发收益约束**：Metal command queue 在当前实现下是全局共享的。即使有线程池，多个 session 的 `generate` 也无法在 GPU 上真正并行——GPU 端仍是串行执行。Phase 2 的主要收益是：
> - 一个 session 在 `generate` 时，另一个 session 的 tokenize/encode 可以并行；
> - 非生成请求（`create_session`、`append_turn`、`replace_history` 等）不被阻塞。

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 2.1 | per-session 请求队列 | 已完成 | `tools/qwen3-engine-daemon.c` | 每个 `session_t` 新增 `pthread_mutex_t mutex`；所有针对 session 的请求（`generate`/`append_turn`/`replace_history`/`reset_session`/`close_session`）先通过 `session_find_and_lock` 或 `session_close` 获取该 session 的互斥锁，确保同一 session 内请求串行执行。 |
| 2.2 | 线程池 + `--workers` 参数 | 已完成 | `tools/qwen3-engine-daemon.c` | 主线程只负责 `accept()`，连接通过 `enqueue_work` 派发到固定大小的 worker 线程池；新增 `-W N` / `--workers`（默认 4）。 |
| 2.3 | 同 session 串行、跨 session 并行 | 已完成 | `tools/qwen3-engine-daemon.c` | 同 session 请求由 per-session mutex 串行；跨 session 的非生成请求可并行。所有 `generate` 调用额外持有全局 `engine_mutex`，因为底层 `ds3_engine_t` / Metal 不是线程安全。 |
| 2.4 | session 空闲超时回收 + `--max-sessions` | 已完成 | `tools/qwen3-engine-daemon.c:session_t` | 新增 `-S SEC` / `--session-timeout`（默认 0，禁用）与 `-M N` / `--max-sessions`（默认 0，不限）。后台 reaper 线程每 5 秒扫描一次，使用 `pthread_mutex_trylock` 关闭空闲超时的 session；`create_session` 在达到上限时尝试淘汰最老的空闲 session，失败则返回 `-32007 Session Limit Reached`。 |
| 2.5 | 流式 generate（`stream: true`） | 已完成 | `tools/qwen3-engine-daemon.c`、协议 §3.4 | daemon 侧 `generate` 新增可选 `stream` 参数；`stream:true` 时通过 `ds3_engine_generate_ex` 回调逐 token 发送 NDJSON 中间 chunk，最终发送含完整 metadata 与 `done:true` 的结束行，再追加 assistant message。Go 侧 `SocketClient` 新增 `GenerateStream`，使用独立连接读取 NDJSON 行并返回 `<-chan SocketStreamResult`，上下文取消时关闭连接。集成测试 `TestIntegrationGenerateStream` 已添加。 |
| 2.6 | 连接空闲超时控制 | 已完成 | `tools/qwen3-engine-daemon.c` | 新增 `-I SEC` / `--conn-idle-timeout`（默认 300）。`handle_connection` 在每次读取 NDJSON 前先用 `poll()` 等待数据，超时则关闭连接。 |

### Phase 4 工程化（剩余）

| # | 工作项 | 当前状态 | 位置 / 文件 | 详细说明 |
|---|--------|----------|-------------|----------|
| 4.3 | TCP 监听可选（`--listen`） | ❌ | `tools/qwen3-engine-daemon.c` | 当前只有 UDS；远程调试需要 SSH forward，不方便。 |
| 4.4 | 结构化日志 / 日志级别 | ❌ | 全局 | 当前只有 `fprintf(stderr, ...)` 和 `--quiet` 开关，缺少 DEBUG/INFO/WARN/ERROR 分级。 |
| 4.5 | Prometheus `/metrics` 端点 | ❌ | `tools/qwen3-engine-daemon.c` | `stats` RPC 已实现（P1 1.5），但尚未提供 Prometheus 格式 `/metrics` HTTP 端点。 |

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

- ~~Go 侧缺少 `SocketClient` 与真实 daemon 的集成测试~~ → `pie/llm/socket_integration_test.go`（`//go:build integration`）已覆盖 health/stats/create/generate/append/continue/replace/close、大请求边界、graceful shutdown。
- 缺少 daemon 断线重连 / session 恢复测试 → 已由 `pie/agent/socket_executor_test.go` 的 mock 测试覆盖，真实 daemon 场景可在集成测试中补充。
- 缺少 socket 模式下 Agent `RunOnce` 的端到端测试 → 可在集成测试中通过 `qw3-agent` 主程序补充。
- 缺少 `Context Overflow`、超时、大请求边界测试 → Context Overflow 与大请求边界已覆盖；真实模型下的 generate 超时触发测试待补充。

---

## 建议推进顺序

1. **P0（3 项）**：已完成。历史同步、Context Overflow、断线恢复已闭环。
2. **P1（10 项）**：已完成。包含超时、stats、请求大小限制、缓存命中字段透传、集成测试、结构化 tool_calls、`preserve_kv_cache`、resume 显式 close/create、graceful shutdown、launchd/systemd 示例。
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
