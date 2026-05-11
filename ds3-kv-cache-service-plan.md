# DS3 KV Cache Service (Phase 4)

Rust 实现的高性能 KV Cache 管理服务，借鉴 [LMCache](https://github.com/lmcache/lmcache) 的分层存储思想，针对 Apple Silicon 统一内存架构深度优化。服务于 Qwen3-235B-A22B 的多 session / 长上下文推理场景。

## 1. 设计目标

| 指标 | 目标值 |
|------|--------|
| 单次 lookup 延迟 | < 1ms (命中 hot cache) |
| Store/Load 吞吐 | > 10 GB/s (利用统一内存零拷贝) |
| 最大管理容量 | 128K context × 94 layers ≈ 12 GB |
| 多 session 共享 | 相同前缀 chunk 自动去重复用 |
| 进程模型 | 独立服务进程，DS3 通过 UDS 调用 |

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      DS3 Engine (C + Metal)                   │
│                                                               │
│   Forward Pass                                                │
│     ├─ Layer 0: Attention → store K/V → next layer           │
│     ├─ Layer 1: lookup K/V (hit!) → skip prefill             │
│     └─ ...                                                    │
│                                                               │
│   ┌──────────────────────────────────────────────────┐       │
│   │  UDS Client (binary protocol, zero-copy mmap)     │       │
│   └──────────────────────┬───────────────────────────┘       │
└──────────────────────────┼───────────────────────────────────┘
                           │ Unix Domain Socket
┌──────────────────────────┼───────────────────────────────────┐
│                          ▼                                     │
│            DS3 KV Cache Service (Rust)                         │
│                                                               │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│   │ Request     │  │ Chunk Index  │  │ Eviction Policy │    │
│   │ Dispatcher  │──│ (Hash Table) │──│ (LRU / ARC)     │    │
│   └─────────────┘  └──────────────┘  └─────────────────┘    │
│                                                               │
│   ┌───────────────────────────────────────────────────────┐  │
│   │              Storage Tiers                              │  │
│   │                                                         │  │
│   │  Tier 0: Shared mmap region (统一内存, zero-copy)       │  │
│   │  Tier 1: File-backed mmap (SSD, ~3 GB/s read)         │  │
│   │  Tier 2: Compressed archive (zstd, cold storage)       │  │
│   └───────────────────────────────────────────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 3. 核心概念

### 3.1 Chunk 分块策略

借鉴 LMCache 的 chunk-based hashing：

- **Chunk Size**: 256 tokens (可配置)
- **Chunk Key**: `SHA-256(token_ids[chunk_start..chunk_end])` 的前 128 bits
- **Per-layer 存储**: 每个 chunk 的每一层独立存储，支持逐层流水线

```
Chunk = {
    key:      [u8; 16],           // 128-bit hash of token sequence
    layer:    u32,                // layer index (0..93)
    k_data:   &[f16],            // [N_HEAD_KV × HEAD_DIM × chunk_size]
    v_data:   &[f16],            // [N_HEAD_KV × HEAD_DIM × chunk_size]
}
```

对于 Qwen3-235B (N_HEAD_KV=4, HEAD_DIM=128):
- 单 chunk 单层: 4 × 128 × 256 × 2 bytes × 2 (K+V) = **512 KB**
- 单 chunk 全层: 512 KB × 94 = **47 MB**
- 512 token prefix: 2 chunks × 47 MB = **94 MB**

### 3.2 Prefix Sharing（前缀共享）

相同 system prompt 的多个 session 自动共享 KV cache：

```
Session A: [system prompt tokens...][user A tokens...]
Session B: [system prompt tokens...][user B tokens...]
                     ↑
            Same chunk hash → single copy in cache
```

节省比例取决于 system prompt 长度，典型场景可节省 30-60% 存储。

### 3.3 Apple Silicon 零拷贝优势

| 传统 GPU (CUDA) | Apple Silicon (Metal) |
|---|---|
| GPU HBM ↔ CPU RAM: PCIe 拷贝 (32 GB/s) | 统一内存：同一物理地址，无需拷贝 |
| 需要 pinned memory + async DMA | mmap 共享区域，改 offset 即可 |
| LMCache 50%+ 代码在做 D2H/H2D | 我们只需管理元数据索引 |

实现方式：
1. Rust 服务 mmap 一块大区域作为 KV cache pool
2. DS3 Engine mmap 同一文件（或 shared memory fd）
3. Store: Engine 写入 mmap region → 通知 Rust 服务更新索引
4. Load: Rust 服务返回 offset → Engine 直接从 mmap 读取，**零拷贝**

## 4. 通信协议

### 4.1 传输层

- **Unix Domain Socket** (SOCK_STREAM)
- Socket path: `/tmp/ds3-kv-cache.sock` (可配置)
- 连接模式: 长连接 + 请求-响应

### 4.2 二进制协议格式

```
Request:
┌────────┬────────┬──────────┬─────────────────┐
│ cmd(1) │ flags(1)│ len(4)   │ payload(len)    │
└────────┴────────┴──────────┴─────────────────┘

Response:
┌──────────┬──────────┬─────────────────┐
│ status(1)│ len(4)   │ payload(len)    │
└──────────┴──────────┴─────────────────┘
```

### 4.3 命令集

| Cmd ID | 名称 | 描述 | Payload |
|--------|------|------|---------|
| 0x01 | STORE | 存储 chunk KV 数据 | chunk_key + layer + mmap_offset + size |
| 0x02 | LOAD | 加载 chunk KV 数据 | chunk_key + layer → 返回 mmap_offset |
| 0x03 | LOOKUP | 查询 chunk 是否存在 | chunk_key + layer_mask → 返回 hit bitmap |
| 0x04 | BATCH_LOOKUP | 批量前缀查询 | n_chunks + chunk_keys[] → hit counts |
| 0x05 | EVICT | 主动淘汰指定 chunk | chunk_key (all layers) |
| 0x06 | SAVE | 持久化到磁盘 | session_id → 确认 |
| 0x07 | RESTORE | 从磁盘恢复 | session_id → 确认 |
| 0x08 | STATS | 获取服务状态 | → capacity, used, hit_rate |
| 0x09 | CLEAR | 清空所有缓存 | → 确认 |

### 4.4 典型调用流程

**Prefill 阶段（新请求到达）：**
```
1. Engine tokenize → 得到 token_ids[]
2. Engine 计算每个 chunk 的 hash key
3. BATCH_LOOKUP(chunk_keys[]) → 返回哪些 chunk 已缓存
4. 对已缓存的 chunk:
     LOAD(key, layer) → 得到 mmap_offset → 直接读取 KV
5. 对未缓存的 chunk:
     正常 forward → compute KV → 写入 mmap region
     STORE(key, layer, offset, size) → 通知服务更新索引
```

**Decode 阶段（逐 token 生成）：**
```
正常 decode，KV cache 在 Engine 侧管理。
当 context 超过阈值时，可选择 EVICT 最旧的 chunk 释放空间。
```

## 5. Rust 实现模块

```
ds3-kv-cache/
├── Cargo.toml
├── src/
│   ├── main.rs              # 服务入口，UDS listener
│   ├── config.rs            # 配置加载 (TOML)
│   ├── protocol.rs          # 二进制协议编解码
│   ├── dispatcher.rs        # 请求路由 + 并发控制
│   ├── chunk_index.rs       # Chunk 元数据索引 (DashMap)
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── mmap_pool.rs     # 共享 mmap 内存池管理
│   │   ├── ssd_tier.rs      # SSD file-backed 存储
│   │   └── compressed.rs    # zstd 压缩冷存储
│   ├── eviction/
│   │   ├── mod.rs
│   │   ├── lru.rs           # LRU 淘汰策略
│   │   └── arc.rs           # ARC 自适应策略 (可选)
│   ├── metrics.rs           # 命中率、延迟统计
│   └── session.rs           # Session 级 save/restore
├── tests/
│   ├── integration_test.rs  # UDS 端到端测试
│   └── bench_throughput.rs  # 吞吐量基准测试
└── ds3_kv_client.h          # C header: DS3 Engine 调用的客户端 API
```

### 5.1 核心依赖

| Crate | 用途 |
|-------|------|
| `tokio` | 异步运行时 (UDS accept + I/O) |
| `memmap2` | 共享 mmap 管理 |
| `dashmap` | 并发安全 HashMap (chunk index) |
| `bytes` | 零拷贝 buffer 操作 |
| `zstd` | 冷数据压缩 |
| `sha2` | Chunk key 计算 (也可在 C 侧完成) |
| `tracing` | 结构化日志 |

### 5.2 C 客户端 API (`ds3_kv_client.h`)

```c
/* DS3 KV Cache Client — 嵌入 Engine 进程 */

typedef struct ds3_kv_client ds3_kv_client_t;

/* 连接到 KV Cache Service */
ds3_kv_client_t *ds3_kv_connect(const char *socket_path);
void             ds3_kv_disconnect(ds3_kv_client_t *client);

/* 批量查询前缀命中 */
int ds3_kv_batch_lookup(ds3_kv_client_t *client,
                        const uint8_t (*chunk_keys)[16],
                        int n_chunks,
                        bool *hits_out);

/* 加载单 chunk 单层 KV (返回 mmap 中的 offset，Engine 直接读取) */
int64_t ds3_kv_load(ds3_kv_client_t *client,
                    const uint8_t chunk_key[16],
                    int layer);

/* 通知服务：Engine 已将 KV 写入 mmap offset */
bool ds3_kv_store(ds3_kv_client_t *client,
                  const uint8_t chunk_key[16],
                  int layer,
                  int64_t mmap_offset,
                  size_t size);

/* Session 持久化 */
bool ds3_kv_save_session(ds3_kv_client_t *client, const char *session_id);
bool ds3_kv_restore_session(ds3_kv_client_t *client, const char *session_id);

/* 服务状态 */
typedef struct {
    size_t capacity_bytes;
    size_t used_bytes;
    float  hit_rate;
    int    n_chunks_cached;
} ds3_kv_stats_t;

bool ds3_kv_stats(ds3_kv_client_t *client, ds3_kv_stats_t *out);
```

## 6. 存储管理细节

### 6.1 mmap 内存池

```
┌────────────────────────────────────────────────────────┐
│           Shared mmap region (e.g. 16 GB)               │
│                                                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────────────┐  │
│  │Chunk0│ │Chunk1│ │Chunk2│ │      Free Space       │  │
│  │L0-L93│ │L0-L93│ │L0-L93│ │                       │  │
│  └──────┘ └──────┘ └──────┘ └──────────────────────┘  │
│                                                          │
└────────────────────────────────────────────────────────┘
  ↑                                                     ↑
  Engine mmap (read/write)              Service mmap (管理元数据)
```

- 分配策略: Bump allocator + free list (回收被 evict 的 chunk)
- 对齐: 4096 bytes (page-aligned，避免跨页开销)
- 文件路径: `/tmp/ds3-kv-pool.mmap` (tmpfs 或 SSD)

### 6.2 淘汰决策

```rust
enum EvictionPolicy {
    LRU,                    // 最近最少使用
    ARC { ghost_size: usize }, // 自适应替换缓存
}

// 淘汰候选选择
fn select_eviction_candidates(&self, needed_bytes: usize) -> Vec<ChunkKey> {
    // 1. 跳过 pinned chunks (正在被 Engine 使用的)
    // 2. 优先淘汰 layer 数不完整的 chunk (部分写入失败的)
    // 3. 按策略排序剩余 candidates
    // 4. 累积释放空间直到 >= needed_bytes
}
```

### 6.3 SSD Tier

当 mmap pool 满时，冷 chunk 降级到 SSD：
- 格式: 每个 session 一个目录，内含 chunk 文件
- 命名: `{session_id}/{chunk_hash_hex}.bin`
- 加载: mmap 回 pool → 更新索引 → 返回 offset

## 7. 性能估算

### 7.1 延迟分析 (Apple Silicon M4 Max / M3 Ultra)

| 操作 | 延迟 |
|------|------|
| LOOKUP (hash table hit) | ~0.5 us |
| BATCH_LOOKUP (10 chunks) | ~5 us |
| STORE (更新索引，数据已在 mmap) | ~2 us |
| LOAD (返回 offset，数据已在 mmap) | ~1 us |
| UDS round-trip overhead | ~10 us |
| **End-to-end LOAD (hot)** | **~12 us** |
| SSD tier load (512KB chunk) | ~170 us (3 GB/s SSD) |

### 7.2 对推理性能的影响

Decode 阶段（无 cache 交互）: **0% overhead**

Prefill 阶段（有 cache 命中时）:
- 每 256 tokens 节省约 50ms 计算 (235B 模型)
- Cache lookup 开销: ~12 us per chunk
- **净收益: 节省 99.97% 的 prefill 时间（命中时）**

### 7.3 内存使用 (Qwen3-235B)

| Context Length | KV Cache Size | 说明 |
|---|---|---|
| 4K tokens (16 chunks) | 752 MB | 单 session 短对话 |
| 32K tokens (128 chunks) | 6 GB | 单 session 长文档 |
| 128K tokens (512 chunks) | 24 GB | 最大上下文 |
| 4 sessions × 4K shared prefix | 752 MB (shared) + delta | 前缀共享 |

## 8. 实现路线

### Phase 4.1: 基础框架 (1 周)

- [ ] Cargo 项目初始化
- [ ] UDS listener + binary protocol 编解码
- [ ] mmap pool 分配器 (bump + free list)
- [ ] Chunk index (DashMap<[u8;16], ChunkMeta>)
- [ ] STORE / LOAD / LOOKUP 基本命令
- [ ] C 客户端库 (ds3_kv_client.h + .c)

### Phase 4.2: 集成 DS3 Engine (1 周)

- [ ] Engine 侧 chunk hash 计算
- [ ] Prefill 路径: batch_lookup → conditional forward
- [ ] Store 路径: forward 完成后通知 service
- [ ] 端到端测试: 相同 prompt 第二次 prefill 跳过

### Phase 4.3: 淘汰 + 持久化 (1 周)

- [ ] LRU 淘汰实现
- [ ] SSD tier: 降级 + 恢复
- [ ] Session save/restore (对话持久化)
- [ ] STATS 命令 + 监控输出

### Phase 4.4: 优化 + 压力测试 (1 周)

- [ ] ARC 自适应策略 (可选)
- [ ] zstd 压缩冷存储
- [ ] 并发压力测试 (多 session 并行)
- [ ] 内存泄漏检测 (valgrind / Instruments)
- [ ] 性能 benchmark 文档

## 9. 与 LMCache 的关键差异

| 维度 | LMCache | DS3 KV Cache Service |
|------|---------|---------------------|
| 语言 | Python + CUDA C | Rust |
| GPU 交互 | CUDA kernel (D2H/H2D 拷贝) | 统一内存 mmap (零拷贝) |
| 耦合度 | 深度嵌入 vLLM 内部 | 独立进程，UDS API |
| 存储层级 | GPU → CPU → SSD → Remote | mmap pool → SSD → compressed |
| 复杂度 | ~15K 行 Python | 预估 ~3K 行 Rust + ~500 行 C client |
| 适用场景 | 多 GPU 集群 | 单机 Apple Silicon |
| Layerwise pipeline | 通过 Python generator | Engine 侧 Metal CommandBuffer 调度 |

## 10. 配置文件示例

```toml
# ds3-kv-cache.toml

[server]
socket_path = "/tmp/ds3-kv-cache.sock"
max_connections = 4

[storage]
mmap_path = "/tmp/ds3-kv-pool.mmap"
mmap_size_gb = 16          # 共享内存池大小
ssd_path = "./kv-cache-ssd"
ssd_max_gb = 64

[policy]
eviction = "lru"           # "lru" | "arc"
chunk_size_tokens = 256

[model]
n_layers = 94
n_head_kv = 4
head_dim = 128
dtype = "f16"              # KV 存储精度

[logging]
level = "info"
```

## 11. 未来扩展

- **Speculative Decoding 集成**: Draft model 的 KV cache 也可复用
- **跨机器共享**: 基于 TCP 的 remote tier (多台 Mac 组成集群)
- **KV 量化**: FP16 → INT8 KV cache，容量翻倍，精度损失 < 0.1% PPL
- **Prefix tree 索引**: 替代 flat hash，支持最长前缀匹配加速
