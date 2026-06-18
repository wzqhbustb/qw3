# Qwen3 专用 Metal 推理引擎：实现方案

> 项目代号：qwen3-engine  
> 目标平台：Apple Silicon Mac (48GB ~ 128GB RAM)  
> 核心假设验证状态：🟢 已通过（非对称量化 PPL 劣化 +8.46% < 10%）  
> 文档版本：v1.0  
> 创建时间：2026-05-12

---

## 一、项目概述

### 1.1 一句话定位

为 **Qwen3-30B-A3B** 和 **Qwen3-235B-A22B** 从零构建的 Metal-only 专用推理引擎。核心创新是**非对称量化**——只压缩 MoE Expert 权重，Attention 和输入输出层保持高精度。

> **为什么不从 ds4 fork？** ds4 的 ~16K 行单文件中，超过 **60% 是 MLA 相关代码**（低秩投影、压缩 KV、indexed attention、compressor），这些对 Qwen3 的标准 GQA 架构**完全无用**。从零按模块化结构写，需要时逐段搬 ds4 的代码（mmap、Metal buffer 管理、session sync），比删改 16K 行更高效。

### 1.2 价值主张

| 场景 | 现有方案 | 本引擎 |
|------|---------|--------|
| 30B on 48GB Mac | llama.cpp Q4_K_M (17GB) | **Asym Q2+Q8 (10GB)**，更快加载 |
| 235B on 128GB Mac | **无法运行**（全 Q4 需 132GB+） | **Asym Q2+Q8 (~80GB)**，可运行 |

### 1.3 核心指标

| 指标 | 目标 |
|------|------|
| 30B Q4_K_M PPL 劣化 | < 5%（vs FP16） |
| 30B Asym Q2+Q8 PPL 劣化 | < 10%（已验证：+8.46%） |
| 235B Asym Q2+Q8 PPL 劣化 | < 15%（待验证） |
| 30B 推理速度 | > 20 tokens/s（M3 Max） |
| 235B 推理速度 | > 5 tokens/s（M3 Max 128GB） |

---

## 二、架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Qwen3 Engine                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Public API Layer                                                        │
│  ├── ds3_engine_open()     加载模型 + 初始化 Metal                      │
│  ├── ds3_engine_generate() 生成 token 序列                              │
│  ├── ds3_session_create()  创建对话会话                                  │
│  └── ds3_session_save()    保存 KV cache 到磁盘                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Model Layer                                                             │
│  ├── GGUF Loader (mmap zero-copy)                                       │
│  ├── Weights Bind (tensor name → struct field)                          │
│  └── Quantization Router (Q2_K / Q4_K_M / Q8_0 / FP16)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Inference Core                                                          │
│  ├── Prefill: 批量处理 prompt tokens                                    │
│  ├── Decode: 自回归生成（单次 1 token）                                  │
│  └── Speculative: draft model 投机解码（后期）                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Metal GPU Layer                                                         │
│  ├── Command Buffer Batching                                            │
│  ├── Kernel Library (Metal Shading Language)                            │
│  └── Memory Management (MTLBuffer + Residency Set)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  KV Cache Layer                                                          │
│  ├── 30B: 内存 Ring Buffer（可选 Q8_0 量化）                            │
│  └── 235B: KV 量化 + 磁盘 offload + 滑动窗口                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Tokenizer                                                               │
│  └── GPT-2 BPE (Qwen3 vocab: 151936 tokens)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
User Input
    │
    ▼
[Tokenizer] ──► Token IDs (BPE encode)
    │
    ▼
[Chat Template] ──► Formatted prompt tokens
    │
    ▼
[Prefill] ──► 批量 forward ──► 更新 KV cache
    │
    ▼
[Decode Loop] ◄────────────────────────────┐
    │                                        │
    ├──► [Metal Graph Eval] ──► Logits      │
    │       ├── Embed                        │
    │       ├── Layer 0..N-1                 │
    │       │   ├── RMS Norm                 │
    │       │   ├── GQA Attention            │
    │       │   ├── MoE FFN (top-8)          │
    │       │   └── Residual                 │
    │       └── Output Norm                  │
    │                                        │
    ├──► [Sampler] ──► Next Token            │
    │                                        │
    └──► [KV Cache Update] ──────────────────┘
    │
    ▼
[Tokenizer] ──► Detokenize ──► Text Output
```

---

## 三、关键技术方案

### 3.0 模型常量表（Qwen3 配置）

| 常量 | Qwen3-30B-A3B | Qwen3-235B-A22B | 说明 |
|------|---------------|-----------------|------|
| `N_LAYER` | **48** | **94** | Transformer 层数 |
| `N_EMBD` | **2048** | **4096** | 隐藏层维度 |
| `N_HEAD` | **32** | **64** | Attention heads（**需在 Phase 1.2 验证 attn_q.weight shape**） |
| `N_HEAD_KV` | **4** | **4** | GQA KV heads |
| `HEAD_DIM` | **128** | **128** | 每个 head 的维度（**需在 Phase 1.2 验证**） |
| `ROPE_THETA` | **1,000,000** | **1,000,000** | RoPE 基频（长上下文关键） |
| `N_EXPERT` | **128** | **128** | MoE 专家总数 |
| `N_EXPERT_USED` | **8** | **8** | 每次激活的专家数 |
| `N_FF_EXP` | **768** | **1536** | 单个专家 FFN 中间维度 |
| `N_FF_SHARED` | **6144** | **12288** | **共享专家 intermediate_size** |
| `N_VOCAB` | **151936** | **151936** | 词表大小 |
| `NORM_TOPK_PROB` | **true** | **true** | 路由权重归一化 |
| `NORM_EPS` | **待确认** | **待确认** | RMSNorm epsilon（需查 config.json） |
| `ROUTER_BIAS` | **待确认** | **待确认** | 是否存在 `ffn_gate_inp_bias`（需查 GGUF） |

> 来源：Qwen3 官方 `config.json` + GGUF tensor 形状验证。

---

### 3.1 模型加载：mmap 零拷贝（直接复用 ds4）

**方案**：`MAP_SHARED` + `newBufferWithBytesNoCopy`

```c
// 伪代码
int fd = open("model.gguf", O_RDONLY);
void *map = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);

// Metal 直接包装 mmap 地址
MTLBuffer *buffer = [device newBufferWithBytesNoCopy:map
                                              length:file_size
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
```

**收益**：
- 模型不占用应用内存，由 OS 页缓存管理
- GPU 直接读取，零拷贝
- 冷启动快（按需换页）

**Mac 特化**：
- `MAP_PRIVATE` fallback 仅用于 CPU debug 路径（规避 Darwin VM panic）
- Residency Set (macOS 15+) 预注册 model view

---

### 3.2 非对称量化：分层精度策略

基于 Perplexity 验证结果，确定以下策略：

#### 30B-A3B 推荐配置

| 层级 | Tensor 模式 | 精度 | 理由 | 体积占比 |
|------|------------|------|------|---------|
| Expert gate/up | `*ffn_gate_exps*` / `*ffn_up_exps*` | **Q2_K** | 验证通过（+8.46%） | ~54% |
| Expert down | `*ffn_down_exps*` | **Q2_K** | 同上 | ~27% |
| Attention Q/K/V/O | `*attn_q*` / `*attn_k*` / `*attn_v*` / `*attn_output*` | **Q8_0** | 敏感，保精度 | ~8% |
| Router / Norm | `*ffn_gate_inp*` / `*norm*` | **FP32** | 数值稳定性 | ~1% |
| Embedding | `token_embd` | **FP16** | 输入质量 | ~5% |
| Output | `output` / `output_norm` | **FP16** | 输出质量 | ~3% |

**预期体积**：
- 全 FP16: ~60 GB
- Q4_K_M: ~17 GB
- **Asym Q2+Q8: ~10-11 GB**

#### 235B-A22B 推荐配置

| 层级 | 精度 | 理由 |
|------|------|------|
| Expert gate/up | **IQ2_XXS** (2.06 bit) | 更激进 |
| Expert down | **Q2_K** (2.5 bit) | 稍保守 |
| Shared Expert | **Q4_K_M** | 高频激活 |
| Attention | **Q8_0** | 敏感 |
| Embedding/Output | **FP16** | 质量关键 |

**预期体积**：
- 全 Q4_K_M: ~132 GB ❌
- **Asym IQ2+Q8: ~75-85 GB** ✅
- **+ KV cache @128K (Q8_0): ~12 GB**
- **总计: ~90-102 GB** → 128GB Mac 可运行（余量 ~20-30 GB）

---

### 3.3 Attention：标准 GQA（重写 ds4 的 MLA）

Qwen3 使用标准 GQA，不是 ds4 的 MLA。

**结构**：
```
Input (hidden=2048)
    │
    ├──► Q_proj [2048, 4096] ──► 32 heads × 128 dim (4096 = 32 × 128)
    ├──► K_proj [2048, 512]  ──► 4 kv_heads × 128 dim (512 = 4 × 128)
    └──► V_proj [2048, 512]  ──► 4 kv_heads × 128 dim (512 = 4 × 128)

Q × K^T ──► Softmax ──► × V ──► O_proj [4096, 2048] ──► Output
```

> 注意：GQA 中 Q 有 32 个 head，但 K/V 只有 4 个 head。每个 Q head 共享对应的 K/V head（32/4 = 8 个 Q head 共享 1 个 KV head）。

**Metal Kernel 设计**：
```metal
// GQA Attention Kernel
// Q head 0-7  → KV head 0
// Q head 8-15 → KV head 1
// Q head 16-23 → KV head 2
// Q head 24-31 → KV head 3
// (contiguous 分组，需在 Phase 1 用单层 CPU reference 验证)
kernel void gqa_attention(
    device const float4 *q [[buffer(0)]],       // [batch, n_q_head=32, head_dim=128]
    device const float4 *k_cache [[buffer(1)]],  // [max_seq, n_kv_head=4, head_dim=128]
    device const float4 *v_cache [[buffer(2)]],  // [max_seq, n_kv_head=4, head_dim=128]
    device float4 *output [[buffer(3)]],
    constant uint &seq_len [[buffer(4)]],
    constant uint &n_q_head [[buffer(5)]],      // 32
    constant uint &n_kv_head [[buffer(6)]],     // 4
    constant uint &head_dim [[buffer(7)]],      // 128
    uint3 tid [[thread_position_in_grid]]
)
```

> **KV head 映射**：GQA 的核心是 4 个 KV head 被 32 个 Q head 共享。通常是 **contiguous 分组**（相邻 8 个 Q head 映射到同一个 KV head），但需在 Phase 1 用单层 CPU reference 验证 Qwen3 是否使用 interleaved 或其他分组方式。

**关键优化**：
- GQA 的 KV cache 只有 4 heads（vs Q 的 32 heads），内存节省 8×
- RoPE 应用在所有 head_dim（不是 ds4 的 tail-only）
- Flash Attention 算法减少 HBM 读写

---

### 3.4 MoE FFN：top-8 of 128 experts（适配 ds4）

**流程**：
```
Input
    │
    ├──► Router (ffn_gate_inp @ input + bias?) ──► Softmax ──► top-8 专家索引
    │       ↑                                              │
    │       └─ 是否存在 bias 需查 GGUF（ffn_gate_inp_bias）   └──► norm_topk_prob 重新归一化
    │                                                              使 8 个权重之和为 1
    │
    ├──► [Shared Expert] (始终执行，无 routing):
    │       gate = SiLU(gate_proj @ input)    // [6144, 2048]
    │       up   = up_proj @ input            // [6144, 2048]
    │       shared_out = (gate * up) @ down_proj  // [2048, 6144]
    │
    └──► 选中的 8 个专家并行计算：
             ├─ Expert_i:
             │      gate = SiLU(gate_proj @ input)   // [768, 2048]
             │      up   = up_proj @ input           // [768, 2048]
             │      ffn_output = (gate * up) @ down_proj × weight_i  // [2048, 768]
             ├─ Expert_j: ...
             └─ ... (共 8 个)
    
         ──► routed_out = 加权求和(8 个 expert 输出)
         ──► Output = shared_out + routed_out    ← 两路相加
```

> **SwiGLU 结构**：
> - 路由专家：`gate_proj [768, 2048]`, `up_proj [768, 2048]`, `down_proj [2048, 768]`
> - 共享专家：`gate_proj [6144, 2048]`, `up_proj [6144, 2048]`, `down_proj [2048, 6144]`
>
> ⚠️ 某些层可能使用 **hash routing**（类似 ds4 的 hash layer），需在 Phase 1 验证 router 是否只有 softmax 逻辑。

> ⚠️ **关键：norm_topk_prob**。config.json 中 `norm_topk_prob: true`，表示选出 top-8 后，要**重新对这 8 个 softmax 权重做归一化**（使其和为 1）。如果遗漏这一步，MoE 输出的幅度会不对。

**Metal Kernel 设计**：
```metal
kernel void routed_moe_q2(
    device const void *expert_gate_up,  // Q2_K 压缩权重
    device const void *expert_down,     // Q2_K 压缩权重
    device const float *input,
    device float *output,
    device const int *expert_ids,       // top-8 索引
    device const float *weights,        // router 权重（已 norm_topk_prob 归一化）
    constant uint &n_experts_used,      // 8
    uint3 tid [[thread_position_in_grid]]
)
```

**norm_topk_prob 实现**：
```metal
// 在 router_select kernel 中，选出 top-8 后：
float sum = 0.0;
for (int i = 0; i < 8; i++) sum += weights[i];
for (int i = 0; i < 8; i++) weights[i] /= sum;  // 重新归一化
```

**与 ds4 的差异**：
- Expert 数：256 → 128
- Top-k：6 → 8
- 量化：IQ2_XXS/Q2_K/Q4_K（混合）→ 统一 Q2_K（30B）或 IQ2_XXS/Q2_K（235B）

---

### 3.5 KV Cache 策略：分模型差异化

#### 30B-A3B（48 layers, 4 kv_heads, head_dim=128）

| 上下文 | KV 大小（FP16） | 策略 |
|--------|----------------|------|
| 4K | ~0.4 GB (FP16) | 内存 ring buffer，无需量化 |
| 32K | ~3 GB (FP16) | 内存 ring buffer，可选 Q8_0 |
| 128K | ~12 GB (FP16) / ~6 GB (Q8_0) | 内存足够，无需复杂策略 |

**实现**：标准 ring buffer，够简单。

#### 235B-A22B（94 layers, 4 kv_heads, head_dim=128）

| 上下文 | KV 大小（FP16） | 策略 |
|--------|----------------|------|
| 4K | ~0.75 GB (FP16) | 内存 ring buffer |
| 32K | ~6 GB (FP16) | 内存 + 可选 KV Q8_0 |
| 128K | ~24 GB (FP16) / ~12 GB (Q8_0) | **KV 量化 + 磁盘 offload + 滑动窗口** |

> ⚠️ **不可照搬 ds4 的三级压缩**。ds4 的三级压缩是为 MLA（低秩潜在向量）设计的：
> - ds4 KV 是 512 维潜在向量，压缩在潜在空间进行
> - Qwen3 使用标准 GQA（4 heads × 128 dim），没有潜在空间可压缩
> - ds4 的 indexer（topK-512 稀疏注意力）依赖 MLA 的特殊结构
>
> **正确方案**：
> 1. **KV 量化**：FP16 → Q8_0（体积减半）或 Q4_K_M（体积减至 1/4）
> 2. **磁盘 offload**：冷 KV（历史早期 token）换出到 SSD，热 KV（最近 token）留内存
> 3. **滑动窗口 + Sink Token**：保留开头若干 token（sink）+ 最近 N 个 token，中间丢弃

---

### 3.6 RoPE：GPT-NeoX 布局（重写 ds4 的 tail-only）

Qwen3 使用 **GPT-NeoX 风格 RoPE**：对每一对 `(d, d + head_dim/2)` 做 2D 旋转，而不是相邻对 `(d, d+1)`。llama.cpp 将 qwen3moe 的 rope type 标记为 `LLAMA_ROPE_TYPE_NEOX`。

> ⚠️ **关键参数**：Qwen3 的 `rope_theta = 1,000,000`（不是常见的 10,000）。theta 错误会导致长上下文能力完全失效。  
> ⚠️ **布局陷阱**：seq_pos=0 时 NeoX 与相邻对布局结果相同，因此单 token 验证无法发现；多 token decode 才会暴露错误并导致 logits 分叉 / 重复循环。

```metal
kernel void rope_neox(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    constant uint &seq_pos [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant float *freq_table [[buffer(4)]], // theta_i = base^(-2i/head_dim)
    uint3 tid [[thread_position_in_grid]]
)
{
    uint i = tid.x;
    uint n = head_dim / 2;
    float cos_val = cos(seq_pos * freq_table[i]);
    float sin_val = sin(seq_pos * freq_table[i]);

    // NeoX rotation: pair (i, i + head_dim/2)
    float q0 = q[i], q1 = q[i + n];
    q[i]     = q0 * cos_val - q1 * sin_val;
    q[i + n] = q0 * sin_val + q1 * cos_val;

    // k 同理...
}
```

**与 ds4 的差异**：
- ds4 只对尾部 64 dim 应用 RoPE，Qwen3 需要全维度。
- ds4 使用相邻对旋转，Qwen3 使用半区对旋转。

---

### 3.8 Prefill 分块策略

128K prompt 无法一次性分配所有中间激活，必须 chunked prefill：

```
Prompt: [t0, t1, ..., t131071]  (128K tokens)
    │
    ├──► Chunk 0:   [t0..t511]     ──► forward ──► 写入 KV cache
    ├──► Chunk 1:   [t512..t1023]  ──► forward ──► 写入 KV cache
    ├──► ...
    └──► Chunk N:   [t130560..t131071] ──► forward ──► 写入 KV cache
         │
         └──► 最后一个 chunk 的最后一个 token 的 hidden state
              作为 decode 的起始输入
```

**参数**：
- `chunk_size = 512`（decode 优先，低延迟）或 `1024`（prefill 吞吐优先）
- 每 chunk 处理后立即写入 KV cache，释放中间激活

**性能预估**：
- 235B @128K：250 chunks × ~50ms ≈ **12.5 秒** 完成全量 prefill
- 30B @128K：速度约快 2-3 倍

---

### 3.7 Tokenizer：GPT-2 BPE（Qwen3 适配）

**数据结构**：
```c
struct ds3_vocab {
    ds3_str *token;           // 151936 个 token 字符串
    int n_vocab;              // 151936
    int bos_id, eos_id;
    int im_start_id, im_end_id;
    str_i32_table token_to_id;  // hash: string → id
    str_i32_table merge_rank;   // hash: merge pair → rank
};
```

**Chat Template**（Qwen3 格式）：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

**Special Token IDs**（需从 `tokenizer_config.json` 提取）：

| Token | 用途 | 获取方式 |
|-------|------|---------|
| `<|im_start|>` | 对话轮次开始 | `tokenizer_config.json` |
| `<|im_end|>` | 对话轮次结束 | `tokenizer_config.json` |
| `<|endoftext|>` | EOS | `tokenizer_config.json` |
| `system` / `user` / `assistant` | Role 标签 | 作为普通文本 tokenize |

**与 ds4 的差异**：
- ds4 使用 `<｜User｜>` / `<｜Assistant｜>` 格式
- Qwen3 使用 `<|im_start|>` + role 标签
- Special token 名称不同（必须从 Qwen3 `tokenizer_config.json` 确认确切 ID）

---

## 四、文件结构

```
qwen3-engine/
├── src/
│   ├── ds3.h                  # 公共 API + 数据结构定义
│   ├── ds3.c                  # 核心引擎（GGUF 加载、推理循环）
│   ├── ds3_metal.h            # Metal C 接口
│   ├── ds3_metal.m            # Metal Objective-C 实现
│   ├── ds3_quant.c            # 量化格式解码（Q2_K / Q4_K_M / Q8_0）
│   ├── ds3_tokenizer.c        # BPE tokenizer
│   ├── ds3_session.c          # 会话管理 + KV cache
│   └── ds3_server.c           # HTTP API server
│
├── metal/
│   ├── common.metal           # 公共函数（量化解码、类型定义）
│   ├── rms_norm.metal         # RMS Normalization
│   ├── rope.metal             # GPT-NeoX 风格 RoPE
│   ├── gqa_attention.metal    # GQA Attention
│   ├── moe_routed.metal       # MoE top-k 路由 + 专家计算
│   ├── matmul_q2.metal        # Q2_K 矩阵乘法
│   ├── matmul_q4.metal        # Q4_K_M 矩阵乘法
│   ├── matmul_q8.metal        # Q8_0 矩阵乘法
│   └── swiglu.metal           # SwiGLU 激活
│
├── tests/
│   ├── test_perplexity.c      # Perplexity 验证工具
│   ├── test_tokenizer.c       # Tokenizer 单元测试
│   └── test_metal.c           # Metal kernel 单元测试
│
├── Makefile
└── README.md
```

---

## 五、开发路线图

### Phase 1: 基础推理 MVP（2-3 周）

**目标**：Qwen3-30B-Q4_K_M 的 GGUF 加载、单层验证、Tokenizer 完备，Metal kernel 框架就位。

| # | 任务 | 工作量 | 依赖 |
|---|------|--------|------|
| 1.1 | **定义模型常量表**（含 `N_FF_SHARED=6144` 等） | 小 | 无 |
| 1.2 | 重写 `weights_bind()`（Qwen3 tensor 命名） | 中 | 1.1 |
| 1.3 | 重写 Tokenizer（BPE + Qwen3 chat template + special token IDs） | 中 | 无 |
| 1.3a | 提取 special token IDs（`<|im_start|>`, `<|im_end|>` 等） | 小 | 1.3 |
| 1.4 | 单层 CPU reference：GQA Attention + RoPE + MoE | 大 | 1.1 |
| 1.5 | 单层 Metal kernel 框架：rms_norm / rope / matmul | 中 | 1.1 |
| 1.6 | 单层验证：对比 CPU reference vs Python ground truth | 中 | 1.4 |

**验收标准**：
- 能正确加载 Qwen3-30B-Q4_K_M.gguf 并解析所有 tensor
- 单层（Layer 0）CPU reference 输出与 Python 实现一致
- Metal kernel 框架编译通过，可 dispatch 单个 kernel

> **Phase 1 不做端到端生成**。端到端在 Phase 2 完成。

---

### Phase 2: 全 Metal 端到端 + 性能优化（2-3 周）

**目标**：完整 Metal 推理路径跑通，速度超过 llama.cpp。

| # | 任务 | 工作量 | 依赖 |
|---|------|--------|------|
| 2.1 | 完善 Metal attention kernel（GQA，全 48 层） | 大 | 1.5 |
| 2.2 | 完善 Metal RoPE kernel（全维度） | 中 | 1.5 |
| 2.3 | 完善 Metal MoE kernel（128 experts, top-8, norm_topk_prob） | 中 | 1.5 |
| 2.4 | 实现 mmap + zero-copy | 中 | 无 |
| 2.5 | Model View Overlap（>32GB 模型） | 中 | 2.4 |
| 2.6 | Metal 路径端到端跑通（48 层完整 forward） | 大 | 2.1-2.5 |
| 2.7 | 性能基准（vs llama.cpp Metal） | 小 | 2.6 |
| 2.8 | **正确性验证**：greedy decoding 输出 vs llama.cpp 逐 token 对比 | 小 | 2.6 |

> **正确性验证方法**：对同一 prompt 做 greedy decoding（temperature=0），对比两个引擎输出的前 100 个 token。如有 1-2 个不同，检查 logits top-5 差异判断是数值精度问题还是逻辑错误。

**验收标准**：
- `./qwen3-cli -m Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello"` 能输出连贯文本
- M3 Max 上 30B-Q4_K_M 达到 >20 tok/s

---

### Phase 3: 量化落地（1-2 周）

**目标**：支持混合精度 GGUF，验证 PPL。

| # | 任务 | 工作量 | 依赖 |
|---|------|--------|------|
| 3.0 | **非对称 GGUF 量化工具**（Python，按 tensor 名模式指定精度） | 中 | 无 |
| 3.1 | 支持 Q4_K_M 加载和推理 | 小 | 2.6 |
| 3.2 | 支持 Q2_K 反量化 Metal kernel | 中 | 2.6 |
| 3.3 | 支持 IQ2_XXS 反量化 Metal kernel（235B 需要） | 大 | 2.6 |
| 3.4 | 支持 Asym Q2+Q8（混合精度调度） | 中 | 3.0-3.3 |
| 3.5 | Perplexity 验证（vs llama.cpp） | 小 | 3.4 |
| 3.6 | 性能对比（Q4 vs Asym Q2） | 小 | 3.5 |

**验收标准**：Asym Q2+Q8 PPL 劣化 < 10%（已验证：+8.46%）。

---

### Phase 4: 235B 准备 + 投机解码（3-5 周）

**目标**：235B 模型能加载并运行，投机解码让速度翻倍。

| # | 任务 | 工作量 | 依赖 |
|---|------|--------|------|
| 4.1 | 修改常量（94层/4096dim） | 小 | Phase 3 |
| 4.2 | 实现 KV 量化 + 磁盘 offload | 大 | 无 |
| 4.3 | KV cache 磁盘持久化 | 中 | 4.2 |
| 4.4 | 更大 batch size 优化 | 中 | 4.1 |
| 4.5 | 235B 模型加载测试 | 小 | 4.1-4.4 |
| 4.6 | 128K 长上下文压力测试 | 中 | 4.5 |
| 4.7 | **投机解码（Qwen3-0.6B draft model）** | **大** | 4.5 |
| 4.8 | **Rust KV Cache Service**（独立进程，UDS 协议，prefix sharing） | **大** | 4.2-4.3 |

> **4.8 详细设计**：见 [`ds3-kv-cache-service-plan.md`](ds3-kv-cache-service-plan.md)。借鉴 LMCache 分层存储思想，利用 Apple Silicon 统一内存零拷贝优势，实现 chunk-based prefix sharing + LRU 淘汰 + session 持久化。

> **为什么投机解码在 Phase 4？** 235B decode 预估仅 5-10 t/s，投机解码（draft model）可能是**唯一能让速度翻倍的手段**。
> 
> Qwen3 **没有 MTP 头**（这是 DeepSeek V3/V4 特有的），无法使用 MTP 投机解码。可行方案：
> - **Qwen3-0.6B draft model**：同 vocab 小模型做草稿，大模型验证（加速 2-3×，工作量中）
> - **Self-speculative**：只跑前 N 层做草稿（加速 1.5-2×，工作量中）
> - **EAGLE-style**：训练额外 head（加速 2-4×，工作量大）

**验收标准**：128GB Mac 上 235B 模型能加载并生成 token。

---

### Phase 5: 工程化（持续）

| # | 任务 | 工作量 |
|---|------|--------|
| 5.1 | Server 模式（HTTP API） | 中 |
| 5.2 | 会话管理和持久化 | 中 |
| 5.3 | 投机解码框架（draft model 支持） | 大 |
| 5.4 | CLI 工具完善（多轮对话、文件输入） | 小 |
| 5.5 | 文档和示例 | 小 |

---

## 六、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Metal kernel 性能不达预期 | 中 | 高 | 先用 CPU 验证正确性，再优化 kernel |
| 235B PPL 劣化 >15% | 中 | 高 | 准备 fallback：expert 从 Q2 改 Q3/Q4 |
| macOS VM bug 触发 | 低 | 高 | 保持 Metal-only，禁用 CPU fallback |
| GGUF 格式兼容问题 | 低 | 中 | 用 gguf-py 验证 tensor 命名和格式 |
| 开发周期超预期 | 中 | 中 | Phase 1-2 优先，先出 MVP |

---

## 七、与 ds4.c 的关键差异汇总

| 维度 | ds4 (DeepSeek V4) | qwen3-engine (Qwen3) |
|------|-------------------|---------------------|
| **通用性** | 单一模型专用 | 单一模型专用 |
| **Attention** | MLA (压缩 Q/KV) | **标准 GQA** |
| **RoPE** | tail-only 64 dim | **全维度 GPT-NeoX 布局** |
| **Experts** | 256, top-6 | **128, top-8** |
| **Hidden dim** | 4096 | **2048 (30B) / 4096 (235B)** |
| **Layers** | 43 | **48 (30B) / 94 (235B)** |
| **Head dim** | 512 | **128** |
| **KV heads** | 1 (MLA) | **4 (GQA)** |
| **Tokenizer** | DeepSeek BPE | **GPT-2 BPE (Qwen3)** |
| **Chat template** | `<｜User｜>` | **`<|im_start|>user`** |
| **量化** | Asym Q2 (expert only) | **Asym Q2 (验证通过)** |
| **KV cache** | 三级压缩 + 磁盘 | **30B: 内存 / 235B: KV 量化 + 磁盘 offload** |
| **代码结构** | 单文件 ~16K 行 | **模块化，多文件** |

---

## 八、参考文档

| 文档 | 路径 |
|------|------|
| 第一轮实验报告 | `qwen3_30b_perplexity_validation.md` |
| 路线 A 报告 | `route_a_imatrix_iq2_xxs_results.md` |
| 路线 B-1 报告 | `route_b1_asymmetric_quantization_results.md` |
| 实验计划 V2 | `qwen3_experiment_plan_v2.md` |
| KV Cache Service 设计 | [`ds3-kv-cache-service-plan.md`](ds3-kv-cache-service-plan.md) |
| ds4 架构分析 | 本文档 Section 3 引用 |

---

*文档版本：v1.0*  
*创建时间：2026-05-12*  
*项目状态：Phase 0（设计完成），准备进入 Phase 1*
