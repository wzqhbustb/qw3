# Qwen3-235B 专属推理引擎计划

基于对 ds4.c 项目的深度分析和多轮讨论，整理出为 Qwen3-235B-A22B 打造专属 Metal 推理引擎的核心结论。

---

## 核心前提

- **Phase 1 硬件**：48GB Mac（M3 Max，现有设备）
- **Phase 1 模型**：Qwen3-30B-A3B（架构验证 + 全流程走通）
- **Phase 2 硬件**：128GB Mac（M5 Max 或同等配置）
- **Phase 2 模型**：Qwen3-235B-A22B（MoE，128 专家选 8，活跃参数 22B）
- 方法论：复用 ds4.c 的设计理念 — 一个模型、一个平台、做到极致
- 策略：两阶段渐进式开发，零硬件投资验证核心假设

---

## 模型架构参数（真实值）

| 参数 | 值 |
|------|-----|
| 总参数 | 235B |
| 活跃参数 | 22B |
| 层数 | 94 |
| hidden_size | 4096 |
| 注意力 | GQA，4 KV heads，head_dim=128 |
| 专家数 | 128 选 8 |
| FFN | MoE + 共享专家 |
| 上下文窗口 | **128K**（官方声明，Qwen 博客/模型卡确认） |
| max_position_embeddings | 40960（预训练位置编码上限，非推理上限） |
| rope_theta | 1,000,000（高频基底，原生支持长上下文外推，无需 YaRN） |
| 滑动窗口 | 无 |

---

## 内存预算

```
路由专家 Q2（非对称量化）:   ~65-70 GB
共享层 + 投影 FP16:          ~12-15 GB
────────────────────────────────────────
模型权重合计:                ~80-85 GB

KV 缓存（每层 = ctx × 128 × 4 × 2 × 2 字节，共 94 层）:
  @40K:                      ~7.3 GB
  @100K:                     ~18.4 GB
  @128K:                     ~23.5 GB
系统 + 推理 scratch:         ~10 GB
────────────────────────────────────────
总计 @40K:                   ~100 GB / 128 GB  （宽裕）
总计 @100K:                  ~113 GB / 128 GB  （可行）
总计 @128K:                  ~119 GB / 128 GB  （非常紧张但可行）
```

> **注意**：128K 满上下文时内存余量仅约 9GB，建议实际使用中以 100K 为默认上限，
> 128K 作为极限能力保留。

---

## 价值主张：为什么值得做

### 核心故事

**"非对称量化 + 128K 上下文 = 128GB Mac 上跑 235B frontier 级模型做 coding agent"**

模型官方支持 128K 上下文（rope_theta=1M 高频基底原生外推），长上下文叙事部分成立。
虽不及 DS4 Flash 的百万 token + MLA 压缩，但 128K 对 coding agent 和长文档处理完全够用。

### "不可能变可能"的逻辑

```
通用引擎的现实:
  - 均匀 Q2 → 质量退化严重，不可用
  - Q4 → ~140GB，128GB Mac 装不下
  - 不支持"只压专家、其余 FP16"的非对称格式

专属引擎:
  - 非对称 Q2 → 85GB 装得下
  - 质量接近 Q4（因为投影/路由/共享层未动）
  - 128GB Mac 上可用的 frontier 级模型
```

---

## 与 ds4.c 的对比

| 维度 | ds4.c + V4 Flash | 你 + Qwen3-235B |
|------|------------------|-----------------|
| "不可能变可能" | 五星 | 四星 |
| 非对称量化质量 | 五星 | 五星（同理） |
| MoE kernel 优化 | 四星 | 四星（128选8 vs 256选6） |
| 磁盘 KV 缓存 | 五星（1GB@100K） | 三星（18GB@100K，SSD 读写约 12 秒） |
| 长上下文 | 五星（百万 token） | 三星半（128K，对 agent 够用但非极致） |
| 通用引擎差距 | 巨大 | 中等 |

### 你有的

- 非对称量化让 235B 塞进 128GB — 质变
- MoE fused kernel — 实打实的速度提升
- 磁盘 KV — agent 会话恢复可接受（@40K 约 5 秒，@100K 约 12 秒）
- 固定形状编译期优化 — 消除运行时分发开销
- GQA 比 MLA 简单 — 开发难度更低

### 你没有的

- MLA 压缩 — KV 缓存体积远大于 DS4 Flash（同上下文约 18x）
- 百万 token 窗口 — 上限 128K，足够实用但非极致

---

## 适用场景

128K 上下文对 coding agent **可用**（系统提示约 25K tokens，剩余约 100K tokens 用于多轮对话、文件内容注入和工具结果，足够支撑十几轮深度 agent 交互）。同时也适合以下场景：

| 场景 | 说明 |
|------|------|
| **Coding agent** | 系统提示 25K + 多轮工具调用，100K 上下文空间充裕 |
| 单轮深度推理 | 难题思考，不需要多轮积累上下文 |
| 长文档问答/摘要 | 一次塞入 10+ 篇论文（128K ≈ 10 万字） |
| 高质量翻译 | 一次翻译 5-8 万字 |
| 数据分析/结构化提取 | 大表格 + 指令，单次处理 |
| 创意写作 | 长篇分章节生成 |
| 复杂指令遵循 | 多约束场景，小模型丢约束，235B 不丢 |
| 私密数据处理 | 合同、病历、财务 — 数据不出本机 |

### 最佳产品形态

**本地私有的 frontier 级推理 API 服务 + coding agent 后端**：

```
引擎 (localhost:8000)
    |-- Coding agent → opencode / Claude Code 兼容
    |-- Obsidian 插件 → 笔记摘要、知识问答
    |-- 翻译工具 → 高质量长文翻译
    |-- 终端工具 → 单轮复杂问题
    |-- RAG 应用 → 检索后单次回答
    |-- 批处理脚本 → 结构化数据提取
```

---

## Token 自由

| | 云端 API | 本地 235B |
|---|---|---|
| 质量 | GPT-4o / Claude Sonnet 级 | 同级 |
| 每百万 token 成本 | 15-60 元 | 0 元 |
| 月重度使用 (~1000 万 token) | 150-600 元 | 0 元 |
| 年费 | 1,800-7,200 元 | 0 元 |
| 硬件溢价（128GB vs 64GB） | - | ~3,000-6,000 元（一次性） |
| 回本周期 | - | 1-4 个月 |
| 数据隐私 | 数据上传第三方 | 完全本地 |
| 可用性 | 依赖网络/供应商 | 离线可用 |

---

## 从 ds4.c 可复用的部分

| 组件 | 复用度 | 说明 |
|------|--------|------|
| mmap 加载 + Metal 零拷贝 | 高 | 直接搬 |
| 磁盘 KV 缓存框架 | 高 | SHA1 键 + 四种保存时机 + 边界对齐 |
| 服务器架构 | 高 | 单 worker + 客户端线程 + OpenAI API |
| MoE kernel (moe.metal) | 中 | 改 top-k 数和 gate 逻辑 |
| 非对称量化策略 | 高 | 同理：只压路由专家 |
| 官方 logits 验证方法论 | 高 | 从 Qwen API 抓取对比 |

## 需要重新实现的部分

| 组件 | 工作量 | 说明 |
|------|--------|------|
| 模型形状常量 | 小 | 替换为 Qwen3 参数 |
| 注意力 kernel (GQA) | 中 | 比 MLA 简单 |
| RoPE | 低 | GPT-NeoX 布局 RoPE，rope_theta=1M 高频基底，无需 YaRN |
| 分词器 | 中 | BPE merge 表不同 |
| Chat 模板 | 小 | Qwen3 格式 |
| GGUF 制作工具 | 中 | 非对称量化打包 |

---

## 练手阶段：Qwen3-30B-A3B on 48GB Mac

在投入 128GB 硬件和 235B 模型之前，先用架构完全相同的小模型在现有硬件上走通全流程。

### 为什么选 Qwen3-30B-A3B

| 参数 | Qwen3-30B-A3B | Qwen3-235B-A22B | 关系 |
|------|---------------|-----------------|------|
| 架构 | MoE | MoE | **完全相同** |
| 专家数 | 128 选 8 | 128 选 8 | **完全相同** |
| KV heads | 4 (GQA) | 4 (GQA) | **完全相同** |
| head_dim | 128 | 128 | **完全相同** |
| 层数 | 48 | 94 | 约 1/2 |
| hidden_size | 2048 | 4096 | 1/2 |
| moe_intermediate_size | 768 | 1536 | 1/2 |
| intermediate_size (共享专家) | 4096 | 12288 | 1/3 |
| FP16 体积 | ~60 GB | ~470 GB | — |
| 非对称 Q2 体积 | **~11 GB** | ~85 GB | 48GB 轻松放下 |

**核心优势**：架构 100% 一致，只是维度减半。写的每一行 Metal kernel、每一个 GGUF 解析函数、
每一个 MoE 路由逻辑，都可以**改几个常量**直接搬到 235B。

### 练手阶段的独特价值

1. **内存极度宽裕** — 11GB 权重 + KV + scratch 总共不超过 20GB，48GB 剩余大量空间用于调试、profiling、跑对比
2. **迭代速度快** — 48 层 vs 94 层，每次测试快一倍；hidden_size 2048 让单层 CPU reference 秒出结果
3. **量化验证一石二鸟** — 在 30B 上验证非对称 Q2 的 perplexity 退化，结果直接指导 235B 的决策
4. **可对比 llama.cpp** — 30B 在 llama.cpp 上跑得很好，可随时对比自己引擎与通用引擎的速度/质量差异，量化自己的进步
5. **完整里程碑** — Phase 1 结束时产出一个能生成 token 的完整引擎，信心价值远大于技术价值
6. **提前暴露风险** — 如果 30B 上非对称 Q2 的 perplexity 就崩了，在投入 128GB 硬件之前就知道了

### 30B 的内存预算（48GB Mac）

```
路由专家 Q2（非对称量化）:     ~7-8 GB
共享层 + 投影 FP16:            ~3-4 GB
────────────────────────────────────────
模型权重合计:                  ~11 GB

KV 缓存（每层 = ctx × 128 × 4 × 2 × 2 字节，共 48 层）:
  @32K:                        ~1.9 GB
  @64K:                        ~3.8 GB
  @128K:                       ~7.5 GB
系统 + 推理 scratch:           ~4 GB
────────────────────────────────────────
总计 @64K:                     ~19 GB / 48 GB  （极度宽裕）
总计 @128K:                    ~23 GB / 48 GB  （非常宽裕）
```

> 48GB Mac 上跑 30B 模型有超过 25GB 的余量，可以同时跑 Python reference 做对比验证。

### 从 30B 到 235B 的升级清单

升级时只需改动的部分：

| 改动项 | 30B → 235B | 工作量 |
|--------|-----------|--------|
| hidden_size | 2048 → 4096 | 改常量 |
| 层数 | 48 → 94 | 改常量 |
| moe_intermediate_size | 768 → 1536 | 改常量 |
| intermediate_size | 4096 → 12288 | 改常量 |
| Metal threadgroup 尺寸 | 可能需调优 | 小 |
| 94 层 command buffer 批处理 | 新增优化 | 中 |
| 内存管理策略 | 从宽裕到紧张 | 中 |
| 磁盘 KV 缓存 | 30B 上可选，235B 上必要 | 中 |

**不需要改动的部分**：MoE 路由逻辑、GQA attention 结构、RoPE 实现、
GGUF 解析框架、量化/反量化 kernel 逻辑、server API、tokenizer。

---

## 风险点

1. **非对称 Q2 质量未验证** — 必须先做实验，确认 Qwen3-235B 在只压专家方案下 coding/推理可用。风险高于 DS4 的三个原因：(a) 每专家参数更少（18.9M vs DS4 的 25.2M，量化噪声占比更大）；(b) 94 层 × 每 token 8 专家 = 752 次量化矩阵乘，是 DS4（43 层 × 6 专家 = 258 次）的近 3 倍，误差累积路径更长；(c) `moe_intermediate_size: 1536` 虽整除 QK_K=256 无对齐问题，但行更短（6 blocks/行 vs DS4 的 8 blocks/行），分布估计的容错空间更小。**代理验证（Qwen3-30B-A3B）应优先关注 perplexity 退化幅度，而非仅看绝对值**
2. **128K 满上下文内存极紧张** — 权重 85GB + KV 24GB + scratch 10GB ≈ 119GB，仅剩 9GB 余量。建议默认 100K 上下文（总占用 ~113GB），128K 作为极限能力
3. **94 层 decode 较慢** — 是 DS4 (43层) 的 2.2 倍，预估生成速度 ~10-15 t/s（ds4 V4 Flash 43 层在 M3 Max 上约 27 t/s，层数翻倍后即使 M5 带宽更好也难超 15 t/s）
4. **通用引擎追赶** — llama.cpp 未来可能支持非对称量化，窗口期有限
5. **模型迭代风险** — Qwen 系列更新频繁（Qwen1.5 → Qwen2 → Qwen2.5 → Qwen3 间隔仅数月），引擎绑定特定模型形状，可能 3-6 个月后需要适配 Qwen3.5 或 Qwen4

---

## 预估工作量

总计 14-16 周，分两个阶段：

- **Phase 1**（6-7 周）：在 48GB M3 Max 上用 Qwen3-30B-A3B 走通全流程
- **Phase 2**（4-5 周）：在 128GB Mac 上升级到 Qwen3-235B-A22B

---

### Phase 1：Qwen3-30B-A3B 全流程（48GB M3 Max，6-7 周）

目标：产出一个**能生成 token 的完整引擎**，同时验证所有技术假设。

#### Phase 1 前置：Metal 基础学习（2 周）

```
□ Apple Metal 官方文档 + MSL 规范精读
□ 练习 1: 向量加法 kernel（理解 threadgroup / grid）
□ 练习 2: 矩阵乘法 naive 版（理解 2D dispatch）
□ 练习 3: 矩阵乘法 tiled 版（理解 shared memory / threadgroup memory）
□ 精读 ds4_metal.m 和 metal/moe.metal，对照写注释
□ 补充学习：Transformer 前向计算（Attention + MoE + RoPE 概念理解）
```

#### Phase 1 Week 1: 量化验证（决定是否继续）

```
□ 从 HF 拉 Qwen3-30B-A3B 的 config.json，确认架构参数
□ 用 transformers 加载 30B 模型（FP16 ~60GB，48GB Mac 需用 offload 或分层加载）
□ 做三组 perplexity 测量：FP16 基线、均匀 Q2、非对称 Q2（只压路由专家）
□ 从 Qwen API 获取 test vectors（greedy decoding + top logprobs）
```

**终止条件（双重标准）：**
- 非对称 Q2 相对 FP16 的 perplexity 退化 > 15% → 终止（量化不可用）
- 非对称 Q2 相对均匀 Q2 的改善 < 5% → 终止（非对称策略无意义）

#### Phase 1 Week 2: GGUF 工具链 + 最小加载器

```
□ 写 Python 脚本：只量化 routed experts（up/gate @ IQ2_XXS, down @ Q2_K），共享层 @ Q4_K
□ 生成 30B 的非对称量化 GGUF 文件（~11GB）
□ 写最小 C 加载器：mmap GGUF 后正确读取所有张量，打印 shape/type/offset
□ 验证 Metal MTLBuffer(bytesNoCopy:) 零拷贝绑定
```

#### Phase 1 Week 3: CPU 单层 reference

```
□ 写单层 CPU kernel：RMSNorm → GQA Attention → RMSNorm → MoE FFN
□ 用 Python（PyTorch）跑完整 30B 模型（可用 offload）抓取每层输入/输出
□ C 代码逐层对比，验证数值等价性（误差 < 1e-4 for FP16 path）
□ 重点验证 MoE 路由（128 选 8 的 top-k + softmax 归一化）
```

#### Phase 1 Week 4-5: Metal kernels

```
□ GQA FlashAttention — decode path（单 token 生成，最常用）
□ GQA FlashAttention — prefill path（prompt 处理）
□ MoE matvec（128 选 8，支持 IQ2_XXS / Q2_K / Q4_K 三种量化格式）
□ RoPE（GPT-NeoX 布局，rope_theta=1M，预留 YaRN freq_base/freq_scale/ext_factor 参数位）
□ RMSNorm、SwiGLU、embedding lookup 等辅助 kernel
□ 端到端单层 Metal vs CPU 对比验证
```

#### Phase 1 Week 6-7: 端到端 + Server

```
□ 把单层串成 48 层完整 graph
□ 对齐 greedy decoding：engine 输出 vs Qwen API 输出，token-level 匹配
□ 接入 server API（复用 ds4_server.c 架构）
□ 跑 benchmark：对比 llama.cpp 跑同一个 30B 模型的速度
□ 性能 profiling：识别瓶颈，为 Phase 2 的 94 层优化积累数据
```

**Phase 1 交付物**：一个完整的 Qwen3-30B-A3B 推理引擎，能通过 HTTP API 提供服务，
在 48GB Mac 上以竞争力的速度生成高质量文本。

---

### Phase 2：升级到 Qwen3-235B-A22B（128GB Mac，4-5 周）

前提：已有 128GB Mac 硬件 + Phase 1 产出的完整 30B 引擎。

#### Phase 2 Week 1: 常量替换 + GGUF 制作

```
□ 替换所有模型形状常量（hidden_size, layers, moe_intermediate_size 等）
□ 用 Phase 1 的量化脚本处理 235B 模型（输出 ~85GB GGUF）
□ 在 128GB Mac 上 mmap 加载，监控 VM 压力（vmmap, memory_pressure）
□ 验证 Metal 零拷贝绑定在大文件下的行为
```

#### Phase 2 Week 2: 单层验证 + 性能调优

```
□ 用 Python 跑 235B 完整模型（需云端或分布式环境）抓取单层 reference
□ C/Metal 单层对比验证（hidden_size 4096 + moe_intermediate_size 1536）
□ Metal threadgroup 尺寸重新调优（维度翻倍后最优配置可能不同）
□ 94 层命令批处理优化（减少 kernel dispatch 开销 — 深窄模型的关键瓶颈）
```

#### Phase 2 Week 3-4: 端到端 + 磁盘 KV

```
□ 把单层串成 94 层完整 graph
□ 端到端 greedy decoding 对齐（从 Qwen API 抓取 10 组 test vectors）
□ 内存预算精细管理（85GB 权重 + KV + scratch 在 128GB 内协调）
□ 实现磁盘 KV 缓存（GQA layout，与 Phase 1 的 KV 序列化格式一致但数据量更大）
□ 100K / 128K 上下文压力测试
```

#### Phase 2 Week 5: Server + 集成 + 发布

```
□ server API 适配（基本不变，只是内存管理更保守）
□ coding agent 兼容性测试（opencode / Claude Code 协议）
□ 完整 benchmark suite（速度、质量、内存、长上下文）
□ 文档 + README + 使用指南
```

**Phase 2 交付物**：完整的 Qwen3-235B-A22B 推理引擎，128K 上下文，
128GB Mac 上可用的 frontier 级本地推理服务。

### 关键设计决策

1. **两阶段渐进式开发** — 先在 48GB Mac 上用 30B 模型走通全流程（Phase 1），
   再在 128GB Mac 上升级到 235B（Phase 2）。30B 和 235B 架构 100% 一致（128 专家选 8，
   GQA 4 KV heads），维度恰好减半，所有 kernel 逻辑可直接复用。
   Phase 1 同时完成量化验证、Metal 学习和完整引擎实现，是零硬件投资的风险验证。

2. **CPU forward pass 降级为单层 reference** — ds4.c 的 `ds4_engine_first_token_test`
   是完整 43 层 CPU 跑一遍。Qwen3 有 94 层，CPU 全量推理在 macOS 上可能触发
   kernel panic。替代方案：Python 跑完整模型作为 ground truth，C/Metal 只验证单层等价性。

3. **先 Metal kernel，后 disk KV** — disk KV 的序列化格式高度依赖 KV cache 的内存布局。
   Qwen3 是标准 GQA，KV layout 和 ds4.c 的 MLA 完全不同。应先让内存中的 KV cache
   跑通，再最后做磁盘持久化。

4. **RoPE 预留扩展位** — Qwen3 使用 GPT-NeoX 布局半区对旋转；虽然 rope_theta=1M
   已支持 128K，仍在 RoPE kernel 中保留 freq_base、freq_scale、ext_factor 参数
   （ds4.c 的 `ds4_metal_rope_tail_tensor` 已有这些），为未来推到 256K+ 预留能力。

5. **Phase 1 产出可独立使用** — 30B 引擎本身就是一个有价值的产品（48GB Mac 上的
   高速 MoE 推理），即使 Phase 2 因硬件或其他原因延迟，Phase 1 的工作不会浪费。

---

## 最终结论

> 值得做。采用两阶段渐进式开发：先在 48GB Mac 上用 Qwen3-30B-A3B 走通全流程（Phase 1，6-7 周），
> 验证技术假设并产出可用引擎；再在 128GB Mac 上升级到 235B（Phase 2，4-5 周）。
>
> 128K 上下文让长上下文叙事成立，coding agent 场景可行。核心价值主张：
> **128GB Mac 上拥有一个 128K 上下文的 frontier 级私人大脑，永久免费调用，数据完全私有。**
>
> Phase 1 的独立价值：即使不升级到 235B，30B 引擎本身也是 48GB Mac 上的高速 MoE 推理方案，
> 可作为轻量 coding agent 后端使用。两阶段策略确保**零硬件投资即可验证所有核心假设**。
>
> 与 ds4.c 的差异化：ds4 做"百万上下文的极致"，本引擎做"128K 上下文的实用 + coding agent 兼容"。
