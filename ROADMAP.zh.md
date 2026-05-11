# Qwen3 Metal 引擎路线图

本文档记录项目方向。当前重点是：

> **P1a — 发布一个稳定、可开源的 Qwen3-30B-A3B 版本。**  
> Decode 优化（P0）已因实际天花板而关闭。235B 适配和聊天/UX 优化仍明确放到 30B 发布之后。

---

## ✅ 已完成

### Phase 4a — Chunk Prefill（分块预填充）
- [x] 层间串行的 chunk prefill 与 overlapped 调度。
- [x] Chunk 内 attention 投影使用批量量化矩阵乘法。
- [x] 与 vec kernel 浮点累加顺序一致的 SIMD-group batched matmul kernel。
- [x] Chunk 路径与 token 路径的正确性验证。

### Phase 4b — Gathered MoE（聚合专家）
- [x] 用于 batched routed expert 的 gather/scatter kernel。
- [x] `forward_chunk_post_gathered`：按选中专家聚合 token。
- [x] Shared expert 以普通 batched matmul 处理。
- [x] 回退开关 `DS3_CHUNK_NO_GATHERED_MOE=1`。
- [x] 正确性与性能验证。

### 配套工作
- [x] `token_embd` 和 `output.weight` 的 mmap zero-copy。
- [x] `AGENTS.md`：架构、构建、测试、环境开关说明。

---

## ✅ P0 — Decode 性能优化（已关闭）

当前 decode 基线约为 **17 tok/s**（M3 Max + `Qwen3-30B-A3B-Q4_K_M.gguf`）。

### 结论

P0 因“受架构/Metal 调度开销限制”而关闭：

- **2-CB 跨层 decode 在数学上不可行。** Transformer 层严格串行：layer `l+1`
  的 attention 输入必须是 layer `l` 的 **FFN 输出**。任何“先跑全部 attention、
  再跑全部 FFN”的调度（即使保存了 pre-FFN hidden/residual）都会破坏这一依赖，
  导致输出错误。实验已确认：第一个 token 一致，第二个 token 起 logits 分叉。
- **GPU-only router top-k + fused 1-CB mega-kernel 已被证伪**（Phase 3：
  **5.5 tok/s**，比默认慢 63%）。
- **`DS3_USE_GPU_ONLY_MOE=1` 已被证伪**（Phase 3：比 per-expert dispatch 慢 47%）。
- 唯一正确的低 CB 数路径是现有的 **overlapped 层间串行调度**（`post(l)` 与
  `pre(l+1)` 共享一个 command buffer），它仍然需要每 token 约 49 次 CB flush，
  已接近天花板。

### P0 任务
- [x] 调研低 CB 数 decode 调度。
- [x] 证明跨层 2-CB 调度在结构上不正确。
- [x] 把 ~17 tok/s 记录为当前架构的实际 decode 天花板。

**结果：** Decode 优化降级为“nice to have”。唯一已知能显著突破天花板的方案是
**speculative decoding**（P4），但它需要一个独立的 draft 模型。

---

## 📦 P1 — Qwen3-30B-A3B 发布打磨

🚧 **当前重点：P1b。**

P1a 已完成。P1 分为 **P1a（开源前必须完成）** 和 **P1b（开源后持续打磨）**。

### P1a — 开源阻塞项 ✅

这些必须在仓库公开前完成。

- [x] **LICENSE**
- [x] **清理 `.gitignore`**
- [x] **README.md**，包含：
  - 项目简介与范围（是什么 / 不是什么）。
  - 系统要求（Apple Silicon、macOS 14+、30B Q4_K_M 约需 20 GB 空闲内存）。
  - 构建说明。
  - 模型下载指引（HuggingFace 仓库 + Qwen3-30B-A3B-Q4_K_M 精确文件名）。
  - 快速开始示例。
- [x] **干净构建验证**
  - 确认从全新 clone 开始，`make qwen3-cli` 和所有 `make test_*` 都能通过。
- [x] **识别结束标记并停止生成**
  - 遇到 `<|im_end|>` 和 `</think>` 时提前停止。
  - 当前实现总是固定生成 `n` 个 token，会拖慢 benchmark 并产生多余内容。
- [x] **默认启用 chunk prefill**
  - `forward_chunk` 已验证正确，且速度快得多（130 token prompt：18 s → 2.5 s）。
  - 短 prompt 自动回退到 token path。
  - 反转环境开关语义，让用户用 `DS3_CHUNK_NO_BATCHED_MATMUL=1` 等选项退出，而不是默认关闭。
- [x] **端到端生成质量验证**
  - 在相同 prompt 和 temperature 下，与 llama.cpp 或 transformers 对比生成文本（先用 argmax，再用固定种子采样）。
  - **重复循环的根因：** RoPE 之前实现了 LLaMA 的相邻对旋转，但 Qwen3 使用 GPT-NeoX 的半区对旋转 `(i, i + head_dim/2)`。已在 `metal/rope.metal` 和 `src/ds3_reference.c` 中修复。
- [x] **Q4_K_M 反量化 ground-truth 验证**
  - 当前只对比 GPU 输出 vs CPU 反量化输出。需要增加与已知正确 ground-truth（如 PyTorch 或 llama.cpp）的对比。

### P1b — 开源后打磨

这些能提升贡献者/用户体验，但可以在首个 release 之后补充。

- [ ] **CI 配置**（macOS + Metal 的 GitHub Actions 或等价方案）。
- [ ] **脚本化验证套件**
  - `-F` 全模型参考对比。
  - Chunk vs token 生成一致性。
  - Gathered vs per-token MoE 一致性。
  - 所有 `make test_*` Metal 单元测试。
- [x] **`docs/BENCHMARK.md`**：可复现的 benchmark 命令与预期数字。
- [ ] **CLI 易用性改进**
  - 更完整的 `--help`。
  - 针对常见失败的友好报错：模型路径缺失、GGUF 加载失败、context 越界、Metal 初始化失败。
  - [x] `-q` / `--quiet` 开关：抑制非错误信息输出。
  - [ ] 可选的输出到文件开关。
- [x] **清理环境开关**
  - 已删除 `DS3_NO_SYNC_LAYERS=1` 及其错误的跨层 batch 路径。
  - 已删除 `DS3_USE_2CB_DECODE=1` 及错误的双 command buffer decode 路径。
  - 只保留有意义、有文档的开关。
- [ ] **标记 interactive 聊天模式（`-i`）为 `[EXPERIMENTAL]`**，或在 P4 完成前移除。

**P1a 验收标准：** 新贡献者只需看 `README.md` 就能完成：clone → `make qwen3-cli` → 下载推荐 GGUF → 运行推理；生成会在结束标记处停止，且质量与已知正确 reference 一致。

---

## 🔮 P2 — 长上下文与内存效率

目标：更高效地支持长上下文。

- [ ] **KV cache 量化（FP8 / INT8）**
  - 降低长序列内存占用。
- [ ] **支持超过 4096 的 context length**
  - 为 prefill 添加 Flash Attention 风格的 tiled attention，避免 O(n²) 内存与计算增长。
  - 优化 seq_len > 8192 时的 decode attention。

**30B 明确不做：** page-based KV cache allocation。30B 模型在 48 层 × 4 KV heads × 128 dim × FP16 × 4096 context 下，KV cache 仅约 384 MB，连续分配完全够用。page-based allocation 在此规模下过度设计。

**准入条件：** P0 和 P1a 完成后。

---

## 🏗️ P3 — 多模型 / 235B 适配（延后）

目标：让引擎不再硬编码模型形状，能够运行 Qwen3-235B-A22B 等变体。

> **范围警告：** 这是一次大规模重构。`DS3_N_LAYER`、`DS3_N_EMBD`、`DS3_N_HEAD`、`DS3_N_HEAD_KV` 等宏遍布 `src/ds3_engine.c`、`src/ds3_metal.m`、`metal/*.metal` 和 `tests/`。把它们改成运行时字段会触及几乎所有文件。

### P3 任务
- [ ] 把编译期模型特性标志改成运行时字段：
  - `has_qk_norm`
  - `has_router_bias`
  - 其他 `#if DS3_HAS_*` / `-DDS3_MODEL_235B` 分支
- [ ] 从 GGUF metadata 读取 `n_layer / n_embd / n_head / n_kv_head / n_ff_exp / n_ff_shared`，替代宏。
- [ ] 根据模型维度动态分配所有 scratch buffer。
- [ ] **用真实 235B 权重验证 shared expert 路径**
  - 30B GGUF 没有 shared expert 张量，因此该路径尚未被真实模型验证。
  - llama.cpp 的 qwen3moe 图当前忽略了可选的 shared-expert 分支；引擎默认与之保持一致（若未来模型或 ground-truth 需要，可用 `DS3_USE_SHARED_EXPERT=1` 开启）。
- [ ] **>32 GB MTLBuffer 的 model-view overlap**
  - 单个 `MTLBuffer` 上限 32 GB；235B 权重需要将 mmap view 拆成多个 buffer 或使用 overlapping view。
- [ ] 验证 235B 模型加载与前向传播。

**准入条件：** 30B 开源发布完成后。

---

## 💬 P4 — 聊天体验与功能增强（低优先级）

目标：让交互式 CLI 更易用。

- [x] ~~改进 chat template，在 assistant prefix 中加入 `<think>\n`。~~
  - 官方 Qwen3 chat template 的 generation prompt 只有 `"assistant\n"`，`<think>\n` 由模型自己生成。强制把它写进 prefix 会改变条件分布，实测效果更差。当前模板已与 GGUF 元数据一致。
- [ ] 添加 repetition penalty。
- [ ] 添加 top-p / top-k 采样。
- [ ] 流式输出。
- [ ] 可选 tool-call 支持。
- [ ] **Speculative decoding（远期）**
  - 小模型 draft + 大模型 verify，以突破当前 decode 速度天花板。
  - 这是已被验证的方向，但需要独立的 draft 模型和 verify kernel。

**准入条件：** P0、P1a 和 P2 初期工作完成后。

---

## 备注

- 优先级顺序刻意保持为：**性能 → 发布 → 规模扩展 → 多模型 → 用户体验**。
- 235B 适配和聊天打磨明确排在 30B 发布之后。
- 随着工作完成，应把条目从本文档移入 `AGENTS.md` 或直接删除。
- **P1a decode 质量修复（NeoX RoPE）：** Qwen3 使用 GPT-NeoX 风格的 RoPE
  `(i, i + head_dim/2)`，而非相邻对 LLaMA 布局。由于 seq_pos=0 时两种布局都是恒等变换，该问题只在多 token decode 中显现。
