# Qwen3 推理引擎：知识补充与学习资料

面向有大数据/数据库内核背景的工程师，为构建 Qwen3 专属 Metal 推理引擎所需的知识体系。
按学习优先级排列，标注与已有经验的映射关系。

---

## 一、Metal GPU 编程（最大缺口，必须先学）

> **预计投入**：2 周
> **你的优势**：mmap/buffer pool、pipeline 调度、内存分层管理可直接迁移

### 1.1 核心概念与数据库类比

| Metal 概念 | 数据库类比 | 在推理引擎中的作用 |
|-----------|-----------|-------------------|
| MTLDevice | 数据库实例 | GPU 设备抽象，所有资源的根 |
| MTLCommandQueue | 连接池 / 工作队列 | 命令提交通道 |
| MTLCommandBuffer | 事务 / WAL batch | 一组 GPU 命令的原子提交单元 |
| MTLComputeCommandEncoder | 查询执行器 | 编码具体的 kernel 调用 |
| MTLComputePipelineState (PSO) | 预编译 query plan | 编译后的 kernel，运行时只切参数 |
| MTLBuffer | buffer pool page | GPU 可访问的内存块 |
| MTLBuffer(bytesNoCopy:) | GPU 直接访问 mmap 的虚拟地址 | GGUF 文件 mmap 后，告诉 Metal "这片地址有效，直接用它"，GPU 通过 UMA 共享访问 |
| Threadgroup (线程组) | 并行扫描的分区 | 一组协作线程，共享 threadgroup memory |
| SIMD group (32 线程) | SIMD 向量化执行 | Apple GPU 的最小调度单位 |
| Threadgroup memory | CPU L1 cache / shared buffer | FlashAttention tiling 的核心 |
| Compute Pipeline | OLAP 执行路径 | 推理引擎只用 compute，不用 render |

### 1.2 学习路径

#### 第一阶段：基础概念（3-4 天）

**1. Apple 官方文档（权威首选）**

- **Compute Processing — 理解 Device → Queue → CommandBuffer → Encoder 调用链**
  - 入门教程：[Performing calculations on a GPU](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
    — 从零搭建完整的 compute pipeline：写 kernel → 找 GPU → 创建 PSO → 建 Queue → 编码 → 提交 → 等待结果
  - 架构总览：[GPU devices and work submission](https://developer.apple.com/documentation/metal/gpu-devices-and-work-submission)
    — Device 选择、Command 组织结构、Thread/Threadgroup 管理
  - 设备选择：[Selecting device objects for compute processing](https://developer.apple.com/documentation/metal/selecting-device-objects-for-compute-processing)
    — `MTLCreateSystemDefaultDevice()`、多 GPU 选择策略

- **Resource Management — 理解 GPU 内存生命周期**
  - 资源协议：[MTLResource](https://developer.apple.com/documentation/metal/mtlresource)
    — `MTLStorageMode`（Shared/Private/Managed）、`MTLCPUCacheMode`、`MTLHazardTrackingMode`
  - 命令结构：[Setting up a command structure](https://developer.apple.com/documentation/metal/setting-up-a-command-structure)
    — CommandQueue → CommandBuffer → CommandEncoder 的完整生命周期
  - 资源堆与围栏：[Metal Programming Guide: Resource Heaps](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/ResourceHeaps/ResourceHeaps.html)
    — `MTLHeap` 子分配、`MTLFence` 资源同步

- **Metal Shading Language (MSL) Specification**
  - 链接：https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
  - 这是写 .metal kernel 文件的语法手册
  - 重点：§2 Data Types（half/float/uint）、§4 Address Spaces（device/threadgroup/constant）、§5 Function Qualifiers（kernel）
  - 用法：不需要通读，作为字典查阅

- **Metal Best Practices Guide**
  - 链接：https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/
  - 重点："Resource Management"（避免不必要的拷贝）、"Command Buffer Management"（批处理）
  - 直接关系：94 层模型的 command buffer 批处理优化

**2. 上手练习（3 个递进的 kernel）**

```
练习 1: 向量加法
  目标：理解 threadgroup size、grid size、thread_position_in_grid
  代码量：kernel ~10 行，host ~50 行
  关键收获：理解"我有 N 个元素，如何分配给 M 个线程"

练习 2: 矩阵乘法 naive 版
  目标：理解 2D dispatch、thread_position_in_threadgroup
  代码量：kernel ~15 行
  关键收获：理解为什么 naive 实现慢（全局内存访问模式差）

练习 3: 矩阵乘法 tiled 版
  目标：理解 threadgroup memory（shared memory）
  代码量：kernel ~40 行
  关键收获：这就是 FlashAttention 的核心思想 —— 数据分块加载到快速存储
  你的类比：和数据库 grace hash join 把数据分 partition 加载到内存一模一样
```

#### 第二阶段：性能模型与 Apple GPU 架构（3-4 天）

**3. WWDC Sessions（Apple 工程师讲解，最实用）**

> **注意**：以下具体年份链接可能需要 Apple Developer 登录或已失效。备用查找方式：在
> [developer.apple.com/videos/](https://developer.apple.com/videos/) 搜索 `Metal compute`、
> `GPU optimization`、`Metal performance`，按最新年份排序。WWDC 2024 之后的内容优先。

- **WWDC 2024: Optimize GPU workloads with Metal**
  - 链接：https://developer.apple.com/videos/play/wwdc2024/10150/
  - 内容：GPU profiling、Timeline 分析、瓶颈定位
  - 直接关系：94 层模型的 kernel dispatch 优化

- **WWDC 2023: Discover Metal for immersive apps**
  - 链接：https://developer.apple.com/videos/play/wwdc2023/10089/
  - 内容：Metal 3 新特性概览

- **WWDC 2022: Scale compute workloads across Apple GPUs**
  - 链接：https://developer.apple.com/videos/play/wwdc2022/10159/
  - 内容：大规模计算任务的组织方式

- **WWDC 2022: Maximize GPU bandwidth for Apple silicon**
  - 链接：https://developer.apple.com/videos/play/wwdc2022/10106/
  - 内容：统一内存架构下的带宽优化
  - 直接关系：推理引擎的核心瓶颈就是内存带宽

**4. Apple GPU 架构理解**

- **Apple GPU Architecture Overview**
  - 链接：https://developer.apple.com/documentation/metal/gpu_devices_and_work_submission
  - 重点：理解 Apple GPU 的 tile-based 架构、ALU/TMU 比例、统一内存

- **M3 Max GPU Specs**（你的开发硬件）
  - 40 GPU cores，带宽 ~400 GB/s
  - 理解：推理是 memory-bound 任务，400 GB/s 带宽决定了理论最高 token 速度

#### 第三阶段：精读 ds4 的 Metal 代码（3-4 天）

**5. ds4_metal.m 精读指南（14,524 行）**

```
必读部分（按顺序）：

1. 初始化流程（~前 200 行）
   - g_device, g_queue, g_library 的创建
   - Pipeline State Object 的编译
   → 理解 Metal 运行时的生命周期

2. MTLBuffer(bytesNoCopy:) 的使用
   - 搜索 "bytesNoCopy"
   → 理解 mmap + GPU 零拷贝的实现方式
   → 你的类比：和数据库的 direct I/O + page cache bypass 类似

3. Command buffer 批处理逻辑
   - 搜索 "g_batch_cb", "g_batch_enc"
   → 理解 ds4 如何把多个 kernel 调用打包进一个 command buffer
   → 直接关系：94 层模型需要更激进的批处理

4. Kernel 调用 wrapper 函数
   - 搜索具体 kernel 名如 "matvec", "rope", "rmsnorm"
   → 理解 host 端如何设置参数、dispatch 线程
```

**6. metal/moe.metal 精读指南（1,737 行）**

```
这是最关键的 kernel 文件，其量化反量化逻辑可供 Qwen3 引擎参考，但 kernel 主体需重写。

关键结构：
- QK_K = 256：量化 block 大小
- block_q2_K 结构：理解 2-bit 量化的内存布局
- kernel 函数签名：理解 [[thread_position_in_grid]] 等属性

重点函数（搜索 kernel void）：
- MoE matvec 主函数：看 top-k 选择 + 稀疏 dispatch 逻辑
- IQ2_XXS 反量化：看码本查找 + 符号恢复
- Q2_K 反量化：看 scale/min 解包 + 2-bit 恢复

改造要点（30B → 235B 时）：
- top-k: 6 → 8（改常量）
- gate 函数逻辑（可能不同）
- expert FFN 维度（2048 → 768/1536）
```

### 1.3 参考项目

| 项目 | 链接 | 价值 |
|------|------|------|
| metal-cpp samples | https://developer.apple.com/metal/sample-code/ | Apple 官方 Metal 示例代码 |
| MLX | https://github.com/ml-explore/mlx | Apple 自己的 ML 框架，Metal kernel 实现可参考 |
| llama.cpp ggml-metal | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ | Metal kernel 的生产级实现 |

---

## 二、Transformer 推理架构（中等缺口）

> **预计投入**：1 周
> **你的优势**：pipeline 执行、DAG 调度、内存管理直接适用
> **注意**：只需理解推理（前向计算），不需要理解训练（反向传播）

### 2.1 整体计算图

```
一次 token 生成的完整计算流程：

Input Token IDs
    ↓
Embedding Lookup        ← 查表，trivial
    ↓
┌─────────────────────────────────────────┐
│  Transformer Block × 94 (或 48 for 30B)  │
│                                         │
│  RMSNorm                                │  ← 归一化，trivial
│     ↓                                   │
│  GQA Self-Attention                     │  ← 核心 1：Q/K/V 投影 + 注意力 + 输出投影
│     ↓                                   │
│  RMSNorm                                │  ← 归一化
│     ↓                                   │
│  MoE FFN                                │  ← 核心 2：路由 + top-8 专家 + 共享专家
│     ↓                                   │
│  Residual Add                           │  ← 残差连接
└─────────────────────────────────────────┘
    ↓
RMSNorm
    ↓
LM Head (线性投影)      ← 映射到 vocab_size 维度
    ↓
Logits → Sampling → Next Token
```

#### 关键补充：Causal Mask（因果掩码）

上述流程图省略了一个核心机制：**decode 时每个 token 只能看到自己和之前的 token，不能看到未来 token**。

- 实现方式：在 `Q @ K^T / sqrt(d)` 之后，将"未来位置"的分数设为 `-inf`（softmax 后变为 0）
- Metal kernel 中的实现：FlashAttention 在 tile 计算时内嵌因果掩码，**不构造 N×N 矩阵**
- 你的类比：数据库的 **range filter** —— 只扫描 `position <= current` 的行
- 预填充（prefill）阶段：可以一次性处理 prompt 中所有 token，但每个 token 仍然只能看到前面的 token
- 解码（decode）阶段：生成新 token 时，Q 只有 1 个向量，K/V 是全部历史，因果掩码退化为"取全部历史"

### 2.2 Self-Attention + GQA

#### 必读论文

**1. Attention Is All You Need (2017)**
- 链接：https://arxiv.org/abs/1706.03762
- 只需读 §3.2 "Scaled Dot-Product Attention" 和 §3.2.2 "Multi-Head Attention"
- 核心公式：`Attention(Q,K,V) = softmax(QK^T / √d_k) × V`
- 你的类比：
  - Q = 查询条件
  - K = 索引键
  - V = 数据值
  - `QK^T` = 全表扫描算相关性分数
  - `softmax` = 归一化为概率分布
  - `× V` = 加权聚合

**2. GQA: Training Generalized Multi-Query Transformer (2023)**
- 链接：https://arxiv.org/abs/2305.13245
- Qwen3 用的就是 GQA：4 个 KV heads 被 32 个 Q heads 共享（235B），或 16 个（30B）
- 关键收益：KV cache 体积缩小 8 倍（32/4）或 4 倍（16/4）
- 你的类比：broadcast join —— 4 个小维度表被 broadcast 到 N 个大分区做 join

**3. Multi-Query Attention (MQA) — 原始版本 (2019)**
- 链接：https://arxiv.org/abs/1911.02150
- GQA 是 MQA 的推广，了解演化关系

#### 推荐教程

- **Jay Alammar: The Illustrated Transformer**
  - 链接：https://jalammar.github.io/illustrated-transformer/
  - 全网最好的可视化 Transformer 教程，用图解释每一步
  - 建议：先看这个建立直觉，再看论文

- **Lilian Weng: The Transformer Family v2**
  - 链接：https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
  - 全面综述，覆盖 MHA → MQA → GQA 的演化

#### 动手练习

```python
# 用 PyTorch 手写 GQA forward（~50 行），以 30B 为例
# 输入：随机 Q(batch, 16, seq, 128), K(batch, 4, seq, 128), V(batch, 4, seq, 128)
# 步骤：
#   1. K, V repeat_interleave 4 次（4 × 4 = 16 Q heads）
#   2. scores = Q @ K.transpose(-2,-1) / sqrt(128)
#   3. scores = scores.masked_fill(causal_mask, -inf)
#   4. attn = softmax(scores, dim=-1)
#   5. output = attn @ V
# 提示：235B 的 Q heads 数为 32，K/V repeat_interleave 8 次，其余逻辑完全相同
# 对比 transformers 库的输出，验证数值一致
```

### 2.3 FlashAttention

#### 必读论文

**1. FlashAttention: Fast and Memory-Efficient Exact Attention (2022)**
- 链接：https://arxiv.org/abs/2205.14135
- 核心思想：不显式构建 N×N 的注意力矩阵，而是分 tile 在 SRAM 中计算
- 你的类比：**grace hash join** —— 数据太大放不进内存，就分 partition，每个 partition 在快速存储（threadgroup memory）里完成计算
- 重点读：§3 Algorithm（分 tile 的具体策略）、Figure 1（内存层次对比）

**2. FlashAttention-2 (2023)**
- 链接：https://arxiv.org/abs/2307.08691
- 优化了线程分工和并行度
- 作为实现参考

**3. Flash-Decoding (2023)**
- 链接：https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- 解码阶段（生成单 token）的优化版本
- 直接关系：推理引擎 90% 的时间在 decode 阶段

#### 推荐教程

- **Tri Dao 的 FlashAttention 讲解视频**
  - 搜索 "Tri Dao FlashAttention talk" 在 YouTube
  - 作者亲自讲解，最权威

- **ELI5: FlashAttention**
  - 链接：https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
  - 通俗易懂的解释

### 2.4 MoE 路由（Mixture of Experts）

#### 必读论文

**1. Switch Transformers (2022)**
- 链接：https://arxiv.org/abs/2101.03961
- MoE 的基础概念：gate network、routing、load balancing
- 你的类比：**一致性哈希 + 负载均衡** —— gate 网络计算 128 个专家的分数，选 top-8，按权重做加权 scatter-gather

**2. Mixtral of Experts (2024)**
- 链接：https://arxiv.org/abs/2401.04088
- 和 Qwen3 最相似的 MoE 实现（top-K 路由）
- 重点读：§2.1 Sparse Mixture of Experts

**3. DeepSeekMoE: Towards Ultimate Expert Specialization (2024)**
- 链接：https://arxiv.org/abs/2401.06066
- 理解共享专家（shared expert）的设计动机
- 直接关系：Qwen3 也有共享专家，其 FFN 走 Q4 而非 Q2

#### MoE 的计算流程（Qwen3 具体实现）

```
输入: hidden_states [batch, hidden_size=4096]

1. Gate 计算:
   gate_logits = hidden_states @ gate_weight    # [batch, 128]
   top8_indices, top8_weights = topk(gate_logits, k=8)
   top8_weights = softmax(top8_weights)         # 归一化

2. 路由专家计算（稀疏，只算 8 个）:
   for i in top8_indices:
       expert_out_i = SwiGLU(hidden_states, expert[i])
       # SwiGLU: up = x @ W_up, gate = x @ W_gate
       #         out = (silu(gate) * up) @ W_down

3. 加权求和:
   routed_output = sum(top8_weights[i] * expert_out_i)

4. 共享专家:
   shared_output = SwiGLU(hidden_states, shared_expert)

5. 最终输出:
   output = routed_output + shared_output
```

#### 必读论文补充

**4. GLU Variants Improve Transformer (SwiGLU)**
- 链接：https://arxiv.org/abs/2002.05202
- 核心：为什么 `silu(gate) * up` 比 ReLU/GELU 更好
- 直接关系：Qwen3 的 FFN（包括共享专家和路由专家）都使用 SwiGLU

### 2.5 RoPE 位置编码

#### 必读论文

**1. RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)**
- 链接：https://arxiv.org/abs/2104.09864
- 核心思想：对 Q/K 向量的每对维度做旋转，角度由位置决定
- 数学本质：复数乘法。`(q_0 + i*q_1) × (cos(mθ) + i*sin(mθ))`
- Qwen3 使用 **GPT-NeoX 布局**：旋转对为 `(i, i + head_dim/2)`，而非相邻对 `(i, i+1)`
- `rope_theta = 1,000,000`：频率基底，值越大支持越长的上下文

**2. YaRN: Efficient Context Window Extension (2023)**
- 链接：https://arxiv.org/abs/2309.00071
- 通过修改 RoPE 的频率参数实现上下文窗口扩展
- 直接关系：虽然 Qwen3 原生支持 128K 不需要 YaRN，但 kernel 中预留参数位以备将来

#### 推荐教程

- **Eleuther AI: Rotary Embeddings — A Relative Revolution**
  - 链接：https://blog.eleuther.ai/rotary-embeddings/
  - 最好的 RoPE 数学推导 + 直觉解释

#### RoPE 的具体计算（Qwen3）

```python
# rope_theta = 1_000_000, head_dim = 128, n_rot = 128（Qwen3 全头旋转）

# 频率计算
dim = 128
freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2) / dim))
# freqs 是 64 个频率值，从高频到低频

# 对位置 pos 的旋转
angles = pos * freqs  # [64] 个角度
cos_angles = cos(angles)
sin_angles = sin(angles)

# 对 Q/K 向量旋转（GPT-NeoX 布局：半区对 (i, i + dim/2)）
n_pairs = dim // 2
q_rotated[0:n_pairs]         = q[0:n_pairs] * cos_angles - q[n_pairs:dim] * sin_angles
q_rotated[n_pairs:dim]       = q[0:n_pairs] * sin_angles + q[n_pairs:dim] * cos_angles

# 注意：若 n_rot < head_dim（如 DeepSeek V4 的 MLA 只旋转前 64 维），
#       则只旋转前 n_rot 维，剩余维度保持不变。Qwen3 通常 n_rot=head_dim。

# GQA 实现注意事项：
# Q 有 num_heads 个（30B:16, 235B:32），K 只有 kv_heads 个（4）
# Metal kernel 中 Q 和 K 的 RoPE 需要不同的 dispatch size：
#   Q: dispatch [num_heads, seq_len]
#   K: dispatch [kv_heads, seq_len]
# 两者不可合并为一个 kernel 调用，否则线程分配效率低
```

### 2.6 RMSNorm

最简单的部分，几行代码：

```python
def rmsnorm(x, weight, eps=1e-6):
    rms = sqrt(mean(x ** 2) + eps)
    return (x / rms) * weight
```

- 论文：https://arxiv.org/abs/1910.07467
- 比 LayerNorm 少一个 mean 减法，计算更快
- Metal kernel：一个 reduce（求 mean(x²)）+ 一个 element-wise 操作

### 2.7 综合学习资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **3Blue1Brown: Attention in transformers** | https://www.youtube.com/watch?v=eMlx5fFNoYc | 最直观的可视化讲解 |
| **Andrej Karpathy: Let's build GPT from scratch** | https://www.youtube.com/watch?v=kCc8FmEb1nY | 2 小时从零实现 GPT，代码级理解 |
| **Andrej Karpathy: llm.c** | https://github.com/karpathy/llm.c | 纯 C 实现的 LLM 训练，代码风格与 ds4.c 相似 |
| **Jay Mody: picoGPT** | https://github.com/jaymody/picoGPT | 60 行 NumPy 实现 GPT-2 推理，理解前向计算最快的方式 |
| **Umar Jamil: Transformer 系列视频** | YouTube 搜索 "Umar Jamil transformer" | 逐行代码讲解 attention、RoPE、MoE |

### 2.8 关于 Tensor Parallelism（为什么不引入）

本引擎是单 Mac 上运行，只有一个 GPU（统一内存架构中没有独立的"多卡"概念，
Apple GPU 的多个 core 共享同一地址空间）。因此：

- **不引入 TP**：没有多余的 GPU 可用，跨多个 Mac 的分布式推理是另一个问题的范畴
- **batch 大小恒为 1**：decode 阶段 token-by-token 生成，没有并发请求的 batch 机会
- **唯一的并行维度在 GPU 内部**：threadgroup 内的几百个线程并行处理同一个 token 的 attention/MoE 计算

这意味着所有优化都是单 batch、单设备场景下的 kernel 效率优化，而非分布式通信优化。

---

## 三、量化理论与实践（小缺口，与压缩经验高度重叠）

> **预计投入**：1 周专门学习（与 Transformer 章节并行）
> **你的优势**：列存压缩（RLE/dict/delta/bitpacking）的经验直接迁移

### 3.1 核心概念映射

| 量化术语 | 数据库/压缩类比 | 说明 |
|----------|-----------------|------|
| block_q2_K (256 元素/block) | 列存 mini-page | 每个 block 有自己的 min/scale 元信息 |
| IQ2_XXS（importance matrix） | 基于数据分布的字典编码 | 用码本（codebook）做查表反量化 |
| 非对称量化 | 冷热数据分层压缩 | 路由专家 Q2（冷），共享层 Q4/FP16（热） |
| Calibration data | 压缩算法的统计采样 | 用真实数据估计量化参数 |
| Perplexity | 有损压缩的质量度量 | 类似解压后的 PSNR |
| Round-to-nearest (RTN) | 最近邻量化 | 最简单，质量最差 |
| GPTQ | 基于 Hessian 的自适应量化 | 考虑参数重要性，逐列补偿误差 |

### 3.2 必读资料

**1. GPTQ: Accurate Post-Training Quantization (2023)**
- 链接：https://arxiv.org/abs/2210.17323
- 核心思想：逐层量化时，用 Hessian 矩阵估计每个权重的重要性，量化一列后调整剩余列以补偿误差
- 你的类比：类似数据库的增量统计更新 —— 改一个值后调整相关统计信息
- 重点读：§3 Algorithm

**2. AWQ: Activation-aware Weight Quantization (2024)**
- 链接：https://arxiv.org/abs/2306.00978
- 核心思想：不是所有权重同等重要，根据激活值的大小决定保护哪些权重
- 直接关系：非对称量化的理论基础 —— 共享层的权重更"重要"，所以保持高精度

**3. QuIP#: Even Better LLM Quantization with Hadamard Incoherence (2024)**
- 链接：https://arxiv.org/abs/2402.04396
- 2-bit 量化的前沿方法，理解 2-bit 的理论极限

### 3.3 llama.cpp 的量化格式（实操必读）

**源码文件**：`ggml/src/ggml-quants.c`（llama.cpp 仓库）

```c
// block_q2_K 结构（你需要精确理解的）
typedef struct {
    uint8_t scales[QK_K/16];   // 16 个 4-bit scale 值
    uint8_t qs[QK_K/4];        // 256 个 2-bit 量化值（64 字节）
    half d;                     // super-block scale
    half dmin;                  // super-block minimum
} block_q2_K;
// 总大小：16 + 64 + 2 + 2 = 84 字节 / 256 个权重 = 2.625 bpw

// IQ2_XXS 结构（码本量化）
typedef struct {
    half d;                     // scale
    uint16_t qs[QK_K/8];       // 码本索引 + 符号位
} block_iq2_xxs;
// 总大小：2 + 64 = 66 字节 / 256 个权重 = 2.0625 bpw
```

- 仓库链接：https://github.com/ggml-org/llama.cpp
- 重点文件：`ggml/src/ggml-quants.c`（量化/反量化）、`ggml/src/ggml-quants.h`（结构定义）
- 目标：理解 `quantize_row_q2_K` 和 `dequantize_row_q2_K` 的完整流程

**4. llama.cpp 内存分配策略（补充必读）**
- **ggml-alloc**：`ggml_gallocr` 如何分析计算图生命周期、重用临时 buffer
  - 链接：`ggml/src/ggml-alloc.c`
  - 直接关系：94 层模型的 scratch buffer 池化策略
- **ggml-backend**：host/device 内存异步调度、多后端抽象
  - 链接：`ggml/src/ggml-backend.c`
  - 参考价值：理解为什么 ds4.c 要自己管理 Metal buffer 而不是用通用 backend

### 3.4 动手练习

```python
import numpy as np

# 练习：手写 block_q2_K 量化/反量化
def quantize_q2_k_block(data_256):
    """将 256 个 float32 值量化为 block_q2_K 格式"""
    # 分成 16 个 sub-block，每个 16 个值
    # 每个 sub-block 有自己的 scale 和 min
    # 值被映射到 0,1,2,3 四个 level
    ...

def dequantize_q2_k_block(block):
    """从 block_q2_K 恢复为 float32"""
    # val = (scale * quant + min) * super_scale
    ...

# 验证：随机矩阵 → 量化 → 反量化 → 计算 MSE
original = np.random.randn(4096, 1536).astype(np.float32)  # 一个 Qwen3 专家的权重
quantized = quantize_block(original)
recovered = dequantize_block(quantized)
mse = np.mean((original - recovered) ** 2)
print(f"MSE: {mse:.6f}, relative error: {np.sqrt(mse) / np.std(original):.4f}")
```

### 3.5 KV Cache 量化（128K 上下文必需）

> **预计投入**：与 Metal kernel 开发同步进行，2-3 天理解原理

#### 为什么必需

| 配置 | FP16 KV | Q8_0 KV | Q4_K KV |
|------|---------|---------|---------|
| 235B @ 128K | **25.2 GB** | 12.6 GB | 6.3 GB |
| 30B @ 128K | **12.9 GB** | 6.45 GB | 3.22 GB |

128GB Mac 跑 235B 非对称 Q2 的完整内存分解：
```
模型权重 (非对称 Q2):  ~85 GB
FP16 KV Cache @128K:   ~25 GB
推理 scratch buffer:   ~8 GB   (每层激活、中间张量、量化缓冲)
macOS + 其他:          ~5 GB
─────────────────────────────────
合计:                  ~123 GB / 128 GB  ← 仅余 5GB，极度紧张
```
**KV cache 量化不是优化，是门槛。** Q8_0 KV 可省 12.6GB，给 scratch 和 OS 留足余量。

#### 技术方案

**1. Per-head 动态缩放（推荐）**
- 每个 attention head（128-dim）维护自己的 scale/zero_point
- 粒度：每 64 或 128 个 token 一个 block
- 写放大：每次 append 新 token 时，可能需要重新量化尾部 block

**2. 与权重量化的关键区别**

| | 权重量化 | KV cache 量化 |
|--|---------|---------------|
| 数据分布 | 静态，训练后固定 | 动态，随输入变化 |
| scale 计算 | 离线（imatrix/GPTQ） | 在线（每 token/每 block） |
| 精度要求 | 可接受 2-3 bit | 通常需要 ≥ 4 bit（Q4_K）或 ≥ 8 bit（Q8_0） |
| kernel 复杂度 | 低（固定 scale） | 中（动态 scale 加载） |

**3. 实现参考**
- llama.cpp 的 `kv_cache_quant` 相关代码（搜索 `quantize_kv`）
- ds4.c 的 `dsv4_fp8_kv_quantize`（FP8 在线量化思路可直接迁移到 Q8_0）
- Metal kernel 中：在 `store_raw_kv` 阶段插入量化，在 `attention_decode` 阶段插入反量化

#### 决策树

```
128GB Mac 跑 235B：
  ├─ 目标 128K 上下文 → 必须 KV Q8_0 或 Q4_K
  ├─ 目标 64K 上下文  → KV Q8_0 推荐，FP16 可接受但紧张
  └─ 目标 32K 上下文  → FP16 可接受

512GB Mac Studio：
  └─ 128K 上下文 → FP16 可接受，但 Q8_0 仍推荐（为 disk KV 预留内存）
```

---

## 四、GGUF 文件格式（最小缺口）

> **预计投入**：1-2 小时
> **你的优势**：数据库文件格式设计经验让这部分几乎无障碍

### 4.1 格式概述

```
GGUF 文件结构：

┌─────────────────────────┐
│ Header                  │  magic ("GGUF") + version + counts
├─────────────────────────┤
│ Metadata KV pairs       │  模型参数、tokenizer 配置等
│  - architecture: "qwen3moe"
│  - num_hidden_layers: 48
│  - ...                  │
├─────────────────────────┤
│ Tensor Descriptors      │  每个张量的 name + shape + type + offset
│  - blk.0.attn_q.weight  │
│  - blk.0.ffn_gate.0.weight  (专家 0)
│  - ...                  │
├─────────────────────────┤
│ Alignment Padding       │  对齐到指定边界（通常 32 或 page size）
├─────────────────────────┤
│ Tensor Data             │  所有张量的二进制数据，按描述符顺序
│  （可 mmap，支持零拷贝）│
└─────────────────────────┘
```

你的类比：这就是一个简化版的数据库 data file —— header + schema（tensor descriptors）+ data pages（tensor data），对齐到页边界支持 mmap。

### 4.2 必读资料

**1. GGUF 官方规范**
- 链接：https://github.com/ggml-org/gguf/blob/main/spec.md
- 核心就几页，数据类型定义 + KV 编码规则 + 张量描述符格式
- 1 小时可读完

**2. ds4.c 的 GGUF 加载代码**
- 在 ds4.c 中搜索 `gguf`
- 看 mmap → 解析 header → 根据 tensor descriptor 算 offset → MTLBuffer(bytesNoCopy:) 绑定
- 这是最好的实战参考

**3. llama.cpp convert 脚本**
- 链接：https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
- Python 实现的 HuggingFace → GGUF 转换
- 你的量化脚本可以基于此修改

### 4.3 关键实现细节

```
非对称量化 GGUF 的制作流程：

1. 加载 HF 模型的 safetensors 权重
2. 遍历每个张量，根据名称决定量化策略：
   - 名称包含 "experts" + ("up" 或 "gate") → IQ2_XXS
   - 名称包含 "experts" + "down"             → Q2_K
   - 名称包含 "shared_expert"                → Q4_K
   - 其他（attention, embedding, norm）       → FP16
3. 对每个张量执行对应的量化函数
4. 写入 GGUF 文件（header + metadata + descriptors + data）
5. 确保 data 段对齐到 page boundary（4096 字节）
```

---

## 五、补充知识：Tokenizer（BPE）

> **预计投入**：半天
> **直接关系**：Qwen3 使用 BPE tokenizer，需要正确实现编码/解码

### 5.1 必读资料

- **BPE 算法原理**
  - 链接：https://huggingface.co/learn/nlp-course/chapter6/5
  - HuggingFace 的 NLP 课程，清晰讲解 BPE 的 merge 规则

- **Qwen3 的 tokenizer**
  - vocab_size: 151936
  - 类型：BPE with special tokens
  - 文件：`tokenizer.json`（merge 规则）+ `tokenizer_config.json`（特殊 token）
  - 实现选择：Qwen3 使用 HuggingFace `tokenizer.json` 格式的 BPE，应手写 BPE encode/decode（参考 ds4.c 约 500 行 C），或链接 `tokenizers` C++ 库；**不要用 sentencepiece，Qwen3 不兼容**

---

## 六、推荐学习顺序总结

```
Week -2 ~ -1: Metal 编程
  ├── Day 1-2: 读 Apple 文档 + 写向量加法 kernel
  ├── Day 3-4: 写 tiled 矩阵乘法 + 看 WWDC 视频
  ├── Day 5-7: 精读 ds4_metal.m + moe.metal
  └── Day 8-10: 精读 ds4.c 的 GGUF 加载 + 计算图调度

Week -1 ~ 0: Transformer 理解
  ├── Day 1: 看 Illustrated Transformer + 3Blue1Brown 视频
  ├── Day 2: 看 Karpathy 的 GPT from scratch 视频
  ├── Day 3: 读 GQA 论文 + 手写 GQA forward
  ├── Day 4: 读 Switch Transformers + Mixtral 论文
  └── Day 5: 读 FlashAttention 论文 + RoPE 博客

Phase 1 Week 1: 量化（专门学习）
  └── 精读 llama.cpp quants.c + 手写 block_q2_K / IQ2_XXS 反量化

后续: GGUF、YaRN、tokenizer 按需查阅
```

---

## 七、工具链

开发过程中需要的软件工具：

| 工具 | 用途 | 安装方式 |
|------|------|---------|
| Xcode + Metal tools | Metal 开发 + GPU profiling | Mac App Store |
| Metal System Trace | GPU 性能分析 | Xcode Instruments 内置 |
| Python 3.10+ | 量化脚本、reference 实现 | brew install python |
| PyTorch | 模型加载、reference forward | pip install torch |
| transformers | HuggingFace 模型加载 | pip install transformers |
| safetensors | 权重文件读取 | pip install safetensors |
| numpy | 数值计算 | pip install numpy |
| huggingface-cli | 模型下载 | pip install huggingface_hub |
| vmmap / memory_pressure | macOS 内存监控 | 系统内置 |
| llama.cpp | 对比基准 | git clone + make |
| `cmake` | 编译 llama.cpp 和 C++ 项目 | `brew install cmake` |
| `git-lfs` | 下载 HuggingFace 大文件（很多 repo 用 LFS） | `brew install git-lfs` |
| Xcode Metal Frame Capture | 逐 kernel 调试 GPU 状态、抓帧分析 | Xcode → Debug → Capture GPU Frame |
| Metal System Trace | GPU 性能 timeline 分析 | Xcode Instruments 内置 |

---

## 八、速查卡片

开发过程中高频查阅的公式和参数：

### Qwen3 关键参数速查

```
30B-A3B:   layers=48,  hidden=2048,  heads=16, kv_heads=4, experts=128, top_k=8,
           expert_ffn=768,  shared_ffn=4096,   head_dim=128, n_rot=128(需确认)

235B-A22B: layers=94,  hidden=4096,  heads=32, kv_heads=4, experts=128, top_k=8,
           expert_ffn=1536, shared_ffn=12288,  head_dim=128, n_rot=128(需确认)

RoPE: GPT-NeoX 布局，rope_theta=1_000_000, 全头旋转 n_rot=head_dim=128
  Q dispatch=[num_heads, seq], K dispatch=[kv_heads, seq]  — 两者维度不同，分开 dispatch
  kernel 中预留 freq_base/freq_scale/ext_factor 参数位以备未来上下文扩展

Attention 实现须知（需从 config.json 确认）:
  □ QK norm (qk_layernorm): 如果为 true，attention kernel 中 Q/K 投影后需额外 RMSNorm
  □ attention_bias: 如果为 true，attention 有 bias 项
  □ 这两个参数直接影响 kernel 实现的正确性，启动前必须确认
```

### 内存计算公式

```
KV 缓存大小 = ctx_len × head_dim × kv_heads × 2(K+V) × 2(FP16 bytes) × num_layers
  30B @ 32K:   32768 × 128 × 4 × 2 × 2 × 48  = 3.22 GB
  30B @ 64K:   65536 × 128 × 4 × 2 × 2 × 48  = 6.44 GB
  30B @128K:  131072 × 128 × 4 × 2 × 2 × 48  = 12.9 GB
  235B @ 64K:   65536 × 128 × 4 × 2 × 2 × 94  = 12.6 GB
  235B @100K:  102400 × 128 × 4 × 2 × 2 × 94  = 19.7 GB
  235B @128K:  131072 × 128 × 4 × 2 × 2 × 94  = 25.2 GB
```

### 量化体积估算

```
FP16:    params × 2 bytes
Q4_K:    params × 4.5 / 8 bytes  (≈ 0.5625 bytes/param)
Q2_K:    params × 2.625 / 8 bytes (≈ 0.328 bytes/param)
IQ2_XXS: params × 2.0625 / 8 bytes (≈ 0.258 bytes/param)
```

### 速度估算

**唯一可得的 Metal 推理速度校准点**（ds4.c README 实测）

```
ds4.c DeepSeek V4 Flash (22B active, 43 层) on M3 Max 128GB, q2, short prompt:
  prefill: 58.5 t/s, decode: 26.7 t/s
```

这是唯一在同一平台（M3 Max）上、用类似策略（非对称 Q2 MoE）跑 Metal 推理的公开数据。
以下所有估算都以这个校准点为基础。

- 30B-A3B 只有 3B active（约 1/7），decode 速度预估 **100-180 t/s**（短上下文）
- 235B-A22B 有 22B active（与 DS4 相当），但 94 层是 DS4 的 2.2 倍，decode 速度预估 **10-15 t/s**

**短上下文（<4K，权重带宽主导）**
```
理论最高 token/s ≈ 内存带宽 / (活跃参数 × bytes_per_param)
  M3 Max (400 GB/s), 30B-A3B Q2: 400 / (3B × 0.33) ≈ 400 t/s (理论上限)
  M3 Max (400 GB/s), 30B-A3B Q4: 400 / (3B × 0.56) ≈ 238 t/s (理论上限)
  实际通常为理论值的 30-50%（kernel 效率、dispatch 开销）
  以 DS4 校准：理论 400/(22×0.33)≈55，实测 27，利用率 ~49%
```

**长上下文（>32K，KV cache 带宽主导）**
```
每次 decode 需读取全部历史 K + V：
  kv_read = ctx_len × head_dim × kv_heads × 2(K+V) × 2(bytes) × layers

例：30B @ 64K
  kv_read = 64,000 × 128 × 4 × 2 × 2 × 48 = 12.6 GB per token
  tps_bound ≈ 400 / 12.6 ≈ 32 t/s（理论上限）

缓解：KV cache Q8_0 → 读取量减半 → ~60 t/s
      KV cache Q4_K → 读取量再减半 → ~120 t/s
```
