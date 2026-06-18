# Qwen3-30B-A3B Perplexity 验证实验计划 V2

> 本文档基于 [第一轮实验报告](qwen3_30b_perplexity_validation.md) 的反馈与缺陷，设计两条补充实验路线，以验证核心假设：**非对称量化能让 235B-A22B 跑在 128GB Mac 上**。

---

## 第一轮实验的问题清单

| # | 问题 | 严重程度 | 说明 |
|---|------|---------|------|
| 1 | **未验证核心假设** | 🔴 严重 | 测的是 llama.cpp `uniform Q2_K`，不是"expert-only 非对称"策略 |
| 2 | **Q8→Q2 再量化不公平** | 🟡 中等 | 叠加两次量化误差，PPL 数据不严格；但作为**趋势参考**仍有价值 |
| 3 | **内存估算错误** | 🔴 严重 | 235B 全 Q4 应为 **~132GB**（非 85GB），128GB **装不下** |

### 修正后的关键结论

> **非对称量化不是"优化选项"，而是 235B on 128GB 的刚需。**
>
> - 235B Q4_K_M 权重 ≈ 132GB + KV cache ≈ 25GB = **157GB** ❌
> - 235B 非对称 Q2 权重 ≈ 85GB + KV cache ≈ 25GB = **110GB** ✅

---

## 实验设计总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         两条路线，先 A 后 B                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  路线 A: imatrix + IQ2_XXS          路线 B: 非对称量化原型                    │
│  ─────────────────────────          ────────────────────                     │
│  时间: ~30 分钟                     时间: ~30-60 分钟                        │
│  成本: 零开发                       成本: 一个 shell 脚本                    │
│  价值: 建立"聪明 2-bit"基线          价值: 验证核心假设                       │
│                                                                             │
│      ↓                                       ↓                              │
│  三点趋势线:                                                                  │
│  Q2_K (naive)  ──→  IQ2_XXS (smart)  ──→  Asymmetric (expert-only)         │
│    +20.56%            ???                    ???                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 为什么先 A 后 B

### 路线 A 的价值（30 分钟，零开发）

| 理由 | 说明 |
|------|------|
| 低成本高信息 | 30 分钟得到一个 IQ2_XXS 的 PPL 数据点 |
| 建立基线 | 形成"naive → smart → asymmetric"三点趋势线 |
| 决策信号 | 如果 IQ2_XXS 已降到 +10% 以内 → 非对称方案信心大增 |
| 止损信号 | 如果 IQ2_XXS 仍 +20%+ → 2-bit 本身有硬上限，路线 B 需更保守 |

### 路线 B 的价值（核心验证）

| 理由 | 说明 |
|------|------|
| 验证核心假设 | "只压 expert 权重，其余保高精度"是否可行 |
| 关键发现 | `llama-quantize --tensor-type name:type` 把"写 Python 2-4 小时"变成"一行命令 10 分钟" |
| 项目基石 | 路线 B 通过 → 235B 故事几乎板上钉钉 |

---

## 实验环境

延续第一轮，无需重新编译：

| 项目 | 配置 |
|------|------|
| 硬件 | Apple M3 Max, 48GB RAM |
| llama.cpp | `/Volumes/ExtremeSSD/qwen3-engine/llama.cpp`（已编译，Metal 启用）|
| 输入模型 | Q8_0 (30.3 GB) — 虽然再量化有误差，但作为统一输入可接受 |
| 测试数据 | `data/test.txt` (148 KB) — 同一文本保证可比性 |
| 已有基准 | Q8_0 PPL=5.5237, Q4_K_M PPL=5.6774, Q2_K PPL=6.6591 |

---

## 路线 A：imatrix + IQ2_XXS

### 目的
验证"聪明的 uniform 2-bit 量化"能挽回多少质量。

### 原理

- **imatrix（Importance Matrix）**：通过跑校准数据，分析每个 weight tensor 对模型输出的敏感度
- **IQ2_XXS**：llama.cpp 的"super-block 2-bit"量化，用 imatrix 指导分配精度，比 naive Q2_K 更精细

### 执行步骤

> ⚠️ **数据隔离原则**：imatrix 校准数据与 PPL 评估数据**不能相同**，否则结果会偏乐观。

```bash
# Step A0: 准备独立的 PPL 评估数据（与校准数据不同）
# 用 ds4 的 README.md 作为评估集（与 test.txt 内容完全不同）
cp /Users/wangyang/gitclonefiles/ds4/README.md \
   /Volumes/ExtremeSSD/qwen3-engine/data/eval.txt

# Step A1: 生成 imatrix（校准数据用 test.txt）
cd /Volumes/ExtremeSSD/qwen3-engine/llama.cpp
./build/bin/llama-imatrix \
  -m /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf \
  -f /Volumes/ExtremeSSD/qwen3-engine/data/test.txt \
  -o /Volumes/ExtremeSSD/qwen3-engine/data/imatrix.dat \
  --threads 16

# Step A2: 用 imatrix 做 IQ2_XXS 量化
./build/bin/llama-quantize \
  --allow-requantize \
  --imatrix /Volumes/ExtremeSSD/qwen3-engine/data/imatrix.dat \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-IQ2_XXS.gguf \
  IQ2_XXS

# Step A3: 跑 perplexity（评估数据用 eval.txt，与校准数据隔离）
./build/bin/llama-perplexity \
  -m /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-IQ2_XXS.gguf \
  -f /Volumes/ExtremeSSD/qwen3-engine/data/eval.txt \
  -ngl 999 \
  > /Volumes/ExtremeSSD/qwen3-engine/data/ppl_iq2_xxs.log 2>&1
```

> 📌 如果无法获取独立评估数据，必须在报告中标注：**"校准数据与评估数据相同，PPL 可能偏低"**。

### 预期结果

> 注意：以下劣化是**相对 Q8_0 在 eval.txt 上的 PPL**，而非第一轮 test.txt 的结果。由于 eval.txt 与 test.txt 内容不同，绝对 PPL 值会有差异，但**相对比较**（Q8_0 vs IQ2_XXS 在同一 eval.txt 上）仍然有效。

| 情景 | PPL 劣化 | 解读 |
|------|---------|------|
| 🟢 理想 | < +10% | "聪明 2-bit"空间很大，非对称方案信心足 |
| 🟡 中等 | +10~15% | 2-bit 有硬上限，非对称必须精心设计 |
| 🔴 悲观 | > +15% | 30B 对 2-bit 太敏感，需考虑 expert Q3/Q4 |

---

## 路线 B：非对称量化原型

### 目的
验证核心假设："MoE expert 权重用 Q2，其余保高精度"。

### 关键发现

`llama-quantize` 支持 **`--tensor-type name:type`**，可以按 tensor 名称指定精度：

```bash
# 默认 Q8_0，对特定 tensor 覆盖为 Q2_K
./llama-quantize \
  --tensor-type "ffn_gate_exps.weight:Q2_K" \
  --tensor-type "ffn_up_exps.weight:Q2_K" \
  --tensor-type "ffn_down_exps.weight:Q2_K" \
  input.gguf output.gguf Q8_0
```

> ⚠️ 这消除了"写 Python 脚本解析 GGUF"的需求，项目难度降了一个台阶。

### 需要确认的坑

#### 坑 1：Tensor 命名格式

Qwen3 MoE 在 GGUF 中的实际命名可能是：

```
# 方案 A：打包式（一个 tensor 包含 128 个专家）
blk.0.ffn_gate_exps.weight   # shape: [768, 2048, 128]
blk.0.ffn_up_exps.weight     # shape: [2048, 768, 128]
blk.0.ffn_down_exps.weight   # shape: [768, 2048, 128]

# 方案 B：逐专家式（128 个独立 tensor）
blk.0.ffn_gate.0.weight
blk.0.ffn_gate.1.weight
...
blk.0.ffn_gate.127.weight
```

**必须先探查**：

```bash
# 方法 1: llama-gguf-dump（如果 llama.cpp 编译了）
./build/bin/llama-gguf-dump model.gguf | grep tensor | head -30

# 方法 2: gguf-py
python3 -c "from gguf import GGUFReader; r=GGUFReader('model.gguf'); [print(t.name, t.shape) for t in r.tensors[:20]]"

# 方法 3: 直接 grep
strings model.gguf | grep -E "blk\.0\.ffn_(gate|up|down)" | sort -u
```

#### 坑 2：通配符支持

`--tensor-type` **不支持通配符**。如果命名是逐专家式（48 层 × 128 专家 × 3 方向 = 18,432 个 tensor），手动写 `--tensor-type` 不现实。

**解决方案**：写一个 Python/shell 脚本生成命令行参数。

#### 坑 3：默认量化类型

命令行最后一个参数（如 `Q8_0`）是**默认量化类型**。未在 `--tensor-type` 中指定的 tensor 走默认。

#### 坑 4：`--tensor-type` 匹配机制（关键！）

文档假设 `--tensor-type "ffn_gate_exps.weight=Q2_K"` 能匹配所有层的 `blk.N.ffn_gate_exps.weight`。但这是**推测**，必须实测确认：

- **精确匹配**：需要写 `blk.0.ffn_gate_exps.weight=Q2_K` ... `blk.47.ffn_gate_exps.weight=Q2_K`（48 条）
- **前缀/后缀匹配**：写 `ffn_gate_exps.weight=Q2_K` 一次即可匹配所有层
- **不支持通配符**：官方文档未提及 `*` 或 `?` 支持

> ✅ **已验证**：`--tensor-type` 使用**后缀匹配**。`ffn_gate_exps.weight=Q2_K` 自动匹配所有 48 层的同名 tensor。语法为 **`=`**（等号），不是 `:`。
>
> 验证命令：
> ```bash
> ./llama-quantize --allow-requantize --tensor-type "ffn_gate_exps.weight=Q2_K" \
>   input.gguf test_match.gguf Q8_0
> # 检查：blk.0/1/23.ffn_gate_exps → type 10 (Q2_K) ✅
> # 检查：blk.0.ffn_up_exps → type 8 (Q8_0) ✅
> ```

### 分层策略

#### 第一优先级（最接近 ds4.c 策略）

| 层级 | Tensor 模式 | 精度 | 理由 |
|------|------------|------|------|
| MoE Expert 权重 | `*ffn_gate_exps*`, `*ffn_up_exps*`, `*ffn_down_exps*` | **Q2_K** | 占模型 80%+，最该压缩 |
| 其余所有层 | default | **Q8_0** | 保精度 |

#### 第二优先级（更精细的非对称）

| 层级 | Tensor 模式 | 精度 | 理由 |
|------|------------|------|------|
| Expert gate/up | `*ffn_gate_exps*`, `*ffn_up_exps*` | **IQ2_XXS** | 更激进的 2.06 bit |
| Expert down | `*ffn_down_exps*` | **Q2_K** | 2.5 bit，稍保守 |
| Attention Q/K/V/O | `*attn_q*`, `*attn_k*`, `*attn_v*`, `*attn_output*` | **Q4_K_M** | 敏感，不能压太狠 |
| Shared Expert | `*shared_expert*` | **Q4_K_M** | 高频激活，保精度 |
| 首尾层 + Norm + Embedding | `token_embd`, `output*`, `output_norm`, `*ln_*` | **FP16** | 输入输出质量关键 |

> 第二优先级是 ds4.c 策略的真正对齐版本，但先从第一优先级开始验证。

### 执行步骤

```bash
# Step B0: 探查 tensor 命名（必须先做）
cd /Volumes/ExtremeSSD/qwen3-engine/llama.cpp
./build/bin/llama-gguf-dump \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf \
  | grep -i "ffn_gate\|ffn_up\|ffn_down\|expert" | head -20

# Step B1: 第一优先级非对称量化（默认 Q8_0，expert 覆盖为 Q2_K）
# 注意：以下 --tensor-type 参数可能需要根据实际命名调整
./build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type "ffn_gate_exps.weight:Q2_K" \
  --tensor-type "ffn_up_exps.weight:Q2_K" \
  --tensor-type "ffn_down_exps.weight:Q2_K" \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-ASYM-Q2-Q8.gguf \
  Q8_0

# Step B2: 跑 perplexity
./build/bin/llama-perplexity \
  -m /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-ASYM-Q2-Q8.gguf \
  -f /Volumes/ExtremeSSD/qwen3-engine/data/test.txt \
  -ngl 999 \
  > /Volumes/ExtremeSSD/qwen3-engine/data/ppl_asym_q2_q8.log 2>&1
```

### 如果第一优先级成功，追加第二优先级

```bash
# Step B3: 更精细的非对称（ds4.c 风格）
./build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type "ffn_gate_exps.weight:IQ2_XXS" \
  --tensor-type "ffn_up_exps.weight:IQ2_XXS" \
  --tensor-type "ffn_down_exps.weight:Q2_K" \
  --tensor-type "attn_q.weight:Q4_K_M" \
  --tensor-type "attn_k.weight:Q4_K_M" \
  --tensor-type "attn_v.weight:Q4_K_M" \
  --tensor-type "attn_output.weight:Q4_K_M" \
  --tensor-type "shared_expert_gate.weight:Q4_K_M" \
  --tensor-type "shared_expert_up.weight:Q4_K_M" \
  --tensor-type "shared_expert_down.weight:Q4_K_M" \
  --tensor-type "token_embd.weight:FP16" \
  --tensor-type "output.weight:FP16" \
  --tensor-type "output_norm.weight:FP16" \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-ASYM-DS4.gguf \
  Q8_0
```

> ⚠️ 以上 tensor 名称为**推测**，必须以 Step B0 的实际输出为准。

---

## 预期结果与 Go/No-Go 标准

### 四点数据对比表

| 实验 | 量化策略 | 文件大小(估) | 预期 PPL 劣化 | 判断标准 |
|------|---------|------------|-------------|---------|
| Q2_K (已有) | Naive uniform Q2 | 10.5 GB | **+20.56%** | 基准下限 |
| **IQ2_XXS** | Smart uniform 2-bit | ~9-11 GB | **10-15%** | 路线 A 目标 |
| **Asym Q2+Q8** | Expert Q2 + 其余 Q8 | ~13-14 GB | **5-10%** | 路线 B 第一优先级 |
| **Asym DS4** | ds4.c 风格精细分层 | ~16-20 GB | **7-12%** | 路线 B 第二优先级 |

> 注：Asym Q2+Q8 的理论估算：30B × (0.8 × 0.33 + 0.2 × 1.0) ≈ 13.9 GB。量化完成后需记录**实际文件大小**更新报告。

### Go/No-Go 决策树

```
┌──────────────────────────────────────────────────────────────┐
│                    非对称 PPL 劣化结果                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  < 5%  ──────→  🟢 GO！核心假设完美验证，235B 项目绿灯      │
│                                                              │
│  5-10% ──────→  🟡 GO（with caution）。可接受，开始写       │
│                Metal 代码，同时优化分层策略                  │
│                                                              │
│  10-15% ─────→  🟠 WATCH。专家权重可能需从 Q2 改为 Q3/     │
│                Q4，重新跑验证                                │
│                                                              │
│  > 15% ──────→  🔴 KILL。非对称策略对 Qwen3 无效，项目     │
│                范围需收缩（如改做 Q4 优化的 30B 引擎）       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 执行时间表

| 步骤 | 内容 | 预估时间 | 依赖 |
|------|------|---------|------|
| 0 | 准备独立评估数据 (eval.txt) | 1 分钟 | 无 |
| 1 | 探查 tensor 命名 (Step B0) | 5 分钟 | 无 |
| 1.5 | 小实验确认 `--tensor-type` 匹配规则 | 5 分钟 | Step 1 |
| 2 | **路线 A**：imatrix + IQ2_XXS | 30-45 分钟 | Step 1 |
| 3 | **路线 B-1**：非对称 Q2+Q8 | 20-30 分钟 | Step 1.5 |
| 4 | **路线 B-2**：ds4.c 风格精细分层 | 20-30 分钟 | Step 3 成功后 |
| 5 | 汇总数据，更新报告 | 15 分钟 | 全部完成后 |
| | **总计** | **~2.5 小时** | |

---

## 附录：llama-quantize --tensor-type 参考

### 用法

```
llama-quantize [options] input.gguf output.gguf <default_type>

Options:
  --tensor-type <name>:<type>   Override quantization type for specific tensor(s)
                                Can be specified multiple times
  --allow-requantize            Allow requantizing already-quantized tensors
  --imatrix <file>              Use importance matrix for better quantization
```

### 可用的量化类型

| 类型 | 位宽 | 说明 |
|------|------|------|
| `Q2_K` | 2.5 bit | 标准 2-bit k-quants |
| `IQ2_XXS` | 2.06 bit | imatrix 优化的 super-block 2-bit |
| `IQ2_XS` | 2.31 bit | 稍高精度 |
| `IQ3_XXS` | 3.06 bit | 3-bit 超激进 |
| `Q4_K_M` | 4.5 bit | 标准 4-bit k-quants |
| `Q4_0` | 4 bit | 简单 uniform |
| `Q5_K_M` | 5.5 bit | 标准 5-bit |
| `Q6_K` | 6.5 bit | 标准 6-bit |
| `Q8_0` | 8 bit | 简单 uniform 8-bit |
| `FP16` | 16 bit | 半精度浮点 |

### 示例：混合精度命令

```bash
# 默认 Q4_K_M，attention 保 Q8_0，expert 压到 Q2_K
llama-quantize \
  --tensor-type "attn_q.weight:Q8_0" \
  --tensor-type "attn_k.weight:Q8_0" \
  --tensor-type "attn_v.weight:Q8_0" \
  --tensor-type "attn_output.weight:Q8_0" \
  --tensor-type "ffn_gate_exps.weight=Q2_K" \
  --tensor-type "ffn_up_exps.weight=Q2_K" \
  --tensor-type "ffn_down_exps.weight=Q2_K" \
  input.gguf output.gguf Q4_K_M
```

---

## 附录：第一轮实验原始数据

| 量化级别 | 文件大小 | Final PPL | 相对 Q8_0 劣化 | 推理速度 |
|---------|---------|-----------|---------------|---------|
| Q8_0 (基准) | 30.3 GB | **5.5237** | — | 1,319 tok/s |
| Q4_K_M | 17.2 GB | **5.6774** | +2.78% | 1,286 tok/s |
| Q2_K (naive) | 10.5 GB | **6.6591** | +20.56% | 1,263 tok/s |

---

*文档版本：V2*  
*创建时间：2026-05-12*  
*上一轮报告：[qwen3_30b_perplexity_validation.md](qwen3_30b_perplexity_validation.md)*
