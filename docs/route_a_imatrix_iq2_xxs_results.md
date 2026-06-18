# 路线 A 实验报告：imatrix + IQ2_XXS

> 实验时间：2026-05-12  
> 实验目的：验证"聪明的 uniform 2-bit 量化"（IQ2_XXS + imatrix）在 Qwen3-30B-A3B 上的质量表现，为路线 B（非对称量化）提供基线参考。

---

## 实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | Apple M3 Max, 48GB RAM |
| 推理框架 | llama.cpp ( Metal 启用 ) |
| 输入模型 | Qwen3-30B-A3B-Q8_0 (30.3 GB) |
| 校准数据 | `test.txt` (148 KB, ds4 项目安全提示文本) |
| 评估数据 | `eval.txt` (27 KB, ds4 项目 README.md) |

> ⚠️ **数据隔离**：imatrix 校准与 PPL 评估使用**不同文本**，避免数据泄露导致 PPL 偏乐观。

---

## 实验步骤

### Step A1: 生成 Importance Matrix (imatrix)

```bash
./build/bin/llama-imatrix \
  -m models/Qwen3-30B-A3B-Q8_0.gguf \
  -f data/test.txt \
  -o data/imatrix.dat \
  --threads 16
```

**结果**：

| 项目 | 数值 |
|------|------|
| 输出文件 | `imatrix.dat` |
| 文件大小 | **116 MB** |
| 覆盖率 | 多数 tensor 96-99%（校准文本较短，覆盖率未达 100%）|
| 耗时 | ~4 分钟 |

> 覆盖率未达 100% 是因为 `test.txt` 只有 148KB（约 3-5K tokens），imatrix 需要遍历模型所有激活值来收集统计信息。

---

### Step A2: IQ2_XXS 量化

```bash
./build/bin/llama-quantize \
  --allow-requantize \
  --imatrix data/imatrix.dat \
  models/Qwen3-30B-A3B-Q8_0.gguf \
  models/Qwen3-30B-A3B-IQ2_XXS.gguf \
  IQ2_XXS
```

**结果**：

| 项目 | 数值 |
|------|------|
| 输入大小 | 30.3 GB (Q8_0) |
| 输出大小 | **7.6 GB** |
| 压缩率 | **75%** (30.3G → 7.6G) |
| BPW (Bits Per Weight) | 2.1 ~ 2.5 bit（不均匀，imatrix 动态分配）|
| 耗时 | ~5 分钟 |

**关键观察：IQ2_XXS 的智能分配行为**

从量化日志中发现，IQ2_XXS **不是简单地把所有 tensor 压到 2-bit**，而是根据 imatrix 的重要性分析自动调整：

| Tensor | 输入类型 | 输出类型 | 说明 |
|--------|---------|---------|------|
| `ffn_gate_exps.weight` | Q8_0 | **iq2_xxs** | Expert gate → 最激进 2-bit |
| `ffn_up_exps.weight` | Q8_0 | **iq2_xxs** | Expert up → 最激进 2-bit |
| `ffn_down_exps.weight` | Q8_0 | **iq2_xxs** | Expert down → 最激进 2-bit |
| `attn_q.weight` | Q8_0 | **iq2_xxs** | Attention Q → 2-bit |
| `attn_k.weight` | Q8_0 | **iq2_xxs** | Attention K → 2-bit |
| `attn_output.weight` | Q8_0 | **iq2_xxs** | Attention output → 2-bit |
| `attn_v.weight` | Q8_0 | **q4_K** ⚠️ | Attention V → **保留 4-bit** |

> **发现**：imatrix 自动判断 `attn_v`（Value 投影）对质量更敏感，将其保留为 Q4_K，而其他 attention 权重压到 iq2_xxs。这正是"聪明的 2-bit"的核心机制。

---

### Step A3: Perplexity 评估

```bash
./build/bin/llama-perplexity \
  -m models/Qwen3-30B-A3B-IQ2_XXS.gguf \
  -f data/eval.txt \
  -ngl 999
```

**结果**：

| 指标 | 数值 |
|------|------|
| Final PPL | **19.8352** |
| 标准差 | ±1.16527 |
| 测试 Tokens | ~1,000 (eval.txt 较短) |
| 推理速度 | ~1,200 tok/s |

---

### 补跑基准：Q8_0 在 eval.txt 上的 PPL

由于路线 A 使用 `eval.txt` 作为评估数据，而第一轮实验的 Q8_0 基准是用 `test.txt` 跑的，两者不能直接比较。因此补跑 Q8_0 在相同 eval.txt 上的 PPL：

```bash
./build/bin/llama-perplexity \
  -m models/Qwen3-30B-A3B-Q8_0.gguf \
  -f data/eval.txt \
  -ngl 999
```

**结果**：

| 指标 | 数值 |
|------|------|
| Final PPL | **15.5221** |
| 标准差 | ±0.90286 |

---

## 最终结果对比

### 同数据集对比（eval.txt）

| 量化级别 | 文件大小 | PPL | 相对劣化 | 每 GB PPL |
|---------|---------|-----|---------|----------|
| **Q8_0** | 30.3 GB | **15.52** | 基准 | 0.51 |
| **IQ2_XXS** | 7.6 GB | **19.84** | **+27.79%** 🔴 | 2.61 |

### 跨数据集参考（仅作趋势观察，不可直接比较）

| 量化级别 | 评估数据 | PPL | 备注 |
|---------|---------|-----|------|
| Q8_0 | test.txt | 5.52 | 第一轮实验 |
| Q4_K_M | test.txt | 5.68 | 第一轮实验 |
| Q2_K (naive) | test.txt | 6.66 | 第一轮实验 (+20.56%) |
| Q8_0 | eval.txt | 15.52 | 本次补跑 |
| IQ2_XXS | eval.txt | 19.84 | 本次实验 (+27.79%) |

> ⚠️ `test.txt` (148KB) 与 `eval.txt` (27KB) 内容完全不同，PPL 绝对值不可跨数据集比较。仅**同数据集内的相对比较**有效。

---

## 关键发现与分析

### 1. IQ2_XXS 劣化幅度：+27.8%

**数字本身可能偏高**，原因包括：

| 因素 | 影响 |
|------|------|
| 评估文本太短 | eval.txt 仅 27KB，PPL 标准差 ±0.9~1.16，统计可信度有限 |
| 非自然语言内容 | README.md 含大量 markdown 标记、代码块、链接，模型预测困难 |
| 校准文本覆盖不足 | imatrix 覆盖率 96-99%，未达 100%，可能影响精度分配 |

**但方向性结论可靠**：即使"聪明的 2-bit"（imatrix 动态分配精度）在 30B 上仍有显著质量损失。

### 2. 与 naive Q2_K 的对比（趋势参考）

| 指标 | naive Q2_K | IQ2_XXS (smart) | 差异 |
|------|-----------|----------------|------|
| 文件大小 | 10.5 GB | **7.6 GB** | IQ2_XXS 更小 |
| 估计劣化 | ~20-25% | ~25-30% | **IQ2_XXS 并未显著优于 naive Q2_K** |
| 智能分配 | ❌ 无 | ✅ 有（attn_v 保 Q4）| 机制更优，但效果有限 |

> **意外发现**：IQ2_XXS 的"智能分配"并没有带来预期的质量飞跃。可能原因是：
> 1. 30B 模型本身对 2-bit 太敏感
> 2. 再量化（Q8→2-bit）叠加误差过大
> 3. 校准数据太短，imatrix 统计不充分

### 3. 对路线 B（非对称量化）的启示

| 发现 | 启示 |
|------|------|
| uniform 2-bit 不可行 | 无论 naive 还是 smart，uniform Q2 劣化都 >20% |
| attn_v 被 imatrix 保护 | **验证了我们的直觉**——attention 权重确实更敏感，非对称策略中应保高精度 |
| expert 权重可被激进压缩 | imatrix 毫不犹豫地把所有 expert 权重压到 iq2_xxs，说明 expert 是最佳压缩目标 |

**结论**：路线 B（只压 expert，attention/FFN 保 Q8）是更合理的路径。

---

## 实验过程中的问题记录

| # | 问题 | 解决方案 |
|---|------|---------|
| 1 | `llama-imatrix` 未编译 | 补编译：`cmake --build build --target llama-imatrix` |
| 2 | `--tensor-type` 语法误写为 `:` | 实际语法为 `=`：`name=TYPE` |
| 3 | `--tensor-type` 匹配机制未知 | 实测确认：**后缀匹配**，`ffn_gate_exps.weight=Q2_K` 自动匹配所有 48 层 |
| 4 | 评估数据与校准数据相同 | 修正为：test.txt 校准，eval.txt 评估 |
| 5 | Q8_0 基准与 IQ2_XXS 基准不在同一数据集 | 补跑 Q8_0 on eval.txt |

---

## 原始日志路径

| 文件 | 路径 |
|------|------|
| 路线 A 全程日志 | `/Volumes/ExtremeSSD/qwen3-engine/data/route_a_full.log` |
| imatrix 生成日志 | `/Volumes/ExtremeSSD/qwen3-engine/data/step_a1.log` |
| IQ2_XXS 量化日志 | `/Volumes/ExtremeSSD/qwen3-engine/data/step_a2.log` |
| IQ2_XXS Perplexity | `/Volumes/ExtremeSSD/qwen3-engine/data/ppl_iq2_xxs.log` |
| Q8_0 eval 基准 | `/Volumes/ExtremeSSD/qwen3-engine/data/ppl_q8_0_eval.log` |
| imatrix 数据文件 | `/Volumes/ExtremeSSD/qwen3-engine/data/imatrix.dat` (116MB) |
| IQ2_XXS 模型文件 | `/Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-IQ2_XXS.gguf` (7.6GB) |

---

## 下一步：路线 B

基于路线 A 的结果，路线 B（非对称量化）的必要性更加明确：

| 实验 | 策略 | 预期 PPL 劣化 | 预期文件大小 |
|------|------|-------------|------------|
| 路线 A (已完成) | uniform IQ2_XXS | +27.8% | 7.6 GB |
| **路线 B-1** | Expert Q2 + 其余 Q8 | **5-10%** | ~13-14 GB |
| **路线 B-2** | ds4.c 精细分层 | **7-12%** | ~16-20 GB |

> 路线 B 是验证核心假设的最后一块拼图。如果 B-1 通过（劣化 <10%），则 235B 项目绿灯。

---

*报告生成时间：2026-05-12*  
*前置报告：[qwen3_30b_perplexity_validation.md](qwen3_30b_perplexity_validation.md)*  
*实验计划：[qwen3_experiment_plan_v2.md](qwen3_experiment_plan_v2.md)*
