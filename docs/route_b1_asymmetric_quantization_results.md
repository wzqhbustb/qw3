# 路线 B-1 实验报告：非对称量化核心假设验证

> 实验时间：2026-05-12  
> 实验目的：验证核心假设——"MoE Expert 权重用 Q2 压缩，Attention/FFN 保 Q8，可在保持质量的前提下大幅减小模型体积"。

---

## 实验背景

### 前置实验结论

| 实验 | 策略 | PPL (test.txt) | 劣化 | 结论 |
|------|------|---------------|------|------|
| 第一轮 | Uniform Q8_0 | 5.5237 | 基准 | ✅ 基准 |
| 第一轮 | Uniform Q4_K_M | 5.6774 | +2.78% | ✅ 通过 |
| 第一轮 | Uniform Q2_K | 6.6591 | +20.56% | ❌ 失败 |
| 路线 A | IQ2_XXS (imatrix) | 19.8352* | +27.79%* | ⚠️ 参考* |

> *路线 A 使用 eval.txt（README.md）评估，与第一轮 test.txt 不直接可比。详见 [route_a 报告](route_a_imatrix_iq2_xxs_results.md)。

### 核心问题

Uniform Q2_K（所有层一视同仁压到 2-bit）劣化 20%+，**不可接受**。但 Uniform 的失败不代表 2-bit 本身不可行——关键在于**哪里该压、哪里该保**。

---

## 实验设计

### 策略：非对称分层量化

```
Expert 权重（占模型 80%+）    → Q2_K  激进压缩
Attention / FFN / Norm 等      → Q8_0  保精度（默认）
```

### 理论依据

| 层级 | 占比 | 敏感度 | 量化策略 | 理由 |
|------|------|--------|---------|------|
| `ffn_gate_exps` | ~27% | 低 | **Q2_K** | 专家 gate，专门化、低信息熵 |
| `ffn_up_exps` | ~27% | 低 | **Q2_K** | 专家 up，专门化、低信息熵 |
| `ffn_down_exps` | ~27% | 低 | **Q2_K** | 专家 down，专门化、低信息熵 |
| `attn_q/k/v/o` | ~8% | 高 | **Q8_0** | Attention 投影，每次推理都参与 |
| `ffn_gate_inp` | ~1% | 中 | **Q8_0** | 路由门，决定专家选择 |
| Norm / Embedding | ~1% | 高 | **FP32/F16** | 数值稳定性关键 |

### 执行命令

```bash
./build/bin/llama-quantize \
  --allow-requantize \
  --tensor-type "ffn_gate_exps.weight=Q2_K" \
  --tensor-type "ffn_up_exps.weight=Q2_K" \
  --tensor-type "ffn_down_exps.weight=Q2_K" \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf \
  /Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-ASYM-Q2-Q8.gguf \
  Q8_0
```

> `--tensor-type` 使用**后缀匹配**：`ffn_gate_exps.weight=Q2_K` 自动匹配所有 48 层的同名 tensor。实测验证通过。

---

## 实验结果

### 量化结果

| 指标 | 数值 |
|------|------|
| 输入模型 | Q8_0 (30.3 GB, 8.51 BPW) |
| 输出模型 | **ASYM-Q2-Q8 (10.4 GB, 2.93 BPW)** |
| 压缩率 | **66.3%** (30.3G → 10.4G) |
| Expert 压缩 | 204 MB → 63 MB 每层 (Q8→Q2) |
| 量化耗时 | ~5 分钟 |

### Perplexity 结果

评估数据：`test.txt`（与第一轮实验同一数据集，确保可比性）

| 指标 | 数值 |
|------|------|
| **Final PPL** | **5.9912** |
| 标准差 | ±0.13012 |
| 测试 Tokens | 36,352 (71 chunks, n_ctx=512) |
| 推理速度 | ~1,260 tok/s |
| 评估耗时 | ~31 秒 |

---

## 全量对比：同数据集（test.txt）

| 量化级别 | 文件大小 | PPL | 相对 Q8_0 劣化 | 状态 |
|---------|---------|-----|---------------|------|
| **Q8_0** (基准) | 30.3 GB | **5.5237** | — | ✅ 基准 |
| **Q4_K_M** | 17.2 GB | **5.6774** | +2.78% | ✅ 优秀 |
| **🎯 Asym Q2+Q8** | **10.4 GB** | **5.9912** | **+8.46%** | 🟢 **通过** |
| Q2_K (naive uniform) | 10.5 GB | 6.6591 | +20.56% | ❌ 失败 |
| IQ2_XXS (smart uniform)* | 7.6 GB | — | — | ⚠️ 参考* |

> *IQ2_XXS 使用 eval.txt 评估，未在 test.txt 上跑，故不列入本表。

---

## 关键发现

### 1. 核心假设验证通过 🟢

**非对称量化的 PPL 劣化仅 8.46%，低于 10% 阈值。**

这意味着：
- ✅ "只压 expert，保 attention" 策略有效
- ✅ 2-bit 量化在 expert 权重上可行
- ✅ 质量损失可接受（人眼/人耳难以感知 8% 的 PPL 差异）

### 2. 非对称 vs Uniform 的碾压性优势

| 对比维度 | Asym Q2+Q8 | Uniform Q2_K |
|---------|-----------|-------------|
| 文件大小 | **10.4 GB** | 10.5 GB |
| PPL | **5.9912** | 6.6591 |
| 劣化 | **+8.46%** | +20.56% |
| 差值 | — | **PPL 好 0.67** |

> **同样 10.5G 大小，非对称比 Uniform Q2 的 PPL 好 10%！** 这证明策略设计比盲目压 bit 更重要。

### 3. 与 Q4_K_M 的性价比对比

| 指标 | Q4_K_M | Asym Q2+Q8 | 差异 |
|------|--------|-----------|------|
| 文件大小 | 17.2 GB | **10.4 GB** | **省 40%** |
| PPL | 5.6774 | 5.9912 | 多 5.5% |
| 劣化 | +2.78% | +8.46% | 多 5.7% |

> 用 **5.7% 的 PPL 劣化** 换取 **40% 的空间节省**——对于 128GB Mac 跑 235B 来说，这是**必须做的交易**。

### 4. 对 235B-A22B 的推算

| 模型 | 量化策略 | 权重体积 | KV @128K | 总计 | 128GB 可行性 |
|------|---------|---------|---------|------|-------------|
| 235B | 全 Q4_K_M | ~132 GB | ~25 GB | **~157 GB** | ❌ 不可行 |
| 235B | 非对称 Q2+Q8 | **~85 GB** | ~25 GB | **~110 GB** | ✅ **可行** |

> **非对称量化不是"优化选项"，而是 235B on 128GB 的刚需。**

---

## Go/No-Go 决策

```
┌─────────────────────────────────────────────────────────────┐
│              非对称量化核心假设验证结果                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PPL 劣化: +8.46%                                          │
│   阈值: 10%                                                 │
│                                                             │
│   判定: 🟢 GO!                                              │
│                                                             │
│   235B 项目可以继续推进。开始写 Metal 引擎骨架。             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 实验过程中的关键确认

| # | 确认项 | 状态 | 说明 |
|---|--------|------|------|
| 1 | `--tensor-type` 后缀匹配 | ✅ 实测 | `ffn_gate_exps.weight=Q2_K` 自动匹配所有 48 层 |
| 2 | `--tensor-type` 语法 | ✅ 实测 | 用 `=`（等号），不是 `:` |
| 3 | Expert 命名模式 | ✅ 确认 | 打包式 `blk.N.ffn_*_exps.weight`，非逐专家 |
| 4 | 非 expert 层未被误压缩 | ✅ 确认 | `attn_*`、`ffn_gate_inp`、`norm` 保持默认 Q8/FP32 |
| 5 | 评估数据一致性 | ✅ 确认 | 使用 test.txt，与第一轮基准同一数据集 |

---

## 下一步建议

### 方案 A：立即开始引擎开发（推荐）

B-1 已通过，核心假设验证完成。可以开始：
1. 搭建 Qwen3-30B Metal 推理引擎骨架
2. 基于 Q4_K_M 或 Asym Q2+Q8 模型进行开发测试
3. 后续再优化量化策略（路线 B-2）

### 方案 B：追加路线 B-2（精细分层）

进一步压榨空间：
```
Expert gate/up → IQ2_XXS (2.06 bit)
Expert down    → Q2_K   (2.5 bit)
Attention      → Q4_K_M (4.5 bit)
首尾层         → FP16
```
预期体积：~16-20 GB，预期劣化：~10-12%。

### 方案 C：两者并行

- 引擎骨架开发立即启动
- B-2 作为后台实验持续进行

---

## 原始日志路径

| 文件 | 路径 |
|------|---------|
| 全程日志 | `/Volumes/ExtremeSSD/qwen3-engine/data/route_b1_full.log` |
| 量化日志 | `/Volumes/ExtremeSSD/qwen3-engine/data/step_b1.log` |
| Perplexity 日志 | `/Volumes/ExtremeSSD/qwen3-engine/data/ppl_asym_q2_q8.log` |
| 量化后模型 | `/Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-ASYM-Q2-Q8.gguf` (10.4 GB) |

---

## 相关文档

- [第一轮实验报告](qwen3_30b_perplexity_validation.md)
- [路线 A 报告](route_a_imatrix_iq2_xxs_results.md)
- [实验计划 V2](qwen3_experiment_plan_v2.md)

---

*报告生成时间：2026-05-12*  
*验证状态：🟢 核心假设通过*  
*项目状态：Go for Metal engine development*
