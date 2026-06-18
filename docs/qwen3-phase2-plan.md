# Qwen3 Engine — Phase 2 Plan

**Goal**: Port the validated CPU reference from Phase 1 to a full Metal GPU
inference path for Qwen3-30B-A3B, and verify it matches llama.cpp within numerical precision.

**Duration estimate**: 2–3 weeks  
**Target model**: Qwen3-30B-A3B-Q4_K_M.gguf  
**Hardware target**: 48 GB Apple Silicon (M3 Max)  
**Success criteria**:
- `./qwen3-cli -m Qwen3-30B-A3B-Q4_K_M.gguf -p "Hello"` produces coherent text.
- Greedy-decoding output matches llama.cpp Metal for the first 20 tokens (or diverges
  only at numerical-precision ties, see 2.7).
- 30B-Q4_K_M reaches >20 tok/s on M3 Max.

---

## 1. Task inventory

| # | Task | Effort | Depends on |
|---|------|--------|------------|
| 2.0 | Metal Q4_K/Q6_K/Q8_0 dequant-matmul kernel | Large | 1.5 |
| 2.1 | Metal GQA attention kernel (decode-only, 48 layers) | Large | 2.0, 2.2 |
| 2.2 | Metal RoPE integration | Medium | 1.5 |
| 2.3 | Metal MoE kernel (128 experts, top-8, norm_topk_prob) | Medium | 2.0 |
| 2.4 | mmap + zero-copy weight binding | Medium | — |
| 2.5 | End-to-end Metal forward (embedding → 48 layers → logits → sampling) | Large | 2.1–2.4 |
| 2.6 | Performance benchmark vs llama.cpp Metal; **optimize MoE quantized path with SIMD kernels** | Small | 2.5 |
| 2.7 | Correctness: greedy decoding vs llama.cpp token-by-token | Small | 2.5 |

> **Deferred to Phase 3 / Phase 4**:
> - Model-view overlap (only needed for 235B / >32 GB buffers; 30B Q4_K_M ≈ 17 GB fits in a single MTLBuffer)
> - Prefill path (chunked multi-token attention with causal mask; decode-only is sufficient for Phase 2)

---

## 2. Detailed breakdown

### 2.0 Metal Q4_K / Q6_K / Q8_0 dequant-matmul kernel

**Why this is first**: Every projection in the model (Q/K/V/O, router, expert gate/up/down) is a
vector × quantized-matrix operation. Without this kernel, nothing else can run on GPU.

**Operation**: `y[out_dim] = x[in_dim] @ W^T[out_dim, in_dim]` where W is stored in Q4_K, Q6_K or Q8_0 format.

**Design**:
- One threadgroup per output element (or small tile of output elements).
- Each thread iterates over blocks of the input dimension, dequantizing on-the-fly.
- Q4_K: decode 256-weight super-blocks (2-level scale+min, 4-bit quants).
- Q6_K: decode 256-weight super-blocks (6-bit quants, 16 int8 scales, fp16 super-scale).
- Q8_0: decode 32-weight blocks (fp16 delta × int8 quants).
- Accumulate in FP32; write FP32 result.

**File targets**:
- `metal/matmul_q4k.metal`
- `metal/matmul_q6k.metal`
- `metal/matmul_q8_0.metal`
- `src/ds3_metal.h` / `src/ds3_metal.m`: add `ds3_metal_vec_matmul_quantized(...)`

**Acceptance**:
- `y = x @ W^T` matches CPU `ds3_ref_vec_matmul(x, dequant(W), y)` with `max_diff < 1e-3`
  (quantized matmul has higher tolerance due to fused dequant rounding).
- `tests/test_matmul_quant_metal.c` covers FP32, Q8_0, Q4_K, and Q6_K base + SIMD variants.

---

### 2.1 Metal GQA attention kernel (decode-only)

**What it replaces**: `ds3_ref_gqa_attention()` in `src/ds3_reference.c`.

**Scope**: Single-token decode only (seq_pos = current token). Prefill (multi-token causal attention)
is deferred to Phase 3 / Phase 4.

**Pipeline** (all on GPU, single command buffer):
1. Q/K/V projections via `ds3_metal_vec_matmul_quantized` (from 2.0).
2. Q/K RMSNorm (reuse `metal/rms_norm.metal`).
3. RoPE on Q and K (reuse `metal/rope.metal`).
4. KV-cache write: append new K, V to FP16 cache at `seq_pos`.
5. Q @ K_cache^T: `[N_HEAD, 1, HEAD_DIM] × [N_HEAD_KV, seq_len, HEAD_DIM]^T` → scores `[N_HEAD, seq_len]`.
6. Scaled softmax over `seq_len` positions. Implemented inline per head in
   `metal/attention.metal` (small `seq_len` during decode; a dedicated softmax
   kernel is only needed if prefill is added later).
7. Scores @ V_cache: `[N_HEAD, seq_len] × [N_HEAD_KV, seq_len, HEAD_DIM]` → `[N_HEAD, HEAD_DIM]`.
8. O projection via `ds3_metal_vec_matmul_quantized`.

**KV cache format**: **FP16** `[N_LAYER][seq_len][N_HEAD_KV][HEAD_DIM]`.
- 30B @ 128K ctx = 48 × 131072 × 4 × 128 × 2 bytes = **6 GB** (FP16).
- FP32 would be 12 GB — too expensive on 48 GB Mac with 17 GB model.

**Key differences from ds4**:
- Qwen3 uses standard GQA (4 KV heads, not MLA).
- No latent compression / decompressor / rope-tail split.
- GQA head mapping: Q heads 0–7 → KV head 0, 8–15 → KV head 1, etc.

**File targets**:
- `metal/attention.metal`
- `src/ds3_metal.h` / `src/ds3_metal.m`: add `ds3_metal_attention(...)`

**Acceptance**:
- Layer 0 attention output matches CPU reference with `max_diff < 1e-3` (quantized weights).

---

### 2.2 Metal RoPE integration

**Status**: ✅ Completed.

**Implementation**:
- Added `ds3_metal_attention_decode_rope()` and `ds3_metal_attention_decode_rope_simd()` to `src/ds3_metal.h` / `src/ds3_metal.m`.
- These functions apply `rope_f32` to Q and K (in-place), then write the new K/V row to the cache, then run the attention compute kernel — all in one ordered command buffer.
- Added `ensure_rope_freq_table()` helper to share the precomputed frequency table between `ds3_metal_rope()` and the new attention path.

**Tests**: `tests/test_attention_metal.c` gained:
- `test_attention_decode_rope_small`
- `test_attention_decode_rope_qwen3_dims` (SIMD variant)

Both pass against the CPU reference (RoPE + attention) with `max_diff < 1e-3`.

---

### 2.3 Metal MoE kernel

**Status**: ✅ Completed (FP32 / Q4_K / Q6_K / Q8_0 expert weights; shared-expert wired; quantized path now uses SIMD kernels).

**Implementation**:
- Added `metal/moe.metal` with two FP32 kernels per selected expert:
  - `moe_expert_gate_up_f32`: computes `hidden = silu(input @ W_gate^T) * (input @ W_up^T)`
  - `moe_expert_down_f32`: computes `output += weight * (hidden @ W_down^T)`
- Added `ds3_metal_moe_ffn()` to `src/ds3_metal.h` / `src/ds3_metal.m`.
- Router path reuses existing quantized vec-matmul (`ds3_metal_vec_matmul_*`) for
  `gate_logits`, then does `softmax + topk + renorm` on the CPU (N=128 is tiny).
- Each selected expert is dispatched as two kernels in one command buffer;
  all 8 experts run sequentially in the same command buffer.
- Made `ds3_ref_moe_ffn()` runtime-parameterized so tests can use non-Qwen3
  dimensions for fast validation.

**Completed details**:
- FP32 expert path: one fused dispatch per expert (gate+up, then down+accumulate).
- Quantized expert path: `dispatch_moe_expert_quant()` dispatches separate
  `matmul_vec_*` kernels for gate / up / down, supporting mixed per-weight types
  (e.g. gate=Q4_K, down=Q6_K) and per-expert byte offsets.
- Added `metal/matmul_q6k.metal` and `ds3_metal_vec_matmul_q6k_simd()` so Q6_K
  expert/attention weights stay on GPU without host-side dequantization.
- Shared expert path wired in `ds3_engine.c`; dispatcher treats shared experts
  identically to routed experts with `n_ff = N_FF_SHARED`.
- `ds3_metal_moe_ffn()` now takes per-weight types and row strides for both
  routed and shared experts.
- Q4_K/Q6_K/Q8_0 matmul kernels gained a `weight_offset` field so a single
  packed expert tensor can be sliced without copies.

**Deferred / follow-up**:
- Fused/batched expert kernel (Phase 3 optimization if dispatch overhead matters).
- Router softmax/top-k on GPU (currently done on CPU after reading 128 logits).

**Tests**: `tests/test_moe_metal.c`
- `test_moe_ffn_small` (256/16/4/128, FP32)
- `test_moe_ffn_qwen3_dims` (2048/128/8/768, FP32)
- `test_moe_ffn_qwen3_dims_q4k` (2048/128/8/768, Q4_K expert weights)
- `test_moe_ffn_qwen3_dims_mixed` (2048/128/8/768, gate=Q4_K/up=Q4_K/down=FP32)
- `test_moe_ffn_mixed_q4k_q6k` (2048/128/8/768, gate=Q4_K/up=Q4_K/down=Q6_K)

All pass against `ds3_ref_moe_ffn()` with `max_diff < 2e-3` (Q4_K / mixed within `4e-5`).

---

### 2.4 mmap + zero-copy weight binding

**Status**: ✅ Implemented (untested on real model; no RSS regression expected).

**What it does**: Avoid copying 17 GB of weights from the mmap'd GGUF into separate GPU buffers.

**Implementation**:
- `ds3_gguf.c` already `mmap()`s the GGUF file.
- Added `ds3_metal_buffer_from_mmap()` in `src/ds3_metal.m` which creates **one** base
  `MTLBuffer` over the entire mmap with
  `-[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]`.
  The mmap base is page-aligned; the length is rounded up to a page multiple.
- Added `ds3_metal_buffer_view()` to create sub-range views with byte offsets.
  Tensor views are not required to be page-aligned; the offset is applied at kernel
  bind time via `setBuffer:offset:`.
- Updated all `setBuffer:` calls in `src/ds3_metal.m` to use `mtl_buf()` / `mtl_off()`
  helpers so views bind with the correct offset.
- `src/ds3_engine.c` now creates a single `weight_base_buf` from `gguf->mmap_base` and
  binds every supported quantized/FP32 layer weight as a view into it.
  Dequantized weights (norm weights that are not FP32, token embedding, output weight)
  still use separate writable GPU buffers.

**Reference**: ds4 `ds4_gpu_model_view` in `ds4/ds4_metal.m` and kernels taking `(const void *model_map, uint64_t weight_offset)`.

**Acceptance**:
- `vmmap ./qwen3-cli` should show only one copy of weight pages in RAM.
- Peak process RSS ≈ model_size + KV cache + scratch, no 2× weight duplication.
- To be validated on the target model once it is available.

---

### 2.5 End-to-end Metal forward

**Status**: ✅ Completed (single-token decode; GPU residual adds; shared-expert and Q6_K MoE wired).

**What it does**: Build the full inference loop and get text output.

**Complete forward path** (decode, single token):
```
token_id
  → token_embd lookup (dequantize the row for `token_id` to FP32; Qwen3-30B
    GGUF stores token_embd as FP16 or a k-quant, so reuse the dequant-matmul
    infrastructure from 2.0 with a one-hot input vector or a dedicated gather kernel).
  → for layer in 0..47:
      residual = x
      x = rms_norm(x, attn_norm_weight)
      x = attention(x, kv_cache[layer], seq_pos)  [decode-only]
      x = x + residual
      residual = x
      x = rms_norm(x, ffn_norm_weight)
      x = moe_ffn(x)
      x = x + residual
  → x = rms_norm(x, output_norm_weight)
  → logits = x @ output_weight^T  (vec-matmul, [1, 2048] × [151936, 2048]^T;
     if `output_weight` is tied to `token_embd`, reuse the same weight tensor)
  → next_token = argmax(logits)  [greedy] or sample(logits, temperature)
```

**Engine responsibilities**:
- Allocate persistent GPU buffers for:
  - KV cache: FP16, `[N_LAYER][max_seq_len][N_HEAD_KV][HEAD_DIM]`
  - Hidden state scratch: 2× `[N_EMBD]` FP32 (ping-pong)
  - Logits: `[N_VOCAB]` FP32
- Token embedding lookup (dequantize the row for the current token_id to FP32).
- Final output norm + logits projection.
- Greedy / temperature sampling on CPU (small N_VOCAB=151936 softmax).
- Decode loop: repeat until EOS or max_tokens.

**File targets**:
- `src/ds3.c` / `src/ds3.h`: `ds3_engine_open()`, `ds3_engine_generate()`
- `tools/qwen3-cli.c`: command-line interface

**Acceptance**:
- `./qwen3-cli -m ... -p "Hello"` prints coherent continuation.

---

### 2.6 Performance benchmark + MoE SIMD optimization

**Status**: ✅ Completed (SIMD MoE + GPU residual adds + benchmark harness).

**MoE quantized path SIMD optimization**:
- `dispatch_moe_expert_quant()` now dispatches SIMD variants
  (`matmul_vec_q4k_simd`, `matmul_vec_q6k_simd`, `matmul_vec_q8_0_simd`) for
  gate/up/down. FP32 expert weights still use the fused `moe_expert_gate_up_f32` /
  `moe_expert_down_f32` path.
- Added `dispatch_quant_matmul_row_simd()` to embed SIMD matmul dispatches inside
  an existing command encoder while preserving per-expert `weight_offset`.
- Fixed Q8_0 `ds3_matmul_q8_0_args` to include `weight_offset` so it behaves
  consistently with Q4_K/Q6_K when slicing packed expert tensors.

**GPU residual adds**:
- Added `metal/elementwise.metal` with small FP32 kernels: `vec_copy_f32`,
  `vec_zero_f32`, `vec_add_f32`, `vec_add3_f32`.
- Added `ds3_metal_buffer_copy`, `ds3_metal_buffer_zero`,
  `ds3_metal_vec_add_inplace`, `ds3_metal_vec_add` to `src/ds3_metal.h/.m`.
- `src/ds3_engine.c` now keeps residual state in a GPU buffer (`buf_residual`)
  and performs attention/FFN residual additions on the GPU. This removes the
  4 small GPU↔CPU transfers per layer that were an active bottleneck.

**Benchmark** (M3 Max, Qwen3-30B-A3B-Q4_K_M.gguf, measured by `qwen3-cli`):
| metric | value |
|--------|-------|
| prompt | `"Hello"` (1 token) |
| generated | 128 tokens, greedy (`t=0`) |
| decode time | 28.73 s |
| decode tok/s | **4.45 tok/s** |
| 64 tokens, `t=0.7` | **3.97 tok/s** |

After implementing command-buffer batching and mmap zero-copy binding, a fresh
benchmark on `Qwen3-30B-A3B-Q4_K_M.gguf` (M3 Max, external SSD, warm cache)
shows:

| metric | value |
|--------|-------|
| 128 tokens greedy (`t=0`) | **≈16.07 tok/s** (was 4.45 tok/s) |
| 64 tokens `t=0.7` | **≈14.34 tok/s** (was 3.97 tok/s) |
| model load time | **≈0.9 s** (was tens of seconds) |

This is a **≈3.6×** decode speed-up from batching and a major reduction in load
latency from zero-copy. The >20 tok/s target is still not met; further kernel
optimizations (e.g. output projection tiling, fused MoE expert dispatch) remain
available if needed.

**Acceptance**:
- ✅ SIMD MoE path matches non-SIMD numerically (`max_diff < 2e-3` against CPU
  reference in `tests/test_moe_metal.c`).
- ⚠️ Decode speed 4.45 tok/s; >20 tok/s not yet reached.

---

### 2.7 Correctness validation

**Method**:
- Run greedy decoding (temperature = 0, top_p = 1) on both engines with the same prompt.
- Compare token IDs for the first 20 tokens, then inspect any divergence.
- If mismatched, dump logits top-5 at divergence and check whether it is a numerical precision difference or a logic bug.

**Acceptance**:
- First 20 tokens identical to llama.cpp; OR
- At the first divergence, the relative difference between ds3 and llama.cpp logits for the
  top-5 candidates is < 0.1%, indicating a numerical-precision tie rather than a logic bug.

> Why 20 tokens? Quantized dequantization order and FP32 vs FP16 accumulation can produce
> tiny logit differences. A top-1 gap < 0.01 at token 21 can cause later tokens to diverge
> even when both implementations are logically correct. Twenty tokens is enough to catch
> real bugs while tolerating benign numerical noise.

---

## 3. Suggested execution order

1. **2.0 Q4_K/Q8_0 dequant-matmul** — the foundational kernel everything depends on. Develop against FP32 buffers first for correctness, then switch to quantized.
2. **2.4 mmap zero-copy** — set up weight binding infrastructure. Can be done in parallel with 2.0.
3. **2.1 Attention + 2.2 RoPE** — largest compute block; dominates decode latency. Debug against FP32 buffers first, then integrate zero-copy.
4. **2.3 MoE** — second largest block; reuses the same dequant-matmul infrastructure.
5. **2.5 End-to-end** — wire everything together (embedding, 48 layers, logits, sampling).
6. **2.6 Benchmark + MoE SIMD + 2.7 Correctness** — final validation gates.

---

## 4. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Q4_K dequant-matmul kernel too slow | High | Profile early; try different tiling strategies (per-row vs per-block) |
| Metal attention kernel slower than llama.cpp | High | Use ds4 f16 flash-attn patterns if needed |
| MoE routing mismatch (norm_topk_prob) | High | Unit-test top-k + renormalization against reference |
| mmap zero-copy fails (page alignment) | Medium | Fall back to copy-to-buffer; profile RSS impact |
| Numerical drift vs llama.cpp | Medium | Use top-5 logits comparison to distinguish precision from bugs |
| KV cache memory too large at 128K | Medium | Default to 32K context for Phase 2; 128K in Phase 4 with KV quantization |

---

## 5. Deliverables

- `metal/matmul_q4k.metal` — Q4_K dequant-matmul kernel
- `metal/matmul_q6k.metal` — Q6_K dequant-matmul kernel
- `metal/matmul_q8_0.metal` — Q8_0 dequant-matmul kernel
- `metal/attention.metal` — GQA decode attention
- `metal/moe.metal` — MoE routing + expert dispatch
- `metal/elementwise.metal` — GPU residual-add / copy / zero kernels
- `src/ds3_metal.h/.m` extended with attention / MoE / elementwise / zero-copy / dequant-matmul APIs
  (embedding lookup and logits projection reuse the quantized dequant-matmul kernel)
- `src/ds3_engine.c` engine graph (embedding → layers → logits → sampling)
- `tools/qwen3-cli.c` command-line tool
- `tests/test_matmul_quant_metal.c` — dequant-matmul validation
- `tests/test_metal.c` — RMSNorm / RoPE / matmul / elementwise validation
- `tests/test_attention_metal.c` — attention validation
- `tests/test_moe_metal.c` — MoE validation
- `tests/test_e2e_metal.c` — full forward validation
- Benchmark report (`phase2-benchmark.md`)
