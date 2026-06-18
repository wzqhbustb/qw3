# Phase 4a MVP: Chunked Prompt Prefill — Benchmark & Notes

**Date:** 2026-06-16  
**Hardware:** Apple M3 Max, 48 GB RAM, macOS 14.5  
**Model:** `/Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q4_K_M.gguf`  
**Code state:** commit after Phase 4a MVP implementation

---

## 1. What was implemented

| Component | Status | Notes |
|-----------|--------|-------|
| Tiled FP32 batched matmul | ✅ | `matmul_f32_tiled` in `metal/matmul.metal`; `ds3_metal_matmul` auto-selects tiled path for `M > 1` |
| Batched quantized matmul | ✅ | `matmul_q4k` / `matmul_q6k` / `matmul_q8_0` in respective `.metal` files; host API `ds3_metal_matmul_quantized_batch` |
| Batched KV cache write | ✅ | `kv_cache_write_batch` kernel + `ds3_metal_kv_cache_write_batch` |
| Naive causal chunk attention | ✅ | `attention_chunk_gqa` kernel + `ds3_metal_attention_chunk`; O(n·m) within chunk, reads past KV from FP16 cache |
| Batched router top-k | ✅ | `moe_router_topk_batch_tokens_f32` handles `[layers][tokens][experts]` logits |
| Engine `forward_chunk()` | ✅ | `src/ds3_engine.c::forward_chunk()` processes the prompt in `DS3_PREFILL_CHUNK_SIZE` (512) tokens |
| Unit tests | ✅ | New tests added for batch quantized matmul, batch KV write, and chunk attention |

### Known Phase-4a simplification

The **MoE FFN is still evaluated per token inside the chunk**. The chunk path batches the attention and all linear projections, but each token’s selected experts are dispatched individually. This keeps the implementation simple and correct, but it is the main remaining bottleneck.

---

## 2. Unit-test results

```
./test_metal                 ... PASSED
./test_matmul_quant_metal    ... PASSED  (incl. new batch Q4_K/Q6_K/Q8_0 tests)
./test_attention_metal       ... PASSED  (incl. new batch KV write + chunk attention tests)
./test_moe_metal             ... PASSED
```

All new kernels pass their numerical checks against CPU references.

---

## 3. End-to-end prefill measurements

`DS3_METAL_PROFILE=1` was used. The model file was pre-loaded into the OS file cache with `dd` before the timed runs to separate cold-SSD effects from GPU/CPU overhead.

### 3.1 256-token prompt (warm cache)

```
[profile prompt] cb_count=8 total_gpu_ms=3420.636 total_cpu_wait_ms=6841.570
[profile step=0] cb_count=2 total_gpu_ms=44.033 total_cpu_wait_ms=53.220
[gen] 1 tokens in 6.94 s
```

* Prompt GPU time: **3.42 s** → ~75 tok/s on GPU
* Prompt wall time: **6.94 s** → ~37 tok/s effective
* Decode step after 256-token context: **~44 ms GPU / ~53 ms CPU**

### 3.2 512-token prompt (warm cache)

```
[profile prompt] cb_count=12 total_gpu_ms=7726.602 total_cpu_wait_ms=16577.330
[profile step=0] cb_count=2 total_gpu_ms=55.671 total_cpu_wait_ms=56.081
[gen] 1 tokens in 16.73 s
```

* Prompt GPU time: **7.73 s** → ~61 tok/s on GPU
* Prompt wall time: **16.73 s** → ~30 tok/s effective
* Decode step after 512-token context: **~56 ms**

### 3.3 Cold-SSD first run

First access after loading the model can be dominated by paging the 17 GB weight file from the external SSD into GPU-resident memory. A 256-token prompt on a cold cache took **~9–20 s** wall time even though GPU active time stayed near **3.4 s**.

---

## 4. Analysis

1. **Batched attention/projections work.** The GPU active time for a 256-token chunk (~3.4 s) is in the right ballpark for a bandwidth-bound 30 B MoE model and is much better than token-by-token prefill.

2. **The wall-time gap is per-token MoE overhead.** CPU wait is 2–3× the GPU time because the per-token expert dispatches inside the chunk create many small compute commands. Grouping layers into 8-layer command buffers helped (it cut CPU wait by ~40% versus one-CB-per-layer), but the fundamental fix is a batched/gathered MoE path.

3. **Flash Attention is not the blocker.** The naive causal chunk attention kernel is fast enough at these chunk sizes; the MoE FFN is the next target.

---

## 5. Next steps (Phase 4b)

* Implement **batched MoE expert execution**:
  * For each layer, group tokens by selected expert.
  * Gather token rows into a contiguous buffer, run the existing batch matmul kernels for gate/up/down, then scatter-accumulate with per-token router weights.
  * This removes the per-token dispatch explosion and should bring wall-time much closer to the GPU active time.
* Consider a **flash-style chunk attention** only if profiling shows attention becoming a bottleneck after MoE is batched.
* Validate end-to-end output quality with a known-good reference once the underlying Q6_K/Q4_K dequantizers have been cross-checked against llama.cpp/Transformers (current unit tests compare GPU vs CPU dequantization, not against a ground-truth model).
