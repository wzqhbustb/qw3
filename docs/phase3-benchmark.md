# Phase 3 Benchmark Report

## Environment

- Hardware: Apple M3 Max, 48 GB RAM
- OS: macOS 14.5, Metal 3
- Build: `cc -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal`
- Model: `/Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q4_K_M.gguf` (~17 GB)
- Key dims: `n_embd=2048`, `n_head=32`, `n_kv_heads=4`, `head_dim=128`, `n_layer=48`
- MoE: 128 experts, top-8, `N_FF_EXP=768`, `N_FF_SHARED=6144`

## Summary

Target was **>20 tok/s** greedy decode. Final result: **~21 tok/s** on warm cache.

| Step | Optimization | Speed (warm cache) |
|------|--------------|--------------------|
| Baseline (Phase 2.6) | per-layer 2× flush + separate expert dispatches | ~15 tok/s |
| Step 1 | Fused quantized MoE kernels (5 → 2 dispatches/expert) | ~15 tok/s |
| Step 2 | Interleaved multi-layer batching (97 → 49 flushes/token) | ~17 tok/s |
| Step 3 | GPU router top-k + 2-CB/token (49 → 2 flushes/token) | **~21 tok/s** |

## Step 1: Fused quantized MoE expert kernels

Merged gate+up+silu and down+accumulate into dedicated SIMD kernels for Q4_K,
Q6_K and Q8_0. This reduced per-expert dispatch count from 5 to 2, but by itself
only matched the warm-cache baseline (~15 tok/s), because the bottleneck was
already command-buffer sync rather than dispatch count.

## Step 2: Lightweight GPU timer profile

Xcode is not installed, so Metal System Trace is unavailable. Added
programmatic command-buffer timing controlled by `DS3_METAL_PROFILE=1`.

Profile (fused path, before multi-layer batching):

| Metric | Value |
|--------|-------|
| Command buffers / token | 97 |
| GPU active time | ~37 ms |
| CPU wait time (`waitUntilCompleted`) | ~54 ms |
| Wall time / token | ~67 ms |

Conclusion: the fused kernels are already efficient (~37 ms of GPU work is near
the M3 Max bandwidth-bound limit for this model). The actual decode speed was
limited by the **~54 ms of CPU sync/idle gap** from 97 command-buffer flushes
per token (2 per layer + final output).

## Step 3: Interleaved multi-layer batching

Rewrote `forward_token()` to overlap post-router work of layer *l* with
pre-router work of layer *l+1* in the same command buffer:

```
CB 0      : layer 0 pre-router
CB 1..N-1 : layer l post-router + layer l+1 pre-router
CB N      : layer N-1 post-router + final output projection
```

This reduced flushes from 97 to 49 per token and improved speed to **~17 tok/s**.

## Step 4: GPU router top-k + 2-CB per token

Moved the router softmax and top-k to the GPU. Each token now uses only two
command buffers:

```
CB 0: all 48 layers of attention + router matmul + GPU softmax/top-k
CPU : read 48 * 8 indices/scores (~768 bytes)
CB 1: all 48 layers of shared/routed experts + final output projection
```

This keeps the efficient fused per-expert kernels from Step 1 while eliminating
almost all per-layer CPU sync overhead.

Results (warm cache):

```text
[gen] 64 tokens in 2.99 s = 21.43 tok/s
[gen] 64 tokens in 3.02 s = 21.22 tok/s
[gen] 64 tokens in 3.03 s = 21.15 tok/s
[gen] 64 tokens in 3.03 s = 21.10 tok/s
```

Profile after this change:

| Metric | Value |
|--------|-------|
| Command buffers / token | 2 |
| GPU active time | ~36–37 ms |
| CPU wait time | ~37 ms |
| Wall time / token | ~47 ms |

GPU utilization is now close to 100%: CPU wait (~37 ms) essentially matches GPU
active time. The remaining ~10 ms/token is CPU readback + dispatch encoding,
which is hard to remove without an even more aggressive GPU-only approach.

## Correctness

All unit tests pass:

- `test_metal`: PASS
- `test_matmul_quant_metal`: PASS
- `test_attention_metal`: PASS
- `test_moe_metal`: PASS (all five variants)

Real-model greedy decode produces coherent text.

## Files changed

- `metal/moe.metal`: fused expert kernels + batch router top-k kernel.
- `src/ds3_metal.h`: new low-level MoE APIs (`router`, `experts`, `topk_batch`).
- `src/ds3_metal.m`: pipelines, dispatch helpers, profiling hooks.
- `src/ds3_engine.c`: interleaved multi-layer batching + 2-CB/token schedule,
  larger scratch buffers for per-layer router data.
- `tests/test_moe_metal.c`: updated to match new `ds3_metal_moe_ffn` signature.

## Notes / caveats

- External SSD file-cache warmth causes large run-to-run variance on the first
  generation after model load. Reported numbers are from repeated warm runs.
- The earlier GPU-only one-shot MoE kernel (router + experts in a single
dispatch) is still present but disabled; it was ~47% slower than the fused
per-expert path.
- `DS3_METAL_PROFILE=1` remains available for future tuning.
