# Agent Notes — Qwen3 Metal Engine

This is a minimal, Metal-only inference engine for Qwen3 MoE models (primary
development target: `Qwen3-30B-A3B-Q4_K_M.gguf`). It is *not* a generic GGUF
runner: shapes and quantization types are hard-coded for the Qwen3 architecture.

## Goals

- Production path is a whole-model Metal graph with as little CPU work as
  possible per token/chunk.
- Weights stay mmap-backed; avoid eager FP32 copies of embedding/output
  matrices.
- Correctness before speed. Any faster path must be explainable and must not
  drift attention, KV cache, or logits.
- Keep the CPU backend (`src/ds3_reference.c`) as reference/debug code only.

## Layout

- `src/ds3.h`, `src/ds3_gguf.c`, `src/ds3_weights.c`: model metadata, GGUF
  loading, tensor binding.
- `src/ds3_tokenizer.c`: BPE tokenizer.
- `src/ds3_reference.c`: FP32 CPU reference forward (correctness baseline).
- `src/ds3_engine.c`: Metal graph scheduling — decode, chunk prefill, KV cache.
- `src/ds3_metal.m`, `src/ds3_metal.h`: Objective-C Metal runtime and kernel
  wrappers.
- `src/ds3_log.c`: shared `ds3_log_quiet` state used by `ds3_print_info()` /
  `ds3_log_info()` macros.
- `metal/*.metal`: compute kernels (attention, matmul, MoE, elementwise, RoPE,
  RMS norm).
- `tools/qwen3-cli.c`: command-line chat/reasoning interface.
- `tests/`: unit tests for GGUF, tokenizer, reference, Metal kernels.
- `scripts/`: Python helpers for extracting ground-truth weights/layer outputs.

## Build

```bash
# Main CLI
make qwen3-cli

# Individual Metal unit tests
make test_metal test_moe_metal test_matmul_quant_metal test_attention_metal
```

Compiler flags are in the `Makefile`:

```
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -fobjc-arc
LDLIBS ?= -framework Foundation -framework Metal
```

No Xcode is required; the Metal library is compiled at runtime from the
`metal/*.metal` source files.

## Testing

Run these before declaring a change done:

1. **Metal unit tests**
   ```bash
   make test_metal test_moe_metal test_matmul_quant_metal test_attention_metal
   ```

2. **Full-model reference comparison** (one token vs FP32 reference)
   ```bash
   ./qwen3-cli -m <model.gguf> -p "1+1=" -F
   ```
   Expected `max_diff` is around `1.7e-3` for `Qwen3-30B-A3B-Q4_K_M.gguf`.

3. **Chunk-prefill vs token-by-token equivalence** (argmax / temperature 0)
   ```bash
   ./qwen3-cli -m <model.gguf> -p "<prompt>" -n 30 -t 0.0
   DS3_NO_CHUNK_PREFILL=1 ./qwen3-cli -m <model.gguf> -p "<prompt>" -n 30 -t 0.0
   ```
   Generated text must be byte-identical.

4. **Gathered MoE vs per-token MoE equivalence** (inside chunk prefill)
   ```bash
   ./qwen3-cli -m <model.gguf> -p "<prompt>" -n 30 -t 0.0
   DS3_CHUNK_NO_GATHERED_MOE=1 ./qwen3-cli -m <model.gguf> -p "<prompt>" -n 30 -t 0.0
   ```
   Generated text must be byte-identical.

5. **Multi-step decode drift check**
   ```bash
   ./qwen3-cli -m <model.gguf> -p "1+1=" -n 10 -t 0.0 -C 30
   ```
   Engine top tokens should match the FP32 reference at every step. For an
   external reference, compare greedy output against llama.cpp on the same
   prompt:
   ```bash
   .llama-cpp/build/bin/llama cli -m <model.gguf> \
       -p $'<|im_start|>user\n<prompt><|im_end|>\n<|im_start|>assistant\n' \
       -n 40 --temp 0 --top-k 1 -ngl 99 --no-jinja --single-turn --no-escape
   ```

## CLI Options

`tools/qwen3-cli.c` is the main chat/reasoning interface:

```bash
./qwen3-cli -m <model.gguf> -p "<prompt>" [-s "<system>"] [-n 128] [-t 0.7]
```

Notable flags:

- `-q` / `--quiet` — suppress non-error informational output (model load summary,
  KV cache size, generation statistics). Errors are still printed to stderr.
- `-d` — dump top-5 logits per generated token.
- `-L`, `-F`, `-P`, `-C N` — diagnostic reference comparisons.
- `-i` — interactive mode (experimental).

## Environment Switches

All switches are presence-based (`getenv(...) != NULL`); setting them to `0`
still enables them. To disable, unset the variable.

- `DS3_NO_CHUNK_PREFILL=1` — disable chunked prefill and process the prompt
  token-by-token. Chunked prefill is on by default for prompts of 8 or more
  tokens; use this only for regression/diagnostics.
- `DS3_PREFILL_CHUNK_SIZE=N` — chunk size for prompt prefill (default: 512).
- `DS3_PREFILL_FALLBACK_TOKENS=N` — prompts shorter than this many tokens use
  the token-by-token prefill path (default: 8).
- `DS3_CHUNK_NO_GATHERED_MOE=1` — inside chunk prefill, fall back to per-token
  expert dispatch for the MoE FFN.
- `DS3_CHUNK_NO_BATCHED_MATMUL=1` — inside chunk prefill, fall back to per-token
  `vec_matmul_*_simd` for attention projections.
- `DS3_CHUNK_DEBUG=1` — run chunk-vs-token divergence checks during prefill
  (slow).
- `DS3_USE_GPU_ONLY_MOE=1` — experimental fused 1-CB MoE for decode (currently
  slower than the default per-expert dispatch).
- `DS3_NO_OVERLAP=1` — disable the `post(l-1)+pre(l)` overlapped decode
  schedule.
- `DS3_USE_BASELINE_ATTN=1` — use the non-SIMD attention decode kernel for
  regression/diagnostics.
- `DS3_USE_SHARED_EXPERT=1` — enable the optional shared-expert branch in the
  MoE FFN. Disabled by default to match llama.cpp's qwen3moe graph.
- `DS3_DEBUG_FORWARD=1` — log the hidden state and sampled/logit top-1 at the
  start/end of every `forward_token()` call.
- `DS3_METAL_PROFILE=1` — print per-phase command-buffer and GPU/CPU-wait
  timings to stderr.

## Architecture Notes

### Decode schedule (default)

Layer-sequential, overlapped:

```
CB 0: pre(0)  + topk(0)
CB 1: post(0) + pre(1) + topk(1)
...
CB L: post(L-1) + output projection
```

`post(l)` (FFN) and `pre(l+1)` (attention + router) live in the same command
buffer. Residual adds are done on the GPU (`vec_add_f32`). This is required for
correctness; the old 2-CB schedule that ran all attention first and all FFN
second reused the post-attention residual as the next layer input and produced
wrong logits.

### Chunk prefill schedule (default)

Also layer-sequential inside the chunk:

```
CB 0: pre(0)  + topk(0)
CB 1: post(0) + pre(1) + topk(1)
...
CB L: post(L-1) + output projection
```

Attention projections use batched quantized SIMD matmul. The MoE FFN uses the
gathered path by default.

### Gathered MoE

For each layer in the chunk:

1. Build per-expert token lists on the CPU from the already-read router top-k
   results.
2. Upload the flat `ids` / `scores` lists to
   `buf_moe_gather_ids` / `buf_moe_gather_scores`.
3. For each active expert:
   - `gather_rows_f32`: collect selected rows from `buf_norm` into
     `buf_moe_expert_down`.
   - batched matmul: `gate = gathered @ W_gate^T` → `buf_moe_hidden`.
   - batched matmul: `up   = gathered @ W_up^T`   → `buf_moe_expert_up`.
   - `silu_mul`: `hidden = silu(gate) * up` in-place into `buf_moe_expert_up`.
   - batched matmul: `down = hidden @ W_down^T`  → `buf_moe_expert_down`.
   - `scatter_add_weighted_f32`: accumulate `score * down` back into
     `buf_ffn_out`.
4. Add residual: `buf_hidden = buf_ffn_out + buf_residual`.

The shared expert, when present, is a plain batched matmul over all `M` tokens
without gather/scatter.

### mmap zero-copy

`token_embd.weight` and `output.weight` are not dequantized into FP32 host/GPU
buffers. Instead:

- `buf_token_embd_view` is a zero-copy GPU view of the quantized mmap.
- Token lookup dequantizes only the needed rows on the CPU into a temporary
  buffer, then writes them to `buf_hidden`.
- Output projection dispatches the appropriate quantized `vec_matmul_*_simd`
  kernel directly against the mmap view.

This saves several GB of allocations for `Qwen3-30B-A3B`.

## KV Cache Service Integration

The daemon can use the external Rust `ds3-kv-cache-svc` for cross-session / cross-daemon prefix caching instead of managing KV locally.

Run the service in tiered mode:

```bash
/Users/wangyang/kv_cache/target/debug/ds3-kv-cache-svc \
    --tiered \
    -s /tmp/ds3-kv-cache.sock \
    -p /tmp/ds3-kv-cache-dir \
    -m 1073741824 \
    --max-daemons 2 \
    --heartbeat-timeout-secs 30
```

Run the daemon connected to the service:

```bash
./qwen3-engine-daemon \
    -m /path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
    -s /tmp/qwen3-engine.sock \
    -c 2048 \
    -k service \
    -K /tmp/ds3-kv-cache.sock \
    -i daemon1
```

Key flags:

- `-k service` — use the Rust KV Cache Service provider.
- `-K <sock>` — Unix Domain Socket path of the Rust service.
- `-i <daemon-id>` — unique daemon id; service uses this to isolate SHM arenas and heartbeat leases.
- `--max-daemons` on the service side limits concurrent daemon arenas.
- `--heartbeat-timeout-secs` controls how long the service waits before cleaning up a vanished daemon's sessions and migrating its Global blocks.

When a second daemon (`-i daemon2`) looks up a prefix owned by `daemon1`, the Rust service returns `daemon1`'s SHM arena name; `daemon2` maps that arena read-only (`O_RDONLY` + `PROT_READ`) to access the shared KV.

### Service-side integration tests

```bash
cd /Users/wangyang/kv_cache
cargo test
```

### Daemon-side integration tests

```bash
cd /Users/wangyang/metal_demo
python3 tests/run_kv_cache_service_test.py
python3 tests/test_two_daemon_cross_read.py
python3 tests/test_milestone3_acceptance.py
python3 tests/test_performance_baseline.py   # longer benchmark run
```

## Performance Notes

- See [`docs/BENCHMARK.md`](../docs/BENCHMARK.md) for reproducible commands,
  expected numbers, and measurement caveats.
- Absolute `tok/s` and prefill latency vary with thermal state and OS GPU
  scheduling. Run benchmarks several times and discard outliers.
- For short prompts the token-by-token path can be competitive because chunk
  prefill has fixed per-layer overhead. The crossover is typically above a few
  dozen tokens.
- Gathered MoE primarily reduces CPU dispatch overhead inside chunk prefill;
  GPU time is largely unchanged. The win grows with prompt length.

## Model Constants

Primary target (`src/ds3.h`):

```c
#define DS3_N_LAYER       48
#define DS3_N_EMBD        2048
#define DS3_N_HEAD        32
#define DS3_N_HEAD_KV     4
#define DS3_HEAD_DIM      128
#define DS3_N_EXPERT      128
#define DS3_N_EXPERT_USED 8
#define DS3_N_FF_EXP      768
#define DS3_N_FF_SHARED   6144
#define DS3_N_VOCAB       151936
```

These are hard-coded. Supporting a different model requires updating the
constants and verifying tensor shapes in `ds3_engine_init`.
