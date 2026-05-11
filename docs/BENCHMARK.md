# Benchmarks — Qwen3 Metal Engine

These numbers were measured on the current code (single-commit `main`) with a
real `Qwen3-30B-A3B-Q4_K_M.gguf` checkpoint. They are **not** extrapolated from
older phase reports.

## Reference environment

- Hardware: Apple MacBook Pro with M3 Max, 48 GB unified memory
- OS: macOS 14.5+, Metal 3
- Build: `make qwen3-cli` (`cc -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal`)
- Model: `Qwen3-30B-A3B-Q4_K_M.gguf` (~17 GB on disk)
- Storage: fast external SSD
- Context: default 4096 tokens
- Methodology: each benchmark was run multiple times; the first run after
  process start was discarded because it is dominated by OS file-cache / paging
  warm-up. Reported values are from warm-cache runs.

## Build

```bash
make qwen3-cli
```

No Xcode is required; Metal kernels are compiled at runtime from
`metal/*.metal`.

## Model load

With the OS file cache warm, loading the model takes roughly **0.03–0.05 s**.
The weights are mmap'd directly from the GGUF, so the engine does not create a
second in-memory copy. From a cold external SSD the first load can take several
seconds while the OS pages the ~17 GB file.

## Decode (token generation)

Decode speed depends strongly on how much KV cache has already been filled.
The numbers below use greedy sampling (`-t 0.0`) so they are deterministic and
do not stop early on end tokens.

### Short-context decode

Prompt is only a few tokens, so the KV cache is small for most of the
generation.

```bash
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
            -p "1+1=" -n 128 -t 0.0
```

Warm-cache results:

```text
[gen] 128 tokens in 6.89 s = 18.58 tok/s
[gen] 128 tokens in 6.87 s = 18.62 tok/s
[gen] 128 tokens in 6.90 s = 18.56 tok/s
```

**Short-context greedy decode: ~18.6 tok/s** (M3 Max, Q4_K_M).

### Decode after a long prompt

Here the engine first prefills a prompt of the given length and then generates
64 tokens. The `[gen]` line printed by `qwen3-cli` includes both prefill and
decode, so the table below reports the **decode-only** time:

```
decode_time = real_wall_time - model_load_time - prefill_wall_time
```

Prefill wall time is taken from the `DS3_METAL_PROFILE=1` prompt profile; model
load is negligible on a warm cache (~0.03 s) and is subtracted off.

```bash
DS3_METAL_PROFILE=1 ./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
                                 -p "$(cat prompt_512.txt)" -n 64 -t 0.0
```

| Prompt tokens | Prefill wall | 64-token decode time | Decode tok/s |
|---------------|--------------|----------------------|--------------|
| ~64           | ~1.7 s       | ~3.4 s               | **~19**      |
| ~256          | ~6.1 s       | ~3.9 s               | **~16**      |
| ~512          | ~14.6 s      | ~4.7 s               | **~14**      |

These numbers are consistent with the short-context headline: the first decode
token after a 64-token prompt still sees a KV cache of only ~64 positions, so
its speed is close to the short-prompt average. As the prefilled context grows
to 256 and 512 positions, the decode attention kernel reads more KV data per
step and throughput falls.

## Prompt prefill

Chunked prefill is enabled by default for prompts of 8 or more tokens. The
effective prefill throughput is the prompt token count divided by the prefill
wall time reported by `DS3_METAL_PROFILE=1`.

| Prompt tokens | Prefill wall | Effective tok/s |
|---------------|--------------|-----------------|
| ~61           | ~1.7 s       | **~36**         |
| ~251          | ~6.2 s       | **~41**         |
| ~511          | ~14.9 s      | **~34**         |

For short prompts (below a few dozen tokens) the token-by-token fallback can be
competitive. To compare:

```bash
# chunked (default)
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
            -p "$(cat prompt_256.txt)" -n 32 -t 0.0

# token-by-token fallback
DS3_NO_CHUNK_PREFILL=1 ./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
                            -p "$(cat prompt_256.txt)" -n 32 -t 0.0
```

Both must produce byte-identical greedy output.

## Gathered vs per-token MoE

Inside chunk prefill the MoE FFN uses a gathered expert path by default.

```bash
# gathered (default)
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
            -p "$(cat prompt_256.txt)" -n 32 -t 0.0

# per-token fallback
DS3_CHUNK_NO_GATHERED_MOE=1 ./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
                            -p "$(cat prompt_256.txt)" -n 32 -t 0.0
```

The gathered path is generally faster for prompts above a few dozen tokens. On a
256-token prompt + 32-token greedy decode it reduces wall time by roughly
**20%** versus the per-token fallback, while producing byte-identical output.

## Unit-test sanity checks

```bash
make test
./test_metal
./test_matmul_quant_metal
./test_attention_metal
./test_moe_metal
./test_reference
```

All tests pass in a few seconds. This is the fastest way to verify that a
build or environment change has not regressed numerical correctness.

## Correctness / quality checks

Before trusting speed numbers, run the reference comparisons:

```bash
# Layer 0 full forward vs FP32 reference
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf -p "hi" -L

# Full-model one-token comparison
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf -p "hi" -F

# Multi-step decode drift check (first 30 prompt+generated tokens)
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
            -p "1+1=" -n 10 -t 0.0 -C 30
```

For an external baseline, compare greedy output against llama.cpp:

```bash
# Adjust the path to match your local llama.cpp build.
.llama-cpp/build/bin/llama-cli \
    -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
    -p $'<|im_start|>user\n<prompt><|im_end|>\n<|im_start|>assistant\n' \
    -n 40 --temp 0 --top-k 1 -ngl 99 --no-jinja --single-turn --no-escape
```

## Profiling

Set `DS3_METAL_PROFILE=1` to print per-phase command-buffer counts and GPU/CPU
wait timings to stderr:

```bash
DS3_METAL_PROFILE=1 ./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
                                 -p "Hello" -n 64 -t 0.0
```

Example output for a short prompt + decode step:

```text
[profile prompt] cb_count=49 total_gpu_ms=278.745 total_cpu_wait_ms=710.224
[profile step=0] cb_count=50 total_gpu_ms=35.809 total_cpu_wait_ms=44.358
...
```

The prompt profile separates the one-time prefill cost from the per-token
decode steps.

## Variance and caveats

- **First run after process start is slow.** The OS pages weight data from the
  SSD into GPU-resident memory; discard that run.
- **External SSD warmth matters.** Warm the OS file cache with a sequential read
  (`dd if=model.gguf of=/dev/null bs=1m`) before timing.
- **Thermal throttling** can reduce sustained throughput by 10–20% after minutes
  of continuous generation.
- **Decode speed is context-dependent.** The headline ~18.6 tok/s figure is for
  short prompts; it falls to ~13 tok/s once the KV cache holds ~512 tokens.
- **Quantization changes numbers.** Q6_K / Q8_0 variants will be slower than
  Q4_K_M because they move more bytes per matmul.
