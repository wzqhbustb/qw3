# Qwen3 Metal Engine

A minimal, native Metal inference engine for **Qwen3-30B-A3B** on Apple Silicon
Macs. It loads GGUF checkpoints and runs the full transformer on the GPU using
hand-written Metal kernels.

> **Status:** beta quality. Chunk prefill, gathered MoE, and ~18.6 tok/s greedy
> decode are working on an M3 Max with `Qwen3-30B-A3B-Q4_K_M.gguf`. Current focus
> is P1a release polish ahead of open-source. See [`ROADMAP.md`](ROADMAP.md).

## What it is / isn't

- **Is:** a from-scratch Metal engine for Qwen3-30B-A3B (GGUF), with chunked
  prefill, quantized MoE, and a tiny CLI.
- **Is not:** a general llama.cpp replacement. It only supports the 30B-A3B
  shape hard-coded in `src/ds3.h`. Multi-model support (235B, etc.) is P3.
- **Is not:** a server or chat product. The CLI is minimal; streaming, tool
  calls, and speculative decoding are future work.

## System requirements

- Apple Silicon Mac (M1/M2/M3/M4 series).
- macOS 14.5+ with Metal 3.
- No Xcode required; the build uses `cc` and the system Metal frameworks.
- RAM:
  - **Qwen3-30B-A3B-Q4_K_M.gguf** (~17 GB on disk) needs **~20 GB** free
    unified memory for a 4096-token context.
  - Q8_0 / BF16 variants need proportionally more memory.

## Build

```bash
make qwen3-cli
```

This produces `./qwen3-cli`. To build the unit-test binaries:

```bash
make test
```

## Download a model

The recommended checkpoint for most users is the official Qwen GGUF:

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF \
    Qwen3-30B-A3B-Q4_K_M.gguf \
    --local-dir ./models

# Or with curl/wget
mkdir -p models
curl -L -o models/Qwen3-30B-A3B-Q4_K_M.gguf \
    https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
```

Other quantizations (Q6_K, Q8_0) from the same repo also work, but Q4_K_M is
fastest on consumer Apple Silicon while keeping good quality. If the repo name
changes, check the [Qwen HuggingFace page](https://huggingface.co/Qwen) for the
latest GGUF location.

## Quick start

```bash
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf \
            -p "What is the capital of France?" \
            -n 128 -t 0.7
```

Options:

- `-m PATH` — path to the GGUF model (required).
- `-p TEXT` — user prompt (required).
- `-s TEXT` — optional system prompt.
- `-n N`    — maximum tokens to generate (default: 128).
- `-t TEMP` — sampling temperature, `0` = argmax (default: 0.7).
- `-c CTX`  — KV cache / context size (default: 4096).
- `-d`      — dump top-5 logits per generated token.
- `-C N`    — run a multi-step decode comparison against the FP32 reference for the first `N` prompt+generated tokens (diagnostic).
- `-i`      — interactive mode (experimental, may change or be removed).
- `-q`      — quiet mode: suppress non-error informational output.

Generation stops automatically when the model emits `<|im_end|>` or
`</think>`.

## Environment variables

These are mainly for debugging or benchmarking. Normal users do not need them.

| Variable | Effect |
|----------|--------|
| `DS3_NO_CHUNK_PREFILL=1` | Disable chunked prefill and process the prompt token-by-token. |
| `DS3_PREFILL_CHUNK_SIZE=N` | Chunk size for prompt prefill (default: 512). |
| `DS3_CHUNK_NO_GATHERED_MOE=1` | Use the non-gathered MoE path for chunk prefill. |
| `DS3_CHUNK_DEBUG=1` | Run chunk-vs-token divergence checks during prefill (slow). |
| `DS3_DEBUG_FORWARD=1` | Log the hidden state at the start/end of every `forward_token()` call. |
| `DS3_METAL_PROFILE=1` | Print per-phase command-buffer and GPU/CPU-wait timings to stderr. |
| `DS3_USE_SHARED_EXPERT=1` | Enable the optional shared-expert branch in the MoE FFN (disabled by default to match llama.cpp's qwen3moe graph). |

## Project layout

```text
.
├── metal/           # Metal shader sources
├── src/             # Engine, tokenizer, GGUF loader, reference CPU kernels
├── tests/           # Unit and integration tests
├── tools/           # qwen3-cli and helper tools
├── docs/            # Benchmarks and design notes
│   └── BENCHMARK.md # Reproducible performance numbers
├── ROADMAP.md       # Current direction and completed milestones
└── LICENSE          # MIT
```

## Tests

Most tests require a GGUF model or are interactive:

```bash
make test

# Tokenizer / chat-template check
./test_tokenizer models/Qwen3-30B-A3B-Q4_K_M.gguf

# GGUF loader check
./test_gguf models/Qwen3-30B-A3B-Q4_K_M.gguf

# Metal kernel sanity checks (no model needed)
./test_metal
./test_matmul_quant_metal
./test_attention_metal
./test_moe_metal

# Full layer-0 reference comparison
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf -p "hi" -L

# Full-model reference comparison for one token
./qwen3-cli -m models/Qwen3-30B-A3B-Q4_K_M.gguf -p "hi" -F
```

## Performance

Measured numbers on an M3 Max with `Qwen3-30B-A3B-Q4_K_M.gguf` and a 4096
context (warm OS file cache, first run discarded):

- **Decode:** ~18.6 tok/s for short prompts; slows to ~14 tok/s once the KV
  cache holds ~512 tokens.
- **Chunk prefill:** ~34–41 tok/s effective throughput for 61–511 token prompts.

Chunk prefill is enabled by default for prompts of 8 or more tokens. See
[`docs/BENCHMARK.md`](docs/BENCHMARK.md) for the exact commands, full tables,
and variance notes.

## License

MIT — see [`LICENSE`](LICENSE).

The engine code is MIT-licensed. Model weights (Qwen3) are subject to their own
model license from Alibaba Cloud.
