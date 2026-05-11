# Qwen3 Metal Engine Roadmap

This document tracks the project direction. The current focus is:

> **P1a — Ship a stable, open-source-ready Qwen3-30B-A3B release.**  
> Decode optimization (P0) is closed after hitting its practical ceiling. 235B
> adaptation and chat/UX polish remain intentionally delayed until after the 30B
> release.

---

## ✅ Completed

### Phase 4a — Chunk Prefill
- [x] Layer-sequential chunk prefill with overlapped schedule.
- [x] Batched quantized attention projections inside the chunk.
- [x] SIMD-group batched matmul kernels that match vec-kernel round-off.
- [x] Chunk vs token correctness validation.

### Phase 4b — Gathered MoE
- [x] Gather/scatter kernels for batched routed experts.
- [x] `forward_chunk_post_gathered` that groups tokens by selected expert.
- [x] Shared expert handled as plain batched matmul.
- [x] Fallback switch `DS3_CHUNK_NO_GATHERED_MOE=1`.
- [x] Correctness and performance validation.

### Supporting Work
- [x] mmap zero-copy for `token_embd` and `output.weight`.
- [x] `AGENTS.md` with architecture, build, test, and env-switch notes.

---

## ✅ P0 — Decode Performance Optimization (Closed)

Current decode baseline is **~17 tok/s** on M3 Max + `Qwen3-30B-A3B-Q4_K_M.gguf`.

### Conclusion

P0 is closed as "bounded by architecture/Metal scheduling overhead":

- **2-CB cross-layer decode is mathematically impossible.** Transformer layers
  are strictly sequential: layer `l+1` attention must consume the *FFN output*
  of layer `l`. Any schedule that runs all attention before all FFN (even with
  saved pre-FFN hidden/residual) violates this dependency and produces incorrect
  output. This was confirmed empirically: the first token matched, subsequent
  tokens diverged.
- **GPU-only router top-k + fused 1-CB mega-kernel was already disproven**
  (Phase 3: **5.5 tok/s**, 63% slower than default).
- **`DS3_USE_GPU_ONLY_MOE=1` was already disproven** (Phase 3: 47% slower than
  per-expert dispatch).
- The only correct low-CB-count path is the existing **overlapped layer-sequential
  schedule** (`post(l)` + `pre(l+1)` share a command buffer), which still needs
  ~49 CB flushes per token and is already near the ceiling.

### P0 TODOs
- [x] Investigate low-CB-count decode schedules.
- [x] Prove the cross-layer 2-CB schedule is structurally incorrect.
- [x] Document ~17 tok/s as the practical decode ceiling for this architecture.

**Outcome:** Decode optimization is downgraded to "nice to have". The only
known path to significantly exceed this ceiling is **speculative decoding**
(P4), which requires a separate draft model.

---

## 📦 P1 — Release Polish for Qwen3-30B-A3B

🚧 **Current focus: P1b.**

P1a is complete. P1 is split into **P1a (must ship before open-source)** and
**P1b (after release)**.

### P1a — Release Blockers ✅

These must be done before publishing the repo.

- [x] **LICENSE**
- [x] **Clean `.gitignore`**
- [x] **README.md** with:
  - Project overview and scope (what it is / isn't).
  - System requirements (Apple Silicon, macOS 14+, ~20 GB free RAM for 30B Q4_K_M).
  - Build instructions.
  - Model download guide (HuggingFace repo + exact GGUF filename for
    Qwen3-30B-A3B-Q4_K_M).
  - Quick-start examples.
- [x] **Clean build verification**
  - Confirm `make qwen3-cli` and `make test_*` work from a fresh clone.
- [x] **Stop generation on end tokens**
  - Recognize `<|im_end|>` and `</think>` and stop early.
  - Currently generation always produces exactly `n` tokens, which inflates
    benchmark numbers and produces trailing garbage.
- [x] **Make chunk prefill the default**
  - `forward_chunk` is correct and much faster (130-token prompt: 18 s → 2.5 s).
  - Short prompts should automatically fall back to the token path.
  - Invert the env switch so users opt *out* with
    `DS3_CHUNK_NO_BATCHED_MATMUL=1` or similar, not opt in.
- [x] **End-to-end generation quality validation**
  - Compare generated text against llama.cpp or transformers on the same prompt
    and temperature (argmax first, then sampling with fixed seed).
  - **Root cause of the repetitive-loop bug:** RoPE was implemented in the
    adjacent-pair (LLaMA) layout, but Qwen3 uses the GPT-NeoX layout
    `(i, i + head_dim/2)`. Fixed in `metal/rope.metal` and `src/ds3_reference.c`.
- [x] **Q4_K_M dequantization ground-truth validation**
  - Current checks only compare GPU output vs CPU dequant output. Add a check
    against known-good ground-truth weights/logits (e.g. from PyTorch or
    llama.cpp).

### P1b — Post-Release Polish

These improve contributor/user experience but can ship after the initial release.

- [ ] **CI configuration** (GitHub Actions on macOS with Metal).
- [ ] **Scripted validation suite**
  - `-F` full-model reference comparison.
  - Chunk vs token generation consistency.
  - Gathered vs per-token MoE consistency.
  - All `make test_*` Metal unit tests.
- [x] **`docs/BENCHMARK.md`** with reproducible commands and expected numbers.
- [ ] **CLI usability improvements**
  - Better `--help`.
  - Friendly errors for missing model, GGUF load failure, context exceeded,
    Metal init failure.
  - [x] `-q` / `--quiet` flag to suppress non-error informational output.
  - [ ] Optional output-file flag.
- [x] **Cleanup env switches**
  - Removed `DS3_NO_SYNC_LAYERS=1` and its broken cross-layer batch path.
  - Removed `DS3_USE_2CB_DECODE=1` and the incorrect two-command-buffer decode
    schedule.
  - Kept only meaningful, documented switches.
- [ ] **Mark interactive chat mode (`-i`) as `[EXPERIMENTAL]`** or remove it until P4.

**Acceptance (P1a):** A new contributor can clone, `make qwen3-cli`, download the
recommended GGUF, and run inference following only `README.md`; generation stops
at end tokens and quality matches a known-good reference.

---

## 🔮 P2 — Long Context & Memory Efficiency

Goal: support longer contexts efficiently.

- [ ] **KV cache quantization (FP8 / INT8)**
  - Reduces memory footprint for long sequences.
- [ ] **Support context lengths beyond 4096**
  - Add Flash Attention style tiled attention for prefill to avoid O(n²) memory
    and compute growth.
  - Optimize decode attention for seq_len > 8192.

**Explicitly out of scope for 30B:** Page-based KV cache allocation. For the
30B model with 48 layers × 4 KV heads × 128 dim × FP16 × 4096 context, the KV
cache is only ~384 MB and can be allocated contiguously. Page-based allocation
is over-engineering at this scale.

**Gate:** After P0 and P1a are complete.

---

## 🏗️ P3 — Multi-Model / 235B Adaptation (Delayed)

Goal: make the engine model-shape-agnostic so it can run Qwen3-235B-A22B and
other Qwen3 variants.

> **Scope warning:** This is a large refactor. `DS3_N_LAYER`, `DS3_N_EMBD`,
> `DS3_N_HEAD`, `DS3_N_HEAD_KV`, and related macros are used throughout
> `src/ds3_engine.c`, `src/ds3_metal.m`, `metal/*.metal`, and `tests/`.
> Converting them to runtime fields will touch almost every file.

### P3 TODOs
- [ ] Convert compile-time model feature flags to runtime fields:
  - `has_qk_norm`
  - `has_router_bias`
  - any other `#if DS3_HAS_*` / `-DDS3_MODEL_235B` branches
- [ ] Read `n_layer / n_embd / n_head / n_kv_head / n_ff_exp / n_ff_shared` from
  GGUF metadata instead of macros.
- [ ] Resize all scratch buffers dynamically based on model dimensions.
- [ ] **Validate shared expert path with real 235B weights**
  - The 30B GGUF has no shared expert tensors, so the shared-expert code path
    has not been exercised by a real model.
  - llama.cpp's qwen3moe graph currently ignores the optional shared-expert
    branch; the engine defaults to the same behaviour (`DS3_USE_SHARED_EXPERT=1`
    can opt back in if a future model/ground-truth requires it).
- [ ] **Model-view overlap for >32 GB MTLBuffer limit**
  - Single `MTLBuffer` is capped at 32 GB; 235B weights will require splitting
    the mmap view into multiple buffers or using overlapping views.
- [ ] Validate 235B model loading and forward pass.

**Gate:** After the 30B open-source release.

---

## 💬 P4 — Chat Experience & Feature Enhancements (Low Priority)

Goal: make the interactive CLI pleasant to use.

- [x] ~~Improve chat template to include `<think>\n` assistant prefix.~~
  - The official Qwen3 chat template uses only `"assistant\n"` as the generation
    prompt; the model emits `<think>\n` itself. Forcing `<think>\n` into the
    prefix changes the conditioning and was empirically worse. The template now
    matches the GGUF metadata.
- [ ] Add repetition penalty.
- [ ] Add top-p / top-k sampling.
- [ ] Streaming output.
- [ ] Optional tool-call support.
- [ ] **Speculative decoding (long-term)**
  - Draft small model + verify with main model for higher decode tok/s.
  - This is the most proven path to push decode speed well beyond the current
    ceiling, but requires a separate draft model and verify kernel.

**Gate:** After P0, P1a, and initial P2 work.

---

## Notes

- Priority order is intentionally **Performance → Release → Scale → Multi-model → UX**.
- 235B and chat polish are explicitly behind the 30B release.
- As work completes, move items from this file into `AGENTS.md` or remove them.
- **P1a decode quality fix (NeoX RoPE):** Qwen3 uses GPT-NeoX style RoPE
  `(i, i + head_dim/2)`, not the adjacent-pair LLaMA layout. This only shows
  up in multi-token decode because seq_pos=0 is the identity for both layouts.
