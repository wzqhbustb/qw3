/*
 * ds3.h — Qwen3 Engine: Public API + Model Constants
 *
 * Compile with -DDS3_MODEL_235B to target Qwen3-235B-A22B.
 * Default (no flag) targets Qwen3-30B-A3B for Phase 1 development.
 *
 * Sources:
 *   - qwen3-engine-implementation-plan.md (Section 3.0, 3.4)
 *   - qwen3-235b-engine-plan.md
 *   - qwen3-engine-learning-guide.md (Section 8)
 *
 * Model constants were derived from config.json/GGUF metadata and verified
 * against the Qwen3-30B-A3B GGUF for Phase 1.
 */

#ifndef DS3_H
#define DS3_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Logging helpers
 * ============================================================================
 * `ds3_log_quiet` controls non-error informational output.  Set it via
 * `ds3_set_quiet()` (e.g. from a `--quiet` CLI flag).  Error messages should
 * always use plain `fprintf(stderr, ...)` so they are never suppressed. */

extern int ds3_log_quiet;

static inline void ds3_set_quiet(int quiet) { ds3_log_quiet = quiet ? 1 : 0; }

#define ds3_print_info(...) do { if (!ds3_log_quiet) printf(__VA_ARGS__); } while (0)
#define ds3_log_info(...)   do { if (!ds3_log_quiet) fprintf(stderr, __VA_ARGS__); } while (0)

/* ============================================================================
 * 1. Model Architecture Constants
 * ============================================================================ */

/* Compile-time model selection */
#if defined(DS3_MODEL_30B) && defined(DS3_MODEL_235B)
#error "Cannot define both DS3_MODEL_30B and DS3_MODEL_235B"
#endif

#if !defined(DS3_MODEL_30B) && !defined(DS3_MODEL_235B)
#define DS3_MODEL_30B 1
#endif

#ifdef DS3_MODEL_30B

/* Qwen3-30B-A3B — Phase 1 development target (48 GB Mac) */
#define DS3_MODEL_NAME      "Qwen3-30B-A3B"
#define DS3_N_LAYER         48          /* Transformer layers */
#define DS3_N_EMBD          2048        /* hidden_size */
/* Attention Q heads — verified from GGUF metadata */
#define DS3_N_HEAD          32
#define DS3_N_HEAD_KV       4           /* GQA KV heads */
#define DS3_HEAD_DIM        128         /* per-head dimension */
#define DS3_N_ROT           128         /* RoPE apply dims (full dimension for Qwen3) */

#define DS3_N_EXPERT        128         /* Total MoE experts */
#define DS3_N_EXPERT_USED   8           /* Top-k selected per token */
#define DS3_N_FF_EXP        768         /* Routed expert intermediate size */
#define DS3_N_FF_SHARED     6144        /* Shared expert intermediate size (3× N_EMBD) */

#define DS3_N_VOCAB         151936      /* Vocab size (GPT-2 BPE) */
#define DS3_ROPE_THETA      1000000.0f  /* High-freq base for long-context extrapolation */
#define DS3_NORM_TOPK_PROB  true        /* Re-normalize top-8 router weights to sum=1 */
#define DS3_NORM_EPS        1e-6f       /* RMSNorm epsilon */

/* Phase 4a: chunked prompt prefill. Process the prompt in chunks of this many
 * tokens to exploit batched matrix multiplies. Must be <= n_ctx. */
#define DS3_PREFILL_CHUNK_SIZE 512

/* Prompts shorter than this use the token-by-token prefill path.  Chunk prefill
 * is correct for M=1 but has higher dispatch overhead than the simple path. */
#define DS3_PREFILL_FALLBACK_TOKENS 8

#define DS3_HAS_ATTN_BIAS   false       /* Qwen3 attention does not use bias */
#define DS3_HAS_QK_NORM     1           /* verified: attn_q_norm.weight & attn_k_norm.weight exist in GGUF */
#define DS3_HAS_ROUTER_BIAS 0           /* verified: ffn_gate_inp.bias not present in GGUF */

#elif defined(DS3_MODEL_235B)

/* Qwen3-235B-A22B — Phase 2 target (128 GB Mac) */
#define DS3_MODEL_NAME      "Qwen3-235B-A22B"
#define DS3_N_LAYER         94
#define DS3_N_EMBD          4096
/* Attention Q heads — 30B verified; 235B from implementation-plan */
#define DS3_N_HEAD          64
#define DS3_N_HEAD_KV       4
#define DS3_HEAD_DIM        128
#define DS3_N_ROT           128         /* full-dimension RoPE */

#define DS3_N_EXPERT        128
#define DS3_N_EXPERT_USED   8
#define DS3_N_FF_EXP        1536        /* 2× 30B value */
#define DS3_N_FF_SHARED     12288       /* 2× 30B value (3× N_EMBD) */

#define DS3_N_VOCAB         151936
#define DS3_ROPE_THETA      1000000.0f
#define DS3_NORM_TOPK_PROB  true
#define DS3_NORM_EPS        1e-6f

#define DS3_HAS_ATTN_BIAS   false
#define DS3_HAS_QK_NORM     0           /* 235B variant does not use Q/K RMSNorm */
#define DS3_HAS_ROUTER_BIAS 0

#else
#error "Define either DS3_MODEL_30B or DS3_MODEL_235B"
#endif

/*
 * Special Token IDs — shared across both models (same vocab).
 *
 * Qwen3 convention:
 *   BOS = <|beginoftext|> = 151643  (add_bos_token=false in Qwen3 config)
 *   EOS = <|im_end|>       = 151645  (also serves as end-of-text)
 *   PAD = <|endoftext|>    = 151643  (shared with BOS)
 *   IM_START = <|im_start|> = 151644
 *   IM_END   = <|im_end|>   = 151645 (same as EOS)
 */
#define DS3_BOS_ID          151643
#define DS3_EOS_ID          151645
#define DS3_PAD_ID          151643
#define DS3_IM_START_ID     151644
#define DS3_IM_END_ID       151645

/* ============================================================================
 * 2. Derived Constants (computed from Section 1)
 * ============================================================================ */

#define DS3_N_HEAD_PER_KV   (DS3_N_HEAD / DS3_N_HEAD_KV)   /* GQA group size: 8 (30B) or 16 (235B) */

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert((DS3_N_HEAD % DS3_N_HEAD_KV) == 0, "N_HEAD must be divisible by N_HEAD_KV for GQA");
#endif
#define DS3_Q_DIM           (DS3_N_HEAD * DS3_HEAD_DIM)    /* Q projection output dim: 4096 (30B) or 8192 (235B) */
#define DS3_KV_DIM          (DS3_N_HEAD_KV * DS3_HEAD_DIM) /* KV projection output dim: 512 (both) */

/*
 * Attention weight shapes
 * NOTE: ROWS = output dimension, COLS = input dimension (the matrix is [out_dim, in_dim] in GGUF)
 */
#define DS3_ATTN_Q_ROWS     DS3_Q_DIM    /* out dim = num_heads * head_dim */
#define DS3_ATTN_Q_COLS     DS3_N_EMBD   /* in dim  = hidden_size */
#define DS3_ATTN_K_ROWS     DS3_KV_DIM   /* out dim = kv_heads * head_dim */
#define DS3_ATTN_K_COLS     DS3_N_EMBD   /* in dim  = hidden_size */
#define DS3_ATTN_V_ROWS     DS3_KV_DIM   /* out dim = kv_heads * head_dim */
#define DS3_ATTN_V_COLS     DS3_N_EMBD   /* in dim  = hidden_size */
#define DS3_ATTN_O_ROWS     DS3_N_EMBD   /* out dim = hidden_size */
#define DS3_ATTN_O_COLS     DS3_Q_DIM    /* in dim  = num_heads * head_dim */

/* Routed expert weight shapes (per expert) */
#define DS3_EXPERT_GATE_UP_ROWS     DS3_N_FF_EXP    /* gate and up are separate tensors in Qwen3 GGUF */
#define DS3_EXPERT_GATE_UP_COLS     DS3_N_EMBD
#define DS3_EXPERT_DOWN_ROWS        DS3_N_EMBD
#define DS3_EXPERT_DOWN_COLS        DS3_N_FF_EXP

/* Shared expert weight shapes */
#define DS3_SHARED_GATE_UP_ROWS     DS3_N_FF_SHARED
#define DS3_SHARED_GATE_UP_COLS     DS3_N_EMBD
#define DS3_SHARED_DOWN_ROWS        DS3_N_EMBD
#define DS3_SHARED_DOWN_COLS        DS3_N_FF_SHARED

/* Router */
#define DS3_ROUTER_ROWS     DS3_N_EXPERT    /* gate logits: [N_EXPERT] */
#define DS3_ROUTER_COLS     DS3_N_EMBD      /* from hidden state */

/* KV cache size formulas (bytes, FP16 = 2 bytes per element) */
/* K + V, each FP16 = 2 bytes → 4 bytes total per (layer, token, head_k, head_dim) */
#define DS3_KV_SIZE_PER_LAYER_TOKEN  (2 * DS3_N_HEAD_KV * DS3_HEAD_DIM * sizeof(uint16_t))
#define DS3_KV_SIZE_TOTAL(ctx_len)   ((size_t)(ctx_len) * DS3_N_LAYER * DS3_KV_SIZE_PER_LAYER_TOKEN)

/* Example pre-computed values for common context lengths */
#define DS3_KV_SIZE_4K      DS3_KV_SIZE_TOTAL(4096)
#define DS3_KV_SIZE_32K     DS3_KV_SIZE_TOTAL(32768)
#define DS3_KV_SIZE_128K    DS3_KV_SIZE_TOTAL(131072)

/* ============================================================================
 * 3. Quantization Format Constants (from llama.cpp ggml)
 * ============================================================================ */

#define DS3_QK_K    256     /* k-quants block size: 256 weights per block */
#define DS3_QK4_0   32
#define DS3_QK8_0   32

/*
 * GGUF type IDs — MUST match llama.cpp ggml_type enum exactly.
 * Source: ggml/include/ggml.h (ggml_type)
 */
typedef enum {
    DS3_TYPE_F32      = 0,
    DS3_TYPE_F16      = 1,
    DS3_TYPE_Q4_0     = 2,
    DS3_TYPE_Q5_0     = 6,      /* NOT 3 */
    DS3_TYPE_Q8_0     = 8,
    DS3_TYPE_Q2_K     = 10,     /* was 16 — CRITICAL FIX */
    DS3_TYPE_Q3_K     = 11,     /* was 17 */
    DS3_TYPE_Q4_K     = 12,     /* was 18 */
    DS3_TYPE_Q5_K     = 13,     /* was 19 */
    DS3_TYPE_Q6_K     = 14,     /* was 20 */
    DS3_TYPE_Q8_K     = 15,     /* was 21 */
    DS3_TYPE_IQ2_XXS  = 16,     /* was 28 — CRITICAL FIX */
    DS3_TYPE_IQ2_XS   = 17,     /* was 29 */
    DS3_TYPE_IQ3_XXS  = 18,     /* was 35 */
} ds3_type_t;

/* Bits-per-weight estimates for volume calculations */
#define DS3_BPW_F32      32.0f
#define DS3_BPW_F16      16.0f
#define DS3_BPW_Q4_0      4.5f   /* 18 bytes / 32 weights: half(2B) + 16×uint8(16B) */
#define DS3_BPW_Q5_0      5.5f   /* 22 bytes / 32 weights: half(2B) + 18×uint8(18B) */
#define DS3_BPW_Q8_0      8.5f   /* 34 bytes / 32 weights: half(2B) + 32×int8(32B) */
#define DS3_BPW_Q4_K      4.5f
#define DS3_BPW_Q2_K      2.625f
#define DS3_BPW_IQ2_XXS   2.0625f

/* ============================================================================
 * 4. Core Data Structures
 * ============================================================================ */

/* Tensor descriptor — points into mmap'd GGUF region */
typedef struct {
    const char  *name;          /* tensor name (e.g. "blk.0.attn_q.weight") */
    ds3_type_t   type;          /* quantization type */
    uint32_t     n_dims;        /* 1..4 */
    uint64_t     ne[4];         /* element counts per dimension (shape) */
    uint64_t     nb[4];         /* byte strides */
    void        *data;          /* pointer into mmap'd GGUF file */
    size_t       size;          /* total bytes */
} ds3_tensor_t;

/* Per-layer weight bundle */
typedef struct {
    ds3_tensor_t *attn_norm;        /* input RMSNorm weight [N_EMBD] */
    ds3_tensor_t *attn_q;           /* Q projection [N_EMBD, Q_DIM] */
    ds3_tensor_t *attn_q_norm;      /* Q RMSNorm [HEAD_DIM] — if HAS_QK_NORM */
    ds3_tensor_t *attn_k;           /* K projection [N_EMBD, KV_DIM] */
    ds3_tensor_t *attn_k_norm;      /* K RMSNorm [HEAD_DIM] — if HAS_QK_NORM */
    ds3_tensor_t *attn_v;           /* V projection [N_EMBD, KV_DIM] */
    ds3_tensor_t *attn_output;      /* O projection [Q_DIM, N_EMBD] */

    ds3_tensor_t *ffn_norm;         /* post-attention RMSNorm [N_EMBD] */
    ds3_tensor_t *ffn_gate_inp;     /* router: [N_EXPERT, N_EMBD] */
    ds3_tensor_t *ffn_gate_exps;    /* routed expert gate: [N_FF_EXP, N_EMBD, N_EXPERT] or packed */
    ds3_tensor_t *ffn_up_exps;      /* routed expert up:   [N_FF_EXP, N_EMBD, N_EXPERT] */
    ds3_tensor_t *ffn_down_exps;    /* routed expert down: [N_EMBD, N_FF_EXP, N_EXPERT] */

    /*
     * NOTE: Qwen3-30B GGUF does NOT have separate shared_expert tensors.
     * The shared expert weights are embedded within the routed expert
     * structure for this model variant (shared_* fields are NULL).
     */
    ds3_tensor_t *shared_expert_gate;   /* [N_FF_SHARED, N_EMBD] — optional, NULL in 30B */
    ds3_tensor_t *shared_expert_up;     /* [N_FF_SHARED, N_EMBD] — optional, NULL in 30B */
    ds3_tensor_t *shared_expert_down;   /* [N_EMBD, N_FF_SHARED] — optional, NULL in 30B */

    /* Optional per-layer tensors */
    ds3_tensor_t *ffn_gate_inp_bias;    /* router bias [N_EXPERT], if DS3_HAS_ROUTER_BIAS */
} ds3_layer_weights_t;

/* Global model weights */
typedef struct {
    ds3_tensor_t *token_embd;       /* [N_VOCAB, N_EMBD] */
    ds3_tensor_t *output_norm;      /* [N_EMBD] */
    ds3_tensor_t *output;           /* [N_VOCAB, N_EMBD]; NULL if tied to token_embd */

    ds3_layer_weights_t layers[DS3_N_LAYER];
} ds3_weights_t;

/* ============================================================================
 * 5. Inference State Structures (forward-pass temporaries)
 * ============================================================================ */

/* Activations for a single layer, single token */
typedef struct {
    float *x;           /* input / residual [N_EMBD] */
    float *xb;          /* post-norm [N_EMBD] */
    float *q;           /* Q [Q_DIM] */
    float *k;           /* K [KV_DIM] */
    float *v;           /* V [KV_DIM] */
    float *attn_out;    /* attention output [N_EMBD] */
    float *ffn_out;     /* MoE output [N_EMBD] */

    /* MoE temporaries */
    float *gate_logits;     /* [N_EXPERT] */
    float *expert_weights;  /* top-8 weights [N_EXPERT_USED] */
    int   *expert_indices;  /* top-8 indices [N_EXPERT_USED] */
    /*
     * SwiGLU needs gate AND up simultaneously: SiLU(x @ W_gate) * (x @ W_up).
     * Each tmp below holds ONE vector. A second scratch of same size is
     * needed at runtime (or fuse inside the kernel). [PHASE-1.4]
     */
    float *expert_tmp;          /* scratch for one routed expert's matvec [N_FF_EXP] */
    float *shared_expert_tmp;   /* scratch for shared expert's matvec [N_FF_SHARED] */

    /*
     * Attention decode scratch: Q @ K_cache^T scores.
     * Shape: [N_HEAD, max_seq_len]. Sized at engine open time (n_ctx).
     * Not used for prefill (FlashAttention tiles internally).
     */
    float *attn_score;
} ds3_layer_state_t;

/* KV cache entry — stored per layer, per sequence position */
typedef struct {
    /* Layout: [seq_len][N_HEAD_KV][HEAD_DIM] interleaved K then V, or separate buffers */
    /*
     * NOTE: pointers are float* for FP32 reference path.
     * Metal / optimized paths may use uint16_t* (FP16) or void* with dynamic type.
     */
    float *k_cache;     /* [max_seq_len][N_HEAD_KV][HEAD_DIM] */
    float *v_cache;     /* [max_seq_len][N_HEAD_KV][HEAD_DIM] */
    size_t capacity;    /* max_seq_len currently allocated */
    size_t seq_len;     /* current sequence length (number of valid tokens) */
} ds3_kv_cache_t;

/* ============================================================================
 * 6. Engine Handle (opaque — full definition in ds3.c)
 * ============================================================================ */

typedef struct ds3_engine ds3_engine_t;

/* ============================================================================
 * 7. Public API (declarations only — implementations in ds3_*.c)
 * ============================================================================ */

/* Engine lifecycle */
ds3_engine_t *ds3_engine_open(const char *gguf_path, int n_ctx);
void          ds3_engine_close(ds3_engine_t *engine);

/* Generation */
int ds3_engine_generate(ds3_engine_t *engine,
                        const int   *prompt_tokens,
                        int          n_prompt,
                        int          n_predict,
                        float        temperature,
                        int         *output_tokens,
                        int          output_capacity);

/* Optional callback invoked after each decode step with the full logits vector
 * and the token that was selected. `step` is 0 for the first generated token. */
typedef void (*ds3_engine_logit_cb_t)(int step,
                                      const float *logits,
                                      int n_vocab,
                                      int selected_token,
                                      void *user);

int ds3_engine_generate_ex(ds3_engine_t *engine,
                           const int   *prompt_tokens,
                           int          n_prompt,
                           int          n_predict,
                           float        temperature,
                           int         *output_tokens,
                           int          output_capacity,
                           ds3_engine_logit_cb_t cb,
                           void *cb_user);

/* Tokenizer — implemented in ds3_tokenizer.c, see ds3_tokenizer.h for full API */
struct ds3_vocab_t;
int  ds3_tokenize(const struct ds3_engine *engine, const char *text, int *tokens, int max_tokens);
void ds3_token_to_str(const struct ds3_engine *engine, int token_id, char *buf, size_t buf_size);
int  ds3_engine_chat_format(const struct ds3_engine *engine,
                            const char *system, const char *user,
                            int *tokens, int max_tokens);
int  ds3_engine_decode_sequence(const struct ds3_engine *engine,
                                const int *tokens, int n_tokens,
                                char *buf, size_t buf_size);

/* Diagnostics */
void ds3_print_tensor_info(const ds3_tensor_t *t);
void ds3_print_model_info(const ds3_engine_t *engine);

/* Run the engine through Layer 0 attention only and compare the resulting
 * hidden state against the FP32 reference implementation. Returns 0 if the
 * max difference is below the tolerance, 1 if it exceeds the tolerance, and
 * -1 on error. */
int ds3_engine_debug_layer0_compare(ds3_engine_t *engine, int token_id);

/* Run the engine through Layer 0 (attention + MoE FFN) and compare the
 * resulting hidden state against the FP32 reference implementation. */
int ds3_engine_debug_layer0_full_compare(ds3_engine_t *engine, int token_id);

/* Run the engine through the full model for a single token and compare the
 * final logits against the FP32 reference implementation. */
int ds3_engine_debug_full_compare(ds3_engine_t *engine, int token_id);

/* Run the engine layer-by-layer for a single token and compare the hidden
 * state after each full layer against the FP32 reference implementation. */
int ds3_engine_debug_per_layer_compare(ds3_engine_t *engine, int token_id);

/* Run the engine through a sequence of input tokens, maintaining the KV cache
 * between steps, and compare final logits at each step against the FP32
 * reference.  This is the diagnostic tool for multi-token decode drift. */
int ds3_engine_debug_decode_compare(ds3_engine_t *engine, const int *tokens,
                                    int n_steps);

/* ============================================================================
 * Weight Loading (implemented in ds3_weights.c)
 * ============================================================================ */

struct ds3_gguf;  /* defined in ds3_gguf.h */

ds3_weights_t *ds3_weights_load(struct ds3_gguf *gguf);
void           ds3_weights_free(ds3_weights_t *w);
void           ds3_weights_print_summary(const ds3_weights_t *w);

#ifdef __cplusplus
}
#endif

#endif /* DS3_H */
