/*
 * ds3_weights.c — Bind GGUF tensors to ds3_weights_t by name
 *
 * Qwen3 GGUF tensor naming convention (from llama.cpp convert_hf_to_gguf.py):
 *
 *   Global:
 *     token_embd.weight          [N_VOCAB, N_EMBD]
 *     output_norm.weight         [N_EMBD]
 *     output.weight              [N_VOCAB, N_EMBD]  (optional, may be tied)
 *
 *   Per-layer (blk.N. where N = 0..N_LAYER-1):
 *     blk.N.attn_norm.weight             [N_EMBD]
 *     blk.N.attn_q.weight                [Q_DIM, N_EMBD]
 *     blk.N.attn_k.weight                [KV_DIM, N_EMBD]
 *     blk.N.attn_v.weight                [KV_DIM, N_EMBD]
 *     blk.N.attn_output.weight           [N_EMBD, Q_DIM]
 *     blk.N.ffn_norm.weight              [N_EMBD]
 *     blk.N.ffn_gate_inp.weight          [N_EXPERT, N_EMBD]   (router)
 *     blk.N.ffn_gate_exps.weight         [N_EXPERT, N_FF_EXP, N_EMBD] or [N_EXPERT×N_FF_EXP, N_EMBD]
 *     blk.N.ffn_up_exps.weight           [N_EXPERT, N_FF_EXP, N_EMBD] or [N_EXPERT×N_FF_EXP, N_EMBD]
 *     blk.N.ffn_down_exps.weight         [N_EXPERT, N_EMBD, N_FF_EXP] or [N_EXPERT×N_EMBD, N_FF_EXP]
 *     blk.N.shared_expert_gate.weight    [N_FF_SHARED, N_EMBD]
 *     blk.N.shared_expert_up.weight      [N_FF_SHARED, N_EMBD]
 *     blk.N.shared_expert_down.weight    [N_EMBD, N_FF_SHARED]
 *
 *   Optional per-layer:
 *     blk.N.ffn_gate_inp.bias            [N_EXPERT]  (if config has router bias)
 */

#include "ds3.h"
#include "ds3_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

/* Ensure ds3_type_t (ds3.h) and ds3_gguf_type_t (ds3_gguf.h) stay in sync */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert((int)DS3_TYPE_Q2_K == (int)DS3_GGUF_TYPE_Q2_K, "type enum mismatch");
_Static_assert((int)DS3_TYPE_Q4_K == (int)DS3_GGUF_TYPE_Q4_K, "type enum mismatch");
_Static_assert((int)DS3_TYPE_Q8_0 == (int)DS3_GGUF_TYPE_Q8_0, "type enum mismatch");
_Static_assert((int)DS3_TYPE_IQ2_XXS == (int)DS3_GGUF_TYPE_IQ2_XXS, "type enum mismatch");
#endif

/* ============================================================================
 * Helpers
 * ============================================================================ */

static ds3_tensor_t *make_tensor(ds3_gguf_t *gguf, const char *name) {
    ds3_gguf_tensor_info_t *gti = ds3_gguf_find_tensor(gguf, name);
    if (!gti) return NULL;

    ds3_tensor_t *t = calloc(1, sizeof(ds3_tensor_t));
    if (!t) return NULL;

    /* name: strdup since GGUF string is not null-terminated */
    t->name = strndup(gti->name.data, gti->name.len);
    t->type = (ds3_type_t)gti->type;
    t->n_dims = gti->n_dims;
    for (uint32_t d = 0; d < 4; d++) {
        t->ne[d] = (d < gti->n_dims) ? gti->ne[d] : 1;
    }
    /* Compute byte strides nb[4] matching llama.cpp ggml layout.
     * nb[0] = type_size (bytes per block, or per-element for F32/F16)
     * nb[1] = type_size * (ne[0] / block_size)  (bytes per row)
     * nb[d] = nb[d-1] * ne[d-1] for d >= 2
     *
     * For F32: type_size=4, block_size=1 → nb[0]=4, nb[1]=4*ne[0] (standard C-order)
     * For Q4_K: type_size=144, block_size=256 → nb[0]=144, nb[1]=144*(ne[0]/256)
     */
    if (gti->n_dims > 0) {
        uint64_t type_size = 1, block_size = 1;
        switch (gti->type) {
            case DS3_GGUF_TYPE_F32:  type_size = 4; break;
            case DS3_GGUF_TYPE_F16:  type_size = 2; break;
            case DS3_GGUF_TYPE_Q4_0: type_size = 18; block_size = 32; break;
            case DS3_GGUF_TYPE_Q5_0: type_size = 22; block_size = 32; break;
            case DS3_GGUF_TYPE_Q8_0: type_size = 34; block_size = 32; break;
            case DS3_GGUF_TYPE_Q2_K: type_size = 84; block_size = 256; break;
            case DS3_GGUF_TYPE_Q3_K: type_size = 110; block_size = 256; break;
            case DS3_GGUF_TYPE_Q4_K: type_size = 144; block_size = 256; break;
            case DS3_GGUF_TYPE_Q5_K: type_size = 176; block_size = 256; break;
            case DS3_GGUF_TYPE_Q6_K: type_size = 210; block_size = 256; break;
            case DS3_GGUF_TYPE_Q8_K: type_size = 290; block_size = 256; break;
            case DS3_GGUF_TYPE_IQ2_XXS: type_size = 66; block_size = 256; break;
            case DS3_GGUF_TYPE_IQ2_XS:  type_size = 74; block_size = 256; break;
            case DS3_GGUF_TYPE_IQ3_XXS: type_size = 98; block_size = 256; break;
            default: break;
        }
        t->nb[0] = type_size;
        if (gti->n_dims >= 2) {
            t->nb[1] = type_size * (t->ne[0] / block_size);
        }
        for (uint32_t d = 2; d < 4; d++) {
            t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
        }
    }
    t->data = (void *)gti->data;  /* GGUF mmap is read-only; cast for API uniformity */
    t->size = gti->size;
    return t;
}

/* ============================================================================
 * Layer tensor binding
 * ============================================================================ */

static bool bind_layer(ds3_gguf_t *gguf, ds3_layer_weights_t *layer, int layer_idx) {
    char name_buf[128];
    bool ok = true;

#define BIND(field, fmt, ...) do {                                      \
    snprintf(name_buf, sizeof(name_buf), fmt, __VA_ARGS__);             \
    layer->field = make_tensor(gguf, name_buf);                         \
    if (!layer->field) {                                                \
        fprintf(stderr, "[weights] Missing: %s\n", name_buf);           \
        ok = false;                                                     \
    }                                                                   \
} while (0)

    /* Attention */
    BIND(attn_norm,    "blk.%d.attn_norm.weight", layer_idx);
    BIND(attn_q,       "blk.%d.attn_q.weight", layer_idx);
#if DS3_HAS_QK_NORM
    BIND(attn_q_norm,  "blk.%d.attn_q_norm.weight", layer_idx);
#endif
    BIND(attn_k,       "blk.%d.attn_k.weight", layer_idx);
#if DS3_HAS_QK_NORM
    BIND(attn_k_norm,  "blk.%d.attn_k_norm.weight", layer_idx);
#endif
    BIND(attn_v,       "blk.%d.attn_v.weight", layer_idx);
    BIND(attn_output,  "blk.%d.attn_output.weight", layer_idx);

    /* FFN Norm */
    BIND(ffn_norm,     "blk.%d.ffn_norm.weight", layer_idx);

    /* Router */
    BIND(ffn_gate_inp, "blk.%d.ffn_gate_inp.weight", layer_idx);

    /* Routed experts */
    BIND(ffn_gate_exps, "blk.%d.ffn_gate_exps.weight", layer_idx);
    BIND(ffn_up_exps,   "blk.%d.ffn_up_exps.weight", layer_idx);
    BIND(ffn_down_exps, "blk.%d.ffn_down_exps.weight", layer_idx);

    /* Shared expert — Qwen3-MoE uses llama.cpp Qwen2MoE naming:
     *   blk.N.ffn_gate_shexp.weight
     *   blk.N.ffn_up_shexp.weight
     *   blk.N.ffn_down_shexp.weight
     * Keep the struct field names for backward compatibility. */
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_shexp.weight", layer_idx);
    layer->shared_expert_gate = make_tensor(gguf, name_buf);
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up_shexp.weight", layer_idx);
    layer->shared_expert_up = make_tensor(gguf, name_buf);
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down_shexp.weight", layer_idx);
    layer->shared_expert_down = make_tensor(gguf, name_buf);

    /* Optional: router bias */
    snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_inp.bias", layer_idx);
    layer->ffn_gate_inp_bias = make_tensor(gguf, name_buf);

#undef BIND
    return ok;
}

/* ============================================================================
 * Global binding
 * ============================================================================ */

ds3_weights_t *ds3_weights_load(ds3_gguf_t *gguf) {
    if (!gguf) return NULL;

    ds3_weights_t *w = calloc(1, sizeof(ds3_weights_t));
    if (!w) return NULL;

    ds3_print_info("[weights] Binding %s (%d layers)...\n", DS3_MODEL_NAME, DS3_N_LAYER);

    bool all_ok = true;

    /* Global tensors */
    w->token_embd  = make_tensor(gguf, "token_embd.weight");
    w->output_norm = make_tensor(gguf, "output_norm.weight");
    w->output      = make_tensor(gguf, "output.weight");

    if (!w->token_embd)  {
        fprintf(stderr, "[weights] Missing: token_embd.weight\n");
        all_ok = false;
    }
    if (!w->output_norm) {
        fprintf(stderr, "[weights] Missing: output_norm.weight\n");
        all_ok = false;
    }
    if (!w->output) {
        /* Output may be tied to embedding — not an error */
        ds3_print_info("[weights] output.weight not found (tied to token_embd?)\n");
    }

    /* Per-layer tensors */
    for (int i = 0; i < DS3_N_LAYER; i++) {
        bool layer_ok = bind_layer(gguf, &w->layers[i], i);
        if (!layer_ok) {
            fprintf(stderr, "[weights] Layer %d binding incomplete\n", i);
            all_ok = false;
        }
    }

    if (!all_ok) {
        fprintf(stderr, "[weights] ERROR: Some required tensors missing. Freeing partial weights.\n");
        ds3_weights_free(w);
        return NULL;
    }

    return w;
}

void ds3_weights_free(ds3_weights_t *w) {
    if (!w) return;
    for (int i = 0; i < DS3_N_LAYER; i++) {
        ds3_layer_weights_t *l = &w->layers[i];
#define FREE_FIELD(f) do { if (l->f) { free((void*)l->f->name); free(l->f); } } while (0)
        FREE_FIELD(attn_norm);
        FREE_FIELD(attn_q);
        FREE_FIELD(attn_q_norm);
        FREE_FIELD(attn_k);
        FREE_FIELD(attn_k_norm);
        FREE_FIELD(attn_v);
        FREE_FIELD(attn_output);
        FREE_FIELD(ffn_norm);
        FREE_FIELD(ffn_gate_inp);
        FREE_FIELD(ffn_gate_exps);
        FREE_FIELD(ffn_up_exps);
        FREE_FIELD(ffn_down_exps);
        FREE_FIELD(shared_expert_gate);
        FREE_FIELD(shared_expert_up);
        FREE_FIELD(shared_expert_down);
        FREE_FIELD(ffn_gate_inp_bias);
#undef FREE_FIELD
    }
    if (w->token_embd)  { free((void*)w->token_embd->name); free(w->token_embd); }
    if (w->output_norm) { free((void*)w->output_norm->name); free(w->output_norm); }
    if (w->output)      { free((void*)w->output->name); free(w->output); }
    free(w);
}

/* ============================================================================
 * Validation / inspection
 * ============================================================================ */

void ds3_weights_print_summary(const ds3_weights_t *w) {
    if (!w) return;
    ds3_print_info("\n═══ Weights Summary ═══\n");
    ds3_print_info("Model: %s\n", DS3_MODEL_NAME);

    if (w->token_embd)  ds3_print_tensor_info(w->token_embd);
    if (w->output_norm) ds3_print_tensor_info(w->output_norm);
    if (w->output)      ds3_print_tensor_info(w->output);

    /* Sample layer 0 */
    ds3_print_info("\n--- Layer 0 sample ---\n");
    const ds3_layer_weights_t *l = &w->layers[0];
    if (l->attn_q)       ds3_print_tensor_info(l->attn_q);
    if (l->attn_k)       ds3_print_tensor_info(l->attn_k);
    if (l->ffn_gate_inp) ds3_print_tensor_info(l->ffn_gate_inp);
    if (l->ffn_gate_exps) ds3_print_tensor_info(l->ffn_gate_exps);
    if (l->shared_expert_gate) ds3_print_tensor_info(l->shared_expert_gate);

    /* Count total tensors found */
    int found = 0, expected = 0;
    if (w->token_embd) found++; expected++;
    if (w->output_norm) found++; expected++;
    if (w->output) found++; expected++;
    for (int i = 0; i < DS3_N_LAYER; i++) {
        const ds3_layer_weights_t *l = &w->layers[i];
#define COUNT(f) do { expected++; if (l->f) found++; } while (0)
        COUNT(attn_norm); COUNT(attn_q); COUNT(attn_q_norm); COUNT(attn_k); COUNT(attn_k_norm);
        COUNT(attn_v); COUNT(attn_output);
        COUNT(ffn_norm); COUNT(ffn_gate_inp);
        COUNT(ffn_gate_exps); COUNT(ffn_up_exps); COUNT(ffn_down_exps);
#undef COUNT
    }
    ds3_print_info("\nTensors bound: %d / %d\n", found, expected);
}

/* Print single tensor info (declared in ds3.h) */
void ds3_print_tensor_info(const ds3_tensor_t *t) {
    if (!t) { ds3_print_info("(null)\n"); return; }
    ds3_print_info("  %-45s  type=%2d  shape=[", t->name, t->type);
    for (uint32_t d = 0; d < t->n_dims; d++) {
        if (d > 0) ds3_print_info(", ");
        ds3_print_info("%llu", (unsigned long long)t->ne[d]);
    }
    ds3_print_info("]  size=%llu\n", (unsigned long long)t->size);
}
