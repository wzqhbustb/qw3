/*
 * test_layer0_e2e.c — End-to-end test for Layer 0 forward
 *
 * Build: cc -O2 -Wall -Wextra -I. -o test_layer0_e2e tests/test_layer0_e2e.c \
 *            src/ds3_reference.c src/ds3_gguf.c src/ds3_weights.c -lm
 * Run:   ./test_layer0_e2e <path/to/Q8_0.gguf>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "src/ds3_gguf.h"
#include "src/ds3.h"
#include "src/ds3_reference.h"

static void save_floats(const char *path, const float *data, int n)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror("fopen");
        return;
    }
    fwrite(data, sizeof(float), (size_t)n, fp);
    fclose(fp);
    printf("Saved %d floats to %s\n", n, path);
}

/*
 * Run attention branch with independent scratch buffers to extract
 * clean intermediate snapshots (state.xb/q/k/attn_out get reused
 * inside ds3_ref_layer_forward, so we can't rely on them).
 */
static void extract_attention_intermediates(
    const ds3_layer_weights_t *layer,
    const float *input,
    ds3_kv_cache_t *kv_cache,
    int seq_pos,
    float *snap_attn_norm,
    float *snap_q, float *snap_k, float *snap_v,
    float *snap_attn_out)
{
    const int n_embd = DS3_N_EMBD;
    const int q_dim  = DS3_Q_DIM;
    const int kv_dim = DS3_KV_DIM;

    /* Dequantize attention weights */
    const float *attn_norm   = ds3_ref_dequantize_tensor(layer->attn_norm);
    const float *attn_q      = ds3_ref_dequantize_tensor(layer->attn_q);
    const float *attn_q_norm = DS3_HAS_QK_NORM ? ds3_ref_dequantize_tensor(layer->attn_q_norm) : NULL;
    const float *attn_k      = ds3_ref_dequantize_tensor(layer->attn_k);
    const float *attn_k_norm = DS3_HAS_QK_NORM ? ds3_ref_dequantize_tensor(layer->attn_k_norm) : NULL;
    const float *attn_v      = ds3_ref_dequantize_tensor(layer->attn_v);
    const float *attn_o      = ds3_ref_dequantize_tensor(layer->attn_output);

    /* 1. RMSNorm */
    ds3_ref_rms_norm(input, attn_norm, DS3_NORM_EPS, n_embd, snap_attn_norm);

    /* 2. Q/K/V projections → saved directly to snap buffers */
    ds3_ref_vec_matmul(snap_attn_norm, attn_q, snap_q, n_embd, q_dim);
    ds3_ref_vec_matmul(snap_attn_norm, attn_k, snap_k, n_embd, kv_dim);
    ds3_ref_vec_matmul(snap_attn_norm, attn_v, snap_v, n_embd, kv_dim);

    /* 3. Call gqa_attention with independent scratch so snap_q/k/v stay intact */
    float *q_scratch = calloc(q_dim,  sizeof(float));
    float *k_scratch = calloc(kv_dim, sizeof(float));
    float *v_scratch = calloc(kv_dim, sizeof(float));
    float *attn_scores = calloc(DS3_N_HEAD * (seq_pos + 1), sizeof(float));

    ds3_ref_gqa_attention(
        snap_attn_norm,
        attn_q, attn_q_norm,
        attn_k, attn_k_norm,
        attn_v,
        attn_o,
        kv_cache, seq_pos, 1,
        snap_attn_out,
        q_scratch, k_scratch, v_scratch,
        attn_scores);

    free(q_scratch); free(k_scratch); free(v_scratch); free(attn_scores);

    /* Cleanup dequantized weights */
    free((void*)attn_norm);   free((void*)attn_q);      free((void*)attn_q_norm);
    free((void*)attn_k);      free((void*)attn_k_norm); free((void*)attn_v);
    free((void*)attn_o);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path/to/model.gguf>\n", argv[0]);
        return 1;
    }

    const char *gguf_path = argv[1];

    /* ── Open GGUF ── */
    ds3_gguf_t *gguf = ds3_gguf_open(gguf_path);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path);
        return 1;
    }
    printf("GGUF opened: %zu tensors\n", (size_t)gguf->n_tensors);

    /* ── Load weights ── */
    ds3_weights_t *weights = ds3_weights_load(gguf);
    if (!weights) {
        fprintf(stderr, "Failed to load weights\n");
        ds3_gguf_close(gguf);
        return 1;
    }
    printf("Weights loaded: Layer 0 has %d tensors\n",
           weights->layers[0].attn_q ? 1 : 0);

    const ds3_layer_weights_t *layer = &weights->layers[0];

    /* ── Allocate layer state ── */
    ds3_layer_state_t state = {0};
    state.xb             = calloc(DS3_N_EMBD,         sizeof(float));
    state.q              = calloc(DS3_Q_DIM,          sizeof(float));
    state.k              = calloc(DS3_KV_DIM,         sizeof(float));
    state.v              = calloc(DS3_KV_DIM,         sizeof(float));
    state.attn_out       = calloc(DS3_N_EMBD,         sizeof(float));
    state.ffn_out        = calloc(DS3_N_EMBD,         sizeof(float));
    state.gate_logits    = calloc(DS3_N_EXPERT,       sizeof(float));
    state.expert_weights = calloc(DS3_N_EXPERT_USED,  sizeof(float));
    state.expert_indices = calloc(DS3_N_EXPERT_USED,  sizeof(int));
    state.expert_tmp     = calloc(DS3_N_FF_EXP,       sizeof(float));
    state.shared_expert_tmp = calloc(DS3_N_FF_SHARED, sizeof(float));
    state.attn_score     = calloc(DS3_N_HEAD * 128,   sizeof(float)); /* max seq_len for testing */

    /* ── Allocate KV cache (seq_len=1 for first token) ── */
    ds3_kv_cache_t kv_cache = {0};
    kv_cache.k_cache = calloc(1 * DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
    kv_cache.v_cache = calloc(1 * DS3_N_HEAD_KV * DS3_HEAD_DIM, sizeof(float));
    kv_cache.capacity = 1;
    kv_cache.seq_len  = 0;

    /* ── Synthetic input: all ones ── */
    float *input = calloc(DS3_N_EMBD, sizeof(float));
    for (int i = 0; i < DS3_N_EMBD; i++) input[i] = 1.0f;

    /* ── Extract attention intermediates with independent buffers ── */
    printf("Extracting attention intermediates...\n");
    float *snap_attn_norm = calloc(DS3_N_EMBD,  sizeof(float));
    float *snap_q         = calloc(DS3_Q_DIM,   sizeof(float));
    float *snap_k         = calloc(DS3_KV_DIM,  sizeof(float));
    float *snap_v         = calloc(DS3_KV_DIM,  sizeof(float));
    float *snap_attn_out  = calloc(DS3_N_EMBD,  sizeof(float));

    extract_attention_intermediates(layer, input, &kv_cache, 0,
                                    snap_attn_norm, snap_q, snap_k, snap_v, snap_attn_out);

    save_floats("layer0_c_attn_norm.bin", snap_attn_norm, DS3_N_EMBD);
    save_floats("layer0_c_q_proj.bin",    snap_q,         DS3_Q_DIM);
    save_floats("layer0_c_k_proj.bin",    snap_k,         DS3_KV_DIM);
    save_floats("layer0_c_v_proj.bin",    snap_v,         DS3_KV_DIM);
    save_floats("layer0_c_attn_out.bin",  snap_attn_out,  DS3_N_EMBD);

    /* Reset KV cache for the full forward pass */
    kv_cache.seq_len = 0;

    /* ── Run full Layer 0 forward ── */
    printf("Running ds3_ref_layer_forward (seq_pos=0)...\n");
    float *output = calloc(DS3_N_EMBD, sizeof(float));
    ds3_ref_layer_forward(layer, input, &kv_cache, 0, 1, output, &state);

    /* ── Statistics ── */
    float mean = 0.0f, std = 0.0f;
    for (int i = 0; i < DS3_N_EMBD; i++) mean += output[i];
    mean /= DS3_N_EMBD;
    for (int i = 0; i < DS3_N_EMBD; i++) {
        float d = output[i] - mean;
        std += d * d;
    }
    std = sqrtf(std / DS3_N_EMBD);
    printf("C output: mean=%.6f std=%.6f\n", mean, std);

    /* ── Save FFN intermediates and final output ── */
    /* NOTE: state.xb was last written by ds3_ref_layer_forward's FFN RMSNorm.
     * This is an implicit coupling — if layer_forward reorders ops, this
     * will silently save the wrong tensor. For robustness we keep the
     * decomposed extract_attention_intermediates() path above. */
    save_floats("layer0_c_ffn_norm.bin", state.xb,      DS3_N_EMBD);
    save_floats("layer0_c_ffn_out.bin",  state.ffn_out, DS3_N_EMBD);
    save_floats("layer0_c_output.bin",   output,        DS3_N_EMBD);

    /* ── Cleanup ── */
    free(input); free(output);
    free(snap_attn_norm); free(snap_q); free(snap_k); free(snap_v); free(snap_attn_out);
    free(state.xb); free(state.q); free(state.k); free(state.v);
    free(state.attn_out); free(state.ffn_out);
    free(state.gate_logits); free(state.expert_weights); free(state.expert_indices);
    free(state.expert_tmp); free(state.shared_expert_tmp); free(state.attn_score);
    free(kv_cache.k_cache); free(kv_cache.v_cache);
    ds3_weights_free(weights);
    ds3_gguf_close(gguf);

    return 0;
}
