/*
 * test_moe_metal.c — Validate Metal MoE FFN kernel.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "src/ds3_metal.h"
#include "src/ds3_reference.h"
#include "tests/test_quant_utils.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL: %s at %s:%d\n", #x, __FILE__, __LINE__); return 1; } } while(0)

static float max_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static int test_moe_ffn_small(void) {
    fprintf(stderr, "test_moe_ffn_small ... ");

    const uint32_t n_embd    = 256;
    const uint32_t n_expert  = 16;
    const uint32_t n_used    = 4;
    const uint32_t n_ff_exp  = 128;

    const size_t input_bytes     = n_embd * sizeof(float);
    const size_t logits_bytes    = n_expert * sizeof(float);
    const size_t hidden_bytes    = n_ff_exp * sizeof(float);
    const size_t output_bytes    = n_embd * sizeof(float);
    const size_t gate_inp_bytes  = n_expert * n_embd * sizeof(float);
    const size_t gate_exps_bytes = n_expert * n_ff_exp * n_embd * sizeof(float);
    const size_t down_exps_bytes = n_expert * n_embd * n_ff_exp * sizeof(float);

    float *input     = calloc(n_embd, sizeof(float));
    float *w_gate_inp = calloc(n_expert * n_embd, sizeof(float));
    float *w_gate_exps = calloc(n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_up_exps   = calloc(n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_down_exps = calloc(n_expert * n_embd * n_ff_exp, sizeof(float));
    float *output_ref  = calloc(n_embd, sizeof(float));
    float *gate_logits = calloc(n_expert, sizeof(float));
    int    expert_indices[4];
    float  expert_scores[4];
    CHECK(input && w_gate_inp && w_gate_exps && w_up_exps && w_down_exps && output_ref && gate_logits);

    srand(70);
    for (uint32_t i = 0; i < n_embd; i++) input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_embd; i++) w_gate_inp[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_ff_exp * n_embd; i++) {
        w_gate_exps[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        w_up_exps[i]   = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < n_expert * n_embd * n_ff_exp; i++) {
        w_down_exps[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    ds3_ref_moe_ffn(input, w_gate_inp, w_gate_exps, w_up_exps, w_down_exps,
                    NULL, NULL, NULL,
                    output_ref, gate_logits, expert_scores, expert_indices,
                    (int)n_embd, (int)n_expert, (int)n_used, (int)n_ff_exp, 0, true);

    ds3_metal_buffer_t *in_buf       = ds3_metal_buffer_alloc(input_bytes);
    ds3_metal_buffer_t *gate_inp_buf = ds3_metal_buffer_alloc(gate_inp_bytes);
    ds3_metal_buffer_t *gate_exps_buf= ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *up_exps_buf  = ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *down_exps_buf= ds3_metal_buffer_alloc(down_exps_bytes);
    ds3_metal_buffer_t *out_buf      = ds3_metal_buffer_alloc(output_bytes);
    ds3_metal_buffer_t *logits_buf   = ds3_metal_buffer_alloc(logits_bytes);
    ds3_metal_buffer_t *hidden_buf   = ds3_metal_buffer_alloc(hidden_bytes);
    CHECK(in_buf && gate_inp_buf && gate_exps_buf && up_exps_buf && down_exps_buf &&
          out_buf && logits_buf && hidden_buf);

    CHECK(ds3_metal_buffer_write(in_buf,        0, input,      input_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_inp_buf,  0, w_gate_inp, gate_inp_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_exps_buf, 0, w_gate_exps,gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(up_exps_buf,   0, w_up_exps,  gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(down_exps_buf, 0, w_down_exps,down_exps_bytes) == 0);

    float *output_mtl = calloc(n_embd, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_write(out_buf, 0, output_mtl, output_bytes) == 0);

    CHECK(ds3_metal_moe_ffn(in_buf, gate_inp_buf, gate_exps_buf, up_exps_buf, down_exps_buf,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            n_embd * sizeof(float), n_embd * sizeof(float), n_ff_exp * sizeof(float),
                            NULL, NULL, NULL,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            0, 0, 0,
                            out_buf, logits_buf, NULL, hidden_buf, NULL, NULL,
                            n_embd, n_expert, n_used, n_ff_exp, 0,
                            n_embd * sizeof(float), DS3_TYPE_F32, true) == 0);

    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, output_bytes) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)n_embd);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(in_buf);
    ds3_metal_buffer_free(gate_inp_buf);
    ds3_metal_buffer_free(gate_exps_buf);
    ds3_metal_buffer_free(up_exps_buf);
    ds3_metal_buffer_free(down_exps_buf);
    ds3_metal_buffer_free(out_buf);
    ds3_metal_buffer_free(logits_buf);
    ds3_metal_buffer_free(hidden_buf);
    free(input); free(w_gate_inp); free(w_gate_exps); free(w_up_exps); free(w_down_exps);
    free(output_ref); free(gate_logits); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_moe_ffn_qwen3_dims(void) {
    fprintf(stderr, "test_moe_ffn_qwen3_dims ... ");

    const uint32_t n_embd    = 2048;
    const uint32_t n_expert  = 128;
    const uint32_t n_used    = 8;
    const uint32_t n_ff_exp  = 768;

    const size_t input_bytes     = n_embd * sizeof(float);
    const size_t logits_bytes    = n_expert * sizeof(float);
    const size_t hidden_bytes    = n_ff_exp * sizeof(float);
    const size_t output_bytes    = n_embd * sizeof(float);
    const size_t gate_inp_bytes  = n_expert * n_embd * sizeof(float);
    const size_t gate_exps_bytes = n_expert * n_ff_exp * n_embd * sizeof(float);
    const size_t down_exps_bytes = n_expert * n_embd * n_ff_exp * sizeof(float);

    float *input     = calloc(n_embd, sizeof(float));
    float *w_gate_inp = calloc(n_expert * n_embd, sizeof(float));
    float *w_gate_exps = calloc(n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_up_exps   = calloc(n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_down_exps = calloc(n_expert * n_embd * n_ff_exp, sizeof(float));
    float *output_ref  = calloc(n_embd, sizeof(float));
    float *gate_logits = calloc(n_expert, sizeof(float));
    int    expert_indices[8];
    float  expert_scores[8];
    CHECK(input && w_gate_inp && w_gate_exps && w_up_exps && w_down_exps && output_ref && gate_logits);

    srand(71);
    for (uint32_t i = 0; i < n_embd; i++) input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_embd; i++) w_gate_inp[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_ff_exp * n_embd; i++) {
        w_gate_exps[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        w_up_exps[i]   = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < n_expert * n_embd * n_ff_exp; i++) {
        w_down_exps[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    ds3_ref_moe_ffn(input, w_gate_inp, w_gate_exps, w_up_exps, w_down_exps,
                    NULL, NULL, NULL,
                    output_ref, gate_logits, expert_scores, expert_indices,
                    (int)n_embd, (int)n_expert, (int)n_used, (int)n_ff_exp, 0, true);

    ds3_metal_buffer_t *in_buf       = ds3_metal_buffer_alloc(input_bytes);
    ds3_metal_buffer_t *gate_inp_buf = ds3_metal_buffer_alloc(gate_inp_bytes);
    ds3_metal_buffer_t *gate_exps_buf= ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *up_exps_buf  = ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *down_exps_buf= ds3_metal_buffer_alloc(down_exps_bytes);
    ds3_metal_buffer_t *out_buf      = ds3_metal_buffer_alloc(output_bytes);
    ds3_metal_buffer_t *logits_buf   = ds3_metal_buffer_alloc(logits_bytes);
    ds3_metal_buffer_t *hidden_buf   = ds3_metal_buffer_alloc(hidden_bytes);
    CHECK(in_buf && gate_inp_buf && gate_exps_buf && up_exps_buf && down_exps_buf &&
          out_buf && logits_buf && hidden_buf);

    CHECK(ds3_metal_buffer_write(in_buf,        0, input,      input_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_inp_buf,  0, w_gate_inp, gate_inp_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_exps_buf, 0, w_gate_exps,gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(up_exps_buf,   0, w_up_exps,  gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(down_exps_buf, 0, w_down_exps,down_exps_bytes) == 0);

    float *output_mtl = calloc(n_embd, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_write(out_buf, 0, output_mtl, output_bytes) == 0);

    CHECK(ds3_metal_moe_ffn(in_buf, gate_inp_buf, gate_exps_buf, up_exps_buf, down_exps_buf,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            n_embd * sizeof(float), n_embd * sizeof(float), n_ff_exp * sizeof(float),
                            NULL, NULL, NULL,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            0, 0, 0,
                            out_buf, logits_buf, NULL, hidden_buf, NULL, NULL,
                            n_embd, n_expert, n_used, n_ff_exp, 0,
                            n_embd * sizeof(float), DS3_TYPE_F32, true) == 0);

    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, output_bytes) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)n_embd);
    CHECK(diff < 1e-3f);

    ds3_metal_buffer_free(in_buf);
    ds3_metal_buffer_free(gate_inp_buf);
    ds3_metal_buffer_free(gate_exps_buf);
    ds3_metal_buffer_free(up_exps_buf);
    ds3_metal_buffer_free(down_exps_buf);
    ds3_metal_buffer_free(out_buf);
    ds3_metal_buffer_free(logits_buf);
    ds3_metal_buffer_free(hidden_buf);
    free(input); free(w_gate_inp); free(w_gate_exps); free(w_up_exps); free(w_down_exps);
    free(output_ref); free(gate_logits); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

static int test_moe_ffn_qwen3_dims_q4k(void) {
    fprintf(stderr, "test_moe_ffn_qwen3_dims_q4k ... ");

    const uint32_t n_embd    = 2048;
    const uint32_t n_expert  = 128;
    const uint32_t n_used    = 8;
    const uint32_t n_ff_exp  = 768;

    const int nb_gate_in = n_embd / Q4K_BLOCK_SIZE;
    const int nb_up_in   = nb_gate_in;
    const int nb_down_in = n_ff_exp / Q4K_BLOCK_SIZE;

    const size_t gate_inp_bytes  = n_expert * n_embd * sizeof(float);
    const size_t gate_exps_bytes = (size_t)n_expert * n_ff_exp * nb_gate_in * sizeof(ds3_test_block_q4_K);
    const size_t up_exps_bytes   = gate_exps_bytes;
    const size_t down_exps_bytes = (size_t)n_expert * n_embd * nb_down_in * sizeof(ds3_test_block_q4_K);

    const size_t input_bytes  = n_embd * sizeof(float);
    const size_t logits_bytes = n_expert * sizeof(float);
    const size_t hidden_bytes = n_ff_exp * sizeof(float);
    const size_t output_bytes = n_embd * sizeof(float);

    float *input     = calloc(n_embd, sizeof(float));
    float *w_gate_inp = calloc(n_expert * n_embd, sizeof(float));
    float *w_gate_exps_f = calloc(n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_up_exps_f   = calloc(n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_down_exps_f = calloc(n_expert * n_embd * n_ff_exp, sizeof(float));
    ds3_test_block_q4_K *w_gate_exps = calloc(1, gate_exps_bytes);
    ds3_test_block_q4_K *w_up_exps   = calloc(1, up_exps_bytes);
    ds3_test_block_q4_K *w_down_exps = calloc(1, down_exps_bytes);
    float *output_ref  = calloc(n_embd, sizeof(float));
    float *gate_logits = calloc(n_expert, sizeof(float));
    int    expert_indices[8];
    float  expert_scores[8];
    CHECK(input && w_gate_inp && w_gate_exps_f && w_up_exps_f && w_down_exps_f &&
          w_gate_exps && w_up_exps && w_down_exps && output_ref && gate_logits);

    srand(72);
    for (uint32_t i = 0; i < n_embd; i++) input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_embd; i++) w_gate_inp[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_ff_exp * n_embd; i++) {
        w_gate_exps_f[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        w_up_exps_f[i]   = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < n_expert * n_embd * n_ff_exp; i++) {
        w_down_exps_f[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    /* Quantize expert weights. */
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < n_ff_exp; r++) {
            quantize_row_q4_K(w_gate_exps_f + (e * n_ff_exp + r) * n_embd,
                              w_gate_exps + (e * n_ff_exp + r) * nb_gate_in,
                              (int)n_embd);
            quantize_row_q4_K(w_up_exps_f + (e * n_ff_exp + r) * n_embd,
                              w_up_exps + (e * n_ff_exp + r) * nb_up_in,
                              (int)n_embd);
        }
        for (uint32_t r = 0; r < n_embd; r++) {
            quantize_row_q4_K(w_down_exps_f + (e * n_embd + r) * n_ff_exp,
                              w_down_exps + (e * n_embd + r) * nb_down_in,
                              (int)n_ff_exp);
        }
    }

    /* CPU reference uses dequantized weights. */
    {
        ds3_tensor_t t_gate = {0}; t_gate.type = DS3_TYPE_Q4_K; t_gate.n_dims = 2;
        t_gate.ne[0] = n_embd; t_gate.ne[1] = n_expert * n_ff_exp;
        t_gate.data = w_gate_exps; t_gate.size = gate_exps_bytes;
        float *d_gate = ds3_ref_dequantize_tensor(&t_gate);
        CHECK(d_gate);

        ds3_tensor_t t_up = {0}; t_up.type = DS3_TYPE_Q4_K; t_up.n_dims = 2;
        t_up.ne[0] = n_embd; t_up.ne[1] = n_expert * n_ff_exp;
        t_up.data = w_up_exps; t_up.size = up_exps_bytes;
        float *d_up = ds3_ref_dequantize_tensor(&t_up);
        CHECK(d_up);

        ds3_tensor_t t_down = {0}; t_down.type = DS3_TYPE_Q4_K; t_down.n_dims = 2;
        t_down.ne[0] = n_ff_exp; t_down.ne[1] = n_expert * n_embd;
        t_down.data = w_down_exps; t_down.size = down_exps_bytes;
        float *d_down = ds3_ref_dequantize_tensor(&t_down);
        CHECK(d_down);

        ds3_ref_moe_ffn(input, w_gate_inp, d_gate, d_up, d_down,
                        NULL, NULL, NULL,
                        output_ref, gate_logits, expert_scores, expert_indices,
                        (int)n_embd, (int)n_expert, (int)n_used, (int)n_ff_exp, 0, true);

        free(d_gate); free(d_up); free(d_down);
    }

    ds3_metal_buffer_t *in_buf       = ds3_metal_buffer_alloc(input_bytes);
    ds3_metal_buffer_t *gate_inp_buf = ds3_metal_buffer_alloc(gate_inp_bytes);
    ds3_metal_buffer_t *gate_exps_buf= ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *up_exps_buf  = ds3_metal_buffer_alloc(up_exps_bytes);
    ds3_metal_buffer_t *down_exps_buf= ds3_metal_buffer_alloc(down_exps_bytes);
    ds3_metal_buffer_t *out_buf      = ds3_metal_buffer_alloc(output_bytes);
    ds3_metal_buffer_t *logits_buf   = ds3_metal_buffer_alloc(logits_bytes);
    ds3_metal_buffer_t *hidden_buf   = ds3_metal_buffer_alloc(hidden_bytes);
    ds3_metal_buffer_t *up_buf       = ds3_metal_buffer_alloc(hidden_bytes);
    ds3_metal_buffer_t *down_buf     = ds3_metal_buffer_alloc(output_bytes);
    CHECK(in_buf && gate_inp_buf && gate_exps_buf && up_exps_buf && down_exps_buf &&
          out_buf && logits_buf && hidden_buf && up_buf && down_buf);

    CHECK(ds3_metal_buffer_write(in_buf,        0, input,      input_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_inp_buf,  0, w_gate_inp, gate_inp_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_exps_buf, 0, w_gate_exps, gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(up_exps_buf,   0, w_up_exps,   up_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(down_exps_buf, 0, w_down_exps, down_exps_bytes) == 0);

    float *output_mtl = calloc(n_embd, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_write(out_buf, 0, output_mtl, output_bytes) == 0);

    const uint64_t q4k_gate_stride = (uint64_t)(n_embd / 256) * 144;
    const uint64_t q4k_up_stride   = q4k_gate_stride;
    const uint64_t q4k_down_stride = (uint64_t)(n_ff_exp / 256) * 144;
    CHECK(ds3_metal_moe_ffn(in_buf, gate_inp_buf, gate_exps_buf, up_exps_buf, down_exps_buf,
                            DS3_TYPE_Q4_K, DS3_TYPE_Q4_K, DS3_TYPE_Q4_K,
                            q4k_gate_stride, q4k_up_stride, q4k_down_stride,
                            NULL, NULL, NULL,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            0, 0, 0,
                            out_buf, logits_buf, NULL, hidden_buf, up_buf, down_buf,
                            n_embd, n_expert, n_used, n_ff_exp, 0,
                            n_embd * sizeof(float), DS3_TYPE_F32, true) == 0);

    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, output_bytes) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)n_embd);
    CHECK(diff < 2e-3f);

    ds3_metal_buffer_free(in_buf);
    ds3_metal_buffer_free(gate_inp_buf);
    ds3_metal_buffer_free(gate_exps_buf);
    ds3_metal_buffer_free(up_exps_buf);
    ds3_metal_buffer_free(down_exps_buf);
    ds3_metal_buffer_free(out_buf);
    ds3_metal_buffer_free(logits_buf);
    ds3_metal_buffer_free(hidden_buf);
    ds3_metal_buffer_free(up_buf);
    ds3_metal_buffer_free(down_buf);
    free(input); free(w_gate_inp);
    free(w_gate_exps_f); free(w_up_exps_f); free(w_down_exps_f);
    free(w_gate_exps); free(w_up_exps); free(w_down_exps);
    free(output_ref); free(gate_logits); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Mixed-type MoE: gate/up are Q4_K, down is kept FP32.
 * Exercises dispatch_moe_expert_quant() with per-weight type selection. */
static int test_moe_ffn_qwen3_dims_mixed(void) {
    fprintf(stderr, "test_moe_ffn_qwen3_dims_mixed ... ");

    const uint32_t n_embd    = 2048;
    const uint32_t n_expert  = 128;
    const uint32_t n_used    = 8;
    const uint32_t n_ff_exp  = 768;

    const size_t input_bytes     = n_embd * sizeof(float);
    const size_t logits_bytes    = n_expert * sizeof(float);
    const size_t hidden_bytes    = n_ff_exp * sizeof(float);
    const size_t output_bytes    = n_embd * sizeof(float);
    const size_t gate_inp_bytes  = n_expert * n_embd * sizeof(float);

    const int nb_gate_in = (int)(n_embd / 256);
    const int nb_up_in   = nb_gate_in;
    const size_t gate_exps_bytes = (size_t)n_expert * n_ff_exp * nb_gate_in * 144;
    const size_t up_exps_bytes   = gate_exps_bytes;
    const size_t down_exps_bytes = (size_t)n_expert * n_embd * n_ff_exp * sizeof(float);

    float *input        = calloc(n_embd, sizeof(float));
    float *w_gate_inp   = calloc(n_expert * n_embd, sizeof(float));
    float *w_gate_exps_f= calloc((size_t)n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_up_exps_f  = calloc((size_t)n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_down_exps_f= calloc((size_t)n_expert * n_embd * n_ff_exp, sizeof(float));
    ds3_test_block_q4_K *w_gate_exps = calloc(1, gate_exps_bytes);
    ds3_test_block_q4_K *w_up_exps   = calloc(1, up_exps_bytes);
    float *output_ref  = calloc(n_embd, sizeof(float));
    float *gate_logits = calloc(n_expert, sizeof(float));
    int    expert_indices[8];
    float  expert_scores[8];
    CHECK(input && w_gate_inp && w_gate_exps_f && w_up_exps_f && w_down_exps_f &&
          w_gate_exps && w_up_exps && output_ref && gate_logits);

    srand(73);
    for (uint32_t i = 0; i < n_embd; i++) input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_embd; i++) w_gate_inp[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_ff_exp * n_embd; i++) {
        w_gate_exps_f[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        w_up_exps_f[i]   = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < n_expert * n_embd * n_ff_exp; i++) {
        w_down_exps_f[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    /* Quantize gate/up only. */
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < n_ff_exp; r++) {
            quantize_row_q4_K(w_gate_exps_f + (e * n_ff_exp + r) * n_embd,
                              w_gate_exps + (e * n_ff_exp + r) * nb_gate_in,
                              (int)n_embd);
            quantize_row_q4_K(w_up_exps_f + (e * n_ff_exp + r) * n_embd,
                              w_up_exps + (e * n_ff_exp + r) * nb_up_in,
                              (int)n_embd);
        }
    }

    /* CPU reference uses dequantized gate/up and original FP32 down. */
    {
        ds3_tensor_t t_gate = {0}; t_gate.type = DS3_TYPE_Q4_K; t_gate.n_dims = 2;
        t_gate.ne[0] = n_embd; t_gate.ne[1] = n_expert * n_ff_exp;
        t_gate.data = w_gate_exps; t_gate.size = gate_exps_bytes;
        float *d_gate = ds3_ref_dequantize_tensor(&t_gate);
        CHECK(d_gate);

        ds3_tensor_t t_up = {0}; t_up.type = DS3_TYPE_Q4_K; t_up.n_dims = 2;
        t_up.ne[0] = n_embd; t_up.ne[1] = n_expert * n_ff_exp;
        t_up.data = w_up_exps; t_up.size = up_exps_bytes;
        float *d_up = ds3_ref_dequantize_tensor(&t_up);
        CHECK(d_up);

        ds3_ref_moe_ffn(input, w_gate_inp, d_gate, d_up, w_down_exps_f,
                        NULL, NULL, NULL,
                        output_ref, gate_logits, expert_scores, expert_indices,
                        (int)n_embd, (int)n_expert, (int)n_used, (int)n_ff_exp, 0, true);

        free(d_gate); free(d_up);
    }

    ds3_metal_buffer_t *in_buf       = ds3_metal_buffer_alloc(input_bytes);
    ds3_metal_buffer_t *gate_inp_buf = ds3_metal_buffer_alloc(gate_inp_bytes);
    ds3_metal_buffer_t *gate_exps_buf= ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *up_exps_buf  = ds3_metal_buffer_alloc(up_exps_bytes);
    ds3_metal_buffer_t *down_exps_buf= ds3_metal_buffer_alloc(down_exps_bytes);
    ds3_metal_buffer_t *out_buf      = ds3_metal_buffer_alloc(output_bytes);
    ds3_metal_buffer_t *logits_buf   = ds3_metal_buffer_alloc(logits_bytes);
    ds3_metal_buffer_t *hidden_buf   = ds3_metal_buffer_alloc(hidden_bytes);
    ds3_metal_buffer_t *up_buf       = ds3_metal_buffer_alloc(hidden_bytes);
    ds3_metal_buffer_t *down_buf     = ds3_metal_buffer_alloc(output_bytes);
    CHECK(in_buf && gate_inp_buf && gate_exps_buf && up_exps_buf && down_exps_buf &&
          out_buf && logits_buf && hidden_buf && up_buf && down_buf);

    CHECK(ds3_metal_buffer_write(in_buf,        0, input,        input_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_inp_buf,  0, w_gate_inp,   gate_inp_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_exps_buf, 0, w_gate_exps,  gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(up_exps_buf,   0, w_up_exps,    up_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(down_exps_buf, 0, w_down_exps_f, down_exps_bytes) == 0);

    float *output_mtl = calloc(n_embd, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_write(out_buf, 0, output_mtl, output_bytes) == 0);

    const uint64_t q4k_gate_stride = (uint64_t)(n_embd / 256) * 144;
    const uint64_t q4k_up_stride   = q4k_gate_stride;
    CHECK(ds3_metal_moe_ffn(in_buf, gate_inp_buf, gate_exps_buf, up_exps_buf, down_exps_buf,
                            DS3_TYPE_Q4_K, DS3_TYPE_Q4_K, DS3_TYPE_F32,
                            q4k_gate_stride, q4k_up_stride, n_ff_exp * sizeof(float),
                            NULL, NULL, NULL,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            0, 0, 0,
                            out_buf, logits_buf, NULL, hidden_buf, up_buf, down_buf,
                            n_embd, n_expert, n_used, n_ff_exp, 0,
                            n_embd * sizeof(float), DS3_TYPE_F32, true) == 0);

    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, output_bytes) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)n_embd);
    CHECK(diff < 2e-3f);

    ds3_metal_buffer_free(in_buf);
    ds3_metal_buffer_free(gate_inp_buf);
    ds3_metal_buffer_free(gate_exps_buf);
    ds3_metal_buffer_free(up_exps_buf);
    ds3_metal_buffer_free(down_exps_buf);
    ds3_metal_buffer_free(out_buf);
    ds3_metal_buffer_free(logits_buf);
    ds3_metal_buffer_free(hidden_buf);
    ds3_metal_buffer_free(up_buf);
    ds3_metal_buffer_free(down_buf);
    free(input); free(w_gate_inp);
    free(w_gate_exps_f); free(w_up_exps_f); free(w_down_exps_f);
    free(w_gate_exps); free(w_up_exps);
    free(output_ref); free(gate_logits); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

/* Mixed-type MoE: gate/up are Q4_K, down is Q6_K.
 * This is the actual combination seen in Qwen3-30B-A3B-Q4_K_M.gguf. */
static int test_moe_ffn_mixed_q4k_q6k(void) {
    fprintf(stderr, "test_moe_ffn_mixed_q4k_q6k ... ");

    const uint32_t n_embd    = 2048;
    const uint32_t n_expert  = 128;
    const uint32_t n_used    = 8;
    const uint32_t n_ff_exp  = 768;

    const size_t input_bytes     = n_embd * sizeof(float);
    const size_t logits_bytes    = n_expert * sizeof(float);
    const size_t hidden_bytes    = n_ff_exp * sizeof(float);
    const size_t output_bytes    = n_embd * sizeof(float);
    const size_t gate_inp_bytes  = n_expert * n_embd * sizeof(float);

    const int nb_gate_in = (int)(n_embd / 256);
    const int nb_up_in   = nb_gate_in;
    const int nb_down_in = (int)(n_ff_exp / 256);
    const size_t gate_exps_bytes = (size_t)n_expert * n_ff_exp * nb_gate_in * 144;
    const size_t up_exps_bytes   = gate_exps_bytes;
    const size_t down_exps_bytes = (size_t)n_expert * n_embd * nb_down_in * 210;

    float *input        = calloc(n_embd, sizeof(float));
    float *w_gate_inp   = calloc(n_expert * n_embd, sizeof(float));
    float *w_gate_exps_f= calloc((size_t)n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_up_exps_f  = calloc((size_t)n_expert * n_ff_exp * n_embd, sizeof(float));
    float *w_down_exps_f= calloc((size_t)n_expert * n_embd * n_ff_exp, sizeof(float));
    ds3_test_block_q4_K *w_gate_exps = calloc(1, gate_exps_bytes);
    ds3_test_block_q4_K *w_up_exps   = calloc(1, up_exps_bytes);
    ds3_test_block_q6_K *w_down_exps = calloc(1, down_exps_bytes);
    float *output_ref  = calloc(n_embd, sizeof(float));
    float *gate_logits = calloc(n_expert, sizeof(float));
    int    expert_indices[8];
    float  expert_scores[8];
    CHECK(input && w_gate_inp && w_gate_exps_f && w_up_exps_f && w_down_exps_f &&
          w_gate_exps && w_up_exps && w_down_exps && output_ref && gate_logits);

    srand(74);
    for (uint32_t i = 0; i < n_embd; i++) input[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_embd; i++) w_gate_inp[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    for (uint32_t i = 0; i < n_expert * n_ff_exp * n_embd; i++) {
        w_gate_exps_f[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        w_up_exps_f[i]   = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (uint32_t i = 0; i < n_expert * n_embd * n_ff_exp; i++) {
        w_down_exps_f[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    /* Quantize gate/up to Q4_K and down to Q6_K. */
    for (uint32_t e = 0; e < n_expert; e++) {
        for (uint32_t r = 0; r < n_ff_exp; r++) {
            quantize_row_q4_K(w_gate_exps_f + (e * n_ff_exp + r) * n_embd,
                              w_gate_exps + (e * n_ff_exp + r) * nb_gate_in,
                              (int)n_embd);
            quantize_row_q4_K(w_up_exps_f + (e * n_ff_exp + r) * n_embd,
                              w_up_exps + (e * n_ff_exp + r) * nb_up_in,
                              (int)n_embd);
        }
        for (uint32_t r = 0; r < n_embd; r++) {
            quantize_row_q6_K(w_down_exps_f + (e * n_embd + r) * n_ff_exp,
                              w_down_exps + (e * n_embd + r) * nb_down_in,
                              (int)n_ff_exp);
        }
    }

    /* CPU reference uses dequantized gate/up/down. */
    {
        ds3_tensor_t t_gate = {0}; t_gate.type = DS3_TYPE_Q4_K; t_gate.n_dims = 2;
        t_gate.ne[0] = n_embd; t_gate.ne[1] = n_expert * n_ff_exp;
        t_gate.data = w_gate_exps; t_gate.size = gate_exps_bytes;
        float *d_gate = ds3_ref_dequantize_tensor(&t_gate);
        CHECK(d_gate);

        ds3_tensor_t t_up = {0}; t_up.type = DS3_TYPE_Q4_K; t_up.n_dims = 2;
        t_up.ne[0] = n_embd; t_up.ne[1] = n_expert * n_ff_exp;
        t_up.data = w_up_exps; t_up.size = up_exps_bytes;
        float *d_up = ds3_ref_dequantize_tensor(&t_up);
        CHECK(d_up);

        ds3_tensor_t t_down = {0}; t_down.type = DS3_TYPE_Q6_K; t_down.n_dims = 2;
        t_down.ne[0] = n_ff_exp; t_down.ne[1] = n_expert * n_embd;
        t_down.data = w_down_exps; t_down.size = down_exps_bytes;
        float *d_down = ds3_ref_dequantize_tensor(&t_down);
        CHECK(d_down);

        ds3_ref_moe_ffn(input, w_gate_inp, d_gate, d_up, d_down,
                        NULL, NULL, NULL,
                        output_ref, gate_logits, expert_scores, expert_indices,
                        (int)n_embd, (int)n_expert, (int)n_used, (int)n_ff_exp, 0, true);

        free(d_gate); free(d_up); free(d_down);
    }

    ds3_metal_buffer_t *in_buf       = ds3_metal_buffer_alloc(input_bytes);
    ds3_metal_buffer_t *gate_inp_buf = ds3_metal_buffer_alloc(gate_inp_bytes);
    ds3_metal_buffer_t *gate_exps_buf= ds3_metal_buffer_alloc(gate_exps_bytes);
    ds3_metal_buffer_t *up_exps_buf  = ds3_metal_buffer_alloc(up_exps_bytes);
    ds3_metal_buffer_t *down_exps_buf= ds3_metal_buffer_alloc(down_exps_bytes);
    ds3_metal_buffer_t *out_buf      = ds3_metal_buffer_alloc(output_bytes);
    ds3_metal_buffer_t *logits_buf   = ds3_metal_buffer_alloc(logits_bytes);
    ds3_metal_buffer_t *hidden_buf   = ds3_metal_buffer_alloc(hidden_bytes);
    ds3_metal_buffer_t *up_buf       = ds3_metal_buffer_alloc(hidden_bytes);
    ds3_metal_buffer_t *down_buf     = ds3_metal_buffer_alloc(output_bytes);
    CHECK(in_buf && gate_inp_buf && gate_exps_buf && up_exps_buf && down_exps_buf &&
          out_buf && logits_buf && hidden_buf && up_buf && down_buf);

    CHECK(ds3_metal_buffer_write(in_buf,        0, input,        input_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_inp_buf,  0, w_gate_inp,   gate_inp_bytes) == 0);
    CHECK(ds3_metal_buffer_write(gate_exps_buf, 0, w_gate_exps,  gate_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(up_exps_buf,   0, w_up_exps,    up_exps_bytes) == 0);
    CHECK(ds3_metal_buffer_write(down_exps_buf, 0, w_down_exps,  down_exps_bytes) == 0);

    float *output_mtl = calloc(n_embd, sizeof(float));
    CHECK(output_mtl);
    CHECK(ds3_metal_buffer_write(out_buf, 0, output_mtl, output_bytes) == 0);

    const uint64_t q4k_gate_stride = (uint64_t)nb_gate_in * 144;
    const uint64_t q4k_up_stride   = q4k_gate_stride;
    const uint64_t q6k_down_stride = (uint64_t)nb_down_in * 210;
    CHECK(ds3_metal_moe_ffn(in_buf, gate_inp_buf, gate_exps_buf, up_exps_buf, down_exps_buf,
                            DS3_TYPE_Q4_K, DS3_TYPE_Q4_K, DS3_TYPE_Q6_K,
                            q4k_gate_stride, q4k_up_stride, q6k_down_stride,
                            NULL, NULL, NULL,
                            DS3_TYPE_F32, DS3_TYPE_F32, DS3_TYPE_F32,
                            0, 0, 0,
                            out_buf, logits_buf, NULL, hidden_buf, up_buf, down_buf,
                            n_embd, n_expert, n_used, n_ff_exp, 0,
                            n_embd * sizeof(float), DS3_TYPE_F32, true) == 0);

    CHECK(ds3_metal_buffer_read(out_buf, 0, output_mtl, output_bytes) == 0);

    float diff = max_diff(output_ref, output_mtl, (int)n_embd);
    CHECK(diff < 2e-3f);

    ds3_metal_buffer_free(in_buf);
    ds3_metal_buffer_free(gate_inp_buf);
    ds3_metal_buffer_free(gate_exps_buf);
    ds3_metal_buffer_free(up_exps_buf);
    ds3_metal_buffer_free(down_exps_buf);
    ds3_metal_buffer_free(out_buf);
    ds3_metal_buffer_free(logits_buf);
    ds3_metal_buffer_free(hidden_buf);
    ds3_metal_buffer_free(up_buf);
    ds3_metal_buffer_free(down_buf);
    free(input); free(w_gate_inp);
    free(w_gate_exps_f); free(w_up_exps_f); free(w_down_exps_f);
    free(w_gate_exps); free(w_up_exps); free(w_down_exps);
    free(output_ref); free(gate_logits); free(output_mtl);

    fprintf(stderr, "PASS (max_diff=%.2e)\n", diff);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    if (ds3_metal_init() != 0) {
        fprintf(stderr, "Metal init failed, skipping tests.\n");
        return 77;
    }

    int failed = 0;
    failed |= test_moe_ffn_small();
    failed |= test_moe_ffn_qwen3_dims();
    failed |= test_moe_ffn_qwen3_dims_q4k();
    failed |= test_moe_ffn_qwen3_dims_mixed();
    failed |= test_moe_ffn_mixed_q4k_q6k();

    ds3_metal_shutdown();

    if (failed) {
        fprintf(stderr, "\n%d test(s) FAILED\n", failed);
        return 1;
    }
    fprintf(stderr, "\nAll MoE tests PASSED\n");
    return 0;
}
