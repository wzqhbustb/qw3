/*
 * tools/dump_logits_llama.cpp — Dump top-5 logits per decode step using llama.cpp.
 *
 * Usage:
 *   ./dump_logits_llama -m model.gguf -p "raw prompt string" [-n steps]
 *
 * The prompt should already include any special tokens / chat formatting that
 * the comparison engine expects (e.g. <|im_start|>user\nHello<|im_end|>...).
 */

#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static void print_usage(int, char **argv) {
    printf("\nusage:\n");
    printf("    %s -m model.gguf -p \"prompt\" [-n n_predict]\n\n", argv[0]);
}

struct top5_t {
    llama_token id;
    float val;
};

static void update_top5(top5_t *top, llama_token id, float val) {
    int worst = 0;
    for (int i = 1; i < 5; i++) {
        if (top[i].val < top[worst].val) worst = i;
    }
    if (val > top[worst].val) {
        top[worst].id = id;
        top[worst].val = val;
    }
}

int main(int argc, char **argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt;
    int n_predict = 20;
    int ngl = 99;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) model_path = argv[++i];
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) prompt = argv[++i];
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) n_predict = std::stoi(argv[++i]);
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) ngl = std::stoi(argv[++i]);
                else { print_usage(argc, argv); return 1; }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        }
    }

    if (model_path.empty() || prompt.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    /* Tokenize raw prompt, parse special tokens, do not add BOS. */
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                         NULL, 0, false, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "%s: error: empty prompt\n", __func__);
        llama_model_free(model);
        return 1;
    }
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(), false, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize prompt\n", __func__);
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict + 32;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = true;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    for (int step = 0; step < n_predict; step++) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s: error: llama_decode failed at step %d\n", __func__, step);
            break;
        }

        const float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            fprintf(stderr, "%s: error: no logits at step %d\n", __func__, step);
            break;
        }

        const int n_vocab = llama_vocab_n_tokens(vocab);
        top5_t top[5];
        for (int i = 0; i < 5; i++) {
            top[i].id = -1;
            top[i].val = -INFINITY;
        }
        for (int i = 0; i < n_vocab; i++) {
            update_top5(top, i, logits[i]);
        }
        std::sort(std::begin(top), std::end(top),
                  [](const top5_t &a, const top5_t &b) { return a.val > b.val; });

        fprintf(stderr, "[logits step=%d]", step);
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, " %d:%.4f", top[i].id, top[i].val);
        }
        fprintf(stderr, "\n");

        /* Greedy next token so we can continue the prefix. */
        llama_token next_id = top[0].id;
        if (llama_vocab_is_eog(vocab, next_id)) break;

        batch = llama_batch_get_one(&next_id, 1);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
