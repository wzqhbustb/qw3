/*
 * tools/qwen3-cli.c — Minimal Qwen3 chat/reasoning CLI
 *
 * Usage:
 *   ./qwen3-cli -m <model.gguf> -p "Hello" [-s "system prompt"] \
 *               [-n 128] [-t 0.7] [-c 4096]
 */

#include "src/ds3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

static void print_usage(const char *argv0)
{
    fprintf(stderr,
        "Usage: %s -m <model.gguf> -p <user prompt> [options]\n"
        "Options:\n"
        "  -m PATH   Path to Qwen3 GGUF model\n"
        "  -p TEXT   User prompt\n"
        "  -s TEXT   System prompt (optional)\n"
        "  -n N      Number of tokens to generate (default: 128)\n"
        "  -t TEMP   Sampling temperature (default: 0.7, 0 = argmax)\n"
        "  -c CTX    Context length / KV cache size (default: 4096)\n"
        "  -d        Dump top-5 logits per generated token (Phase 2.7)\n"
        "  -L        Run Layer 0 reference comparison and exit\n"
        "  -F        Run full-model reference comparison for one token and exit\n"
        "  -P        Run per-layer reference comparison for one token and exit\n"
        "  -C N      Run multi-step decode compare on first N prompt+gen tokens\n"
        "  -i        Interactive mode (prompt after each response)\n"
        "  -q        Quiet mode: suppress non-error informational output\n",
        argv0);
}

typedef struct {
    int id;
    float val;
} top5_t;

static void update_top5(top5_t *top, int id, float val)
{
    int worst = 0;
    for (int i = 1; i < 5; i++) {
        if (top[i].val < top[worst].val) worst = i;
    }
    if (val > top[worst].val) {
        top[worst].id = id;
        top[worst].val = val;
    }
}

static void print_top5_logits(int step, const float *logits, int n_vocab,
                              int selected_token, void *user)
{
    (void)user;
    top5_t top[5];
    for (int i = 0; i < 5; i++) {
        top[i].id = -1;
        top[i].val = -INFINITY;
    }
    for (int i = 0; i < n_vocab; i++) {
        update_top5(top, i, logits[i]);
    }

    /* Sort descending. */
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 5; j++) {
            if (top[j].val > top[i].val) {
                top5_t tmp = top[i];
                top[i] = top[j];
                top[j] = tmp;
            }
        }
    }

    fprintf(stderr, "[logits step=%d selected=%d]", step, selected_token);
    for (int i = 0; i < 5; i++) {
        fprintf(stderr, " %d:%.4f", top[i].id, top[i].val);
    }
    fprintf(stderr, "\n");
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *user_prompt = NULL;
    const char *system_prompt = NULL;
    int n_predict = 128;
    float temperature = 0.7f;
    int n_ctx = 4096;
    bool interactive = false;
    bool dump_logits = false;
    bool layer0_compare = false;
    bool full_compare = false;
    bool per_layer_compare = false;
    bool quiet = false;
    int  decode_compare_steps = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            user_prompt = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0) {
            dump_logits = true;
        } else if (strcmp(argv[i], "-L") == 0) {
            layer0_compare = true;
        } else if (strcmp(argv[i], "-F") == 0) {
            full_compare = true;
        } else if (strcmp(argv[i], "-P") == 0) {
            per_layer_compare = true;
        } else if (strcmp(argv[i], "-C") == 0 && i + 1 < argc) {
            decode_compare_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            quiet = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (quiet) {
        ds3_set_quiet(1);
    }

    if (!model_path || !user_prompt) {
        print_usage(argv[0]);
        return 1;
    }

    ds3_engine_t *engine = ds3_engine_open(model_path, n_ctx);
    if (!engine) {
        fprintf(stderr, "Failed to load engine from %s\n", model_path);
        return 1;
    }

    ds3_print_model_info(engine);

    if (layer0_compare) {
        int rc_attn = ds3_engine_debug_layer0_compare(engine, DS3_BOS_ID);
        int rc_full = ds3_engine_debug_layer0_full_compare(engine, DS3_BOS_ID);
        ds3_engine_close(engine);
        return (rc_attn == 0 && rc_full == 0) ? 0 : 1;
    }

    if (full_compare) {
        int rc = ds3_engine_debug_full_compare(engine, DS3_BOS_ID);
        ds3_engine_close(engine);
        return (rc == 0) ? 0 : 1;
    }

    if (per_layer_compare) {
        int rc = ds3_engine_debug_per_layer_compare(engine, DS3_BOS_ID);
        ds3_engine_close(engine);
        return (rc == 0) ? 0 : 1;
    }

    int *tokens = (int *)malloc(8192 * sizeof(int));
    if (!tokens) {
        fprintf(stderr, "Out of memory\n");
        ds3_engine_close(engine);
        return 1;
    }

    do {
        int n_tokens = ds3_engine_chat_format(engine, system_prompt, user_prompt,
                                              tokens, 8192);
        if (n_tokens < 0) {
            fprintf(stderr, "Failed to format chat prompt\n");
            break;
        }
        fprintf(stderr, "\n<|user|>\n%s\n\n<|assistant|>\n", user_prompt);
        fflush(stderr);

        int *output = (int *)malloc((size_t)n_predict * sizeof(int));
        if (!output) {
            fprintf(stderr, "Out of memory\n");
            break;
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ds3_engine_logit_cb_t cb = dump_logits ? print_top5_logits : NULL;
        int n_gen = ds3_engine_generate_ex(engine, tokens, n_tokens, n_predict,
                                           temperature, output, n_predict,
                                           cb, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (n_gen < 0) {
            fprintf(stderr, "Generation failed\n");
            free(output);
            break;
        }
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        ds3_log_info("\n[gen] %d tokens in %.2f s = %.2f tok/s\n",
                n_gen, elapsed, n_gen / elapsed);

        /* Up to ~64 bytes per token is enough for any Qwen3 BPE token. */
        size_t text_cap = (size_t)n_gen * 64 + 1;
        char *text = (char *)malloc(text_cap);
        if (text) {
            int bytes = ds3_engine_decode_sequence(engine, output, n_gen,
                                                   text, text_cap);
            if (bytes > 0) {
                printf("%s\n", text);
                fflush(stdout);
            }
            free(text);
        }

        if (decode_compare_steps > 0) {
            int n_compare = n_tokens + n_gen;
            if (decode_compare_steps < n_compare) n_compare = decode_compare_steps;
            int *compare_tokens = (int *)malloc((size_t)n_compare * sizeof(int));
            if (compare_tokens) {
                memcpy(compare_tokens, tokens, (size_t)n_tokens * sizeof(int));
                int gen_part = n_compare - n_tokens;
                if (gen_part > 0) {
                    memcpy(compare_tokens + n_tokens, output,
                           (size_t)gen_part * sizeof(int));
                }
                ds3_engine_debug_decode_compare(engine, compare_tokens, n_compare);
                free(compare_tokens);
            }
        }

        free(output);

        if (!interactive) break;

        printf("\n> ");
        fflush(stdout);
        static char line[4096];
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';
        if (line[0] == '\0') break;
        user_prompt = line;
        system_prompt = NULL;  /* only use system prompt on first turn */
    } while (1);

    free(tokens);
    ds3_engine_close(engine);
    return 0;
}
