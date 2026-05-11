/*
 * tests/test_tokenizer.c — Tokenizer unit test
 *
 * Usage:
 *   ./test_tokenizer model.gguf              # Load vocab, test encode/decode/chat
 *   ./test_tokenizer model.gguf "Hello!"     # Tokenize specific text
 *
 * Does NOT require Metal. Validates:
 *   1. Vocab loading from GGUF metadata
 *   2. BPE encoding (text → tokens)
 *   3. Token decode (token → string)
 *   4. Chat template formatting
 *   5. Special token IDs
 */

#include "src/ds3_tokenizer.h"
#include "src/ds3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [text to tokenize]\n", argv[0]);
        return 1;
    }

    const char *gguf_path = argv[1];
    const char *test_text = argc > 2 ? argv[2] : NULL;

    /* Open GGUF */
    printf("Opening: %s\n", gguf_path);
    ds3_gguf_t *gguf = ds3_gguf_open(gguf_path);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF file.\n");
        return 1;
    }

    /* Load vocabulary */
    printf("\n═══ Vocab Load ═══\n");
    ds3_vocab_t vocab;
    if (!ds3_vocab_load(&vocab, gguf)) {
        fprintf(stderr, "Failed to load vocabulary from GGUF.\n");
        ds3_gguf_close(gguf);
        return 1;
    }

    /* Check special tokens */
    printf("\n═══ Special Tokens ═══\n");
    printf("  BOS:       %d → %s\n", vocab.bos_id,
           vocab.bos_id < vocab.n_vocab ? vocab.token[vocab.bos_id] : "OOB");
    printf("  EOS:       %d → %s\n", vocab.eos_id,
           vocab.eos_id < vocab.n_vocab ? vocab.token[vocab.eos_id] : "OOB");
    printf("  PAD:       %d → %s\n", vocab.pad_id,
           vocab.pad_id < vocab.n_vocab ? vocab.token[vocab.pad_id] : "OOB");
    printf("  IM_START:  %d → %s\n", vocab.im_start_id,
           vocab.im_start_id < vocab.n_vocab ? vocab.token[vocab.im_start_id] : "OOB");
    printf("  IM_END:    %d → %s\n", vocab.im_end_id,
           vocab.im_end_id < vocab.n_vocab ? vocab.token[vocab.im_end_id] : "OOB");

    /* Compare with ds3.h constants */
    printf("\n  ds3.h vs GGUF:\n");
    printf("    BOS:      ds3.h=%d  gguf=%d  %s\n",
           DS3_BOS_ID, vocab.bos_id,
           DS3_BOS_ID == vocab.bos_id ? "OK" : "MISMATCH");
    printf("    EOS:      ds3.h=%d  gguf=%d  %s\n",
           DS3_EOS_ID, vocab.eos_id,
           DS3_EOS_ID == vocab.eos_id ? "OK" : "MISMATCH");
    printf("    IM_START: ds3.h=%d  gguf=%d  %s\n",
           DS3_IM_START_ID, vocab.im_start_id,
           DS3_IM_START_ID == vocab.im_start_id ? "OK" : "MISMATCH");
    printf("    IM_END:   ds3.h=%d  gguf=%d  %s\n",
           DS3_IM_END_ID, vocab.im_end_id,
           DS3_IM_END_ID == vocab.im_end_id ? "OK" : "MISMATCH");

    /* Test encode/decode round-trip */
    if (test_text) {
        printf("\n═══ Encode/Decode: \"%s\" ═══\n", test_text);

        int tokens[1024];
        int n = ds3_vocab_encode(&vocab, test_text, tokens, 1024);
        printf("  Tokens (%d):", n);
        for (int i = 0; i < n; i++) {
            printf(" %d", tokens[i]);
        }
        printf("\n");

        /* Raw token strings */
        printf("  Raw tokens:");
        for (int i = 0; i < n && i < 20; i++) {
            const char *s = ds3_vocab_decode(&vocab, tokens[i]);
            if (s) printf(" [%s]", s);
        }
        if (n > 20) printf(" ...");
        printf("\n");

        /* Human-readable decode */
        char decoded[4096];
        int dlen = ds3_vocab_decode_sequence(&vocab, tokens, n,
                                             decoded, sizeof(decoded));
        if (dlen >= 0) {
            printf("  Decoded text: \"%s\"\n", decoded);
        }
    } else {
        /* Default: test some sample texts */
        const char *samples[] = {
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "int main() { return 0; }",
            "人工智能",
            "Qwen3 is a large language model.",
        };
        int n_samples = (int)(sizeof(samples) / sizeof(samples[0]));
        for (int si = 0; si < n_samples; si++) {
            printf("\n═══ Encode: \"%s\" ═══\n", samples[si]);
            int tokens[512];
            int n = ds3_vocab_encode(&vocab, samples[si], tokens, 512);
            printf("  Tokens (%d):", n);
            for (int i = 0; i < n && i < 30; i++) {
                printf(" %d", tokens[i]);
            }
            if (n > 30) printf(" ...");
            printf("\n");

            /* Human-readable decode */
            char decoded[4096];
            int dlen = ds3_vocab_decode_sequence(&vocab, tokens, n,
                                                 decoded, sizeof(decoded));
            if (dlen >= 0) {
                printf("  Decoded text: \"%s\"\n", decoded);
            }
        }
    }

    /* Test chat template */
    {
        printf("\n═══ Chat Template ═══\n");
        int chat_tokens[2048];
        int n = ds3_chat_format(&vocab,
                                "You are a helpful assistant.",
                                "What is the capital of France?",
                                chat_tokens, 2048);
        printf("  Chat tokens (%d):", n);
        for (int i = 0; i < n && i < 40; i++) {
            const char *s = ds3_vocab_decode(&vocab, chat_tokens[i]);
            if (s) {
                if (chat_tokens[i] == vocab.im_start_id ||
                    chat_tokens[i] == vocab.im_end_id) {
                    printf(" [<%s>]", s);
                } else {
                    printf(" %s", s);
                }
            } else {
                printf(" %d", chat_tokens[i]);
            }
        }
        if (n > 40) printf(" ...");
        printf("\n");

        /* Human-readable version */
        char decoded[8192];
        int dlen = ds3_vocab_decode_sequence(&vocab, chat_tokens, n,
                                             decoded, sizeof(decoded));
        if (dlen >= 0) {
            printf("  Decoded: %s\n", decoded);
        }
    }

    /* Cleanup */
    ds3_vocab_free(&vocab);
    ds3_gguf_close(gguf);
    printf("\nDone.\n");
    return 0;
}
