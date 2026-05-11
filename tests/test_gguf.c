/*
 * tests/test_gguf.c — Standalone GGUF loader test
 *
 * Usage:
 *   ./test_gguf model.gguf           # Parse and print summary + all tensors
 *   ./test_gguf model.gguf -v        # Verbose: also print metadata
 *   ./test_gguf model.gguf -b        # Also try binding to ds3_weights_t
 *
 * This does NOT require Metal. It validates:
 *   1. mmap works
 *   2. Header parses correctly
 *   3. All tensor descriptors are readable
 *   4. Tensor data pointers land inside mmap
 *   5. (optional) Qwen3 tensor naming matches ds3_weights.c expectations
 */

#include "src/ds3_gguf.h"
#include "src/ds3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.gguf> [-v] [-b]\n", prog);
    fprintf(stderr, "  -v  Verbose: print all metadata KV pairs\n");
    fprintf(stderr, "  -b  Bind: try to match tensors to ds3_weights_t structure\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *gguf_path = argv[1];
    bool verbose = false;
    bool do_bind = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose = true;
        if (strcmp(argv[i], "-b") == 0) do_bind = true;
    }

    printf("Opening: %s\n", gguf_path);
    ds3_gguf_t *gguf = ds3_gguf_open(gguf_path);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF file.\n");
        return 1;
    }

    /* ── Summary ── */
    ds3_gguf_print_summary(gguf);

    /* ── Metadata ── */
    if (verbose) {
        ds3_gguf_print_metadata(gguf);
    } else {
        /* Print a few key metadata even in non-verbose mode */
        printf("\n═══ Key Metadata ═══\n");
        const char *arch = NULL;
        size_t arch_len = 0;
        if (ds3_gguf_get_metadata_string(gguf, "general.architecture", &arch, &arch_len)) {
            printf("  Architecture: %.*s\n", (int)arch_len, arch);
        }
        uint32_t file_type = 0;
        if (ds3_gguf_get_metadata_u32(gguf, "general.file_type", &file_type)) {
            printf("  File type: %u\n", file_type);
        }
    }

    /* ── Tensors ── */
    ds3_gguf_print_tensors(gguf);

    /* ── Validation: check tensor data pointers are inside mmap ── */
    printf("\n══─ Validation ─══\n");
    bool ptrs_ok = true;
    uint64_t total_size = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        const ds3_gguf_tensor_info_t *t = &gguf->tensors[i];
        total_size += t->size;
        const uint8_t *end = (const uint8_t *)t->data + t->size;
        const uint8_t *mmap_end = (const uint8_t *)gguf->mmap_base + gguf->file_size;
        if (end > mmap_end) {
            fprintf(stderr, "  FAIL: tensor '%.*s' data range exceeds mmap!\n",
                    (int)t->name.len, t->name.data);
            ptrs_ok = false;
        }
    }
    if (ptrs_ok) {
        printf("  ✓ All %llu tensor data pointers inside mmap\n", (unsigned long long)gguf->n_tensors);
        printf("  ✓ Total weight bytes: %llu (%.2f MB)\n",
               (unsigned long long)total_size, total_size / (1024.0 * 1024.0));
    }

    /* ── Optional: bind to ds3_weights_t ── */
    if (do_bind) {
        printf("\n══─ Weight Binding ─══\n");
        ds3_weights_t *w = ds3_weights_load(gguf);
        if (w) {
            ds3_weights_print_summary(w);
            ds3_weights_free(w);
        } else {
            fprintf(stderr, "  Weight binding failed.\n");
        }
    }

    ds3_gguf_close(gguf);
    printf("\nDone.\n");
    return 0;
}
