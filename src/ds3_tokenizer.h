/*
 * ds3_tokenizer.h — GPT-2 BPE tokenizer for Qwen3
 *
 * Loads token vocabulary and BPE merge ranks from GGUF metadata.
 * Implements standard GPT-2 byte-level BPE encoding/decoding with
 * Qwen3 chat template support.
 */

#ifndef DS3_TOKENIZER_H
#define DS3_TOKENIZER_H

#include "ds3_gguf.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Data Structures
 * ============================================================================ */

/* Open-addressing hash: string → int32 */
typedef struct {
    ds3_gguf_str_t key;
    int            value;
    bool           used;
} ds3_str_i32_entry_t;

typedef struct {
    ds3_str_i32_entry_t *entry;
    uint32_t              cap;
    uint32_t              used;
} ds3_str_i32_table_t;

/*
 * Qwen3 vocabulary.
 *
 * Loaded from GGUF metadata keys:
 *   tokenizer.ggml.tokens  — [STRING]  151936 token strings
 *   tokenizer.ggml.scores  — [FLOAT32] token scores (optional, for reference)
 *   tokenizer.ggml.merges  — [STRING]  BPE merge pairs (e.g., "Ġ t")
 *   tokenizer.ggml.bos_token_id, eos_token_id, etc.
 */
typedef struct {
    char    **token;          /* null-terminated token strings, indexed by token ID */
    int       n_vocab;

    /* Special token IDs (looked up from vocabulary) */
    int       bos_id;
    int       eos_id;
    int       pad_id;
    int       im_start_id;    /* <|im_start|> */
    int       im_end_id;      /* <|im_end|> (same as eos_id in Qwen3) */
    int       think_end_id;   /* </think> */

    /* Hash tables for encoding */
    ds3_str_i32_table_t  token_to_id;   /* token string → token ID */
    ds3_str_i32_table_t  merge_rank;    /* "a b" → BPE merge rank */

    /* Added tokens (e.g., <|im_start|>) — matched before BPE pre-tokenization */
    char    **added_token;
    int      *added_token_id;
    int       n_added;
} ds3_vocab_t;

/* ============================================================================
 * API
 * ============================================================================ */

/* Load vocabulary from GGUF metadata. Returns false if required data is missing. */
bool ds3_vocab_load(ds3_vocab_t *vocab, const ds3_gguf_t *gguf);
void ds3_vocab_free(ds3_vocab_t *vocab);

/* Encode text into token IDs using BPE. Returns number of tokens written (may be
 * fewer than max_tokens if text is short). Returns -1 on error. */
int ds3_vocab_encode(const ds3_vocab_t *vocab, const char *text,
                     int *tokens, int max_tokens);

/* Decode a single token ID to its raw BPE token string.
 * NOTE: This is the byte-encoded form (e.g. "ĠHello").  For human-readable
 * output, use ds3_vocab_decode_sequence which does byte-decoding. */
const char *ds3_vocab_decode(const ds3_vocab_t *vocab, int token_id);

/* Decode a token sequence into human-readable text (byte-decoded, UTF-8).
 * Returns number of bytes written (excluding null terminator), or -1 on error.
 * The output is always null-terminated. Truncates if buf_size is insufficient. */
int ds3_vocab_decode_sequence(const ds3_vocab_t *vocab,
                              const int *tokens, int n_tokens,
                              char *buf, size_t buf_size);

/* Build a Qwen3 chat prompt.
 *
 * Template:
 *   <|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{USER}<|im_end|>\n<|im_start|>assistant\n
 *
 * If system is NULL or empty, the system message is omitted.
 * The returned tokens include the assistant prefix, ready for model generation.
 * Returns number of tokens written (may be fewer than max_tokens). */
int ds3_chat_format(const ds3_vocab_t *vocab,
                    const char *system,
                    const char *user,
                    int *tokens, int max_tokens);

#ifdef __cplusplus
}
#endif

#endif /* DS3_TOKENIZER_H */
