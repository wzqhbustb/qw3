/*
 * ds3_tokenizer.c — Qwen3 BPE tokenizer (tiktoken/qwen2 style)
 *
 * Implements byte-level BPE encoding matching the Qwen3 tokenizer behaviour.
 * Vocabulary and merge ranks are loaded from GGUF metadata (tokenizer.ggml.* keys).
 *
 * Pre-tokenization follows the Qwen3 regex (tokenizer.ggml.pre = qwen2):
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   |[^\r\n\p{L}\p{N}]?\p{L}+
 *   |\p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 *
 * Byte-level encoding maps raw bytes 0-255 into visible Unicode characters,
 * following the GPT-2 bytes_to_unicode() table.
 */

#include "ds3_tokenizer.h"
#include "ds3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ============================================================================
 * Hash: FNV-1a-ish
 * ============================================================================ */

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = (const uint8_t *)ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* ============================================================================
 * Hash table: string → int32 (open addressing, linear probing)
 * ============================================================================ */

static uint32_t next_pow2(uint32_t n) {
    uint32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(ds3_str_i32_table_t *t, uint32_t expected) {
    /* Capacity = next power-of-2 of ~2× expected → load factor ≤ 50%.
     * 151K tokens at 50% → ~300K slots, linear probing is fine. */
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = calloc(t->cap, sizeof(t->entry[0]));
}

static void table_free(ds3_str_i32_table_t *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(ds3_str_i32_table_t *t, ds3_gguf_str_t key, int value) {
    uint32_t mask = t->cap - 1;
    uint32_t i = (uint32_t)(hash_bytes(key.data, key.len) & mask);

    while (t->entry[i].used) {
        if (t->entry[i].key.len == key.len &&
            memcmp(t->entry[i].key.data, key.data, key.len) == 0) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used  = true;
    t->entry[i].key   = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const ds3_str_i32_table_t *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint32_t mask = t->cap - 1;
    uint32_t i = (uint32_t)(hash_bytes(ptr, len) & mask);

    while (t->entry[i].used) {
        if (t->entry[i].key.len == len &&
            memcmp(t->entry[i].key.data, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

/* ============================================================================
 * Token vector (dynamic int array)
 * ============================================================================ */

typedef struct {
    int *data;
    int  len;
    int  cap;
} ds3_token_vec_t;

static void token_vec_init(ds3_token_vec_t *tv) {
    memset(tv, 0, sizeof(*tv));
}

static void token_vec_push(ds3_token_vec_t *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->data = realloc(tv->data, (size_t)tv->cap * sizeof(tv->data[0]));
    }
    tv->data[tv->len++] = token;
}

static void token_vec_free(ds3_token_vec_t *tv) {
    free(tv->data);
    memset(tv, 0, sizeof(*tv));
}

/* ============================================================================
 * GPT-2 byte ↔ Unicode mapping
 *
 * GPT-2 BPE maps raw bytes to printable Unicode codepoints so that BPE merges
 * can operate on valid UTF-8 text. Bytes 33-126, 161-172, 174-255 map to
 * themselves; the other bytes map to codepoints 256+.
 * ============================================================================ */

/* GPT-2 byte → codepoint mapping table.
 * Bytes 33-126, 161-172, 174-255 map to themselves.
 * The remaining 94 bytes map to codepoints 256+.
 * Generated once at init time. */
static uint16_t g_byte_to_cp[256];
static uint8_t  g_cp_to_byte[512];  /* codepoints up to 256+94=350 */
static bool     g_byte_tables_ready;

static void init_byte_tables(void) {
    if (g_byte_tables_ready) return;
    uint32_t n = 0;
    for (int b = 0; b < 256; b++) {
        uint8_t byte = (uint8_t)b;
        if ((byte >= 33 && byte <= 126) ||
            (byte >= 161 && byte <= 172) ||
            (byte >= 174)) {
            g_byte_to_cp[byte] = byte;
            g_cp_to_byte[byte] = byte;
        } else {
            uint32_t cp = 256 + n;
            g_byte_to_cp[byte] = (uint16_t)cp;
            if (cp < 512) g_cp_to_byte[cp] = byte;
            n++;
        }
    }
    g_byte_tables_ready = true;
}

static inline uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if (!g_byte_tables_ready) init_byte_tables();
    return g_byte_to_cp[b];
}

static inline void init_cp_to_byte(void) {
    if (!g_byte_tables_ready) init_byte_tables();
}

/* ============================================================================
 * UTF-8 helpers
 * ============================================================================ */

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    (void)s;
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

/* Decode one UTF-8 codepoint, advance *pos past it */
static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    uint8_t c0 = (uint8_t)s[*pos];
    int n = utf8_len_from_first_byte(c0);
    if (*pos + (uint64_t)n > len) n = 1;

    uint32_t cp;
    if (n == 1)      cp = c0;
    else if (n == 2)  cp = ((uint32_t)(c0 & 0x1f) << 6)  | ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f));
    else if (n == 3)  cp = ((uint32_t)(c0 & 0x0f) << 12) | ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6)
                         | ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f));
    else              cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12)
                         | ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6)
                         | ((uint32_t)((uint8_t)s[*pos + 3] & 0x3f));
    *pos += (uint64_t)n;
    return cp;
}

/* ============================================================================
 * Unicode letter classification (approximation of \p{L})
 * ============================================================================ */

static bool is_unicode_letter_cp(uint32_t cp) {
    /* ASCII */
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;

    /* Latin-1 Supplement & Extended */
    if (cp >= 0x00C0 && cp <= 0x024F) return true;

    /* Greek */
    if (cp >= 0x0370 && cp <= 0x03FF) return true;

    /* Cyrillic */
    if (cp >= 0x0400 && cp <= 0x04FF) return true;

    /* Armenian, Hebrew, Arabic, Syriac, Thaana */
    if (cp >= 0x0530 && cp <= 0x05FF) return true;
    if (cp >= 0x0600 && cp <= 0x06FF) return true;

    /* Devanagari & other Indic scripts */
    if (cp >= 0x0900 && cp <= 0x0D7F) return true;

    /* Thai, Lao, Tibetan, Myanmar, Georgian */
    if (cp >= 0x0E00 && cp <= 0x10FF) return true;

    /* Hangul Jamo */
    if (cp >= 0x1100 && cp <= 0x11FF) return true;

    /* Ethiopic, Cherokee, etc. */
    if (cp >= 0x1200 && cp <= 0x137F) return true;

    /* CJK Unified Ideographs */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;

    /* CJK Extension A */
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;

    /* Hiragana, Katakana, Bopomofo */
    if (cp >= 0x3040 && cp <= 0x312F) return true;

    /* Hangul Syllables */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;

    /* CJK Compatibility Ideographs */
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;

    /* Fullwidth Latin */
    if (cp >= 0xFF21 && cp <= 0xFF3A) return true;  /* A-Z */
    if (cp >= 0xFF41 && cp <= 0xFF5A) return true;  /* a-z */

    /* CJK Extensions B-G */
    if (cp >= 0x20000 && cp <= 0x2EBEF) return true;

    return false;
}

static bool is_letter_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c = (uint8_t)s[pos];
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');

    uint64_t p = pos;
    uint32_t cp = utf8_decode_one(s, len, &p);
    return is_unicode_letter_cp(cp);
}

/* ============================================================================
 * Unicode digit classification (approximation of \p{N})
 * ============================================================================ */

static bool is_unicode_digit_cp(uint32_t cp) {
    /* ASCII */
    if (cp >= '0' && cp <= '9') return true;

    /* Arabic-Indic */
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    /* Extended Arabic-Indic */
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;

    /* Devanagari */
    if (cp >= 0x0966 && cp <= 0x096F) return true;
    /* Bengali */
    if (cp >= 0x09E6 && cp <= 0x09EF) return true;
    /* Gurmukhi */
    if (cp >= 0x0A66 && cp <= 0x0A6F) return true;
    /* Gujarati */
    if (cp >= 0x0AE6 && cp <= 0x0AEF) return true;
    /* Oriya */
    if (cp >= 0x0B66 && cp <= 0x0B6F) return true;
    /* Tamil */
    if (cp >= 0x0BE6 && cp <= 0x0BEF) return true;
    /* Telugu */
    if (cp >= 0x0C66 && cp <= 0x0C6F) return true;
    /* Kannada */
    if (cp >= 0x0CE6 && cp <= 0x0CEF) return true;
    /* Malayalam */
    if (cp >= 0x0D66 && cp <= 0x0D6F) return true;

    /* Thai */
    if (cp >= 0x0E50 && cp <= 0x0E59) return true;
    /* Lao */
    if (cp >= 0x0ED0 && cp <= 0x0ED9) return true;
    /* Tibetan */
    if (cp >= 0x0F20 && cp <= 0x0F29) return true;
    /* Myanmar */
    if (cp >= 0x1040 && cp <= 0x1049) return true;
    /* Khmer */
    if (cp >= 0x17E0 && cp <= 0x17E9) return true;
    /* Mongolian */
    if (cp >= 0x1810 && cp <= 0x1819) return true;

    /* Fullwidth */
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;

    return false;
}

static bool is_digit_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c = (uint8_t)s[pos];
    if (c < 0x80) return c >= '0' && c <= '9';

    uint64_t p = pos;
    uint32_t cp = utf8_decode_one(s, len, &p);
    return is_unicode_digit_cp(cp);
}

/* ============================================================================
 * Byte-level encode: raw bytes → visible UTF-8
 * ============================================================================ */

static char *byte_encode(const char *in, uint64_t in_len, uint64_t *out_len) {
    char *out = malloc((size_t)in_len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in_len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

/*
 * Reverse of byte_encode: map the byte-encoded token string back to raw
 * bytes (which are the original UTF-8 text).  This is needed because BPE
 * token strings use mapped codepoints (e.g. "Ġ" = codepoint 288 = raw
 * byte 0x20 = ASCII space).
 */
static char *byte_decode(const char *in, uint64_t in_len, uint64_t *out_len) {
    init_cp_to_byte();

    char *out = malloc((size_t)in_len + 1);
    char *p = out;
    uint64_t pos = 0;

    while (pos < in_len) {
        uint32_t cp = utf8_decode_one(in, in_len, &pos);
        if (cp < 512) {
            *p++ = (char)g_cp_to_byte[cp];
        }
    }

    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

/* ============================================================================
 * BPE merge rank lookup
 *
 * Merge pairs are stored with a space separator in GGUF (e.g. "Ġ t").
 * We reconstruct the pair string "a b" and look it up in merge_rank.
 * ============================================================================ */

typedef struct {
    char    *ptr;
    uint64_t len;
} owned_str_t;

static owned_str_t owned_copy(const char *ptr, uint64_t len) {
    owned_str_t s;
    s.ptr = malloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const ds3_vocab_t *vocab,
                    const owned_str_t *a, const owned_str_t *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : malloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* ============================================================================
 * BPE encode a single pre-tokenized piece
 *
 * 1. Byte-encode the raw text piece → visible UTF-8
 * 2. Split into individual UTF-8 characters
 * 3. Apply BPE merges (lowest rank first) until no more merges possible
 * 4. Look up each resulting symbol in token_to_id
 * ============================================================================ */

static void bpe_encode_piece(const ds3_vocab_t *vocab,
                             const char *raw, uint64_t raw_len,
                             ds3_token_vec_t *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw, raw_len, &encoded_len);

    /* Split into UTF-8 characters */
    int n_sym = 0;
    int cap_sym = 32;
    owned_str_t *sym = calloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = realloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    /*
     * Iteratively apply BPE merges.
     *
     * TODO: O(n² × m) — each iteration scans all pairs for the lowest-rank
     * merge.  Long pieces (rare in practice after pre-tokenization) can be
     * slow.  If 128K-prompt prefill becomes bottlenecked here, replace with
     * a priority-queue (min-heap keyed by rank).
     */
    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        /* Merge sym[best_i] and sym[best_i+1] */
        owned_str_t merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = malloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr,
               (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    /* Look up each symbol in vocab */
    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            /* Fallback: emit byte-by-byte */
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

/* ============================================================================
 * ASCII helpers
 * ============================================================================ */

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

/* Returns exact match length (0 = no contraction match) */
static int match_contraction(const char *s, uint64_t len, uint64_t pos) {
    if (pos >= len || s[pos] != '\'') return 0;
    uint64_t rem = len - pos - 1;
    const char *p = s + pos + 1;

    /* 're, 've, 'll — 3 chars total */
    if (rem >= 2 && ((p[0] == 'r' || p[0] == 'R') && (p[1] == 'e' || p[1] == 'E'))) return 3;
    if (rem >= 2 && ((p[0] == 'v' || p[0] == 'V') && (p[1] == 'e' || p[1] == 'E'))) return 3;
    if (rem >= 2 && ((p[0] == 'l' || p[0] == 'L') && (p[1] == 'l' || p[1] == 'L'))) return 3;
    /* 's, 't, 'm, 'd — 2 chars total */
    if (rem >= 1 && (p[0] == 's' || p[0] == 'S' || p[0] == 't' || p[0] == 'T' ||
                      p[0] == 'm' || p[0] == 'M' || p[0] == 'd' || p[0] == 'D')) return 2;

    return 0;
}

/* ============================================================================
 * Qwen3 Pre-tokenizer
 *
 * Implements the Qwen3 tiktoken-style regex (from tokenizer.ggml.pre = qwen2):
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   |[^\r\n\p{L}\p{N}]?\p{L}+
 *   |\p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 *
 * Branches are evaluated in left-to-right priority order.
 * ============================================================================ */

static void bpe_tokenize_segment(const ds3_vocab_t *vocab, const char *text,
                                   uint64_t seg_start, uint64_t seg_end,
                                   ds3_token_vec_t *out) {
    uint64_t len = seg_end;
    uint64_t pos = seg_start;

    while (pos < len) {
        uint64_t start = pos;

        /* 1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd */
        int clen = match_contraction(text, len, pos);
        if (clen > 0) {
            pos += clen;
            bpe_encode_piece(vocab, text + start, pos - start, out);
            continue;
        }

        /* 2. [^\r\n\p{L}\p{N}]?\p{L}+
         * Optional prefix (not \r\n, not letter, not digit) + one or more letters */
        if (!ascii_newline((uint8_t)text[pos]) && is_letter_at(text, len, pos)) {
            /* Current char is a letter, consume consecutive letters */
            while (pos < len && !ascii_newline((uint8_t)text[pos]) &&
                   is_letter_at(text, len, pos)) {
                pos = next_utf8_char(text, len, pos);
            }
            bpe_encode_piece(vocab, text + start, pos - start, out);
            continue;
        } else {
            /* Current char is not a letter. Check if it's a valid prefix
             * (not \r, not \n, not letter, not digit) followed by a letter. */
            uint64_t next_pos = next_utf8_char(text, len, pos);
            if (next_pos < len &&
                !ascii_newline((uint8_t)text[pos]) &&
                !is_digit_at(text, len, pos) &&
                !ascii_newline((uint8_t)text[next_pos]) &&
                is_letter_at(text, len, next_pos)) {
                /* Prefix + letter sequence */
                pos = next_pos;
                while (pos < len && !ascii_newline((uint8_t)text[pos]) &&
                       is_letter_at(text, len, pos)) {
                    pos = next_utf8_char(text, len, pos);
                }
                bpe_encode_piece(vocab, text + start, pos - start, out);
                continue;
            }
        }

        /* 3. \p{N} — single digit */
        if (is_digit_at(text, len, pos)) {
            pos++;
            bpe_encode_piece(vocab, text + start, pos - start, out);
            continue;
        }

        /* 4. ?[^\s\p{L}\p{N}]+[\r\n]*
         * Optional leading space + one or more non-space non-letter non-digit
         * + optional trailing newlines */
        {
            uint64_t p = pos;
            /* Optional leading space */
            if (p < len && text[p] == ' ') {
                p++;
            }
            /* Content: non-space non-letter non-digit non-newline */
            uint64_t content_start = p;
            while (p < len) {
                uint8_t c = (uint8_t)text[p];
                if (ascii_space(c) || ascii_newline(c) ||
                    is_letter_at(text, len, p) || is_digit_at(text, len, p)) {
                    break;
                }
                p++;
            }
            if (p > content_start) {
                /* Trailing newlines */
                while (p < len && ascii_newline((uint8_t)text[p])) p++;
                pos = p;
                bpe_encode_piece(vocab, text + start, pos - start, out);
                continue;
            }
        }

        /* 5. \s*[\r\n]+
         * Whitespace + one or more newlines, up to the end of newline sequence */
        {
            uint64_t p = pos;
            /* Consume whitespace (space, tab, newline) */
            while (p < len && (ascii_space((uint8_t)text[p]) || ascii_newline((uint8_t)text[p]))) {
                p++;
            }
            /* Find the last newline in the consumed range */
            uint64_t last_nl_end = 0;
            for (uint64_t i = pos; i < p; i++) {
                if (ascii_newline((uint8_t)text[i])) {
                    last_nl_end = i + 1;
                }
            }
            if (last_nl_end > pos) {
                pos = last_nl_end;
                bpe_encode_piece(vocab, text + start, pos - start, out);
                continue;
            }
        }

        /* 6. \s+(?!\S) — trailing whitespace
         * One or more whitespace, with no non-whitespace after the match.
         * Greedy: match as many as possible, but lookahead must succeed. */
        {
            uint64_t p = pos;
            while (p < len && (ascii_space((uint8_t)text[p]) || ascii_newline((uint8_t)text[p]))) {
                p++;
            }
            if (p > pos) {
                uint64_t best_end = pos;
                for (uint64_t end = pos + 1; end <= p; end++) {
                    if (end >= len ||
                        ascii_space((uint8_t)text[end]) ||
                        ascii_newline((uint8_t)text[end])) {
                        best_end = end;
                    } else {
                        break;
                    }
                }
                if (best_end > pos) {
                    pos = best_end;
                    bpe_encode_piece(vocab, text + start, pos - start, out);
                    continue;
                }
            }
        }

        /* 7. \s+ — other whitespace */
        if (ascii_space((uint8_t)text[pos]) || ascii_newline((uint8_t)text[pos])) {
            while (pos < len && (ascii_space((uint8_t)text[pos]) || ascii_newline((uint8_t)text[pos]))) {
                pos++;
            }
            bpe_encode_piece(vocab, text + start, pos - start, out);
            continue;
        }

        /* Fallback */
        pos = next_utf8_char(text, len, pos);
    }
}

static void bpe_tokenize(const ds3_vocab_t *vocab, const char *text,
                         ds3_token_vec_t *out) {
    bpe_tokenize_segment(vocab, text, 0, strlen(text), out);
}

/* ============================================================================
 * Vocabulary loading from GGUF metadata
 * ============================================================================ */

static int vocab_lookup(const ds3_vocab_t *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "[tokenizer] required token missing from vocab: %s\n", text);
        return -1;
    }
    return token;
}

bool ds3_vocab_load(ds3_vocab_t *vocab, const ds3_gguf_t *gguf) {
    if (!vocab || !gguf) return false;
    memset(vocab, 0, sizeof(*vocab));

    /* --- Token strings --- */
    const ds3_gguf_metadata_kv_t *kv_tokens = ds3_gguf_find_metadata(gguf, "tokenizer.ggml.tokens");
    if (!kv_tokens || kv_tokens->value_type != DS3_GGUF_METADATA_TYPE_ARRAY ||
        kv_tokens->value.array.type != DS3_GGUF_METADATA_TYPE_STRING) {
        fprintf(stderr, "[tokenizer] GGUF missing tokenizer.ggml.tokens\n");
        return false;
    }

    uint64_t n_vocab = kv_tokens->value.array.len;
    if (n_vocab > INT32_MAX) return false;
    vocab->n_vocab = (int)n_vocab;

    /* Allocate token string array */
    vocab->token = calloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    if (!vocab->token) return false;

    table_init(&vocab->token_to_id, (uint32_t)n_vocab);

    /* Iterate string array */
    const uint8_t *p = (const uint8_t *)kv_tokens->value.array.data;
    for (int i = 0; i < vocab->n_vocab; i++) {
        ds3_gguf_str_t s;
        p = ds3_gguf_read_str(p, &s);
        vocab->token[i] = strndup(s.data, s.len);
        /* Use original GGUF string (points into mmap) for hash key */
        table_put(&vocab->token_to_id, s, i);
    }

    /* --- BPE merges --- */
    const ds3_gguf_metadata_kv_t *kv_merges = ds3_gguf_find_metadata(gguf, "tokenizer.ggml.merges");
    if (!kv_merges || kv_merges->value_type != DS3_GGUF_METADATA_TYPE_ARRAY ||
        kv_merges->value.array.type != DS3_GGUF_METADATA_TYPE_STRING) {
        fprintf(stderr, "[tokenizer] GGUF missing tokenizer.ggml.merges\n");
        ds3_vocab_free(vocab);
        return false;
    }

    uint64_t n_merges = kv_merges->value.array.len;
    table_init(&vocab->merge_rank, (uint32_t)n_merges);

    p = (const uint8_t *)kv_merges->value.array.data;
    for (uint64_t i = 0; i < n_merges; i++) {
        ds3_gguf_str_t s;
        p = ds3_gguf_read_str(p, &s);
        table_put(&vocab->merge_rank, s, (int)i);
    }

    /* --- Special token IDs --- */
    /* BOS / EOS from GGUF metadata */
    uint32_t bos_u32 = 151643, eos_u32 = 151645, pad_u32 = 151643;
    ds3_gguf_get_metadata_u32(gguf, "tokenizer.ggml.bos_token_id", &bos_u32);
    ds3_gguf_get_metadata_u32(gguf, "tokenizer.ggml.eos_token_id", &eos_u32);
    ds3_gguf_get_metadata_u32(gguf, "tokenizer.ggml.padding_token_id", &pad_u32);
    vocab->bos_id = (int)bos_u32;
    vocab->eos_id = (int)eos_u32;
    vocab->pad_id = (int)pad_u32;

    /* Look up Qwen3 special tokens from vocab */
    vocab->im_start_id = vocab_lookup(vocab, "<|im_start|>");
    vocab->im_end_id   = vocab_lookup(vocab, "<|im_end|>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");

    if (vocab->im_start_id < 0) {
        fprintf(stderr, "[tokenizer] WARNING: <|im_start|> not found in vocab\n");
        vocab->im_start_id = 151644; /* fallback */
    }
    if (vocab->im_end_id < 0) {
        vocab->im_end_id = vocab->eos_id;
    }
    if (vocab->think_end_id < 0) {
        vocab->think_end_id = -1;
    }

    /* --- Added tokens (Qwen3 specific) --- */
    static const char *added_token_strs[] = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<tool_call>", "</tool_call>",
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
        "<|repo_name|>", "<|file_sep|>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>",
        NULL
    };

    int n_added = 0;
    for (int i = 0; added_token_strs[i]; i++) {
        int id = vocab_lookup(vocab, added_token_strs[i]);
        if (id >= 0) n_added++;
    }

    vocab->added_token = calloc((size_t)n_added, sizeof(vocab->added_token[0]));
    vocab->added_token_id = calloc((size_t)n_added, sizeof(vocab->added_token_id[0]));
    vocab->n_added = 0;

    for (int i = 0; added_token_strs[i]; i++) {
        int id = vocab_lookup(vocab, added_token_strs[i]);
        if (id >= 0) {
            vocab->added_token[vocab->n_added] = strdup(added_token_strs[i]);
            vocab->added_token_id[vocab->n_added] = id;
            vocab->n_added++;
        }
    }

    ds3_print_info("[tokenizer] Loaded %d tokens, %llu merges, %d added tokens\n",
           vocab->n_vocab, (unsigned long long)n_merges, vocab->n_added);
    ds3_print_info("[tokenizer] BOS=%d EOS=%d IM_START=%d IM_END=%d THINK_END=%d\n",
           vocab->bos_id, vocab->eos_id, vocab->im_start_id, vocab->im_end_id,
           vocab->think_end_id);

    return true;
}

void ds3_vocab_free(ds3_vocab_t *vocab) {
    if (!vocab) return;
    if (vocab->token) {
        for (int i = 0; i < vocab->n_vocab; i++) {
            free(vocab->token[i]);
        }
        free(vocab->token);
    }
    if (vocab->added_token) {
        for (int i = 0; i < vocab->n_added; i++) {
            free(vocab->added_token[i]);
        }
        free(vocab->added_token);
    }
    free(vocab->added_token_id);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* ============================================================================
 * Public API
 * ============================================================================ */

int ds3_vocab_encode(const ds3_vocab_t *vocab, const char *text,
                     int *tokens, int max_tokens) {
    if (!vocab || !text || !tokens || max_tokens <= 0) return -1;

    ds3_token_vec_t out;
    token_vec_init(&out);

    uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        /* Try to match an added token at current position */
        int best_added = -1;
        uint64_t best_added_len = 0;
        for (int i = 0; i < vocab->n_added; i++) {
            size_t pat_len = strlen(vocab->added_token[i]);
            if (pat_len > 0 && pos + pat_len <= len &&
                memcmp(text + pos, vocab->added_token[i], pat_len) == 0) {
                if (pat_len > best_added_len) {
                    best_added_len = pat_len;
                    best_added = i;
                }
            }
        }

        if (best_added >= 0) {
            token_vec_push(&out, vocab->added_token_id[best_added]);
            pos += best_added_len;
            continue;
        }

        /* Find the nearest added token to determine the text segment */
        uint64_t next_added_pos = len;
        for (int i = 0; i < vocab->n_added; i++) {
            const char *found = strstr(text + pos, vocab->added_token[i]);
            if (found) {
                uint64_t fp = (uint64_t)(found - text);
                if (fp < next_added_pos) next_added_pos = fp;
            }
        }

        /* Tokenize the plain-text segment */
        bpe_tokenize_segment(vocab, text, pos, next_added_pos, &out);
        pos = next_added_pos;
    }

    int n = out.len < max_tokens ? out.len : max_tokens;
    memcpy(tokens, out.data, (size_t)n * sizeof(int));
    token_vec_free(&out);
    return n;
}

const char *ds3_vocab_decode(const ds3_vocab_t *vocab, int token_id) {
    if (!vocab || token_id < 0 || token_id >= vocab->n_vocab) return NULL;
    return vocab->token[token_id];
}

int ds3_vocab_decode_sequence(const ds3_vocab_t *vocab,
                              const int *tokens, int n_tokens,
                              char *buf, size_t buf_size) {
    if (!vocab || !tokens || n_tokens <= 0 || !buf || buf_size == 0) return -1;

    size_t written = 0;
    buf[0] = '\0';

    for (int i = 0; i < n_tokens; i++) {
        const char *raw = ds3_vocab_decode(vocab, tokens[i]);
        if (!raw) continue;

        uint64_t decoded_len = 0;
        char *decoded = byte_decode(raw, strlen(raw), &decoded_len);
        if (!decoded) continue;

        size_t remain = buf_size - written;
        if (decoded_len >= remain) {
            /* Truncate: write what fits and stop */
            if (remain > 1) {
                memcpy(buf + written, decoded, remain - 1);
                written += remain - 1;
            }
            buf[written] = '\0';
            free(decoded);
            return (int)written;
        }

        memcpy(buf + written, decoded, (size_t)decoded_len);
        written += (size_t)decoded_len;
        buf[written] = '\0';
        free(decoded);
    }

    return (int)written;
}

/* ============================================================================
 * Chat template
 *
 * Qwen3 format:
 *   <|im_start|>system\n{SYSTEM}<|im_end|>\n
 *   <|im_start|>user\n{USER}<|im_end|>\n
 *   <|im_start|>assistant\n
 *
 * System message is omitted if system is NULL or empty.
 * ============================================================================ */

int ds3_chat_format(const ds3_vocab_t *vocab,
                    const char *system,
                    const char *user,
                    int *tokens, int max_tokens) {
    if (!vocab || !user || !tokens || max_tokens <= 0) return -1;

    ds3_token_vec_t out;
    token_vec_init(&out);

    bool has_system = system && system[0];

    /* System message */
    if (has_system) {
        token_vec_push(&out, vocab->im_start_id);
        /* Tokenize "system\n{SYSTEM}" as text, then emit tokens */
        /* Strategy: push im_start, then tokenize "system\n" + system */
        bpe_tokenize(vocab, "system\n", &out);
        bpe_tokenize(vocab, system, &out);
        token_vec_push(&out, vocab->im_end_id);
        /* Add newline between messages */
        bpe_tokenize(vocab, "\n", &out);
    }

    /* User message */
    token_vec_push(&out, vocab->im_start_id);
    bpe_tokenize(vocab, "user\n", &out);
    bpe_tokenize(vocab, user, &out);
    token_vec_push(&out, vocab->im_end_id);
    bpe_tokenize(vocab, "\n", &out);

    /* Assistant prefix.
     *
     * The official Qwen3 chat template (from the GGUF metadata) adds only
     * "<|im_start|>assistant\n" as the generation prompt.  The model itself
     * emits "<think>\n...\n</think>\n\n" as the start of the response, so we
     * must NOT include <think> in the prefix — doing so changes the context
     * and causes the decode path to drift from the reference behaviour. */
    token_vec_push(&out, vocab->im_start_id);
    bpe_tokenize(vocab, "assistant\n", &out);

    int n = out.len < max_tokens ? out.len : max_tokens;
    memcpy(tokens, out.data, (size_t)n * sizeof(int));
    token_vec_free(&out);
    return n;
}
