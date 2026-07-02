/*
 * Verify C/Rust byte layout agreement for ds3 KV Cache.
 *
 * Compile:
 *   cc -I../src -o test_kv_cache_layout test_kv_cache_layout.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds3_kv_cache.h"

#define CHECK_SIZE(name, actual, expected) \
    do { \
        if ((actual) != (expected)) { \
            fprintf(stderr, "SIZE_MISMATCH: %s actual=%zu expected=%zu\n", \
                    (name), (size_t)(actual), (size_t)(expected)); \
            return 1; \
        } \
        printf("OK %s size=%zu align=%zu\n", (name), (size_t)(actual), (size_t)_Alignof(typeof(actual))); \
    } while (0)

#define CHECK_VAL(name, actual, expected) \
    do { \
        if ((actual) != (expected)) { \
            fprintf(stderr, "VALUE_MISMATCH: %s actual=%llu expected=%llu\n", \
                    (name), (unsigned long long)(actual), (unsigned long long)(expected)); \
            return 1; \
        } \
        printf("OK %s=%llu\n", (name), (unsigned long long)(actual)); \
    } while (0)

int main(void)
{
    printf("ds3 KV Cache layout verification\n");
    printf("================================\n");

    CHECK_SIZE("ds3_kv_region_header_t", sizeof(ds3_kv_region_header_t), 4096);
    CHECK_SIZE("ds3_kv_block_header_t", sizeof(ds3_kv_block_header_t), 128);

    CHECK_VAL("REGION_MAGIC", DS3_KVC_REGION_MAGIC, 0x323043564B335344ULL);
    CHECK_VAL("BLOCK_MAGIC", DS3_KVC_BLOCK_MAGIC, 0x000000004B43564BULL);
    CHECK_VAL("VERSION", DS3_KVC_VERSION, 2);

    CHECK_VAL("per_token_kv_bytes",
              (unsigned long long)ds3_kv_per_token_bytes(48, 4, 128, 2),
              98304ULL);

    CHECK_VAL("block_total_size",
              (unsigned long long)ds3_kv_block_total_size(16, 48, 4, 128, 2),
              1572992ULL);

    /* Verify field offsets match Rust expectations. */
    ds3_kv_region_header_t rh;
    CHECK_VAL("region_header.magic offset", (unsigned long long)((char *)&rh.magic - (char *)&rh), 0ULL);
    CHECK_VAL("region_header.version offset", (unsigned long long)((char *)&rh.version - (char *)&rh), 8ULL);
    CHECK_VAL("region_header.block_size offset", (unsigned long long)((char *)&rh.block_size - (char *)&rh), 12ULL);
    CHECK_VAL("region_header.num_blocks offset", (unsigned long long)((char *)&rh.num_blocks - (char *)&rh), 16ULL);

    ds3_kv_block_header_t bh;
    CHECK_VAL("block_header.magic offset", (unsigned long long)((char *)&bh.magic - (char *)&bh), 0ULL);
    CHECK_VAL("block_header.state offset", (unsigned long long)((char *)&bh.state - (char *)&bh), 8ULL);
    CHECK_VAL("block_header.ref_count offset", (unsigned long long)((char *)&bh.ref_count - (char *)&bh), 12ULL);
    CHECK_VAL("block_header.token_hash offset", (unsigned long long)((char *)&bh.token_hash - (char *)&bh), 16ULL);
    CHECK_VAL("block_header.block_id offset", (unsigned long long)((char *)&bh.block_id - (char *)&bh), 32ULL);
    CHECK_VAL("block_header.session_owner offset", (unsigned long long)((char *)&bh.session_owner - (char *)&bh), 40ULL);

    printf("================================\n");
    printf("All layout checks passed.\n");
    return 0;
}
