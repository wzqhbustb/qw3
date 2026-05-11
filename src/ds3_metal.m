#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#include "ds3_metal.h"
#include "ds3_reference.h"

/* ------------------------------------------------------------------ */
/* Internal state                                                     */
/* ------------------------------------------------------------------ */

static id<MTLDevice>              g_device   = nil;
static id<MTLCommandQueue>        g_queue    = nil;
static id<MTLLibrary>             g_library  = nil;
static id<MTLComputePipelineState> g_pipe_rms_norm       = nil;
static id<MTLComputePipelineState> g_pipe_rope           = nil;
static id<MTLComputePipelineState> g_pipe_matmul         = nil;
static id<MTLComputePipelineState> g_pipe_matmul_tiled   = nil;
static id<MTLComputePipelineState> g_pipe_matmul_vec     = nil;
static id<MTLComputePipelineState> g_pipe_matmul_vec_simd = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q4k     = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q4k_simd = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q4k_half = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q4k_batch     = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q4k_batch_simd = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q8_0    = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q8_0_simd = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q8_0_batch     = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q8_0_batch_simd = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q6k     = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q6k_simd = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q6k_batch     = nil;
static id<MTLComputePipelineState> g_pipe_matmul_q6k_batch_simd = nil;
static id<MTLComputePipelineState> g_pipe_kv_cache_write = nil;
static id<MTLComputePipelineState> g_pipe_kv_cache_write_batch = nil;
static id<MTLComputePipelineState> g_pipe_attention_decode = nil;
static id<MTLComputePipelineState> g_pipe_attention_decode_simd = nil;
static id<MTLComputePipelineState> g_pipe_attention_chunk = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_gate_up     = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_down      = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_silu_mul  = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_accumulate = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_gate_up_q4k   = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_gate_up_q6k   = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_gate_up_q8_0  = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_down_q4k    = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_down_q6k    = nil;
static id<MTLComputePipelineState> g_pipe_moe_expert_down_q8_0   = nil;
static id<MTLComputePipelineState> g_pipe_moe_router_hidden_q4k  = nil;
static id<MTLComputePipelineState> g_pipe_moe_output_q4k         = nil;
static id<MTLComputePipelineState> g_pipe_moe_output_q6k         = nil;
static id<MTLComputePipelineState> g_pipe_moe_router_topk_batch  = nil;
static id<MTLComputePipelineState> g_pipe_moe_router_topk_batch_tokens = nil;
static id<MTLComputePipelineState> g_pipe_vec_copy       = nil;
static id<MTLComputePipelineState> g_pipe_vec_zero       = nil;
static id<MTLComputePipelineState> g_pipe_vec_add        = nil;
static id<MTLComputePipelineState> g_pipe_vec_add3       = nil;
static id<MTLComputePipelineState> g_pipe_gather_rows    = nil;
static id<MTLComputePipelineState> g_pipe_scatter_add    = nil;
static id<MTLComputePipelineState> g_pipe_silu_mul       = nil;
static id<MTLCommandBuffer>        g_last_cb             = nil;

/* Batch-mode state: when active, public dispatch functions append to a single
 * shared command buffer instead of creating a new one per call. */
static bool                g_batch_mode = false;
static id<MTLCommandBuffer> g_batch_cb  = nil;
static id<MTLComputeCommandEncoder> g_batch_enc = nil;
static id<MTLBuffer>               g_rope_freq_buffer = nil;

/* Lightweight GPU profiling (enabled by DS3_METAL_PROFILE=1). */
static bool     g_profile = false;
static int      g_profile_cb_count = 0;
static double   g_profile_total_gpu_ms = 0.0;
static double   g_profile_total_cpu_ms = 0.0;

/* Return the shared batch encoder if batching is active, otherwise nil.
 * Lazily creates the encoder on the current batch command buffer. */
static id<MTLComputeCommandEncoder> active_encoder(void)
{
    if (!g_batch_mode || !g_batch_cb) return nil;
    if (!g_batch_enc) g_batch_enc = [g_batch_cb computeCommandEncoder];
    return g_batch_enc;
}

struct ds3_metal_buffer {
    __strong id<MTLBuffer> mtl;
    size_t bytes;
    size_t offset;      /* byte offset applied at kernel bind time for views */
    bool   no_copy;     /* true for mmap-backed buffers; writes are disallowed */
};

/* Return the actual Metal buffer handle and byte offset for a public buffer.
 * Views share the underlying MTLBuffer of their parent; the offset is applied
 * when the buffer is bound to a kernel encoder. */
static id<MTLBuffer> mtl_buf(const ds3_metal_buffer_t *b) { return b ? b->mtl : nil; }
static NSUInteger    mtl_off(const ds3_metal_buffer_t *b) { return (NSUInteger)(b ? b->offset : 0); }

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Choose a divisor of 32 that is <= n. This guarantees an integer number of
 * rows per SIMD group for the *_simd kernels. Must match the Metal helper of
 * the same name. */
static int lanes_per_row_for_simd(int n)
{
    if (n >= 32) return 32;
    if (n >= 16) return 16;
    if (n >=  8) return  8;
    if (n >=  4) return  4;
    if (n >=  2) return  2;
    return 1;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> lib, const char *name) {
    NSString *nsName = [NSString stringWithUTF8String:name];
    id<MTLFunction> fn = [lib newFunctionWithName:nsName];
    if (!fn) {
        fprintf(stderr, "ds3_metal: kernel '%s' not found in library\n", name);
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pipe = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pipe) {
        fprintf(stderr, "ds3_metal: failed to create pipeline for '%s': %s\n",
                name, err.localizedDescription.UTF8String);
        return nil;
    }
    return pipe;
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

int ds3_metal_init(void) {
    g_profile = (getenv("DS3_METAL_PROFILE") != NULL);

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        /* MTLCreateSystemDefaultDevice can return nil in headless/SSH
         * sessions even when a GPU is present.  Fall back to the first
         * device returned by MTLCopyAllDevices. */
        NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
        if (all.count > 0) {
            g_device = all[0];
        }
        if (!g_device) {
            fprintf(stderr, "ds3_metal: no Metal device found\n");
            return -1;
        }
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "ds3_metal: failed to create command queue\n");
        return -1;
    }

    /* Try compiled .metallib first, otherwise compile from source. */
    NSError *err = nil;
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"qwen3" ofType:@"metallib"];
    if (libPath) {
        g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    }
    if (!g_library) {
        /* Compile from individual .metal files in the metal/ directory.
         * Concatenate sources and compile into a single library. */
        NSString *rmsSrc = [NSString stringWithContentsOfFile:@"metal/rms_norm.metal"
                                                     encoding:NSUTF8StringEncoding
                                                        error:nil];
        NSString *ropeSrc = [NSString stringWithContentsOfFile:@"metal/rope.metal"
                                                      encoding:NSUTF8StringEncoding
                                                         error:nil];
        NSString *mmSrc = [NSString stringWithContentsOfFile:@"metal/matmul.metal"
                                                    encoding:NSUTF8StringEncoding
                                                       error:nil];
        NSString *mmVecSrc = [NSString stringWithContentsOfFile:@"metal/matmul_vec.metal"
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        NSString *mmQ4kSrc = [NSString stringWithContentsOfFile:@"metal/matmul_q4k.metal"
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        NSString *mmQ8_0Src = [NSString stringWithContentsOfFile:@"metal/matmul_q8_0.metal"
                                                        encoding:NSUTF8StringEncoding
                                                           error:nil];
        NSString *mmQ6kSrc = [NSString stringWithContentsOfFile:@"metal/matmul_q6k.metal"
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        NSString *attnSrc = [NSString stringWithContentsOfFile:@"metal/attention.metal"
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        NSString *moeSrc = [NSString stringWithContentsOfFile:@"metal/moe.metal"
                                                      encoding:NSUTF8StringEncoding
                                                         error:nil];
        NSString *elemSrc = [NSString stringWithContentsOfFile:@"metal/elementwise.metal"
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        if (!rmsSrc || !ropeSrc || !mmSrc || !mmVecSrc || !mmQ4kSrc || !mmQ8_0Src || !mmQ6kSrc || !attnSrc || !moeSrc || !elemSrc) {
            fprintf(stderr, "ds3_metal: failed to read .metal source files\n");
            return -1;
        }
        NSString *combined = rmsSrc;
        combined = [combined stringByAppendingString:ropeSrc];
        combined = [combined stringByAppendingString:mmSrc];
        combined = [combined stringByAppendingString:mmVecSrc];
        combined = [combined stringByAppendingString:mmQ4kSrc];
        combined = [combined stringByAppendingString:mmQ8_0Src];
        combined = [combined stringByAppendingString:mmQ6kSrc];
        combined = [combined stringByAppendingString:attnSrc];
        combined = [combined stringByAppendingString:moeSrc];
        combined = [combined stringByAppendingString:elemSrc];
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        g_library = [g_device newLibraryWithSource:combined options:opts error:&err];
        if (!g_library) {
            fprintf(stderr, "ds3_metal: failed to compile combined library: %s\n",
                    err.localizedDescription.UTF8String);
            return -1;
        }
    }

    g_pipe_rms_norm        = make_pipeline(g_library, "rms_norm_f32");
    g_pipe_rope            = make_pipeline(g_library, "rope_f32");
    g_pipe_matmul          = make_pipeline(g_library, "matmul_f32");
    g_pipe_matmul_tiled    = make_pipeline(g_library, "matmul_f32_tiled");
    g_pipe_matmul_vec      = make_pipeline(g_library, "matmul_vec_f32");
    g_pipe_matmul_vec_simd = make_pipeline(g_library, "matmul_vec_f32_simd");
    g_pipe_matmul_q4k      = make_pipeline(g_library, "matmul_vec_q4k");
    g_pipe_matmul_q4k_simd = make_pipeline(g_library, "matmul_vec_q4k_simd");
    g_pipe_matmul_q4k_half = make_pipeline(g_library, "matmul_vec_q4k_half");
    g_pipe_matmul_q4k_batch     = make_pipeline(g_library, "matmul_q4k");
    g_pipe_matmul_q4k_batch_simd = make_pipeline(g_library, "matmul_q4k_batch_simd");
    g_pipe_matmul_q8_0     = make_pipeline(g_library, "matmul_vec_q8_0");
    g_pipe_matmul_q8_0_simd = make_pipeline(g_library, "matmul_vec_q8_0_simd");
    g_pipe_matmul_q8_0_batch     = make_pipeline(g_library, "matmul_q8_0");
    g_pipe_matmul_q8_0_batch_simd = make_pipeline(g_library, "matmul_q8_0_batch_simd");
    g_pipe_matmul_q6k      = make_pipeline(g_library, "matmul_vec_q6k");
    g_pipe_matmul_q6k_simd = make_pipeline(g_library, "matmul_vec_q6k_simd");
    g_pipe_matmul_q6k_batch     = make_pipeline(g_library, "matmul_q6k");
    g_pipe_matmul_q6k_batch_simd = make_pipeline(g_library, "matmul_q6k_batch_simd");
    g_pipe_kv_cache_write    = make_pipeline(g_library, "kv_cache_write");
    g_pipe_kv_cache_write_batch = make_pipeline(g_library, "kv_cache_write_batch");
    g_pipe_attention_decode  = make_pipeline(g_library, "attention_decode_gqa");
    g_pipe_attention_decode_simd = make_pipeline(g_library, "attention_decode_gqa_simd");
    g_pipe_attention_chunk   = make_pipeline(g_library, "attention_chunk_gqa");
    g_pipe_moe_expert_gate_up     = make_pipeline(g_library, "moe_expert_gate_up_f32");
    g_pipe_moe_expert_down      = make_pipeline(g_library, "moe_expert_down_f32");
    g_pipe_moe_expert_silu_mul  = make_pipeline(g_library, "moe_expert_silu_mul_f32");
    g_pipe_moe_expert_accumulate = make_pipeline(g_library, "moe_expert_accumulate_f32");
    g_pipe_moe_expert_gate_up_q4k  = make_pipeline(g_library, "moe_expert_gate_up_q4k_f32");
    g_pipe_moe_expert_gate_up_q6k  = make_pipeline(g_library, "moe_expert_gate_up_q6k_f32");
    g_pipe_moe_expert_gate_up_q8_0 = make_pipeline(g_library, "moe_expert_gate_up_q8_0_f32");
    g_pipe_moe_expert_down_q4k   = make_pipeline(g_library, "moe_expert_down_q4k_f32");
    g_pipe_moe_expert_down_q6k   = make_pipeline(g_library, "moe_expert_down_q6k_f32");
    g_pipe_moe_expert_down_q8_0  = make_pipeline(g_library, "moe_expert_down_q8_0_f32");
    g_pipe_moe_router_hidden_q4k = make_pipeline(g_library, "moe_router_hidden_q4k_f32");
    g_pipe_moe_output_q4k        = make_pipeline(g_library, "moe_output_q4k_f32");
    g_pipe_moe_output_q6k        = make_pipeline(g_library, "moe_output_q6k_f32");
    g_pipe_moe_router_topk_batch = make_pipeline(g_library, "moe_router_topk_batch_f32");
    g_pipe_moe_router_topk_batch_tokens = make_pipeline(g_library, "moe_router_topk_batch_tokens_f32");
    g_pipe_vec_copy  = make_pipeline(g_library, "vec_copy_f32");
    g_pipe_vec_zero  = make_pipeline(g_library, "vec_zero_f32");
    g_pipe_vec_add   = make_pipeline(g_library, "vec_add_f32");
    g_pipe_vec_add3  = make_pipeline(g_library, "vec_add3_f32");
    g_pipe_gather_rows = make_pipeline(g_library, "gather_rows_f32");
    g_pipe_scatter_add = make_pipeline(g_library, "scatter_add_weighted_f32");
    g_pipe_silu_mul    = make_pipeline(g_library, "moe_expert_silu_mul_f32");
    if (!g_pipe_rms_norm || !g_pipe_rope || !g_pipe_matmul || !g_pipe_matmul_tiled ||
        !g_pipe_matmul_vec || !g_pipe_matmul_vec_simd ||
        !g_pipe_matmul_q4k || !g_pipe_matmul_q4k_simd || !g_pipe_matmul_q4k_half ||
        !g_pipe_matmul_q4k_batch || !g_pipe_matmul_q4k_batch_simd ||
        !g_pipe_matmul_q8_0 || !g_pipe_matmul_q8_0_simd ||
        !g_pipe_matmul_q8_0_batch || !g_pipe_matmul_q8_0_batch_simd ||
        !g_pipe_matmul_q6k || !g_pipe_matmul_q6k_simd ||
        !g_pipe_matmul_q6k_batch || !g_pipe_matmul_q6k_batch_simd ||
        !g_pipe_kv_cache_write || !g_pipe_kv_cache_write_batch ||
        !g_pipe_attention_decode || !g_pipe_attention_decode_simd || !g_pipe_attention_chunk ||
        !g_pipe_moe_expert_gate_up || !g_pipe_moe_expert_down ||
        !g_pipe_moe_expert_silu_mul || !g_pipe_moe_expert_accumulate ||
        !g_pipe_moe_expert_gate_up_q4k || !g_pipe_moe_expert_gate_up_q6k ||
        !g_pipe_moe_expert_gate_up_q8_0 ||
        !g_pipe_moe_expert_down_q4k || !g_pipe_moe_expert_down_q6k ||
        !g_pipe_moe_expert_down_q8_0 ||
        !g_pipe_moe_router_hidden_q4k || !g_pipe_moe_output_q4k || !g_pipe_moe_output_q6k ||
        !g_pipe_moe_router_topk_batch || !g_pipe_moe_router_topk_batch_tokens ||
        !g_pipe_vec_copy || !g_pipe_vec_zero || !g_pipe_vec_add || !g_pipe_vec_add3 ||
        !g_pipe_gather_rows || !g_pipe_scatter_add || !g_pipe_silu_mul) return -1;

    /* Precompute RoPE frequency table for head_dim up to 128.
     * freq[i] = theta_base ^ (-2i / head_dim) for i = 0..63.
     * We allocate for 64 floats (max pairs for head_dim=128) and
     * fill it lazily on first rope call with the actual theta_base.
     * For now create an empty buffer; it will be populated by
     * ds3_metal_rope with the caller's theta_base. */
    g_rope_freq_buffer = [g_device newBufferWithLength:64 * sizeof(float)
                                               options:MTLResourceStorageModeShared];

    const char *name = g_device.name ? [g_device.name UTF8String] : "unknown";
    ds3_log_info("ds3_metal: initialized on %s\n", name);
    return 0;
}

void ds3_metal_shutdown(void) {
    g_pipe_rms_norm         = nil;
    g_pipe_rope             = nil;
    g_pipe_matmul           = nil;
    g_pipe_matmul_tiled     = nil;
    g_pipe_matmul_vec       = nil;
    g_pipe_matmul_vec_simd  = nil;
    g_pipe_matmul_q4k       = nil;
    g_pipe_matmul_q4k_simd  = nil;
    g_pipe_matmul_q4k_half  = nil;
    g_pipe_matmul_q4k_batch     = nil;
    g_pipe_matmul_q4k_batch_simd = nil;
    g_pipe_matmul_q8_0             = nil;
    g_pipe_matmul_q8_0_simd        = nil;
    g_pipe_matmul_q8_0_batch     = nil;
    g_pipe_matmul_q8_0_batch_simd = nil;
    g_pipe_matmul_q6k              = nil;
    g_pipe_matmul_q6k_simd         = nil;
    g_pipe_matmul_q6k_batch     = nil;
    g_pipe_matmul_q6k_batch_simd = nil;
    g_pipe_kv_cache_write          = nil;
    g_pipe_kv_cache_write_batch    = nil;
    g_pipe_attention_decode        = nil;
    g_pipe_attention_decode_simd   = nil;
    g_pipe_attention_chunk         = nil;
    g_pipe_moe_expert_gate_up      = nil;
    g_pipe_moe_expert_down         = nil;
    g_pipe_moe_expert_silu_mul     = nil;
    g_pipe_moe_expert_accumulate   = nil;
    g_pipe_moe_expert_gate_up_q4k  = nil;
    g_pipe_moe_expert_gate_up_q6k  = nil;
    g_pipe_moe_expert_gate_up_q8_0 = nil;
    g_pipe_moe_expert_down_q4k     = nil;
    g_pipe_moe_expert_down_q6k     = nil;
    g_pipe_moe_expert_down_q8_0    = nil;
    g_pipe_moe_router_hidden_q4k   = nil;
    g_pipe_moe_output_q4k          = nil;
    g_pipe_moe_output_q6k          = nil;
    g_pipe_moe_router_topk_batch   = nil;
    g_pipe_moe_router_topk_batch_tokens = nil;
    g_pipe_vec_copy                = nil;
    g_pipe_vec_zero                = nil;
    g_pipe_vec_add                 = nil;
    g_pipe_vec_add3                = nil;
    g_pipe_gather_rows             = nil;
    g_pipe_scatter_add             = nil;
    g_pipe_silu_mul                = nil;
    g_library                      = nil;
    g_last_cb          = nil;
    g_rope_freq_buffer = nil;
    g_queue            = nil;
    g_device           = nil;

    /* Clear any leftover batch state (e.g. error paths). */
    g_batch_mode = false;
    g_batch_cb   = nil;
    g_batch_enc  = nil;
}

/* ------------------------------------------------------------------ */
/* Buffers                                                            */
/* ------------------------------------------------------------------ */

ds3_metal_buffer_t * ds3_metal_buffer_alloc(size_t bytes) {
    if (bytes == 0) return NULL;
    id<MTLBuffer> mtl = [g_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!mtl) return NULL;
    ds3_metal_buffer_t *buf = calloc(1, sizeof(ds3_metal_buffer_t));
    if (!buf) return NULL;
    buf->mtl   = mtl;
    buf->bytes = bytes;
    buf->offset = 0;
    buf->no_copy = false;
    return buf;
}

ds3_metal_buffer_t * ds3_metal_buffer_from_mmap(const void *base, size_t bytes) {
    if (!base || bytes == 0) return NULL;

    long page_size_l = sysconf(_SC_PAGESIZE);
    if (page_size_l <= 0) page_size_l = 4096;
    size_t page_size = (size_t)page_size_l;

    if (((uintptr_t)base) & (page_size - 1)) {
        fprintf(stderr, "ds3_metal: mmap base %p is not page-aligned\n", base);
        return NULL;
    }

    id<MTLBuffer> mtl = [g_device newBufferWithBytesNoCopy:(void *)base
                                                     length:bytes
                                                    options:MTLResourceStorageModeShared
                                                 deallocator:^(void * _Nonnull p, NSUInteger len) {
                                                     /* mmap is owned by ds3_gguf; no-op here. */
                                                     (void)p; (void)len;
                                                 }];
    if (!mtl) return NULL;
    ds3_metal_buffer_t *buf = calloc(1, sizeof(ds3_metal_buffer_t));
    if (!buf) { mtl = nil; return NULL; }
    buf->mtl     = mtl;
    buf->bytes   = bytes;
    buf->offset  = 0;
    buf->no_copy = true;
    return buf;
}

ds3_metal_buffer_t * ds3_metal_buffer_view(const ds3_metal_buffer_t *base,
                                           size_t offset, size_t bytes) {
    if (!base || !base->mtl || offset > base->bytes ||
        bytes == 0 || offset + bytes > base->bytes) {
        return NULL;
    }
    ds3_metal_buffer_t *v = calloc(1, sizeof(ds3_metal_buffer_t));
    if (!v) return NULL;
    v->mtl     = base->mtl;            /* shares (and retains) the underlying MTLBuffer */
    v->bytes   = bytes;
    v->offset  = base->offset + offset;
    v->no_copy = base->no_copy;
    return v;
}

void ds3_metal_buffer_free(ds3_metal_buffer_t *buf) {
    if (!buf) return;
    buf->mtl = nil;
    free(buf);
}

int ds3_metal_buffer_write(ds3_metal_buffer_t *buf, size_t offset,
                           const void *src, size_t bytes) {
    if (!buf || !src || offset + bytes > buf->bytes || buf->no_copy) return -1;
    memcpy((char *)buf->mtl.contents + buf->offset + offset, src, bytes);
    return 0;
}

int ds3_metal_buffer_read(ds3_metal_buffer_t *buf, size_t offset,
                          void *dst, size_t bytes) {
    if (!buf || !dst || offset + bytes > buf->bytes) return -1;
    memcpy(dst, (char *)buf->mtl.contents + buf->offset + offset, bytes);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Kernel dispatch                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t n;
    uint32_t n_rows;
    float    eps;
    uint64_t src_stride;
    uint64_t dst_stride;
} ds3_rms_norm_args_host;

typedef struct {
    uint32_t n_heads;
    uint32_t head_dim;
    uint32_t n_rows;
    float    theta_base;
    uint64_t src_stride;
    uint64_t dst_stride;
} ds3_rope_args_host;

typedef struct {
    uint32_t n;
} ds3_moe_silu_mul_args_host;

/* Populate the precomputed RoPE frequency table for the given theta_base and
 * head_dim.  The table has 64 entries (enough for head_dim <= 128). */
static int ensure_rope_freq_table(float theta_base, uint32_t head_dim)
{
    if (!g_rope_freq_buffer) return -1;
    if (head_dim % 2 != 0) return -1;
    const uint32_t n_pairs = head_dim / 2;
    if (n_pairs == 0 || n_pairs > 64) return -1;
    float *freq = (float *)g_rope_freq_buffer.contents;
    for (uint32_t i = 0; i < n_pairs; i++) {
        freq[i] = powf(theta_base, -2.0f * (float)i / (float)head_dim);
    }
    return 0;
}

int ds3_metal_rms_norm(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *weight,
    ds3_metal_buffer_t       *out,
    uint32_t n, uint32_t n_rows, float eps)
{
    if (!x || !weight || !out) return -1;

    ds3_rms_norm_args_host args = {
        .n          = n,
        .n_rows     = n_rows,
        .eps        = eps,
        .src_stride = n * sizeof(float),
        .dst_stride = n * sizeof(float),
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_rms_norm];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(weight) offset:mtl_off(weight) atIndex:1];
    [enc setBuffer:mtl_buf(out) offset:mtl_off(out) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    NSUInteger tgSize = g_pipe_rms_norm.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > n) tgSize = n;          /* avoid threads that do no work */
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake(n_rows, 1, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);

    /* Allocate enough threadgroup memory for 32 floats regardless of
     * actual simdgroup count, so the kernel can safely index shmem[tiisg]
     * (tiisg is always in 0..31 on Apple GPUs). */
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_rope(
    const ds3_metal_buffer_t *src,
    ds3_metal_buffer_t       *dst,
    const ds3_metal_buffer_t *positions,
    uint32_t n_heads, uint32_t head_dim, uint32_t n_rows,
    float theta_base)
{
    if (!src || !dst || !positions) return -1;

    ds3_rope_args_host args = {
        .n_heads    = n_heads,
        .head_dim   = head_dim,
        .n_rows     = n_rows,
        .theta_base = theta_base,
        .src_stride = n_heads * head_dim * sizeof(float),
        .dst_stride = n_heads * head_dim * sizeof(float),
    };

    if (ensure_rope_freq_table(theta_base, head_dim) != 0) return -1;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_rope];
    [enc setBuffer:mtl_buf(src) offset:mtl_off(src) atIndex:0];
    [enc setBuffer:mtl_buf(dst) offset:mtl_off(dst) atIndex:1];
    [enc setBytes:&args length:sizeof(args) atIndex:2];
    [enc setBuffer:mtl_buf(positions) offset:mtl_off(positions) atIndex:3];
    [enc setBuffer:g_rope_freq_buffer offset:0 atIndex:4];

    NSUInteger tgSize = g_pipe_rope.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > head_dim / 2) tgSize = head_dim / 2;
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake(n_rows, n_heads, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_matmul(
    const ds3_metal_buffer_t *A,
    const ds3_metal_buffer_t *B,
    ds3_metal_buffer_t       *C,
    uint32_t M, uint32_t N, uint32_t K)
{
    if (!A || !B || !C) return -1;

    typedef struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint64_t lda;
        uint64_t ldb;
        uint64_t ldc;
    } ds3_matmul_args_host;

    ds3_matmul_args_host args = {
        .M   = M,
        .N   = N,
        .K   = K,
        .lda = K,
        .ldb = N,
        .ldc = N,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setBuffer:mtl_buf(A) offset:mtl_off(A) atIndex:0];
    [enc setBuffer:mtl_buf(B) offset:mtl_off(B) atIndex:1];
    [enc setBuffer:mtl_buf(C) offset:mtl_off(C) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    if (M > 1) {
        /* Tiled path for prefill-style matrix-matrix multiply. */
        [enc setComputePipelineState:g_pipe_matmul_tiled];
        const NSUInteger TG = 16;
        MTLSize threads = MTLSizeMake(N, M, 1);
        MTLSize tgrp = MTLSizeMake(TG, TG, 1);
        [enc dispatchThreads:threads threadsPerThreadgroup:tgrp];
    } else {
        /* Decode-style vector-matrix multiply. */
        [enc setComputePipelineState:g_pipe_matmul];
        const NSUInteger TG_X = 8;
        const NSUInteger TG_Y = 8;
        MTLSize grid = MTLSizeMake((M + TG_X - 1) / TG_X, (N + TG_Y - 1) / TG_Y, 1);
        MTLSize tgrp = MTLSizeMake(TG_X, TG_Y, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];
    }

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* Host-side argument block shared by all vec-matmul kernels.
 * row_stride is always in bytes, regardless of data type. */
typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_stride;
} ds3_matmul_vec_args_host;

int ds3_metal_vec_matmul_f32(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride)
{
    if (!x || !W || !y) return -1;
    uint64_t min_row_stride = (uint64_t)in_dim * sizeof(float);
    if (row_stride < min_row_stride) {
        fprintf(stderr, "ds3_metal: row_stride=%llu must be >= %llu bytes for in_dim=%u\n",
                (unsigned long long)row_stride, (unsigned long long)min_row_stride, in_dim);
        return -1;
    }

    ds3_matmul_vec_args_host args = {
        .in_dim     = in_dim,
        .out_dim    = out_dim,
        .row_stride = row_stride,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_matmul_vec];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    NSUInteger tgSize = g_pipe_matmul_vec.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > out_dim) tgSize = out_dim;
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake((out_dim + tgSize - 1) / tgSize, 1, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* Common dispatch helper for the SIMD-group reduction variants.
 * block_size == 0 means FP32 (no blocks); otherwise n_blocks = in_dim / block_size. */
static int dispatch_vec_matmul_simd(
    id<MTLComputePipelineState> pipe,
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride,
    uint64_t weight_offset,
    int block_size)
{
    typedef struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint64_t row_stride;
        uint64_t weight_offset;
    } simd_args_t;
    simd_args_t args = {
        .in_dim        = in_dim,
        .out_dim       = out_dim,
        .row_stride    = row_stride,
        .weight_offset = weight_offset,
    };

    int n_blocks      = (block_size > 0) ? (int)(in_dim / block_size) : (int)in_dim;
    int lanes_per_row = lanes_per_row_for_simd(n_blocks);
    int rows_per_simd = 32 / lanes_per_row;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:weight_offset + mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    const NSUInteger TG_SIZE = 32;
    MTLSize grid = MTLSizeMake((out_dim + rows_per_simd - 1) / rows_per_simd, 1, 1);
    MTLSize tgrp = MTLSizeMake(TG_SIZE, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_vec_matmul_f32_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride)
{
    if (!x || !W || !y) return -1;
    uint64_t min_row_stride = (uint64_t)in_dim * sizeof(float);
    if (row_stride < min_row_stride) {
        fprintf(stderr, "ds3_metal: row_stride=%llu must be >= %llu bytes for in_dim=%u\n",
                (unsigned long long)row_stride, (unsigned long long)min_row_stride, in_dim);
        return -1;
    }
    return dispatch_vec_matmul_simd(g_pipe_matmul_vec_simd, x, W, y,
                                    in_dim, out_dim, row_stride, 0, 0);
}

int ds3_metal_vec_matmul_q8_0_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride)
{
    if (!x || !W || !y) return -1;
    if (in_dim % 32 != 0) {
        fprintf(stderr, "ds3_metal: Q8_0 simd in_dim=%u must be a multiple of 32\n", in_dim);
        return -1;
    }

    typedef struct { uint32_t in_dim; uint32_t out_dim; uint64_t row_stride; uint64_t weight_offset; } q8_0_args_t;
    q8_0_args_t args = {
        .in_dim        = in_dim,
        .out_dim       = out_dim,
        .row_stride    = row_stride,
        .weight_offset = 0,
    };

    int n_blocks      = (int)(in_dim / 32);
    int lanes_per_row = lanes_per_row_for_simd(n_blocks);
    int rows_per_simd = 32 / lanes_per_row;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_matmul_q8_0_simd];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    const NSUInteger TG_SIZE = 32;
    MTLSize grid = MTLSizeMake((out_dim + rows_per_simd - 1) / rows_per_simd, 1, 1);
    MTLSize tgrp = MTLSizeMake(TG_SIZE, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_vec_matmul_q4k_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride)
{
    if (!x || !W || !y) return -1;
    if (in_dim % 256 != 0) {
        fprintf(stderr, "ds3_metal: Q4_K simd in_dim=%u must be a multiple of 256\n", in_dim);
        return -1;
    }

    typedef struct { uint32_t in_dim; uint32_t out_dim; uint64_t row_stride; uint64_t weight_offset; } q4k_args_t;
    q4k_args_t args = {
        .in_dim     = in_dim,
        .out_dim    = out_dim,
        .row_stride = row_stride,
        .weight_offset = 0,
    };

    int n_blocks      = (int)(in_dim / 256);
    int lanes_per_row = lanes_per_row_for_simd(n_blocks);
    int rows_per_simd = 32 / lanes_per_row;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_matmul_q4k_simd];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    const NSUInteger TG_SIZE = 32;
    MTLSize grid = MTLSizeMake((out_dim + rows_per_simd - 1) / rows_per_simd, 1, 1);
    MTLSize tgrp = MTLSizeMake(TG_SIZE, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_vec_matmul_q6k_simd(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride)
{
    if (!x || !W || !y) return -1;
    if (in_dim % 256 != 0) {
        fprintf(stderr, "ds3_metal: Q6_K simd in_dim=%u must be a multiple of 256\n", in_dim);
        return -1;
    }

    typedef struct { uint32_t in_dim; uint32_t out_dim; uint64_t row_stride; uint64_t weight_offset; } q6k_args_t;
    q6k_args_t args = {
        .in_dim     = in_dim,
        .out_dim    = out_dim,
        .row_stride = row_stride,
        .weight_offset = 0,
    };

    int n_blocks      = (int)(in_dim / 256);
    int lanes_per_row = lanes_per_row_for_simd(n_blocks);
    int rows_per_simd = 32 / lanes_per_row;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_matmul_q6k_simd];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    const NSUInteger TG_SIZE = 32;
    MTLSize grid = MTLSizeMake((out_dim + rows_per_simd - 1) / rows_per_simd, 1, 1);
    MTLSize tgrp = MTLSizeMake(TG_SIZE, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Elementwise kernels (copy / zero / add)                            */
/* ------------------------------------------------------------------ */

static int dispatch_elementwise_1d(id<MTLComputePipelineState> pipe,
                                   uint32_t n,
                                   const ds3_metal_buffer_t *b0,
                                   const ds3_metal_buffer_t *b1,
                                   ds3_metal_buffer_t       *b2,
                                   const void *bytes0, size_t bytes0_len,
                                   int arg_count)
{
    if (n == 0) return 0;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:pipe];
    if (b0) [enc setBuffer:mtl_buf(b0) offset:mtl_off(b0) atIndex:0];
    if (b1) [enc setBuffer:mtl_buf(b1) offset:mtl_off(b1) atIndex:1];
    if (b2) [enc setBuffer:mtl_buf(b2) offset:mtl_off(b2) atIndex:2];
    if (bytes0 && bytes0_len > 0) [enc setBytes:bytes0 length:bytes0_len atIndex:arg_count - 1];

    NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > n) tgSize = n;
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake((n + tgSize - 1) / tgSize, 1, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_buffer_copy_f32(const ds3_metal_buffer_t *src,
                              ds3_metal_buffer_t       *dst,
                              size_t bytes)
{
    if (!src || !dst || bytes == 0) return -1;
    if (bytes % sizeof(float) != 0) {
        fprintf(stderr, "ds3_metal: buffer_copy_f32 bytes=%zu must be a multiple of %zu\n",
                bytes, sizeof(float));
        return -1;
    }
    uint32_t n = (uint32_t)(bytes / sizeof(float));
    if (n == 0) return 0;
    return dispatch_elementwise_1d(g_pipe_vec_copy, n, src, dst, NULL, &n, sizeof(n), 3);
}

int ds3_metal_buffer_zero_f32(ds3_metal_buffer_t *buf, size_t bytes)
{
    if (!buf || bytes == 0) return -1;
    if (bytes % sizeof(float) != 0) {
        fprintf(stderr, "ds3_metal: buffer_zero_f32 bytes=%zu must be a multiple of %zu\n",
                bytes, sizeof(float));
        return -1;
    }
    uint32_t n = (uint32_t)(bytes / sizeof(float));
    if (n == 0) return 0;
    return dispatch_elementwise_1d(g_pipe_vec_zero, n, buf, NULL, NULL, &n, sizeof(n), 2);
}

int ds3_metal_vec_add_inplace(ds3_metal_buffer_t       *acc,
                              const ds3_metal_buffer_t *addend,
                              uint32_t n)
{
    if (!acc || !addend) return -1;
    return dispatch_elementwise_1d(g_pipe_vec_add, n, addend, acc, NULL, &n, sizeof(n), 3);
}

int ds3_metal_vec_add(const ds3_metal_buffer_t *a,
                      const ds3_metal_buffer_t *b,
                      ds3_metal_buffer_t       *c,
                      uint32_t n)
{
    if (!a || !b || !c) return -1;
    return dispatch_elementwise_1d(g_pipe_vec_add3, n, a, b, c, &n, sizeof(n), 4);
}

int ds3_metal_gather_rows_f32(const ds3_metal_buffer_t *src,
                              const ds3_metal_buffer_t *ids,
                              ds3_metal_buffer_t       *dst,
                              uint32_t n_cols, uint32_t count, uint32_t offset)
{
    if (!src || !ids || !dst || n_cols == 0 || count == 0) return -1;
    typedef struct { uint32_t n_cols; uint32_t count; uint32_t offset; } args_t;
    args_t args = { .n_cols = n_cols, .count = count, .offset = offset };
    return dispatch_elementwise_1d(g_pipe_gather_rows, count * n_cols,
                                   src, ids, dst, &args, sizeof(args), 4);
}

int ds3_metal_scatter_add_weighted_f32(const ds3_metal_buffer_t *src,
                                       const ds3_metal_buffer_t *scores,
                                       const ds3_metal_buffer_t *ids,
                                       ds3_metal_buffer_t       *dst,
                                       uint32_t n_cols, uint32_t count, uint32_t offset)
{
    if (!src || !scores || !ids || !dst || n_cols == 0 || count == 0) return -1;
    typedef struct { uint32_t n_cols; uint32_t count; uint32_t offset; } args_t;
    args_t args = { .n_cols = n_cols, .count = count, .offset = offset };
    uint32_t n = count * n_cols;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_scatter_add];
    [enc setBuffer:mtl_buf(src)    offset:mtl_off(src)    atIndex:0];
    [enc setBuffer:mtl_buf(scores) offset:mtl_off(scores) atIndex:1];
    [enc setBuffer:mtl_buf(ids)    offset:mtl_off(ids)    atIndex:2];
    [enc setBuffer:mtl_buf(dst)    offset:mtl_off(dst)    atIndex:3];
    [enc setBytes:&args length:sizeof(args) atIndex:4];

    NSUInteger tgSize = g_pipe_scatter_add.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > n) tgSize = n;
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake((n + tgSize - 1) / tgSize, 1, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_silu_mul_f32(const ds3_metal_buffer_t *gate,
                           const ds3_metal_buffer_t *up,
                           ds3_metal_buffer_t       *out,
                           uint32_t n)
{
    if (!gate || !up || !out) return -1;
    ds3_moe_silu_mul_args_host args = { .n = n };
    return dispatch_elementwise_1d(g_pipe_silu_mul, n, gate, up, out,
                                   &args, sizeof(args), 4);
}

int ds3_metal_vec_matmul_q4k_half(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride)
{
    if (!x || !W || !y) return -1;
    if (in_dim % 256 != 0) {
        fprintf(stderr, "ds3_metal: Q4_K half in_dim=%u must be a multiple of 256\n", in_dim);
        return -1;
    }

    typedef struct { uint32_t in_dim; uint32_t out_dim; uint64_t row_stride; uint64_t weight_offset; } q4k_half_args_t;
    q4k_half_args_t args = {
        .in_dim     = in_dim,
        .out_dim    = out_dim,
        .row_stride = row_stride,
        .weight_offset = 0,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:g_pipe_matmul_q4k_half];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    NSUInteger tgSize = g_pipe_matmul_q4k_half.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > out_dim) tgSize = out_dim;
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake((out_dim + tgSize - 1) / tgSize, 1, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* Host-side argument block shared by both attention kernels. */
typedef struct {
    uint32_t seq_pos;
    uint32_t max_seq_len;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
} ds3_attention_args_host;

static int validate_attention_args(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    if (!q || !k || !v || !k_cache || !v_cache || !output) return -1;
    if (n_q_heads == 0 || n_kv_heads == 0 || head_dim == 0) {
        fprintf(stderr, "ds3_metal: n_q_heads/n_kv_heads/head_dim must be > 0\n");
        return -1;
    }
    if (n_q_heads % n_kv_heads != 0) {
        fprintf(stderr, "ds3_metal: n_q_heads=%u must be divisible by n_kv_heads=%u\n",
                n_q_heads, n_kv_heads);
        return -1;
    }
    if (head_dim > 128) {
        fprintf(stderr, "ds3_metal: head_dim=%u exceeds kernel MAX_HEAD_DIM=128\n", head_dim);
        return -1;
    }
    if (seq_pos >= max_seq_len) {
        fprintf(stderr, "ds3_metal: seq_pos=%u must be < max_seq_len=%u\n",
                seq_pos, max_seq_len);
        return -1;
    }

    const uint32_t q_dim  = n_q_heads * head_dim;
    const uint32_t kv_dim = n_kv_heads * head_dim;
    const size_t   q_bytes     = (size_t)q_dim * sizeof(float);
    const size_t   kv_bytes    = (size_t)kv_dim * sizeof(float);
    const size_t   cache_bytes = (size_t)max_seq_len * kv_dim * sizeof(uint16_t);

    if (q->bytes < q_bytes || k->bytes < kv_bytes || v->bytes < kv_bytes ||
        output->bytes < q_bytes ||
        k_cache->bytes < cache_bytes || v_cache->bytes < cache_bytes) {
        fprintf(stderr, "ds3_metal: attention buffer too small\n");
        return -1;
    }
    return 0;
}

/* Validate arguments for the RoPE-integrated attention variants.
 * Reuses validate_attention_args() and adds the head_dim evenness check
 * required by RoPE. */
static int validate_attention_decode_rope_args(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    if (validate_attention_args(q, k, v, k_cache, v_cache, output,
                                seq_pos, max_seq_len,
                                n_q_heads, n_kv_heads, head_dim) != 0) return -1;
    if (head_dim % 2 != 0) {
        fprintf(stderr, "ds3_metal: head_dim=%u must be even for RoPE\n", head_dim);
        return -1;
    }
    return 0;
}

/* Append a RoPE dispatch for one buffer to an existing compute encoder.
 * The buffer is modified in-place. */
static void dispatch_rope_on_buffer(
    id<MTLComputeCommandEncoder> enc,
    ds3_metal_buffer_t *buf,
    uint32_t n_heads,
    uint32_t head_dim,
    float theta_base,
    int32_t seq_pos,
    MTLSize tgrp)
{
    ds3_rope_args_host rope_args = {
        .n_heads    = n_heads,
        .head_dim   = head_dim,
        .n_rows     = 1,
        .theta_base = theta_base,
        .src_stride = (uint64_t)n_heads * head_dim * sizeof(float),
        .dst_stride = (uint64_t)n_heads * head_dim * sizeof(float),
    };
    [enc setComputePipelineState:g_pipe_rope];
    [enc setBuffer:mtl_buf(buf) offset:mtl_off(buf) atIndex:0];
    [enc setBuffer:mtl_buf(buf) offset:mtl_off(buf) atIndex:1];
    [enc setBytes:&rope_args length:sizeof(rope_args) atIndex:2];
    [enc setBytes:&seq_pos length:sizeof(seq_pos) atIndex:3];
    [enc setBuffer:g_rope_freq_buffer offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, n_heads, 1)
        threadsPerThreadgroup:tgrp];
}

/* Dispatch attention decode in one command buffer:
 *   1. rope_f32 on q   : q -> q (in-place)
 *   2. rope_f32 on k   : k -> k (in-place)
 *   3. kv_cache_write  : k, v -> k_cache[seq_pos], v_cache[seq_pos]
 *   4. attention_*     : q, cache -> output
 * All dispatches are ordered within the same command buffer.
 */
static int dispatch_attention_decode_rope(
    id<MTLComputePipelineState> attn_pipe,
    ds3_metal_buffer_t       *q,
    ds3_metal_buffer_t       *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    const ds3_attention_args_host *args,
    float theta_base,
    MTLSize kv_grid,
    MTLSize kv_tgrp,
    MTLSize attn_grid,
    MTLSize attn_tgrp)
{
    if (ensure_rope_freq_table(theta_base, args->head_dim) != 0) return -1;

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    const int32_t seq_pos_i32 = (int32_t)args->seq_pos;

    NSUInteger ropeTgSize = g_pipe_rope.maxTotalThreadsPerThreadgroup;
    if (ropeTgSize > 1024) ropeTgSize = 1024;
    if (ropeTgSize > args->head_dim / 2) ropeTgSize = args->head_dim / 2;
    if (ropeTgSize == 0) ropeTgSize = 1;
    MTLSize rope_tgrp = MTLSizeMake(ropeTgSize, 1, 1);

    /* Pass 1: RoPE on Q. */
    dispatch_rope_on_buffer(enc, q, args->n_q_heads, args->head_dim,
                            theta_base, seq_pos_i32, rope_tgrp);

    /* Pass 2: RoPE on K. */
    dispatch_rope_on_buffer(enc, k, args->n_kv_heads, args->head_dim,
                            theta_base, seq_pos_i32, rope_tgrp);

    /* Pass 3: write the new K/V row into the cache. */
    [enc setComputePipelineState:g_pipe_kv_cache_write];
    [enc setBuffer:mtl_buf(k) offset:mtl_off(k) atIndex:0];
    [enc setBuffer:mtl_buf(v) offset:mtl_off(v) atIndex:1];
    [enc setBuffer:mtl_buf(k_cache) offset:mtl_off(k_cache) atIndex:2];
    [enc setBuffer:mtl_buf(v_cache) offset:mtl_off(v_cache) atIndex:3];
    [enc setBytes:args length:sizeof(*args) atIndex:4];
    [enc dispatchThreadgroups:kv_grid threadsPerThreadgroup:kv_tgrp];

    /* Pass 4: compute attention using the just-written cache. */
    [enc setComputePipelineState:attn_pipe];
    [enc setBuffer:mtl_buf(q) offset:mtl_off(q) atIndex:0];
    [enc setBuffer:mtl_buf(k_cache) offset:mtl_off(k_cache) atIndex:1];
    [enc setBuffer:mtl_buf(v_cache) offset:mtl_off(v_cache) atIndex:2];
    [enc setBuffer:mtl_buf(output) offset:mtl_off(output) atIndex:3];
    [enc setBytes:args length:sizeof(*args) atIndex:4];
    [enc dispatchThreadgroups:attn_grid threadsPerThreadgroup:attn_tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* Dispatch both kernels in one command buffer:
 *   1. kv_cache_write  : k, v -> k_cache[seq_pos], v_cache[seq_pos]
 *   2. attention_*     : q, cache -> output
 * Metal guarantees that all writes from the first dispatch are visible to the
 * second dispatch in the same command buffer.
 */
static int dispatch_attention_decode(
    id<MTLComputePipelineState> attn_pipe,
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    const ds3_attention_args_host *args,
    MTLSize kv_grid,
    MTLSize kv_tgrp,
    MTLSize attn_grid,
    MTLSize attn_tgrp)
{
    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    /* Pass 1: write the new K/V row into the cache. */
    [enc setComputePipelineState:g_pipe_kv_cache_write];
    [enc setBuffer:mtl_buf(k) offset:mtl_off(k) atIndex:0];
    [enc setBuffer:mtl_buf(v) offset:mtl_off(v) atIndex:1];
    [enc setBuffer:mtl_buf(k_cache) offset:mtl_off(k_cache) atIndex:2];
    [enc setBuffer:mtl_buf(v_cache) offset:mtl_off(v_cache) atIndex:3];
    [enc setBytes:args length:sizeof(*args) atIndex:4];
    [enc dispatchThreadgroups:kv_grid threadsPerThreadgroup:kv_tgrp];

    /* Pass 2: compute attention using the just-written cache. */
    [enc setComputePipelineState:attn_pipe];
    [enc setBuffer:mtl_buf(q) offset:mtl_off(q) atIndex:0];
    [enc setBuffer:mtl_buf(k_cache) offset:mtl_off(k_cache) atIndex:1];
    [enc setBuffer:mtl_buf(v_cache) offset:mtl_off(v_cache) atIndex:2];
    [enc setBuffer:mtl_buf(output) offset:mtl_off(output) atIndex:3];
    [enc setBytes:args length:sizeof(*args) atIndex:4];
    [enc dispatchThreadgroups:attn_grid threadsPerThreadgroup:attn_tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_attention_decode(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    if (validate_attention_args(q, k, v, k_cache, v_cache, output,
                                 seq_pos, max_seq_len,
                                 n_q_heads, n_kv_heads, head_dim) != 0) return -1;

    ds3_attention_args_host args = {
        .seq_pos     = seq_pos,
        .max_seq_len = max_seq_len,
        .n_q_heads   = n_q_heads,
        .n_kv_heads  = n_kv_heads,
        .head_dim    = head_dim,
    };

    NSUInteger tgSize = g_pipe_attention_decode.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize == 0) tgSize = 1;

    NSUInteger kvTgSize = tgSize;
    if (kvTgSize > n_kv_heads) kvTgSize = n_kv_heads;
    if (kvTgSize == 0) kvTgSize = 1;
    MTLSize kv_grid = MTLSizeMake((n_kv_heads + kvTgSize - 1) / kvTgSize, 1, 1);
    MTLSize kv_tgrp = MTLSizeMake(kvTgSize, 1, 1);

    NSUInteger attnTgSize = tgSize;
    if (attnTgSize > n_q_heads) attnTgSize = n_q_heads;
    if (attnTgSize == 0) attnTgSize = 1;
    MTLSize attn_grid = MTLSizeMake((n_q_heads + attnTgSize - 1) / attnTgSize, 1, 1);
    MTLSize attn_tgrp = MTLSizeMake(attnTgSize, 1, 1);

    return dispatch_attention_decode(g_pipe_attention_decode, q, k, v, k_cache, v_cache, output,
                                     &args, kv_grid, kv_tgrp, attn_grid, attn_tgrp);
}

int ds3_metal_attention_decode_simd(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    if (validate_attention_args(q, k, v, k_cache, v_cache, output,
                                 seq_pos, max_seq_len,
                                 n_q_heads, n_kv_heads, head_dim) != 0) return -1;

    ds3_attention_args_host args = {
        .seq_pos     = seq_pos,
        .max_seq_len = max_seq_len,
        .n_q_heads   = n_q_heads,
        .n_kv_heads  = n_kv_heads,
        .head_dim    = head_dim,
    };

    const NSUInteger KV_TG_SIZE = 32;
    MTLSize kv_grid = MTLSizeMake((n_kv_heads + KV_TG_SIZE - 1) / KV_TG_SIZE, 1, 1);
    MTLSize kv_tgrp = MTLSizeMake(KV_TG_SIZE, 1, 1);

    const NSUInteger TG_SIZE = 32;
    MTLSize attn_grid = MTLSizeMake(n_q_heads, 1, 1);
    MTLSize attn_tgrp = MTLSizeMake(TG_SIZE, 1, 1);
    return dispatch_attention_decode(g_pipe_attention_decode_simd, q, k, v, k_cache, v_cache, output,
                                     &args, kv_grid, kv_tgrp, attn_grid, attn_tgrp);
}

int ds3_metal_attention_decode_rope(
    ds3_metal_buffer_t       *q,
    ds3_metal_buffer_t       *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float theta_base)
{
    if (validate_attention_decode_rope_args(q, k, v, k_cache, v_cache, output,
                                            seq_pos, max_seq_len,
                                            n_q_heads, n_kv_heads, head_dim) != 0) return -1;

    ds3_attention_args_host args = {
        .seq_pos     = seq_pos,
        .max_seq_len = max_seq_len,
        .n_q_heads   = n_q_heads,
        .n_kv_heads  = n_kv_heads,
        .head_dim    = head_dim,
    };

    NSUInteger tgSize = g_pipe_attention_decode.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize == 0) tgSize = 1;

    NSUInteger kvTgSize = tgSize;
    if (kvTgSize > n_kv_heads) kvTgSize = n_kv_heads;
    if (kvTgSize == 0) kvTgSize = 1;
    MTLSize kv_grid = MTLSizeMake((n_kv_heads + kvTgSize - 1) / kvTgSize, 1, 1);
    MTLSize kv_tgrp = MTLSizeMake(kvTgSize, 1, 1);

    NSUInteger attnTgSize = tgSize;
    if (attnTgSize > n_q_heads) attnTgSize = n_q_heads;
    if (attnTgSize == 0) attnTgSize = 1;
    MTLSize attn_grid = MTLSizeMake((n_q_heads + attnTgSize - 1) / attnTgSize, 1, 1);
    MTLSize attn_tgrp = MTLSizeMake(attnTgSize, 1, 1);

    return dispatch_attention_decode_rope(g_pipe_attention_decode, q, k, v, k_cache, v_cache, output,
                                          &args, theta_base,
                                          kv_grid, kv_tgrp, attn_grid, attn_tgrp);
}

int ds3_metal_attention_decode_rope_simd(
    ds3_metal_buffer_t       *q,
    ds3_metal_buffer_t       *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    float theta_base)
{
    if (getenv("DS3_USE_BASELINE_ATTN") != NULL) {
        /* Diagnostic fallback to the non-SIMD attention kernel. */
        return ds3_metal_attention_decode_rope(q, k, v, k_cache, v_cache, output,
                                               seq_pos, max_seq_len,
                                               n_q_heads, n_kv_heads, head_dim,
                                               theta_base);
    }

    if (validate_attention_decode_rope_args(q, k, v, k_cache, v_cache, output,
                                            seq_pos, max_seq_len,
                                            n_q_heads, n_kv_heads, head_dim) != 0) return -1;

    ds3_attention_args_host args = {
        .seq_pos     = seq_pos,
        .max_seq_len = max_seq_len,
        .n_q_heads   = n_q_heads,
        .n_kv_heads  = n_kv_heads,
        .head_dim    = head_dim,
    };

    const NSUInteger KV_TG_SIZE = 32;
    MTLSize kv_grid = MTLSizeMake((n_kv_heads + KV_TG_SIZE - 1) / KV_TG_SIZE, 1, 1);
    MTLSize kv_tgrp = MTLSizeMake(KV_TG_SIZE, 1, 1);

    const NSUInteger TG_SIZE = 32;
    MTLSize attn_grid = MTLSizeMake(n_q_heads, 1, 1);
    MTLSize attn_tgrp = MTLSizeMake(TG_SIZE, 1, 1);

    return dispatch_attention_decode_rope(g_pipe_attention_decode_simd, q, k, v, k_cache, v_cache, output,
                                          &args, theta_base,
                                          kv_grid, kv_tgrp, attn_grid, attn_tgrp);
}

/* -------------------------------------------------------------------------- */
/* Batched KV cache write for chunked prefill                                 */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint32_t seq_pos;
    uint32_t n_tokens;
    uint32_t max_seq_len;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
} ds3_attention_batch_args_host;

int ds3_metal_kv_cache_write_batch(
    const ds3_metal_buffer_t *k,
    const ds3_metal_buffer_t *v,
    ds3_metal_buffer_t       *k_cache,
    ds3_metal_buffer_t       *v_cache,
    uint32_t seq_pos,
    uint32_t n_tokens,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    if (!k || !v || !k_cache || !v_cache) return -1;
    if (n_tokens == 0) return 0;
    if (seq_pos + n_tokens > max_seq_len) {
        fprintf(stderr, "ds3_metal: kv_cache_write_batch seq_pos=%u n_tokens=%u exceeds max_seq_len=%u\n",
                seq_pos, n_tokens, max_seq_len);
        return -1;
    }

    ds3_attention_batch_args_host args = {
        .seq_pos     = seq_pos,
        .n_tokens    = n_tokens,
        .max_seq_len = max_seq_len,
        .n_q_heads   = n_q_heads,
        .n_kv_heads  = n_kv_heads,
        .head_dim    = head_dim,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    [enc setComputePipelineState:g_pipe_kv_cache_write_batch];
    [enc setBuffer:mtl_buf(k) offset:mtl_off(k) atIndex:0];
    [enc setBuffer:mtl_buf(v) offset:mtl_off(v) atIndex:1];
    [enc setBuffer:mtl_buf(k_cache) offset:mtl_off(k_cache) atIndex:2];
    [enc setBuffer:mtl_buf(v_cache) offset:mtl_off(v_cache) atIndex:3];
    [enc setBytes:&args length:sizeof(args) atIndex:4];

    const NSUInteger TG_X = 8;
    const NSUInteger TG_Y = 8;
    MTLSize grid = MTLSizeMake((n_tokens + TG_X - 1) / TG_X,
                               (n_kv_heads + TG_Y - 1) / TG_Y, 1);
    MTLSize tgrp = MTLSizeMake(TG_X, TG_Y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* Naive causal chunk attention for prefill                                   */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint32_t seq_pos;
    uint32_t n_tokens;
    uint32_t max_seq_len;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
} ds3_attention_chunk_args_host;

int ds3_metal_attention_chunk(
    const ds3_metal_buffer_t *q,
    const ds3_metal_buffer_t *k_cache,
    const ds3_metal_buffer_t *v_cache,
    ds3_metal_buffer_t       *output,
    uint32_t seq_pos,
    uint32_t n_tokens,
    uint32_t max_seq_len,
    uint32_t n_q_heads,
    uint32_t n_kv_heads,
    uint32_t head_dim)
{
    if (!q || !k_cache || !v_cache || !output) return -1;
    if (n_tokens == 0) return 0;
    if (n_q_heads % n_kv_heads != 0) return -1;
    if (head_dim > 128) return -1;
    if (seq_pos + n_tokens > max_seq_len) {
        fprintf(stderr, "ds3_metal: attention_chunk seq_pos=%u n_tokens=%u exceeds max_seq_len=%u\n",
                seq_pos, n_tokens, max_seq_len);
        return -1;
    }

    ds3_attention_chunk_args_host args = {
        .seq_pos     = seq_pos,
        .n_tokens    = n_tokens,
        .max_seq_len = max_seq_len,
        .n_q_heads   = n_q_heads,
        .n_kv_heads  = n_kv_heads,
        .head_dim    = head_dim,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    [enc setComputePipelineState:g_pipe_attention_chunk];
    [enc setBuffer:mtl_buf(q)         offset:mtl_off(q)         atIndex:0];
    [enc setBuffer:mtl_buf(k_cache)   offset:mtl_off(k_cache)   atIndex:1];
    [enc setBuffer:mtl_buf(v_cache)   offset:mtl_off(v_cache)   atIndex:2];
    [enc setBuffer:mtl_buf(output)    offset:mtl_off(output)    atIndex:3];
    [enc setBytes:&args length:sizeof(args) atIndex:4];

    const NSUInteger TG_X = 8;
    const NSUInteger TG_Y = 8;
    MTLSize grid = MTLSizeMake((n_tokens + TG_X - 1) / TG_X,
                               (n_q_heads + TG_Y - 1) / TG_Y, 1);
    MTLSize tgrp = MTLSizeMake(TG_X, TG_Y, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* MoE FFN                                                                    */
/* -------------------------------------------------------------------------- */

typedef struct {
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint64_t gate_offset;
    uint64_t up_offset;
} ds3_moe_gate_up_args_host;

typedef struct {
    uint32_t n_embd;
    uint32_t n_ff_exp;
    float    weight;
    uint64_t down_offset;
} ds3_moe_down_args_host;

typedef struct {
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t norm_topk_prob;
} ds3_moe_routed_args_host;

typedef struct {
    uint32_t n;
    float    weight;
} ds3_moe_accumulate_args_host;

static void dispatch_moe_expert_f32(
    id<MTLComputeCommandEncoder> enc,
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_exps,
    const ds3_metal_buffer_t *w_up_exps,
    const ds3_metal_buffer_t *w_down_exps,
    ds3_metal_buffer_t       *hidden,
    ds3_metal_buffer_t       *output,
    uint32_t eid,
    float weight,
    uint32_t n_embd,
    uint32_t n_ff_exp,
    MTLSize gate_up_grid,
    MTLSize gate_up_tgrp,
    MTLSize down_grid,
    MTLSize down_tgrp)
{
    const uint64_t gate_offset = (uint64_t)eid * n_ff_exp * n_embd * sizeof(float);
    const uint64_t up_offset   = (uint64_t)eid * n_ff_exp * n_embd * sizeof(float);
    const uint64_t down_offset = (uint64_t)eid * n_embd * n_ff_exp * sizeof(float);

    ds3_moe_gate_up_args_host gate_up_args = {
        .n_embd     = n_embd,
        .n_ff_exp   = n_ff_exp,
        .gate_offset = gate_offset,
        .up_offset   = up_offset,
    };

    ds3_moe_down_args_host down_args = {
        .n_embd     = n_embd,
        .n_ff_exp   = n_ff_exp,
        .weight     = weight,
        .down_offset = down_offset,
    };

    [enc setComputePipelineState:g_pipe_moe_expert_gate_up];
    [enc setBuffer:mtl_buf(input) offset:mtl_off(input) atIndex:0];
    [enc setBuffer:mtl_buf(w_gate_exps) offset:mtl_off(w_gate_exps) atIndex:1];
    [enc setBuffer:mtl_buf(w_up_exps) offset:mtl_off(w_up_exps) atIndex:2];
    [enc setBuffer:mtl_buf(hidden) offset:mtl_off(hidden) atIndex:3];
    [enc setBytes:&gate_up_args length:sizeof(gate_up_args) atIndex:4];
    [enc dispatchThreadgroups:gate_up_grid threadsPerThreadgroup:gate_up_tgrp];

    [enc setComputePipelineState:g_pipe_moe_expert_down];
    [enc setBuffer:mtl_buf(hidden) offset:mtl_off(hidden) atIndex:0];
    [enc setBuffer:mtl_buf(w_down_exps) offset:mtl_off(w_down_exps) atIndex:1];
    [enc setBuffer:mtl_buf(output) offset:mtl_off(output) atIndex:2];
    [enc setBytes:&down_args length:sizeof(down_args) atIndex:3];
    [enc dispatchThreadgroups:down_grid threadsPerThreadgroup:down_tgrp];
}

/* ============================================================================
 * Mixed-type quantized MoE expert dispatch
 * ============================================================================ */

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint64_t row_stride;
    uint64_t weight_offset;
} ds3_matmul_quant_args_host;

static id<MTLComputePipelineState> pipe_for_quant_type(ds3_type_t type)
{
    if (type == DS3_TYPE_Q4_K) return g_pipe_matmul_q4k;
    if (type == DS3_TYPE_Q6_K) return g_pipe_matmul_q6k;
    if (type == DS3_TYPE_Q8_0) return g_pipe_matmul_q8_0;
    return nil;
}

static id<MTLComputePipelineState> pipe_for_quant_type_simd(ds3_type_t type)
{
    if (type == DS3_TYPE_Q4_K) return g_pipe_matmul_q4k_simd;
    if (type == DS3_TYPE_Q6_K) return g_pipe_matmul_q6k_simd;
    if (type == DS3_TYPE_Q8_0) return g_pipe_matmul_q8_0_simd;
    return nil;
}

static int block_size_for_quant_type(ds3_type_t type)
{
    if (type == DS3_TYPE_Q4_K || type == DS3_TYPE_Q6_K) return 256;
    if (type == DS3_TYPE_Q8_0) return 32;
    return 0;
}

static bool quant_type_ok_for_moe(ds3_type_t type, uint32_t in_dim, const char *name)
{
    if (type == DS3_TYPE_F32) return true;
    if (type == DS3_TYPE_Q4_K || type == DS3_TYPE_Q6_K) {
        if (in_dim % 256 != 0) {
            fprintf(stderr, "ds3_metal: %s type=%d requires in_dim=%u multiple of 256\n",
                    name, type, in_dim);
            return false;
        }
        return true;
    }
    if (type == DS3_TYPE_Q8_0) {
        if (in_dim % 32 != 0) {
            fprintf(stderr, "ds3_metal: %s type=%d requires in_dim=%u multiple of 32\n",
                    name, type, in_dim);
            return false;
        }
        return true;
    }
    fprintf(stderr, "ds3_metal: unsupported %s type %d\n", name, type);
    return false;
}

static void dispatch_quant_matmul_row(id<MTLComputeCommandEncoder> enc,
                                      ds3_type_t type,
                                      const ds3_metal_buffer_t *x,
                                      const ds3_metal_buffer_t *W,
                                      ds3_metal_buffer_t       *y,
                                      uint32_t in_dim, uint32_t out_dim,
                                      uint64_t row_stride, uint64_t weight_offset)
{
    NSUInteger tg;
    MTLSize grid, tgrp;

    if (type == DS3_TYPE_F32) {
        ds3_matmul_vec_args_host args = {
            .in_dim     = in_dim,
            .out_dim    = out_dim,
            .row_stride = row_stride,
        };
        tg = g_pipe_matmul_vec.maxTotalThreadsPerThreadgroup;
        if (tg > 1024) tg = 1024;
        if (tg > out_dim) tg = out_dim;
        if (tg == 0) tg = 1;
        grid = MTLSizeMake((out_dim + tg - 1) / tg, 1, 1);
        tgrp = MTLSizeMake(tg, 1, 1);
        [enc setComputePipelineState:g_pipe_matmul_vec];
        [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
        [enc setBuffer:mtl_buf(W) offset:weight_offset + mtl_off(W) atIndex:1];
        [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
        [enc setBytes:&args length:sizeof(args) atIndex:3];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];
        return;
    }

    id<MTLComputePipelineState> pipe = pipe_for_quant_type(type);
    if (!pipe) return;

    ds3_matmul_quant_args_host args = {
        .in_dim        = in_dim,
        .out_dim       = out_dim,
        .row_stride    = row_stride,
        .weight_offset = weight_offset,
    };

    tg = pipe.maxTotalThreadsPerThreadgroup;
    if (tg > 1024) tg = 1024;
    if (tg > out_dim) tg = out_dim;
    if (tg == 0) tg = 1;
    grid = MTLSizeMake((out_dim + tg - 1) / tg, 1, 1);
    tgrp = MTLSizeMake(tg, 1, 1);

    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];
}

/* SIMD-group reduction variant for use inside an existing command encoder.
 * Must match the lane-splitting logic in the *_simd Metal kernels. */
static void dispatch_quant_matmul_row_simd(id<MTLComputeCommandEncoder> enc,
                                           ds3_type_t type,
                                           const ds3_metal_buffer_t *x,
                                           const ds3_metal_buffer_t *W,
                                           ds3_metal_buffer_t       *y,
                                           uint32_t in_dim, uint32_t out_dim,
                                           uint64_t row_stride, uint64_t weight_offset)
{
    id<MTLComputePipelineState> pipe = pipe_for_quant_type_simd(type);
    if (!pipe) return;

    ds3_matmul_quant_args_host args = {
        .in_dim        = in_dim,
        .out_dim       = out_dim,
        .row_stride    = row_stride,
        .weight_offset = weight_offset,
    };

    int block_size = block_size_for_quant_type(type);
    int n_blocks   = (block_size > 0) ? (int)(in_dim / block_size) : (int)in_dim;
    int lanes_per_row = lanes_per_row_for_simd(n_blocks);
    int rows_per_simd = 32 / lanes_per_row;

    MTLSize grid = MTLSizeMake((out_dim + rows_per_simd - 1) / rows_per_simd, 1, 1);
    MTLSize tgrp = MTLSizeMake(32, 1, 1);

    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];
}

/* Helpers to select the fused gate-up / down pipeline for a given weight type. */
static id<MTLComputePipelineState> pipe_for_fused_gate_up(ds3_type_t type)
{
    if (type == DS3_TYPE_F32)  return g_pipe_moe_expert_gate_up;
    if (type == DS3_TYPE_Q4_K) return g_pipe_moe_expert_gate_up_q4k;
    if (type == DS3_TYPE_Q6_K) return g_pipe_moe_expert_gate_up_q6k;
    if (type == DS3_TYPE_Q8_0) return g_pipe_moe_expert_gate_up_q8_0;
    return nil;
}

static id<MTLComputePipelineState> pipe_for_fused_down(ds3_type_t type)
{
    if (type == DS3_TYPE_F32)  return g_pipe_moe_expert_down;
    if (type == DS3_TYPE_Q4_K) return g_pipe_moe_expert_down_q4k;
    if (type == DS3_TYPE_Q6_K) return g_pipe_moe_expert_down_q6k;
    if (type == DS3_TYPE_Q8_0) return g_pipe_moe_expert_down_q8_0;
    return nil;
}

/* Fused gate-up: hidden = silu(input @ W_gate^T) * (input @ W_up^T).
 * Gate and up must share the same type. */
static void dispatch_moe_expert_gate_up_fused(
    id<MTLComputeCommandEncoder> enc,
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate,
    const ds3_metal_buffer_t *w_up,
    ds3_metal_buffer_t       *hidden,
    ds3_type_t type,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint32_t n_embd,
    uint32_t n_ff)
{
    id<MTLComputePipelineState> pipe = pipe_for_fused_gate_up(type);
    if (!pipe) return;

    ds3_moe_gate_up_args_host args = {
        .n_embd      = n_embd,
        .n_ff_exp    = n_ff,
        .gate_offset = gate_offset,
        .up_offset   = up_offset,
    };

    MTLSize grid, tgrp;
    if (type == DS3_TYPE_F32) {
        NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
        if (tg > 1024) tg = 1024;
        if (tg > n_ff) tg = n_ff;
        if (tg == 0) tg = 1;
        grid = MTLSizeMake((n_ff + tg - 1) / tg, 1, 1);
        tgrp = MTLSizeMake(tg, 1, 1);
    } else {
        int block_size = block_size_for_quant_type(type);
        int n_blocks   = (int)(n_embd / block_size);
        int lanes_per_row = lanes_per_row_for_simd(n_blocks);
        int rows_per_simd = 32 / lanes_per_row;
        grid = MTLSizeMake((n_ff + rows_per_simd - 1) / rows_per_simd, 1, 1);
        tgrp = MTLSizeMake(32, 1, 1);
    }

    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(input) offset:mtl_off(input) atIndex:0];
    [enc setBuffer:mtl_buf(w_gate) offset:mtl_off(w_gate) atIndex:1];
    [enc setBuffer:mtl_buf(w_up) offset:mtl_off(w_up) atIndex:2];
    [enc setBuffer:mtl_buf(hidden) offset:mtl_off(hidden) atIndex:3];
    [enc setBytes:&args length:sizeof(args) atIndex:4];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];
}

/* Fused down+accumulate: output += weight * (hidden @ W_down^T). */
static void dispatch_moe_expert_down_fused(
    id<MTLComputeCommandEncoder> enc,
    const ds3_metal_buffer_t *hidden,
    const ds3_metal_buffer_t *w_down,
    ds3_metal_buffer_t       *output,
    ds3_type_t type,
    uint64_t down_offset,
    float weight,
    uint32_t n_embd,
    uint32_t n_ff)
{
    id<MTLComputePipelineState> pipe = pipe_for_fused_down(type);
    if (!pipe) return;

    ds3_moe_down_args_host args = {
        .n_embd     = n_embd,
        .n_ff_exp   = n_ff,
        .weight     = weight,
        .down_offset = down_offset,
    };

    MTLSize grid, tgrp;
    if (type == DS3_TYPE_F32) {
        NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
        if (tg > 1024) tg = 1024;
        if (tg > n_embd) tg = n_embd;
        if (tg == 0) tg = 1;
        grid = MTLSizeMake((n_embd + tg - 1) / tg, 1, 1);
        tgrp = MTLSizeMake(tg, 1, 1);
    } else {
        int block_size = block_size_for_quant_type(type);
        int n_blocks   = (int)(n_ff / block_size);
        int lanes_per_row = lanes_per_row_for_simd(n_blocks);
        int rows_per_simd = 32 / lanes_per_row;
        grid = MTLSizeMake((n_embd + rows_per_simd - 1) / rows_per_simd, 1, 1);
        tgrp = MTLSizeMake(32, 1, 1);
    }

    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(hidden) offset:mtl_off(hidden) atIndex:0];
    [enc setBuffer:mtl_buf(w_down) offset:mtl_off(w_down) atIndex:1];
    [enc setBuffer:mtl_buf(output) offset:mtl_off(output) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];
}

/* Dispatch one expert using fused kernels when gate/up types match and the
 * individual types are supported; otherwise fall back to separate matmuls. */
static void dispatch_moe_expert_quant(
    id<MTLComputeCommandEncoder> enc,
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate,
    const ds3_metal_buffer_t *w_up,
    const ds3_metal_buffer_t *w_down,
    ds3_type_t gate_type,
    ds3_type_t up_type,
    ds3_type_t down_type,
    uint64_t gate_stride,
    uint64_t up_stride,
    uint64_t down_stride,
    ds3_metal_buffer_t       *gate,
    ds3_metal_buffer_t       *up,
    ds3_metal_buffer_t       *down,
    ds3_metal_buffer_t       *output,
    uint32_t eid,
    float weight,
    uint32_t n_embd,
    uint32_t n_ff)
{
    const uint64_t gate_offset = (uint64_t)eid * n_ff * gate_stride;
    const uint64_t up_offset   = (uint64_t)eid * n_ff * up_stride;
    const uint64_t down_offset = (uint64_t)eid * n_embd * down_stride;

    const bool gate_up_fusible = (gate_type == up_type) && (pipe_for_fused_gate_up(gate_type) != nil);
    const bool down_fusible    = pipe_for_fused_down(down_type) != nil;

    if (gate_up_fusible) {
        dispatch_moe_expert_gate_up_fused(enc, input, w_gate, w_up, gate,
                                          gate_type, gate_offset, up_offset,
                                          n_embd, n_ff);
    } else {
        /* gate */
        if (gate_type == DS3_TYPE_F32) {
            dispatch_quant_matmul_row(enc, gate_type, input, w_gate, gate,
                                      n_embd, n_ff, gate_stride, gate_offset);
        } else {
            dispatch_quant_matmul_row_simd(enc, gate_type, input, w_gate, gate,
                                           n_embd, n_ff, gate_stride, gate_offset);
        }

        /* up */
        if (up_type == DS3_TYPE_F32) {
            dispatch_quant_matmul_row(enc, up_type, input, w_up, up,
                                      n_embd, n_ff, up_stride, up_offset);
        } else {
            dispatch_quant_matmul_row_simd(enc, up_type, input, w_up, up,
                                           n_embd, n_ff, up_stride, up_offset);
        }

        /* silu(gate) * up -> gate */
        ds3_moe_silu_mul_args_host silu_args = { .n = n_ff };
        NSUInteger silu_tg = g_pipe_moe_expert_silu_mul.maxTotalThreadsPerThreadgroup;
        if (silu_tg > 1024) silu_tg = 1024;
        if (silu_tg > n_ff) silu_tg = n_ff;
        if (silu_tg == 0) silu_tg = 1;
        MTLSize silu_grid = MTLSizeMake((n_ff + silu_tg - 1) / silu_tg, 1, 1);
        MTLSize silu_tgrp = MTLSizeMake(silu_tg, 1, 1);
        [enc setComputePipelineState:g_pipe_moe_expert_silu_mul];
        [enc setBuffer:mtl_buf(gate) offset:mtl_off(gate) atIndex:0];
        [enc setBuffer:mtl_buf(up) offset:mtl_off(up) atIndex:1];
        [enc setBuffer:mtl_buf(gate) offset:mtl_off(gate) atIndex:2];
        [enc setBytes:&silu_args length:sizeof(silu_args) atIndex:3];
        [enc dispatchThreadgroups:silu_grid threadsPerThreadgroup:silu_tgrp];
    }

    if (down_fusible) {
        dispatch_moe_expert_down_fused(enc, gate, w_down, output,
                                       down_type, down_offset, weight,
                                       n_embd, n_ff);
    } else {
        /* down */
        if (down_type == DS3_TYPE_F32) {
            dispatch_quant_matmul_row(enc, down_type, gate, w_down, down,
                                      n_ff, n_embd, down_stride, down_offset);
        } else {
            dispatch_quant_matmul_row_simd(enc, down_type, gate, w_down, down,
                                           n_ff, n_embd, down_stride, down_offset);
        }

        /* output += weight * down */
        ds3_moe_accumulate_args_host acc_args = { .n = n_embd, .weight = weight };
        NSUInteger acc_tg = g_pipe_moe_expert_accumulate.maxTotalThreadsPerThreadgroup;
        if (acc_tg > 1024) acc_tg = 1024;
        if (acc_tg > n_embd) acc_tg = n_embd;
        if (acc_tg == 0) acc_tg = 1;
        MTLSize acc_grid = MTLSizeMake((n_embd + acc_tg - 1) / acc_tg, 1, 1);
        MTLSize acc_tgrp = MTLSizeMake(acc_tg, 1, 1);
        [enc setComputePipelineState:g_pipe_moe_expert_accumulate];
        [enc setBuffer:mtl_buf(down) offset:mtl_off(down) atIndex:0];
        [enc setBuffer:mtl_buf(output) offset:mtl_off(output) atIndex:1];
        [enc setBytes:&acc_args length:sizeof(acc_args) atIndex:2];
        [enc dispatchThreadgroups:acc_grid threadsPerThreadgroup:acc_tgrp];
    }
}

int ds3_metal_moe_ffn_experts(
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_exps,
    const ds3_metal_buffer_t *w_up_exps,
    const ds3_metal_buffer_t *w_down_exps,
    ds3_type_t gate_type,
    ds3_type_t up_type,
    ds3_type_t down_type,
    uint64_t gate_row_stride,
    uint64_t up_row_stride,
    uint64_t down_row_stride,
    const ds3_metal_buffer_t *w_shared_gate,
    const ds3_metal_buffer_t *w_shared_up,
    const ds3_metal_buffer_t *w_shared_down,
    ds3_type_t shared_gate_type,
    ds3_type_t shared_up_type,
    ds3_type_t shared_down_type,
    uint64_t shared_gate_row_stride,
    uint64_t shared_up_row_stride,
    uint64_t shared_down_row_stride,
    ds3_metal_buffer_t       *output,
    ds3_metal_buffer_t       *hidden,
    ds3_metal_buffer_t       *expert_up,
    ds3_metal_buffer_t       *expert_down,
    uint32_t n_embd,
    uint32_t n_used,
    uint32_t n_ff_exp,
    uint32_t n_ff_shared,
    const int32_t            *indices_host,
    const float              *scores_host)
{
    if (!input || !w_gate_exps || !w_up_exps || !w_down_exps || !output ||
        !indices_host || !scores_host) return -1;
    if (n_embd == 0 || n_used == 0 || n_ff_exp == 0) return -1;

    const bool has_shared = (w_shared_gate && w_shared_up && w_shared_down);
    const bool routed_all_f32 = (gate_type == DS3_TYPE_F32 &&
                                 up_type   == DS3_TYPE_F32 &&
                                 down_type == DS3_TYPE_F32);
    const bool shared_all_f32 = !has_shared ||
                                (shared_gate_type == DS3_TYPE_F32 &&
                                 shared_up_type   == DS3_TYPE_F32 &&
                                 shared_down_type == DS3_TYPE_F32);

    /* Grid sizes for the F32 combined gate_up/down path. */
    NSUInteger gate_up_tg = g_pipe_moe_expert_gate_up.maxTotalThreadsPerThreadgroup;
    if (gate_up_tg > 1024) gate_up_tg = 1024;
    if (gate_up_tg > n_ff_exp) gate_up_tg = n_ff_exp;
    if (gate_up_tg == 0) gate_up_tg = 1;
    MTLSize gate_up_grid = MTLSizeMake((n_ff_exp + gate_up_tg - 1) / gate_up_tg, 1, 1);
    MTLSize gate_up_tgrp = MTLSizeMake(gate_up_tg, 1, 1);

    NSUInteger down_tg = g_pipe_moe_expert_down.maxTotalThreadsPerThreadgroup;
    if (down_tg > 1024) down_tg = 1024;
    if (down_tg > n_embd) down_tg = n_embd;
    if (down_tg == 0) down_tg = 1;
    MTLSize down_grid = MTLSizeMake((n_embd + down_tg - 1) / down_tg, 1, 1);
    MTLSize down_tgrp = MTLSizeMake(down_tg, 1, 1);

    /* Grid sizes for the optional shared expert (F32 path). */
    NSUInteger shared_gate_up_tg = gate_up_tg;
    if (shared_gate_up_tg > n_ff_shared) shared_gate_up_tg = n_ff_shared;
    if (shared_gate_up_tg == 0) shared_gate_up_tg = 1;
    MTLSize shared_gate_up_grid = MTLSizeMake((n_ff_shared + shared_gate_up_tg - 1) / shared_gate_up_tg, 1, 1);
    MTLSize shared_gate_up_tgrp = MTLSizeMake(shared_gate_up_tg, 1, 1);

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    /* Shared expert is always active and weighted 1.0. */
    if (has_shared) {
        if (shared_all_f32) {
            dispatch_moe_expert_f32(enc, input, w_shared_gate, w_shared_up, w_shared_down,
                                    hidden, output,
                                    0, 1.0f,
                                    n_embd, n_ff_shared,
                                    shared_gate_up_grid, shared_gate_up_tgrp,
                                    down_grid, down_tgrp);
        } else {
            dispatch_moe_expert_quant(enc, input, w_shared_gate, w_shared_up, w_shared_down,
                                      shared_gate_type, shared_up_type, shared_down_type,
                                      shared_gate_row_stride, shared_up_row_stride, shared_down_row_stride,
                                      hidden, expert_up, expert_down, output,
                                      0, 1.0f,
                                      n_embd, n_ff_shared);
        }
    }

    for (uint32_t i = 0; i < n_used; i++) {
        if (routed_all_f32) {
            dispatch_moe_expert_f32(enc, input, w_gate_exps, w_up_exps, w_down_exps,
                                    hidden, output,
                                    (uint32_t)indices_host[i], scores_host[i],
                                    n_embd, n_ff_exp,
                                    gate_up_grid, gate_up_tgrp, down_grid, down_tgrp);
        } else {
            dispatch_moe_expert_quant(enc, input, w_gate_exps, w_up_exps, w_down_exps,
                                      gate_type, up_type, down_type,
                                      gate_row_stride, up_row_stride, down_row_stride,
                                      hidden, expert_up, expert_down, output,
                                      (uint32_t)indices_host[i], scores_host[i],
                                      n_embd, n_ff_exp);
        }
    }

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_moe_ffn_router(
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_inp,
    ds3_metal_buffer_t       *gate_logits,
    uint32_t n_embd,
    uint32_t n_expert,
    uint64_t router_row_stride,
    ds3_type_t router_type)
{
    if (router_type == DS3_TYPE_F32) {
        return ds3_metal_vec_matmul_f32(input, w_gate_inp, gate_logits,
                                        n_embd, n_expert, router_row_stride);
    } else {
        return ds3_metal_vec_matmul_quantized(input, w_gate_inp, gate_logits,
                                              n_embd, n_expert, router_row_stride,
                                              router_type);
    }
}

int ds3_metal_moe_router_topk_batch(
    const ds3_metal_buffer_t *logits,
    ds3_metal_buffer_t       *indices,
    ds3_metal_buffer_t       *scores,
    uint32_t n_layers,
    uint32_t n_expert,
    uint32_t n_used,
    bool norm_topk_prob)
{
    if (!logits || !indices || !scores || n_layers == 0 || n_expert == 0 || n_used == 0) return -1;

    typedef struct __attribute__((packed)) {
        uint32_t n_expert;
        uint32_t n_used;
        uint32_t norm_topk_prob;
    } args_t;
    args_t args = {
        .n_expert = n_expert,
        .n_used   = n_used,
        .norm_topk_prob = norm_topk_prob ? 1u : 0u,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    [enc setComputePipelineState:g_pipe_moe_router_topk_batch];
    [enc setBuffer:mtl_buf(logits)  offset:mtl_off(logits)  atIndex:0];
    [enc setBuffer:mtl_buf(indices) offset:mtl_off(indices) atIndex:1];
    [enc setBuffer:mtl_buf(scores)  offset:mtl_off(scores)  atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_layers, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* GPU softmax + top-k for a batch of router logits with multiple tokens per layer.
 * logits:  [n_layers][n_tokens][n_expert]
 * indices: [n_layers][n_tokens][n_used]
 * scores:  [n_layers][n_tokens][n_used] */
int ds3_metal_moe_router_topk_batch_tokens(
    const ds3_metal_buffer_t *logits,
    ds3_metal_buffer_t       *indices,
    ds3_metal_buffer_t       *scores,
    uint32_t n_layers,
    uint32_t n_tokens,
    uint32_t n_expert,
    uint32_t n_used,
    bool norm_topk_prob)
{
    if (!logits || !indices || !scores || n_layers == 0 || n_tokens == 0 ||
        n_expert == 0 || n_used == 0) return -1;

    typedef struct __attribute__((packed)) {
        uint32_t n_expert;
        uint32_t n_used;
        uint32_t n_tokens;
        uint32_t norm_topk_prob;
    } args_t;
    args_t args = {
        .n_expert = n_expert,
        .n_used   = n_used,
        .n_tokens = n_tokens,
        .norm_topk_prob = norm_topk_prob ? 1u : 0u,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }

    [enc setComputePipelineState:g_pipe_moe_router_topk_batch_tokens];
    [enc setBuffer:mtl_buf(logits)  offset:mtl_off(logits)  atIndex:0];
    [enc setBuffer:mtl_buf(indices) offset:mtl_off(indices) atIndex:1];
    [enc setBuffer:mtl_buf(scores)  offset:mtl_off(scores)  atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_layers, n_tokens, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

int ds3_metal_moe_ffn(
    const ds3_metal_buffer_t *input,
    const ds3_metal_buffer_t *w_gate_inp,
    const ds3_metal_buffer_t *w_gate_exps,
    const ds3_metal_buffer_t *w_up_exps,
    const ds3_metal_buffer_t *w_down_exps,
    ds3_type_t gate_type,
    ds3_type_t up_type,
    ds3_type_t down_type,
    uint64_t gate_row_stride,
    uint64_t up_row_stride,
    uint64_t down_row_stride,
    const ds3_metal_buffer_t *w_shared_gate,
    const ds3_metal_buffer_t *w_shared_up,
    const ds3_metal_buffer_t *w_shared_down,
    ds3_type_t shared_gate_type,
    ds3_type_t shared_up_type,
    ds3_type_t shared_down_type,
    uint64_t shared_gate_row_stride,
    uint64_t shared_up_row_stride,
    uint64_t shared_down_row_stride,
    ds3_metal_buffer_t       *output,
    ds3_metal_buffer_t       *gate_logits,
    const ds3_metal_buffer_t *expert_offsets,
    ds3_metal_buffer_t       *hidden,
    ds3_metal_buffer_t       *expert_up,
    ds3_metal_buffer_t       *expert_down,
    uint32_t n_embd,
    uint32_t n_expert,
    uint32_t n_used,
    uint32_t n_ff_exp,
    uint32_t n_ff_shared,
    uint64_t router_row_stride,
    ds3_type_t router_type,
    bool norm_topk_prob)
{
    const bool has_shared = (w_shared_gate != NULL) && (w_shared_up != NULL) && (w_shared_down != NULL);

    if (!input || !w_gate_inp || !w_gate_exps || !w_up_exps || !w_down_exps ||
        !output || !gate_logits || !hidden) {
        return -1;
    }

    /* Validate per-weight types and alignment. */
    if (!quant_type_ok_for_moe(gate_type, n_embd, "gate") ||
        !quant_type_ok_for_moe(up_type,   n_embd, "up") ||
        !quant_type_ok_for_moe(down_type, n_ff_exp, "down")) {
        return -1;
    }
    if (has_shared) {
        if (!quant_type_ok_for_moe(shared_gate_type, n_embd,     "shared_gate") ||
            !quant_type_ok_for_moe(shared_up_type,   n_embd,     "shared_up") ||
            !quant_type_ok_for_moe(shared_down_type, n_ff_shared, "shared_down")) {
            return -1;
        }
    }

    const bool routed_quantized = (gate_type != DS3_TYPE_F32) ||
                                  (up_type   != DS3_TYPE_F32) ||
                                  (down_type != DS3_TYPE_F32);
    const bool shared_quantized = has_shared &&
                                  ((shared_gate_type != DS3_TYPE_F32) ||
                                   (shared_up_type   != DS3_TYPE_F32) ||
                                   (shared_down_type != DS3_TYPE_F32));
    if ((routed_quantized || shared_quantized) && (!expert_up || !expert_down)) {
        fprintf(stderr, "ds3_metal: quantized MoE path requires expert_up/expert_down scratch buffers\n");
        return -1;
    }

    if (n_embd == 0 || n_expert == 0 || n_used == 0 || n_ff_exp == 0) return -1;
    if (n_used > n_expert) return -1;

    /* GPU-only path for the real Qwen3 layout: Q4_K gate/up, Q4_K or Q6_K
     * down, FP32 router, and a precomputed offset table. This avoids the CPU
     * router readback and the per-expert dispatches entirely. */
    const bool use_gpu_only = (expert_offsets != NULL) &&
                              (gate_type == DS3_TYPE_Q4_K) &&
                              (up_type   == DS3_TYPE_Q4_K) &&
                              (down_type == DS3_TYPE_Q4_K || down_type == DS3_TYPE_Q6_K) &&
                              (router_type == DS3_TYPE_F32) &&
                              (n_used <= 8) && (n_ff_exp <= 768);
    if (use_gpu_only) {
        id<MTLComputeCommandEncoder> enc = active_encoder();
        bool own_cb = (enc == nil);
        id<MTLCommandBuffer> cb = nil;
        if (own_cb) {
            cb = [g_queue commandBuffer];
            enc = [cb computeCommandEncoder];
        }

        /* Optional shared expert (same fused path, no CPU readback). */
        const bool shared_all_f32 = !has_shared ||
                                    (shared_gate_type == DS3_TYPE_F32 &&
                                     shared_up_type   == DS3_TYPE_F32 &&
                                     shared_down_type == DS3_TYPE_F32);
        if (has_shared) {
            if (shared_all_f32) {
                NSUInteger gtg = g_pipe_moe_expert_gate_up.maxTotalThreadsPerThreadgroup;
                if (gtg > 1024) gtg = 1024;
                if (gtg > n_ff_shared) gtg = n_ff_shared;
                if (gtg == 0) gtg = 1;
                MTLSize gug = MTLSizeMake((n_ff_shared + gtg - 1) / gtg, 1, 1);
                MTLSize gut = MTLSizeMake(gtg, 1, 1);

                NSUInteger dtg = g_pipe_moe_expert_down.maxTotalThreadsPerThreadgroup;
                if (dtg > 1024) dtg = 1024;
                if (dtg > n_embd) dtg = n_embd;
                if (dtg == 0) dtg = 1;
                MTLSize dg = MTLSizeMake((n_embd + dtg - 1) / dtg, 1, 1);
                MTLSize dt = MTLSizeMake(dtg, 1, 1);

                dispatch_moe_expert_f32(enc, input, w_shared_gate, w_shared_up, w_shared_down,
                                        hidden, output,
                                        0, 1.0f,
                                        n_embd, n_ff_shared,
                                        gug, gut, dg, dt);
            } else {
                dispatch_moe_expert_quant(enc, input, w_shared_gate, w_shared_up, w_shared_down,
                                          shared_gate_type, shared_up_type, shared_down_type,
                                          shared_gate_row_stride, shared_up_row_stride, shared_down_row_stride,
                                          hidden, expert_up, expert_down, output,
                                          0, 1.0f,
                                          n_embd, n_ff_shared);
            }
        }

        /* Two-pass GPU-only routed MoE: router+hidden, then down projection.
         * The scratch `hidden` buffer holds [n_used][n_ff_exp] floats. */
        ds3_moe_routed_args_host routed_args = {
            .n_embd        = n_embd,
            .n_ff_exp      = n_ff_exp,
            .n_expert      = n_expert,
            .n_used        = n_used,
            .norm_topk_prob = norm_topk_prob ? 1u : 0u,
        };

        /* Pass 1: one SIMD group per hidden element. */
        int hidden_n_blocks = (int)(n_embd / 256);
        int hidden_lanes_per_row = lanes_per_row_for_simd(hidden_n_blocks);
        int hidden_rows_per_simd = 32 / hidden_lanes_per_row;
        MTLSize hidden_grid = MTLSizeMake((n_used * n_ff_exp + hidden_rows_per_simd - 1) / hidden_rows_per_simd, 1, 1);
        MTLSize hidden_tgrp = MTLSizeMake(32, 1, 1);

        [enc setComputePipelineState:g_pipe_moe_router_hidden_q4k];
        [enc setBuffer:mtl_buf(input)        offset:mtl_off(input)        atIndex:0];
        [enc setBuffer:mtl_buf(w_gate_inp)   offset:mtl_off(w_gate_inp)   atIndex:1];
        [enc setBuffer:mtl_buf(w_gate_exps)  offset:mtl_off(w_gate_exps)  atIndex:2];
        [enc setBuffer:mtl_buf(w_up_exps)    offset:mtl_off(w_up_exps)    atIndex:3];
        [enc setBuffer:mtl_buf(expert_offsets) offset:mtl_off(expert_offsets) atIndex:4];
        [enc setBuffer:mtl_buf(hidden)       offset:mtl_off(hidden)       atIndex:5];
        [enc setBytes:&routed_args length:sizeof(routed_args) atIndex:6];
        [enc dispatchThreadgroups:hidden_grid threadsPerThreadgroup:hidden_tgrp];

        /* Pass 2: one SIMD group per output element. */
        int out_n_blocks = (int)(n_ff_exp / 256);
        int out_lanes_per_row = lanes_per_row_for_simd(out_n_blocks);
        int out_rows_per_simd = 32 / out_lanes_per_row;
        MTLSize out_grid = MTLSizeMake((n_embd + out_rows_per_simd - 1) / out_rows_per_simd, 1, 1);
        MTLSize out_tgrp = MTLSizeMake(32, 1, 1);

        id<MTLComputePipelineState> out_pipe =
            (down_type == DS3_TYPE_Q4_K) ? g_pipe_moe_output_q4k : g_pipe_moe_output_q6k;
        [enc setComputePipelineState:out_pipe];
        [enc setBuffer:mtl_buf(input)        offset:mtl_off(input)        atIndex:0];
        [enc setBuffer:mtl_buf(w_gate_inp)   offset:mtl_off(w_gate_inp)   atIndex:1];
        [enc setBuffer:mtl_buf(hidden)       offset:mtl_off(hidden)       atIndex:2];
        [enc setBuffer:mtl_buf(w_down_exps)  offset:mtl_off(w_down_exps)  atIndex:3];
        [enc setBuffer:mtl_buf(expert_offsets) offset:mtl_off(expert_offsets) atIndex:4];
        [enc setBuffer:mtl_buf(output)       offset:mtl_off(output)       atIndex:5];
        [enc setBytes:&routed_args length:sizeof(routed_args) atIndex:6];
        [enc dispatchThreadgroups:out_grid threadsPerThreadgroup:out_tgrp];

        if (own_cb) {
            [enc endEncoding];
            [cb commit];
            g_last_cb = cb;
            [cb waitUntilCompleted];
            return cb.error ? -1 : 0;
        }
        return 0;
    }

    /* 1. Compute router logits on GPU. */
    bool was_batch = g_batch_mode;
    int router_rc = ds3_metal_moe_ffn_router(input, w_gate_inp, gate_logits,
                                              n_embd, n_expert, router_row_stride,
                                              router_type);
    if (router_rc != 0) return -1;

    /* The router logits must be visible on the host before we can read them. */
    if (was_batch) ds3_metal_end_batch();

    /* 2. Read logits back to host, softmax + topk + renorm. */
    float *logits_host = (float *)malloc(n_expert * sizeof(float));
    int32_t *indices_host = (int32_t *)malloc(n_used * sizeof(int32_t));
    float *scores_host = (float *)malloc(n_used * sizeof(float));
    if (!logits_host || !indices_host || !scores_host) {
        free(logits_host); free(indices_host); free(scores_host);
        return -1;
    }

    if (ds3_metal_buffer_read(gate_logits, 0, logits_host, n_expert * sizeof(float)) != 0) {
        free(logits_host); free(indices_host); free(scores_host);
        return -1;
    }

    ds3_ref_softmax(logits_host, (int)n_expert);
    ds3_ref_topk(logits_host, (int)n_expert, (int)n_used, indices_host, scores_host);

    if (norm_topk_prob) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < n_used; i++) sum += scores_host[i];
        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            for (uint32_t i = 0; i < n_used; i++) scores_host[i] *= inv_sum;
        }
    }

    free(logits_host);

    /* 3. Dispatch selected experts in a single encoder/command buffer. */
    /* Re-open a batch for the expert dispatches if the caller originally
     * started one; otherwise each expert dispatch will use its own CB. */
    if (was_batch) {
        if (ds3_metal_begin_batch() != 0) {
            free(indices_host);
            free(scores_host);
            return -1;
        }
    }

    int expert_rc = ds3_metal_moe_ffn_experts(
        input,
        w_gate_exps, w_up_exps, w_down_exps,
        gate_type, up_type, down_type,
        gate_row_stride, up_row_stride, down_row_stride,
        w_shared_gate, w_shared_up, w_shared_down,
        shared_gate_type, shared_up_type, shared_down_type,
        shared_gate_row_stride, shared_up_row_stride, shared_down_row_stride,
        output, hidden, expert_up, expert_down,
        n_embd, n_used, n_ff_exp, n_ff_shared,
        indices_host, scores_host);

    free(indices_host);
    free(scores_host);
    return expert_rc;
}

int ds3_metal_vec_matmul_quantized(
    const ds3_metal_buffer_t *x,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *y,
    uint32_t in_dim, uint32_t out_dim, uint64_t row_stride,
    ds3_type_t type)
{
    if (!x || !W || !y) return -1;

    id<MTLComputePipelineState> pipe = nil;
    if (type == DS3_TYPE_Q4_K) {
        if (in_dim % 256 != 0) {
            fprintf(stderr, "ds3_metal: Q4_K in_dim=%u must be a multiple of 256\n", in_dim);
            return -1;
        }
        pipe = g_pipe_matmul_q4k;
    } else if (type == DS3_TYPE_Q8_0) {
        if (in_dim % 32 != 0) {
            fprintf(stderr, "ds3_metal: Q8_0 in_dim=%u must be a multiple of 32\n", in_dim);
            return -1;
        }
        pipe = g_pipe_matmul_q8_0;
    } else if (type == DS3_TYPE_Q6_K) {
        if (in_dim % 256 != 0) {
            fprintf(stderr, "ds3_metal: Q6_K in_dim=%u must be a multiple of 256\n", in_dim);
            return -1;
        }
        pipe = g_pipe_matmul_q6k;
    } else {
        fprintf(stderr, "ds3_metal: unsupported quantized matmul type %d\n", type);
        return -1;
    }

    typedef struct { uint32_t in_dim; uint32_t out_dim; uint64_t row_stride; uint64_t weight_offset; } args_t;
    args_t args = {
        .in_dim     = in_dim,
        .out_dim    = out_dim,
        .row_stride = row_stride,
        .weight_offset = 0,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(x) offset:mtl_off(x) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(y) offset:mtl_off(y) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
    if (tgSize > 1024) tgSize = 1024;
    if (tgSize > out_dim) tgSize = out_dim;
    if (tgSize == 0) tgSize = 1;
    MTLSize grid = MTLSizeMake((out_dim + tgSize - 1) / tgSize, 1, 1);
    MTLSize tgrp = MTLSizeMake(tgSize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

/* Batched matrix × quantized matrix: C = A @ W^T
 *   A: [M][K] FP32 row-major
 *   W: [N][K] quantized (Q4_K / Q6_K / Q8_0) row-major
 *   C: [M][N] FP32 row-major
 * `weight_row_stride` is the byte stride between rows of W.
 * `weight_offset` is a byte offset to the start of the matrix inside W. */
int ds3_metal_matmul_quantized_batch(
    const ds3_metal_buffer_t *A,
    const ds3_metal_buffer_t *W,
    ds3_metal_buffer_t       *C,
    uint32_t M, uint32_t N, uint32_t K,
    uint64_t weight_row_stride,
    uint64_t weight_offset,
    ds3_type_t type)
{
    if (!A || !W || !C) return -1;

    int block_size = 0;
    id<MTLComputePipelineState> pipe = nil;
    if (type == DS3_TYPE_Q4_K) {
        if (K % 256 != 0) {
            fprintf(stderr, "ds3_metal: Q4_K batch K=%u must be a multiple of 256\n", K);
            return -1;
        }
        block_size = 256;
        pipe = g_pipe_matmul_q4k_batch_simd;
    } else if (type == DS3_TYPE_Q8_0) {
        if (K % 32 != 0) {
            fprintf(stderr, "ds3_metal: Q8_0 batch K=%u must be a multiple of 32\n", K);
            return -1;
        }
        block_size = 32;
        pipe = g_pipe_matmul_q8_0_batch_simd;
    } else if (type == DS3_TYPE_Q6_K) {
        if (K % 256 != 0) {
            fprintf(stderr, "ds3_metal: Q6_K batch K=%u must be a multiple of 256\n", K);
            return -1;
        }
        block_size = 256;
        pipe = g_pipe_matmul_q6k_batch_simd;
    } else {
        fprintf(stderr, "ds3_metal: unsupported quantized batch matmul type %d\n", type);
        return -1;
    }

    typedef struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint64_t a_stride;
        uint64_t c_stride;
        uint64_t weight_row_stride;
        uint64_t weight_offset;
    } args_t;
    args_t args = {
        .M = M,
        .N = N,
        .K = K,
        .a_stride = K,
        .c_stride = N,
        .weight_row_stride = weight_row_stride,
        .weight_offset = weight_offset,
    };

    id<MTLComputeCommandEncoder> enc = active_encoder();
    bool own_cb = (enc == nil);
    id<MTLCommandBuffer> cb = nil;
    if (own_cb) {
        cb = [g_queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:mtl_buf(A) offset:mtl_off(A) atIndex:0];
    [enc setBuffer:mtl_buf(W) offset:mtl_off(W) atIndex:1];
    [enc setBuffer:mtl_buf(C) offset:mtl_off(C) atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];

    const int n_blocks      = (int)(K / (uint32_t)block_size);
    const int lanes_per_row = lanes_per_row_for_simd(n_blocks);
    const int rows_per_simd = 32 / lanes_per_row;

    MTLSize grid = MTLSizeMake(M, (N + rows_per_simd - 1) / rows_per_simd, 1);
    MTLSize tgrp = MTLSizeMake(32, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgrp];

    if (own_cb) {
        [enc endEncoding];
        [cb commit];
        g_last_cb = cb;
        [cb waitUntilCompleted];
        return cb.error ? -1 : 0;
    }
    return 0;
}

void ds3_metal_profile_reset(void)
{
    g_profile_cb_count = 0;
    g_profile_total_gpu_ms = 0.0;
    g_profile_total_cpu_ms = 0.0;
}

void ds3_metal_profile_print(const char *label)
{
    if (!g_profile) return;
    fprintf(stderr, "[profile %s] cb_count=%d total_gpu_ms=%.3f total_cpu_wait_ms=%.3f\n",
            label ? label : "",
            g_profile_cb_count,
            g_profile_total_gpu_ms,
            g_profile_total_cpu_ms);
}

int ds3_metal_begin_batch(void)
{
    if (g_batch_mode) {
        fprintf(stderr, "ds3_metal: begin_batch called while already in batch mode\n");
        return -1;
    }
    g_batch_cb = [g_queue commandBuffer];
    if (!g_batch_cb) return -1;
    g_batch_mode = true;
    g_batch_enc  = nil;
    return 0;
}

void ds3_metal_end_batch(void)
{
    if (!g_batch_mode) return;
    if (g_batch_enc) {
        [g_batch_enc endEncoding];
        g_batch_enc = nil;
    }

    struct timespec cpu_start, cpu_end;
    if (g_profile) {
        clock_gettime(CLOCK_MONOTONIC, &cpu_start);
    }

    [g_batch_cb commit];
    g_last_cb = g_batch_cb;
    [g_batch_cb waitUntilCompleted];

    if (g_profile) {
        clock_gettime(CLOCK_MONOTONIC, &cpu_end);
        double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                        (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6;
        double gpu_ms = (g_batch_cb.GPUEndTime - g_batch_cb.GPUStartTime) * 1000.0;
        g_profile_total_cpu_ms += cpu_ms;
        g_profile_total_gpu_ms += gpu_ms;
        g_profile_cb_count++;
    }

    g_batch_cb  = nil;
    g_batch_mode = false;
}

void ds3_metal_synchronize(void) {
    /* Currently all dispatch functions commit and wait internally,
     * so g_last_cb is always already completed.  We keep this
     * function so that future async dispatch APIs have a correct
     * synchronization point. */
    if (g_last_cb) {
        [g_last_cb waitUntilCompleted];
        g_last_cb = nil;
    }
}
