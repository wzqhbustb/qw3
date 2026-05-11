#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static NSString *read_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", path);
        return nil;
    }
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    fread(buf, 1, (size_t)len, fp);
    buf[len] = '\0';
    fclose(fp);
    NSString *src = [NSString stringWithUTF8String:buf];
    free(buf);
    return src;
}

// CPU reference for verification
static void matmul_cpu(const float *A, const float *B, float *C,
                       uint M, uint N, uint K) {
    for (uint i = 0; i < M; i++) {
        for (uint j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(void) {
    @autoreleasepool {

        // ── Matrix dimensions ──────────────────────────────────
        //
        //  C[M×N] = A[M×K] × B[K×N]
        //
        //  Picked as multiples of TILE_SIZE (16) for clean tiling.

        const uint M = 256;
        const uint N = 256;
        const uint K = 256;

        const NSUInteger sizeA = sizeof(float) * M * K;
        const NSUInteger sizeB = sizeof(float) * K * N;
        const NSUInteger sizeC = sizeof(float) * M * N;

        printf("C[%u×%u] = A[%u×%u] × B[%u×%u]\n", M, N, M, K, K, N);
        printf("Total FLOPs: %u multiply-adds\n\n", M * N * K);

        // ── 1. Device ──────────────────────────────────────────
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            for (id<MTLDevice> d in devices) {
                if (!d.isLowPower && !d.isHeadless) { device = d; break; }
            }
            if (!device && devices.count > 0) device = devices[0];
        }
        if (!device) {
            fprintf(stderr, "No Metal GPU found\n");
            return 1;
        }
        printf("GPU: %s\n", [device.name UTF8String]);

        // ── 2. Compile kernel ──────────────────────────────────
        NSString *source = read_file("matmul_tiled.metal");
        if (!source) return 1;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:&error];
        if (!library) {
            fprintf(stderr, "Metal compile error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLFunction> func = [library newFunctionWithName:@"matmul_tiled"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }

        // ── 3. Dispatch configuration (2D) ─────────────────────
        //
        //  Grid:  (N, M, 1) = one thread per element of C
        //  TG:    (TILE_SIZE, TILE_SIZE, 1) = 16×16 threadgroup
        //
        //  Each threadgroup computes one 16×16 tile of C.
        //  Total threadgroups = ceil(N/16) × ceil(M/16).

        const uint TILE = 16;
        MTLSize grid       = MTLSizeMake(N, M, 1);
        MTLSize threadgroup = MTLSizeMake(TILE, TILE, 1);

        NSUInteger tgX = (N + TILE - 1) / TILE;
        NSUInteger tgY = (M + TILE - 1) / TILE;

        printf("\n═══ Dispatch Configuration (2D) ═══\n");
        printf("Grid:                (%u, %u, 1)  ← one thread per C element\n", N, M);
        printf("Threadgroup:         (%u, %u, 1)  ← %u×%u tile\n",
               TILE, TILE, TILE, TILE);
        printf("Threadgroups:        %lu × %lu = %lu\n",
               (unsigned long)tgX, (unsigned long)tgY,
               (unsigned long)(tgX * tgY));
        printf("threadExecutionWidth: %lu\n",
               (unsigned long)pipeline.threadExecutionWidth);
        printf("\n");

        // ── 4. Initialize matrices ─────────────────────────────
        float *hA = malloc(sizeA);
        float *hB = malloc(sizeB);
        for (uint i = 0; i < M * K; i++) hA[i] = (float)(i % 100) / 100.0f;
        for (uint i = 0; i < K * N; i++) hB[i] = (float)((i + 50) % 100) / 100.0f;

        // ── 5. GPU buffers ─────────────────────────────────────
        id<MTLBuffer> bufA = [device newBufferWithBytes:hA length:sizeA
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:hB length:sizeB
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:sizeC
                                                 options:MTLResourceStorageModeShared];

        // ── 6. Encode & dispatch ───────────────────────────────
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(M) atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&K length:sizeof(K) atIndex:5];
        [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];

        [enc endEncoding];

        double t0 = now_sec();
        [cb commit];
        [cb waitUntilCompleted];
        double t1 = now_sec();

        float *gpuC = (float *)bufC.contents;
        double gflops = (2.0 * M * N * K) / ((t1 - t0) * 1e9);
        printf("GPU time: %.3f ms  (%.2f GFLOPS)\n\n",
               (t1 - t0) * 1000.0, gflops);

        // ── 7. Verify with CPU ─────────────────────────────────
        float *cpuC = malloc(sizeC);
        matmul_cpu(hA, hB, cpuC, M, N, K);

        NSUInteger errors = 0;
        float maxErr = 0.0f;
        for (uint i = 0; i < M * N; i++) {
            float err = fabsf(gpuC[i] - cpuC[i]);
            if (err > maxErr) maxErr = err;
            if (err > 1e-4f) {  // FP32 matmul: tolerate small rounding
                if (errors < 5) {
                    printf("MISMATCH [%u]: GPU %.6f  CPU %.6f  (diff %.2e)\n",
                           i, gpuC[i], cpuC[i], err);
                }
                errors++;
            }
        }

        if (errors == 0) {
            printf("All %u elements verified OK (max error: %.2e).\n",
                   M * N, maxErr);
        } else {
            printf("FAIL: %lu mismatches out of %u (max error: %.2e).\n",
                   (unsigned long)errors, M * N, maxErr);
        }

        // ── 8. Show a few results ──────────────────────────────
        printf("\nSample (corner of C):\n");
        for (uint i = 0; i < 3 && i < M; i++) {
            printf("  row %u: ", i);
            for (uint j = 0; j < 4 && j < N; j++) {
                printf("%8.4f ", gpuC[i * N + j]);
            }
            printf("\n");
        }

        free(hA);
        free(hB);
        free(cpuC);
    }
    return 0;
}
