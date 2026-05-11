#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

int main(int argc, char **argv) {
    @autoreleasepool {

        NSUInteger count = 100;
        if (argc > 1) count = (NSUInteger)atoi(argv[1]);
        if (count < 1) count = 1;

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
        printf("GPU: %s\n\n", [device.name UTF8String]);

        // ── 2. Compile kernel ──────────────────────────────────
        NSString *source = read_file("add_vectors.metal");
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
        id<MTLFunction> addFunc = [library newFunctionWithName:@"add_vectors"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:addFunc error:&error];
        if (!pipeline) {
            fprintf(stderr, "Pipeline error: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }

        // ── 3. Dispatch configuration ──────────────────────────
        //
        //  Key APIs on the pipeline:
        //    threadExecutionWidth        = 32  (SIMD group width — the hardware's
        //                                       native execution unit)
        //    maxTotalThreadsPerThreadgroup = 1024 (hard limit per threadgroup)
        //
        //  Best practice: set threadgroup size to a multiple of
        //  threadExecutionWidth. For 1D work, 64 or 128 is typical.
        //  Metal's dispatchThreads: auto-calculates the grid of
        //  threadgroups so you don't need to handle non-uniform N.

        NSUInteger execWidth   = pipeline.threadExecutionWidth;      // 32
        NSUInteger maxTG       = pipeline.maxTotalThreadsPerThreadgroup; // 1024
        NSUInteger tgSize      = execWidth * 2;  // 64 threads per threadgroup

        MTLSize grid           = MTLSizeMake(count, 1, 1);
        MTLSize threadgroup    = MTLSizeMake(tgSize, 1, 1);

        NSUInteger tgCount     = (count + tgSize - 1) / tgSize;
        NSUInteger totalThreads = tgCount * tgSize;

        printf("═══ Dispatch Configuration ═══\n");
        printf("Element count (N):        %lu\n", (unsigned long)count);
        printf("threadExecutionWidth:     %lu  ← SIMD group width\n",
               (unsigned long)execWidth);
        printf("maxTotalThreadsPerTG:     %lu  ← hard upper bound\n",
               (unsigned long)maxTG);
        printf("Threadgroup size (chosen): %lu  ← %lu × threadExecutionWidth\n",
               (unsigned long)tgSize, (unsigned long)(tgSize / execWidth));
        printf("Grid size:                (%lu, 1, 1)\n",
               (unsigned long)count);
        printf("Threadgroups launched:    %lu\n", (unsigned long)tgCount);
        printf("Total threads launched:   %lu\n", (unsigned long)totalThreads);
        if (totalThreads > count) {
            printf("  → %lu threads will be outside the grid (masked by HW)\n",
                   (unsigned long)(totalThreads - count));
        }
        printf("\n");

        // ── 4. Prepare data ────────────────────────────────────
        NSUInteger dataSize = sizeof(float) * count;
        float *a = malloc(dataSize);
        float *b = malloc(dataSize);
        for (NSUInteger i = 0; i < count; i++) {
            a[i] = (float)i;
            b[i] = (float)(i * 10);
        }

        // ── 5. Buffers (Shared = CPU + GPU both see the same memory) ──
        id<MTLBuffer> bufA = [device newBufferWithBytes:a
                                                 length:dataSize
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b
                                                 length:dataSize
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufR = [device newBufferWithLength:dataSize
                                                 options:MTLResourceStorageModeShared];

        // ── 6. Encode & dispatch ───────────────────────────────
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufR offset:0 atIndex:2];
        [enc dispatchThreads:grid threadsPerThreadgroup:threadgroup];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        // ── 7. Verify ──────────────────────────────────────────
        float *result = (float *)bufR.contents;
        NSUInteger errors = 0;
        for (NSUInteger i = 0; i < count; i++) {
            float expected = a[i] + b[i];
            if (fabsf(result[i] - expected) > 1e-6f) {
                if (errors < 5) {
                    printf("MISMATCH [%lu]: expected %.0f, got %.0f\n",
                           (unsigned long)i, expected, result[i]);
                }
                errors++;
            }
        }
        if (errors == 0) {
            printf("All %lu elements verified OK.\n", (unsigned long)count);
        } else {
            printf("FAIL: %lu mismatches out of %lu.\n",
                   (unsigned long)errors, (unsigned long)count);
        }

        // ── 8. Show a few results ──────────────────────────────
        printf("\nSample results:\n");
        NSUInteger show = count < 8 ? count : 5;
        for (NSUInteger i = 0; i < show; i++) {
            printf("  [%lu]  %.0f + %.0f = %.0f\n",
                   (unsigned long)i, a[i], b[i], result[i]);
        }
        if (count > show + 3) printf("  ...\n");
        for (NSUInteger i = count - 3; i < count && count > show + 3; i++) {
            printf("  [%lu]  %.0f + %.0f = %.0f\n",
                   (unsigned long)i, a[i], b[i], result[i]);
        }

        free(a);
        free(b);
    }
    return 0;
}
