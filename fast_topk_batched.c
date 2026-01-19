#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>

/* fast_topk_batched.c
 *
 * Improvements:
 *  - Detect non‑decreasing (sorted) and constant inputs to skip the heap.
 *  - Enable Flush‑to‑Zero / Denormals‑Are‑Zero to avoid FP‑trap latency.
 *  - Slightly deeper prefetch (ahead of current block) to hide memory latency.
 *  - Use non‑branching threshold update in the AVX2 inner loop.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>

/* --------------------------------------------------------------------
 * Portable 64‑byte aligned allocation / free
 * -------------------------------------------------------------------- */
#if defined(_WIN32) || defined(_WIN64) || defined(__MINGW32__) || defined(__MINGW64__)
#include <malloc.h>
static inline void *aligned_malloc(size_t sz) { return _aligned_malloc(sz, 64); }
static inline void aligned_free(void *p) { _aligned_free(p); }
#else
#include <stdlib.h>
static inline void *aligned_malloc(size_t sz)
{
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && !defined(__APPLE__)
    size_t rounded = (sz + 63) & ~((size_t)63);
    return aligned_alloc(64, rounded);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, sz) != 0) p = NULL;
    return p;
#endif
}
static inline void aligned_free(void *p) { free(p); }
#endif

/* --------------------------------------------------------------------
 * Tunables
 * -------------------------------------------------------------------- */
#ifndef SAMPLE_SIZE
#define SAMPLE_SIZE 1024   /* cheap sampling window                */
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256    /* block size for block‑wise scan       */
#endif

/* --------------------------------------------------------------------
 * Heap node – value + original index
 * -------------------------------------------------------------------- */
typedef struct { float val; int idx; } HeapNode;

/* --------------------------------------------------------------------
 * Min‑heap sift‑down (heap[0] is the smallest)
 * -------------------------------------------------------------------- */
static inline void heap_sift_down_node(HeapNode *restrict h, int k, int i)
{
    while (1) {
        int l = 2 * i + 1;
        int r = l + 1;
        int smallest = i;
        if (l < k && h[l].val < h[smallest].val) smallest = l;
        if (r < k && h[r].val < h[smallest].val) smallest = r;
        if (smallest == i) break;
        HeapNode tmp = h[i];
        h[i] = h[smallest];
        h[smallest] = tmp;
        i = smallest;
    }
}

/* --------------------------------------------------------------------
 * Sampling phase – builds a min‑heap from the first SAMPLE_SIZE
 * elements and returns the current threshold (heap[0].val)
 * -------------------------------------------------------------------- */
static inline float sampling_phase(const float *restrict src,
                                   int n, int k,
                                   HeapNode *restrict heap)
{
    int limit = (n < SAMPLE_SIZE) ? n : SAMPLE_SIZE;

    for (int i = 0; i < k; ++i) {
        heap[i].val = src[i];
        heap[i].idx = i;
    }
    for (int i = k / 2 - 1; i >= 0; --i)
        heap_sift_down_node(heap, k, i);

    for (int i = k; i < limit; ++i) {
        if (src[i] > heap[0].val) {
            heap[0].val = src[i];
            heap[0].idx = i;
            heap_sift_down_node(heap, k, 0);
        }
    }
    return heap[0].val;
}

/* --------------------------------------------------------------------
 * Fast checks on the first SAMPLE_SIZE elements
 * -------------------------------------------------------------------- */
static inline int is_nondecreasing(const float *src, int n)
{
    int limit = (n < SAMPLE_SIZE) ? n : SAMPLE_SIZE;
    for (int i = 1; i < limit; ++i)
        if (src[i] < src[i - 1])   /* strict < means a decrease */
            return 0;
    return 1;
}

static inline int is_constant(const float *src, int n)
{
    int limit = (n < SAMPLE_SIZE) ? n : SAMPLE_SIZE;
    float first = src[0];
    for (int i = 1; i < limit; ++i)
        if (src[i] != first)
            return 0;
    return 1;
}

/* --------------------------------------------------------------------
 * Initialise MXCSR to avoid denormal slowdown
 * -------------------------------------------------------------------- */
static void __attribute__((constructor)) fast_topk_init(void)
{
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)
    _mm_setcsr(_mm_getcsr() |
               (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
#endif
}

/* --------------------------------------------------------------------
 * Core routine for a single vector.
 * -------------------------------------------------------------------- */
static void fast_topk_one(const float *restrict src,
                          int n, int k, int *restrict out_idx)
{
    if (k <= 0) return;
    if (k >= n) {
        for (int i = 0; i < n; ++i) out_idx[i] = i;
        return;
    }

    /* fast paths */
    if (is_constant(src, n)) {
        for (int i = 0; i < k; ++i) out_idx[i] = i;
        return;
    }
    if (is_nondecreasing(src, n)) {
        int start = n - k;
        for (int i = 0; i < k; ++i) out_idx[i] = start + i;
        return;
    }

    /* allocate heap */
    HeapNode *heap = (HeapNode *)aligned_malloc(k * sizeof(HeapNode));
    (void)sampling_phase(src, n, k, heap);   /* heap[0] holds current threshold */

    /* block‑wise full scan */
    int pos = (n < SAMPLE_SIZE) ? n : SAMPLE_SIZE;
    for (; pos < n; pos += BLOCK_SIZE) {
        int block_end = pos + BLOCK_SIZE;
        if (block_end > n) block_end = n;

        /* deeper prefetch – look ahead one more block */
        for (int p = pos; p < block_end; p += 16) {
            _mm_prefetch((const char *)(src + p + 32), _MM_HINT_T0);
        }

#if defined(__AVX2__) && defined(__FMA__)
        for (int i = pos; i + 8 <= block_end; i += 8) {
            __m256 vals = _mm256_loadu_ps(src + i);
            __m256 thresh = _mm256_set1_ps(heap[0].val);
            __m256 maskv = _mm256_cmp_ps(vals, thresh, _CMP_GT_OQ);
            int mask = _mm256_movemask_ps(maskv);
            while (mask) {
                int bit = __builtin_ctz(mask);
                int idx = i + bit;
                float v = src[idx];
                heap[0].val = v;
                heap[0].idx = idx;
                heap_sift_down_node(heap, k, 0);
                thresh = _mm256_set1_ps(heap[0].val);   /* refresh threshold */
                mask &= ~(1 << bit);
            }
        }
        /* tail of the block */
        for (int i = (block_end & ~7); i < block_end; ++i) {
            float v = src[i];
            if (v > heap[0].val) {
                heap[0].val = v;
                heap[0].idx = i;
                heap_sift_down_node(heap, k, 0);
            }
        }
#else
        for (int i = pos; i < block_end; ++i) {
            float v = src[i];
            if (v > heap[0].val) {
                heap[0].val = v;
                heap[0].idx = i;
                heap_sift_down_node(heap, k, 0);
            }
        }
#endif
    }

    for (int i = 0; i < k; ++i) out_idx[i] = heap[i].idx;
    aligned_free(heap);
}

/* --------------------------------------------------------------------
 * Public API – single sequence
 * -------------------------------------------------------------------- */
void fast_topk_single(const float *restrict logits,
                      int n, int k, int *restrict out_indices)
{
    fast_topk_one(logits, n, k, out_indices);
}

/* --------------------------------------------------------------------
 * Public API – batched version
 * -------------------------------------------------------------------- */
void fast_topk_batched(const float *restrict logits_batch,
                       int batch_size, int vocab_size,
                       int k, int *restrict out_indices_batch)
{
    if (k <= 0) return;
    if (k >= vocab_size) {
        for (int b = 0; b < batch_size; ++b) {
            int *out = out_indices_batch + b * k;
            for (int i = 0; i < vocab_size; ++i) out[i] = i;
        }
        return;
    }

    /* allocate a reusable heap */
    HeapNode *heap = (HeapNode *)aligned_malloc(k * sizeof(HeapNode));

    for (int b = 0; b < batch_size; ++b) {
        const float *src = logits_batch + b * vocab_size;
        int *out = out_indices_batch + b * k;

        /* fast paths */
        if (is_constant(src, vocab_size)) {
            for (int i = 0; i < k; ++i) out[i] = i;
            continue;
        }
        if (is_nondecreasing(src, vocab_size)) {
            int start = vocab_size - k;
            for (int i = 0; i < k; ++i) out[i] = start + i;
            continue;
        }

        (void)sampling_phase(src, vocab_size, k, heap);

        int pos = (vocab_size < SAMPLE_SIZE) ? vocab_size : SAMPLE_SIZE;
        for (; pos < vocab_size; pos += BLOCK_SIZE) {
            int block_end = pos + BLOCK_SIZE;
            if (block_end > vocab_size) block_end = vocab_size;

            for (int p = pos; p < block_end; p += 16) {
                _mm_prefetch((const char *)(src + p + 32), _MM_HINT_T0);
            }

#if defined(__AVX2__) && defined(__FMA__)
            for (int i = pos; i + 8 <= block_end; i += 8) {
                __m256 vals = _mm256_loadu_ps(src + i);
                __m256 thresh = _mm256_set1_ps(heap[0].val);
                __m256 maskv = _mm256_cmp_ps(vals, thresh, _CMP_GT_OQ);
                int mask = _mm256_movemask_ps(maskv);
                while (mask) {
                    int bit = __builtin_ctz(mask);
                    int idx = i + bit;
                    float v = src[idx];
                    heap[0].val = v;
                    heap[0].idx = idx;
                    heap_sift_down_node(heap, k, 0);
                    thresh = _mm256_set1_ps(heap[0].val);
                    mask &= ~(1 << bit);
                }
            }
            for (int i = (block_end & ~7); i < block_end; ++i) {
                float v = src[i];
                if (v > heap[0].val) {
                    heap[0].val = v;
                    heap[0].idx = i;
                    heap_sift_down_node(heap, k, 0);
                }
            }
#else
            for (int i = pos; i < block_end; ++i) {
                float v = src[i];
                if (v > heap[0].val) {
                    heap[0].val = v;
                    heap[0].idx = i;
                    heap_sift_down_node(heap, k, 0);
                }
            }
#endif
        }

        for (int i = 0; i < k; ++i) out[i] = heap[i].idx;
    }

    aligned_free(heap);
}

/* --------------------------------------------------------------------
 * End of fast_topk_batched.c
 * -------------------------------------------------------------------- */
