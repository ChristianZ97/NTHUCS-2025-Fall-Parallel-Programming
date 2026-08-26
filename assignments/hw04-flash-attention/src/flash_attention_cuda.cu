// Canonical CUDA implementation.

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tuned full-tile configuration for the target NVIDIA GPU.
#define BR 128
#define BC 16
#define PADDING 1
#define NUM_STREAM 8

static int B, N, d;
static float *Q, *K, *V, *O;
static float *d_Q, *d_K, *d_V, *d_O;

static bool checked_multiply(size_t lhs, size_t rhs, size_t *result);
static bool read_input(const char *filename, size_t *batch_elements,
                       size_t *element_count, size_t *byte_count);
static bool write_output(const char *filename, size_t element_count);
static void release_host_memory();
static bool cuda_ok(cudaError_t error, const char *operation);
static bool release_cuda_resources(cudaStream_t streams[], int stream_count);

void flash_attention(float *d_Q, float *d_K, float *d_V, float *d_O,
                     int batch_count, int sequence_length, int head_dimension,
                     cudaStream_t stream);

__global__ void __launch_bounds__(BR, 2)
flash_attention_kernel(float *__restrict__ d_Q, float *__restrict__ d_K,
                       float *__restrict__ d_V, float *__restrict__ d_O,
                       int sequence_length, int head_dimension);

int main(int argc, char *argv[]) {
    cudaStream_t streams[NUM_STREAM] = {};
    size_t batch_elements = 0;
    size_t element_count = 0;
    size_t byte_count = 0;
    size_t batch_offset = 0;
    size_t base_chunk = 0;
    size_t remainder = 0;
    int stream_count = 0;
    int exit_code = EXIT_FAILURE;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (!read_input(argv[1], &batch_elements, &element_count, &byte_count)) {
        goto cleanup;
    }

    if (!cuda_ok(cudaMalloc(&d_Q, byte_count), "cudaMalloc(Q)") ||
        !cuda_ok(cudaMalloc(&d_K, byte_count), "cudaMalloc(K)") ||
        !cuda_ok(cudaMalloc(&d_V, byte_count), "cudaMalloc(V)") ||
        !cuda_ok(cudaMalloc(&d_O, byte_count), "cudaMalloc(O)")) {
        goto cleanup;
    }

    for (; stream_count < NUM_STREAM; ++stream_count) {
        if (!cuda_ok(cudaStreamCreate(&streams[stream_count]),
                     "cudaStreamCreate")) {
            goto cleanup;
        }
    }

    base_chunk = (size_t)B / NUM_STREAM;
    remainder = (size_t)B % NUM_STREAM;

    // Partition complete batches across streams without changing tile geometry.
    for (int i = 0; i < NUM_STREAM; ++i) {
        const size_t chunk_batches = base_chunk + ((size_t)i < remainder);
        size_t chunk_elements = 0;
        size_t chunk_bytes = 0;
        size_t stream_offset = 0;

        if (chunk_batches == 0) {
            continue;
        }

        if (!checked_multiply(chunk_batches, batch_elements, &chunk_elements) ||
            !checked_multiply(chunk_elements, sizeof(float), &chunk_bytes) ||
            !checked_multiply(batch_offset, batch_elements, &stream_offset)) {
            fprintf(stderr, "Batch offset overflow\n");
            goto cleanup;
        }

        if (!cuda_ok(cudaMemcpyAsync(d_Q + stream_offset, Q + stream_offset,
                                     chunk_bytes, cudaMemcpyHostToDevice,
                                     streams[i]),
                     "cudaMemcpyAsync(Q)") ||
            !cuda_ok(cudaMemcpyAsync(d_K + stream_offset, K + stream_offset,
                                     chunk_bytes, cudaMemcpyHostToDevice,
                                     streams[i]),
                     "cudaMemcpyAsync(K)") ||
            !cuda_ok(cudaMemcpyAsync(d_V + stream_offset, V + stream_offset,
                                     chunk_bytes, cudaMemcpyHostToDevice,
                                     streams[i]),
                     "cudaMemcpyAsync(V)")) {
            goto cleanup;
        }

        flash_attention(d_Q + stream_offset, d_K + stream_offset,
                        d_V + stream_offset, d_O + stream_offset,
                        (int)chunk_batches, N, d, streams[i]);
        if (!cuda_ok(cudaGetLastError(), "flash_attention kernel launch")) {
            goto cleanup;
        }

        if (!cuda_ok(cudaMemcpyAsync(O + stream_offset, d_O + stream_offset,
                                     chunk_bytes, cudaMemcpyDeviceToHost,
                                     streams[i]),
                     "cudaMemcpyAsync(O)")) {
            goto cleanup;
        }

        batch_offset += chunk_batches;
    }

    if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        goto cleanup;
    }
    if (!write_output(argv[2], element_count)) {
        goto cleanup;
    }

    exit_code = EXIT_SUCCESS;

cleanup:
    if (!release_cuda_resources(streams, stream_count)) {
        exit_code = EXIT_FAILURE;
    }
    release_host_memory();
    return exit_code;
}

__global__ void __launch_bounds__(BR, 2)
flash_attention_kernel(float *__restrict__ d_Q, float *__restrict__ d_K,
                       float *__restrict__ d_V, float *__restrict__ d_O,
                       const int N, const int d) {
    const size_t batch_offset =
        (size_t)blockIdx.y * (size_t)N * (size_t)d;
    const size_t row_stride = (size_t)d + PADDING;
    const size_t row_offset_q = (size_t)blockIdx.x * BR;
    const int tx = threadIdx.x;
    const int bd = blockDim.x;

    d_Q += batch_offset;
    d_K += batch_offset;
    d_V += batch_offset;
    d_O += batch_offset;

    // Q, K, and V tiles share one padded dynamic-memory allocation.
    extern __shared__ float sram[];
    float *sm_Q = sram;
    float *sm_K = sm_Q + (size_t)BR * row_stride;
    float *sm_V = sm_K + (size_t)BC * row_stride;
    float O_i[64];

    #pragma unroll
    for (int t = 0; t < d; t++)
        O_i[t] = 0.0f;

    float l_i = 0.0f;
    float m_i = -FLT_MAX;

    #pragma unroll
    for (int i = tx; i < BR * d; i += bd) {
        const int r = i / d;
        const int c = i % d;
        sm_Q[(size_t)r * row_stride + (size_t)c] =
            d_Q[row_offset_q * (size_t)d + (size_t)i];
    }

    for (int j = 0; j < N; j += BC) {
        const size_t column_offset = (size_t)j * (size_t)d;

        #pragma unroll
        for (int i = tx; i < BC * d; i += bd) {
            const int r = i / d;
            const int c = i % d;
            const size_t tile_offset =
                (size_t)r * row_stride + (size_t)c;
            sm_K[tile_offset] = d_K[column_offset + (size_t)i];
            sm_V[tile_offset] = d_V[column_offset + (size_t)i];
        }

        __syncthreads();

        float S_ij[BC];
        float m_ij_tilde = -FLT_MAX;

        for (int k = 0; k < BC; k++) {
            float score = 0.0f;
            for (int t = 0; t < d; t++)
                score += sm_Q[(size_t)tx * row_stride + (size_t)t] *
                         sm_K[(size_t)k * row_stride + (size_t)t];
            score *= (1.0f / sqrtf((float)d));

            S_ij[k] = score;
            m_ij_tilde = fmaxf(m_ij_tilde, score);
        }

        // Maintain max and sum-exp online so the N-by-N matrix is never stored.
        float m_i_new = fmaxf(m_i, m_ij_tilde);
        float exp_m_diff_old = expf(m_i - m_i_new);
        float P_ij[BC];
        float l_ij_sum_exp = 0.0f;

        for (int k = 0; k < BC; k++) {
            P_ij[k] = expf(S_ij[k] - m_i_new);
            l_ij_sum_exp += P_ij[k];
        }

        float l_i_new = l_i * exp_m_diff_old + l_ij_sum_exp;

        for (int t = 0; t < d; t++) {
            float P_ij_V_j = 0.0f;
            for (int k = 0; k < BC; k++)
                P_ij_V_j +=
                    P_ij[k] * sm_V[(size_t)k * row_stride + (size_t)t];

            O_i[t] = O_i[t] * exp_m_diff_old + P_ij_V_j;
        }

        m_i = m_i_new;
        l_i = l_i_new;
        __syncthreads();
    }

    float *sm_O = sram;
    #pragma unroll
    for (int t = 0; t < d; t++)
        sm_O[(size_t)tx * row_stride + (size_t)t] = O_i[t] / l_i;

    __syncthreads();

    const size_t output_offset = row_offset_q * (size_t)d;
    #pragma unroll
    for (int i = tx; i < BR * d; i += bd) {
        const int r = i / d;
        const int c = i % d;
        d_O[output_offset + (size_t)i] =
            sm_O[(size_t)r * row_stride + (size_t)c];
    }
}

void flash_attention(float *d_Q, float *d_K, float *d_V, float *d_O,
                     const int B_chunk, const int N, const int d,
                     cudaStream_t stream) {
    dim3 blocks_per_grid((unsigned int)(((size_t)N + BR - 1) / BR),
                         (unsigned int)B_chunk);
    dim3 threads_per_block(BR);
    const size_t row_stride = (size_t)d + PADDING;
    const size_t shared_elements =
        ((size_t)BR + 2 * (size_t)BC) * row_stride;
    const size_t sram_size = shared_elements * sizeof(float);

    flash_attention_kernel<<<blocks_per_grid, threads_per_block, sram_size,
                             stream>>>(d_Q, d_K, d_V, d_O, N, d);
}

static bool checked_multiply(size_t lhs, size_t rhs, size_t *result) {
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

static bool read_input(const char *filename, size_t *batch_elements,
                       size_t *element_count, size_t *byte_count) {
    FILE *file = fopen(filename, "rb");
    int dimensions[3];

    if (file == NULL) {
        fprintf(stderr, "Cannot open input file: %s\n", filename);
        return false;
    }
    if (fread(dimensions, sizeof(dimensions[0]), 3, file) != 3) {
        fprintf(stderr, "Cannot read B, N, and d from: %s\n", filename);
        fclose(file);
        return false;
    }

    B = dimensions[0];
    N = dimensions[1];
    d = dimensions[2];
    if (B <= 0 || N <= 0 || d <= 0 || d > 64 ||
        N % BR != 0 || N % BC != 0) {
        fprintf(stderr,
                "Unsupported dimensions B=%d N=%d d=%d; require B>0, "
                "N>0 divisible by %d and %d, and 1<=d<=64\n",
                B, N, d, BR, BC);
        fclose(file);
        return false;
    }

    if (!checked_multiply((size_t)N, (size_t)d, batch_elements) ||
        !checked_multiply((size_t)B, *batch_elements, element_count) ||
        !checked_multiply(*element_count, sizeof(float), byte_count)) {
        fprintf(stderr, "Input dimensions overflow addressable memory\n");
        fclose(file);
        return false;
    }

    Q = (float *)malloc(*byte_count);
    K = (float *)malloc(*byte_count);
    V = (float *)malloc(*byte_count);
    O = (float *)malloc(*byte_count);
    if (Q == NULL || K == NULL || V == NULL || O == NULL) {
        fprintf(stderr, "Cannot allocate host buffers\n");
        fclose(file);
        release_host_memory();
        return false;
    }

    for (int i = 0; i < B; ++i) {
        const size_t offset = (size_t)i * *batch_elements;
        if (fread(Q + offset, sizeof(float), *batch_elements, file) !=
                *batch_elements ||
            fread(K + offset, sizeof(float), *batch_elements, file) !=
                *batch_elements ||
            fread(V + offset, sizeof(float), *batch_elements, file) !=
                *batch_elements) {
            fprintf(stderr, "Input tensor data is truncated: %s\n", filename);
            fclose(file);
            release_host_memory();
            return false;
        }
    }

    memset(O, 0, *byte_count);
    if (fclose(file) != 0) {
        fprintf(stderr, "Cannot close input file: %s\n", filename);
        release_host_memory();
        return false;
    }
    return true;
}

static bool write_output(const char *filename, size_t element_count) {
    FILE *file = fopen(filename, "wb");
    bool success = true;

    if (file == NULL) {
        fprintf(stderr, "Cannot open output file: %s\n", filename);
        return false;
    }
    if (fwrite(O, sizeof(float), element_count, file) != element_count) {
        fprintf(stderr, "Cannot write output tensor: %s\n", filename);
        success = false;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Cannot close output file: %s\n", filename);
        success = false;
    }
    return success;
}

static void release_host_memory() {
    free(Q);
    free(K);
    free(V);
    free(O);
    Q = K = V = O = NULL;
}

static bool cuda_ok(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return false;
}

static bool release_cuda_resources(cudaStream_t streams[], int stream_count) {
    bool success = true;

    for (int i = 0; i < stream_count; ++i) {
        if (!cuda_ok(cudaStreamDestroy(streams[i]), "cudaStreamDestroy")) {
            success = false;
        }
    }
    if (d_Q != NULL && !cuda_ok(cudaFree(d_Q), "cudaFree(Q)")) {
        success = false;
    }
    if (d_K != NULL && !cuda_ok(cudaFree(d_K), "cudaFree(K)")) {
        success = false;
    }
    if (d_V != NULL && !cuda_ok(cudaFree(d_V), "cudaFree(V)")) {
        success = false;
    }
    if (d_O != NULL && !cuda_ok(cudaFree(d_O), "cudaFree(O)")) {
        success = false;
    }
    d_Q = d_K = d_V = d_O = NULL;
    return success;
}
