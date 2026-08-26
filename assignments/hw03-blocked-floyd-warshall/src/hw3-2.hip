// hw3-2-opt.hip
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "io_utils.h"

#define BLOCKING_FACTOR 64
#define HALF_BLOCK BLOCKING_FACTOR / 2
#define INF ((1 << 30) - 1)

static int *D;
static int *d_D;
static int V, E;
static int V_padded;
static size_t matrix_size;

hipStream_t stream_main, stream_row, stream_col;
hipEvent_t event_p1_done, event_p2_row_done, event_p2_col_done;

__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded);

void input(char *infile);
void output(char *outfile);
void block_FW();

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    input(argv[1]);

    hipStreamCreate(&stream_main);
    hipStreamCreate(&stream_row);
    hipStreamCreate(&stream_col);
    hipEventCreate(&event_p1_done);
    hipEventCreate(&event_p2_row_done);
    hipEventCreate(&event_p2_col_done);

    const size_t size = matrix_size;
    hipMalloc(&d_D, size);

    hipMemcpy(d_D, D, size, hipMemcpyHostToDevice);

    block_FW();
    hipMemcpy(D, d_D, size, hipMemcpyDeviceToHost);

    output(argv[2]);

    hipFree(d_D);
    hipStreamDestroy(stream_main);
    hipStreamDestroy(stream_row);
    hipStreamDestroy(stream_col);
    hipEventDestroy(event_p1_done);
    hipEventDestroy(event_p2_row_done);
    hipEventDestroy(event_p2_col_done);

    return 0;
}

void block_FW() {
    const int round = V_padded / BLOCKING_FACTOR;
    dim3 block_dim(HALF_BLOCK, HALF_BLOCK);

    // Phase 1 uses one block, phase 2 uses one block per round, and phase 3
    // covers the round-by-round block grid.

    for (int r = 0; r < round; ++r) {
        // 1. Phase 1: Pivot Block (Main Stream)
        kernel_phase1<<<1, block_dim, 0, stream_main>>>(d_D, r, V_padded);
        hipEventRecord(event_p1_done, stream_main);

        // 2. Phase 2: Row & Col (Separate Streams)

        // Stream Row waits for Phase 1
        hipStreamWaitEvent(stream_row, event_p1_done, 0);
        kernel_phase2_row<<<round, block_dim, 0, stream_row>>>(d_D, r, V_padded);
        hipEventRecord(event_p2_row_done, stream_row);

        // Stream Col waits for Phase 1
        hipStreamWaitEvent(stream_col, event_p1_done, 0);
        kernel_phase2_col<<<round, block_dim, 0, stream_col>>>(d_D, r, V_padded);
        hipEventRecord(event_p2_col_done, stream_col);

        // 3. Phase 3: Remaining Blocks (Main Stream)
        // Must wait for both Phase 2 parts to finish
        hipStreamWaitEvent(stream_main, event_p2_row_done, 0);
        hipStreamWaitEvent(stream_main, event_p2_col_done, 0);

        dim3 grid_phase3(round, round);
        kernel_phase3<<<grid_phase3, block_dim, 0, stream_main>>>(d_D, r, V_padded);
    }
}

void input(char *infile) {
    FILE *file = open_file_or_die(infile, "rb");
    read_exact_or_die(&V, sizeof(V), 1, file, infile, "vertex count");
    read_exact_or_die(&E, sizeof(E), 1, file, infile, "edge count");
    validate_graph_header_or_die(V, E, 0, infile);

    V_padded = padded_width_or_die(V, BLOCKING_FACTOR, infile);
    matrix_size = matrix_bytes_or_die(V_padded, infile);

    hipHostAlloc(&D, matrix_size, hipHostAllocDefault);

    // #pragma unroll

    for (int i = 0; i < V_padded; ++i)
        for (int j = 0; j < V_padded; ++j)
            D[(size_t)i * V_padded + j] = (i == j) ? 0 : INF;

    int pair[3];
    for (int i = 0; i < E; ++i) {
        read_exact_or_die(pair, sizeof(int), 3, file, infile, "edge record");
        validate_edge_or_die(pair, V, infile);
        D[(size_t)pair[0] * V_padded + pair[1]] = pair[2];
    }
    close_file_or_die(file, infile);
}

void output(char *outfile) {
    FILE *f = open_file_or_die(outfile, "wb");
    for (int i = 0; i < V; ++i)
        write_exact_or_die(
            &D[(size_t)i * V_padded], sizeof(int), (size_t)V, f, outfile);
    close_file_or_die(f, outfile);

    hipFreeHost(D);
}

// Helper functions and kernels.

// int4 Vectorized Load
__device__ __forceinline__ void load_block_int4(const int *__restrict__ src, const size_t stride, int dst[BLOCKING_FACTOR][BLOCKING_FACTOR + 1]) {

    const int tid = threadIdx.y * HALF_BLOCK + threadIdx.x;
    const int iterations = (BLOCKING_FACTOR * BLOCKING_FACTOR) / (HALF_BLOCK * HALF_BLOCK * 4);

#pragma unroll

    for (int i = 0; i < iterations; ++i) {
        const int offset = tid + i * (HALF_BLOCK * HALF_BLOCK);
        const int r = offset / (BLOCKING_FACTOR / 4);
        const int c = (offset % (BLOCKING_FACTOR / 4)) * 4;

        int4 tmp = *((int4 *)(src + r * stride + c));
        dst[r][c] = tmp.x;
        dst[r][c + 1] = tmp.y;
        dst[r][c + 2] = tmp.z;
        dst[r][c + 3] = tmp.w;
    }
}

// int4 Vectorized Store
__device__ __forceinline__ void store_block_int4(int *__restrict__ dst, const size_t stride, int src[BLOCKING_FACTOR][BLOCKING_FACTOR + 1]) {

    const int tid = threadIdx.y * HALF_BLOCK + threadIdx.x;
    const int iterations = (BLOCKING_FACTOR * BLOCKING_FACTOR) / (HALF_BLOCK * HALF_BLOCK * 4);

#pragma unroll

    for (int i = 0; i < iterations; ++i) {
        const int offset = tid + i * (HALF_BLOCK * HALF_BLOCK);
        const int r = offset / (BLOCKING_FACTOR / 4);
        const int c = (offset % (BLOCKING_FACTOR / 4)) * 4;

        int4 tmp;
        tmp.x = src[r][c];
        tmp.y = src[r][c + 1];
        tmp.z = src[r][c + 2];
        tmp.w = src[r][c + 3];
        *((int4 *)(dst + r * stride + c)) = tmp;
    }
}

__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    const size_t b_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;

    load_block_int4(d_D + b_start, stride, sm_pivot);
    __syncthreads();

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_pivot[ty][k];
        const int r1 = sm_pivot[ty + HALF_BLOCK][k];
        const int c0 = sm_pivot[k][tx];
        const int c1 = sm_pivot[k][tx + HALF_BLOCK];

        sm_pivot[ty][tx] = min(sm_pivot[ty][tx], r0 + c0);
        sm_pivot[ty][tx + HALF_BLOCK] = min(sm_pivot[ty][tx + HALF_BLOCK], r0 + c1);
        sm_pivot[ty + HALF_BLOCK][tx] = min(sm_pivot[ty + HALF_BLOCK][tx], r1 + c0);
        sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = min(sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK], r1 + c1);
        __syncthreads();
    }

    store_block_int4(d_D + b_start, stride, sm_pivot);
}

__global__ void __launch_bounds__(1024) kernel_phase2_row(int *d_D, const int r, const int V_padded) {

    const int b_idx_x = blockIdx.x;
    if (b_idx_x == r) return;  // Skip the pivot block itself (handled in Phase 1)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_y = r;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const size_t pivot_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;
    const size_t self_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;

    load_block_int4(d_D + pivot_start, stride, sm_pivot);
    load_block_int4(d_D + self_start, stride, sm_self);
    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = sm_self[ty][tx];
    reg_self[0][1] = sm_self[ty][tx + HALF_BLOCK];
    reg_self[1][0] = sm_self[ty + HALF_BLOCK][tx];
    reg_self[1][1] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_pivot[ty][k];
        const int r1 = sm_pivot[ty + HALF_BLOCK][k];
        const int c0 = sm_self[k][tx];
        const int c1 = sm_self[k][tx + HALF_BLOCK];

        reg_self[0][0] = min(reg_self[0][0], r0 + c0);
        reg_self[0][1] = min(reg_self[0][1], r0 + c1);
        reg_self[1][0] = min(reg_self[1][0], r1 + c0);
        reg_self[1][1] = min(reg_self[1][1], r1 + c1);
        // __syncthreads();
    }

    sm_self[ty][tx] = reg_self[0][0];
    sm_self[ty][tx + HALF_BLOCK] = reg_self[0][1];
    sm_self[ty + HALF_BLOCK][tx] = reg_self[1][0];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = reg_self[1][1];

    __syncthreads();
    store_block_int4(d_D + self_start, stride, sm_self);
}

__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded) {

    const int b_idx_y = blockIdx.x;
    if (b_idx_y == r) return;  // Skip the pivot block itself (handled in Phase 1)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_x = r;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const size_t pivot_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;
    const size_t self_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;

    load_block_int4(d_D + pivot_start, stride, sm_pivot);
    load_block_int4(d_D + self_start, stride, sm_self);
    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = sm_self[ty][tx];
    reg_self[0][1] = sm_self[ty][tx + HALF_BLOCK];
    reg_self[1][0] = sm_self[ty + HALF_BLOCK][tx];
    reg_self[1][1] = sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK];

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_self[ty][k];
        const int r1 = sm_self[ty + HALF_BLOCK][k];
        const int c0 = sm_pivot[k][tx];
        const int c1 = sm_pivot[k][tx + HALF_BLOCK];

        reg_self[0][0] = min(reg_self[0][0], r0 + c0);
        reg_self[0][1] = min(reg_self[0][1], r0 + c1);
        reg_self[1][0] = min(reg_self[1][0], r1 + c0);
        reg_self[1][1] = min(reg_self[1][1], r1 + c1);
        // __syncthreads();
    }

    sm_self[ty][tx] = reg_self[0][0];
    sm_self[ty][tx + HALF_BLOCK] = reg_self[0][1];
    sm_self[ty + HALF_BLOCK][tx] = reg_self[1][0];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = reg_self[1][1];

    __syncthreads();
    store_block_int4(d_D + self_start, stride, sm_self);
}

__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded) {

    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y;
    if (b_idx_x == r || b_idx_y == r) return;  // Skip Phase 1 & 2 blocks

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];  // Row Block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];  // Col Block (r, x)

    const size_t row_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)r * BLOCKING_FACTOR;
    const size_t col_start =
        (size_t)r * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;
    const size_t self_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;

    load_block_int4(d_D + row_start, stride, sm_row);
    load_block_int4(d_D + col_start, stride, sm_col);
    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = d_D[self_start + ty * stride + tx];
    reg_self[0][1] = d_D[self_start + ty * stride + (tx + HALF_BLOCK)];
    reg_self[1][0] = d_D[self_start + (ty + HALF_BLOCK) * stride + tx];
    reg_self[1][1] = d_D[self_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];

#pragma unroll 32

    for (int k = 0; k < BLOCKING_FACTOR; ++k) {
        const int r0 = sm_row[ty][k];
        const int r1 = sm_row[ty + HALF_BLOCK][k];
        const int c0 = sm_col[k][tx];
        const int c1 = sm_col[k][tx + HALF_BLOCK];

        reg_self[0][0] = min(reg_self[0][0], r0 + c0);
        reg_self[0][1] = min(reg_self[0][1], r0 + c1);
        reg_self[1][0] = min(reg_self[1][0], r1 + c0);
        reg_self[1][1] = min(reg_self[1][1], r1 + c1);
    }

    d_D[self_start + ty * stride + tx] = reg_self[0][0];
    d_D[self_start + ty * stride + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * stride + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)] = reg_self[1][1];
}
