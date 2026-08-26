// hw3-3.hip

/* Headers */
#include <hip/hip_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "io_utils.h"

/* Constants & Global Variables */
#define BLOCKING_FACTOR 64
#define HALF_BLOCK 32
#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D[2];   // Device pointers for 2 GPUs
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)
static size_t matrix_size;

hipStream_t stream_main[2], stream_row[2], stream_col[2];
hipEvent_t event_p1_done[2], event_p2_row_done[2], event_p2_col_done[2];

/* Helper Functions for int4 Optimization (From hw3-2) */
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

/* Function Prototypes */
__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit);
__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset);

void input(char *infile);
void output(char *outfile);
void block_FW(const int dev_id);

/* Main */
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    input(argv[1]);

    const size_t size = matrix_size;

    // Enable peer access between 2 GPUs
    // Assuming 2 GPUs are available as per hw3-3 template
    int n_devices;
    hipGetDeviceCount(&n_devices);
    if (n_devices > 1) {
        hipSetDevice(0);
        hipDeviceEnablePeerAccess(1, 0);
        hipSetDevice(1);
        hipDeviceEnablePeerAccess(0, 0);
    }

#pragma omp parallel num_threads(2)
    {
        const int dev_id = omp_get_thread_num();
        hipSetDevice(dev_id);

        hipStreamCreate(&stream_main[dev_id]);
        hipStreamCreate(&stream_row[dev_id]);
        hipStreamCreate(&stream_col[dev_id]);

        hipEventCreate(&event_p1_done[dev_id]);
        hipEventCreate(&event_p2_row_done[dev_id]);
        hipEventCreate(&event_p2_col_done[dev_id]);

        hipMalloc(&d_D[dev_id], size);

        // Initialize each GPU with a full matrix copy.
        hipMemcpy(d_D[dev_id], D, size, hipMemcpyHostToDevice);

#pragma omp barrier

        block_FW(dev_id);

        hipDeviceSynchronize();

        const int round = V_padded / BLOCKING_FACTOR;
        const int row_mid = round / 2;
        const int row_block_start = (dev_id == 0) ? 0 : row_mid;
        const int row_block_count = (dev_id == 0) ? row_mid : (round - row_mid);

        const int row_start = row_block_start * BLOCKING_FACTOR;
        const int row_num = row_block_count * BLOCKING_FACTOR;

        // Copy back only the owned rows from each GPU
        if (row_num > 0) {
            hipMemcpy(D + (size_t)row_start * V_padded, d_D[dev_id] + (size_t)row_start * V_padded, (size_t)row_num * V_padded * sizeof(int), hipMemcpyDeviceToHost);
        }

        hipFree(d_D[dev_id]);

        hipStreamDestroy(stream_main[dev_id]);
        hipStreamDestroy(stream_row[dev_id]);
        hipStreamDestroy(stream_col[dev_id]);

        hipEventDestroy(event_p1_done[dev_id]);
        hipEventDestroy(event_p2_row_done[dev_id]);
        hipEventDestroy(event_p2_col_done[dev_id]);
    }  // end parallel region

    output(argv[2]);
    return 0;
}

/* Function Definitions */

void block_FW(const int dev_id) {
    const int round = V_padded / BLOCKING_FACTOR;
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);

    const int row_mid = round / 2;
    const int start_row_block = (dev_id == 0) ? 0 : row_mid;
    const int num_row_blocks = (dev_id == 0) ? row_mid : (round - row_mid);

    // Grids
    dim3 grid_p2_row(round, 1);
    dim3 grid_p2_col(num_row_blocks, 1);
    dim3 grid_p3(round, num_row_blocks);

    for (int r = 0; r < round; ++r) {
        const int owner_id = (r < row_mid) ? 0 : 1;

        // Phase 1 + Phase 2 row on owner device
        if (dev_id == owner_id) {
            // Phase 1: Pivot block
            kernel_phase1<<<1, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded);
            hipEventRecord(event_p1_done[dev_id], stream_main[dev_id]);

            hipStreamWaitEvent(stream_row[dev_id], event_p1_done[dev_id], 0);

            // Phase 2: Pivot row
            kernel_phase2_row<<<grid_p2_row, threads_per_block, 0, stream_row[dev_id]>>>(d_D[dev_id], r, V_padded);
            hipEventRecord(event_p2_row_done[dev_id], stream_row[dev_id]);

            // Send updated pivot row to peer device
            size_t copy_size = (size_t)BLOCKING_FACTOR * V_padded * sizeof(int);
            size_t offset = (size_t)r * BLOCKING_FACTOR * V_padded;
            const int peer_id = 1 - dev_id;

            // Only sync P2P copy if 2 GPUs exist
            hipMemcpyPeer(d_D[peer_id] + offset, peer_id, d_D[dev_id] + offset, dev_id, copy_size);
        }

#pragma omp barrier

        // The synchronous peer copy and host barrier make the pivot row visible
        // before either device launches the remaining phases.

        if (dev_id == owner_id) {
            hipStreamWaitEvent(stream_main[dev_id], event_p2_row_done[dev_id], 0);
        }
        // The peer uses the row copied before the barrier.

        if (num_row_blocks > 0) {
            kernel_phase2_col<<<grid_p2_col, threads_per_block, 0, stream_col[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block, num_row_blocks);
            hipEventRecord(event_p2_col_done[dev_id], stream_col[dev_id]);

            hipStreamWaitEvent(stream_main[dev_id], event_p2_col_done[dev_id], 0);

            kernel_phase3<<<grid_p3, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block);
        }
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

/* Kernels - Logic from hw3-2, Adapted for Multi-GPU (offsets) */

__global__ void __launch_bounds__(1024) kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t stride = (size_t)V_padded;

    // Shared Memory with padding (+1) from hw3-2
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

    const size_t b_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;

    // Use vectorized load
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
    if (b_idx_x == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_y = r;  // Fixed row for this phase
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
    }

    sm_self[ty][tx] = reg_self[0][0];
    sm_self[ty][tx + HALF_BLOCK] = reg_self[0][1];
    sm_self[ty + HALF_BLOCK][tx] = reg_self[1][0];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = reg_self[1][1];

    __syncthreads();
    store_block_int4(d_D + self_start, stride, sm_self);
}

__global__ void __launch_bounds__(1024) kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit) {
    const int b_idx_y = blockIdx.x + row_offset;  // Apply Offset
    if (b_idx_y == r) return;
    // Safety check for multi-gpu bounds (though launch grid should be exact)
    if (b_idx_y >= row_offset + row_limit) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_x = r;  // Fixed col for this phase
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
    }

    sm_self[ty][tx] = reg_self[0][0];
    sm_self[ty][tx + HALF_BLOCK] = reg_self[0][1];
    sm_self[ty + HALF_BLOCK][tx] = reg_self[1][0];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = reg_self[1][1];

    __syncthreads();
    store_block_int4(d_D + self_start, stride, sm_self);
}

__global__ void __launch_bounds__(1024) kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset) {
    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y + row_offset;  // Apply Offset

    if (b_idx_x == r || b_idx_y == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR + 1];

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
    // Direct global load for self block as in hw3-2 logic (register blocking)
    // Note: hw3-2 Phase 3 loaded self into registers directly, not shared memory
    reg_self[0][0] = d_D[self_start + (size_t)ty * stride + tx];
    reg_self[0][1] =
        d_D[self_start + (size_t)ty * stride + (tx + HALF_BLOCK)];
    reg_self[1][0] =
        d_D[self_start + (size_t)(ty + HALF_BLOCK) * stride + tx];
    reg_self[1][1] =
        d_D[self_start + (size_t)(ty + HALF_BLOCK) * stride +
            (tx + HALF_BLOCK)];

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

    d_D[self_start + (size_t)ty * stride + tx] = reg_self[0][0];
    d_D[self_start + (size_t)ty * stride + (tx + HALF_BLOCK)] =
        reg_self[0][1];
    d_D[self_start + (size_t)(ty + HALF_BLOCK) * stride + tx] =
        reg_self[1][0];
    d_D[self_start + (size_t)(ty + HALF_BLOCK) * stride +
        (tx + HALF_BLOCK)] = reg_self[1][1];
}
