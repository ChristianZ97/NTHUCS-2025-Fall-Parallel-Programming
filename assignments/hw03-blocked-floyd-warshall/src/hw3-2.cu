// hw3-2.cu

/* Headers*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "io_utils.h"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(error));          \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/*
 * Blocked Floyd–Warshall with CUDA
 */

/* Constants & Global Variables */
// [OPT] Blocking factor (tile size). Tune this (e.g., 32/64/128) to study "large blocking factor" vs occupancy & bandwidth.
#define BLOCKING_FACTOR 64  // Matches v2 (64x64 data block)

// [OPT] Thread block dimension (32x32 = 1024 threads). This controls CUDA 2D alignment and occupancy.
#define HALF_BLOCK BLOCKING_FACTOR / 2  // Thread block dimension (32x32 threads)

#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D;      // Device pointer
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)
static size_t matrix_size;

// [OPT] Multiple streams & events are used to overlap different phases (streaming / reduce idle time).
cudaStream_t stream_main, stream_row, stream_col;
cudaEvent_t event_p1_done, event_p2_row_done, event_p2_col_done;

/* Function Prototypes */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded);

void input(char *infile);
void output(char *outfile);
void block_FW();

/* Main */

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    input(argv[1]);

    CUDA_CHECK(cudaStreamCreate(&stream_main));
    CUDA_CHECK(cudaStreamCreate(&stream_row));
    CUDA_CHECK(cudaStreamCreate(&stream_col));
    CUDA_CHECK(cudaEventCreate(&event_p1_done));
    CUDA_CHECK(cudaEventCreate(&event_p2_row_done));
    CUDA_CHECK(cudaEventCreate(&event_p2_col_done));

    const size_t size = matrix_size;
    CUDA_CHECK(cudaMalloc(&d_D, size));

    // -------------------------
    // H2D transfer
    // -------------------------
    CUDA_CHECK(cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice));

    // -------------------------
    // Kernel execution
    // -------------------------
    block_FW();

    // -------------------------
    // D2H transfer
    // -------------------------
    CUDA_CHECK(cudaMemcpy(D, d_D, size, cudaMemcpyDeviceToHost));

    output(argv[2]);

    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaStreamDestroy(stream_main));
    CUDA_CHECK(cudaStreamDestroy(stream_row));
    CUDA_CHECK(cudaStreamDestroy(stream_col));
    CUDA_CHECK(cudaEventDestroy(event_p1_done));
    CUDA_CHECK(cudaEventDestroy(event_p2_row_done));
    CUDA_CHECK(cudaEventDestroy(event_p2_col_done));

    return 0;
}

/* Function Definitions */
void block_FW() {

    const int round = V_padded / BLOCKING_FACTOR;

    // [OPT] 2D thread block (HALF_BLOCK x HALF_BLOCK) for good CUDA 2D alignment and coalesced accesses.
    dim3 threads_per_block(HALF_BLOCK, HALF_BLOCK);  // 32x32 threads

    for (int r = 0; r < round; ++r) {
        // 1. Phase 1: Pivot Block
        kernel_phase1<<<1, threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(event_p1_done, stream_main);

        // 2. Phase 2: Pivot Row (row stream)
        cudaStreamWaitEvent(stream_row, event_p1_done, 0);
        kernel_phase2_row<<<round, threads_per_block, 0, stream_row>>>(d_D, r, V_padded);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(event_p2_row_done, stream_row);

        // 2. Phase 2: Pivot Col (col stream)
        cudaStreamWaitEvent(stream_col, event_p1_done, 0);
        kernel_phase2_col<<<round, threads_per_block, 0, stream_col>>>(d_D, r, V_padded);
        CUDA_CHECK(cudaGetLastError());
        cudaEventRecord(event_p2_col_done, stream_col);

        cudaStreamWaitEvent(stream_main, event_p2_row_done, 0);
        cudaStreamWaitEvent(stream_main, event_p2_col_done, 0);

        // 3. Phase 3: Independent Blocks
        // (round, round) blocks per grid
        kernel_phase3<<<dim3(round, round), threads_per_block, 0, stream_main>>>(d_D, r, V_padded);
        CUDA_CHECK(cudaGetLastError());
    }
}

void input(char *infile) {
    FILE *file = open_file_or_die(infile, "rb");
    read_exact_or_die(&V, sizeof(V), 1, file, infile, "vertex count");
    read_exact_or_die(&E, sizeof(E), 1, file, infile, "edge count");
    validate_graph_header_or_die(V, E, 0, infile);

    // Calculate Padded Size (Round up to multiple of 64)
    // [OPT] Padding V up to a multiple of BLOCKING_FACTOR simplifies index math and improves coalesced access.
    V_padded = padded_width_or_die(V, BLOCKING_FACTOR, infile);
    matrix_size = matrix_bytes_or_die(V_padded, infile);

    // Use Pinned Memory for faster host-device transfer
    // [OPT] Pinned host memory (cudaHostAlloc) increases H2D/D2H bandwidth.
    CUDA_CHECK(cudaHostAlloc(&D, matrix_size, cudaHostAllocDefault));

    // Initialize with INF (and 0 diagonal)
    // Note: Padding areas are also initialized to avoid side effects

#pragma unroll 32

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

    // Write only the valid part (V x V), skipping padding
    for (int i = 0; i < V; ++i)
        write_exact_or_die(
            &D[(size_t)i * V_padded], sizeof(int), (size_t)V, f, outfile);

    close_file_or_die(f, outfile);
    CUDA_CHECK(cudaFreeHost(D));
}

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31
    const size_t stride = (size_t)V_padded;

    // Shared Memory for the 64x64 block
    // [OPT] Shared memory tile + extra column (+1) to reduce global memory traffic and avoid shared memory bank conflicts.
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];

    // Global Memory Offset for the Pivot Block (r, r)
    const size_t b_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;

    // 1. Load Global -> Shared (Each thread loads 4 ints)
    // [OPT] Coalesced global memory loads: threads in a warp read contiguous elements of each row in the 64x64 tile.
    // Access pattern: Top-Left, Top-Right, Bottom-Left, Bottom-Right relative to thread
    sm_pivot[ty][tx] = d_D[b_start + ty * stride + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[b_start + ty * stride + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[b_start + (ty + HALF_BLOCK) * stride + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[b_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
    __syncthreads();

    // 2. Floyd-Warshall Computation within the block

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

    // 3. Write Shared -> Global
    d_D[b_start + ty * stride + tx] = sm_pivot[ty][tx];
    d_D[b_start + ty * stride + (tx + HALF_BLOCK)] = sm_pivot[ty][tx + HALF_BLOCK];
    d_D[b_start + (ty + HALF_BLOCK) * stride + tx] = sm_pivot[ty + HALF_BLOCK][tx];
    d_D[b_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)] = sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded) {

    const int b_idx_x = blockIdx.x;
    if (b_idx_x == r) return;  // Skip the pivot block itself (handled in Phase 1)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_y = r;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const size_t pivot_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;
    const size_t self_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;

    // Pivot
    sm_pivot[ty][tx] = d_D[pivot_start + ty * stride + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * stride + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * stride + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
    // Self
    sm_self[ty][tx] = d_D[self_start + ty * stride + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * stride + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * stride + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
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

    d_D[self_start + ty * stride + tx] = reg_self[0][0];
    d_D[self_start + ty * stride + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * stride + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)] = reg_self[1][1];
}

__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded) {

    const int b_idx_y = blockIdx.x;
    if (b_idx_y == r) return;  // Skip the pivot block itself (handled in Phase 1)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_x = r;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const size_t pivot_start =
        (size_t)r * BLOCKING_FACTOR * stride + (size_t)r * BLOCKING_FACTOR;
    const size_t self_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;

    // Pivot
    sm_pivot[ty][tx] = d_D[pivot_start + ty * stride + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * stride + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * stride + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
    // Self
    sm_self[ty][tx] = d_D[self_start + ty * stride + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * stride + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * stride + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
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

    d_D[self_start + ty * stride + tx] = reg_self[0][0];
    d_D[self_start + ty * stride + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * stride + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)] = reg_self[1][1];
}

__global__ void kernel_phase3(int *d_D, const int r, const int V_padded) {

    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y;
    if (b_idx_x == r || b_idx_y == r) return;  // Skip Phase 1 & 2 blocks

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t stride = (size_t)V_padded;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Row Block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Col Block (r, x)

    const size_t row_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)r * BLOCKING_FACTOR;
    const size_t col_start =
        (size_t)r * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;
    const size_t self_start =
        (size_t)b_idx_y * BLOCKING_FACTOR * stride +
        (size_t)b_idx_x * BLOCKING_FACTOR;

    // Row Block
    sm_row[ty][tx] = d_D[row_start + ty * stride + tx];
    sm_row[ty][tx + HALF_BLOCK] = d_D[row_start + ty * stride + (tx + HALF_BLOCK)];
    sm_row[ty + HALF_BLOCK][tx] = d_D[row_start + (ty + HALF_BLOCK) * stride + tx];
    sm_row[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[row_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
    // Col Block
    sm_col[ty][tx] = d_D[col_start + ty * stride + tx];
    sm_col[ty][tx + HALF_BLOCK] = d_D[col_start + ty * stride + (tx + HALF_BLOCK)];
    sm_col[ty + HALF_BLOCK][tx] = d_D[col_start + (ty + HALF_BLOCK) * stride + tx];
    sm_col[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[col_start + (ty + HALF_BLOCK) * stride + (tx + HALF_BLOCK)];
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
