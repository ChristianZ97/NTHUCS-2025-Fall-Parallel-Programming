// hw3-3.cu

/* Headers */

#include <cuda.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Constants & Global Variables */

#define BLOCKING_FACTOR 64
#define HALF_BLOCK 32
#define INF ((1 << 30) - 1)

static int *D;        // Host pointer
static int *d_D[2];   // Device pointers for 2 GPUs
static int V, E;      // Original vertices, edges
static int V_padded;  // Padded vertices (multiple of 64)

cudaStream_t stream_main[2], stream_row[2], stream_col[2];
cudaEvent_t event_p1_done[2], event_p2_row_done[2], event_p2_col_done[2];

/* Function Prototypes */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded);
__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit);
__global__ void kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset);

void input(char *infile);
void output(char *outfile);

void block_FW(const int dev_id);

/* Main */

int main(int argc, char *argv[]) {

    /*
        if (argc != 3) {
            printf("Usage: %s <input> <output>\n", argv[0]);
            return 1;
        }
    */

    input(argv[1]);

    // Use size_t for total size calculation
    size_t size = (size_t)V_padded * V_padded * sizeof(int);

    // Enable peer access between 2 GPUs
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

#pragma omp parallel num_threads(2)
    {
        const int dev_id = omp_get_thread_num();

        cudaSetDevice(dev_id);

        cudaStreamCreate(&stream_main[dev_id]);
        cudaStreamCreate(&stream_row[dev_id]);
        cudaStreamCreate(&stream_col[dev_id]);

        cudaEventCreate(&event_p1_done[dev_id]);
        cudaEventCreate(&event_p2_row_done[dev_id]);
        cudaEventCreate(&event_p2_col_done[dev_id]);

        cudaMalloc(&d_D[dev_id], size);

        // H2D: copy full matrix to each GPU
        cudaMemcpy(d_D[dev_id], D, size, cudaMemcpyHostToDevice);

#pragma omp barrier

        block_FW(dev_id);

        cudaDeviceSynchronize();

        const int round = V_padded / BLOCKING_FACTOR;
        const int row_mid = round / 2;
        const int row_block_start = (dev_id == 0) ? 0 : row_mid;
        const int row_block_count = (dev_id == 0) ? row_mid : (round - row_mid);
        const int row_start = row_block_start * BLOCKING_FACTOR;
        const int row_num = row_block_count * BLOCKING_FACTOR;

        // Copy back only the owned rows from each GPU
        cudaMemcpy(D + (size_t)row_start * V_padded, d_D[dev_id] + (size_t)row_start * V_padded, (size_t)row_num * V_padded * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_D[dev_id]);

        cudaStreamDestroy(stream_main[dev_id]);
        cudaStreamDestroy(stream_row[dev_id]);
        cudaStreamDestroy(stream_col[dev_id]);

        cudaEventDestroy(event_p1_done[dev_id]);
        cudaEventDestroy(event_p2_row_done[dev_id]);
        cudaEventDestroy(event_p2_col_done[dev_id]);
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

    dim3 grid_p2_row(round, 1);
    dim3 grid_p2_col(num_row_blocks, 1);
    dim3 grid_p3(round, num_row_blocks);

    for (int r = 0; r < round; ++r) {
        const int owner_id = (r < row_mid) ? 0 : 1;

        // Phase 1 + Phase 2 row on owner device
        if (dev_id == owner_id) {
            // Phase 1: Pivot block
            kernel_phase1<<<1, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded);
            cudaEventRecord(event_p1_done[dev_id], stream_main[dev_id]);

            cudaStreamWaitEvent(stream_row[dev_id], event_p1_done[dev_id], 0);

            // Phase 2: Pivot row
            kernel_phase2_row<<<grid_p2_row, threads_per_block, 0, stream_row[dev_id]>>>(d_D[dev_id], r, V_padded);
            cudaEventRecord(event_p2_row_done[dev_id], stream_row[dev_id]);

            // Send updated pivot row to peer device
            size_t copy_size = (size_t)BLOCKING_FACTOR * V_padded * sizeof(int);
            size_t offset = (size_t)r * BLOCKING_FACTOR * V_padded;
            const int peer_id = 1 - dev_id;

            cudaMemcpyPeer(d_D[peer_id] + offset, peer_id, d_D[dev_id] + offset, dev_id, copy_size);
        }

#pragma omp barrier

        // Phase 2 col + Phase 3 on both devices, using updated pivot row
        cudaStreamWaitEvent(stream_main[dev_id], event_p2_row_done[dev_id], 0);

        kernel_phase2_col<<<grid_p2_col, threads_per_block, 0, stream_col[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block, num_row_blocks);
        cudaEventRecord(event_p2_col_done[dev_id], stream_col[dev_id]);

        cudaStreamWaitEvent(stream_main[dev_id], event_p2_col_done[dev_id], 0);

        kernel_phase3<<<grid_p3, threads_per_block, 0, stream_main[dev_id]>>>(d_D[dev_id], r, V_padded, start_row_block);

        cudaDeviceSynchronize();

#pragma omp barrier
    }
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    // Calculate padded size (round up to multiple of 64)
    V_padded = (V + BLOCKING_FACTOR - 1) / BLOCKING_FACTOR * BLOCKING_FACTOR;

    // Use pinned memory for faster host-device transfer
    cudaHostAlloc(&D, (size_t)V_padded * V_padded * sizeof(int), cudaHostAllocDefault);

    // Initialize with INF (and 0 diagonal), including padding
    for (int i = 0; i < V_padded; ++i)
        for (int j = 0; j < V_padded; ++j)
            D[(size_t)i * V_padded + j] = (i == j) ? 0 : INF;

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[(size_t)pair[0] * V_padded + pair[1]] = pair[2];
    }

    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "wb");
    // Write only the valid part (V x V), skipping padding
    for (int i = 0; i < V; ++i)
        fwrite(&D[(size_t)i * V_padded], sizeof(int), V, f);
    fclose(f);

    cudaFreeHost(D);  // Free pinned memory
}

/* Kernels */

__global__ void kernel_phase1(int *d_D, const int r, const int V_padded) {
    const int tx = threadIdx.x;  // 0..31
    const int ty = threadIdx.y;  // 0..31

    // Shared Memory for the 64x64 block
    // [OPT] Shared memory tile + extra column (+1) to reduce global memory traffic and avoid shared memory bank conflicts.
    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];

    // Global Memory Offset for the Pivot Block (r, r)
    const int b_start = (r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);

    // 1. Load Global -> Shared (Each thread loads 4 ints)
    // [OPT] Coalesced global memory loads: threads in a warp read contiguous elements of each row in the 64x64 tile.
    // Access pattern: Top-Left, Top-Right, Bottom-Left, Bottom-Right relative to thread
    sm_pivot[ty][tx] = d_D[b_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];
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
    d_D[b_start + ty * V_padded + tx] = sm_pivot[ty][tx];
    d_D[b_start + ty * V_padded + (tx + HALF_BLOCK)] = sm_pivot[ty][tx + HALF_BLOCK];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + tx] = sm_pivot[ty + HALF_BLOCK][tx];
    d_D[b_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK];
}

__global__ void kernel_phase2_row(int *d_D, const int r, const int V_padded) {
    const int b_idx_x = blockIdx.x;
    if (b_idx_x == r) return;  // Skip pivot block itself

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int b_idx_y = r;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const size_t pivot_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Load pivot
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load self
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

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
        // no syncthreads needed here
    }

    // Write back
    d_D[self_start + ty * V_padded + tx] = reg_self[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = reg_self[1][1];
}

__global__ void kernel_phase2_col(int *d_D, const int r, const int V_padded, const int row_offset, const int row_limit) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int b_idx_y = blockIdx.x + row_offset;
    if (b_idx_y == r || b_idx_y >= row_offset + row_limit) return;  // Skip pivot block and out-of-range blocks

    const int b_idx_x = r;

    __shared__ int sm_pivot[BLOCKING_FACTOR][BLOCKING_FACTOR];
    __shared__ int sm_self[BLOCKING_FACTOR][BLOCKING_FACTOR];

    const size_t pivot_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Load pivot (r, r)
    sm_pivot[ty][tx] = d_D[pivot_start + ty * V_padded + tx];
    sm_pivot[ty][tx + HALF_BLOCK] = d_D[pivot_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_pivot[ty + HALF_BLOCK][tx] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_pivot[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[pivot_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load self (block (y, r))
    sm_self[ty][tx] = d_D[self_start + ty * V_padded + tx];
    sm_self[ty][tx + HALF_BLOCK] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_self[ty + HALF_BLOCK][tx] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_self[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

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
        // no syncthreads needed here
    }

    // Write back
    d_D[self_start + ty * V_padded + tx] = reg_self[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = reg_self[1][1];
}

__global__ void kernel_phase3(int *d_D, const int r, const int V_padded, const int row_offset) {
    const int b_idx_x = blockIdx.x;
    const int b_idx_y = blockIdx.y + row_offset;

    if (b_idx_x == r || b_idx_y == r) return;  // Skip Phase 1 & 2 blocks

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ int sm_row[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Row block (y, r)
    __shared__ int sm_col[BLOCKING_FACTOR][BLOCKING_FACTOR];  // Col block (r, x)

    const size_t row_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (r * BLOCKING_FACTOR);
    const size_t col_start = (size_t)(r * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);
    const size_t self_start = (size_t)(b_idx_y * BLOCKING_FACTOR) * V_padded + (b_idx_x * BLOCKING_FACTOR);

    // Load row block
    sm_row[ty][tx] = d_D[row_start + ty * V_padded + tx];
    sm_row[ty][tx + HALF_BLOCK] = d_D[row_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_row[ty + HALF_BLOCK][tx] = d_D[row_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_row[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[row_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    // Load col block
    sm_col[ty][tx] = d_D[col_start + ty * V_padded + tx];
    sm_col[ty][tx + HALF_BLOCK] = d_D[col_start + ty * V_padded + (tx + HALF_BLOCK)];
    sm_col[ty + HALF_BLOCK][tx] = d_D[col_start + (ty + HALF_BLOCK) * V_padded + tx];
    sm_col[ty + HALF_BLOCK][tx + HALF_BLOCK] = d_D[col_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

    __syncthreads();

    int reg_self[2][2];
    reg_self[0][0] = d_D[self_start + ty * V_padded + tx];
    reg_self[0][1] = d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)];
    reg_self[1][0] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx];
    reg_self[1][1] = d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)];

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

    // Write back
    d_D[self_start + ty * V_padded + tx] = reg_self[0][0];
    d_D[self_start + ty * V_padded + (tx + HALF_BLOCK)] = reg_self[0][1];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + tx] = reg_self[1][0];
    d_D[self_start + (ty + HALF_BLOCK) * V_padded + (tx + HALF_BLOCK)] = reg_self[1][1];
}
