#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* -------------------------------------------------------- */
/* 資料結構與常數                                           */
/* -------------------------------------------------------- */

typedef struct {
    double x, y, z;    /* position */
    double vx, vy, vz; /* velocity */
    double m;          /* mass */
} Body;

#define G 1.0
#define SOFTENING 1e-3
#define BLOCK_SIZE 256  // 必須與 Kernel Launch 的 blockDim 保持一致

/* -------------------------------------------------------- */
/* Device Functions (物理邏輯層 - 保持不變)                 */
/* -------------------------------------------------------- */

__device__ void accumulate_interaction(double xi, double yi, double zi, Body other, double &axi, double &ayi, double &azi) {
    double dx = other.x - xi;
    double dy = other.y - yi;
    double dz = other.z - zi;

    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;

    double invDist = 1.0 / sqrt(distSqr);
    double invDist3 = invDist * invDist * invDist;

    double s = G * other.m * invDist3;

    axi += dx * s;
    ayi += dy * s;
    azi += dz * s;
}

__device__ void update_velocity(Body &b, double ax, double ay, double az, double scale) {
    b.vx += ax * scale;
    b.vy += ay * scale;
    b.vz += az * scale;
}

__device__ void update_position(Body &b, double dt) {
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    b.z += b.vz * dt;
}

/* -------------------------------------------------------- */
/* GPU Kernels (Shared Memory 優化版)                       */
/* -------------------------------------------------------- */

/* * Tiled Calculation Kernel
 * 每次將 BLOCK_SIZE 個粒子載入 Shared Memory，重複使用以節省頻寬
 */
__global__ void compute_acceleration_kernel(const Body *bodies, int N, double *ax, double *ay, double *az) {
    // 宣告 Shared Memory 緩衝區
    __shared__ Body cache[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 預先讀取「本體」的位置到 Register (避免重複訪問 Global/Shared Memory)
    double xi = 0, yi = 0, zi = 0;
    if (i < N) {
        xi = bodies[i].x;
        yi = bodies[i].y;
        zi = bodies[i].z;
    }

    double val_ax = 0.0;
    double val_ay = 0.0;
    double val_az = 0.0;

    // 計算 Grid 中有多少個 Tile (區塊)
    int numTiles = (N + blockDim.x - 1) / blockDim.x;

    // --- Tiling Loop: 遍歷每一個區塊 ---
    for (int tile = 0; tile < numTiles; ++tile) {

        // 1. 協同載入 (Cooperative Loading)
        // 每個 Thread 負責載入此 Tile 中的對應粒子
        int load_idx = tile * blockDim.x + threadIdx.x;

        if (load_idx < N) {
            cache[threadIdx.x] = bodies[load_idx];
        } else {
            // 邊界檢查：如果超出範圍，載入一個「幽靈粒子」
            // 質量設為 0，這樣計算引力時貢獻為 0，不影響結果
            Body dummy = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            cache[threadIdx.x] = dummy;
        }

        // 等待所有 Thread 完成載入
        __syncthreads();

        // 2. 計算與此 Tile 中所有粒子的交互作用 (只在 Shared Memory 運算)
        // 只有當本體 i 是有效粒子時才計算
        if (i < N) {
// 這裡為了效能優化 (Loop Unrolling)，通常編譯器會自動處理
// 我們直接遍歷 cache 裡的所有粒子
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                accumulate_interaction(xi, yi, zi, cache[j], val_ax, val_ay, val_az);
            }
        }

        // 等待所有 Thread 計算完畢，才能進入下一輪載入
        __syncthreads();
    }

    // 3. 寫回結果
    if (i < N) {
        ax[i] = val_ax;
        ay[i] = val_ay;
        az[i] = val_az;
    }
}

/* Leapfrog 積分 Kernel (無需 Shared Memory，維持原樣) */
__global__ void update_phase1_kernel(Body *bodies, int N, double dt, const double *ax, const double *ay, const double *az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Body b = bodies[i];
    update_velocity(b, ax[i], ay[i], az[i], 0.5 * dt);
    update_position(b, dt);
    bodies[i] = b;
}

__global__ void update_phase2_kernel(Body *bodies, int N, double dt, const double *ax, const double *ay, const double *az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    Body b = bodies[i];
    update_velocity(b, ax[i], ay[i], az[i], 0.5 * dt);
    bodies[i] = b;
}

/* -------------------------------------------------------- */
/* Host Helper Functions                                    */
/* -------------------------------------------------------- */

// double getTimeStamp() {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return (double)tv.tv_usec/1000000 + tv.tv_sec;
// }

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out) {
    FILE *fin = fopen(filename, "r");
    if (!fin) return 0;

    if (fscanf(fin, "%d", N) != 1) {
        fclose(fin);
        return 0;
    }
    if (fscanf(fin, "%lf %lf %d", total_time, dt, dump_interval) != 3) {
        fclose(fin);
        return 0;
    }

    Body *bodies = (Body *)malloc((*N) * sizeof(Body));
    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &bodies[i].x, &bodies[i].y, &bodies[i].z, &bodies[i].vx, &bodies[i].vy, &bodies[i].vz, &bodies[i].m) != 7) {
            free(bodies);
            fclose(fin);
            return 0;
        }
    }
    fclose(fin);
    *bodies_out = bodies;
    return 1;
}

void dump_traj_step(FILE *ftraj, int step, double t, int N, const Body *bodies) {
    for (int i = 0; i < N; ++i) {
        fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].vx, bodies[i].vy, bodies[i].vz, bodies[i].m);
    }
}

/* -------------------------------------------------------- */
/* Main Function                                            */
/* -------------------------------------------------------- */

int main(int argc, char **argv) {
    clock_t start = clock();
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <traj_output_csv>\n", argv[0]);
        return 1;
    }

    int N, dump_interval;
    double total_time, dt;
    Body *h_bodies = NULL;

    if (!read_input(argv[1], &N, &total_time, &dt, &dump_interval, &h_bodies)) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }

    int steps = (int)(total_time / dt);

    // Allocate Device Memory
    Body *d_bodies;
    double *d_ax, *d_ay, *d_az;
    cudaMalloc((void **)&d_bodies, N * sizeof(Body));
    cudaMalloc((void **)&d_ax, N * sizeof(double));
    cudaMalloc((void **)&d_ay, N * sizeof(double));
    cudaMalloc((void **)&d_az, N * sizeof(double));

    cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    FILE *ftraj = fopen(argv[2], "w");
    if (ftraj) fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");

    // printf("Running Shared Memory Optimized Simulation: N=%d, Steps=%d\n", N, steps);
    // double start = getTimeStamp();

    // CUDA Configuration
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double t = 0.0;

    // --- Initial Step ---
    compute_acceleration_kernel<<<gridSize, BLOCK_SIZE>>>(d_bodies, N, d_ax, d_ay, d_az);
    cudaDeviceSynchronize();

    cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
    if (ftraj) dump_traj_step(ftraj, 0, t, N, h_bodies);

    // --- Main Loop ---
    for (int s = 1; s <= steps; ++s) {
        // Phase 1: Update v(1/2) and x(1)
        update_phase1_kernel<<<gridSize, BLOCK_SIZE>>>(d_bodies, N, dt, d_ax, d_ay, d_az);

        // Phase 2: Compute a(1) (使用優化的 Shared Mem Kernel)
        compute_acceleration_kernel<<<gridSize, BLOCK_SIZE>>>(d_bodies, N, d_ax, d_ay, d_az);

        // Phase 3: Update v(1)
        update_phase2_kernel<<<gridSize, BLOCK_SIZE>>>(d_bodies, N, dt, d_ax, d_ay, d_az);

        t += dt;

        if (s % dump_interval == 0 && ftraj) {
            cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
            dump_traj_step(ftraj, s, t, N, h_bodies);
        }
    }

    cudaDeviceSynchronize();

    if (ftraj) fclose(ftraj);
    free(h_bodies);
    cudaFree(d_bodies);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);

    // double end = getTimeStamp();
    // printf("Total Execution Time: %.3f seconds\n", end - start);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "ELAPSED_TIME: %.6f\n", elapsed);
    return 0;
}