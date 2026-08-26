#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* -------------------------------------------------------- */
/* 設定與常數                                               */
/* -------------------------------------------------------- */

#define BLOCK_SIZE 256
#define SOFTENING 1e-3
#define G 1.0

/* -------------------------------------------------------- */
/* Device Functions: 物理計算                               */
/* -------------------------------------------------------- */

__device__ void bodyBodyInteraction(double xi, double yi, double zi, double xj, double yj, double zj, double mj, double &ai_x, double &ai_y, double &ai_z) {
    double rx = xj - xi;
    double ry = yj - yi;
    double rz = zj - zi;

    double distSqr = rx * rx + ry * ry + rz * rz + SOFTENING;
    double distInv = 1.0 / sqrt(distSqr);
    double distInv3 = distInv * distInv * distInv;

    double s = mj * distInv3 * G;

    ai_x += rx * s;
    ai_y += ry * s;
    ai_z += rz * s;
}

/* -------------------------------------------------------- */
/* GPU Kernels: Leapfrog SoA Optimized                      */
/* -------------------------------------------------------- */

/* * Kernel 1: 更新位置 + 半步速度 (Phase 1)
 * v(t+0.5) = v(t) + 0.5 * a(t) * dt
 * x(t+1)   = x(t) + v(t+0.5) * dt
 */
__global__ void leapfrog_step1_soa(int N,
                                   double dt,
                                   double *__restrict__ pos_x,
                                   double *__restrict__ pos_y,
                                   double *__restrict__ pos_z,
                                   double *__restrict__ vel_x,
                                   double *__restrict__ vel_y,
                                   double *__restrict__ vel_z,
                                   const double *__restrict__ acc_x,
                                   const double *__restrict__ acc_y,
                                   const double *__restrict__ acc_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double vx = vel_x[i] + 0.5 * acc_x[i] * dt;
    double vy = vel_y[i] + 0.5 * acc_y[i] * dt;
    double vz = vel_z[i] + 0.5 * acc_z[i] * dt;

    pos_x[i] += vx * dt;
    pos_y[i] += vy * dt;
    pos_z[i] += vz * dt;

    // 寫回半步速度，供下一步使用
    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;
}

/* * Kernel 2: 計算新力 + 更新後半步速度 (Compute + Phase 2)
 * a(t+1) = Force(x(t+1))
 * v(t+1) = v(t+0.5) + 0.5 * a(t+1) * dt
 *
 * 這裡融合了 "計算力" 和 "更新速度"，省下一個 Kernel
 */
__global__ void leapfrog_step2_soa(int N,
                                   double dt,
                                   // 讀取位置 (x, y, z, m)
                                   const double *__restrict__ pos_x,
                                   const double *__restrict__ pos_y,
                                   const double *__restrict__ pos_z,
                                   const double *__restrict__ mass,
                                   // 更新速度、寫入加速度
                                   double *__restrict__ vel_x,
                                   double *__restrict__ vel_y,
                                   double *__restrict__ vel_z,
                                   double *__restrict__ acc_x,
                                   double *__restrict__ acc_y,
                                   double *__restrict__ acc_z) {
    // Shared Memory Tiling
    __shared__ double sh_x[BLOCK_SIZE];
    __shared__ double sh_y[BLOCK_SIZE];
    __shared__ double sh_z[BLOCK_SIZE];
    __shared__ double sh_m[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double my_x, my_y, my_z;
    if (i < N) {
        my_x = pos_x[i];
        my_y = pos_y[i];
        my_z = pos_z[i];
    }

    double new_ax = 0.0, new_ay = 0.0, new_az = 0.0;

    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 遍歷所有節點
    for (int tile = 0; tile < numTiles; tile++) {
        int idx = tile * BLOCK_SIZE + threadIdx.x;

        // 放入一個tile
        if (idx < N) {
            sh_x[threadIdx.x] = pos_x[idx];
            sh_y[threadIdx.x] = pos_y[idx];
            sh_z[threadIdx.x] = pos_z[idx];
            sh_m[threadIdx.x] = mass[idx];
        } else {
            sh_x[threadIdx.x] = 0.0;
            sh_y[threadIdx.x] = 0.0;
            sh_z[threadIdx.x] = 0.0;
            sh_m[threadIdx.x] = 0.0;
        }

        __syncthreads();
        // 遍歷這個BLOCK
        if (i < N) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                bodyBodyInteraction(my_x, my_y, my_z, sh_x[j], sh_y[j], sh_z[j], sh_m[j], new_ax, new_ay, new_az);
            }
        }
        __syncthreads();
    }

    if (i < N) {
        // 1. 儲存新加速度 a(t+1) (供下一輪的 Step 1 使用)
        acc_x[i] = new_ax;
        acc_y[i] = new_ay;
        acc_z[i] = new_az;

        // 2. 完成速度更新
        // v(t+1) = v(t+0.5) + 0.5 * a(t+1) * dt
        vel_x[i] += 0.5 * new_ax * dt;
        vel_y[i] += 0.5 * new_ay * dt;
        vel_z[i] += 0.5 * new_az * dt;
    }
}

/* -------------------------------------------------------- */
/* Host Helper Functions                                    */
/* -------------------------------------------------------- */

// double getTimeStamp() {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return (double)tv.tv_usec/1000000 + tv.tv_sec;
// }

// 讀取 Input 並轉成 SoA 陣列
int read_input_soa(
    const char *filename, int *N, double *total_time, double *dt, int *dump_interval, double **h_x, double **h_y, double **h_z, double **h_vx, double **h_vy, double **h_vz, double **h_m) {
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

    *h_x = (double *)malloc(*N * sizeof(double));
    *h_y = (double *)malloc(*N * sizeof(double));
    *h_z = (double *)malloc(*N * sizeof(double));
    *h_vx = (double *)malloc(*N * sizeof(double));
    *h_vy = (double *)malloc(*N * sizeof(double));
    *h_vz = (double *)malloc(*N * sizeof(double));
    *h_m = (double *)malloc(*N * sizeof(double));

    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &(*h_x)[i], &(*h_y)[i], &(*h_z)[i], &(*h_vx)[i], &(*h_vy)[i], &(*h_vz)[i], &(*h_m)[i]) != 7) {
            fclose(fin);
            return 0;
        }
    }
    fclose(fin);
    return 1;
}

void dump_traj_soa(FILE *ftraj, int step, double t, int N, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *m) {
    for (int i = 0; i < N; ++i) {
        fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]);
    }
}

/* -------------------------------------------------------- */
/* Main Function                                            */
/* -------------------------------------------------------- */

/* 初始加速度計算 (Leapfrog 啟動需要 a(0)) */
__global__ void compute_initial_acc_soa(int N,
                                        const double *__restrict__ pos_x,
                                        const double *__restrict__ pos_y,
                                        const double *__restrict__ pos_z,
                                        const double *__restrict__ mass,
                                        double *__restrict__ acc_x,
                                        double *__restrict__ acc_y,
                                        double *__restrict__ acc_z) {
    // (省略代碼：與 leapfrog_step2_soa 的計算部分完全相同，只是不更新速度)
    // 為了完整性，這裡應該複製 leapfrog_step2 的計算邏輯，或者直接把 step2 改寫成支援 init mode
    // 這裡為了簡潔，使用簡單版邏輯

    __shared__ double sh_x[BLOCK_SIZE];
    __shared__ double sh_y[BLOCK_SIZE];
    __shared__ double sh_z[BLOCK_SIZE];
    __shared__ double sh_m[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_x, my_y, my_z;
    if (i < N) {
        my_x = pos_x[i];
        my_y = pos_y[i];
        my_z = pos_z[i];
    }

    double ax = 0, ay = 0, az = 0;
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        int idx = tile * BLOCK_SIZE + threadIdx.x;
        if (idx < N) {
            sh_x[threadIdx.x] = pos_x[idx];
            sh_y[threadIdx.x] = pos_y[idx];
            sh_z[threadIdx.x] = pos_z[idx];
            sh_m[threadIdx.x] = mass[idx];
        } else {
            sh_x[threadIdx.x] = 0;
            sh_y[threadIdx.x] = 0;
            sh_z[threadIdx.x] = 0;
            sh_m[threadIdx.x] = 0;
        }
        __syncthreads();
        if (i < N) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                bodyBodyInteraction(my_x, my_y, my_z, sh_x[j], sh_y[j], sh_z[j], sh_m[j], ax, ay, az);
            }
        }
        __syncthreads();
    }
    if (i < N) {
        acc_x[i] = ax;
        acc_y[i] = ay;
        acc_z[i] = az;
    }
}

int main(int argc, char **argv) {
    clock_t start = clock();
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <traj_output_csv>\n", argv[0]);
        return 1;
    }

    int N, dump_interval;
    double total_time, dt;
    double *h_x, *h_y, *h_z, *h_vx, *h_vy, *h_vz, *h_m;

    if (!read_input_soa(argv[1], &N, &total_time, &dt, &dump_interval, &h_x, &h_y, &h_z, &h_vx, &h_vy, &h_vz, &h_m)) return 1;

    int steps = (int)(total_time / dt);

    // Device Arrays (SoA)
    // Leapfrog 需要儲存加速度，所以需要 d_ax, d_ay, d_az
    double *d_x, *d_y, *d_z;
    double *d_vx, *d_vy, *d_vz;
    double *d_ax, *d_ay, *d_az;
    double *d_m;

    size_t bytes = N * sizeof(double);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);
    cudaMalloc(&d_vx, bytes);
    cudaMalloc(&d_vy, bytes);
    cudaMalloc(&d_vz, bytes);
    cudaMalloc(&d_ax, bytes);
    cudaMalloc(&d_ay, bytes);
    cudaMalloc(&d_az, bytes);
    cudaMalloc(&d_m, bytes);

    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);

    FILE *ftraj = fopen(argv[2], "w");
    if (ftraj) fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");

    // printf("Running Leapfrog SoA Optimized Simulation: N=%d, Steps=%d\n", N, steps);
    // double start = getTimeStamp();

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double t = 0.0;

    // --- Initial Step: 計算 a(0) ---
    compute_initial_acc_soa<<<gridSize, BLOCK_SIZE>>>(N, d_x, d_y, d_z, d_m, d_ax, d_ay, d_az);
    cudaDeviceSynchronize();

    if (ftraj) dump_traj_soa(ftraj, 0, t, N, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_m);

    // --- Main Loop ---
    for (int s = 1; s <= steps; ++s) {

        // Kernel 1: 更新位置 + 半步速度 (Read a(t), Write x(t+1), v(t+0.5))
        leapfrog_step1_soa<<<gridSize, BLOCK_SIZE>>>(N, dt, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);

        // Kernel 2: 計算新力 + 更新後半步速度 (Read x(t+1), Write a(t+1), v(t+1))
        leapfrog_step2_soa<<<gridSize, BLOCK_SIZE>>>(N, dt, d_x, d_y, d_z, d_m, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);

        t += dt;

        if (s % dump_interval == 0 && ftraj) {
            cudaMemcpy(h_x, d_x, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vx, d_vx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy, d_vy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vz, d_vz, bytes, cudaMemcpyDeviceToHost);
            dump_traj_soa(ftraj, s, t, N, h_x, h_y, h_z, h_vx, h_vy, h_vz, h_m);
        }
    }

    cudaDeviceSynchronize();
    // double end = getTimeStamp();
    // printf("Total Execution Time: %.3f seconds\n", end - start);

    if (ftraj) fclose(ftraj);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_vx);
    free(h_vy);
    free(h_vz);
    free(h_m);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);
    cudaFree(d_m);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "ELAPSED_TIME: %.6f\n", elapsed);
    return 0;
}

// 現在的作法是 每個thread launch 到一個pixel
// 先根據加速度更新速度及位置
// step2
// 先開shared mem
// 然後開一個tile loop
// 每個thread 都將自己的資訊放到shared mem
// sync
// 然後開一個block size loop
// 更新自己與shared mem中每個點的交互 並且累加更新加速度
// 然後便利玩shared mem每個點後 再去抓下一個tile到自己的shared mem

// 直到跑完所有的點

// 最後把加速度跟速度寫回global memory