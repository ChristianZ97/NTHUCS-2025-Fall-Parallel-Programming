#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define SOFTENING 1e-3
#define G 1.0

/* -------------------------------------------------------- */
/* Device Functions & Kernels (保持不變)                    */
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
    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;
}

__global__ void leapfrog_step2_soa(int N,
                                   double dt,
                                   const double *__restrict__ pos_x,
                                   const double *__restrict__ pos_y,
                                   const double *__restrict__ pos_z,
                                   const double *__restrict__ mass,
                                   double *__restrict__ vel_x,
                                   double *__restrict__ vel_y,
                                   double *__restrict__ vel_z,
                                   double *__restrict__ acc_x,
                                   double *__restrict__ acc_y,
                                   double *__restrict__ acc_z) {
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
#pragma unroll 16
            for (int j = 0; j < BLOCK_SIZE; j++) {
                bodyBodyInteraction(my_x, my_y, my_z, sh_x[j], sh_y[j], sh_z[j], sh_m[j], new_ax, new_ay, new_az);
            }
        }
        __syncthreads();
    }
    if (i < N) {
        acc_x[i] = new_ax;
        acc_y[i] = new_ay;
        acc_z[i] = new_az;
        vel_x[i] += 0.5 * new_ax * dt;
        vel_y[i] += 0.5 * new_ay * dt;
        vel_z[i] += 0.5 * new_az * dt;
    }
}

__global__ void compute_initial_acc_soa(int N,
                                        const double *__restrict__ pos_x,
                                        const double *__restrict__ pos_y,
                                        const double *__restrict__ pos_z,
                                        const double *__restrict__ mass,
                                        double *__restrict__ acc_x,
                                        double *__restrict__ acc_y,
                                        double *__restrict__ acc_z) {
    // (為了節省篇幅，邏輯同上，僅省略)
    // 實際使用時請填入完整的初始化 kernel 代碼
    // ...
    // 這裡直接使用簡單版邏輯模擬
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

/* -------------------------------------------------------- */
/* Helper Functions                                         */
/* -------------------------------------------------------- */

// double getTimeStamp() {
//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     return (double)tv.tv_usec/1000000 + tv.tv_sec;
// }

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

/* -------------------------------------------------------- */
/* Main Function                                            */
/* -------------------------------------------------------- */

int main(int argc, char **argv) {
    clock_t start = clock();

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <traj_output_csv>\n", argv[0]);
        return 1;
    }
    // double start = getTimeStamp();
    int N, dump_interval;
    double total_time, dt;
    double *h_x, *h_y, *h_z, *h_vx, *h_vy, *h_vz, *h_m;

    if (!read_input_soa(argv[1], &N, &total_time, &dt, &dump_interval, &h_x, &h_y, &h_z, &h_vx, &h_vy, &h_vz, &h_m)) return 1;

    int steps = (int)(total_time / dt);

    // 1. 計算總共需要存多少個 Snapshot
    // 通常包含 t=0 的初始狀態，以及每 dump_interval 存一次
    int num_snapshots = 1 + (steps / dump_interval);

    // printf("Config: N=%d, Steps=%d, Dump Interval=%d\n", N, steps, dump_interval);
    // printf("Optimization: Allocating buffer for %d snapshots on GPU.\n", num_snapshots);

    // --- GPU Memory Allocation ---

    // A. 當前狀態 (Current State)
    double *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_ax, *d_ay, *d_az, *d_m;
    size_t bytes_curr = N * sizeof(double);
    cudaMalloc(&d_x, bytes_curr);
    cudaMalloc(&d_y, bytes_curr);
    cudaMalloc(&d_z, bytes_curr);
    cudaMalloc(&d_vx, bytes_curr);
    cudaMalloc(&d_vy, bytes_curr);
    cudaMalloc(&d_vz, bytes_curr);
    cudaMalloc(&d_ax, bytes_curr);
    cudaMalloc(&d_ay, bytes_curr);
    cudaMalloc(&d_az, bytes_curr);
    cudaMalloc(&d_m, bytes_curr);

    // B. 歷史軌跡緩衝區 (Trajectory History Buffer)
    // 不需要存 Mass (因為 Mass 不變)
    double *d_traj_x, *d_traj_y, *d_traj_z, *d_traj_vx, *d_traj_vy, *d_traj_vz;
    size_t bytes_traj = num_snapshots * N * sizeof(double);

    // 檢查顯存分配結果 (這是最可能失敗的地方)
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_traj_x, bytes_traj);
    if (err != cudaSuccess) {
        printf("Error alloc traj buffer: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMalloc(&d_traj_y, bytes_traj);
    cudaMalloc(&d_traj_z, bytes_traj);
    cudaMalloc(&d_traj_vx, bytes_traj);
    cudaMalloc(&d_traj_vy, bytes_traj);
    cudaMalloc(&d_traj_vz, bytes_traj);

    // --- Copy Initial Data ---
    cudaMemcpy(d_x, h_x, bytes_curr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes_curr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, bytes_curr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, bytes_curr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, bytes_curr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, bytes_curr, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, bytes_curr, cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --- Initial Calculation ---
    compute_initial_acc_soa<<<gridSize, BLOCK_SIZE>>>(N, d_x, d_y, d_z, d_m, d_ax, d_ay, d_az);
    cudaDeviceSynchronize();

    // --- Save Snapshot 0 (t=0) ---
    // 使用 DeviceToDevice 複製，極快
    // 儲存到 d_traj 的第 0 個 slot
    cudaMemcpy(d_traj_x, d_x, bytes_curr, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_traj_y, d_y, bytes_curr, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_traj_z, d_z, bytes_curr, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_traj_vx, d_vx, bytes_curr, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_traj_vy, d_vy, bytes_curr, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_traj_vz, d_vz, bytes_curr, cudaMemcpyDeviceToDevice);

    int saved_count = 1;

    // --- Main Simulation Loop ---
    for (int s = 1; s <= steps; ++s) {

        leapfrog_step1_soa<<<gridSize, BLOCK_SIZE>>>(N, dt, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);

        leapfrog_step2_soa<<<gridSize, BLOCK_SIZE>>>(N, dt, d_x, d_y, d_z, d_m, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);

        // 如果需要 Dump，則在 GPU 內部備份
        if (s % dump_interval == 0) {
            // 計算偏移量 (Offset)
            size_t offset = (size_t)saved_count * N;

            // DeviceToDevice Copy
            cudaMemcpy(d_traj_x + offset, d_x, bytes_curr, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_traj_y + offset, d_y, bytes_curr, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_traj_z + offset, d_z, bytes_curr, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_traj_vx + offset, d_vx, bytes_curr, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_traj_vy + offset, d_vy, bytes_curr, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_traj_vz + offset, d_vz, bytes_curr, cudaMemcpyDeviceToDevice);

            saved_count++;
        }
    }

    cudaDeviceSynchronize();

    // --- Retrieve All Data (Batch Copy) ---
    // 這裡需要在 Host 分配足夠大的記憶體來接收結果
    // 由於我們只在最後做一次，所以 Host 記憶體夠不夠大也是個考量點
    double *h_traj_x = (double *)malloc(bytes_traj);
    double *h_traj_y = (double *)malloc(bytes_traj);
    double *h_traj_z = (double *)malloc(bytes_traj);
    double *h_traj_vx = (double *)malloc(bytes_traj);
    double *h_traj_vy = (double *)malloc(bytes_traj);
    double *h_traj_vz = (double *)malloc(bytes_traj);

    if (!h_traj_x) {
        printf("Host memory allocation failed for trajectory!\n");
        return 1;
    }

    // 一次性從 Device 傳回 Host
    cudaMemcpy(h_traj_x, d_traj_x, bytes_traj, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_traj_y, d_traj_y, bytes_traj, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_traj_z, d_traj_z, bytes_traj, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_traj_vx, d_traj_vx, bytes_traj, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_traj_vy, d_traj_vy, bytes_traj, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_traj_vz, d_traj_vz, bytes_traj, cudaMemcpyDeviceToHost);

    // --- Write to File ---
    FILE *ftraj = fopen(argv[2], "w");
    if (ftraj) {
        fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");

        for (int k = 0; k < saved_count; k++) {
            int step = k * dump_interval;
            double t = step * dt;
            size_t offset = (size_t)k * N;

            for (int i = 0; i < N; ++i) {
                // Mass 使用 h_m[i] (假設質量不變，節省顯存)
                fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, h_traj_x[offset + i], h_traj_y[offset + i], h_traj_z[offset + i], h_traj_vx[offset + i],
                        h_traj_vy[offset + i], h_traj_vz[offset + i], h_m[i]);
            }
        }
        fclose(ftraj);
    }

    // Free Host Memory
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_vx);
    free(h_vy);
    free(h_vz);
    free(h_m);
    free(h_traj_x);
    free(h_traj_y);
    free(h_traj_z);
    free(h_traj_vx);
    free(h_traj_vy);
    free(h_traj_vz);

    // Free Device Memory
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
    cudaFree(d_traj_x);
    cudaFree(d_traj_y);
    cudaFree(d_traj_z);
    cudaFree(d_traj_vx);
    cudaFree(d_traj_vy);
    cudaFree(d_traj_vz);
    // double end = getTimeStamp();
    // printf("Simulation Completed. Total Execution Time : %.3f seconds\n", end - start);
    // printf("Done.\n");
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "ELAPSED_TIME: %.6f\n", elapsed);
    return 0;
}