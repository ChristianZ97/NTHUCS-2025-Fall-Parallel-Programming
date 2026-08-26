#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// double getTimeStamp() {
//     struct timeval tv;
//     gettimeofday( &tv, NULL );
//     return (double) tv.tv_usec/1000000 + tv.tv_sec;
// }

/* 單一質點 (Body) 的狀態 */
typedef struct {
    double x, y, z;    /* position */
    double vx, vy, vz; /* velocity */
    double m;          /* mass */
} Body;

/* 物理常數 */
#define G 1.0
#define SOFTENING 1e-3

/* -------------------------------------------------------- */
/* GPU Kernels (核心運算邏輯)                               */
/* -------------------------------------------------------- */

/* Kernel: 計算所有粒子的加速度 (O(N^2))
   對應原代碼 compute_acceleration */
__global__ void compute_acceleration_kernel(const Body *bodies, int N, double *ax, double *ay, double *az) {
    // 每個 thread 負責計算一個粒子 i 的加速度
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        double xi = bodies[i].x;
        double yi = bodies[i].y;
        double zi = bodies[i].z;

        double axi = 0.0, ayi = 0.0, azi = 0.0;

        // 每個 thread 都要遍歷所有其他粒子 j (暴力法)
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;

            double dx = bodies[j].x - xi;
            double dy = bodies[j].y - yi;
            double dz = bodies[j].z - zi;

            double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            double invDist = 1.0 / sqrt(distSqr);  // CUDA 特有的反平方根函數，也可寫 1.0/sqrt(...)
            double invDist3 = invDist * invDist * invDist;

            double force = G * bodies[j].m * invDist3;
            axi += dx * force;
            ayi += dy * force;
            azi += dz * force;
        }

        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}

/* Kernel: Leapfrog 第一階段 (更新半步速度 + 更新全步位置)
   對應 step_leapfrog 的第 1, 2 部分 */
__global__ void update_phase1_kernel(Body *bodies, int N, double dt, const double *ax, const double *ay, const double *az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // 1) v(t+1/2) = v(t) + 0.5 * a(t) * dt
        bodies[i].vx += 0.5 * ax[i] * dt;
        bodies[i].vy += 0.5 * ay[i] * dt;
        bodies[i].vz += 0.5 * az[i] * dt;

        // 2) x(t+1) = x(t) + v(t+1/2) * dt
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

/* Kernel: Leapfrog 第二階段 (更新剩下的半步速度)
   對應 step_leapfrog 的第 4 部分 */
__global__ void update_phase2_kernel(Body *bodies, int N, double dt, const double *ax, const double *ay, const double *az) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // 4) v(t+1) = v(t+1/2) + 0.5 * a(t+1) * dt
        bodies[i].vx += 0.5 * ax[i] * dt;
        bodies[i].vy += 0.5 * ay[i] * dt;
        bodies[i].vz += 0.5 * az[i] * dt;
    }
}

/* -------------------------------------------------------- */
/* Host Helper Functions (CPU 端讀檔/寫檔)                  */
/* -------------------------------------------------------- */

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out) {
    FILE *fin = fopen(filename, "r");
    if (!fin) {
        fprintf(stderr, "Error: cannot open input file %s\n", filename);
        return 0;
    }

    if (fscanf(fin, "%d", N) != 1 || *N <= 0) {
        fprintf(stderr, "Error: invalid N in input file\n");
        fclose(fin);
        return 0;
    }

    if (fscanf(fin, "%lf %lf %d", total_time, dt, dump_interval) != 3 || *total_time <= 0.0 || *dt <= 0.0 || *dump_interval <= 0) {
        fprintf(stderr, "Error: invalid total_time / dt / dump_interval\n");
        fclose(fin);
        return 0;
    }

    Body *bodies = (Body *)malloc((*N) * sizeof(Body));
    if (!bodies) {
        fprintf(stderr, "Error: malloc failed for %d bodies\n", *N);
        fclose(fin);
        return 0;
    }

    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &bodies[i].x, &bodies[i].y, &bodies[i].z, &bodies[i].vx, &bodies[i].vy, &bodies[i].vz, &bodies[i].m) != 7) {
            fprintf(stderr, "Error: invalid body line at index %d\n", i);
            free(bodies);
            fclose(fin);
            return 0;
        }
    }

    fclose(fin);
    *bodies_out = bodies;
    return 1;
}

FILE *init_traj_file(const char *filename) {
    FILE *ftraj = fopen(filename, "w");
    if (!ftraj) {
        fprintf(stderr, "Error: cannot open trajectory file %s\n", filename);
        return NULL;
    }
    fprintf(ftraj, "step,t,id,x,y,z,vx,vy,vz,m\n");
    return ftraj;
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

    // double start, end;
    // start = getTimeStamp();

    int N, dump_interval;
    double total_time, dt;
    Body *h_bodies = NULL;  // host bodies

    // 1. 讀取 Host 端資料
    if (!read_input(argv[1], &N, &total_time, &dt, &dump_interval, &h_bodies)) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }

    int steps = (int)(total_time / dt);

    // 2. 配置 Device (GPU) 記憶體
    Body *d_bodies;
    double *d_ax, *d_ay, *d_az;

    cudaMalloc((void **)&d_bodies, N * sizeof(Body));
    cudaMalloc((void **)&d_ax, N * sizeof(double));
    cudaMalloc((void **)&d_ay, N * sizeof(double));
    cudaMalloc((void **)&d_az, N * sizeof(double));

    // 3. 將初始資料從 Host 複製到 Device
    cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    FILE *ftraj = init_traj_file(argv[2]);
    // printf("Running GPU simulation: N=%d, steps=%d, dump_interval=%d\n", N, steps, dump_interval);

    // 設定 CUDA Grid/Block
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    double t = 0.0;

    // --- 初始步驟 ---
    // 計算 t=0 加速度 (GPU)
    compute_acceleration_kernel<<<gridSize, blockSize>>>(d_bodies, N, d_ax, d_ay, d_az);
    cudaDeviceSynchronize();  // 等待 GPU 完成

    // 輸出初始狀態 (需要先將資料複製回 Host)
    cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
    dump_traj_step(ftraj, 0, t, N, h_bodies);

    // --- 主要迴圈 ---
    for (int s = 1; s <= steps; ++s) {

        // 1) v(t+0.5), x(t+1) 更新
        update_phase1_kernel<<<gridSize, blockSize>>>(d_bodies, N, dt, d_ax, d_ay, d_az);

        // 2) 計算新位置的加速度 a(t+1)
        compute_acceleration_kernel<<<gridSize, blockSize>>>(d_bodies, N, d_ax, d_ay, d_az);

        // 3) v(t+1) 更新
        update_phase2_kernel<<<gridSize, blockSize>>>(d_bodies, N, dt, d_ax, d_ay, d_az);

        t += dt;

        // 需要輸出時，才從 GPU 搬運資料回 CPU
        if (s % dump_interval == 0) {
            cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
            dump_traj_step(ftraj, s, t, N, h_bodies);
        }
    }

    // 清理資源
    fclose(ftraj);
    free(h_bodies);
    cudaFree(d_bodies);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);

    // end = getTimeStamp();
    // printf("Time: %.3f seconds\n", end - start);

    // printf("Done. Output written to %s\n", argv[2]);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "ELAPSED_TIME: %.6f\n", elapsed);
    return 0;
}