// nbody.cu

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = (call);                                                                       \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

typedef struct {
    double *x, *y, *z;    /* position arrays */
    double *vx, *vy, *vz; /* velocity arrays */
    double *m;            /* mass array */
} BodySoA;

#define G 1.0
#define SOFTENING 1e-3
#define SM_SIZE 64
#define ILP 1

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, BodySoA *soa);
FILE *init_traj_file(const char *filename);
void dump_traj_step(FILE *ftraj, int step, double t, int N, const BodySoA *soa);

__global__ void compute_velocity_position_kernel(
    double *d_x, double *d_y, double *d_z, double *d_vx, double *d_vy, double *d_vz, const int N, const double dt, const double *d_ax, const double *d_ay, const double *d_az) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* 1) v(t+1/2) = v(t) + 0.5 * a(t) * dt */
    d_vx[i] += 0.5 * d_ax[i] * dt;
    d_vy[i] += 0.5 * d_ay[i] * dt;
    d_vz[i] += 0.5 * d_az[i] * dt;

    /* 2) x(t+1) = x(t) + v(t+1/2) * dt */
    d_x[i] += d_vx[i] * dt;
    d_y[i] += d_vy[i] * dt;
    d_z[i] += d_vz[i] * dt;
}

__global__ void compute_acceleration_kernel(const double *__restrict__ px,
                                            const double *__restrict__ py,
                                            const double *__restrict__ pz,
                                            const double *__restrict__ pm,
                                            const int N,
                                            double *__restrict__ d_ax,
                                            double *__restrict__ d_ay,
                                            double *__restrict__ d_az) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    double xi[ILP], yi[ILP], zi[ILP];
    double axi[ILP] = {0}, ayi[ILP] = {0}, azi[ILP] = {0};
    int body[ILP];

#pragma unroll
    for (int k = 0; k < ILP; k++) {
        body[k] = idx + k * stride;

        if (body[k] < N) {
            xi[k] = px[body[k]];
            yi[k] = py[body[k]];
            zi[k] = pz[body[k]];
        }
    }

    __shared__ double sm_x[SM_SIZE];
    __shared__ double sm_y[SM_SIZE];
    __shared__ double sm_z[SM_SIZE];
    __shared__ double sm_m[SM_SIZE];

    const int num_tiles = (N + SM_SIZE - 1) / SM_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {

        const int load_idx = tile * blockDim.x + threadIdx.x;

        if (load_idx < N) {
            sm_x[threadIdx.x] = px[load_idx];
            sm_y[threadIdx.x] = py[load_idx];
            sm_z[threadIdx.x] = pz[load_idx];
            sm_m[threadIdx.x] = pm[load_idx];
        } else {

            sm_x[threadIdx.x] = 0.0;
            sm_y[threadIdx.x] = 0.0;
            sm_z[threadIdx.x] = 0.0;
            sm_m[threadIdx.x] = 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < SM_SIZE; j++) {

            const double jx = sm_x[j];
            const double jy = sm_y[j];
            const double jz = sm_z[j];
            const double jm = sm_m[j];

#pragma unroll
            for (int k = 0; k < ILP; k++) {
                if (body[k] < N) {
                    const double dx = jx - xi[k];
                    const double dy = jy - yi[k];
                    const double dz = jz - zi[k];
                    const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    const double invDist = 1.0 / sqrt(distSqr);  // const double invDist = rsqrt((double)distSqr);
                    const double invDist3 = invDist * invDist * invDist;
                    const double force = G * jm * invDist3;

                    axi[k] += dx * force;
                    ayi[k] += dy * force;
                    azi[k] += dz * force;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (body[k] < N) {

            d_ax[body[k]] = axi[k];
            d_ay[body[k]] = ayi[k];
            d_az[body[k]] = azi[k];
        }
    }
}

__global__ void compute_velocity_kernel(double *d_vx, double *d_vy, double *d_vz, const int N, const double dt, const double *d_ax, const double *d_ay, const double *d_az) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* 4) v(t+1) = v(t+1/2) + 0.5 * a(t+1) * dt */
    d_vx[i] += 0.5 * d_ax[i] * dt;
    d_vy[i] += 0.5 * d_ay[i] * dt;
    d_vz[i] += 0.5 * d_az[i] * dt;
}

void step_leapfrog(const int N, const double dt, double *d_x, double *d_y, double *d_z, double *d_vx, double *d_vy, double *d_vz, double *d_m, double *d_ax, double *d_ay, double *d_az) {
    const int blockSize = SM_SIZE;
    const int gridSize = (N + blockSize - 1) / blockSize;

    /* 1) v(t+1/2) = v(t) + 0.5 * a(t) * dt */
    /* 2) x(t+1) = x(t) + v(t+1/2) * dt */

    compute_velocity_position_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, dt, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());

    /* 3) GPU 計算 a(t+1) */
    const int bodies_per_thread = ILP;

    const int total_threads = (N + bodies_per_thread - 1) / bodies_per_thread;
    const int acc_gridSize = (total_threads + blockSize - 1) / blockSize;

    compute_acceleration_kernel<<<acc_gridSize, blockSize>>>(d_x, d_y, d_z, d_m, N, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());

    /* 4) v(t+1) = v(t+1/2) + 0.5 * a(t+1) * dt */

    compute_velocity_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, N, dt, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char **argv) {

    clock_t start = clock();
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_csv>\n", argv[0]);
        return 1;
    }

    int N = 0;
    int dump_interval = 0;
    double total_time = 0.0;
    double dt = 0.0;

    BodySoA h_soa = {0};

    if (!read_input(argv[1], &N, &total_time, &dt, &dump_interval, &h_soa)) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }

    const int steps = (int)(total_time / dt);
    if (steps <= 0) {
        fprintf(stderr, "Error: total_time / dt <= 0\n");
        free(h_soa.x);
        free(h_soa.y);
        free(h_soa.z);
        free(h_soa.vx);
        free(h_soa.vy);
        free(h_soa.vz);
        free(h_soa.m);
        return 1;
    }

    FILE *ftraj = init_traj_file(argv[2]);
    if (!ftraj) {
        free(h_soa.x);
        free(h_soa.y);
        free(h_soa.z);
        free(h_soa.vx);
        free(h_soa.vy);
        free(h_soa.vz);
        free(h_soa.m);
        return 1;
    }

    double t = 0.0;

    double *d_x, *d_y, *d_z;
    double *d_vx, *d_vy, *d_vz;
    double *d_m;
    double *d_ax, *d_ay, *d_az;

    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vx, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vy, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vz, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_m, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ax, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ay, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_az, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, h_soa.x, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_soa.y, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_soa.z, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_soa.vx, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_soa.vy, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_soa.vz, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_soa.m, N * sizeof(double), cudaMemcpyHostToDevice));

    const int bodies_per_thread = ILP;
    const int total_threads = (N + bodies_per_thread - 1) / bodies_per_thread;
    const int threads_per_block = SM_SIZE;
    const int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
    compute_acceleration_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, d_z, d_m, N, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());

    dump_traj_step(ftraj, 0, t, N, &h_soa);

    for (int s = 1; s <= steps; ++s) {

        step_leapfrog(N, dt, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_m, d_ax, d_ay, d_az);

        t += dt;

        if (s % dump_interval == 0) {

            CUDA_CHECK(cudaMemcpy(h_soa.x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_soa.y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_soa.z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_soa.vx, d_vx, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_soa.vy, d_vy, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_soa.vz, d_vz, N * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_soa.m, d_m, N * sizeof(double), cudaMemcpyDeviceToHost));

            dump_traj_step(ftraj, s, t, N, &h_soa);
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_ax));
    CUDA_CHECK(cudaFree(d_ay));
    CUDA_CHECK(cudaFree(d_az));

    fclose(ftraj);
    free(h_soa.x);
    free(h_soa.y);
    free(h_soa.z);
    free(h_soa.vx);
    free(h_soa.vy);
    free(h_soa.vz);
    free(h_soa.m);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "ELAPSED_TIME: %.6f\n", elapsed);
    return 0;
}

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, BodySoA *soa) {
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

    soa->x = (double *)malloc(*N * sizeof(double));
    soa->y = (double *)malloc(*N * sizeof(double));
    soa->z = (double *)malloc(*N * sizeof(double));
    soa->vx = (double *)malloc(*N * sizeof(double));
    soa->vy = (double *)malloc(*N * sizeof(double));
    soa->vz = (double *)malloc(*N * sizeof(double));
    soa->m = (double *)malloc(*N * sizeof(double));

    if (!soa->x || !soa->y || !soa->z || !soa->vx || !soa->vy || !soa->vz || !soa->m) {
        fprintf(stderr, "Error: malloc failed for SoA arrays\n");

        if (soa->x) free(soa->x);
        if (soa->y) free(soa->y);
        if (soa->z) free(soa->z);
        if (soa->vx) free(soa->vx);
        if (soa->vy) free(soa->vy);
        if (soa->vz) free(soa->vz);
        if (soa->m) free(soa->m);
        fclose(fin);
        return 0;
    }

    for (int i = 0; i < *N; ++i) {
        if (fscanf(fin, "%lf %lf %lf %lf %lf %lf %lf", &soa->x[i], &soa->y[i], &soa->z[i], &soa->vx[i], &soa->vy[i], &soa->vz[i], &soa->m[i]) != 7) {
            fprintf(stderr, "Error: invalid body line at index %d\n", i);
            free(soa->x);
            free(soa->y);
            free(soa->z);
            free(soa->vx);
            free(soa->vy);
            free(soa->vz);
            free(soa->m);
            fclose(fin);
            return 0;
        }
    }

    fclose(fin);
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

void dump_traj_step(FILE *ftraj, int step, double t, int N, const BodySoA *soa) {
    for (int i = 0; i < N; ++i) {
        fprintf(ftraj, "%d,%.5f,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", step, t, i, soa->x[i], soa->y[i], soa->z[i], soa->vx[i], soa->vy[i], soa->vz[i], soa->m[i]);
    }
}
