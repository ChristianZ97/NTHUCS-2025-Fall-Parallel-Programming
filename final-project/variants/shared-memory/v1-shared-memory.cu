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
    double x, y, z;    /* position */
    double vx, vy, vz; /* velocity */
    double m;          /* mass */
} Body;

#define G 1.0
#define SOFTENING 1e-3
#define SM_SIZE 256

int read_input(const char *filename, int *N, double *total_time, double *dt, int *dump_interval, Body **bodies_out);
FILE *init_traj_file(const char *filename);
void dump_traj_step(FILE *ftraj, int step, double t, int N, const Body *bodies);

__global__ void compute_velocity_position_kernel(Body *d_bodies, const int N, const double dt, double *d_ax, double *d_ay, double *d_az) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* 1) v(t+1/2) = v(t) + 0.5 * a(t) * dt */
    d_bodies[i].vx += 0.5 * d_ax[i] * dt;
    d_bodies[i].vy += 0.5 * d_ay[i] * dt;
    d_bodies[i].vz += 0.5 * d_az[i] * dt;

    /* 2) x(t+1) = x(t) + v(t+1/2) * dt */
    d_bodies[i].x += d_bodies[i].vx * dt;
    d_bodies[i].y += d_bodies[i].vy * dt;
    d_bodies[i].z += d_bodies[i].vz * dt;
}

__global__ void compute_acceleration_kernel(const Body *d_bodies, const int N, double *d_ax, double *d_ay, double *d_az) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double xi = (i < N) ? d_bodies[i].x : 0;
    const double yi = (i < N) ? d_bodies[i].y : 0;
    const double zi = (i < N) ? d_bodies[i].z : 0;

    double axi = 0.0;
    double ayi = 0.0;
    double azi = 0.0;

    __shared__ Body sm_bodies[SM_SIZE];
    for (int tile = 0; tile < gridDim.x; tile++) {

        const int load_idx = tile * blockDim.x + threadIdx.x;
        if (load_idx < N)
            sm_bodies[threadIdx.x] = d_bodies[load_idx];
        else {
            sm_bodies[threadIdx.x].x = 0.0;
            sm_bodies[threadIdx.x].y = 0.0;
            sm_bodies[threadIdx.x].z = 0.0;
            sm_bodies[threadIdx.x].m = 0.0;
        }
        __syncthreads();

        if (i < N) {

#pragma unroll 16

            for (int j = 0; j < SM_SIZE; j++) {

                // if ((tile * blockDim.x + j) == i) continue;

                const double dx = sm_bodies[j].x - xi;
                const double dy = sm_bodies[j].y - yi;
                const double dz = sm_bodies[j].z - zi;

                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);  // const double invDist = rsqrt(distSqr); fast, but wrong answer
                const double invDist3 = invDist * invDist * invDist;
                const double force = G * sm_bodies[j].m * invDist3;

                axi += dx * force;
                ayi += dy * force;
                azi += dz * force;
            }
        }
        __syncthreads();
    }

    d_ax[i] = (i < N) ? axi : 0;
    d_ay[i] = (i < N) ? ayi : 0;
    d_az[i] = (i < N) ? azi : 0;
}

__global__ void compute_velocity_kernel(Body *d_bodies, const int N, const double dt, double *d_ax, double *d_ay, double *d_az) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* 4) v(t+1) = v(t+1/2) + 0.5 * a(t+1) * dt */
    d_bodies[i].vx += 0.5 * d_ax[i] * dt;
    d_bodies[i].vy += 0.5 * d_ay[i] * dt;
    d_bodies[i].vz += 0.5 * d_az[i] * dt;
}

void step_leapfrog(Body *h_bodies, const int N, const double dt, double *ax, double *ay, double *az, Body *d_bodies, double *d_ax, double *d_ay, double *d_az) {

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    /* 1) v(t+1/2) = v(t) + 0.5 * a(t) * dt */
    /* 2) x(t+1) = x(t) + v(t+1/2) * dt */
    compute_velocity_position_kernel<<<gridSize, blockSize>>>(d_bodies, N, dt, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());

    /* 3) GPU 計算 a(t+1) */
    compute_acceleration_kernel<<<gridSize, blockSize>>>(d_bodies, N, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());

    /* 4) v(t+1) = v(t+1/2) + 0.5 * a(t+1) * dt */
    compute_velocity_kernel<<<gridSize, blockSize>>>(d_bodies, N, dt, d_ax, d_ay, d_az);
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
    Body *h_bodies = NULL;

    if (!read_input(argv[1], &N, &total_time, &dt, &dump_interval, &h_bodies)) {
        fprintf(stderr, "Error reading input.\n");
        return 1;
    }

    const int steps = (int)(total_time / dt);
    if (steps <= 0) {
        fprintf(stderr, "Error: total_time / dt <= 0\n");
        free(h_bodies);
        return 1;
    }

    double *ax = (double *)malloc(N * sizeof(double));
    double *ay = (double *)malloc(N * sizeof(double));
    double *az = (double *)malloc(N * sizeof(double));
    if (!ax || !ay || !az) {
        fprintf(stderr, "Error: malloc failed for acceleration buffers\n");
        free(h_bodies);
        free(ax);
        free(ay);
        free(az);
        return 1;
    }

    FILE *ftraj = init_traj_file(argv[2]);
    if (!ftraj) {
        free(h_bodies);
        free(ax);
        free(ay);
        free(az);
        return 1;
    }

    // printf("Running simulation (v0 logic, v1 style): N=%d, steps=%d, dump_interval=%d\n", N, steps, dump_interval);

    double t = 0.0;
    Body *d_bodies = NULL;
    double *d_ax = NULL;
    double *d_ay = NULL;
    double *d_az = NULL;

    CUDA_CHECK(cudaMalloc(&d_bodies, N * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_ax, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ay, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_az, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    compute_acceleration_kernel<<<gridSize, blockSize>>>(d_bodies, N, d_ax, d_ay, d_az);
    CUDA_CHECK(cudaGetLastError());

    dump_traj_step(ftraj, 0, t, N, h_bodies);
    for (int s = 1; s <= steps; ++s) {

        step_leapfrog(h_bodies, N, dt, ax, ay, az, d_bodies, d_ax, d_ay, d_az);
        t += dt;
        if (s % dump_interval == 0) {
            CUDA_CHECK(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost));
            dump_traj_step(ftraj, s, t, N, h_bodies);
        }
    }

    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_ax));
    CUDA_CHECK(cudaFree(d_ay));
    CUDA_CHECK(cudaFree(d_az));

    fclose(ftraj);
    free(h_bodies);
    free(ax);
    free(ay);
    free(az);

    // printf("Done. Output written to %s\n", argv[2]);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "ELAPSED_TIME: %.6f\n", elapsed);
    return 0;
}

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