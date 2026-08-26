/*
 * This program generates a visualization of the Mandelbrot set.
 * It is an MPI+OpenMP hybrid implementation that calculates the set for a specified
 * region of the complex plane and saves the output as a PNG image.
 *
 * NOTE: Profiling code is added within #ifdef PROFILING blocks.
 * Compile with -DPROFILING to enable timing measurements.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <assert.h>     // For assertions (assert)
#include <emmintrin.h>  // SSE2
#include <mpi.h>
#include <omp.h>
#include <png.h>     // For the libpng library, used to write PNG files
#include <sched.h>   // For CPU affinity functions (sched_getaffinity, CPU_COUNT)
#include <stdio.h>   // For standard I/O (printf, fopen, etc.)
#include <stdlib.h>  // For general utilities (strtol, strtod, malloc, free)
#include <string.h>  // For memory manipulation (memset)

#include <algorithm>
#include <cmath>

#ifdef PROFILING
#include <nvtx3/nvToolsExt.h>

// ============================================================================
// NVTX Color Definitions for Performance Profiling
// ============================================================================
#define COLOR_IO 0xFF5C9FFF       // Azure Blue
#define COLOR_COMPUTE 0xFF5FD068  // Spring Green (Computation)
#define COLOR_COMM 0xFFFFA500     // Orange (Communication)
#define COLOR_SETUP 0xFFFFBE3D    // Sunflower Yellow
#define COLOR_DEFAULT 0xFFCBD5E0  // Soft Gray

// ============================================================================
// Smart NVTX Macro: Auto-selects color based on range name pattern
// ============================================================================
#define NVTX_PUSH(name)                                                                \
    do {                                                                               \
        uint32_t color = COLOR_DEFAULT;                                                \
        if (strstr(name, "IO_") != NULL) {                                             \
            color = COLOR_IO;                                                          \
        } else if (strstr(name, "Compute") != NULL) {                                  \
            color = COLOR_COMPUTE;                                                     \
        } else if (strstr(name, "Comm_") != NULL || strstr(name, "Gather") != NULL) {  \
            color = COLOR_COMM;                                                        \
        } else if (strcmp(name, "MPI_Setup") == 0 || strcmp(name, "Mem_Alloc") == 0) { \
            color = COLOR_SETUP;                                                       \
        } else {                                                                       \
            color = COLOR_DEFAULT;                                                     \
        }                                                                              \
        nvtxEventAttributes_t eventAttrib = {0};                                       \
        eventAttrib.version = NVTX_VERSION;                                            \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                              \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                                       \
        eventAttrib.color = color;                                                     \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                             \
        eventAttrib.message.ascii = name;                                              \
        nvtxRangePushEx(&eventAttrib);                                                 \
    } while (0)

#define NVTX_POP() nvtxRangePop()
#endif

void write_png(const char *filename, int iters, int width, int height, const int *buffer);

int main(int argc, char *argv[]) {
    // ========================================================================
    // Block 1: MPI Initialization
    // ========================================================================
    MPI_Init(&argc, &argv);

#ifdef PROFILING
    double total_start_time, temp_start, io_time = 0.0, comm_time = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting the main timer
    total_start_time = MPI_Wtime();
    NVTX_PUSH("MPI_Setup");
#endif

    int numtasks = 0, my_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    const char *filename = argv[1];
    const int iters = strtol(argv[2], NULL, 10);
    const double left = strtod(argv[3], NULL);   // Real part start (x0)
    const double right = strtod(argv[4], NULL);  // Real part end (x1)
    const double lower = strtod(argv[5], NULL);  // Imaginary part start (y0)
    const double upper = strtod(argv[6], NULL);  // Imaginary part end (y1)
    const int width = strtol(argv[7], NULL, 10);
    const int height = strtol(argv[8], NULL, 10);

    const int num_threads = CPU_COUNT(&cpu_set);

    const int base_chunk_size = height / numtasks;
    const int remainder = height % numtasks;
    const int my_count = (my_rank < remainder) ? base_chunk_size + 1 : base_chunk_size;

#ifdef PROFILING
    NVTX_POP();  // end MPI_Setup
    NVTX_PUSH("Mem_Alloc");
#endif

    int *global_image = NULL;
    int *image = (int *)malloc(width * my_count * sizeof(int));
    if (my_rank == 0) global_image = (int *)malloc(width * height * sizeof(int));

    const double x_scale = (right - left) / width;
    const double y_scale = (upper - lower) / height;

#ifdef PROFILING
    NVTX_POP();  // end Mem_Alloc
#endif

    // ========================================================================
    // Block 2: Main Computation (Hybrid MPI+OpenMP)
    // ========================================================================
#ifdef PROFILING
    NVTX_PUSH("Compute_Mandelbrot");
#endif

    const __m128d four = _mm_set1_pd(4.0);
    const __m128d one = _mm_set1_pd(1.0);

#pragma omp parallel num_threads(num_threads)
#pragma omp for schedule(dynamic, 1) nowait

    for (int local_j = 0; local_j < my_count; ++local_j) {
        const int j = my_rank + local_j * numtasks;
        const double y0 = j * y_scale + lower;
        const __m128d y0_vec = _mm_set1_pd(y0);

        for (int i = 0; i <= width - 8; i += 8) {
            __m128d repeats_vec_0 = _mm_setzero_pd();
            __m128d repeats_vec_1 = _mm_setzero_pd();
            __m128d repeats_vec_2 = _mm_setzero_pd();
            __m128d repeats_vec_3 = _mm_setzero_pd();

            __m128d x_vec_0 = _mm_setzero_pd();
            __m128d y_vec_0 = _mm_setzero_pd();
            __m128d x_vec_1 = _mm_setzero_pd();
            __m128d y_vec_1 = _mm_setzero_pd();
            __m128d x_vec_2 = _mm_setzero_pd();
            __m128d y_vec_2 = _mm_setzero_pd();
            __m128d x_vec_3 = _mm_setzero_pd();
            __m128d y_vec_3 = _mm_setzero_pd();

            const __m128d x0_vec_0 = _mm_setr_pd(i * x_scale + left, (i + 1) * x_scale + left);
            const __m128d x0_vec_1 = _mm_setr_pd((i + 2) * x_scale + left, (i + 3) * x_scale + left);
            const __m128d x0_vec_2 = _mm_setr_pd((i + 4) * x_scale + left, (i + 5) * x_scale + left);
            const __m128d x0_vec_3 = _mm_setr_pd((i + 6) * x_scale + left, (i + 7) * x_scale + left);

            for (int r = 0; r < iters; ++r) {
                __m128d x2_0 = _mm_mul_pd(x_vec_0, x_vec_0);
                __m128d y2_0 = _mm_mul_pd(y_vec_0, y_vec_0);
                __m128d x2_1 = _mm_mul_pd(x_vec_1, x_vec_1);
                __m128d y2_1 = _mm_mul_pd(y_vec_1, y_vec_1);
                __m128d x2_2 = _mm_mul_pd(x_vec_2, x_vec_2);
                __m128d y2_2 = _mm_mul_pd(y_vec_2, y_vec_2);
                __m128d x2_3 = _mm_mul_pd(x_vec_3, x_vec_3);
                __m128d y2_3 = _mm_mul_pd(y_vec_3, y_vec_3);

                __m128d len_sq_0 = _mm_add_pd(x2_0, y2_0);
                __m128d cmp_0 = _mm_cmplt_pd(len_sq_0, four);
                __m128d len_sq_1 = _mm_add_pd(x2_1, y2_1);
                __m128d cmp_1 = _mm_cmplt_pd(len_sq_1, four);
                __m128d len_sq_2 = _mm_add_pd(x2_2, y2_2);
                __m128d cmp_2 = _mm_cmplt_pd(len_sq_2, four);
                __m128d len_sq_3 = _mm_add_pd(x2_3, y2_3);
                __m128d cmp_3 = _mm_cmplt_pd(len_sq_3, four);

                repeats_vec_0 = _mm_add_pd(repeats_vec_0, _mm_and_pd(cmp_0, one));
                repeats_vec_1 = _mm_add_pd(repeats_vec_1, _mm_and_pd(cmp_1, one));
                repeats_vec_2 = _mm_add_pd(repeats_vec_2, _mm_and_pd(cmp_2, one));
                repeats_vec_3 = _mm_add_pd(repeats_vec_3, _mm_and_pd(cmp_3, one));

                const int mask_0 = _mm_movemask_pd(cmp_0);
                const int mask_1 = _mm_movemask_pd(cmp_1);
                const int mask_2 = _mm_movemask_pd(cmp_2);
                const int mask_3 = _mm_movemask_pd(cmp_3);
                if (!mask_0 && !mask_1 && !mask_2 && !mask_3) break;

                __m128d xy_0 = _mm_mul_pd(x_vec_0, y_vec_0);
                y_vec_0 = _mm_add_pd(_mm_add_pd(xy_0, xy_0), y0_vec);
                x_vec_0 = _mm_add_pd(_mm_sub_pd(x2_0, y2_0), x0_vec_0);

                __m128d xy_1 = _mm_mul_pd(x_vec_1, y_vec_1);
                y_vec_1 = _mm_add_pd(_mm_add_pd(xy_1, xy_1), y0_vec);
                x_vec_1 = _mm_add_pd(_mm_sub_pd(x2_1, y2_1), x0_vec_1);

                __m128d xy_2 = _mm_mul_pd(x_vec_2, y_vec_2);
                y_vec_2 = _mm_add_pd(_mm_add_pd(xy_2, xy_2), y0_vec);
                x_vec_2 = _mm_add_pd(_mm_sub_pd(x2_2, y2_2), x0_vec_2);

                __m128d xy_3 = _mm_mul_pd(x_vec_3, y_vec_3);
                y_vec_3 = _mm_add_pd(_mm_add_pd(xy_3, xy_3), y0_vec);
                x_vec_3 = _mm_add_pd(_mm_sub_pd(x2_3, y2_3), x0_vec_3);
            }

            __m128i int_vec_0 = _mm_cvtpd_epi32(repeats_vec_0);
            __m128i int_vec_1 = _mm_cvtpd_epi32(repeats_vec_1);
            __m128i int_vec_2 = _mm_cvtpd_epi32(repeats_vec_2);
            __m128i int_vec_3 = _mm_cvtpd_epi32(repeats_vec_3);

            int *image_ptr = &image[local_j * width + i];
            _mm_storel_epi64((__m128i *)image_ptr, int_vec_0);
            _mm_storel_epi64((__m128i *)(image_ptr + 2), int_vec_1);
            _mm_storel_epi64((__m128i *)(image_ptr + 4), int_vec_2);
            _mm_storel_epi64((__m128i *)(image_ptr + 6), int_vec_3);
        }

        for (int i = (width / 8) * 8; i < width; ++i) {
            const double x0 = i * x_scale + left;
            double x = 0, y = 0;
            int repeats = 0;
            for (; repeats < iters; ++repeats) {
                const double x2 = x * x, y2 = y * y;
                if (x2 + y2 >= 4) break;
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;
            }
            image[local_j * width + i] = repeats;
        }
    }

#ifdef PROFILING
    NVTX_POP();  // end Compute_Mandelbrot
#endif

    // ========================================================================
    // Block 3: MPI Communication (Gatherv)
    // ========================================================================
#ifdef PROFILING
    temp_start = MPI_Wtime();
    NVTX_PUSH("Comm_Gatherv");
#endif

    if (my_rank == 0) {
        int *recvcounts = (int *)malloc(numtasks * sizeof(int));
        int *displacements = (int *)malloc(numtasks * sizeof(int));

        for (int rank = 0; rank < numtasks; ++rank) {
            const int count = (rank < remainder) ? (base_chunk_size + 1) : base_chunk_size;
            recvcounts[rank] = width * count;
            displacements[rank] = rank * width * base_chunk_size + std::min(rank, remainder) * width;
        }

        int *temp_image = (int *)malloc(width * height * sizeof(int));
        MPI_Gatherv(image, width * my_count, MPI_INT, temp_image, recvcounts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

        for (int rank = 0; rank < numtasks; ++rank) {
            const int count = (rank < remainder) ? (base_chunk_size + 1) : base_chunk_size;
            const int offset = displacements[rank];

            for (int local_j = 0; local_j < count; ++local_j) {
                const int global_row = rank + local_j * numtasks;
                memcpy(&global_image[global_row * width], &temp_image[offset + local_j * width], width * sizeof(int));
            }
        }

        free(temp_image);
        free(recvcounts);
        free(displacements);

    } else MPI_Gatherv(image, width * my_count, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);

#ifdef PROFILING
    comm_time += MPI_Wtime() - temp_start;
    NVTX_POP();  // end Comm_Gatherv
#endif

    // ========================================================================
    // Block 4: Output I/O
    // ========================================================================
#ifdef PROFILING
    temp_start = MPI_Wtime();
    NVTX_PUSH("IO_Write");
#endif

    if (my_rank == 0) write_png(filename, iters, width, height, global_image);

#ifdef PROFILING
    io_time += MPI_Wtime() - temp_start;
    NVTX_POP();  // end IO_Write
#endif

    free(global_image);
    free(image);

    // ========================================================================
    // Block 5: Profiling Results Summary
    // ========================================================================
#ifdef PROFILING
    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = MPI_Wtime() - total_start_time;
    double compute_time = total_time - io_time - comm_time;

    // Aggregate statistics across all processes
    double avg_total = 0.0, avg_compute = 0.0, avg_comm = 0.0, avg_io = 0.0;
    double max_total = 0.0, min_total = total_time;
    double max_compute = 0.0, min_compute = compute_time;

    MPI_Reduce(&total_time, &avg_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &avg_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &avg_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&io_time, &avg_io, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &min_total, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &min_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (my_rank == 0 && numtasks > 0) {
        avg_total /= numtasks;
        avg_compute /= numtasks;
        avg_comm /= numtasks;
        avg_io /= numtasks;

        double total_imbalance = (max_total > 0) ? (max_total - min_total) / max_total * 100 : 0.0;
        double compute_imbalance = (max_compute > 0) ? (max_compute - min_compute) / max_compute * 100 : 0.0;
        double parallel_efficiency = (avg_compute / avg_total) * 100;
        double comm_overhead = (avg_comm / avg_compute) * 100;
        double io_overhead = (avg_io / avg_compute) * 100;
        int total_cores = numtasks * num_threads;

        // === Report-friendly output ===
        printf("==========================================\n");
        printf("  MPI Processes:       %d\n", numtasks);
        printf("  OpenMP Threads:      %d\n", num_threads);
        printf("  Total Time:          %.4f s\n", avg_total);
        printf("  Compute:             %.4f s  (%.1f%%)\n", avg_compute, (avg_compute / avg_total) * 100);
        printf("  Communication:       %.4f s  (%.1f%%)\n", avg_comm, (avg_comm / avg_total) * 100);
        printf("  IO:                  %.4f s  (%.1f%%)\n\n", avg_io, (avg_io / avg_total) * 100);
        printf("  Parallel Efficiency: %.2f%%\n", parallel_efficiency);
        printf("  Comm Overhead:       %.2f%% (comm/compute)\n", comm_overhead);
        printf("  IO Overhead:         %.2f%% (io/compute)\n\n", io_overhead);
        printf("  Max Process:         %.4f s\n", max_total);
        printf("  Min Process:         %.4f s\n", min_total);
        printf("  Process Imbalance:   %.2f%%\n", total_imbalance);
        printf("  Max Thread:          %.4f s\n", max_compute);
        printf("  Min Thread:          %.4f s\n", min_compute);
        printf("  Thread Imbalance:    %.2f%%\n", compute_imbalance);
        printf("==========================================\n");
    }
#endif

    MPI_Finalize();
    return 0;
}

/**
 * @brief Writes the raw iteration data into a PNG image file.
 *
 * @param filename The name of the output PNG file.
 * @param iters The maximum number of iterations used for the calculation.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param buffer A pointer to an integer array of size width*height, where each
 *               element stores the number of iterations for the corresponding
 *               pixel.
 */
void write_png(const char *filename, int iters, int width, int height, const int *buffer) {
    // Open the file for writing in binary mode.
    FILE *fp = fopen(filename, "wb");
    // assert(fp);  // Ensure the file was opened successfully.

    // Initialize the libpng structures for writing.
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // assert(png_ptr);  // Ensure the png_struct was created.
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // assert(info_ptr);  // Ensure the info_struct was created.

    // Set up the output stream for libpng.
    png_init_io(png_ptr, fp);

    // Write the PNG header information.
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Disable PNG filtering for simplicity and speed.
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);

    // Write the info chunk to the file.
    png_write_info(png_ptr, info_ptr);

    // Set the compression level (1 is a good balance between speed and size).
    png_set_compression_level(png_ptr, 1);

    // Allocate a buffer for a single row of the image. Each pixel is 3 bytes (R, G, B).
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);

    // Iterate through each row of the image data.
    for (int y = 0; y < height; ++y) {
        // Clear the row buffer to black (0,0,0).
        memset(row, 0, row_size);

        // Iterate through each pixel in the row.
        for (int x = 0; x < width; ++x) {
            // Get the iteration count for the current pixel.
            int p = buffer[(height - 1 - y) * width + x];

            // Get a pointer to the start of the color data for this pixel.
            png_bytep color = row + x * 3;

            // If the point escaped (p != iters), assign a color to it.
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;                       // Red component
                    color[1] = color[2] = (p % 16) * 16;  // Green and Blue components
                } else {
                    color[0] = (p % 16) * 16;  // Red component
                }
            }
        }

        // Write the completed row to the PNG file.
        png_write_row(png_ptr, row);
    }

    // Free the memory for the row buffer.
    free(row);

    // Finalize the PNG file.
    png_write_end(png_ptr, NULL);

    // Clean up the libpng structures.
    png_destroy_write_struct(&png_ptr, &info_ptr);

    // Close the file.
    fclose(fp);
}
