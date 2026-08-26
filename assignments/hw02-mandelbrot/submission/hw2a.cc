/*
 * This program generates a visualization of the Mandelbrot set.
 * It is a pthread implementation that calculates the set for a specified
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
#include <png.h>        // For the libpng library, used to write PNG files
#include <pthread.h>
#include <sched.h>   // For CPU affinity functions (sched_getaffinity, CPU_COUNT)
#include <stdio.h>   // For standard I/O (printf, fopen, etc.)
#include <stdlib.h>  // For general utilities (strtol, strtod, malloc, free)
#include <string.h>  // For memory manipulation (memset)

#ifdef PROFILING
#include <nvtx3/nvToolsExt.h>
#include <sys/time.h>

// ============================================================================
// NVTX Color Definitions for Performance Profiling
// ============================================================================
#define COLOR_IO 0xFF5C9FFF       // Azure Blue
#define COLOR_COMPUTE 0xFF5FD068  // Spring Green (Computation)
#define COLOR_SYNC 0xFFFF6B81     // Rose Red (Synchronization)
#define COLOR_SETUP 0xFFFFBE3D    // Sunflower Yellow
#define COLOR_DEFAULT 0xFFCBD5E0  // Soft Gray

// ============================================================================
// Smart NVTX Macro: Auto-selects color based on range name pattern
// ============================================================================
#define NVTX_PUSH(name)                                                            \
    do {                                                                           \
        uint32_t color = COLOR_DEFAULT;                                            \
        if (strstr(name, "IO_") != NULL) {                                         \
            color = COLOR_IO;                                                      \
        } else if (strstr(name, "Compute") != NULL) {                              \
            color = COLOR_COMPUTE;                                                 \
        } else if (strstr(name, "Sync") != NULL || strstr(name, "Join") != NULL) { \
            color = COLOR_SYNC;                                                    \
        } else if (strcmp(name, "Setup") == 0 || strcmp(name, "Mem_Alloc") == 0) { \
            color = COLOR_SETUP;                                                   \
        } else {                                                                   \
            color = COLOR_DEFAULT;                                                 \
        }                                                                          \
        nvtxEventAttributes_t eventAttrib = {0};                                   \
        eventAttrib.version = NVTX_VERSION;                                        \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
        eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
        eventAttrib.color = color;                                                 \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
        eventAttrib.message.ascii = name;                                          \
        nvtxRangePushEx(&eventAttrib);                                             \
    } while (0)

#define NVTX_POP() nvtxRangePop()

// Helper function to get wall time
static inline double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}
#endif

#define CHUNK_SIZE 1

pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
int next_row = 0;

typedef struct {
    int iters;
    int width;  // fix, since we divide by y
    int height;
    int my_height_start;
    int my_height_end;
    int *image;
    double left;   // Real part start (x0)
    double right;  // Real part end (x1)
    double lower;  // Imaginary part start (y0)
    double upper;  // Imaginary part end (y1)

#ifdef PROFILING
    double compute_time;
    double sync_time;
#endif

} ThreadArg;

void write_png(const char *filename, int iters, int width, int height, const int *buffer);
void *local_mandelbrot(void *argv);

int main(int argc, char *argv[]) {

#ifdef PROFILING
    double total_start_time, temp_start, io_time = 0.0;
    total_start_time = get_wall_time();
    NVTX_PUSH("Setup");
#endif

    // ========================================================================
    // Block 1: Initialization and Setup
    // ========================================================================
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

#ifdef PROFILING
    NVTX_POP();  // end Setup
    NVTX_PUSH("Mem_Alloc");
#endif

    int *image = (int *)malloc(width * height * sizeof(int));
    int num_threads = CPU_COUNT(&cpu_set);
    pthread_t threads[num_threads];
    ThreadArg t_args[num_threads];

#ifdef PROFILING
    NVTX_POP();  // end Mem_Alloc
#endif

    // ========================================================================
    // Block 2: Thread Creation and Computation
    // ========================================================================
#ifdef PROFILING
    NVTX_PUSH("Compute_Main");
    NVTX_PUSH("Thread_Creation");
#endif

    for (int i = 0; i < num_threads; i++) {
        t_args[i].iters = iters;
        t_args[i].width = width;
        t_args[i].height = height;
        t_args[i].image = image;
        t_args[i].left = left;
        t_args[i].right = right;
        t_args[i].lower = lower;
        t_args[i].upper = upper;

#ifdef PROFILING
        t_args[i].compute_time = 0.0;
        t_args[i].sync_time = 0.0;
#endif

        pthread_create(&threads[i], NULL, local_mandelbrot, (void *)&t_args[i]);
    }

#ifdef PROFILING
    NVTX_POP();  // end Thread_Creation
    NVTX_POP();  // end Compute_Main
    NVTX_PUSH("Thread_Join");
#endif

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

#ifdef PROFILING
    NVTX_POP();  // end Thread_Join
#endif

    // ========================================================================
    // Block 3: Output I/O
    // ========================================================================
#ifdef PROFILING
    temp_start = get_wall_time();
    NVTX_PUSH("IO_Write");
#endif

    write_png(filename, iters, width, height, image);

#ifdef PROFILING
    io_time += get_wall_time() - temp_start;
    NVTX_POP();  // end IO_Write
#endif

    free(image);

    // ========================================================================
    // Block 4: Profiling Results Summary
    // ========================================================================
#ifdef PROFILING
    double total_time = get_wall_time() - total_start_time;

    // Aggregate thread timings and find min/max
    double total_compute_time = 0.0, total_sync_time = 0.0;
    double max_compute = 0.0, min_compute = 1e9;
    int slowest_thread = 0, fastest_thread = 0;

    for (int i = 0; i < num_threads; i++) {
        total_compute_time += t_args[i].compute_time;
        total_sync_time += t_args[i].sync_time;
        if (t_args[i].compute_time > max_compute) {
            max_compute = t_args[i].compute_time;
            slowest_thread = i;
        }
        if (t_args[i].compute_time < min_compute) {
            min_compute = t_args[i].compute_time;
            fastest_thread = i;
        }
    }

    double avg_compute = total_compute_time / num_threads;
    double avg_sync = total_sync_time / num_threads;
    double imbalance = (max_compute > 0) ? (max_compute - min_compute) / max_compute * 100 : 0.0;
    double parallel_efficiency = (avg_compute / total_time) * 100;
    double sync_overhead = (avg_sync / avg_compute) * 100;

    // === Report-friendly output ===
    printf("==========================================\n");
    printf("  Pthreads:            %d\n", num_threads);
    printf("  Total Time:          %.4f s\n", total_time);
    printf("  Compute:             %.4f s  (%.1f%%)\n", avg_compute, (avg_compute / total_time) * 100);
    printf("  Communication:       %.4f s  (%.1f%%)\n", avg_sync, (avg_sync / total_time) * 100);
    printf("  IO:                  %.4f s  (%.1f%%)\n\n", io_time, (io_time / total_time) * 100);
    printf("  Parallel Efficiency: %.2f%%\n", parallel_efficiency);
    printf("  Comm Overhead:       %.2f%% (sync/compute)\n\n", sync_overhead);
    printf("  Max Thread (T%d):    %.4f s\n", slowest_thread, max_compute);
    printf("  Min Thread (T%d):    %.4f s\n", fastest_thread, min_compute);
    printf("  Thread Imbalance:    %.2f%% ((max-min)/max)\n", imbalance);
    printf("==========================================\n");
#endif

    return 0;
}

void *local_mandelbrot(void *argv) {

    ThreadArg *t_arg = (ThreadArg *)argv;
    const int iters = t_arg->iters;
    const int width = t_arg->width;
    const int height = t_arg->height;
    const double left = t_arg->left;
    const double right = t_arg->right;
    const double lower = t_arg->lower;
    const double upper = t_arg->upper;
    const double x_scale = (right - left) / width;
    const double y_scale = (upper - lower) / height;
    int *image = t_arg->image;

#ifdef PROFILING
    double local_compute_time = 0.0;
    double local_sync_time = 0.0;
#endif

    while (1) {

#ifdef PROFILING
        double sync_start = get_wall_time();
#endif
        pthread_mutex_lock(&task_mutex);
        const int my_start_row = next_row;
        next_row += CHUNK_SIZE;
        pthread_mutex_unlock(&task_mutex);

#ifdef PROFILING
        local_sync_time += get_wall_time() - sync_start;
#endif

        if (my_start_row >= height) break;
        const int my_end_row = (my_start_row + CHUNK_SIZE > height) ? height : my_start_row + CHUNK_SIZE;
        const __m128d four = _mm_set1_pd(4.0);
        const __m128d one = _mm_set1_pd(1.0);

#ifdef PROFILING
        double compute_start = get_wall_time();
#endif

        for (int j = my_start_row; j < my_end_row; ++j) {
            const double y0 = j * y_scale + lower;
            const __m128d y0_vec = _mm_set1_pd(y0);

            for (int i = 0; i <= width - 8; i += 8) {
                __m128d repeats_vec_0 = _mm_setzero_pd();  // [repeats[0], repeats[1]]
                __m128d repeats_vec_1 = _mm_setzero_pd();  // [repeats[2], repeats[3]]
                __m128d repeats_vec_2 = _mm_setzero_pd();  // [repeats[4], repeats[5]]
                __m128d repeats_vec_3 = _mm_setzero_pd();  // [repeats[6], repeats[7]]

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

                int *image_ptr = &image[j * width + i];
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
                image[j * width + i] = repeats;
            }
        }

#ifdef PROFILING
        local_compute_time += get_wall_time() - compute_start;
#endif

    }

#ifdef PROFILING
    t_arg->compute_time = local_compute_time;
    t_arg->sync_time = local_sync_time;
#endif

    return NULL;
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

    // Set the compression level (1 is a good balance between speed and
    // size).
    png_set_compression_level(png_ptr, 1);

    // Allocate a buffer for a single row of the image. Each pixel is 3
    // bytes (R, G, B).
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

            // Get a pointer to the start of the color data for this
            // pixel.
            png_bytep color = row + x * 3;

            // If the point escaped (p != iters), assign a color to
            // it.
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
