#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mandelbrot_kernel.h"
#include "options.h"
#include "png_writer.h"

#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int available_cpu_count() {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        return 1;
    }
    return std::max(1, CPU_COUNT(&cpu_set));
}

void print_usage(const char *program) {
    std::fprintf(
        stderr,
        "Usage: %s <output.png> <iterations> <left> <right> "
        "<lower> <upper> <width> <height>\n",
        program);
}

}  // namespace

int main(int argc, char **argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(
            &argc,
            &argv,
            MPI_THREAD_FUNNELED,
            &provided_thread_level) != MPI_SUCCESS) {
        std::fprintf(stderr, "Failed to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int rank_count = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(
                stderr,
                "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    mandelbrot::Options options;
    if (!mandelbrot::parse_options(argc, argv, &options)) {
        if (rank == 0) {
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const mandelbrot::View &view = options.view;
    const int rows_per_rank = view.height / rank_count;
    const int extra_rows = view.height % rank_count;
    const int local_row_count =
        rows_per_rank + (rank < extra_rows ? 1 : 0);

    std::vector<int> local_image(
        static_cast<size_t>(view.width) * local_row_count);
    const int thread_count = available_cpu_count();

#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int local_row = 0; local_row < local_row_count; ++local_row) {
        const int global_row = rank + local_row * rank_count;
        mandelbrot::render_row(
            view,
            global_row,
            local_image.data() + static_cast<size_t>(local_row) * view.width);
    }

    std::vector<int> global_image;
    std::vector<int> receive_counts;
    std::vector<int> displacements;
    std::vector<int> gathered_image;

    if (rank == 0) {
        global_image.resize(
            static_cast<size_t>(view.width) * view.height);
        gathered_image.resize(global_image.size());
        receive_counts.resize(rank_count);
        displacements.resize(rank_count);

        int offset = 0;
        for (int source_rank = 0; source_rank < rank_count; ++source_rank) {
            const int source_rows =
                rows_per_rank + (source_rank < extra_rows ? 1 : 0);
            receive_counts[source_rank] = view.width * source_rows;
            displacements[source_rank] = offset;
            offset += receive_counts[source_rank];
        }
    }

    MPI_Gatherv(
        local_image.data(),
        view.width * local_row_count,
        MPI_INT,
        gathered_image.data(),
        receive_counts.data(),
        displacements.data(),
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    if (rank == 0) {
        for (int source_rank = 0; source_rank < rank_count; ++source_rank) {
            const int source_rows =
                rows_per_rank + (source_rank < extra_rows ? 1 : 0);
            const int source_offset = displacements[source_rank];

            for (int local_row = 0; local_row < source_rows; ++local_row) {
                const int global_row =
                    source_rank + local_row * rank_count;
                const int *source =
                    gathered_image.data() + source_offset +
                    static_cast<size_t>(local_row) * view.width;
                int *destination =
                    global_image.data() +
                    static_cast<size_t>(global_row) * view.width;
                std::memcpy(
                    destination,
                    source,
                    static_cast<size_t>(view.width) * sizeof(int));
            }
        }

        if (!mandelbrot::write_png(
                options.output_path,
                view.max_iterations,
                view.width,
                view.height,
                global_image.data())) {
            std::fprintf(stderr, "Failed to write %s\n", options.output_path);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
