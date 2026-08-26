#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mandelbrot_kernel.h"
#include "options.h"
#include "png_writer.h"

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct WorkerContext {
    const mandelbrot::View *view;
    int *image;
    std::atomic<int> *next_row;
};

int available_cpu_count() {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        return 1;
    }
    return std::max(1, CPU_COUNT(&cpu_set));
}

void *render_rows(void *opaque_context) {
    auto *context = static_cast<WorkerContext *>(opaque_context);
    const mandelbrot::View &view = *context->view;

    while (true) {
        const int row =
            context->next_row->fetch_add(1, std::memory_order_relaxed);
        if (row >= view.height) {
            break;
        }

        mandelbrot::render_row(
            view, row, context->image + static_cast<size_t>(row) * view.width);
    }

    return nullptr;
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
    mandelbrot::Options options;
    if (!mandelbrot::parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const mandelbrot::View &view = options.view;
    std::vector<int> image(
        static_cast<size_t>(view.width) * static_cast<size_t>(view.height));

    std::atomic<int> next_row{0};
    WorkerContext context{&view, image.data(), &next_row};
    const int thread_count = std::min(available_cpu_count(), view.height);
    std::vector<pthread_t> threads(thread_count);

    int created_threads = 0;
    for (; created_threads < thread_count; ++created_threads) {
        if (pthread_create(
                &threads[created_threads], nullptr, render_rows, &context) != 0) {
            break;
        }
    }
    for (int index = 0; index < created_threads; ++index) {
        pthread_join(threads[index], nullptr);
    }

    if (created_threads != thread_count) {
        std::fprintf(stderr, "Failed to create all worker threads\n");
        return EXIT_FAILURE;
    }
    if (!mandelbrot::write_png(
            options.output_path,
            view.max_iterations,
            view.width,
            view.height,
            image.data())) {
        std::fprintf(stderr, "Failed to write %s\n", options.output_path);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
