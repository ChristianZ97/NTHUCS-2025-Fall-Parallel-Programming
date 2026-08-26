#ifndef MANDELBROT_OPTIONS_H
#define MANDELBROT_OPTIONS_H

#include "mandelbrot_kernel.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>

namespace mandelbrot {

struct Options {
    const char *output_path;
    View view;
};

inline bool parse_integer(const char *text, int *value) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > INT_MAX) {
        return false;
    }

    *value = static_cast<int>(parsed);
    return true;
}

inline bool parse_double(const char *text, double *value) {
    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }

    *value = parsed;
    return true;
}

inline bool parse_options(int argc, char **argv, Options *options) {
    if (argc != 9) {
        return false;
    }

    options->output_path = argv[1];
    return parse_integer(argv[2], &options->view.max_iterations) &&
           parse_double(argv[3], &options->view.left) &&
           parse_double(argv[4], &options->view.right) &&
           parse_double(argv[5], &options->view.lower) &&
           parse_double(argv[6], &options->view.upper) &&
           parse_integer(argv[7], &options->view.width) &&
           parse_integer(argv[8], &options->view.height) &&
           options->view.width <= INT_MAX / options->view.height;
}

}  // namespace mandelbrot

#endif
