#ifndef HW3_IO_UTILS_H
#define HW3_IO_UTILS_H

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static inline FILE *open_file_or_die(const char *path, const char *mode) {
    FILE *file = fopen(path, mode);
    if (file == NULL) {
        perror(path);
        exit(EXIT_FAILURE);
    }
    return file;
}

static inline void read_exact_or_die(
    void *buffer,
    size_t element_size,
    size_t element_count,
    FILE *file,
    const char *path,
    const char *description) {
    if (fread(buffer, element_size, element_count, file) != element_count) {
        fprintf(stderr, "Failed to read %s from '%s'\n", description, path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
}

static inline void write_exact_or_die(
    const void *buffer,
    size_t element_size,
    size_t element_count,
    FILE *file,
    const char *path) {
    if (fwrite(buffer, element_size, element_count, file) != element_count) {
        fprintf(stderr, "Failed to write distance matrix to '%s'\n", path);
        fclose(file);
        exit(EXIT_FAILURE);
    }
}

static inline void close_file_or_die(FILE *file, const char *path) {
    if (fclose(file) != 0) {
        perror(path);
        exit(EXIT_FAILURE);
    }
}

static inline void validate_graph_header_or_die(
    int vertices,
    int edges,
    int max_vertices,
    const char *path) {
    if (vertices <= 0 || edges < 0 || (max_vertices > 0 && vertices > max_vertices)) {
        fprintf(
            stderr,
            "Invalid graph header in '%s': V=%d, E=%d\n",
            path,
            vertices,
            edges);
        exit(EXIT_FAILURE);
    }
}

static inline void validate_edge_or_die(
    const int edge[3],
    int vertices,
    const char *path) {
    if (edge[0] < 0 || edge[0] >= vertices || edge[1] < 0 || edge[1] >= vertices) {
        fprintf(
            stderr,
            "Invalid edge endpoint in '%s': (%d, %d)\n",
            path,
            edge[0],
            edge[1]);
        exit(EXIT_FAILURE);
    }
}

static inline int padded_width_or_die(int vertices, int block_size, const char *path) {
    const size_t padded =
        ((size_t)vertices + (size_t)block_size - 1) / (size_t)block_size *
        (size_t)block_size;
    if (padded > (size_t)INT_MAX) {
        fprintf(stderr, "Padded graph width is too large in '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    return (int)padded;
}

static inline size_t matrix_bytes_or_die(int width, const char *path) {
    if (width <= 0) {
        fprintf(stderr, "Invalid distance matrix width in '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    const size_t side = (size_t)width;
    const size_t max_size = (size_t)-1;
    if (side > max_size / side) {
        fprintf(stderr, "Distance matrix is too large in '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    const size_t elements = side * side;
    if (elements > max_size / sizeof(int)) {
        fprintf(stderr, "Distance matrix is too large in '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    return elements * sizeof(int);
}

#endif
