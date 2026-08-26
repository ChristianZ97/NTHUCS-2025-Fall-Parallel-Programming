#define PNG_NO_SETJMP

#include "png_writer.h"

#include <png.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace mandelbrot {

bool write_png(
    const char *path,
    int max_iterations,
    int width,
    int height,
    const int *image) {
    FILE *file = std::fopen(path, "wb");
    if (file == nullptr) {
        return false;
    }

    png_structp png =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png == nullptr) {
        std::fclose(file);
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (info == nullptr) {
        png_destroy_write_struct(&png, nullptr);
        std::fclose(file);
        return false;
    }

    png_init_io(png, file);
    png_set_IHDR(
        png,
        info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png, 0, PNG_NO_FILTERS);
    png_set_compression_level(png, 1);
    png_write_info(png, info);

    std::vector<png_byte> pixels(static_cast<size_t>(width) * 3);
    for (int output_row = 0; output_row < height; ++output_row) {
        std::fill(pixels.begin(), pixels.end(), 0);
        const int source_row = height - 1 - output_row;

        for (int column = 0; column < width; ++column) {
            const int iterations =
                image[static_cast<size_t>(source_row) * width + column];
            if (iterations == max_iterations) {
                continue;
            }

            png_byte *color =
                pixels.data() + static_cast<size_t>(column) * 3;
            if ((iterations & 16) != 0) {
                color[0] = 240;
                color[1] = color[2] = (iterations % 16) * 16;
            } else {
                color[0] = (iterations % 16) * 16;
            }
        }

        png_write_row(png, pixels.data());
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    return std::fclose(file) == 0;
}

}  // namespace mandelbrot
