#ifndef MANDELBROT_PNG_WRITER_H
#define MANDELBROT_PNG_WRITER_H

namespace mandelbrot {

bool write_png(
    const char *path,
    int max_iterations,
    int width,
    int height,
    const int *image);

}  // namespace mandelbrot

#endif
