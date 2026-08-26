#ifndef MANDELBROT_KERNEL_H
#define MANDELBROT_KERNEL_H

#include <emmintrin.h>

namespace mandelbrot {

struct View {
    int max_iterations;
    int width;
    int height;
    double left;
    double right;
    double lower;
    double upper;
};

inline void render_row(const View &view, int row_index, int *output) {
    constexpr int kValuesPerVector = 2;
    constexpr int kVectorsPerBatch = 4;
    constexpr int kBatchSize = kValuesPerVector * kVectorsPerBatch;

    const double x_scale = (view.right - view.left) / view.width;
    const double y_scale = (view.upper - view.lower) / view.height;
    const double y0 = row_index * y_scale + view.lower;

    const __m128d four = _mm_set1_pd(4.0);
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d y0_vector = _mm_set1_pd(y0);

    const int vectorized_width = view.width - view.width % kBatchSize;
    for (int column = 0; column < vectorized_width; column += kBatchSize) {
        __m128d x[kVectorsPerBatch];
        __m128d y[kVectorsPerBatch];
        __m128d x0[kVectorsPerBatch];
        __m128d counts[kVectorsPerBatch];

        for (int lane = 0; lane < kVectorsPerBatch; ++lane) {
            const int offset = column + lane * kValuesPerVector;
            x[lane] = _mm_setzero_pd();
            y[lane] = _mm_setzero_pd();
            counts[lane] = _mm_setzero_pd();
            x0[lane] = _mm_setr_pd(
                offset * x_scale + view.left,
                (offset + 1) * x_scale + view.left);
        }

        for (int iteration = 0; iteration < view.max_iterations; ++iteration) {
            int active_mask = 0;
            for (int lane = 0; lane < kVectorsPerBatch; ++lane) {
                const __m128d x_squared = _mm_mul_pd(x[lane], x[lane]);
                const __m128d y_squared = _mm_mul_pd(y[lane], y[lane]);
                const __m128d active =
                    _mm_cmplt_pd(_mm_add_pd(x_squared, y_squared), four);

                counts[lane] =
                    _mm_add_pd(counts[lane], _mm_and_pd(active, one));
                active_mask |= _mm_movemask_pd(active);

                const __m128d xy = _mm_mul_pd(x[lane], y[lane]);
                y[lane] = _mm_add_pd(_mm_add_pd(xy, xy), y0_vector);
                x[lane] = _mm_add_pd(
                    _mm_sub_pd(x_squared, y_squared), x0[lane]);
            }

            if (active_mask == 0) {
                break;
            }
        }

        for (int lane = 0; lane < kVectorsPerBatch; ++lane) {
            const __m128i packed_counts = _mm_cvtpd_epi32(counts[lane]);
            _mm_storel_epi64(
                reinterpret_cast<__m128i *>(
                    output + column + lane * kValuesPerVector),
                packed_counts);
        }
    }

    for (int column = vectorized_width; column < view.width; ++column) {
        const double x0 = column * x_scale + view.left;
        double x = 0.0;
        double y = 0.0;
        int iterations = 0;

        while (iterations < view.max_iterations && x * x + y * y < 4.0) {
            const double x_squared = x * x;
            const double y_squared = y * y;
            y = 2.0 * x * y + y0;
            x = x_squared - y_squared + x0;
            ++iterations;
        }

        output[column] = iterations;
    }
}

}  // namespace mandelbrot

#endif
