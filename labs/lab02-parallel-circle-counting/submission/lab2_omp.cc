#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS 12

int main(int argc, char *argv[]) {

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	const unsigned long long r = atoll(argv[1]);
	const unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	
	const unsigned long long r_squared = r * r;

	#pragma omp parallel num_threads(NUM_THREADS)
	{
		unsigned long long local_pixels = 0;

		#pragma omp for schedule(static) nowait
			for (unsigned long long x = 0; x < r; x++) {

			    const unsigned long long x_squared = x * x;
			    if (x_squared >= r_squared) continue;

			    unsigned long long y = (unsigned long long)sqrtl((long double)(r_squared - x_squared));
			    if (y * y < r_squared - x_squared) y++;

				local_pixels += y;
				if (x % 10000 == 0) local_pixels %= k;
			}

		local_pixels %= k;

	    #pragma omp critical
	    {
	        pixels += local_pixels;
	        pixels %= k;
	    }
	}

	printf("%llu\n", (4 * pixels) % k);
    return 0;    
}