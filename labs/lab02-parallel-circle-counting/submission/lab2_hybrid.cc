#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <algorithm>

#define NUM_THREADS 12

int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);
    int numtasks = 0, my_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	const unsigned long long r = atoll(argv[1]);
	const unsigned long long k = atoll(argv[2]);

	const unsigned long long base_chunk_size = r / numtasks;
	const unsigned long long remainder = r % numtasks;
	const unsigned long long my_count = (my_rank < remainder) ? base_chunk_size + 1 : base_chunk_size;
    const unsigned long long my_start = (my_rank * base_chunk_size + std::min((const unsigned long long)my_rank, remainder));

	unsigned long long pixels = 0;
	
	const unsigned long long r_squared = r * r;

	#pragma omp parallel num_threads(NUM_THREADS)
	{
		unsigned long long local_pixels = 0;

		#pragma omp for schedule(static) nowait
			for (unsigned long long x = my_start; x < my_start + my_count; x++) {

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

	unsigned long long global_pixels = 0;
	MPI_Allreduce(&pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	if (my_rank == 0) printf("%llu\n", (4 * global_pixels) % k);

    MPI_Finalize();
    return 0;
}