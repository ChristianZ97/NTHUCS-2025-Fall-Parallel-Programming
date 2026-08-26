#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>

int main(int argc, char** argv)
{
	int rc = MPI_Init(&argc, &argv);
        if (rc != MPI_SUCCESS) {
        	printf ("Error starting MPI program. Terminating.\n");
        	MPI_Abort (MPI_COMM_WORLD, rc);
        }

	int numtasks, rank;
        MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // printf ("Number of tasks= %d My rank= %d\n", numtasks, rank);

	if (argc != 3) {
		if (rank == 0) {
			fprintf(stderr, "must provide exactly 2 arguments!\n");
		}
		MPI_Finalize();
		return 1;
	}

	unsigned long long r = atoll(argv[1]),
		           k = atoll(argv[2]),
			   pixels = 0;

	unsigned long long chunk = r / numtasks;
	unsigned long long start = rank * chunk;
	unsigned long long end = (rank == numtasks - 1) ? r : (start + chunk);

	for (unsigned long long x = start; x < end; x++)
	{
		unsigned long long y = ceil(sqrtl(r * r - x * x));
		pixels += y;
		pixels %= k;
	}

	unsigned long long total = 0;
	MPI_Reduce(&pixels, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("%llu\n", (4 * total) % k);
	}


	MPI_Finalize();
	return 0;
}

