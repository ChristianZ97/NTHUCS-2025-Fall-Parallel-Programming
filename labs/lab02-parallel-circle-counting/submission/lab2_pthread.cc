#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS 12

void *calculate_pixels (void *argv);

typedef struct {
	int tid;
    unsigned long long r;
    unsigned long long k;
    unsigned long long result;
} ThreadArg;

int main(int argc, char *argv[]) {

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	const unsigned long long r = atoll(argv[1]);
	const unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	pthread_t threads[NUM_THREADS];
	ThreadArg t_args[NUM_THREADS];

	for (int i = 0; i < NUM_THREADS; i++) {
		t_args[i].tid = i;
		t_args[i].r = r;
		t_args[i].k = k;
		t_args[i].result = 0;
		pthread_create(&threads[i], NULL, calculate_pixels, (void *)&t_args[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
		pixels += t_args[i].result;
		pixels %= k;
	}
	
	printf("%llu\n", (4 * pixels) % k);
	pthread_exit(NULL);
    return 0;
}

void *calculate_pixels(void *argv) {

	ThreadArg *t_arg = (ThreadArg *)argv;
	const int tid = t_arg->tid;
	const unsigned long long r = t_arg->r;
	const unsigned long long k = t_arg->k;
	unsigned long long local_pixels = 0;

	const unsigned long long my_start = tid * (r / NUM_THREADS);
	const unsigned long long my_end = (tid == NUM_THREADS - 1) ? r : (tid + 1) * (r / NUM_THREADS);

    const unsigned long long r_squared = r * r;
	for (unsigned long long x = my_start; x < my_end; x++) {
		
	    const unsigned long long x_squared = x * x;
	    if (x_squared >= r_squared) continue;
	    
	    unsigned long long y = (unsigned long long)sqrtl((long double)(r_squared - x_squared));
	    if (y * y < r_squared - x_squared) y++;
	    
	    local_pixels = (local_pixels + y) % k;
	}


	t_arg->result = local_pixels;
	pthread_exit(NULL);
}
