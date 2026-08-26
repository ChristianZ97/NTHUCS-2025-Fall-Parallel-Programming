// hw3-1.cc

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREAD 12
#define INF ((1 << 30) - 1)

#define MAX_V 50010
#define BLOCKING_FACTOR 512

void input(char *infile);
void output(char *outfile);

void block_FW();
int ceil_div(int a, int b);
void cal(int B, int r, int block_start_x, int block_start_y, int block_width, int block_height);

int V, E;
static int D[MAX_V][MAX_V];

int main(int argc, char *argv[]) {

    input(argv[1]);

    block_FW();

    output(argv[2]);

    return 0;
}

void input(char *infile) {
    FILE *file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i == j) {
                D[i][j] = 0;
            } else {
                D[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[pair[0]][pair[1]] = pair[2];
    }

    fclose(file);
}

void output(char *outfile) {
    FILE *f = fopen(outfile, "w");
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (D[i][j] >= INF) D[i][j] = INF;
        }
        fwrite(D[i], sizeof(int), V, f);
    }
    fclose(f);
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void block_FW() {
    const int round = ceil_div(V, BLOCKING_FACTOR);

    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);

        /* Phase 1 */
        cal(BLOCKING_FACTOR, r, r, r, 1, 1);

        /* Phase 2 */
        cal(BLOCKING_FACTOR, r, r, 0, r, 1);
        cal(BLOCKING_FACTOR, r, r, r + 1, round - r - 1, 1);
        cal(BLOCKING_FACTOR, r, 0, r, 1, r);
        cal(BLOCKING_FACTOR, r, r + 1, r, 1, round - r - 1);

        /* Phase 3 */
        cal(BLOCKING_FACTOR, r, 0, 0, r, r);
        cal(BLOCKING_FACTOR, r, 0, r + 1, round - r - 1, r);
        cal(BLOCKING_FACTOR, r, r + 1, 0, r, round - r - 1);
        cal(BLOCKING_FACTOR, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(int B, int r, int block_start_x, int block_start_y, int block_width, int block_height) {

    const int block_end_x = block_start_x + block_height;
    const int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            for (int k = r * B; k < (r + 1) * B && k < V; ++k) {

                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > V) block_internal_end_x = V;
                if (block_internal_end_y > V) block_internal_end_y = V;

#pragma omp parallel for num_threads(NUM_THREAD) schedule(dynamic, 12)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (D[i][k] + D[k][j] < D[i][j]) {
                            D[i][j] = D[i][k] + D[k][j];
                        }
                    }
                }
            }
        }
    }
}
