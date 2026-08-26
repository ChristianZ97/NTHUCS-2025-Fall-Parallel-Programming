# Lab 2: Parallel Circle Counting

## Task

Compute the same integer circle count as Lab 1 while comparing shared-memory
and hybrid parallel implementations. Each program prints four times the
quadrant count modulo `k`.

## Parallel Models

- `lab2_pthread.cc` uses 12 POSIX threads with contiguous x-coordinate ranges.
- `lab2_omp.cc` uses a 12-thread OpenMP region and a static loop schedule.
- `lab2_hybrid.cc` distributes x-coordinate ranges across MPI ranks, uses
  OpenMP within each rank, and combines rank-local counts with `MPI_Allreduce`.

## Files

- `submission/lab2_pthread.cc`: POSIX threads implementation.
- `submission/lab2_omp.cc`: OpenMP implementation.
- `submission/lab2_hybrid.cc`: MPI and OpenMP implementation.
- `submission/Makefile`: build rules for all three programs.
- `submission/module.list`: OpenMPI and UCX module configuration from the
  course cluster.

The `submission/` directory preserves the submitted source and build artifacts.
The report containing personal or course identifiers is kept only in the
private archival snapshot.

## Build And Run

The shared-memory programs require GCC-compatible C++ support for Pthreads and
OpenMP. The hybrid program also requires an MPI implementation.

```bash
make -C submission

submission/lab2_pthread <radius> <modulus>
submission/lab2_omp <radius> <modulus>
mpirun -np <ranks> submission/lab2_hybrid <radius> <modulus>
```
