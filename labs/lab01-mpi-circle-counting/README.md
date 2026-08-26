# Lab 1: MPI Circle Counting

## Task

Count integer lattice points in one quadrant of a circle with radius `r`, then
print four times that count modulo `k`.

## Parallel Model

`submission/lab1.cc` partitions the integer x-coordinate range into contiguous
chunks, assigns one chunk to each MPI rank, and combines the partial counts with
`MPI_Reduce`. Rank 0 prints the result.

## Files

- `submission/lab1.cc`: MPI implementation.
- `submission/Makefile`: C++17 build using `mpicxx`.
- `submission/module.list`: module search paths from the course cluster.

The `submission/` directory preserves the submitted source and build artifacts.
The report containing personal or course identifiers is kept only in the
private archival snapshot.

## Build And Run

An MPI implementation with `mpicxx` and `mpirun` is required.

```bash
make -C submission
mpirun -np <ranks> submission/lab1 <radius> <modulus>
```

The program writes one unsigned integer to standard output.
