# Assignments

This directory contains five assignments from CS542200 Parallel Programming.
The projects cover distributed CPU programs, shared-memory parallelism, CUDA and
HIP kernels, and UCX communication experiments.

| Assignment | Topic | Parallel model or platform |
| --- | --- | --- |
| [HW1](hw01-odd-even-sort/) | Odd-even sort | MPI and MPI-IO |
| [HW2](hw02-mandelbrot/) | Mandelbrot rendering | Pthreads, MPI, and OpenMP |
| [HW3](hw03-blocked-floyd-warshall/) | All-pairs shortest paths | OpenMP, CUDA, and HIP |
| [HW4](hw04-flash-attention/) | Exact attention | CUDA and HIP |
| [HW5](hw05-ucx/) | UCX transport behavior | Open MPI, UCX, and OSU benchmarks |

## Repository Convention

Each assignment exposes a canonical portfolio implementation under `src/`, or
an apply-ready patch when the work modifies an external project. Where the
source is standalone, the root Makefile builds that version by default. The
corresponding `submission/` directory preserves the grading-time artifact
without cleanup, so profiling instrumentation and report-specific output
remain traceable without obscuring the primary code.

Compact result directories keep only representative tables and figures.
Development variants and raw experiment output remain in the private legacy
history. These figures document the grading-time implementation lineage; the
portfolio cleanup has not been presented as a fresh benchmark run.

Course handouts, private test cases, raw profiler captures, and reports whose
filenames or contents expose student identifiers are not part of the public
tree. They remain available in the private legacy snapshot.

The build files retain the compiler and architecture settings used on the
course cluster. Reproducing a run elsewhere may require different GPU targets,
MPI launch commands, or module configuration.
