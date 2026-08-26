# Parallel Programming Portfolio

Coursework from CS542200 Parallel Programming, Fall 2025. The repository
collects implementations and experiment artifacts for distributed CPU
programming, GPU kernels, and communication runtimes.

![N-body simulation with 16,384 bodies](final-project/presentation/16384.gif)

## Projects

| Project | Parallel model | Focus |
| --- | --- | --- |
| [HW1: Odd-Even Sort](assignments/hw01-odd-even-sort/) | MPI, MPI-IO | Distributed sorting, merge-split communication, and scaling experiments |
| [HW2: Mandelbrot Set](assignments/hw02-mandelbrot/) | Pthreads, MPI + OpenMP | Dynamic work distribution and SIMD-assisted rendering |
| [HW3: Blocked Floyd-Warshall](assignments/hw03-blocked-floyd-warshall/) | OpenMP, CUDA, HIP | Blocked all-pairs shortest paths on one or more GPUs |
| [HW4: FlashAttention](assignments/hw04-flash-attention/) | CUDA, HIP | Tiled attention kernels and GPU performance experiments |
| [HW5: UCX](assignments/hw05-ucx/) | UCX, MPI | Transport-selection instrumentation and OSU microbenchmarks |
| [Final Project: N-Body Simulation](final-project/) | C, CUDA, HIP | Velocity-Verlet integration with shared-memory GPU tiling |

Shorter exercises are indexed under [labs](labs/). They cover MPI, Pthreads,
OpenMP, OpenACC, CUDA, and FlashAttention benchmarking.

## Repository Layout

```text
assignments/   Portfolio implementations and selected result summaries
labs/          Focused parallel-programming exercises
final-project/ Team N-body implementation, validation tools, and visualizations
```

Each assignment presents a cleaned canonical implementation separately from
the unmodified grading-time artifact in `submission/`. Representative
measurements stay beside the code; development variants, raw logs, and
parameter-sweep sources remain in the private legacy history. There is no
single root build: enter a project directory and follow its README.

## Reproducibility

Most programs target a course HPC cluster. MPI launch commands, Slurm
partitions, CUDA compute capabilities, ROCm targets, and profiler availability
are environment-specific. The per-project READMEs record the original build
and run interface; reproduce performance measurements on comparable hardware.

The public-facing tree omits course handouts, report PDFs containing personal
identifiers, raw profiler captures, and large presentation media. Those files
are not required to inspect the implementations.

## Academic Use

This repository is published as a portfolio and learning reference. Current
students should follow their institution's academic-integrity rules and should
not submit this work as their own. Unless a file states otherwise, no license
is granted for copying or redistribution.
