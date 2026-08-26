# Parallel Programming Labs

This directory contains the retained implementation work from five course labs.
Each lab README describes the task, parallel programming model, available files,
and the environment needed to build or reproduce the work.

| Lab | Topic | Parallel model |
| --- | --- | --- |
| [Lab 1](lab01-mpi-circle-counting/) | Integer circle counting | MPI |
| [Lab 2](lab02-parallel-circle-counting/) | Shared-memory and hybrid circle counting | Pthreads, OpenMP, MPI + OpenMP |
| [Lab 3](lab03-gpu-offload/) | Neural-network inference and Sobel filtering | OpenACC, CUDA |
| [Lab 4](lab04-cuda-sobel/) | CUDA Sobel filtering | CUDA |
| [Lab 6](lab06-flash-attention-benchmark/) | Attention-kernel benchmarking | CUDA through PyTorch and FlashAttention |

Where present, `submission/` preserves the source and build artifacts that were
submitted for the lab. Reports containing personal or course identifiers are
kept only in a private archival snapshot and are not included in this
public-facing tree.
