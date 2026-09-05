# Lab 3: GPU Offload

## Tasks

This lab contains two GPU programming exercises:

- run the matrix operations and activation stage of a small MNIST classifier
  with OpenACC;
- apply a Sobel edge filter to a PNG image with a CUDA kernel.

## Parallel Models

`submission/mnist.cpp` uses OpenACC data regions and parallel loops for matrix
multiplication and the sigmoid activation. `submission/sobel.cu` launches a
two-dimensional CUDA grid so threads process output pixels in parallel.

## Files

- `submission/mnist.cpp`: OpenACC inference implementation.
- `submission/sobel.cu`: CUDA Sobel implementation with PNG input and output.

The `submission/` directory preserves the submitted source artifacts. Course
build files remain private; no separate student-authored report was found in
the retained remote refs or history.

## Requirements

The MNIST program requires an OpenACC-capable C++ compiler, an NVIDIA GPU
runtime, `mnist/mnist_reader.hpp`, the MNIST dataset, and the four binary weight
files referenced by the source. Those external data and header files are not
included here, and this public tree does not provide a build recipe for that
program. With the original dependencies available, it accepts either no
arguments for the course-cluster defaults or:

```text
<mnist-binary> <mnist-data-directory> <predictions-output>
```

The Sobel program requires the CUDA toolkit, libpng, and zlib:

```bash
nvcc -O3 -std=c++11 submission/sobel.cu -o sobel -lpng -lz
./sobel <input.png> <output.png>
```
