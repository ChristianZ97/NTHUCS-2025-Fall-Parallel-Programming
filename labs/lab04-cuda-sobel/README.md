# Lab 4: CUDA Sobel Filtering

## Task

Apply a Sobel edge filter to a PNG image on an NVIDIA GPU.

## Parallel Model

`submission/sobel-opt.cu` launches a two-dimensional CUDA grid with 16 by 16
thread blocks. Each in-range thread computes one output pixel, while color
channels are handled within that thread.

## Files

- `submission/sobel-opt.cu`: CUDA implementation and PNG input/output helpers.
- `submission/Makefile`: build rule for the `sobel-opt` executable.

The `submission/` directory preserves the submitted source and build artifacts.
No separate student-authored report was found in the retained remote refs or
history.

## Build And Run

The current Makefile uses `nvcc`, targets compute capability `sm_61`, and links
against libpng and zlib. A compatible NVIDIA GPU, CUDA toolkit, and development
packages for both libraries are required.

```bash
make -C submission
submission/sobel-opt <input.png> <output.png>
```

The Makefile also records HIP compiler flags, but its current `sobel-opt` target
uses the CUDA compiler.
