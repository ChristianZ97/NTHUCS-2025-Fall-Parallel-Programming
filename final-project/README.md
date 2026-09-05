# N-Body Simulation on CPU, CUDA, and HIP

This team project simulates gravitational interaction among many bodies with a
velocity-Verlet integrator. A scalar C implementation provides the reference
path, while CUDA and HIP implementations keep body state in structure-of-arrays
form and tile the all-pairs force calculation through GPU shared memory.

![Simulation of 16,384 bodies](presentation/16384.gif)

## Implementation

Every time step applies the same four operations:

1. Advance velocity by half a step using the current acceleration.
2. Advance position by a full step.
3. Recompute acceleration from the new positions.
4. Complete the velocity update with the new acceleration.

The CPU version stores bodies as an array of structures. The GPU versions use
separate position, velocity, mass, and acceleration arrays so adjacent threads
access adjacent elements. The force kernel stages a tile of bodies in shared
memory before accumulating pairwise interactions.

## Build

The Makefile exposes independent targets because the three toolchains are not
usually installed on the same machine:

```bash
make c       # gcc -> nbody_c
make cuda    # nvcc -> nbody_cu
make hip     # hipcc -> nbody_hip
```

`make`, `make debug`, and `make prof` build all standard, debug, or profiling
targets. The checked-in flags target CUDA `sm_61` and ROCm `gfx908`; adjust
those architecture flags for other GPUs.

## Run

Generate an input or use one of the checked-in test cases:

```bash
uv run --with numpy gen_input.py 2048 input.txt
./nbody_c input.txt cpu.csv
```

The generator samples random initial conditions and adds one central body when
`CENTRAL_MASS` is nonzero. Pass the desired number of orbiting bodies as its
first argument.

The original cluster commands were:

```bash
srun -p nvidia -N1 -n1 --gres=gpu:1 ./nbody_cu input.txt cuda.csv
srun -p amd -N1 -n1 --gres=gpu:1 ./nbody_hip input.txt hip.csv
```

Input files contain the body count, simulation parameters, and one line of
position, velocity, and mass values per body. Each executable writes the same
trajectory CSV schema:

```text
step,t,id,x,y,z,vx,vy,vz,m
```

## Validate

Compare a GPU trajectory with the CPU result. Rows must align by simulation
step and body identifier; positions and velocities are checked with an absolute
tolerance of `1e-5`.

```bash
uv run --with pandas --with numpy \
  compare_nbody.py cpu.csv cuda.csv
```

The `testcases/` directory includes two inputs and their expected trajectories.
`run_judge.py` and `run_judge_amd.py` automate repeated CUDA and HIP comparisons
on the course cluster.

## Visualize

Create a GIF from any trajectory:

```bash
uv run --with pandas --with numpy --with matplotlib --with pillow \
  animate.py cuda.csv simulation.gif
```

The recorded profiling and timing figures are under `presentation/`. They are
hardware-specific experiment artifacts, not portable performance guarantees.

## Original presentation

The final 27-slide team deck is preserved byte-for-byte at
[`submission/presentation.pptx`](submission/presentation.pptx). It names the
three contributors with their submitted student identifiers and retains its
original Office authorship metadata. Its hash provenance is recorded in the
root [`SUBMISSIONS.md`](../SUBMISSIONS.md). The replaced draft deck and the
presentation video remain private.

## Layout

```text
nbody.c, nbody.cu, nbody.hip   Reference and GPU implementations
compare_nbody.py               Numerical comparison utility
gen_input.py                   Random initial-condition generator
animate.py                     Trajectory renderer
testcases/                     Inputs and expected CSV trajectories
variants/                      GPU design snapshots grouped by optimization idea
presentation/                  Recorded plots and animation
submission/presentation.pptx  Byte-exact final team presentation
```

The variant directories preserve intermediate design exploration. The root
`nbody.cu` and `nbody.hip` files are the final integration points.
