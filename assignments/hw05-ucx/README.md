# HW5: UCX Transport Study

## Objective

Trace how UCX selects communication transports, record the selected transport
when endpoint configurations are created, and compare UCX transport choices
with OSU point-to-point and one-sided benchmarks.

## Implementation

`patches/ucx-transport-info.patch` records changes to UCX's
`src/ucp/core/ucp_worker.c` and `src/ucs/config/parser.c`. The changes print the
configured `UCX_TLS` value and the transport description assembled for a UCP
endpoint configuration. An unset `UCX_TLS` is reported as `<unset>` instead of
passing a null pointer to formatted output.

The benchmark scripts compare `ud_verbs` with UCX's automatic selection for
single-node runs. A separate Slurm batch script records two-node latency.

The patch targets the `pp2024` branch of
[`NTHU-LSALAB/UCX-lsalab`](https://github.com/NTHU-LSALAB/UCX-lsalab) at commit
[`84e459e73df4f02aecd044c44e4584d88f4b9b0e`](https://github.com/NTHU-LSALAB/UCX-lsalab/commit/84e459e73df4f02aecd044c44e4584d88f4b9b0e).
It is diagnostic instrumentation rather than a production logging interface.
UCX only reaches the modified transport-reporting path when protocol v2 is
disabled (`proto_enable=false`).

## Layout

- `patches/ucx-transport-info.patch`: canonical, directly applicable patch
- `submission/hw5.diff`: exact grading-time diff retained for provenance
- `submission/report.md`: identifier-free public copy of the written analysis
- `run_osu.sh`: single-node OSU point-to-point and one-sided benchmark runner
- `run_multi_node.batch`: two-node OSU latency job
- `osu_results/`: single-node CSV tables
- `osu_results_multi/`: multi-node latency table
- `plot.py`: figure-generation script
- `plots/`: generated comparison figures and UCX architecture diagram

## Reproduction

The UCX source tree and OSU executables are not vendored here. Apply the
canonical patch to the pinned course fork before rebuilding UCX:

```bash
git clone https://github.com/NTHU-LSALAB/UCX-lsalab.git
cd UCX-lsalab
git checkout 84e459e73df4f02aecd044c44e4584d88f4b9b0e
git apply ../patches/ucx-transport-info.patch
```

On the original course environment, the recorded benchmark workflow was:

```bash
module load openmpi/ucx-pp
bash run_osu.sh
sbatch run_multi_node.batch
python plot.py
```

`plot.py` requires pandas, Matplotlib, and Seaborn. The two runner scripts
contain original `$HOME/hw5` paths and expect the course UCX build and `mpiucx`
launcher at those locations. Use the canonical patch for reproduction;
`submission/hw5.diff` contains the terminal color escapes present in the
grading-time artifact. The canonical patch also includes the null-safe
environment output and formatting cleanup documented above.

## Results

`osu_results/` contains latency and bandwidth tables for point-to-point and
one-sided operations under both tested transport settings. The runner can
regenerate verbose logs when repeating the experiment.
`osu_results_multi/multi_node_latency.csv` adds the two-node latency series.
The corresponding plots are kept in `plots/`.

## Environment

Rebuilding the patched library requires the course UCX checkout and its build
configuration. Re-running the measurements also requires Open MPI with UCX,
the OSU micro-benchmarks, access to the configured InfiniBand device, and
Slurm for the multi-node job.

`submission/` preserves the grading-time patch and an identifier-free report
copy. Original report artifacts containing a student identifier remain only on
the private legacy snapshot.
