# Lab 6: FlashAttention Benchmark

## Task

Measure forward and backward execution time, calculated throughput, and peak
allocated GPU memory for a PyTorch attention implementation and FlashAttention
v1 across changes in batch size, sequence length, head count, embedding
dimension, and causal masking.

## Parallel Model

Both implementations execute on a CUDA GPU. The benchmark uses PyTorch
autograd, CUDA events, and repeated forward and backward passes. The
FlashAttention path uses packed QKV input and the unpadded FlashAttention v1
interface.

## Files

- `benchmark.ipynb`: notebook used to configure and run the experiments.
- `benchmark.py`: notebook export; it retains notebook shell directives and is
  not a standalone Python script.
- `results/benchmark-results.csv`: recorded benchmark data.
- `results/plot.py`: script that reads the CSV and recreates the figures.
- `results/plot_results/`: generated result figures.
- `submission/report.pdf`: byte-exact submitted report.

The retained files preserve the submitted benchmark and result artifacts. The
byte-exact [`submission/report.pdf`](submission/report.pdf) intentionally
retains its original author identity; hash provenance is recorded in the root
[`SUBMISSIONS.md`](../../SUBMISSIONS.md).

## Requirements

Re-running the benchmark requires Jupyter, an NVIDIA CUDA GPU, a
CUDA-compatible PyTorch installation, `flash-attn==1.0.9`, `einops`, and the
experiment-tracking client imported by the notebook. Run the notebook
interactively because its exported `.py` file contains notebook-only `!pip`
commands.

Recreating the figures requires pandas, Matplotlib, Seaborn, and NumPy. The
plotting script expects to run from the `results/` directory:

```bash
cd results
python plot.py
```

## Representative Figures

- [Sequence length versus total time](results/plot_results/01_seq_len_vs_time.png)
- [FlashAttention v1 speedup over PyTorch](results/plot_results/04_flash1_speedup.png)
- [Peak allocated memory comparison](results/plot_results/08_memory_usage.png)
- [Forward and backward throughput comparison](results/plot_results/12_efficiency_comparison.png)
