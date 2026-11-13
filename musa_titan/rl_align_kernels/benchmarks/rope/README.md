# Paired RoPE benchmark

Run from the repository root on an idle MUSA device:

```bash
PYTHONPATH=. MUSA_VISIBLE_DEVICES=1 python3 \
  musa_titan/rl_align_kernels/benchmarks/rope/benchmark_rope.py \
  --warmup 10 --iterations 100
```

This measures the standalone paired Q/K **forward operator**, with BF16,
head dimension 128, Q/K heads 32/4, and T=8,32,512,1024,4096,16384.
The `[262400,128]` cos/sin cache is prepared once outside timing.
It does not benchmark cache generation, backward, or the complete Attention
forward, and does not modify the production cache generator.

Each call must contain exactly one `_rope_qk` device kernel. Profiler host-launch
correlations associate device events with calls. Results report mean, median,
minimum, and maximum device kernel duration in microseconds; CPU launch time,
kernel gaps, compilation, and input preparation are excluded.

Effective GB/s counts Q/K reads and writes, selected cache rows, and position
indices. It is logical bandwidth, not measured DRAM traffic. Inputs are reused
without flushing device caches.

Traces, per-call samples, source hashes, software versions, and summary JSON
are written to a fresh ignored `results/profiler-*` directory, or to a new
directory specified with `--output-dir`. Measurements from occupied GPUs must
not be presented as performance acceptance.
