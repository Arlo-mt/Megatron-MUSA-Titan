# RoPE correctness tests

Run from the repository root on MUSA:

```bash
PYTHONPATH=. python3 -m pytest musa_titan/rl_align_kernels/tests/ops/rope -v -ra
```

Tests import `musa_titan.rl_align_kernels.kernels.rope`, verify the source
location, and record source hashes and software versions. MUSA is required;
there is no CPU operator fallback or skipped acceptance.

The cache is test input prepared outside the operator. This suite does not
replace or validate Megatron's cache generator.

## References

Forward uses an independent CPU FP32 formula with the exact dtype-rounded
inputs and selected cache rows, followed by one output cast. Outputs must match
the resulting original-dtype bytes. FP32 outputs also use `atol=rtol=2e-5`
against the uncast reference; BF16 uses `atol=rtol=2e-2`.

Backward uses the saved cache and the transpose rotation:

```text
d_first  = d_out_first * cos + d_out_second * sin
d_second = d_out_second * cos - d_out_first * sin
```

The independent CPU FP32 result is converted once to input dtype and compared
by bytes. Checks also require finite, nonzero gradients for used branches.
These references follow the paired operator's rounding contract, not native
eager BF16 intermediate rounding.

## Test matrix

All groups run in FP32 and BF16. Cache shape is `[262400, 128]`, with base `1e7`.

| Group | Cases | Checks |
| --- | --- | --- |
| Forward | T=8,32,512,1024,4096,16384; Q/K heads=32/4 | Independent reference, unchanged inputs/cache, separate outputs |
| Tail masks and layouts | T=1,7,31,33,129; heads=1/1,5/3 | 2D/3D equivalence, masked tails |
| Batch invariance | T=8,32,33,512,16384; target at first/middle/last row | Byte equality across neighbors and block-size changes |
| Repeat/permutation/split | T=129; ten runs; chunks=8/32/89 | Original-dtype byte equality |
| Rotation orientation | cos=0, sin=1 | Neox half pairing |
| Contracts | Invalid shapes, dtypes, positions dtype, noncontiguous inputs, cache gradients | `ValueError` |
| Backward | T=33; heads=5/3; both outputs, Q only, K only | Independent dQ/dK reference; unused branch is zero |

Positions are valid indices supplied by the caller. This suite does not measure
performance or establish full-model/multi-rank training correctness.
