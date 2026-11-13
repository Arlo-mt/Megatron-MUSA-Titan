# RMSNorm correctness tests

Use pytest to validate both Triton RMSNorm forward interfaces and fused autograd support.

## Scope and organization

- Run the Triton operators only on MUSA through pytest.
- Cover `torch.float32` and `torch.bfloat16`, with matching input, weight and residual dtypes.
- Check ordinary RMSNorm and fused add + RMSNorm forward accuracy, repeatability,
  batch invariance and interface contracts.
- Check that fused first-order gradients for input, residual and weight can be
  computed and are finite and nonzero.
- Exclude benchmarks, performance thresholds, independent gradient accuracy
  comparisons and end-to-end training acceptance.

This directory contains:

```text
musa_titan/rl_align_kernels/tests/ops/rmsnorm/
├── README.md
├── conftest.py                  # MUSA fixtures, seeds, references and comparisons
├── test_rmsnorm.py              # Shared accuracy, invariance and weight checks
└── test_fused_add_rmsnorm.py     # Residual semantics, rounding, errors and autograd
```

Tests import the implementations from
`musa_titan/rl_align_kernels/kernels/rmsnorm/` and verify their actual import locations.

## Test methods

### Independent references

Generate data on CPU with fixed seeds, convert it to the tested dtype, then send
that same data to MUSA. CPU is used only to generate inputs and compute reference
values; it is not a fallback backend for the operator under test. Reference
computations neither call the tested kernel nor copy its Triton reduction tree.

Ordinary RMSNorm uses an independent PyTorch FP32 formula:

```text
x32 = FP32(x)
w32 = FP32(weight)
rstd_ref = 1 / sqrt(mean(x32 * x32, dim=-1) + eps)
y_ref = x32 * rstd_ref[..., None] * w32
```

Cross-check this formula with CPU `torch.nn.functional.rms_norm`. Reference values
must use inputs already converted to the tested dtype, so input quantization is
not counted as kernel error. Convert MUSA outputs to CPU FP32 for comparison;
this includes rounding from the final output conversion.

Fused add + RMSNorm uses:

```text
s32 = FP32(x) + FP32(residual)
rstd_ref = 1 / sqrt(mean(s32 * s32, dim=-1) + eps)
y_ref = s32 * rstd_ref[..., None] * FP32(weight)
residual_ref = cast(s32, input.dtype)
```

Normalization uses `s32` before conversion back to the input dtype. BF16
`rms_norm(x + residual)` is not a valid fused reference because it rounds the
residual sum before normalization.

### Numerical accuracy

Use `torch.testing.assert_close` with these fixed tolerances:

| Result | atol | rtol |
| --- | ---: | ---: |
| FP32 normalized output | 2e-5 | 2e-5 |
| BF16 normalized output, converted to FP32 for comparison | 2e-2 | 2e-2 |
| Fused FP32 `rstd`, for both input dtypes | 2e-5 | 2e-5 |

All finite-input cases check that outputs are finite. Investigate numerical
mismatches before changing tolerances. Different reduction orders can satisfy an
independent accuracy reference without producing identical bytes.

### Byte equality

First check shape and dtype. Copy tensors to CPU, make them contiguous, flatten
them and compare their original bytes. Floating-point equality or
`max_abs_diff == 0` alone can miss bit-pattern differences such as signed zero.

- Repeatability: run the same input 10 times and compare every result with the first.
- Batch invariance: hold the target row and weight fixed while changing batch
  size, target-row position and neighboring rows.
- Row layout: insert unrelated padding rows, permute and restore row order, and
  compare a complete batch with independently processed chunks.
- Tensor layout: use different leading dimensions and contiguous or strided
  storage, then restore the same logical shape for comparison.

Ordinary RMSNorm checks its normalized output. Fused RMSNorm checks all four
outputs, including FP32 `rstd` and `residual_sum_fp32`. Every invariance check
within a backend requires zero differing bytes.

### Fused outputs and interface contracts

| Output | Shape / dtype | Check |
| --- | --- | --- |
| `normalized` | Same as input | Compare with `y_ref` using the input-dtype tolerance |
| `residual_sum` | Same as input | Compare bytes with `residual_ref` |
| `rstd` | `input.shape[:-1]` / FP32 | Compare with `rstd_ref` using the FP32 tolerance |
| `residual_sum_fp32` | Same as input / FP32 | Compare bytes with `s32` |

Also check output devices, unchanged inputs and weights, and independent storage
for the four nonempty fused outputs. A one-dimensional input must produce scalar
`rstd`.

Exception checks follow Titan's interfaces: invalid ordinary RMSNorm weight
shapes and invalid fused shapes, dtypes or epsilon values raise `ValueError`.

Fused RMSNorm supports first-order autograd. Enable gradients separately for
input, residual and weight, and construct a loss from the sums of the normalized
and residual outputs. Check the resulting gradient's shape, dtype, finiteness and
nonzero magnitude. Confirm that FP32 `rstd` and `residual_sum_fp32` do not require
gradients. This verifies backward integration, not independent gradient accuracy
or backward batch invariance.

## Test matrix

Both operators cover FP32 and BF16. Use `eps=1e-6` except in the dedicated epsilon
cases. Parameterize each group separately; do not take a Cartesian product across
groups.

| Test group | Parameters and data | Acceptance |
| --- | --- | --- |
| Regular accuracy | `H={128,1024,2048,4096}` x rows `{1,7,64,257}` x both dtypes; random non-unit weights; seed=0 | Independent reference tolerances, finiteness, shape/dtype/device |
| Chunk boundaries | `H={1023,1025,2047,2049}` x rows `{1,7}` x both dtypes; seed=1 | Tail masking and accuracy across 1024-column boundaries |
| Repeatability | `H=2048`, 64 rows, both dtypes; seed=2; 10 repetitions | Original bytes match |
| Batch and row layout | `H={128,2048}`, batch rows `{1,7,64,257}`; target at the first, middle and last positions; both dtypes; target seed=3, neighbor seeds=30/31, padding seed=33 | Single-row, padded, restored-permutation and chunked results match byte for byte |
| Tensor layout | `H=2048`; one-dimensional `[H]` versus `[1,H]`; the same 6 rows as `[6,H]` and `[2,3,H]`; additional strided inputs, residuals and weights; both dtypes; seed=4 | Bytes match after restoring the same logical layout |
| Numerical boundaries | `H=2048`, 7 rows; zeros, small values around `1e-4`, mixed signs and finite magnitudes from `1e-4` to `1e4`; both dtypes; seed=5 | Accuracy and finiteness |
| Epsilon | `H=2048`, 7 rows; small inputs; `eps={1e-6,1e-5}`; both dtypes; seed=6 | Reference confirms epsilon is inside the square root; zero inputs are not the only evidence |
| Weight semantics | `H=2048`, 7 rows; random, all-zero, all-one and all-two weights; both dtypes; seed=7 | Apply `weight` correctly, without `1+weight` semantics |
| Fused residual semantics | `H=2048`, 7 rows; zero residual, exact cancellation and near cancellation; additional BF16 addition-rounding boundary case; seed=8 | All four outputs are correct; reference data distinguishes premature rounding from normalization of the FP32 sum |
| Interface boundaries | Invalid weight shape; fused residual shape, mismatched dtypes, empty hidden dimension and invalid epsilon; both dtypes; seed=9 | Expected `ValueError` exceptions |
| Fused autograd | `H=2048`, 7 rows; set `requires_grad=True` separately for input, residual and weight; both dtypes; seed=9; 6 cases | First-order gradients have correct shape/dtype and are finite and nonzero; FP32 caches do not require gradients |

Before using the BF16 rounding-boundary case, confirm in the reference computation
that premature rounding changes the expected result. Random non-unit weights are
the primary accuracy inputs; all-one weights are used only for specific semantic
checks.

## Running and acceptance

From the repository root, use a MUSA environment with `pytest`, `torch`,
`torch_musa` and a compatible Triton version installed:

```bash
python -m pytest musa_titan/rl_align_kernels/tests/ops/rmsnorm -v -ra
```

Tests register `torch_musa` and require an available MUSA device. Missing
dependencies or devices cause explicit failures; there is no CPU fallback or
acceptance based on skipped tests. Check GPU availability before running and do
not terminate unrelated workloads.

Failure diagnostics include the operator, shape, dtype, epsilon, seed, maximum
absolute difference, number of elements outside tolerance and first mismatch
location. Byte comparisons also report the number of differing bytes. The pytest
header records kernel source locations and SHA256 hashes, PyTorch/torch_musa/Triton
versions and device information. Retain the Git revision and execution log for
each run. The terminal summary reports passed, failed, errored and skipped cases
separately for `accuracy`, `invariance` and `contract`.

## References

Reference RL-Kernel commit `49941801f942428fd52ac5b66a1b0c070418724e`:

- [RMSNorm tests](https://github.com/RL-Align/RL-Kernel/blob/49941801f942428fd52ac5b66a1b0c070418724e/tests/ops/norm/test_rms_norm.py)
- [PyTorch reference implementation](https://github.com/RL-Align/RL-Kernel/blob/49941801f942428fd52ac5b66a1b0c070418724e/rl_engine/reference/norm/rms_norm.py)

The suite follows the separation of accuracy and invariance checks and the dtype
specific tolerances. Its scope is MUSA FP32/BF16, with an additional fused autograd
support check matching Titan's interface.
