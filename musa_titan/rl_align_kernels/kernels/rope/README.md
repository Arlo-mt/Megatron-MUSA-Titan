# Paired Q/K RoPE

The implementation lives in `musa_titan.rl_align_kernels.kernels.rope`.
Megatron integration lives in `musa_titan/megatron/rope.py`, separate from
the operator implementation:

```text
musa_titan/rl_align_kernels/
├── kernels/rope/          # Operator, autograd and license
├── tests/ops/rope/        # Correctness and invariance tests
└── benchmarks/rope/      # Standalone forward measurements
```

See the [test matrix](../../tests/ops/rope/README.md) and
[benchmark instructions](../../benchmarks/rope/README.md).

```python
from musa_titan.rl_align_kernels.kernels.rope import rope_forward
```

`rope_forward(positions, query, key, cache)` rotates Q and K in one Triton
launch and returns independent outputs. It supports contiguous MUSA BF16/FP32
inputs shaped `[T, heads, 128]` or `[T, heads * 128]`, with different Q/K head
counts. Positions are contiguous int32/int64 indices into a same-dtype
`[capacity, 128]` cache containing `[cos(64), sin(64)]`.
Callers must supply positions within the cache bounds.

The forward retains the supplied paired Neox kernel: FP32 multiplication and
addition, no FMA, and one cast on output. `T <= 32` uses block 128; larger
batches use block 1024. Both use four warps and one stage by default.
The source license is retained in `LICENSE.sglang`.

## Megatron integration

Call `apply_pretrain_patch()` or `apply_rl_patch()` before constructing the
model. Both install the replacement `Attention.forward`, inherited by
`SelfAttention`. Only its RoPE boundary adds the paired call; QKV projection,
Q/K normalization, core attention, projection, and checkpoint scheduling retain
the upstream implementation. Modified blocks are marked `# hack(dinglei)`.
The copied function records its Megatron development revision in the patch.

Supported paths use Neox, full head dimension 128, BF16/FP32, context-parallel
size one, and no inference context. Both ordinary `[S, B, H, D]` and packed
THD layouts are supported. Q and K must share the frequency tensor and, for
THD, the same cumulative-length tensor. Padded cumulative lengths take
precedence; positions reset within each packed sequence without a host read.
Noncontiguous Q/K are made contiguous for the supplied kernel.

Megatron's `RotaryEmbedding`, inverse-frequency calculation, offsets, scaling,
and frequency cache are unchanged. At the apply boundary, the adapter computes
cos/sin from the existing frequencies, casts to the input dtype, and packs
their first halves for the kernel. It does not install a new cache generator
or change the existing YaRN cache-capacity patch.

Unsupported layouts, dimensions, dtypes, CP, FP8, interleaved/multimodal/MLA
RoPE, different Q/K positions, and inference contexts retain the previous
Titan/Megatron path. The paired path does not promise to repair limitations of
that existing fallback.

## Backward

Autograd saves positions and the actual forward cache, without saving Q/K.
The same paired kernel applies the transpose rotation in backward:

```text
d_first  = d_out_first * cos + d_out_second * sin
d_second = d_out_second * cos - d_out_first * sin
```

Both input gradients flow back through Megatron's original Q/K norms and QKV
projection. Megatron's parameter objects and optimizer interfaces are unchanged.
TE's angle-based backward is not used here: recomputing trigonometric values
from FP32 angles would not preserve the dtype-rounded forward cache.

Only first-order Q/K gradients are supported. Cache gradients are rejected;
saved positions/cache must not be modified before backward. Forward results
can differ from native eager BF16 RoPE, which rounds intermediate products.
Unchanged cache generation alone does not establish end-to-end SGLang alignment.
