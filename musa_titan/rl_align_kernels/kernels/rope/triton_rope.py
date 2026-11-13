# Copyright 2023-2024 SGLang Team
# Licensed under Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
# Modified by dinglei: add first-order autograd with the saved forward cache.
"""Paired, out-of-place Neox RoPE with first-order MUSA autograd."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


# hack(dinglei)
# torch_musa lacks bucketize. Find packed sequence starts on device so the
# adapter supports graph replay and does not synchronize cu_seqlens to the host.
@triton.jit
def _packed_positions(CU, Positions, tokens, BOUNDARIES: tl.constexpr,
                      STEPS: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), BOUNDARIES, tl.int32)
    for _ in range(STEPS):
        mid = (low + high) // 2
        start = tl.load(CU + mid, mid < BOUNDARIES, other=2147483647)
        right = (mid < BOUNDARIES) & (start <= token)
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    start = tl.load(CU + low - 1, (token < tokens) & (low > 0), other=0)
    tl.store(Positions + token, token - start, token < tokens)


def _positions_from_cu_seqlens(cu_seqlens, tokens):
    positions = torch.empty(tokens, device=cu_seqlens.device, dtype=torch.int64)
    _packed_positions[(triton.cdiv(tokens, 256),)](
        cu_seqlens.contiguous(), positions, tokens, cu_seqlens.numel(),
        cu_seqlens.numel().bit_length(), 256, num_warps=4,
    )
    return positions
#


@triton.jit
def _rotate(X, Y, Positions, Cache, tokens, HEADS: tl.constexpr,
            BLOCK: tl.constexpr, block, BACKWARD: tl.constexpr):
    pair = block * BLOCK + tl.arange(0, BLOCK)
    valid = pair < tokens * HEADS * 64
    token = pair // (HEADS * 64)
    dim = pair % 64
    offset = (pair // 64) * 128 + dim
    position = tl.load(Positions + token, valid, other=0)
    cos = tl.load(Cache + position * 128 + dim, valid, other=0).to(tl.float32)
    sin = tl.load(Cache + position * 128 + dim + 64, valid, other=0).to(tl.float32)
    x1 = tl.load(X + offset, valid, other=0).to(tl.float32)
    x2 = tl.load(X + offset + 64, valid, other=0).to(tl.float32)
    # hack(dinglei)
    # Apply the transpose rotation to gradients using the exact forward cache.
    # Recomputing trig from FP32 angles would lose the BF16 cache rounding.
    if BACKWARD:
        tl.store(Y + offset, x1 * cos + x2 * sin, valid)
        tl.store(Y + offset + 64, x2 * cos - x1 * sin, valid)
    else:
        tl.store(Y + offset, x1 * cos - x2 * sin, valid)
        tl.store(Y + offset + 64, x2 * cos + x1 * sin, valid)
    #


@triton.jit
def _rope_qk(Q, K, OQ, OK, Positions, Cache, tokens,
             Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr, BLOCK: tl.constexpr,
             BACKWARD: tl.constexpr = False):
    block = tl.program_id(0)
    q_blocks = tl.cdiv(tokens * Q_HEADS * 64, BLOCK)
    if block < q_blocks:
        _rotate(Q, OQ, Positions, Cache, tokens, Q_HEADS, BLOCK, block, BACKWARD)
    else:
        _rotate(K, OK, Positions, Cache, tokens, K_HEADS, BLOCK, block - q_blocks, BACKWARD)


def _rope_forward(positions, query, key, cache, *, block=None, num_warps=4):
    """Rotate contiguous Q/K with a packed [capacity, 128] cos/sin cache.

    Q/K are [T, heads, 128] or [T, heads*128], with potentially different head
    counts. Positions must be in [0, cache.shape[0]); callers validate values
    before dispatch, avoiding a GPU-to-host synchronization in each forward.
    FP32 intermediates, no FMA, one output cast; forward only, out of place.
    """
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a 1D int32 or int64 tensor")
    if query.device.type != "musa" or any(
        t.device != query.device for t in (key, cache, positions)
    ):
        raise ValueError("all tensors must share a MUSA device")
    if query.dtype not in (torch.float32, torch.bfloat16) or any(
        t.dtype != query.dtype for t in (key, cache)
    ):
        raise ValueError("query, key and cache must share float32 or bfloat16 dtype")
    if not all(t.is_contiguous() for t in (positions, query, key, cache)):
        raise ValueError("all tensors must be contiguous")
    if cache.ndim != 2 or cache.shape[0] == 0 or cache.shape[1] != 128:
        raise ValueError("cache must have shape (capacity, 128)")
    tokens = positions.numel()
    if tokens == 0:
        raise ValueError("positions must be nonempty")
    for tensor in (query, key):
        if (tensor.ndim not in (2, 3) or tensor.shape[0] != tokens
                or tensor.numel() == 0 or tensor.shape[-1] % 128
                or (tensor.ndim == 3 and tensor.shape[-1] != 128)):
            raise ValueError("query/key must have shape (T, heads, 128) or (T, heads*128)")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, cache)):
        raise RuntimeError("This forward-only operator does not implement autograd")
    if block is not None and (block <= 0 or block & (block - 1)):
        raise ValueError("block must be a positive power of two")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if block is None:
        block = 128 if tokens <= 32 else 1024   # token 数多时用较大的 block 数
    q_heads = query.numel() // (tokens * 128)
    k_heads = key.numel() // (tokens * 128)
    out_q, out_k = torch.empty_like(query), torch.empty_like(key)
    grid = (triton.cdiv(tokens * q_heads * 64, block)
            + triton.cdiv(tokens * k_heads * 64, block),)
    _rope_qk[grid](query, key, out_q, out_k, positions, cache, tokens,
                   q_heads, k_heads, block, num_warps=num_warps,
                   # Match Inductor: FP32 intermediates, no FMA contraction,
                   # and a single BF16 cast when storing each output.
                   num_stages=1, enable_fp_fusion=False)
    return out_q, out_k


# hack(dinglei)
# Save only positions and the actual cos/sin values; RoPE's linear derivative
# needs no Q/K activation cache and must not regenerate frequencies in backward.
class _PairedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, query, key, cache, block, num_warps):
        outputs = _rope_forward(positions, query, key, cache, block=block, num_warps=num_warps)
        ctx.save_for_backward(positions, cache)
        ctx.shapes = (query.shape, key.shape)
        ctx.block = block if block is not None else (128 if positions.numel() <= 32 else 1024)
        ctx.num_warps = num_warps
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_q, grad_k):
        positions, cache = ctx.saved_tensors
        grads = tuple(
            torch.zeros(shape, dtype=cache.dtype, device=cache.device)
            if grad is None else grad.contiguous()
            for grad, shape in zip((grad_q, grad_k), ctx.shapes)
        )
        outputs = tuple(torch.empty_like(grad) for grad in grads)
        tokens = positions.numel()
        heads = tuple(grad.numel() // (tokens * 128) for grad in grads)
        grid = (sum(triton.cdiv(tokens * h * 64, ctx.block) for h in heads),)
        _rope_qk[grid](
            *grads, *outputs, positions, cache, tokens, *heads, ctx.block,
            BACKWARD=True, num_warps=ctx.num_warps, num_stages=1, enable_fp_fusion=False,
        )
        return (None, outputs[0] if ctx.needs_input_grad[1] else None,
                outputs[1] if ctx.needs_input_grad[2] else None, None, None, None)


def rope_forward(positions, query, key, cache, *, block=None, num_warps=4):
    """Rotate contiguous [T, heads, 128] or [T, heads*128] Q/K together.

    The caller supplies a contiguous [capacity, 128] cos/sin cache in Q/K dtype
    and valid int32/int64 positions. BF16/FP32 MUSA supports first-order Q/K
    gradients. Cache gradients are unsupported; inputs are never overwritten.
    """
    if torch.is_grad_enabled():
        if cache.requires_grad:
            raise ValueError("RoPE cache gradients are unsupported")
        if query.requires_grad or key.requires_grad:
            return _PairedRoPE.apply(positions, query, key, cache, block, num_warps)
    return _rope_forward(positions, query, key, cache, block=block, num_warps=num_warps)
#
