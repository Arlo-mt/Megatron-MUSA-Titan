"""Independent forward/backward references, batch invariance and contracts."""

import pytest
import torch

from musa_titan.rl_align_kernels.kernels.rope import rope_forward

TOKENS = [8, 32, 512, 1024, 4096, 16384]


def inputs(tokens, dtype, device, heads=(32, 4), seed=0):
    gen = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(tokens, h, 128, generator=gen).to(dtype).to(device) for h in heads)


def reference(x, positions, cache):
    x = x.detach().cpu().float()
    cs = cache[positions.long()].detach().cpu().float().unsqueeze(1)
    first, second = x[..., :64], x[..., 64:]
    cos, sin = cs[..., :64], cs[..., 64:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


@pytest.mark.parametrize("tokens", TOKENS)
def test_forward_accuracy(tokens, cache, dtype, musa, assert_bytes):
    q, k = inputs(tokens, dtype, musa)
    # Cover the top of the full cache, not just the prefix used by the benchmark.
    positions = torch.arange(262400 - tokens, 262400, device=musa)
    before = (q.clone(), k.clone(), positions.clone())
    cache_rows = cache[positions].clone()
    out_q, out_k = rope_forward(positions, q, k, cache)
    for name, actual, x in (("q", out_q, q), ("k", out_k, k)):
        assert actual.shape == x.shape and actual.dtype == dtype and actual.device == musa
        assert torch.isfinite(actual).all()
        expected = reference(x, positions, cache)
        tol = 2e-5 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(actual.cpu().float(), expected, atol=tol, rtol=tol,
                                   msg=f"{name}, T={tokens}, dtype={dtype}, seed=0")
        assert_bytes(actual, expected.to(dtype), f"FP32 arithmetic, single cast: {name}, T={tokens}")
    for a, b in zip((q, k, positions), before):
        assert_bytes(a, b, "input unchanged")
    assert_bytes(cache[positions], cache_rows, "cache unchanged")
    assert len({t.untyped_storage().data_ptr() for t in (q, k, cache, out_q, out_k)}) == 5


@pytest.mark.parametrize("tokens", [1, 7, 31, 33, 129])
@pytest.mark.parametrize("heads", [(1, 1), (5, 3)])
def test_tail_heads_and_flattened_layout(tokens, heads, cache, dtype, musa, assert_bytes):
    q, k = inputs(tokens, dtype, musa, heads, seed=1)
    positions = torch.arange(tokens, device=musa) % 3
    shaped = rope_forward(positions, q, k, cache)
    flat = rope_forward(positions, q.flatten(1), k.flatten(1), cache)
    for a, b, x in zip(shaped, flat, (q, k)):
        assert_bytes(a, b.reshape(a.shape), "flattened layout")
        assert_bytes(a, reference(x, positions, cache).to(dtype), "masked tails")


@pytest.mark.parametrize("tokens", [8, 32, 33, 512, 16384])
def test_batch_position_invariance(tokens, cache, dtype, musa, assert_bytes):
    target_q, target_k = inputs(1, dtype, musa, seed=2)
    target_pos = torch.tensor([262399], device=musa)
    expected = rope_forward(target_pos, target_q, target_k, cache)
    for seed in (3, 4):
        q, k = inputs(tokens, dtype, musa, seed=seed)
        for row in sorted({0, tokens // 2, tokens - 1}):
            positions = torch.arange(tokens, device=musa)
            q[row:row + 1], k[row:row + 1], positions[row] = target_q, target_k, 262399
            actual = rope_forward(positions, q, k, cache)
            for a, b in zip(actual, expected):
                assert_bytes(a[row:row + 1], b, f"T={tokens}, row={row}, neighbors={seed}")


def test_repeat_permutation_and_split(cache, dtype, musa, assert_bytes):
    q, k = inputs(129, dtype, musa, seed=5)
    positions = torch.arange(129, device=musa).remainder(17)
    expected = rope_forward(positions, q, k, cache)
    for _ in range(9):
        for a, b in zip(rope_forward(positions, q, k, cache), expected):
            assert_bytes(a, b, "repeat")
    order = torch.randperm(129, generator=torch.Generator().manual_seed(6)).to(musa)
    inverse = order.argsort()
    permuted = rope_forward(positions[order], q[order], k[order], cache)
    pieces = [rope_forward(positions[a:b], q[a:b], k[a:b], cache)
              for a, b in [(0, 8), (8, 40), (40, 129)]]
    for i in range(2):
        assert_bytes(permuted[i][inverse], expected[i], "permutation")
        assert_bytes(torch.cat([part[i] for part in pieces]), expected[i], "split across block switch")


def test_rotation_orientation(dtype, musa, assert_bytes):
    q, k = inputs(1, dtype, musa, heads=(1, 1), seed=7)
    # Exact quarter-turn distinguishes Neox halves from adjacent-pair rotation.
    synthetic = torch.cat((torch.zeros(1, 64), torch.ones(1, 64)), -1).to(musa, dtype)
    actual = rope_forward(torch.zeros(1, device=musa, dtype=torch.int64), q, k, synthetic)
    for out, x in zip(actual, (q, k)):
        assert_bytes(out, torch.cat((-x[..., 64:], x[..., :64]), -1), "Neox orientation")


@pytest.mark.parametrize("invalid", ["shape", "cache_shape", "positions_dtype", "noncontiguous", "dtype", "cache_grad"])
def test_input_contract(invalid, cache, dtype, musa):
    q, k = inputs(8, dtype, musa)
    positions = torch.arange(8, device=musa)
    error = ValueError
    if invalid == "shape":
        q = q[..., :64].contiguous()
    elif invalid == "cache_shape":
        cache = cache[:, :64].contiguous()
    elif invalid == "positions_dtype":
        positions = positions.float()
    elif invalid == "noncontiguous":
        q = q.transpose(0, 1)
    elif invalid == "dtype":
        k = k.to(torch.float32 if dtype == torch.bfloat16 else torch.bfloat16)
    else:
        cache = cache.detach().requires_grad_(True)
    with pytest.raises(error):
        rope_forward(positions, q, k, cache)



@pytest.mark.parametrize("branch", ["both", "q", "k"])
def test_backward_reference(branch, cache, dtype, musa, assert_bytes):
    q, k = inputs(33, dtype, musa, heads=(5, 3), seed=8)
    q.requires_grad_(True)
    k.requires_grad_(True)
    positions = torch.arange(33, device=musa).remainder(7)
    out = rope_forward(positions, q, k, cache)
    generator = torch.Generator().manual_seed(9)
    grads = [torch.randn(x.shape, generator=generator).to(musa, dtype) for x in out]
    selected = [0, 1] if branch == "both" else [0 if branch == "q" else 1]
    torch.autograd.backward([out[i] for i in selected], [grads[i] for i in selected])
    c, s = cache[positions].cpu().float().unsqueeze(1).chunk(2, -1)
    for index, (x, grad) in enumerate(zip((q, k), grads)):
        if index not in selected:
            assert torch.count_nonzero(x.grad) == 0
            continue
        first, second = grad.cpu().float().chunk(2, -1)
        expected = torch.cat((first*c + second*s, second*c - first*s), -1).to(dtype)
        assert_bytes(x.grad.cpu(), expected, "saved-cache transpose gradient")
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
