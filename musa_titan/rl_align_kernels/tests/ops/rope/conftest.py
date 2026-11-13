"""MUSA fixtures and byte-level comparisons for the Titan RoPE module."""

import hashlib
from pathlib import Path

import pytest
import torch
import torch_musa  # noqa: F401
import triton

from musa_titan.rl_align_kernels.kernels.rope import triton_rope

ROOT = Path(__file__).resolve().parents[5]


def pytest_report_header(config):
    lines = [f"torch={torch.__version__}, torch_musa={torch_musa.__version__}, triton={triton.__version__}"]
    for module in (triton_rope,):
        path = Path(module.__file__).resolve()
        assert path.parent == ROOT / "musa_titan" / "rl_align_kernels" / "kernels" / "rope"
        lines.append(f"source={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    return lines


@pytest.fixture(scope="session", autouse=True)
def musa():
    expected = ROOT / "musa_titan/rl_align_kernels/kernels/rope/triton_rope.py"
    assert Path(triton_rope.__file__).resolve() == expected
    if not torch.musa.is_available():
        pytest.fail("MUSA is required; no CPU fallback or skipped acceptance")
    return torch.device("musa", torch.musa.current_device())


@pytest.fixture(scope="session", params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(scope="session")
def cache(dtype, musa):
    # Test input only; production Megatron cache generation is not patched.
    inverse = 1.0 / (10_000_000 ** (torch.arange(0, 128, 2, device=musa, dtype=torch.float32) / 128))
    angles = torch.outer(torch.arange(262400, device=musa, dtype=torch.float32), inverse)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)


@pytest.fixture
def assert_bytes():
    def compare(actual, expected, label=""):
        assert actual.shape == expected.shape, label
        assert actual.dtype == expected.dtype, label
        a = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        bad = (a != b).nonzero().flatten()
        if bad.numel():
            pytest.fail(f"{label}: byte_mismatches={bad.numel()}, first_byte={bad[0].item()}")
    return compare
