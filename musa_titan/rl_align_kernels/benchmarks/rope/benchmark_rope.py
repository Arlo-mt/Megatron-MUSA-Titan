"""MUSA profiler measurements of the paired Q/K RoPE forward operator."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

import torch
import torch_musa  # noqa: F401
import triton

from musa_titan.rl_align_kernels.kernels.rope import triton_rope

ROOT = Path(__file__).resolve().parents[4]
TOKENS = (8, 32, 512, 1024, 4096, 16384)
CAPACITY, DIM, BASE = 262400, 128, 10_000_000
Q_HEADS, K_HEADS = 32, 4


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def parse_trace(path, iterations, expected_kernel=None):
    """Associate device kernels with calls via host launch correlations, not gaps."""
    events = json.loads(path.read_text())["traceEvents"]
    scopes = [e for e in events if e.get("cat") == "user_annotation"
              and e.get("name", "").startswith("rope_bench/")]
    if len(scopes) != iterations:
        raise RuntimeError(f"Expected {iterations} call scopes, got {len(scopes)}")
    cpu = {e["args"]["External id"]: e for e in events
           if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("privateuse1_driver", "privateuse1_runtime")
                and "correlation" in e.get("args", {})}
    groups = [[] for _ in range(iterations)]
    for kernel in (e for e in events if e.get("cat") == "kernel"):
        args = kernel.get("args", {})
        host = cpu.get(args.get("External id"))
        if host is None:
            host = launches.get(args.get("correlation"))
        if host is None:
            raise RuntimeError(f"Missing host correlation for {kernel['name']}")
        matches = [s for s in scopes if s["tid"] == host["tid"]
                   and s["ts"] <= host["ts"] < s["ts"] + s["dur"]]
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous call attribution for {kernel['name']}")
        duration = float(kernel["dur"])
        if kernel.get("ph") != "X" or not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("Invalid kernel duration")
        index = int(matches[0]["name"].split("/")[-1])
        groups[index].append({"name": kernel["name"], "duration_us": duration})
    if any(not group for group in groups):
        raise RuntimeError("At least one call has no device kernel samples")
    signatures = [Counter(e["name"] for e in group) for group in groups]
    if any(signature != signatures[0] for signature in signatures):
        raise RuntimeError("Kernel composition changed between calls")
    if expected_kernel and any(s != Counter({expected_kernel: 1}) for s in signatures):
        raise RuntimeError(f"Expected one {expected_kernel} kernel per call")
    return groups


def profile_call(function, name, args, directory, expected_kernel=None):
    for _ in range(args.warmup):
        function()
    torch.musa.synchronize()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.PrivateUse1,
    ]) as profiler:
        for index in range(args.iterations):
            with torch.profiler.record_function(f"rope_bench/{index}"):
                function()
        torch.musa.synchronize()
    path = directory / f"{name}.trace.json"
    profiler.export_chrome_trace(str(path))
    groups = parse_trace(path, args.iterations, expected_kernel)
    samples = [sum(e["duration_us"] for e in group) for group in groups]
    return {
        "stage": name, "samples": len(samples), "kernel_mean_us": statistics.mean(samples),
        "kernel_median_us": statistics.median(samples), "kernel_min_us": min(samples),
        "kernel_max_us": max(samples), "kernels_per_call": len(groups[0]),
        "call_kernel_time_us": samples, "kernel_events_by_call": groups, "trace": path.name,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is required")
    sources = [Path(__file__).resolve()]
    for module in (triton_rope,):
        path = Path(module.__file__).resolve()
        if path.parent != ROOT / "musa_titan" / "rl_align_kernels" / "kernels" / "rope":
            raise RuntimeError(f"Unexpected kernel source: {path}")
        sources.append(path)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if args.output_dir is None:
        base = Path(__file__).resolve().parent / "results"
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="profiler-", dir=base))
    else:
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "torch": torch.__version__, "torch_musa": torch_musa.__version__, "triton": triton.__version__,
        "device": str(torch.musa.get_device_properties(torch.musa.current_device())),
        "source_sha256": hashes, "base": BASE, "d": DIM, "cache_shape": [CAPACITY, DIM],
        "dtype": "bfloat16", "tokens": TOKENS, "q_heads": Q_HEADS, "k_heads": K_HEADS,
        "positions": "arange(1024, 1024 + T), int64", "seed": 0,
        "warmup": args.warmup, "iterations": args.iterations,
        "timing": "Mean per-call sum of profiler kernel durations; no CPU time or kernel gaps",
        "cache_stage": "Prepared once outside timing; production cache generation is unchanged",
        "bandwidth": "RoPE only: Q/K read+write, T selected cache rows and T int64 positions; logical GB/s, not DRAM counters",
    }
    save(directory / "metadata.json", metadata)
    print(f"Device: {metadata['device']}", flush=True)
    print("BF16, base=1e7, d=128, cache=[262400,128], Q/K heads=32/4", flush=True)
    print(f"{'stage':<22} {'T':>7} {'kernels/call':>13} {'mean_us':>12} {'median_us':>12} {'effective_GB/s':>16}", flush=True)
    # Benchmark inputs only. Cache preparation is excluded from all timings.
    inverse = 1.0 / (BASE ** (torch.arange(0, DIM, 2, device="musa", dtype=torch.float32) / DIM))
    angles = torch.outer(torch.arange(CAPACITY, device="musa", dtype=torch.float32), inverse)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    del angles, inverse
    results = []
    assert cache.shape == (CAPACITY, DIM) and cache.dtype == torch.bfloat16
    torch.manual_seed(0)
    for tokens in TOKENS:
        q = torch.randn(tokens, Q_HEADS, DIM, dtype=torch.bfloat16, device=cache.device)
        k = torch.randn(tokens, K_HEADS, DIM, dtype=q.dtype, device=q.device)
        positions = torch.arange(1024, 1024 + tokens, dtype=torch.int64, device=q.device)
        result = profile_call(lambda: triton_rope.rope_forward(positions, q, k, cache),
                              f"rope_T{tokens}", args, directory, "_rope_qk")
        # Only the selected T cache rows are consumed, not the entire capacity.
        qk_bytes = 2 * (q.numel() + k.numel()) * q.element_size()
        cache_bytes = tokens * DIM * cache.element_size()
        position_bytes = positions.numel() * positions.element_size()
        logical_bytes = qk_bytes + cache_bytes + position_bytes
        result.update(tokens=tokens, logical_bytes=logical_bytes,
                      qk_read_write_bytes=qk_bytes, selected_cache_bytes=cache_bytes,
                      position_bytes=position_bytes,
                      effective_GB_s=logical_bytes / result["kernel_mean_us"] / 1000.0)
        results.append(result)
        save(directory / "summary.json", results)
        print(f"{'rope':<22} {tokens:>7} {result['kernels_per_call']:>13} "
              f"{result['kernel_mean_us']:>12.3f} {result['kernel_median_us']:>12.3f} "
              f"{result['effective_GB_s']:>16.3f}", flush=True)
        del q, k, positions
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f"Source changed during measurement: {p}")
    print(f"Artifacts: {directory}", flush=True)


if __name__ == "__main__":
    main()
