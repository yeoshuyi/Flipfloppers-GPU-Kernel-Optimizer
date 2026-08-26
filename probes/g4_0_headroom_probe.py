"""
G4.0 feasibility probe -- how much GPU-side kernel-to-kernel overhead is
actually left in the CUDA-graphed forward pass?

G4.0 (docs/MEGAKERNEL.md) proposes replacing ~10 ops/layer with 2 hand-fused
kernels/layer ("12 launches instead of ~60, captured in one CUDA graph").
G2.4 already captures the whole non-causal forward in ONE CUDA graph, so the
CPU-side dispatch cost of those ~60 launches is already gone.  What remains,
and what G4.0 would target, is the GPU-side per-kernel cost: the gap between
consecutive kernels in a graph replay (grid launch/teardown, tail effects).

The decisive number is therefore:

    gap_fraction = 1 - sum(kernel GPU durations) / wall-clock forward time

If gap_fraction is small, there is no kernel-to-kernel overhead for fusion to
remove, and G4.0's premise is dead on arrival independently of how good the
fused kernel is.  Measured here three ways:

  A. wall time per forward, torch.cuda.Event, no profiler attached
  B. kernel count + summed kernel durations per forward, via CUPTI kernel
     tracing (torch.profiler -> chrome trace, cat=="kernel"), which DOES see
     kernels launched from inside a CUDA graph replay
  C. an independent calibration: capture a CUDA graph containing K trivial
     kernels, sweep K, and take the slope -> marginal cost of one extra
     kernel in a graph replay on this GPU.  That is the per-kernel price
     G4.0 would be buying back.

Prints a compact summary only -- no raw traces reach anyone's context.
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/work")

from benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
)

DEV = torch.device("cuda")
DTYPE = torch.float32

# MUST match benchmark.py main()'s defaults, or the GEMMs land on plain
# FP32 CUDA-core sgemm instead of the TF32 tensor-core path the shipped
# numbers were measured on (first run of this probe made exactly that
# mistake -- default came out 1.48 ms instead of the sweep's 0.87 ms).
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# (name, batch_size, seq_len) -- every non-causal regime in the sweep.
SHAPES = [
    ("tiny", 1, 64),
    ("default", 8, 128),
    ("long_seq", 8, 1024),
    ("large_batch", 256, 128),
]

PEAK_TF32 = 82.6e12  # CLAUDE.md ground truth


def model_gflop(batch: int, seq: int) -> float:
    """FLOP for one forward at this shape (matches CLAUDE.md's 40.27 GFLOP
    at B=8,S=128)."""
    d, ffn, layers = 512, 2048, 6
    tokens = batch * seq
    per_tok_linear = 2 * d * (3 * d) + 2 * d * d + 2 * d * ffn + 2 * ffn * d
    linear = per_tok_linear * tokens * layers
    attn = 4 * batch * seq * seq * d * layers
    return (linear + attn) / 1e9


def build(batch: int, seq: int):
    cfg = TransformerConfig(
        batch_size=batch, seq_len=seq, d_model=512, num_heads=8,
        ffn_dim=2048, num_layers=6, causal=False,
    )
    base = BaselineTransformer(cfg).to(DEV, DTYPE).eval()
    opt = UserOptimizedTransformer(cfg).to(DEV, DTYPE).eval()
    copy_model_weights(base, opt, strict=True)
    x, mask = generate_random_case(cfg, DEV, DTYPE, seed=1234,
                                   padding_ratio=0.0, input_scale=1.0)
    return opt, x, mask


def wall_ms(model, x, mask, iters=200, warmup=60) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model(x, mask)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def kernel_trace(model, x, mask, iters=20):
    """Returns (kernels_per_forward, summed_kernel_us_per_forward,
    top5 [(name, count, total_us)])."""
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        for _ in range(20):  # ensure steady-state replay, past compile
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    os.unlink(path)

    per_name = {}
    total_us = 0.0
    count = 0
    for ev in events:
        cat = (ev.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(ev.get("dur", 0.0))
        name = ev.get("name", "?")
        total_us += dur
        count += 1
        slot = per_name.setdefault(name, [0, 0.0])
        slot[0] += 1
        slot[1] += dur

    top = sorted(per_name.items(), key=lambda kv: -kv[1][1])[:6]
    top = [(n[:58], c // iters, us / iters) for n, (c, us) in top]

    # Split kernel time into "GEMM/attention" (the tensor-core work G4.0
    # would still have to do, just inside its own kernel) vs "everything
    # else" (LN, GELU, residual adds, transposes, splitK reduces) -- the
    # latter is the only part fusion can actually delete.
    gemm_us = other_us = 0.0
    gemm_n = other_n = 0
    for name, (c, us) in per_name.items():
        low = name.lower()
        is_gemm = ("gemm" in low or "fmha" in low or "cutlass" in low
                   or "implicit" in low or "conv" in low)
        if is_gemm:
            gemm_us += us
            gemm_n += c
        else:
            other_us += us
            other_n += c
    split = (gemm_n / iters, gemm_us / iters, other_n / iters, other_us / iters)
    return count / iters, total_us / iters, top, split


def graph_kernel_slope():
    """Marginal cost of one more kernel inside a CUDA graph replay."""
    t = torch.ones(1, device=DEV)
    out = []
    for k in (16, 64, 128, 256):
        # warm the capture stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                t.add_(0.0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(k):
                t.add_(0.0)
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        out.append((k, start.elapsed_time(end) / 200 * 1000.0))  # us
    # least-squares slope over the sweep
    n = len(out)
    mx = sum(k for k, _ in out) / n
    my = sum(v for _, v in out) / n
    num = sum((k - mx) * (v - my) for k, v in out)
    den = sum((k - mx) ** 2 for k, _ in out)

    # How much of that marginal cost is the no-op kernel's own device-side
    # execution, vs a genuine dispatch gap? Trace the K=256 graph.
    from torch.profiler import ProfilerActivity, profile
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(256):
            t.add_(0.0)
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    os.unlink(path)
    durs = [float(e["dur"]) for e in events
            if (e.get("cat") or "").lower() == "kernel"]
    body_us = (sum(durs) / len(durs)) if durs else float("nan")
    return out, num / den, body_us


def main() -> int:
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    print()

    sweep, slope_us, trivial_body_us = graph_kernel_slope()
    print("=== C. CUDA-graph marginal cost per kernel ===")
    for k, us in sweep:
        print(f"  K={k:4d} trivial kernels in one graph: {us:8.2f} us "
              f"({us / k:6.3f} us/kernel)")
    print(f"  least-squares slope: {slope_us:.4f} us per extra kernel")
    print(f"  of which the kernel BODY itself (CUPTI device duration of one "
          f"no-op add_): {trivial_body_us:.4f} us")
    print(f"  => true dispatch gap per kernel in a graph replay: "
          f"{slope_us - trivial_body_us:.4f} us")
    print()

    for name, batch, seq in SHAPES:
        gflop = model_gflop(batch, seq)
        floor_ms = gflop * 1e9 / PEAK_TF32 * 1e3
        model, x, mask = build(batch, seq)

        w = wall_ms(model, x, mask)
        nk, kus, top, split = kernel_trace(model, x, mask)
        gemm_n, gemm_us, other_n, other_us = split
        kms = kus / 1000.0
        gap_ms = w - kms

        print(f"=== {name} (B={batch}, S={seq}, tokens={batch * seq}) ===")
        print(f"  A. wall per forward       : {w:8.4f} ms")
        print(f"  B. kernels per forward    : {nk:8.1f}")
        print(f"     summed kernel duration : {kms:8.4f} ms "
              f"({kms / w * 100:5.1f}% of wall)")
        print(f"     inter-kernel gap       : {gap_ms:8.4f} ms "
              f"({gap_ms / w * 100:5.1f}% of wall)"
              f" -> {gap_ms * 1000 / max(nk, 1):.3f} us/kernel")
        print(f"  roofline: {gflop:8.2f} GFLOP, TF32 floor {floor_ms:.4f} ms, "
              f"current = {gflop * 1e9 / (w * 1e-3) / 1e12:.1f} TFLOPS "
              f"({gflop * 1e9 / (w * 1e-3) / PEAK_TF32 * 100:.1f}% of peak)")
        print(f"  D. GEMM/attn kernels      : {gemm_n:5.1f} launches, "
              f"{gemm_us / 1000:8.4f} ms ({gemm_us / max(kus, 1) * 100:5.1f}% "
              f"of kernel time)")
        print(f"     everything else        : {other_n:5.1f} launches, "
              f"{other_us / 1000:8.4f} ms "
              f"({other_us / max(kus, 1) * 100:5.1f}% of kernel time)"
              f"  <- the only part fusion can delete")
        print("     top kernels by total time (per forward):")
        for n, c, us in top:
            print(f"       {c:4d}x {us:9.2f} us  {n}")
        print()

        del model, x, mask
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
