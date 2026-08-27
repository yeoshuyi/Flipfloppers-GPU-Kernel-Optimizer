#!/usr/bin/env python3
"""
G6.7 step 1 (cheap gate, NO model integration) -- does step 33's cuBLASLt
algorithm-search win reproduce for the ATTENTION path's FP16 GEMMs?

Fact this extends (docs/PROGRESS.md step 33 / G6.6): at the TINY shape (M=64),
the best of cublasLtMatmulAlgoGetHeuristic's candidates beat PyTorch's default
kernel choice by 1.32-1.49x on the FFN's two TF32 GEMMs, via split-K variants
the default heuristic does not pick at small M. That shipped, gated to
tok <= _LT_MAX_TOKENS = 127.

Since G6.4b (step 28) the attention path runs in FP16, not TF32:
    qkv      M=tok, K=512, N=1536   (fused q/k/v projection, bias)
    out_proj M=tok, K=512, N=512    (bias)
At tok=64 these are also small GEMMs. Open question: does "the default
heuristic misses a better split-K variant at small M" reproduce for FP16, or is
FP16 already well-optimised by cuBLASLt's default heuristic (FP16 being a far
more common ML dtype than TF32)? A clean negative is a perfectly good answer.

DECISIVE GATE: best candidate must beat PyTorch's own FP16 F.linear by >10%,
reproducibly, on at least one of the two shapes. Below that -> stop, report a
clean negative, do NOT integrate.

PRECISION DISCIPLINE: benchmark.py sets allow_fp16_reduced_precision_reduction
= False (G6.4b). The cuBLASLt equivalent is the split-K reduction scheme, so
every shape is measured twice: reduction_mask=2 (fp32 partials only, the
policy-compliant set) and reduction_mask=7 (everything, for information only --
a win that exists ONLY at mask=7 is a win bought with fp16 partial sums and
would not be admissible without a separate accuracy argument).

This probe DOES NOT touch benchmark.py and DOES NOT touch the shipped
csrc/cublaslt_algo.cpp -- it builds csrc/cublaslt_algo_fp16.cpp separately.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "csrc", "cublaslt_algo_fp16.cpp")

D = 512
QKV_N = 3 * D           # 1536, fused QKV
M_LIST = [64, 1024]     # tiny (the hypothesis) + default (context)
REQUESTED = 16
MAX_WS = 32 * 1024 * 1024
WARMUP = 30
ITERS = 200
REPEATS = 3             # whole-measurement repeats, for reproducibility


def build_ext():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    # with_cuda=True is mandatory even with no .cu source (step 32).
    return load(name="cublaslt_algo_fp16", sources=[SRC],
                build_directory=build_dir, with_cuda=True,
                extra_ldflags=["-lcublasLt"], verbose=False)


def time_pytorch(fn, warmup=WARMUP, iters=ITERS):
    """CUDA-event GPU time AND CPU issue time, mirroring time_algo2."""
    import time as _time
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    c0 = _time.perf_counter()
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    c1 = _time.perf_counter()          # BEFORE the sync: issue time only
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters, (c1 - c0) * 1e3 / iters


# ---------------------------------------------------------------------------
# Layout confirmation -- re-derived for FP16, not assumed from the FP32 file.
# ---------------------------------------------------------------------------
def confirm_layout(ext, dev):
    """Independent check that (transa=T on W as K x N, transb=N on In as K x M,
    C as N x M) really computes F.linear for CUDA_R_16F layouts.

    Uses a deliberately ASYMMETRIC shape with M != N != K and all three
    pairwise-unequal, so any transposition or ld mistake is either a hard shape
    error or a large numerical mismatch -- a square shape could hide one.
    """
    print("\n=== layout confirmation (FP16, asymmetric M=7 K=24 N=13) ===",
          flush=True)
    M, K, N = 7, 24, 13
    gen = torch.Generator(device=dev)
    gen.manual_seed(4242)
    inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=gen)
    w = torch.randn(N, K, device=dev, dtype=torch.float16, generator=gen)
    b = torch.randn(N, device=dev, dtype=torch.float16, generator=gen)
    out = torch.empty(M, N, device=dev, dtype=torch.float16)

    ok = True
    for use_bias in (False, True):
        pid = ext.create_problem(M, N, K, use_bias, MAX_WS, REQUESTED, -1)
        n = ext.num_algos(pid)
        if n == 0:
            print(f"  bias={use_bias}: no candidate returned", flush=True)
            ok = False
            continue
        ext.run(pid, 0, inp, w, b if use_bias else None, out)
        torch.cuda.synchronize()
        ref = F.linear(inp, w, b if use_bias else None)
        md = (out.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        good = md <= 5e-3 * max(scale, 1.0)
        ok = ok and good
        print(f"  bias={use_bias}: maxdiff={md:.3e} (|ref|max={scale:.3f}) "
              f"-> {'MATCH' if good else 'MISMATCH'}", flush=True)
    print(f"  layout mapping {'CONFIRMED' if ok else 'NOT CONFIRMED'} for FP16",
          flush=True)
    return ok


def kernel_names(fn, label, n=3):
    """What kernel does this actually dispatch to? Cheap confirmation that
    PyTorch's own FP16 path is the same 'tn' configuration the layout mapping
    assumes (the FP32 file confirmed this by matching '_tn_' in the profiled
    CUTLASS name; this is the FP16 equivalent, measured not guessed)."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    names = []
    for ev in prof.key_averages():
        nm = ev.key
        if ev.self_device_time_total > 0 and any(
                t in nm.lower() for t in ("gemm", "cutlass", "sm80", "sm89",
                                          "xmma", "matmul", "ampere")):
            names.append((ev.self_device_time_total, nm))
    names.sort(reverse=True)
    print(f"  [{label}] dispatched kernels:", flush=True)
    for t, nm in names[:3]:
        print(f"      {nm}", flush=True)
    if not names:
        print("      (none matched the gemm-name filter)", flush=True)
    return [nm for _, nm in names]


# ---------------------------------------------------------------------------
def one_case(ext, name, M, K, N, red_mask, dev, gen, do_profile=False):
    gen.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
    inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=gen)
    w = (torch.randn(N, K, device=dev, dtype=torch.float16, generator=gen)
         * 0.02).half()
    b = (torch.randn(N, device=dev, dtype=torch.float16, generator=gen)
         * 0.02).half()
    out = torch.empty(M, N, device=dev, dtype=torch.float16)
    flops = 2.0 * M * N * K

    ref_fn = lambda: F.linear(inp, w, b)                       # noqa: E731
    ref = ref_fn()
    torch.cuda.synchronize()

    tag = f"{name}  M={M} K={K} N={N} bias=True  red_mask={red_mask}"
    print(f"\n=== {tag} ===", flush=True)

    # PyTorch reference, repeated -- this is the number the gate is against.
    ref_runs = [time_pytorch(ref_fn) for _ in range(REPEATS)]
    t_ref = min(r[0] for r in ref_runs)
    cpu_ref = min(r[1] for r in ref_runs)
    spread = (max(r[0] for r in ref_runs) - t_ref) / t_ref * 100.0
    print(f"  pytorch F.linear : gpu {t_ref*1000:8.2f} us  "
          f"cpu-issue {cpu_ref*1000:7.2f} us  "
          f"{flops/(t_ref*1e-3)/1e12:6.2f} TFLOPS  (spread {spread:.1f}%)",
          flush=True)

    if do_profile:
        kernel_names(ref_fn, "pytorch F.linear fp16")

    pid = ext.create_problem(M, N, K, True, MAX_WS, REQUESTED, red_mask)
    n = ext.num_algos(pid)
    print(f"  heuristic returned {n} candidate(s)", flush=True)
    if n == 0:
        return None

    rows = []
    for i in range(n):
        try:
            ts = [ext.time_algo2(pid, i, inp, w, b, out, WARMUP, ITERS)
                  for _ in range(REPEATS)]
        except Exception as exc:                               # noqa: BLE001
            print(f"  [{i:2d}] SKIP  {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:90]}", flush=True)
            continue
        t = min(x[0] for x in ts)
        cpu = min(x[1] for x in ts)
        ext.run(pid, i, inp, w, b, out)
        torch.cuda.synchronize()
        md = (out.float() - ref.float()).abs().max().item()
        rows.append((t, i, md, cpu))
        print(f"  [{i:2d}] gpu {t*1000:8.2f} us  cpu {cpu*1000:6.2f} us  "
              f"{flops/(t*1e-3)/1e12:6.2f} TF  x{t_ref/t:5.3f}  "
              f"maxdiff={md:.3e}  {ext.algo_info(pid, i)}", flush=True)

    if not rows:
        print("  no usable candidate", flush=True)
        return None
    rows.sort()
    t_best, i_best, md_best, cpu_best = rows[0]
    speedup = t_ref / t_best
    if speedup > 1.10:
        verdict = "WIN (>10% gate PASSED)"
    elif speedup > 1.02:
        verdict = "MARGINAL (<10% gate FAILED)"
    elif speedup > 0.98:
        verdict = "NULL"
    else:
        verdict = "LOSS"
    launch_bound = cpu_best >= 0.9 * t_best
    print(f"  BEST -> algo[{i_best}]  {t_best*1000:.2f} us  "
          f"speedup vs pytorch = {speedup:.4f}  ({verdict})  "
          f"maxdiff={md_best:.3e}", flush=True)
    print(f"  launch-bound? cpu-issue {cpu_best*1000:.2f} us vs gpu "
          f"{t_best*1000:.2f} us -> "
          f"{'YES -- treat any delta as dispatch cost' if launch_bound else 'no, GPU-bound'}",
          flush=True)
    if do_profile:
        kernel_names(lambda: ext.run(pid, i_best, inp, w, b, out),
                     f"cublasLt algo[{i_best}]")
    return (tag, speedup, t_ref, t_best, md_best, launch_bound)


def main():
    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # G6.4b's own discipline: fp32 accumulate for fp16 GEMMs, both sides.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
          "| cuda", torch.version.cuda, flush=True)
    print("allow_fp16_reduced_precision_reduction="
          f"{torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}",
          flush=True)

    ext = build_ext()
    if not confirm_layout(ext, dev):
        print("\nLAYOUT NOT CONFIRMED -- timings below would be meaningless. "
              "Aborting.", flush=True)
        return 2

    gen = torch.Generator(device=dev)
    summary = []
    for M in M_LIST:
        for red_mask in (2, 7):     # 2 = fp32 partials only; 7 = anything
            prof = (M == 64 and red_mask == 2)
            r = one_case(ext, "qkv     ", M, D, QKV_N, red_mask, dev, gen, prof)
            if r:
                summary.append(r)
            r = one_case(ext, "out_proj", M, D, D, red_mask, dev, gen, prof)
            if r:
                summary.append(r)

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY (gate: >1.10x, reproducible, on at least one shape at M=64 "
          "with red_mask=2)", flush=True)
    for tag, sp, tr, tb, md, lb in summary:
        print(f"  {tag:<52} x{sp:6.4f}  "
              f"({tr*1000:7.2f} -> {tb*1000:7.2f} us)  maxdiff={md:.2e}"
              f"{'  [launch-bound]' if lb else ''}", flush=True)
    tiny_ok = [s for s in summary
               if " M=64 " in s[0] and "red_mask=2" in s[0] and s[1] > 1.10]
    print("\nVERDICT: " + ("GATE PASSED -- " + ", ".join(t[0].strip() for t in tiny_ok)
                           if tiny_ok else
                           "GATE FAILED -- no M=64 fp32-partial candidate beats "
                           "PyTorch's FP16 F.linear by >10%. Clean negative: "
                           "cuBLASLt's default heuristic is already well-tuned "
                           "for FP16 at this shape."), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
