#!/usr/bin/env python3
"""
G4.4 Stage 0 -- RAW THROUGHPUT ONLY for the hand-written mma.sync
FP16-storage/FP16-accumulate GEMM.  No accuracy work here beyond
"does it launch and produce finite, non-degenerate output".

Kill gate (stated up front, before any number is seen): a clear, repeatable
>= 1.3-1.5x over the measured cuBLASLt/PyTorch F.linear floor at
qkv / large_batch (M=32768, K=512, N=1536), confirmed under CUDA-GRAPH REPLAY
-- never a bare host timing loop.  docs/PROGRESS.md step 34 produced a false
"1.26-1.96x win" from exactly that mistake and a graph-replay remeasurement
corrected it to a clean negative; the same protocol is mandatory here.

PRIMARY judgment shape is qkv/large_batch: 32768/64 x 1536/128 = 6144 blocks
= 48 waves on 128 SMs, so occupancy/launch geometry is not the story and the
number is cleanly attributable to the accumulate tier.  out_proj at default
(M=1024, N=512 -> 4x16 = 64 blocks = half a wave) and anything at tiny are
reported for completeness but are NOT judged -- they are geometry-bound
regardless of accumulation type.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CU = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cu")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cpp")

D = 512
QKV_N = 3 * D

# (name, M, K, N, judged?)
CASES = [
    ("qkv      large_batch", 32768, D, QKV_N, True),    # PRIMARY
    ("qkv      long_seq   ",  8192, D, QKV_N, False),
    ("qkv      default    ",  1024, D, QKV_N, True),     # secondary
    ("out_proj large_batch", 32768, D, D,     False),
    ("out_proj default    ",  1024, D, D,     False),
]


def build():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="g4_4_mma_gemm",
        sources=[SRC_CPP, SRC_CU],
        build_directory=build_dir,
        with_cuda=True,
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "-Xptxas", "-v"],
        verbose=True,
    )


def graph_time(call, iters, replays, repeats=5):
    """us per call with all per-call host dispatch removed by graph capture."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            call()
    torch.cuda.synchronize()

    best = None
    for _ in range(repeats):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        t = e0.elapsed_time(e1) / replays / iters * 1000.0
        best = t if best is None else min(best, t)
    return best


def prof_kernel_us(call, launches=50):
    from torch.profiler import profile, ProfilerActivity
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(launches):
            call()
        torch.cuda.synchronize()
    tot, names = 0.0, []
    for ev in prof.key_averages():
        if ev.self_device_time_total <= 0:
            continue
        nm = ev.key
        low = nm.lower()
        if any(t in low for t in ("gemm", "cutlass", "wmma", "mma_", "xmma",
                                  "ampere", "sm80", "sm89")):
            tot += ev.self_device_time_total
            names.append(nm.split("(")[0][:70])
    return tot / launches, names


def tflops(M, K, N, us):
    return 2.0 * M * K * N / (us * 1e-6) / 1e12


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda}", flush=True)
    ext = build()
    ncfg = ext.num_cfg()
    print("\nconfigs:")
    for c in range(ncfg):
        print(f"  [{c}] {ext.cfg_name(c)}")

    g = torch.Generator(device=dev)

    # ---------------------------------------------------------------- launch
    # Debug correctness-of-launch on out_proj first (N=512, fewest column
    # tiles).  This is NOT the Stage 1 numerical check -- only "runs, finite,
    # non-degenerate, roughly the right magnitude".
    print("\n" + "=" * 78)
    print("LAUNCH SANITY (out_proj M=1024 K=512 N=512) -- finite / "
          "non-degenerate only, Stage 1 does the real numerics")
    g.manual_seed(0)
    inp = torch.randn(1024, D, device=dev, dtype=torch.float16, generator=g)
    w = (torch.randn(D, D, device=dev, dtype=torch.float16, generator=g) * 0.02).half()
    b = (torch.randn(D, device=dev, dtype=torch.float16, generator=g) * 0.02).half()
    ref = F.linear(inp, w, b)
    good_cfgs = []
    for c in range(ncfg):
        try:
            out = ext.mma_linear(c, inp, w, b)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  cfg[{c}] LAUNCH FAILED: {str(e)[:160]}")
            continue
        finite = bool(torch.isfinite(out).all())
        std = out.float().std().item()
        mx = (out.float() - ref.float()).abs().max().item()
        rel = ((out.float() - ref.float()).abs().max()
               / ref.float().abs().max().clamp_min(1e-9)).item()
        ok = finite and std > 1e-4
        print(f"  cfg[{c}] finite={finite} std={std:.5f} (ref std "
              f"{ref.float().std().item():.5f})  maxabs_vs_cublas={mx:.3e} "
              f"relmax={rel:.3e}  {'OK' if ok else 'DEGENERATE'}")
        if ok:
            good_cfgs.append(c)
    if not good_cfgs:
        print("\nNO CONFIG LAUNCHES CLEANLY -- stop.")
        return 1
    print(f"  launching configs: {good_cfgs}")

    # ---------------------------------------------------------------- timing
    for name, M, K, N, judged in CASES:
        print("\n" + "=" * 78)
        print(f"=== {name}  M={M} K={K} N={N}   "
              f"{'PRIMARY/JUDGED' if judged else 'reported, not judged'}")
        blocks = (M // 64) * (N // 128)
        print(f"    grid {M//64} x {N//128} = {blocks} blocks = "
              f"{blocks/128:.2f} waves on 128 SMs")
        g.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16, generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16, generator=g) * 0.02).half()
        out = torch.empty(M, N, device=dev, dtype=torch.float16)

        # graph iteration counts scale with problem size to keep capture cheap
        if M >= 32768:
            iters, replays = 5, 20
        elif M >= 8192:
            iters, replays = 10, 40
        else:
            iters, replays = 50, 200

        ref_call = lambda: torch.mm(inp, w.t(), out=out).add_(b)
        # use F.linear's own fused-bias path as the reference instead:
        ref_out = torch.empty(M, N, device=dev, dtype=torch.float16)

        def ref_call():
            torch.addmm(b, inp, w.t(), out=ref_out)

        t_ref = graph_time(ref_call, iters, replays)
        print(f"    pytorch F.linear(addmm) {t_ref:8.3f} us/call   "
              f"{tflops(M,K,N,t_ref):7.2f} TFLOPS   [cuBLASLt floor]")

        best = None
        for c in good_cfgs:
            call = lambda c=c: ext.mma_gemm(c, inp, w, b, out)
            try:
                t = graph_time(call, iters, replays)
            except Exception as e:
                print(f"    cfg[{c}] timing failed: {str(e)[:120]}")
                continue
            sp = t_ref / t
            print(f"    cfg[{c}] {ext.cfg_name(c):48s} {t:8.3f} us  "
                  f"{tflops(M,K,N,t):7.2f} TF  x{sp:.3f}")
            if best is None or t < best[1]:
                best = (c, t)
        if best is None:
            continue
        c, t = best
        sp = t_ref / t
        print(f"    BEST cfg[{c}]  {t:.3f} us  {tflops(M,K,N,t):.2f} TF  "
              f"x{sp:.3f} vs pytorch")

        # independent cross-check: profiler kernel time (step 34's rule --
        # two independent measurements that must agree).
        pr, pn = prof_kernel_us(ref_call, 50)
        pm, pmn = prof_kernel_us(lambda: ext.mma_gemm(c, inp, w, b, out), 50)
        print(f"    [prof] pytorch {pr:8.3f} us  {pn[:1]}")
        print(f"    [prof] mma     {pm:8.3f} us  {pmn[:1]}")
        if pm > 0:
            print(f"    [prof] ratio x{pr/pm:.3f}   (graph said x{sp:.3f}, "
                  f"delta {abs(pr/pm - sp)/sp*100:.1f}%)")

        if judged:
            gate = 1.3
            print(f"    >>> GATE ({gate}x): "
                  f"{'MET' if sp >= gate else 'NOT MET'}  (x{sp:.3f})")

    print("\n" + "=" * 78)
    print("Stage 0 complete. Judge on qkv/large_batch under graph replay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
