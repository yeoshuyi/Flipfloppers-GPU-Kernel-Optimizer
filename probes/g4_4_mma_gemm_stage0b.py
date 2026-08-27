#!/usr/bin/env python3
"""
G4.4 Stage 0b -- the MECHANISM test, run after Stage 0 (job 96) returned
x1.057 at the primary shape against a 1.3x gate.

Stage 0's number on its own only says "this particular hand-written kernel is
1.06x". It does NOT say whether the FP16-accumulate TIER is worth anything at
these shapes, because a slow hand-written kernel and a tier with no headroom
look identical from outside. Two additions separate them:

  1. BIGGER TILE (cfg 10-14, BM=128). At BM=64 each warp owns a 32x32 output
     tile -> MT*NT = 2*4 = 8 mma per k-step against 2 A-ldmatrix + 4
     B-ldmatrix. At BM=128 the warp owns 64x32 -> 16 mma against 4+4. That
     doubles the arithmetic per shared-memory read, which is the standard way
     a hand-written kernel gets from ~50% to ~80% of the mma issue ceiling.
     If FP16 accumulate has headroom here, this is where it shows up.

  2. FP32-ACCUMULATE CONTROLS (cfg 15-18). Identical tile, identical cp.async
     pipeline, identical ldmatrix, identical epilogue -- the ONLY difference
     is mma...f32.f16.f16.f32 instead of mma...f16.f16.f16.f16. This is the
     decisive A/B, in the spirit of docs/PROGRESS.md step 35's finding 3
     (in-place vs out-of-place split-K: change exactly one thing and price
     it). Ada's FP32-accumulate tensor-core rate is half its FP16-accumulate
     rate, so if the accumulate tier is what binds, cfg 15-18 must be
     dramatically slower than their FP16 twins. If they are NOT, the tier is
     not the binding constraint and no tuning of the FP16 path can recover
     the 2x it nominally offers.

Same fair-measurement protocol as Stage 0 and as docs/PROGRESS.md step 34:
CUDA-graph replay, cross-checked against torch.profiler kernel time.
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

CASES = [
    ("qkv      large_batch", 32768, D, QKV_N, True),    # PRIMARY
    ("qkv      default    ",  1024, D, QKV_N, True),    # secondary
    ("out_proj large_batch", 32768, D, D,     False),
]

# accF16 cfg -> its accF32 twin
TWIN = {0: 15, 1: 16, 10: 17, 11: 18}

# CLAUDE.md's precision ladder, dense, RTX 4090
PEAK_F32ACC = 165.2     # fp16/bf16 storage, FP32 accumulate (GeForce half rate)
PEAK_F16ACC = 330.3     # fp16 storage, FP16 accumulate  <- what G4.4 targets
HBM_GBS = 1008.0        # RTX 4090 spec bandwidth


def build():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="g4_4_mma_gemm_v2",
        sources=[SRC_CPP, SRC_CU],
        build_directory=build_dir,
        with_cuda=True,
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "-Xptxas", "-v", "-diag-suppress", "179"],
        verbose=True,
    )


def graph_time(call, iters, replays, repeats=5):
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
        low = ev.key.lower()
        if any(t in low for t in ("gemm", "cutlass", "wmma", "mma_", "xmma",
                                  "ampere", "sm80", "sm89")):
            tot += ev.self_device_time_total
            names.append(ev.key.split("(")[0][:70])
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
        print(f"  [{c:2d}] {ext.cfg_name(c)}")

    g = torch.Generator(device=dev)
    summary = []

    for name, M, K, N, judged in CASES:
        print("\n" + "=" * 78)
        print(f"=== {name}  M={M} K={K} N={N}   "
              f"{'PRIMARY/JUDGED' if judged else 'reported, not judged'}")
        flop = 2.0 * M * K * N
        bytes_min = (M * K + N * K + M * N) * 2.0
        print(f"    {flop/1e9:.2f} GFLOP, >= {bytes_min/1e6:.1f} MB of "
              f"compulsory HBM traffic")
        print(f"    memory-bound floor at {HBM_GBS:.0f} GB/s: "
              f"{bytes_min/HBM_GBS/1e3:.2f} us "
              f"({flop/(bytes_min/HBM_GBS/1e9)/1e12:.1f} TFLOPS ceiling)")
        print(f"    tier ceilings: FP32-acc {PEAK_F32ACC} TF, "
              f"FP16-acc {PEAK_F16ACC} TF")

        g.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ref_out = torch.empty(M, N, device=dev, dtype=torch.float16)

        if M >= 32768:
            iters, replays = 5, 20
        elif M >= 8192:
            iters, replays = 10, 40
        else:
            iters, replays = 50, 200

        def ref_call():
            torch.addmm(b, inp, w.t(), out=ref_out)

        t_ref = graph_time(ref_call, iters, replays)
        tf_ref = tflops(M, K, N, t_ref)
        print(f"\n    pytorch F.linear(addmm) {t_ref:9.3f} us  {tf_ref:7.2f} TF"
              f"   = {tf_ref/PEAK_F32ACC*100:5.1f}% of the FP32-acc tier "
              f"ceiling   [THE FLOOR TO BEAT]")

        times = {}
        for c in range(ncfg):
            call = lambda c=c: ext.mma_gemm(c, inp, w, b, out)
            try:
                call()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"    cfg[{c:2d}] unavailable: {str(e)[:100]}")
                continue
            if not bool(torch.isfinite(out).all()) or out.float().std() < 1e-4:
                print(f"    cfg[{c:2d}] DEGENERATE output, skipped")
                continue
            t = graph_time(call, iters, replays)
            times[c] = t
            tf = tflops(M, K, N, t)
            peak = PEAK_F32ACC if ext.cfg_name(c).startswith("accF32") else PEAK_F16ACC
            print(f"    cfg[{c:2d}] {ext.cfg_name(c):44s} {t:9.3f} us "
                  f"{tf:7.2f} TF  x{t_ref/t:.3f}  ({tf/peak*100:4.1f}% of "
                  f"its own tier ceiling)")

        f16 = {c: t for c, t in times.items()
               if not ext.cfg_name(c).startswith("accF32")}
        if not f16:
            continue
        bc = min(f16, key=f16.get)
        bt = f16[bc]
        sp = t_ref / bt
        print(f"\n    BEST FP16-accumulate cfg[{bc}]  {bt:.3f} us  "
              f"{tflops(M,K,N,bt):.2f} TF  x{sp:.3f} vs pytorch")

        pr, pn = prof_kernel_us(ref_call, 50)
        pm, pmn = prof_kernel_us(lambda: ext.mma_gemm(bc, inp, w, b, out), 50)
        print(f"    [prof] pytorch {pr:9.3f} us  {pn[:1]}")
        print(f"    [prof] mma     {pm:9.3f} us  {pmn[:1]}")
        print(f"    [prof] ratio x{pr/pm:.3f}  (graph x{sp:.3f}, agree to "
              f"{abs(pr/pm - sp)/sp*100:.1f}%)")

        print("\n    --- THE A/B: same kernel, only the accumulate type "
              "changes ---")
        for cf16, cf32 in TWIN.items():
            if cf16 in times and cf32 in times:
                r = times[cf32] / times[cf16]
                print(f"    cfg[{cf16:2d}] accF16 {times[cf16]:9.3f} us  vs  "
                      f"cfg[{cf32:2d}] accF32 {times[cf32]:9.3f} us   "
                      f"-> FP16 accumulate is x{r:.3f}"
                      f"   (tier says it should be ~x2.0 if compute-bound)")
                summary.append((name, cf16, r))

        if judged:
            print(f"\n    >>> GATE (1.3x): {'MET' if sp >= 1.3 else 'NOT MET'}"
                  f"  (x{sp:.3f})")

    print("\n" + "=" * 78)
    print("A/B SUMMARY -- value of the FP16-accumulate tier, everything else "
          "held identical")
    for nm, c, r in summary:
        print(f"  {nm}  cfg[{c}]  accF32/accF16 = x{r:.3f}")
    print("\nIf these ratios are near 1.0 the accumulate tier is not the "
          "binding constraint at these shapes and G4.4 has no headroom to "
          "recover, regardless of how well the kernel is tuned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
