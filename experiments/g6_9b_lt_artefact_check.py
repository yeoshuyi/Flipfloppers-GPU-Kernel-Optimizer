#!/usr/bin/env python3
"""
G6.9 Phase 2 step 5 -- artefact rejection for the two retained qkv candidates.

g6_9 found best cuBLASLt heuristic candidate beats idx0 by +21% (M8192 d128)
and +2.9% (M8192 d1024) for the fused qkv GEMM.  But step 43's census showed
PyTorch's F.linear for M8192 d128 qkv already runs `ampere_fp16_s16816gemm_
fp16_128x64` at ~11.5us == our "best", NOT idx0's 14.5us.  If the SHIPPED path
(F.linear) is already at the fast algo, "idx0" is a strawman and the +21% is
an artefact of comparing the cuBLASLt-heuristic-default vs a better cuBLASLt
algo -- a win that does not exist end-to-end.

Times, on the SAME [M,K] fp16 input / [N,K] fp16 weight / [N] fp16 bias:
  A. torch F.linear(x, w, b)          -- the actual shipped dispatch
  B. cuBLASLt run(idx0)               -- g6_9's baseline
  C. cuBLASLt run(best_k)             -- g6_9's winner
  D. torch.addmm(b, x, w.t())         -- F.linear's usual lowering, for cross-check
and reports the kernel name each dispatches (torch.profiler).

VERDICT:
  * F.linear ~= C (both << B)  -> idx0 is a strawman; NO real opportunity. STOP.
  * F.linear ~= B  and C << B  -> real: offline selection could help. -> Phase 3.
"""
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F16_SRC = os.path.join(ROOT, "csrc", "cublaslt_algo_fp16.cpp")
DEV = torch.device("cuda")
WARM, ITERS, REPS = 40, 200, 4
MAX_WS = 32 * 1024 * 1024


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    return load(name="g6_9b_lt_f16", sources=[F16_SRC], build_directory=bd,
                with_cuda=True, extra_ldflags=["-lcublasLt"], verbose=False)


def etime(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(REPS):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        for _ in range(ITERS):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / ITERS)
    return best * 1e3  # us


def kernels(fn, n=12):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        p = fh.name
    prof.export_chrome_trace(p)
    ev = json.load(open(p))["traceEvents"]
    os.unlink(p)
    agg = {}
    for e in ev:
        if (e.get("cat") or "").lower() == "kernel":
            agg[e["name"]] = agg.get(e["name"], [0, 0.0])
            agg[e["name"]][0] += 1
            agg[e["name"]][1] += float(e.get("dur", 0))
    return sorted(([k, v[0] / n, v[1] / n] for k, v in agg.items()),
                  key=lambda r: -r[2])


def main():
    ext = build()
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"gpu={torch.cuda.get_device_name(0)}")
    try:
        print("preferred_blas_library:",
              torch.backends.cuda.preferred_blas_library())
    except Exception:  # noqa: BLE001
        pass
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print()

    for (M, N, K, want_k) in [(8192, 384, 128, 5), (8192, 3072, 1024, 1)]:
        print(f"==== qkv  M{M} N{N} K{K}  (fp16, BIAS) ====")
        g = torch.Generator(device=DEV).manual_seed(0)
        x = (torch.randn(M, K, device=DEV, dtype=torch.float16, generator=g) * 0.1)
        w = (torch.randn(N, K, device=DEV, dtype=torch.float16, generator=g) * 0.05)
        b = (torch.randn(N, device=DEV, dtype=torch.float16, generator=g) * 0.05)
        o = torch.empty(M, N, device=DEV, dtype=torch.float16)

        pid = ext.create_problem(M, N, K, True, MAX_WS, 32, 2)
        na = ext.num_algos(pid)
        # find the actual best index (reproduce g6_9)
        times = [ext.time_algo(pid, k, x, w, b, o, WARM, ITERS) for k in range(na)]
        bk = min(range(na), key=lambda k: times[k])
        print(f"  cuBLASLt: {na} candidates; idx0={times[0]*1e3:.2f}us  "
              f"best=idx{bk} {times[bk]*1e3:.2f}us  "
              f"({(times[0]-times[bk])/times[0]*100:+.1f}%)")
        print(f"  idx0  algo: {ext.algo_info(pid, 0)}")
        print(f"  best  algo: {ext.algo_info(pid, bk)}")

        tA = etime(lambda: F.linear(x, w, b))
        tB = etime(lambda: ext.run(pid, 0, x, w, b, o))
        tC = etime(lambda: ext.run(pid, bk, x, w, b, o))
        tD = etime(lambda: torch.addmm(b, x, w.t(), out=o))
        print(f"  A F.linear(x,w,b)      {tA:8.2f} us")
        print(f"  B cuBLASLt run(idx0)   {tB:8.2f} us")
        print(f"  C cuBLASLt run(best)   {tC:8.2f} us")
        print(f"  D torch.addmm          {tD:8.2f} us")

        # correctness: best vs F.linear
        yA = F.linear(x, w, b)
        yC = ext.lt_linear(pid, bk, x, w, b)
        print(f"  max|C - A| = {(yC.float()-yA.float()).abs().max().item():.3e}")

        print("  F.linear dispatches:")
        for k, c, u in kernels(lambda: F.linear(x, w, b))[:4]:
            print(f"     {u:8.2f}us x{c:.2f}  {k[:78]}")
        print("  cuBLASLt run(best) dispatches:")
        for k, c, u in kernels(lambda: ext.run(pid, bk, x, w, b, o))[:4]:
            print(f"     {u:8.2f}us x{c:.2f}  {k[:78]}")

        # verdict
        if tA <= tB * 1.05 and tC <= tB * 0.92:
            v = ("F.linear ~= idx0 (slow); best is real -> Phase 3"
                 if tA > tC * 1.05 else
                 "F.linear ~= best (already fast); idx0 is a STRAWMAN -> no opportunity")
        elif tA <= tC * 1.05:
            v = "F.linear already at/below best -> idx0 is a STRAWMAN -> no opportunity"
        else:
            v = "F.linear slower than best by >5% -> offline selection could help -> Phase 3"
        print(f"  VERDICT: {v}\n")
        del x, w, b, o
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
