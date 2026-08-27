#!/usr/bin/env python3
"""
G6.6 step 1 -- is there a faster NATIVE cuBLASLt algorithm for the FFN's two
TF32 GEMM shapes than the one PyTorch's default heuristic already picks?

Fact this is built on (docs/PROGRESS.md step 31): a fresh ncu pass on the
shipped model found the FFN's TF32 CUTLASS GEMM (ffn_in / ffn_out, plain
F.linear) at ~47% of TF32 peak (82.6 TFLOPS) with ~26% occupancy.

Why this is not step 25 again: step 25's max-autotune failure was caused by
inductor substituting TRITON GEMMs, which emulate TF32 with a 3-pass FP32
decomposition -- a different reduction algorithm. Here every candidate comes
from cublasLtMatmulAlgoGetHeuristic, i.e. the same library and the same native
TF32 tensor-core datapath PyTorch itself dispatches into; only tile size,
split-K, stages and swizzle differ.

This probe DOES NOT touch benchmark.py. It is the cheap gate: if no
heuristic candidate beats PyTorch's own F.linear by >5% on the same buffers,
the conclusion is "cuBLASLt's default heuristic is already near-optimal for
this shape" and the investigation stops here.

Shapes come from the real sweep (jobs/g1_6_g2_3_sweep.sbatch):
  tiny B1 S64    -> M=64
  default B8 S128 -> M=1024      <- the shape step 31 profiled
  long_seq B8 S1024 -> M=8192
  large_batch B256 S128 -> M=32768
with (K=512, N=2048) for ffn_in and (K=2048, N=512) for ffn_out.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "csrc", "cublaslt_algo.cpp")

D, FF = 512, 2048
M_LIST = [64, 1024, 8192, 32768]
REQUESTED = 16
MAX_WS = 32 * 1024 * 1024
WARMUP = 30
ITERS = 200


def build_ext():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    # with_cuda=True is mandatory even with no .cu source (step 32).
    return load(name="cublaslt_algo", sources=[SRC], build_directory=build_dir,
                with_cuda=True, extra_ldflags=["-lcublasLt"], verbose=False)


def time_pytorch(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def one_case(ext, name, M, K, N, use_bias, dev, gen):
    inp = torch.randn(M, K, device=dev, dtype=torch.float32, generator=gen)
    w = torch.randn(N, K, device=dev, dtype=torch.float32, generator=gen) * 0.02
    b = (torch.randn(N, device=dev, dtype=torch.float32, generator=gen) * 0.02
         if use_bias else None)
    out = torch.empty(M, N, device=dev, dtype=torch.float32)

    flops = 2.0 * M * N * K

    # ---- reference: what PyTorch already achieves, same buffers ----
    if use_bias:
        ref_fn = lambda: F.linear(inp, w, b)          # noqa: E731
    else:
        ref_fn = lambda: F.linear(inp, w)             # noqa: E731
    t_ref = time_pytorch(ref_fn)
    ref = ref_fn()
    torch.cuda.synchronize()

    pid = ext.create_problem(M, N, K, use_bias, MAX_WS, REQUESTED)
    n = ext.num_algos(pid)

    print(f"\n=== {name}  M={M} K={K} N={N} bias={use_bias} ===", flush=True)
    print(f"  pytorch F.linear : {t_ref*1000:9.2f} us   "
          f"{flops/(t_ref*1e-3)/1e12:6.2f} TFLOPS", flush=True)
    print(f"  heuristic returned {n} candidate(s)", flush=True)
    if n == 0:
        return

    rows = []
    for i in range(n):
        try:
            t = ext.time_algo(pid, i, inp, w, b, out, WARMUP, ITERS)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [{i:2d}] SKIP  {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:90]}", flush=True)
            continue
        ext.run(pid, i, inp, w, b, out)
        torch.cuda.synchronize()
        md = (out - ref).abs().max().item()
        rows.append((t, i, md))
        print(f"  [{i:2d}] {t*1000:9.2f} us  {flops/(t*1e-3)/1e12:6.2f} TFLOPS  "
              f"x{t_ref/t:5.3f}  maxdiff={md:.3e}  {ext.algo_info(pid, i)}",
              flush=True)

    if not rows:
        print("  no usable candidate", flush=True)
        return
    rows.sort()
    t_best, i_best, md_best = rows[0]
    speedup = t_ref / t_best
    verdict = "WIN" if speedup > 1.05 else ("noise" if speedup > 0.95 else "LOSS")
    print(f"  BEST -> algo[{i_best}]  {t_best*1000:.2f} us  "
          f"speedup vs pytorch = {speedup:.4f}  ({verdict})  "
          f"maxdiff={md_best:.3e}", flush=True)


def main():
    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
          "| cuda", torch.version.cuda, flush=True)
    print(f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)

    ext = build_ext()
    gen = torch.Generator(device=dev)

    for M in M_LIST:
        for use_bias in (False, True):
            gen.manual_seed(1234)
            one_case(ext, "ffn_in ", M, D, FF, use_bias, dev, gen)
            gen.manual_seed(5678)
            one_case(ext, "ffn_out", M, FF, D, use_bias, dev, gen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
