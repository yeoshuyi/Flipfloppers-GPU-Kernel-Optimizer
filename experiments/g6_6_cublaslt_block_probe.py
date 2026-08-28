#!/usr/bin/env python3
"""
G6.6 step 1b -- the two questions run 71 left open.

Run 71 (results/g6_6_cublaslt_algo_probe_run71.log) said:
  * pure GEMM, M=1024 (the DEFAULT shape step 31 profiled): the best of 8
    heuristic candidates beats PyTorch's own pick by 1.001x / 1.001x. Clean
    negative -- cuBLASLt's default heuristic is already the right algorithm.
  * M=64 (TINY): 1.32x / 1.49x, from split-K variants PyTorch does not pick.
  * bias=True everywhere: eager F.linear is slower than a cuBLASLt BIAS
    epilogue (1.07x ffn_in, 1.38x ffn_out at M=1024).

Both of the apparent wins have a confound that has to be excluded before any of
this reaches benchmark.py:

Q1 (TINY). At M=64 the GEMM is 134 MFLOP; a ~9us "GPU time" measured over a
   back-to-back launch loop may just be the CPU's issue rate. time_algo2
   returns the CPU issue time for the same loop -- if cpu ~= gpu, the number is
   launch overhead, and the shipped TINY path already runs under CUDA graphs
   where launch overhead is gone. That would make the 1.49x an artifact.

Q2 (bias). The shipped path is torch.compile(mode="reduce-overhead"), NOT eager.
   Inductor routinely lowers addmm to mm + a pointwise epilogue fused into the
   following op (here GELU for ffn_in, the residual add for ffn_out), which is
   the same saving the cuBLASLt BIAS epilogue buys. Comparing raw cuBLASLt
   against EAGER F.linear therefore measures a gap the shipped model may not
   have.

So this probe compares the real FFN block (ffn_in + bias -> GELU -> ffn_out +
bias) three ways, with launch overhead removed identically from all of them by
capturing each into a CUDA graph -- the same condition the shipped model runs
under:
   A  graph_torch : F.linear / gelu / F.linear, CUDA-graph replayed
   B  graph_lt    : cuBLASLt best algo + BIAS epilogue / gelu / same, replayed
   C  compiled    : torch.compile(mode="reduce-overhead"), the shipped path
Anything worth shipping has to beat C, not eager.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "csrc", "cublaslt_algo.cpp")

D, FF = 512, 2048
REQUESTED = 16
MAX_WS = 32 * 1024 * 1024
WARMUP = 30
ITERS = 300


def build_ext():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(name="cublaslt_algo", sources=[SRC], build_directory=build_dir,
                with_cuda=True, extra_ldflags=["-lcublasLt"], verbose=False)


def pick_best(ext, M, K, N, use_bias, inp, w, b, out):
    """Search + time all heuristic candidates; return (idx, gpu_ms, pid)."""
    pid = ext.create_problem(M, N, K, use_bias, MAX_WS, REQUESTED)
    best = None
    for i in range(ext.num_algos(pid)):
        try:
            t = ext.time_algo(pid, i, inp, w, b, out, WARMUP, ITERS)
        except Exception:                                  # noqa: BLE001
            continue
        if best is None or t < best[1]:
            best = (i, t)
    return best[0], best[1], pid


def cpu_bound_check(ext, M, K, N, name):
    """Q1: is the M=64 measurement CPU-issue-bound rather than GPU-bound?"""
    dev = torch.device("cuda")
    inp = torch.randn(M, K, device=dev)
    w = torch.randn(N, K, device=dev) * 0.02
    b = torch.randn(N, device=dev) * 0.02
    out = torch.empty(M, N, device=dev)
    pid = ext.create_problem(M, N, K, True, MAX_WS, REQUESTED)
    print(f"\n--- Q1 launch-bound check: {name} M={M} K={K} N={N} ---", flush=True)
    for i in range(ext.num_algos(pid)):
        try:
            gpu_ms, cpu_ms = ext.time_algo2(pid, i, inp, w, b, out, WARMUP, ITERS)
        except Exception:                                  # noqa: BLE001
            continue
        flag = "LAUNCH-BOUND" if cpu_ms > 0.9 * gpu_ms else "gpu-bound"
        print(f"  [{i:2d}] gpu={gpu_ms*1000:8.2f} us  cpu_issue={cpu_ms*1000:8.2f} us"
              f"   {flag}   {ext.algo_info(pid, i)}", flush=True)


def graph_time(fn, warmup=WARMUP, iters=ITERS):
    """Capture fn into a CUDA graph, then time replays. Removes ALL launch
    overhead, which is the condition the shipped reduce-overhead path runs in."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters, res


def plain_time(fn, warmup=WARMUP, iters=ITERS):
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


def block_case(ext, M):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev); g.manual_seed(4242)
    x = torch.randn(M, D, device=dev, generator=g)
    w1 = torch.randn(FF, D, device=dev, generator=g) * 0.02
    b1 = torch.randn(FF, device=dev, generator=g) * 0.02
    w2 = torch.randn(D, FF, device=dev, generator=g) * 0.02
    b2 = torch.randn(D, device=dev, generator=g) * 0.02

    h = torch.empty(M, FF, device=dev)
    o = torch.empty(M, D, device=dev)

    print(f"\n=== FFN block, M={M} (K=512->2048->512) ===", flush=True)

    # ---- pick the best cuBLASLt algo for each half, WITH the bias epilogue ----
    i1, t1, p1 = pick_best(ext, M, D, FF, True, x, w1, b1, h)
    gel = F.gelu(h, approximate="none")
    i2, t2, p2 = pick_best(ext, M, FF, D, True, gel, w2, b2, o)
    print(f"  best lt algo ffn_in  = [{i1}] {t1*1000:.2f} us  {ext.algo_info(p1, i1)}",
          flush=True)
    print(f"  best lt algo ffn_out = [{i2}] {t2*1000:.2f} us  {ext.algo_info(p2, i2)}",
          flush=True)

    def torch_block():
        hh = F.linear(x, w1, b1)
        return F.linear(F.gelu(hh, approximate="none"), w2, b2)

    def lt_block():
        ext.run(p1, i1, x, w1, b1, h)
        gg = F.gelu(h, approximate="none")
        ext.run(p2, i2, gg, w2, b2, o)
        return o

    t_eager = plain_time(torch_block)
    t_gtorch, ref = graph_time(torch_block)
    t_glt, got = graph_time(lt_block)

    compiled = torch.compile(torch_block, mode="reduce-overhead")
    t_comp = plain_time(compiled)

    torch.cuda.synchronize()
    md = (got.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().item()

    print(f"  eager    F.linear chain      : {t_eager*1000:8.2f} us", flush=True)
    print(f"  A graph_torch (no launch ovh): {t_gtorch*1000:8.2f} us", flush=True)
    print(f"  C compiled reduce-overhead   : {t_comp*1000:8.2f} us   <- SHIPPED",
          flush=True)
    print(f"  B graph_lt  (cuBLASLt best)  : {t_glt*1000:8.2f} us", flush=True)
    print(f"  B vs A  = {t_gtorch/t_glt:.4f}x     B vs C = {t_comp/t_glt:.4f}x",
          flush=True)
    print(f"  maxdiff(B,A) = {md:.3e}   (|ref|max = {denom:.3f})", flush=True)


def main():
    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)

    ext = build_ext()

    # Q1
    cpu_bound_check(ext, 64, D, FF, "ffn_in ")
    cpu_bound_check(ext, 64, FF, D, "ffn_out")

    # Q2
    for M in (64, 1024, 8192):
        try:
            block_case(ext, M)
        except Exception as exc:                            # noqa: BLE001
            print(f"  block_case M={M} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
