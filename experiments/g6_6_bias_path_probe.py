#!/usr/bin/env python3
"""
G6.6 step 1c -- localise the M=1024 win. Is it worth an extension at all?

Run 72 (results/g6_6_cublaslt_block_probe_run72.log), all CUDA-graph captured so
launch overhead is removed identically from every variant:
    M=64    graph_lt 14.83us vs graph_torch 21.69us   1.46x
    M=1024  graph_lt 70.94us vs graph_torch 84.64us   1.19x
    M=8192  graph_lt 586.7us vs graph_torch 584.4us   1.00x

But run 71 already showed the pure-GEMM algorithm search is worth 1.001x at
M=1024. So the 1.19x is NOT algorithm selection -- it is PyTorch's BIAS path.
Run 71's per-GEMM numbers pin it down:

    ffn_out M=1024:  F.linear no bias 32.37us | F.linear WITH bias 44.62us
                     cuBLASLt same algo WITH bias epilogue        32.61us

i.e. adding a bias costs PyTorch 12.2us and costs cuBLASLt 0.2us. And it is not
an extra memory pass: ffn_in's output is 4x larger (8MB vs 2MB) yet its bias
only costs 2.5us. PyTorch is picking a different, slower kernel when a bias is
present, not doing extra bandwidth.

If that is all it is, the fix does not need a C++ extension at all -- writing
`F.linear(x, w) + b` instead of `F.linear(x, w, b)` would recover it in one
line of pure PyTorch, bit-exact in the GEMM and with the bias add fusable by
inductor into the GELU / residual that already follows. That is a far better
thing to ship than a hand-rolled cuBLASLt path, so it has to be excluded first.

Variants, each timed both CUDA-graph-captured (GPU kernel time, no launch
overhead) and under torch.compile(mode="reduce-overhead") (the shipped
condition):
    A  F.linear(x,w1,b1) -> gelu -> F.linear(h,w2,b2)      [what ships today]
    D  F.linear(x,w1)+b1 -> gelu -> F.linear(h,w2)+b2      [one-line torch fix]
    E  bias split for ffn_out only (the half with the 12us gap)
    B  cuBLASLt best algo + BIAS epilogue                   [the extension]
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


def pick_best(ext, M, K, N, inp, w, b, out):
    pid = ext.create_problem(M, N, K, True, MAX_WS, REQUESTED)
    best = None
    for i in range(ext.num_algos(pid)):
        try:
            t = ext.time_algo(pid, i, inp, w, b, out, WARMUP, ITERS)
        except Exception:                                   # noqa: BLE001
            continue
        if best is None or t < best[1]:
            best = (i, t)
    return best[0], pid


def graph_time(fn):
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
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / ITERS, res


def plain_time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / ITERS


def case(ext, M):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev); g.manual_seed(4242)
    x = torch.randn(M, D, device=dev, generator=g)
    w1 = torch.randn(FF, D, device=dev, generator=g) * 0.02
    b1 = torch.randn(FF, device=dev, generator=g) * 0.02
    w2 = torch.randn(D, FF, device=dev, generator=g) * 0.02
    b2 = torch.randn(D, device=dev, generator=g) * 0.02
    h = torch.empty(M, FF, device=dev)
    o = torch.empty(M, D, device=dev)

    print(f"\n=== FFN block, M={M} ===", flush=True)

    def A():
        hh = F.linear(x, w1, b1)
        return F.linear(F.gelu(hh, approximate="none"), w2, b2)

    def Dv():
        hh = F.linear(x, w1) + b1
        return F.linear(F.gelu(hh, approximate="none"), w2) + b2

    def E():
        hh = F.linear(x, w1, b1)
        return F.linear(F.gelu(hh, approximate="none"), w2) + b2

    i1, p1 = pick_best(ext, M, D, FF, x, w1, b1, h)
    gel = F.gelu(h, approximate="none")
    i2, p2 = pick_best(ext, M, FF, D, gel, w2, b2, o)

    def B():
        ext.run(p1, i1, x, w1, b1, h)
        gg = F.gelu(h, approximate="none")
        ext.run(p2, i2, gg, w2, b2, o)
        return o

    tA, refA = graph_time(A)
    tD, refD = graph_time(Dv)
    tE, refE = graph_time(E)
    tB, refB = graph_time(B)

    cA = plain_time(torch.compile(A, mode="reduce-overhead"))
    cD = plain_time(torch.compile(Dv, mode="reduce-overhead"))
    cE = plain_time(torch.compile(E, mode="reduce-overhead"))

    torch.cuda.synchronize()
    mx = refA.float().abs().max().item()
    dD = (refD.float() - refA.float()).abs().max().item()
    dE = (refE.float() - refA.float()).abs().max().item()
    dB = (refB.float() - refA.float()).abs().max().item()

    print(f"  cuda-graph kernel time (no launch overhead):", flush=True)
    print(f"    A  F.linear(+bias)      {tA*1000:8.2f} us   1.0000x  (shipped)",
          flush=True)
    print(f"    D  mm + bias both       {tD*1000:8.2f} us  {tA/tD:7.4f}x  "
          f"maxdiff={dD:.3e}", flush=True)
    print(f"    E  mm + bias ffn_out    {tE*1000:8.2f} us  {tA/tE:7.4f}x  "
          f"maxdiff={dE:.3e}", flush=True)
    print(f"    B  cuBLASLt bias-epi    {tB*1000:8.2f} us  {tA/tB:7.4f}x  "
          f"maxdiff={dB:.3e}", flush=True)
    print(f"  torch.compile reduce-overhead (shipped condition):", flush=True)
    print(f"    A {cA*1000:8.2f} us | D {cD*1000:8.2f} us ({cA/cD:.4f}x) | "
          f"E {cE*1000:8.2f} us ({cA/cE:.4f}x)", flush=True)
    print(f"  |ref|max = {mx:.3f}", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    ext = build_ext()
    for M in (64, 1024, 8192, 32768):
        try:
            case(ext, M)
        except Exception as exc:                            # noqa: BLE001
            print(f"  M={M} FAILED {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
