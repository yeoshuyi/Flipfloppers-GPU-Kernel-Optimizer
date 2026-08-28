#!/usr/bin/env python3
"""
G6.7 step 1b -- FAIR re-measurement of the FP16 attention GEMMs.

Run 82 (probes/g6_7_cublaslt_fp16_probe.py) appeared to show 1.26x/1.96x at
M=64, but its own time_algo2 instrumentation flags the result as an artifact:

  * PyTorch's F.linear reference was LAUNCH-BOUND in every M=64 case --
    cpu-issue ~= gpu time (5.63/5.66, 7.75/7.83, 7.70/7.74, 7.80/7.83 us).
    Its "GPU time" is really the Python dispatch rate, not kernel time.
  * The identical measurement moved 5.63 -> 7.70 us between two passes (37%).
  * The two loops have DIFFERENT dispatch floors: PyTorch's Python loop bottoms
    out at ~7.8 us/call, time_algo2's pure-C++ loop at ~3.3-3.9 us/call. Every
    M=64 FP16 kernel here is below both floors, so neither loop resolves it.
  * torch.profiler shows PyTorch and the "winning" cuBLASLt candidate
    dispatching the SAME kernel:
    cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8
    -- and maxdiff is exactly 0.000e+00 (bit-identical output). One kernel
    cannot be 1.96x faster than itself.

(This is precisely the confound step 33 anticipated with time_algo2. It did not
bite there because TF32's M=64 GEMMs were 7.7-30 us -- above both floors. FP16
is ~3x faster, which puts these GEMMs underneath the dispatch floor.)

So this probe removes launch overhead SYMMETRICALLY and measures kernel time
two independent ways:
  A. CUDA-graph replay -- N back-to-back calls captured in one graph, replayed
     and timed with CUDA events. Zero per-call dispatch on either side. This is
     also what the real model does (torch.compile mode="reduce-overhead").
  B. torch.profiler per-kernel device time -- self_device_time_total / launches,
     the GPU's own duration for the kernel, independent of any host loop.

If A and B both say the two sides are equal, the run-82 "win" was harness
overhead and G6.7 is a clean negative.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "csrc", "cublaslt_algo_fp16.cpp")

D = 512
QKV_N = 3 * D
M_LIST = [64]
REQUESTED = 16
MAX_WS = 32 * 1024 * 1024
RED_MASK = 2          # fp32 partial reduction only (G6.4b discipline)
GRAPH_ITERS = 50      # calls captured inside one graph
REPLAYS = 200
REPEATS = 5
PROF_LAUNCHES = 100


def build_ext():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(name="cublaslt_algo_fp16", sources=[SRC],
                build_directory=build_dir, with_cuda=True,
                extra_ldflags=["-lcublasLt"], verbose=False)


# --------------------------------------------------------------------------
# A. CUDA-graph replay timing -- identical harness for both sides.
# --------------------------------------------------------------------------
def graph_time(call, iters=GRAPH_ITERS, replays=REPLAYS, repeats=REPEATS):
    """us per call, with all per-call host dispatch removed by graph capture."""
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
        t = e0.elapsed_time(e1) / replays / iters * 1000.0   # us per call
        best = t if best is None else min(best, t)
    return best, g


# --------------------------------------------------------------------------
# B. Profiler per-kernel device time.
# --------------------------------------------------------------------------
def prof_time(call, launches=PROF_LAUNCHES):
    """(us per launch summed over gemm kernels, [kernel names])."""
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
        if any(t in nm.lower() for t in ("gemm", "cutlass", "xmma", "wmma",
                                         "ampere", "sm80", "sm89", "splitk",
                                         "reduce")):
            tot += ev.self_device_time_total
            names.append(nm)
    return tot / launches, names


def short(nm):
    nm = nm.replace("void cutlass::Kernel2<", "").replace("void ", "")
    return nm.split("(")[0][:96]


def one_case(ext, name, M, K, N, dev, gen):
    print(f"\n{'='*78}\n=== {name}  M={M} K={K} N={N} bias=True  "
          f"red_mask={RED_MASK} ===", flush=True)
    gen.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
    inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=gen)
    w = (torch.randn(N, K, device=dev, dtype=torch.float16, generator=gen)
         * 0.02).half()
    b = (torch.randn(N, device=dev, dtype=torch.float16, generator=gen)
         * 0.02).half()
    out = torch.empty(M, N, device=dev, dtype=torch.float16)
    ref = F.linear(inp, w, b)
    torch.cuda.synchronize()

    pid = ext.create_problem(M, N, K, True, MAX_WS, REQUESTED, RED_MASK)
    n = ext.num_algos(pid)

    # ---- A: CUDA-graph replay, both sides, same harness ----
    py_out = [None]

    def py_call():
        py_out[0] = F.linear(inp, w, b)

    t_py, _g = graph_time(py_call)
    print(f"  [graph] pytorch F.linear      {t_py:7.3f} us/call", flush=True)

    rows = []
    for i in range(n):
        try:
            t, _gi = graph_time(lambda: ext.run(pid, i, inp, w, b, out))
        except Exception as exc:                               # noqa: BLE001
            print(f"  [graph] algo[{i:2d}] SKIP {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:70]}", flush=True)
            continue
        ext.run(pid, i, inp, w, b, out)
        torch.cuda.synchronize()
        md = (out.float() - ref.float()).abs().max().item()
        rows.append((t, i, md))
        print(f"  [graph] algo[{i:2d}]              {t:7.3f} us/call  "
              f"x{t_py/t:6.4f}  maxdiff={md:.3e}  {ext.algo_info(pid, i)}",
              flush=True)

    if not rows:
        print("  no usable candidate", flush=True)
        return None
    rows.sort()
    t_best, i_best, md_best = rows[0]
    sp_graph = t_py / t_best

    # ---- B: profiler kernel time, both sides ----
    t_py_k, py_names = prof_time(lambda: F.linear(inp, w, b))
    t_lt_k, lt_names = prof_time(lambda: ext.run(pid, i_best, inp, w, b, out))
    print(f"\n  [prof]  pytorch kernel time   {t_py_k:7.3f} us/launch", flush=True)
    for nm in dict.fromkeys(py_names):
        print(f"            {short(nm)}", flush=True)
    print(f"  [prof]  algo[{i_best}] kernel time   {t_lt_k:7.3f} us/launch  "
          f"x{t_py_k/max(t_lt_k,1e-9):6.4f}", flush=True)
    for nm in dict.fromkeys(lt_names):
        print(f"            {short(nm)}", flush=True)
    same_kernel = set(map(short, py_names)) == set(map(short, lt_names))
    print(f"  same kernel dispatched by both sides? "
          f"{'YES' if same_kernel else 'no'}", flush=True)

    sp_prof = t_py_k / max(t_lt_k, 1e-9)
    print(f"\n  BEST algo[{i_best}]:  graph x{sp_graph:.4f}   "
          f"prof x{sp_prof:.4f}   maxdiff={md_best:.3e}", flush=True)
    return (f"{name} M={M} K={K} N={N}", sp_graph, sp_prof, t_py, t_best,
            md_best, same_kernel)


def main():
    if not torch.cuda.is_available():
        print("no CUDA", file=sys.stderr)
        return 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
          "| cuda", torch.version.cuda, flush=True)

    ext = build_ext()
    gen = torch.Generator(device=dev)
    summary = []
    for M in M_LIST:
        for nm, K, N in (("qkv     ", D, QKV_N), ("out_proj", D, D)):
            r = one_case(ext, nm, M, K, N, dev, gen)
            if r:
                summary.append(r)

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY -- launch overhead removed symmetrically", flush=True)
    print(f"  {'shape':<34} {'graph':>9} {'prof':>9}   maxdiff   same-kernel",
          flush=True)
    for tag, sg, sp, tpy, tb, md, sk in summary:
        print(f"  {tag:<34} x{sg:7.4f} x{sp:7.4f}   {md:.1e}   {sk}", flush=True)
    passed = [s for s in summary if s[1] > 1.10 and s[2] > 1.10]
    print("\nVERDICT: " + (
        "GATE PASSED on " + ", ".join(p[0].strip() for p in passed)
        if passed else
        "GATE FAILED -- with per-call dispatch removed from BOTH sides, no "
        "cuBLASLt candidate beats PyTorch's own FP16 F.linear by >10% at M=64. "
        "Run 82's apparent 1.26x/1.96x was the gap between two harness dispatch "
        "floors (PyTorch's Python loop ~7.8us/call vs the C++ loop ~3.9us/call), "
        "not kernel time. CLEAN NEGATIVE."), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
