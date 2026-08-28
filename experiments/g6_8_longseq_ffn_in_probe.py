#!/usr/bin/env python3
"""
G6.8 -- does step 33's cuBLASLt algorithm win reproduce at LONG-SEQ (M=8192)?

New evidence being chased: run 71 (results/g6_6_cublaslt_algo_probe_run71.log,
"M=8192" block) reports for ffn_in K=512 N=2048 bias=True:

    pytorch F.linear :  274.08 us
    BEST -> algo[3]     245.22 us   x1.1177  (WIN)

Step 33 diagnosed a structurally identical M=1024 "win" as an artifact of the
probe's own baseline (eager ops in a raw CUDA graph still pay PyTorch's addmm
bias penalty; torch.compile's real full-model lowering does not).  Whether the
same is true at M=8192 has never been checked.  Two independent tells are
already visible in run 71 itself and must be confirmed or refuted here:

  * the cuBLASLt candidate times are the SAME with and without bias
    (244-253 us vs 245-254 us) -- the whole 1.1177x lives in the reference,
    which moves 251.55 -> 274.08 us when a bias is added.  That is the bias
    path, not algorithm selection.  The pure-algorithm win at this shape is
    1.0289x (bias=False), i.e. noise by the >10% gate.
  * run 73 (results/g6_6_bias_path_probe_run73.log, "FFN block, M=8192")
    already measured the full FFN block under CUDA-graph capture:
    cuBLASLt best-algo + bias epilogue = 0.9983x of shipped F.linear.

PARTS
  A  end-to-end, the only number that decides anything: the REAL shipped model
     at B=8 S=1024 under its own torch.compile(reduce-overhead), timed against
     the identical model with the G6.6 gate raised so the cuBLASLt path is
     actually taken.  Interleaved A/B/A/B to cancel drift.  Bit-identity of the
     two outputs is checked -- step 34's cheap tell.
  B  torch.profiler kernel census of the shipped compiled model at this shape:
     is there an addmm bias penalty (a distinct/slower GEMM, or a separate
     bias-add kernel) for ffn_in at all?
  C  isolated M=8192 ffn_in A/B under CUDA-graph replay AND profiler kernel
     time (step 34's fair-measurement protocol), with kernel names + maxdiff.
"""
import os
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark as B  # noqa: E402

D, FF = 512, 2048
BATCH, SEQ = 8, 1024
M = BATCH * SEQ
WARMUP = 30
ITERS = 200


def build_pair(device, dtype):
    cfg = B.TransformerConfig(batch_size=BATCH, seq_len=SEQ, d_model=D,
                              num_heads=8, ffn_dim=FF, num_layers=6,
                              causal=False)
    torch.manual_seed(1234)
    base = B.BaselineTransformer(cfg).to(device=device, dtype=dtype).eval()
    opt = B.UserOptimizedTransformer(cfg).to(device=device, dtype=dtype).eval()
    B.copy_model_weights(base, opt, strict=True)
    return cfg, base, opt


def time_model(model, x, mask, iters=ITERS):
    B.warmup_model(model, x, mask, WARMUP, x.device)
    s = B.benchmark_once(model, x, mask, iters, x.device)
    return statistics.median(s)


def part_a(device, dtype):
    print("\n########## PART A: end-to-end real compiled model, B=8 S=1024 "
          "##########", flush=True)
    cfg, base, _ = build_pair(device, dtype)
    x, mask = B.generate_random_case(cfg, device, dtype, seed=4242,
                                     padding_ratio=0.0, input_scale=1.0)

    def fresh(max_tokens):
        B._LT_MAX_TOKENS = max_tokens
        _, _, opt = build_pair(device, dtype)
        return opt

    shipped_ms, cand_ms = [], []
    out_s = out_c = None
    for rnd in range(3):
        m_s = fresh(127)
        t = time_model(m_s, x, mask)
        shipped_ms.append(t)
        with torch.inference_mode():
            out_s = m_s(x, mask).clone().float()
        plan_s = m_s._lt_cur
        del m_s
        torch.cuda.empty_cache()

        m_c = fresh(10 ** 9)
        t = time_model(m_c, x, mask)
        cand_ms.append(t)
        with torch.inference_mode():
            out_c = m_c(x, mask).clone().float()
        plan_c = m_c._lt_cur
        del m_c
        torch.cuda.empty_cache()

        print(f"  round {rnd}: shipped {shipped_ms[-1]*1000:8.1f} us | "
              f"lt-gate-raised {cand_ms[-1]*1000:8.1f} us | "
              f"plan shipped={plan_s} cand={plan_c}", flush=True)

    B._LT_MAX_TOKENS = 127
    s = min(shipped_ms)
    c = min(cand_ms)
    md = (out_c - out_s).abs().max().item()
    print(f"\n  best-of-3 shipped        : {s*1000:9.1f} us", flush=True)
    print(f"  best-of-3 lt-gate-raised : {c*1000:9.1f} us", flush=True)
    print(f"  SPEEDUP (shipped/cand)   : {s/c:7.4f}x", flush=True)
    print(f"  maxdiff between outputs  : {md:.3e}  "
          f"({'BIT-IDENTICAL' if md == 0.0 else 'differs'})", flush=True)
    print(f"  |out|max = {out_s.abs().max().item():.4f}", flush=True)


def part_b(device, dtype):
    print("\n########## PART B: profiler kernel census, shipped model "
          "##########", flush=True)
    B._LT_MAX_TOKENS = 127
    cfg, _, opt = build_pair(device, dtype)
    x, mask = B.generate_random_case(cfg, device, dtype, seed=4242,
                                     padding_ratio=0.0, input_scale=1.0)
    B.warmup_model(opt, x, mask, WARMUP, device)
    n = 20
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                opt(x, mask)
            torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        if e.self_device_time_total <= 0 or e.device_time_total <= 0:
            continue
        if e.count == 0:
            continue
        rows.append((e.self_device_time_total / n, e.count / n, e.key))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    print(f"  total device time/forward = {tot:8.1f} us   "
          f"(over {n} forwards)", flush=True)
    for us, cnt, key in rows[:14]:
        print(f"   {us:9.2f} us  x{cnt:6.2f}/fwd  {us/max(cnt,1e-9):8.2f} "
              f"us/launch  {key[:78]}", flush=True)


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
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(ITERS):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / ITERS * 1000.0, res


def prof_kernels(fn, n=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    out = []
    for e in prof.key_averages():
        if e.self_device_time_total > 0 and "aten::" not in e.key:
            out.append((e.self_device_time_total / n, e.key))
    out.sort(reverse=True)
    return sum(o[0] for o in out), out[:3]


def part_c(device, dtype):
    print("\n########## PART C: isolated ffn_in M=8192 K=512 N=2048, fair "
          "##########", flush=True)
    ext = B._lt_ext()
    if ext is None:
        print("  extension unavailable", flush=True)
        return
    g = torch.Generator(device=device); g.manual_seed(4242)
    x = torch.randn(M, D, device=device, generator=g)
    w = torch.randn(FF, D, device=device, generator=g) * 0.02
    b = torch.randn(FF, device=device, generator=g) * 0.02
    out = torch.empty(M, FF, device=device)

    pid = ext.create_problem(M, FF, D, True, B._LT_WS_BYTES, B._LT_REQUESTED)
    n = ext.num_algos(pid)
    best = None
    for i in range(n):
        try:
            t = ext.time_algo(pid, i, x, w, b, out, 10, 100)
        except Exception:                                     # noqa: BLE001
            continue
        if best is None or t < best[1]:
            best = (i, t)
    ref_eager = B.UserOptimizedTransformer._time_eager(
        lambda: F.linear(x, w, b), 10, 100)
    ref_nobias = B.UserOptimizedTransformer._time_eager(
        lambda: F.linear(x, w), 10, 100)
    print(f"  run71-style eager loop: F.linear(bias)   {ref_eager*1000:8.2f} us",
          flush=True)
    print(f"                          F.linear(nobias) {ref_nobias*1000:8.2f} us",
          flush=True)
    print(f"                          best lt algo[{best[0]}]  {best[1]*1000:8.2f} us"
          f"   x{ref_eager/best[1]:.4f} vs bias / x{ref_nobias/best[1]:.4f} vs nobias",
          flush=True)

    def A():
        return F.linear(x, w, b)

    def C():
        ext.run(pid, best[0], x, w, b, out)
        return out

    tA, rA = graph_time(A)
    tC, rC = graph_time(C)
    md = (rC.float() - rA.float()).abs().max().item()
    pA, kA = prof_kernels(A)
    pC, kC = prof_kernels(C)
    print(f"\n  CUDA-graph replay : F.linear {tA:8.2f} us | lt {tC:8.2f} us"
          f"   x{tA/tC:.4f}", flush=True)
    print(f"  profiler kernel   : F.linear {pA:8.2f} us | lt {pC:8.2f} us"
          f"   x{pA/pC:.4f}", flush=True)
    print(f"  maxdiff {md:.3e}", flush=True)
    print("  F.linear kernels:", flush=True)
    for us, k in kA:
        print(f"     {us:8.2f} us  {k[:80]}", flush=True)
    print("  cuBLASLt kernels:", flush=True)
    for us, k in kC:
        print(f"     {us:8.2f} us  {k[:80]}", flush=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    dt = torch.float32
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
          flush=True)
    print(f"ext available: {B._lt_ext() is not None}", flush=True)
    for fn in (part_c, part_b, part_a):
        try:
            fn(dev, dt)
        except Exception as exc:                              # noqa: BLE001
            import traceback
            print(f"  {fn.__name__} FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
