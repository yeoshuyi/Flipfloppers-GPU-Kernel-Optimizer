"""
G4.0 Phase-2 -- the CEILING, measured on hardware instead of modelled.

MEGAKERNEL.md's gate is ">15% launch overhead or GPU idle at the tiny regime
after CUDA Graphs".  Step 20 answered the GPU-idle half decisively (-0.7%, and
this session's census reproduces it at -0.5%: summed kernel time is 100.5% of
wall).  What that measurement does NOT capture, and what this probe measures,
is the GPU-side per-kernel-boundary cost that lives INSIDE each kernel's own
duration: SM drain/refill, pipeline restart, and the fixed body time even a
no-op kernel pays.  A genuinely fused kernel could win via that mechanism with
zero measured CPU-side idle.

Two numbers bound it from above and below, both measured here:

  UPPER BOUND -- "perfect free fusion".  Replay ONLY the 42 GEMM/attention
  kernels the tiny forward issues (same shapes, same dtypes, same six distinct
  weight sets so the ~63 MB DRAM/L2 working set is reproduced), in one CUDA
  graph.  The gap to the real forward's wall time is everything the 20
  elementwise launches cost -- their fixed launch cost AND the real LayerNorm /
  GELU / residual work they do.  A megakernel still has to do that work, in
  registers instead of through HBM, so this is a strict over-estimate of what
  G4 could ever recover.  It is the number that cannot be argued past: if even
  THIS is under 15%, the gate cannot be met by any fusion.

  LAUNCH-COST COMPONENT -- the honest reading of "launch overhead".  Re-measure
  the marginal cost of one extra kernel inside a graph replay on this GPU
  (step 20 measured 0.855 us; re-measured here rather than cited), multiply by
  the number of elementwise launches that fusion would delete.  This separates
  "the kernel exists" from "the kernel does work".

Reports both against the 15% gate.
"""

import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")

from benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
    _lt_ext,
)

DEV = torch.device("cuda")
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

B, S = 1, 64
TOK = B * S
D, H, HD, FFN, L = 512, 8, 64, 2048, 6
WS = 32 * 1024 * 1024
REQ = 16


# ---------------------------------------------------------------- helpers --
def graph_replay_us(build_fn, replays=300, best_of=7):
    """Capture build_fn()'s work in one CUDA graph, return best mean us/replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            build_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_fn()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(best_of):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / replays * 1000.0)
    return best, g


def trace_graph(g, replays=30):
    """Kernel count + summed duration for one replay of graph g."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    os.unlink(path)
    n = 0
    us = 0.0
    for ev in events:
        if (ev.get("cat") or "").lower() != "kernel":
            continue
        n += 1
        us += float(ev.get("dur", 0.0))
    return n / replays, us / replays


def noop_marginal_us():
    """Marginal cost of one more kernel inside a graph replay, re-measured."""
    t = torch.ones(1, device=DEV)
    pts = []
    for k in (16, 64, 128, 256):
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
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(300):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        pts.append((k, e0.elapsed_time(e1) / 300 * 1000.0))
    n = len(pts)
    mx = sum(k for k, _ in pts) / n
    my = sum(v for _, v in pts) / n
    num = sum((k - mx) * (v - my) for k, v in pts)
    den = sum((k - mx) ** 2 for k, _ in pts)
    return pts, num / den


# ---------------------------------------------------- the real model, tiny --
def build_model():
    cfg = TransformerConfig(batch_size=B, seq_len=S, d_model=D, num_heads=H,
                            ffn_dim=FFN, num_layers=L, causal=False)
    base = BaselineTransformer(cfg).to(DEV, DTYPE).eval()
    opt = UserOptimizedTransformer(cfg).to(DEV, DTYPE).eval()
    copy_model_weights(base, opt, strict=True)
    x, mask = generate_random_case(cfg, DEV, DTYPE, seed=1234,
                                   padding_ratio=0.0, input_scale=1.0)
    return base, opt, x, mask


def wall_ms(model, x, mask, iters=400, warmup=100):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            model(x, mask)
        e1.record()
        torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def census_model(model, x, mask, iters=20):
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        for _ in range(30):
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
    g_n = g_us = o_n = o_us = 0
    for ev in events:
        if (ev.get("cat") or "").lower() not in ("kernel", "gpu_memcpy",
                                                 "gpu_memset"):
            continue
        nm = ev.get("name", "").lower()
        dur = float(ev.get("dur", 0.0))
        if any(t in nm for t in ("gemm", "fmha", "flash", "cutlass", "splitk")):
            g_n += 1
            g_us += dur
        else:
            o_n += 1
            o_us += dur
    return g_n / iters, g_us / iters, o_n / iters, o_us / iters


# --------------------------- GEMM/attention-only reproduction of the forward -
def build_gemm_only(opt):
    """Exactly the 42 GEMM/attention launches the tiny forward issues, with the
    real six weight sets (so the DRAM/L2 working set is the real one) and no
    elementwise work at all."""
    ext = _lt_ext()
    if ext is None:
        raise RuntimeError("cuBLASLt extension unavailable")

    # cuBLASLt problems for the two FFN GEMMs, best algorithm each -- the same
    # selection the shipped G6.6 path makes.
    a1 = torch.randn(TOK, D, device=DEV, dtype=torch.float32)
    a2 = torch.randn(TOK, FFN, device=DEV, dtype=torch.float32)
    o1 = torch.empty(TOK, FFN, device=DEV, dtype=torch.float32)
    o2 = torch.empty(TOK, D, device=DEV, dtype=torch.float32)
    l0 = opt.layers[0]
    plan = []
    for K_, N_, inp_, w_, b_, out_ in (
            (D, FFN, a1, l0._ffn_in_weight, l0._ffn_in_bias, o1),
            (FFN, D, a2, l0.ffn_out.weight, l0.ffn_out.bias, o2)):
        pid = ext.create_problem(TOK, N_, K_, True, WS, REQ)
        best = None
        for i in range(ext.num_algos(pid)):
            try:
                t = ext.time_algo(pid, i, inp_, w_, b_, out_, 5, 30)
            except Exception:                                    # noqa: BLE001
                continue
            if best is None or t < best[1]:
                best = (i, t)
        plan.append((pid, best[0]))
    print(f"  ffn_in  algo idx={plan[0][1]}  {ext.algo_info(*plan[0])}")
    print(f"  ffn_out algo idx={plan[1][1]}  {ext.algo_info(*plan[1])}")

    # Per-layer buffers. Distinct weights per layer, matching the real model.
    n1 = torch.randn(TOK, D, device=DEV, dtype=torch.float16)
    ctx = torch.randn(TOK, D, device=DEV, dtype=torch.float16)
    q = torch.randn(B, H, S, HD, device=DEV, dtype=torch.float16)
    k = torch.randn(B, H, S, HD, device=DEV, dtype=torch.float16)
    v = torch.randn(B, H, S, HD, device=DEV, dtype=torch.float16)
    n2 = torch.randn(TOK, D, device=DEV, dtype=torch.float32)
    hid = torch.randn(TOK, FFN, device=DEV, dtype=torch.float32)

    def run():
        for layer in opt.layers:
            at = layer.attention
            F.linear(n1, at._qkv_weight_fp16, at._qkv_bias_fp16)
            F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                           is_causal=False, scale=1.0)
            F.linear(ctx, at._out_proj_weight_fp16, at._out_proj_bias_fp16)
            ext.lt_linear(plan[0][0], plan[0][1], n2, layer._ffn_in_weight,
                          layer._ffn_in_bias)
            ext.lt_linear(plan[1][0], plan[1][1], hid, layer.ffn_out.weight,
                          layer.ffn_out.bias)
    return run


def main() -> int:
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    print()

    print("=== per-kernel marginal cost in a CUDA graph (re-measured) ===")
    pts, slope = noop_marginal_us()
    for k, us in pts:
        print(f"  K={k:4d} no-op kernels: {us:8.2f} us ({us / k:6.3f} us/kernel)")
    print(f"  least-squares marginal cost: {slope:.4f} us per extra kernel")
    print()

    base, opt, x, mask = build_model()
    w_opt = wall_ms(opt, x, mask)
    w_base = wall_ms(base, x, mask)
    g_n, g_us, o_n, o_us = census_model(opt, x, mask)
    print(f"=== shipped tiny forward (B={B} S={S}) ===")
    print(f"  baseline wall     : {w_base * 1000:9.2f} us")
    print(f"  optimized wall    : {w_opt * 1000:9.2f} us  ({w_base / w_opt:.3f}x)")
    print(f"  GEMM/attn kernels : {g_n:5.1f} launches {g_us:9.2f} us")
    print(f"  elementwise       : {o_n:5.1f} launches {o_us:9.2f} us "
          f"({o_us / (w_opt * 1000) * 100:.2f}% of wall)")
    print(f"  summed / wall     : "
          f"{(g_us + o_us) / (w_opt * 1000) * 100:.2f}%   "
          f"(GPU idle = {100 - (g_us + o_us) / (w_opt * 1000) * 100:+.2f}%)")
    print()

    print("=== GEMM/attention-only graph (perfect free fusion, upper bound) ===")
    run = build_gemm_only(opt)
    us_go, g = graph_replay_us(run)
    n_go, sum_go = trace_graph(g)
    print(f"  kernels in graph  : {n_go:5.1f}  (target {g_n:.0f})")
    print(f"  summed kernel time: {sum_go:9.2f} us (target {g_us:.2f} us, "
          f"{sum_go / max(g_us, 1e-9) * 100:.1f}% of it)")
    print(f"  graph replay wall : {us_go:9.2f} us")
    print()

    ceiling_us = w_opt * 1000 - us_go
    print("=== THE GATE ===")
    print(f"  A. GPU idle after CUDA graphs        : "
          f"{100 - (g_us + o_us) / (w_opt * 1000) * 100:+.2f}%   (gate: >15%)")
    print(f"  B. perfect-free-fusion ceiling       : {ceiling_us:7.2f} us / "
          f"{w_opt * 1000:.2f} us = {ceiling_us / (w_opt * 1000) * 100:.2f}%"
          f"   (gate: >15%)")
    print(f"     -- strict OVER-estimate: counts the LayerNorm/GELU/residual")
    print(f"        arithmetic itself as recoverable, which it is not.")
    print(f"  C. pure launch-cost component        : "
          f"{o_n:.0f} launches x {slope:.3f} us = {o_n * slope:7.2f} us = "
          f"{o_n * slope / (w_opt * 1000) * 100:.2f}%   (gate: >15%)")
    print(f"     -- the honest reading of 'launch overhead': what remains if")
    print(f"        the work is done anyway, just inside another kernel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
