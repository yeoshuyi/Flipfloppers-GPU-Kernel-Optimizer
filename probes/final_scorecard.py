"""
FINAL SCORECARD -- the shipped causal stack vs the FP32 baseline on all 13
runnable rows of CLAUDE.md's OFFICIAL CAUSAL EVALUATION MATRIX (row 14 OOMs
the baseline: no end-to-end path).

Per row, in ONE job:
  1. baseline wall  -- median per-iter latency, fixed input, torch.cuda.Event
                       (same method as benchmark.py's harness)
  2. shipped  wall  -- ditto, on the compiled/graphed UserOptimizedTransformer
  3. speedup
  4. stage breakdown -- CUPTI census of the real graphed forward, kernels
     bucketed SDPA / GEMM / GELU / ELEM / OTHER (us/fwd and % of wall) + count
  5. accuracy-legal roofline components, computed from first principles:
        gemm_flop      = 12*M*d^2*L                (QKV 6 + out_proj 2 + FFN 4, ffn=d)
        attn_flop      = 4*B*S^2*d*L  (full) ; 2*B*S^2*d*L  (causal-effective)
        compute_floor  = gemm_flop / 165.2e12  +  measured SDPA time
                         (165.2 TFLOP/s = FP16 storage / FP32 accumulate =
                          the fastest tensor tier that passes atol=0.002;
                          SDPA is already at its accuracy-legal precision)
        mem_traffic    = 36*M*d*L bytes  (irreducible boundary-crossing model;
                         see docs/PARETO_FRONTIER_ANALYSIS.md sec 4)
        mem_floor      = mem_traffic / measured_BW   [only binds when the
                         [M,d] fp32 working set exceeds L2]
        launch_floor   = kernels/fwd * 0.855 us     (measured graph body floor)
  6. verdict: which floor binds, and shipped / roofline ratio.

No build. benchmark.py is not modified by this probe.
"""
import json
import os
import statistics
import sys
import tempfile

import torch

sys.path.insert(0, "/work")
from benchmark import (  # noqa: E402
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    copy_model_weights, generate_random_case,
)

DEV = torch.device("cuda")
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LEGAL_TF = 165.2e12          # FP16 store / FP32 accum -- accuracy-legal peak
KBODY_US = 0.855            # measured CUDA-graph per-kernel body floor
L2_BYTES = 72 * 1024**2

# name, batch, seq, d_model, heads, ffn_dim, layers   (CLAUDE.md official matrix)
ROWS = [
    ("row01",  64,    128, 128,  4,  128, 4),
    ("row02",  1,     128, 128,  4,  128, 4),
    ("row03",  4,     128, 128,  4,  128, 4),
    ("row04",  16,    128, 128,  4,  128, 4),
    ("row05",  128,   128, 128,  4,  128, 4),
    ("row06",  10000, 128, 128,  4,  128, 4),
    ("row07",  64,    128, 32,   4,  32,  4),
    ("row08",  64,    128, 1024, 4,  1024, 4),
    ("row09",  64,    128, 128,  1,  128, 4),
    ("row10",  64,    128, 128,  2,  128, 4),
    ("row11",  64,    128, 128,  16, 128, 4),
    ("row12",  64,    32,  128,  4,  128, 4),
    ("row13",  64,    1024, 128, 4,  128, 4),
    # row14 (B32 d1024 h16 S100000 L2) -- baseline OOM, no e2e path
]


def build(b, s, d, h, f, L):
    cfg = TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                            ffn_dim=f, num_layers=L, causal=True)
    cfg.validate()
    base = BaselineTransformer(cfg).to(DEV, DTYPE).eval()
    opt = UserOptimizedTransformer(cfg).to(DEV, DTYPE).eval()
    copy_model_weights(base, opt, strict=True)
    x, mask = generate_random_case(cfg, DEV, DTYPE, seed=1234,
                                   padding_ratio=0.0, input_scale=1.0)
    return cfg, base, opt, x, mask


def wall_median_ms(model, x, mask, iters, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iters):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            model(x, mask)
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e))
    samples.sort()
    return statistics.median(samples), samples[0]


def census(model, x, mask, iters=15):
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        for _ in range(25):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.unlink(path)
    per = {}
    for e in ev:
        if (e.get("cat") or "").lower() not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        slot = per.setdefault(e.get("name", "?"), [0, 0.0])
        slot[0] += 1
        slot[1] += float(e.get("dur", 0.0))
    rows = sorted(([n, c / iters, u / iters] for n, (c, u) in per.items()),
                  key=lambda r: -r[2])
    return rows


def bucket(name):
    n = name.lower()
    if any(t in n for t in ("fmha", "flash", "attention", "mha", "fused_attention",
                            "scaled_dot")):
        return "SDPA"
    if "gelu" in n or ("erf" in n and "gemm" not in n):
        return "GELU"
    if any(t in n for t in ("gemm", "cutlass", "wmma", "s16816", "s1688",
                            "splitk", "implicit_gemm", "ampere_", "sm80_xmma",
                            "sm89", "sgemm", "dot_kernel", "addmm")):
        return "GEMM"
    if any(t in n for t in ("layer_norm", "native_layer_norm", "elementwise",
                            "vectorized", "triton_poi", "triton_per", "_add",
                            "copy", "memcpy", "memset", "cat_", "fill")):
        return "ELEM"
    return "OTHER"


def measure_bw():
    """One-off: streaming copy bandwidth on this GPU (GB/s)."""
    n = 1 << 24  # 16M fp32 = 64 MB each way
    src = torch.randn(n, device=DEV, dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(50):
        dst.copy_(src)
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / 50 * 1e-3
    return (n * 4 * 2) / t / 1e9


def main():
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    bw = measure_bw()
    print(f"measured streaming BW = {bw:.0f} GB/s   |   legal tensor peak = "
          f"{LEGAL_TF/1e12:.1f} TFLOP/s (FP16 store / FP32 accum)\n")

    hdr = (f"{'row':<6}{'B':>7}{'S':>6}{'d':>6}{'H':>4}{'tok':>10}"
           f"{'base_ms':>11}{'ship_ms':>10}{'speedup':>9}{'kern':>6}"
           f"{'SDPA':>9}{'GEMM':>9}{'GELU':>8}{'ELEM':>9}{'OTHER':>8}"
           f"{'gemm_GF':>9}{'attnGF':>8}{'cmp_flr':>9}{'mem_GB':>8}"
           f"{'mem_flr':>9}{'lnch_flr':>9}{'roofline':>10}{'ratio':>7}  bound")
    print(hdr)
    print("-" * len(hdr))

    summary = []
    for name, b, s, d, h, f, L in ROWS:
        cfg, base, opt, x, mask = build(b, s, d, h, f, L)
        M = b * s

        # baseline: very stable, fewer iters; guard OOM
        try:
            bmed, bmin = wall_median_ms(base, x, mask,
                                        iters=(20 if M >= 500_000 else 60),
                                        warmup=10)
        except RuntimeError as exc:
            bmed = bmin = float("nan")
            print(f"{name}: baseline OOM/err {str(exc)[:60]}")
        omed, omin = wall_median_ms(opt, x, mask,
                                    iters=(60 if M >= 500_000 else 200),
                                    warmup=(20 if M >= 500_000 else 50))
        sp = bmed / omed if bmed == bmed else float("nan")

        rows = census(opt, x, mask)
        kern = sum(r[1] for r in rows)
        buks = {}
        for n_, c_, u_ in rows:
            buks[bucket(n_)] = buks.get(bucket(n_), 0.0) + u_ / 1000.0  # -> ms
        sdpa = buks.get("SDPA", 0.0)
        gemm = buks.get("GEMM", 0.0)
        gelu = buks.get("GELU", 0.0)
        elem = buks.get("ELEM", 0.0)
        other = buks.get("OTHER", 0.0)

        gemm_flop = 12.0 * M * d * d * L
        attn_flop_full = 4.0 * b * s * s * d * L
        attn_flop_causal = 2.0 * b * s * s * d * L
        compute_floor = gemm_flop / LEGAL_TF * 1e3 + sdpa            # ms
        mem_traffic = 36.0 * M * d * L                              # bytes
        mem_floor = mem_traffic / (bw * 1e9) * 1e3                  # ms
        ws_bytes = M * d * 4
        dram_bound = ws_bytes > 0.7 * L2_BYTES
        launch_floor = kern * KBODY_US / 1000.0                    # ms

        # pick the binding floor + label
        cands = [("compute", compute_floor), ("launch", launch_floor)]
        if dram_bound:
            cands.append(("membw", max(mem_floor, compute_floor)))
        roofline = max(v for _, v in cands)
        if dram_bound and mem_floor >= compute_floor and mem_floor >= launch_floor:
            bound = "memory-BW"
        elif sdpa >= 0.45 * (sdpa + gemm + gelu + elem + other) and sdpa > 0:
            bound = "attn O(S^2)"
        elif launch_floor >= compute_floor:
            bound = "launch/latency"
        else:
            bound = "compute 165TF"
        ratio = omed / roofline if roofline > 0 else float("nan")

        print(f"{name:<6}{b:>7}{s:>6}{d:>6}{h:>4}{M:>10}"
              f"{bmed:>11.4f}{omed:>10.4f}{sp:>8.2f}x{kern:>6.0f}"
              f"{sdpa:>9.4f}{gemm:>9.4f}{gelu:>8.4f}{elem:>9.4f}{other:>8.4f}"
              f"{gemm_flop/1e9:>9.2f}{attn_flop_causal/1e9:>8.2f}"
              f"{compute_floor:>9.4f}{mem_traffic/1e9:>8.2f}"
              f"{mem_floor:>9.4f}{launch_floor:>9.4f}{roofline:>10.4f}"
              f"{ratio:>6.2f}x  {bound}")

        # top kernels for this row (context)
        for n_, c_, u_ in rows[:6]:
            print(f"        {u_:9.2f}us x{c_:5.2f}  [{bucket(n_):5s}] {n_[:78]}")

        summary.append((name, bmed, omed, sp, roofline, ratio, bound))
        del base, opt, x, mask
        torch.cuda.empty_cache()

    print("\n==================== FINAL SCORECARD ====================")
    print(f"{'row':<7}{'baseline_ms':>13}{'shipped_ms':>13}{'speedup':>10}"
          f"{'roofline_ms':>13}{'ship/roof':>11}  bounded_by")
    tb = to = 0.0
    for name, bmed, omed, sp, roof, ratio, bound in summary:
        print(f"{name:<7}{bmed:>13.4f}{omed:>13.4f}{sp:>9.2f}x"
              f"{roof:>13.4f}{ratio:>10.2f}x  {bound}")
        if bmed == bmed:
            tb += bmed
            to += omed
    print("-" * 68)
    print(f"{'TOTAL':<7}{tb:>13.4f}{to:>13.4f}{tb/to:>9.2f}x   "
          f"(sum over the 13 runnable rows; geomean speedup also reported below)")
    gm = 1.0
    k = 0
    for _, bmed, omed, sp, *_ in summary:
        if sp == sp:
            gm *= sp
            k += 1
    print(f"geomean speedup over {k} rows = {gm ** (1.0 / k):.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
