#!/usr/bin/env python3
"""
G4.3 Stage 0c -- does the warp-spec win survive at the shapes the HARNESS
actually runs, as opposed to the two step-37 chose?

Steps 37 and this investigation's rounds 1/2 both judged on d_model=512
(K=512, N=1536), which is benchmark.py's DEFAULT config. But CLAUDE.md's
official causal evaluation matrix runs QKV dim 128 for 11 of its 14 shapes,
32 for one, and 1024 for two. K=128 means only TWO k-tiles at BK=64: the
cp.async pipeline barely fills, and a kernel that wins at K=512 can easily
lose at K=128. That has to be measured, not assumed, before anything is wired
into benchmark.py.

Also measured here: the FFN GEMMs (ffn_in / ffn_out). They are 8/3 of the
attention GEMMs' FLOPs combined, so they set the size of the prize IF an
epilogue (GELU / residual) were added later. Integration is NOT proposed for
them in this step -- this is a sizing measurement only.

Same protocol as every other measurement in this project: CUDA-graph replay,
best of 5, pytorch F.linear (cuBLASLt) as the floor.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")

# The finalists from rounds 1 and 2 (results/g4_3_warpspec_sweep_r2_run126.log)
CFGS = [8, 14, 26, 29, 33, 37, 39, 45]

# (label, M, K, N)
# --- benchmark.py DEFAULT config: d_model=512, ffn 2048, 6 layers ----------
CASES = [
    ("d512 tiny      qkv     ",   128,  512, 1536),
    ("d512 tiny      out_proj",   128,  512,  512),
    ("d512 default   qkv     ",  1024,  512, 1536),
    ("d512 default   out_proj",  1024,  512,  512),
    ("d512 default   ffn_in  ",  1024,  512, 2048),
    ("d512 default   ffn_out ",  1024, 2048,  512),
    ("d512 long_seq  qkv     ",  8192,  512, 1536),
    ("d512 long_seq  out_proj",  8192,  512,  512),
    ("d512 long_seq  ffn_in  ",  8192,  512, 2048),
    ("d512 long_seq  ffn_out ",  8192, 2048,  512),
    ("d512 largebat  qkv     ", 32768,  512, 1536),
    ("d512 largebat  out_proj", 32768,  512,  512),
    ("d512 largebat  ffn_in  ", 32768,  512, 2048),
    ("d512 largebat  ffn_out ", 32768, 2048,  512),
    # --- OFFICIAL MATRIX, d_model=128 (shapes 1-6, 9-13) ------------------
    ("d128 #2  B1     qkv     ",   128,  128,  384),
    ("d128 #3  B4     qkv     ",   512,  128,  384),
    ("d128 #4  B16    qkv     ",  2048,  128,  384),
    ("d128 #1  B64    qkv     ",  8192,  128,  384),
    ("d128 #1  B64    out_proj",  8192,  128,  128),
    ("d128 #1  B64    ffn_in  ",  8192,  128,  128),
    ("d128 #5  B128   qkv     ", 16384,  128,  384),
    ("d128 #13 S1024  qkv     ", 65536,  128,  384),
    ("d128 #13 S1024  out_proj", 65536,  128,  128),
    ("d128 #6  B10000 qkv     ", 1280000, 128, 384),
    ("d128 #6  B10000 out_proj", 1280000, 128, 128),
    # --- OFFICIAL MATRIX, d_model=1024 (shape 8) --------------------------
    ("d1024 #8 B64    qkv     ",  8192, 1024, 3072),
    ("d1024 #8 B64    out_proj",  8192, 1024, 1024),
    ("d1024 #8 B64    ffn_in  ",  8192, 1024, 1024),
]

PEAK_F16ACC = 330.3


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name="g4_3_warpspec", sources=[WS_CPP, WS_CU],
                build_directory=bd, with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-diag-suppress", "179"], verbose=False)


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


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    ws = build()
    print("\nconfigs under test:")
    for c in CFGS:
        print(f"  [{c:2d}] {ws.cfg_name(c)}")

    g = torch.Generator(device=dev)
    rows = []
    print("\n" + "=" * 118)
    hdr = f"{'shape':26s} {'M':>8s} {'K':>5s} {'N':>5s} {'torch us':>10s} " \
          f"{'best us':>9s} {'cfg':>4s} {'speedup':>8s} {'TF':>7s} {'%tier':>6s}"
    print(hdr)
    print("-" * 118)

    for (label, M, K, N) in CASES:
        g.manual_seed((M * 31 + K * 7 + N) & 0x7FFFFFFF)
        try:
            inp = torch.randn(M, K, device=dev, dtype=torch.float16,
                              generator=g)
            w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                             generator=g) * 0.02).half()
            b = (torch.randn(N, device=dev, dtype=torch.float16,
                             generator=g) * 0.02).half()
            out = torch.empty(M, N, device=dev, dtype=torch.float16)
            ref_out = torch.empty(M, N, device=dev, dtype=torch.float16)
        except RuntimeError as e:
            print(f"{label:26s} {M:8d} {K:5d} {N:5d}   OOM: {str(e)[:40]}")
            continue

        flop = 2.0 * M * K * N
        if M * N >= 1 << 26:
            iters, replays = 3, 12
        elif M >= 8192:
            iters, replays = 10, 40
        else:
            iters, replays = 50, 200

        t_ref = graph_time(lambda: torch.addmm(b, inp, w.t(), out=ref_out),
                           iters, replays)
        best_t, best_c = None, None
        per_cfg = {}
        for c in CFGS:
            try:
                ws.ws_gemm(c, inp, w, b, out)
                torch.cuda.synchronize()
            except Exception:
                continue
            t = graph_time(lambda c=c: ws.ws_gemm(c, inp, w, b, out),
                           iters, replays)
            per_cfg[c] = t
            if best_t is None or t < best_t:
                best_t, best_c = t, c
        if best_t is None:
            print(f"{label:26s} {M:8d} {K:5d} {N:5d} {t_ref:10.3f} "
                  f"{'--':>9s} {'--':>4s}  NO CONFIG FITS (tile divisibility)")
            rows.append((label, M, K, N, t_ref, None, None))
            del inp, w, b, out, ref_out
            torch.cuda.empty_cache()
            continue
        tf = flop / (best_t * 1e-6) / 1e12
        print(f"{label:26s} {M:8d} {K:5d} {N:5d} {t_ref:10.3f} {best_t:9.3f} "
              f"{best_c:4d} {'x%.3f' % (t_ref/best_t):>8s} {tf:7.2f} "
              f"{tf/PEAK_F16ACC*100:5.1f}%"
              f"   [{'  '.join('%d:%.2f' % (c, per_cfg[c]) for c in CFGS if c in per_cfg)}]")
        rows.append((label, M, K, N, t_ref, best_t, best_c))
        del inp, w, b, out, ref_out
        torch.cuda.empty_cache()

    print("\n" + "=" * 118)
    print("GENERALIST CHECK -- one config that must be picked for ALL shapes "
          "if this is to be a static gate")
    # Which single cfg is best on average across the shapes it can run?
    print("(see the per-cfg times in brackets above)")

    print("\nWHOLE-MODEL DILUTION (attention GEMMs only: qkv + out_proj, "
          "x6 layers at d512 / x4 at d128):")
    for tag, keys, layers, fwd_us in [
        ("d512 default   ", ("d512 default   qkv     ",
                             "d512 default   out_proj"), 6, 654.3),
        ("d512 long_seq  ", ("d512 long_seq  qkv     ",
                             "d512 long_seq  out_proj"), 6, 5418.1),
        ("d512 largebat  ", ("d512 largebat  qkv     ",
                             "d512 largebat  out_proj"), 6, 21263.4),
    ]:
        tot_ref = tot_new = 0.0
        okall = True
        for k in keys:
            r = [x for x in rows if x[0] == k]
            if not r or r[0][5] is None:
                okall = False
                break
            tot_ref += r[0][4]
            tot_new += r[0][5]
        if not okall:
            continue
        tot_ref *= layers
        tot_new *= layers
        save = tot_ref - tot_new
        print(f"  {tag} GEMMs {tot_ref:9.1f} us -> {tot_new:9.1f} us "
              f"(save {save:7.1f} us) of a {fwd_us:9.1f} us optimized forward "
              f"(run90) = {tot_ref/fwd_us*100:5.2f}% of forward, "
              f"saving {save/fwd_us*100:5.2f}%  -> model speedup x"
              f"{1.0/(1.0 - save/fwd_us):.4f}")
    print("\nNOTE: run90's forward times are a STALE reference (the elites "
          "have moved since). They are used only to size the fraction, and "
          "the real number must come from benchmark.py itself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
