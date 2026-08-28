#!/usr/bin/env python3
"""
G6.9 -- does offline cuBLASLt algorithm selection beat the default heuristic
for any GEMM signature produced by the 14 official causal shapes?

Phase 1 inventory (analytic, see the header table below) + Phase 2 isolated
search.  The causal forward (G4.7c inert on every official shape: max
ffn_dim=1024 < 2048) issues, per layer:
    qkv      F.linear(n1_fp16 [M,d],  W [3d,d], b)  -> (M, N=3d, K=d)  fp16 BIAS
    out_proj F.linear(ctx  [M,d],     W [d,d],  b)  -> (M, N=d,  K=d)  fp16 BIAS
    ffn_in   F.linear(n2_fp16 [M,d],  W [ffn,d],b)  -> (M, N=ffn,K=d)  fp16 BIAS
    ffn_out  F.linear(act_fp32 [M,ffn],W [d,ffn],b) -> (M, N=d,  K=ffn) TF32 BIAS
d == ffn for all 14 shapes, so out_proj == ffn_in as a signature, and ffn_out
shares (M,N,K) with them but at TF32.  M = batch*seq_len.

Unique (M, d) over shapes 1-13 (14 OOMs the baseline -> no end-to-end path):
    d=128 : M in {128, 512, 2048, 8192, 16384, 65536, 1280000}
    d=32  : M in {8192}
    d=1024: M in {8192}
=> 9 (M,d) pairs x 3 signature types (qkv / proj / ffn_out) = 27 signatures.

Prior coverage (no EXACT match, but patterns):
    G6.6  small-M FFN at d=512/K=512 -> split-K win exists (bias-path era).
    G6.7  fp16 attention GEMMs, small M, d=512 -> default heuristic optimal (neg).
    G6.8  ffn_in fp16, M=8192, d=512  -> 0.36% whole-model (neg).
The large-M signatures (M>=8192) inherit G6.8's negative pattern; the only
signatures with a prior *reason* to expect a win are the small-M ones
(M in {128,512,2048}, shapes 2/3/4/12) -- but at K=128, not K=512, so the
split-K lever is weak.  This sweep checks ALL of them anyway.

Comparison is artefact-free: best heuristic candidate (idx k) vs idx 0 (the
heuristic's own top pick), BOTH through the same cublasLtMatmul run() path.
PRECISION DISCIPLINE: benchmark.py sets allow_fp16_reduced_precision_reduction
= False, so fp16 GEMMs are searched with reduction_mask=2 (fp32 partials only);
mask=7 is informational -- a win only at mask=7 is inadmissible.

RETAIN a signature only if the best candidate beats idx 0 by > 2%,
reproducibly across all REPEATS, at mask=2 (fp16) / any (tf32).
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F16_SRC = os.path.join(ROOT, "csrc", "cublaslt_algo_fp16.cpp")
F32_SRC = os.path.join(ROOT, "csrc", "cublaslt_algo.cpp")

REQUESTED = 32
MAX_WS = 32 * 1024 * 1024
WARMUP = 30
REPEATS = 3
THRESH = 0.02

# (M, d)  -- d == ffn for all official shapes
MD = [
    (128, 128), (512, 128), (2048, 128), (8192, 128),
    (16384, 128), (65536, 128), (1280000, 128),
    (8192, 32), (8192, 1024),
]


def build(name, src):
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name=name, sources=[src], build_directory=bd, with_cuda=True,
                extra_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)


def iters_for(M, N, K):
    flop = 2.0 * M * N * K
    if flop > 5e10:
        return 20
    if flop > 5e9:
        return 60
    return 250


def search(ext, M, N, K, dtype, masks, dev):
    """Return list of (mask, best_idx, best_ms, idx0_ms, delta_frac, info) rows,
    one per mask, each the min over REPEATS of the best-candidate improvement."""
    rows = []
    for mask in masks:
        try:
            if mask == -1:
                pid = ext.create_problem(M, N, K, True, MAX_WS, REQUESTED)
            else:
                pid = ext.create_problem(M, N, K, True, MAX_WS, REQUESTED, mask)
        except Exception as e:  # noqa: BLE001
            rows.append((mask, -1, float("nan"), float("nan"), float("nan"),
                         f"create_problem failed: {str(e)[:60]}"))
            continue
        na = ext.num_algos(pid)
        if na == 0:
            rows.append((mask, -1, float("nan"), float("nan"), float("nan"),
                         "0 candidates"))
            continue
        inp = torch.randn(M, K, device=dev, dtype=dtype) * 0.1
        w = torch.randn(N, K, device=dev, dtype=dtype) * 0.05
        b = torch.randn(N, device=dev, dtype=dtype) * 0.05
        out = torch.empty(M, N, device=dev, dtype=dtype)
        it = iters_for(M, N, K)

        # best-candidate improvement over idx0, taken as the WORST (min) across
        # REPEATS so a one-off fluke does not survive.
        best_impr = 1e9
        best_idx_final, best_ms_final, idx0_final, info_final = -1, 0.0, 0.0, ""
        for rep in range(REPEATS):
            t0 = ext.time_algo(pid, 0, inp, w, b, out, WARMUP, it)
            cand = [(0, t0)]
            for k in range(1, na):
                try:
                    tk = ext.time_algo(pid, k, inp, w, b, out, WARMUP, it)
                except Exception:  # noqa: BLE001
                    continue
                cand.append((k, tk))
            bk, bms = min(cand, key=lambda z: z[1])
            impr = (t0 - bms) / t0
            if impr < best_impr:
                best_impr = impr
                best_idx_final, best_ms_final, idx0_final = bk, bms, t0
                info_final = ext.algo_info(pid, bk)
        rows.append((mask, best_idx_final, best_ms_final, idx0_final,
                     best_impr, info_final))
        del inp, w, b, out
        torch.cuda.empty_cache()
    return rows


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    f16 = build("g6_9_lt_f16", F16_SRC)
    f32 = build("g6_9_lt_f32", F32_SRC)
    try:
        import subprocess
        v = subprocess.run(["python3", "-c",
            "import torch;print(torch.backends.cuda.preferred_blas_library())"],
            capture_output=True, text=True).stdout.strip()
        print("preferred_blas_library:", v)
    except Exception:  # noqa: BLE001
        pass
    print()

    retained = []
    for (M, d) in MD:
        for label, N, K, ext, dtype, masks in [
            (f"qkv     M{M} d{d}", 3 * d, d, f16, torch.float16, [2, 7]),
            (f"proj    M{M} d{d}", d,     d, f16, torch.float16, [2, 7]),
            (f"ffn_out M{M} d{d}", d,     d, f32, torch.float32, [-1]),
        ]:
            rows = search(ext, M, N, K, dtype, masks, dev)
            for (mask, bidx, bms, t0, impr, info) in rows:
                tag = {2: "m2", 7: "m7", -1: "tf32"}[mask]
                flag = ""
                if impr == impr and impr > THRESH and mask in (2, -1):
                    flag = "  <== RETAIN"
                    retained.append((label, tag, impr, bidx, bms, t0, info))
                elif impr == impr and impr > THRESH and mask == 7:
                    flag = "  (m7 only -- inadmissible w/o accuracy arg)"
                if impr != impr:
                    print(f"  {label:22s} {tag:4s}  {info}")
                else:
                    print(f"  {label:22s} {tag:4s}  idx0 {t0*1e3:8.2f}us -> "
                          f"best[{bidx}] {bms*1e3:8.2f}us  ({impr*100:+5.2f}%){flag}")
        print()

    print("=" * 72)
    if not retained:
        print("PHASE 2 RESULT: no uncovered signature beats idx0 by >2% "
              "(policy-compliant). STOP -- clean negative, model untouched.")
        return 0
    print("PHASE 2 RESULT: candidates surviving isolated >2% (reproducible):")
    for r in retained:
        print(f"  {r[0]:22s} {r[1]:4s}  {r[2]*100:+.2f}%  algo {r[6]}")
    print("-> proceed to Phase 3 (static lookup, affected shapes only) for these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
