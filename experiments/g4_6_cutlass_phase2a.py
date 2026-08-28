#!/usr/bin/env python3
"""
G4.6 Phase 2a -- fp64 ground-truth correctness of the CUTLASS FP16-accumulate
GEMM, and the accuracy-BUDGET arithmetic at the real shapes.

Entered only because the user explicitly lowered the 80%-of-tier bar that step
39 stopped at (Phase 1 reached 71.4-71.7%).  Speed is already established; this
file is about whether the numerics are affordable.

WHY THIS IS THE REAL GATE, NOT A FORMALITY.  At large_batch the model already
sits at `max_abs = 0.000905529` against a 1e-3 atol (step 28) -- ~90% of the
budget spent on FP16 *storage* alone.  FP16 *accumulation* over K=512 spends
more of the SAME budget.  Phase 1 already showed the GEMM-level cost:
normalised error 2.91e-03 (accF16) vs 3.91e-04 (accF32), i.e. ~7.4x worse.
The external gate is DISJUNCTIVE per element though --
`abs_err <= 1e-3 OR abs_err <= 1e-2 * |ref|` (benchmark.py compare_outputs) --
so exceeding 1e-3 is not automatically fatal, which is why this is measured
rather than assumed either way.

Reference is fp64 throughout -- NEVER cuBLAS and NEVER the hand-written
g4_4 kernel, both of which are themselves approximations of the thing being
checked.  Three tiers, mirroring step 37 Stage 0a:
  1. deterministic input whose exact answer is representable in fp16 even
     under fp16 accumulation -> expect EXACTLY 0.0 error.
  2. one-hot sweep -> catches any index/layout transposition that a symmetric
     random test would hide.
  3. random fp16 at the REAL shapes -> the honest error magnitude, priced
     against the fp32-accumulate twin and against cuBLAS in the same harness.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_gemm.cpp")
NUM_CFG = 24
SRC_CU = [os.path.join(ROOT, "csrc", f"g4_6_cutlass_cfg{i:02d}.cu")
          for i in range(NUM_CFG)]

# Phase 1's winners (job 102). cfg[6]/cfg[15] are within noise of each other at
# qkv/large_batch; cfg[6] had by far the better repeatability (0.67% vs 4.76%)
# so it is the one carried forward. cfg[18] won out_proj/large_batch.
CFG_QKV = 6
CFG_OUT = 18
CFG_ACCF32_TWIN = 12      # accF32, for the same-harness precision A/B

D = 512
QKV_N = 3 * D
ATOL = 1e-3
RTOL = 1e-2


def build():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="g4_6_cutlass_gemm",
        sources=[SRC_CPP] + SRC_CU,
        build_directory=build_dir,
        with_cuda=True,
        extra_include_paths=[os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools", "util", "include")],
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "--expt-relaxed-constexpr",
                           "--expt-extended-lambda",
                           "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"],
        verbose=False,
    )


def disjunctive(ref64, got):
    """Exactly benchmark.py compare_outputs' criterion, on fp32 views."""
    ref = ref64.float()
    opt = got.float()
    err = (opt - ref).abs()
    ok = (err <= ATOL) | (err <= RTOL * ref.abs())
    return (float(err.max().item()),
            int((~ok).sum().item()),
            int(ok.numel()))


def main():
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    ext = build()
    ws = torch.empty(1, dtype=torch.uint8, device=dev)
    ok_all = True

    # ---- tier 1: exact-integer input, EXACT expected answer ---------------
    # A,B in {-1,0,1} and K=512 -> every partial sum is an integer with
    # |sum| <= 512.  fp16 represents every integer up to 2048 exactly, so the
    # fp16-ACCUMULATE answer is the exact integer answer.  Any nonzero error
    # here is a layout/addressing bug, not rounding.
    print("\n=== tier 1: deterministic integer input, exact expected answer ===")
    M, K, N = 1024, D, QKV_N
    mm = torch.arange(M, device=dev).view(M, 1)
    kk = torch.arange(K, device=dev).view(1, K)
    nn = torch.arange(N, device=dev).view(N, 1)
    A = ((mm + kk) % 3 - 1).to(torch.float16)
    W = ((nn * 7 + kk) % 3 - 1).to(torch.float16)
    b = ((nn.view(N) % 3) - 1).to(torch.float16)
    ref64 = A.double() @ W.double().t() + b.double()
    for cfg in (CFG_QKV, CFG_OUT, CFG_ACCF32_TWIN):
        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ext.cutlass_gemm(cfg, A, W, b, out, ws)
        e = (out.double() - ref64).abs().max().item()
        print(f"  cfg[{cfg:2d}] {ext.cfg_name(cfg):42s} max_abs vs fp64 = "
              f"{e:.6e}  {'EXACT' if e == 0.0 else 'NOT EXACT -- BUG'}")
        ok_all &= (e == 0.0)

    # ---- tier 2: one-hot sweep, catches index transposition ---------------
    print("\n=== tier 2: one-hot sweep (any transposition is a hard fail) ===")
    Mo, Ko, No = 256, 128, 192   # deliberately NOT tile-aligned: also exercises
                                 # CUTLASS's predication on ragged shapes
    bad = 0
    trials = 0
    zb = torch.zeros(No, device=dev, dtype=torch.float16)
    for m in (0, 1, 127, 255):
        for k in (0, 1, 63, 127):
            for n in (0, 1, 191):
                A = torch.zeros(Mo, Ko, device=dev, dtype=torch.float16)
                W = torch.zeros(No, Ko, device=dev, dtype=torch.float16)
                A[m, k] = 2.0
                W[n, k] = 3.0
                out = torch.empty(Mo, No, device=dev, dtype=torch.float16)
                ext.cutlass_gemm(CFG_QKV, A, W, zb, out, ws)
                exp = torch.zeros(Mo, No, device=dev, dtype=torch.float16)
                exp[m, n] = 6.0
                trials += 1
                if not torch.equal(out, exp):
                    bad += 1
                    if bad <= 3:
                        print(f"    MISMATCH m={m} k={k} n={n}: nonzero at "
                              f"{out.nonzero().tolist()[:4]}")
    print(f"  one-hot mismatches: {bad} / {trials}")
    ok_all &= (bad == 0)

    # ---- tier 3: real shapes, honest error, priced against the twins ------
    print("\n=== tier 3: random fp16 at the real shapes, vs fp64 ===")
    print(f"    gate is DISJUNCTIVE per element: "
          f"abs<={ATOL} OR abs<={RTOL}*|ref|")
    g = torch.Generator(device=dev)
    CASES = [("qkv      large_batch", 32768, D, QKV_N, CFG_QKV),
             ("out_proj large_batch", 32768, D, D,     CFG_OUT),
             ("qkv      default    ",  1024, D, QKV_N, CFG_QKV)]
    for name, M, K, N, cfg in CASES:
        g.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
        # Post-LayerNorm activations are ~unit variance; weights are small.
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        ref64 = inp.double() @ w.double().t() + b.double()

        cub = F.linear(inp, w, b)                       # fp16 storage/fp32 acc
        e_cub, f_cub, n_tot = disjunctive(ref64, cub)

        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ext.cutlass_gemm(cfg, inp, w, b, out, ws)
        e_c16, f_c16, _ = disjunctive(ref64, out)

        out32 = torch.empty(M, N, device=dev, dtype=torch.float16)
        ext.cutlass_gemm(CFG_ACCF32_TWIN, inp, w, b, out32, ws)
        e_c32, f_c32, _ = disjunctive(ref64, out32)

        print(f"\n  {name}  M={M} K={K} N={N}   (|ref| max = "
              f"{ref64.abs().max().item():.4f})")
        print(f"    cuBLAS   F.linear  (fp16 store, fp32 acc) max_abs="
              f"{e_cub:.6e}  gate-failing elements {f_cub}/{n_tot}")
        print(f"    CUTLASS  cfg[{CFG_ACCF32_TWIN}] (fp16 store, fp32 acc) "
              f"max_abs={e_c32:.6e}  gate-failing elements {f_c32}/{n_tot}")
        print(f"    CUTLASS  cfg[{cfg}] (fp16 store, FP16 acc) max_abs="
              f"{e_c16:.6e}  gate-failing elements {f_c16}/{n_tot}")
        print(f"    -> FP16 accumulate costs x{e_c16/max(e_cub,1e-12):.2f} in "
              f"GEMM-level max_abs vs the cuBLAS call it would replace")

    print("\n" + "=" * 78)
    print("PHASE 2a KERNEL VERDICT:",
          "PASS -- CUTLASS computes the right answer (tiers 1-2 exact); "
          "tier 3 is the price, judged by the full sweep next"
          if ok_all else
          "FAIL -- CUTLASS is NOT computing the right answer; do NOT integrate")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
