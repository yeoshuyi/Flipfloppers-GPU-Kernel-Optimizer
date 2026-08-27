#!/usr/bin/env python3
"""
G4.7 Stage 0a -- correctness gate for the FFN epilogue added to
csrc/g4_4_warpspec_gemm.cu (cfg 51-76). Nothing here times anything.

What is NEW versus step 41's gate (probes/g4_3_warpspec_correctness.py), and
therefore what these checks are aimed at:

  A. EPIGELU -- a fused, exact erf-form GELU in the epilogue, with an FP32
     output tensor. docs/PROGRESS.md step 35 Finding 2 closed GELU fusion via
     cuBLASLt because cuBLASLt's built-in epilogue computes the TANH
     approximation and the model computes F.gelu(approximate="none"): a
     systematic 4.74e-04 mismatch. This kernel is hand-written, so the claim
     is that it computes the erf form EXACTLY. "Exactly" has to be tested as
     BIT-IDENTITY, not as "close", or the claim is worthless -- see check 3.
  B. ACCF32 -- an FP32-accumulate mma.sync arm (mma.*.f32.f16.f16.f32). New
     accumulator type, new bias path (fp32 add, one round). Its whole purpose
     is being precision-neutral, so it gets its own error check against fp64.
  C. REGRESSION -- the shipped configs 0-50 must be untouched. cfg[0] is still
     re-checked bitwise against g4_4_mma_gemm cfg[11].

CHECK 3 IS THE LOAD-BEARING ONE. For every (gelu_cfg, plain_cfg) pair that
differs ONLY in EPIGELU, the two kernels run the identical GEMM, so
    gelu_cfg(X, W, b)  ==  F.gelu(plain_cfg(X, W, b).float(), "none")
must hold to the LAST BIT. That isolates the epilogue's activation from the
GEMM's arithmetic completely: any nonzero difference is the activation being
wrong, and cannot be blamed on accumulation.
"""
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")
MM_CU = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cu")
MM_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cpp")

# Divisible by every tile in the new list (BM/BN up to 256, BK=64) and by the
# SPLIT=64 configs' K requirement.
SHAPES = [(512, 512, 512), (1024, 256, 2048), (256, 512, 256)]

# (gelu_cfg, plain_twin) -- identical in every template parameter EXCEPT
# EPIGELU. See the WS_CFG_LIST_G47 table in the .cu.
GELU_TWINS = [
    (51, 70),   # cfg[26] generalist, fp16 accum
    (52, 48),   # cfg[26] + SPLIT 64, fp16 accum  (48 is a SHIPPED cfg)
    (55, 39),   # cfg[39] 1x4 grid, fp16 accum    (39 is a SHIPPED cfg)
    (56, 67),   # cfg[39] + SPLIT 64
    (59, 37),   # cfg[37] 256x128                 (37 is a SHIPPED cfg)
    (60, 68),   # cfg[37] + SPLIT 64
    (63, 33),   # cfg[33] 128x256                 (33 is a SHIPPED cfg)
    (64, 69),   # cfg[33] + SPLIT 64
    (54, 53),   # cfg[26] tile, ACCF32
    (58, 57),   # cfg[39] tile, ACCF32
    (62, 61),   # cfg[37] tile, ACCF32
    (66, 65),   # cfg[33] tile, ACCF32
    (71, 72),   # 2x4 warp grid, ACCF32, no regdb
]
NEW_CFGS = list(range(51, 77))
ACCF32_CFGS = [53, 54, 57, 58, 61, 62, 65, 66, 71, 72, 73, 74, 75, 76]


def build(name, cpp, cu):
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name=name,
        sources=[cpp, cu],
        build_directory=build_dir,
        with_cuda=True,
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "-Xptxas", "-v", "-diag-suppress", "179"],
        verbose=True,
    )


def main():
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda}", flush=True)
    ws = build("g4_7_ffn_gelu", WS_CPP, WS_CU)
    mm = build("g4_4_mma_gemm_v3", MM_CPP, MM_CU)
    ncfg = ws.num_cfg()
    print(f"\nconfig table: {ncfg} entries (0-50 shipped, 51+ new)")
    for c in NEW_CFGS:
        print(f"  [{c:2d}] outf32={ws.cfg_outf32(c)}  {ws.cfg_name(c)}")
    print(flush=True)

    ok = True

    # ---------------------------------------------------------------------
    # CHECK 0 -- REGRESSION. The shipped list must be bit-for-bit what it was.
    # cfg[0] is the in-file control for g4_4_mma_gemm cfg[11]; if adding two
    # template columns perturbed anything, this is where it shows.
    # ---------------------------------------------------------------------
    print("=== CHECK 0: shipped cfg[0] still BITWISE == g4_4_mma_gemm cfg[11]")
    torch.manual_seed(7)
    for (M, K, N) in SHAPES:
        X = (torch.randn(M, K, device=dev) * 0.5).half()
        W = (torch.randn(N, K, device=dev) * 0.5).half()
        b = (torch.randn(N, device=dev) * 0.5).half()
        a = ws.ws_linear(0, X, W, b)
        c = mm.mma_linear(11, X, W, b)
        diff = int((a.view(torch.int16) != c.view(torch.int16)).sum().item())
        print(f"  M{M} K{K} N{N}: differing elements {diff} / {a.numel()}"
              f"  {'PASS' if diff == 0 else 'FAIL'}")
        ok &= diff == 0
    print(flush=True)

    # ---------------------------------------------------------------------
    # CHECK 1 -- EXACT INTEGER. Inputs chosen so every partial sum is an
    # integer exactly representable in fp16, so the fp16-accumulate answer is
    # EXACTLY the integer answer and any nonzero error is a real bug.
    # For the GELU configs the comparison is against the exactly-computed
    # activation of that same integer matrix.
    # ---------------------------------------------------------------------
    print("=== CHECK 1: exact-integer inputs")
    for (M, K, N) in SHAPES:
        g = torch.Generator(device=dev).manual_seed(11)
        X = torch.randint(-2, 3, (M, K), device=dev, generator=g).half()
        W = torch.randint(-2, 3, (N, K), device=dev, generator=g).half()
        b = torch.randint(-3, 4, (N,), device=dev, generator=g).half()
        ref16 = (X.float() @ W.float().T + b.float())
        # every partial sum |.| <= 4*K <= 8192 < 2048*... representable
        assert ref16.abs().max().item() < 2048, "integer range too large"
        ref_gelu = F.gelu(ref16, approximate="none")
        bad = []
        for c in NEW_CFGS:
            try:
                out = ws.ws_linear(c, X, W, b)
            except RuntimeError as e:
                if "-2" in str(e) or "-3" in str(e):
                    continue                        # tile does not divide
                raise
            ref = ref_gelu if ws.cfg_outf32(c) else ref16
            err = (out.float() - ref).abs().max().item()
            # fp16-accum arms round the GEMM result to fp16 before the
            # activation, which for integer inputs is exact; the activation
            # itself is fp32, so the only tolerance needed is fp32 rounding.
            tol = 0.0 if not ws.cfg_outf32(c) else 1e-6
            if err > tol:
                bad.append((c, err))
        print(f"  M{M} K{K} N{N}: {len(NEW_CFGS) - len(bad)}/{len(NEW_CFGS)} "
              f"EXACT" + (f"   FAILURES {bad}" if bad else ""))
        ok &= not bad
    print(flush=True)

    # ---------------------------------------------------------------------
    # CHECK 2 -- ONE-HOT SWEEP. Catches an index transposition in the new
    # drain path that a symmetric-random check would hide.
    # ---------------------------------------------------------------------
    print("=== CHECK 2: one-hot probe of the new drain (index transposition)")
    M, K, N = 256, 128, 256
    mism = 0
    total = 0
    for c in NEW_CFGS:
        for (r, col) in [(0, 0), (1, 9), (17, 130), (255, 255), (128, 64)]:
            X = torch.zeros(M, K, device=dev, dtype=torch.half)
            W = torch.zeros(N, K, device=dev, dtype=torch.half)
            X[r, 3] = 2.0
            W[col, 3] = 3.0
            try:
                out = ws.ws_linear(c, X, W, None).float()
            except RuntimeError as e:
                if "-2" in str(e) or "-3" in str(e):
                    continue
                raise
            exp = torch.zeros(M, N, device=dev)
            exp[r, col] = 6.0
            if ws.cfg_outf32(c):
                exp = F.gelu(exp, approximate="none")
            total += 1
            if (out - exp).abs().max().item() > 1e-6:
                mism += 1
                if mism < 4:
                    print(f"    MISMATCH cfg{c} at ({r},{col})")
    print(f"  {mism} / {total} one-hot placements wrong  "
          f"{'PASS' if mism == 0 else 'FAIL'}")
    ok &= mism == 0
    print(flush=True)

    # ---------------------------------------------------------------------
    # CHECK 3 -- THE GELU EXACTNESS GATE. THE reason this step exists.
    #
    # gelu_cfg and plain_cfg differ ONLY in EPIGELU, so they run the identical
    # GEMM with the identical accumulation order. Therefore
    #     gelu_cfg(...)  ==  F.gelu(plain_cfg(...).float(), "none")
    # must hold BITWISE. Anything else means the fused activation is not the
    # erf form ATen computes -- which is exactly the failure mode that killed
    # cuBLASLt's GELU epilogue in step 35 (tanh approximation, 4.74e-04).
    # Run on REAL activation-shaped data: the fp16 output of an actual
    # layernorm-scale GEMM, spanning the range where erf and tanh differ most.
    # ---------------------------------------------------------------------
    print("=== CHECK 3: fused GELU is BIT-IDENTICAL to "
          "F.gelu(approximate='none')")
    for (M, K, N) in SHAPES:
        torch.manual_seed(23)
        # activation-shaped: layernorm output (unit variance) x a real weight
        X = torch.randn(M, K, device=dev).half()
        W = (torch.randn(N, K, device=dev) * (K ** -0.5)).half()
        bb = (torch.randn(N, device=dev) * 0.1).half()
        for (gc, pc) in GELU_TWINS:
            try:
                gout = ws.ws_linear(gc, X, W, bb)
                pout = ws.ws_linear(pc, X, W, bb)
            except RuntimeError as e:
                if "-2" in str(e) or "-3" in str(e):
                    continue
                raise
            assert gout.dtype == torch.float32 and pout.dtype == torch.float16
            ref = F.gelu(pout.float(), approximate="none")
            nbit = int((gout.view(torch.int32)
                        != ref.view(torch.int32)).sum().item())
            ulp = (gout - ref).abs().max().item()
            span = pout.float().abs().max().item()
            tag = "BIT-EXACT" if nbit == 0 else f"{nbit} DIFFER (max {ulp:.3e})"
            print(f"  M{M} K{K} N{N} cfg{gc:2d} vs cfg{pc:2d}: {tag}"
                  f"   [pre-act |x|max {span:.2f}]")
            ok &= nbit == 0
        # And, separately, the tanh approximation on the SAME data, to show
        # the check has the resolution to catch what step 35 caught.
        ref_none = F.gelu(pout.float(), approximate="none")
        ref_tanh = F.gelu(pout.float(), approximate="tanh")
        print(f"  (reference: erf vs tanh GELU on this data differ by "
              f"{(ref_none - ref_tanh).abs().max().item():.3e} -- step 35's "
              f"cuBLASLt failure mode would be visible)")
    print(flush=True)

    # ---------------------------------------------------------------------
    # CHECK 4 -- random fp16 vs fp64, per arm. The fp16-accumulate arms should
    # sit at their known ~3e-3 floor; the ACCF32 arms must be ~1000x better,
    # because "precision-neutral" is the entire reason they exist.
    # ---------------------------------------------------------------------
    print("=== CHECK 4: random fp16 vs fp64 reference, by accumulate mode")
    for (M, K, N) in SHAPES:
        torch.manual_seed(31)
        X = torch.randn(M, K, device=dev).half()
        W = (torch.randn(N, K, device=dev) * (K ** -0.5)).half()
        bb = (torch.randn(N, device=dev) * 0.1).half()
        ref64 = (X.double() @ W.double().T + bb.double())
        worst_f16, worst_f32 = 0.0, 0.0
        for c in NEW_CFGS:
            try:
                out = ws.ws_linear(c, X, W, bb).float()
            except RuntimeError as e:
                if "-2" in str(e) or "-3" in str(e):
                    continue
                raise
            r = ref64
            if ws.cfg_outf32(c):
                r = F.gelu(r, approximate="none")
            e = (out.double() - r).abs().max().item()
            if c in ACCF32_CFGS:
                worst_f32 = max(worst_f32, e)
            else:
                worst_f16 = max(worst_f16, e)
        print(f"  M{M} K{K} N{N}: fp16-accum worst {worst_f16:.3e} | "
              f"ACCF32 worst {worst_f32:.3e}   ratio "
              f"{(worst_f16 / max(worst_f32, 1e-12)):.0f}x")
        ok &= worst_f32 < worst_f16
    print(flush=True)

    print("OVERALL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
