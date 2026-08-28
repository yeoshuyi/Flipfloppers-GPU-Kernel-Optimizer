#!/usr/bin/env python3
"""
G4.4 Stage 0a -- PTX micro-unit-test for mma.sync.aligned.m16n8k16.f16.f16
(FP16 ACCUMULATE) + ldmatrix fragment addressing.

Kill gate for the whole G4.4 investigation: if fragment addressing cannot be
made verifiably correct on a single 16x16x16 problem with one warp, there is
no point building a tiled kernel on top of an unverified assumption.

Three checks, in increasing strength:
  1. hand-picked deterministic integer input  -> exact match against an fp64
     reference computed in numpy/torch (values chosen small enough that fp16
     represents them exactly and the k=16 reduction is exact, so the expected
     answer is EXACT, not approximate).
  2. random fp16 input -> match against fp64 reference within fp16-accumulate
     rounding.
  3. D_ld vs D_man    -> ldmatrix path vs hand-assembled-fragment path must be
     bit-identical; a mismatch isolates ldmatrix addressing from mma layout.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CU = os.path.join(ROOT, "csrc", "g4_4_mma_micro.cu")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_micro.cpp")


def build():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="g4_4_mma_micro",
        sources=[SRC_CPP, SRC_CU],
        build_directory=build_dir,
        with_cuda=True,
        # TORCH_CUDA_ARCH_LIST is not exported in this container; pass the
        # gencode explicitly.  -Xptxas -v prints the register/smem footprint.
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "-Xptxas", "-v", "--use_fast_math"],
        verbose=True,
    )


def report(name, D, ref64, D_other=None):
    err = (D.double() - ref64).abs().max().item()
    print(f"  {name}: max_abs vs fp64 = {err:.6e}")
    if D_other is not None:
        same = torch.equal(D, D_other)
        print(f"  ldmatrix path == hand-assembled path (bitwise): {same}")
        return err, same
    return err, None


def main():
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda}", flush=True)
    ext = build()
    print("\n" + "=" * 78)

    ok = True

    # ---- check 1: exact integer case -------------------------------------
    # A[i,k] = (i + k) % 5 - 2   in {-2..2};  W[n,k] = (n * 3 + k) % 4 - 1
    # products in [-6,6], k=16 terms -> |acc| <= 96, all integers, exactly
    # representable in fp16 (integers up to 2048 are exact).  So the fp16
    # accumulate answer is EXACTLY the integer answer.
    ii = torch.arange(16, device=dev).view(16, 1)
    kk = torch.arange(16, device=dev).view(1, 16)
    nn = torch.arange(8, device=dev).view(8, 1)
    A_i = ((ii + kk) % 5 - 2).to(torch.float16)
    W_i = ((nn * 3 + kk) % 4 - 1).to(torch.float16)
    ref_i = A_i.double() @ W_i.double().t()
    D_ld, D_man = ext.mma_micro(A_i.contiguous(), W_i.contiguous())
    print("check 1 -- deterministic integer input, EXACT expected answer")
    e, same = report("D_ld ", D_ld, ref_i, D_man)
    ok &= (e == 0.0) and same
    if e != 0.0:
        print("  ldmatrix path FAILED. first mismatching rows:")
        print("  got:\n", D_ld.double().cpu())
        print("  exp:\n", ref_i.cpu())
        em = (D_man.double() - ref_i).abs().max().item()
        print(f"  hand-assembled path max_abs = {em:.6e}  "
              f"({'mma layout wrong' if em != 0 else 'ldmatrix addressing wrong'})")

    # ---- check 2: random fp16 -------------------------------------------
    print("\ncheck 2 -- random fp16 input")
    g = torch.Generator(device=dev)
    worst = 0.0
    all_same = True
    for seed in range(8):
        g.manual_seed(seed)
        A = torch.randn(16, 16, device=dev, dtype=torch.float16, generator=g)
        W = torch.randn(8, 16, device=dev, dtype=torch.float16, generator=g)
        ref = A.double() @ W.double().t()
        D_ld, D_man = ext.mma_micro(A, W)
        e = (D_ld.double() - ref).abs().max().item()
        worst = max(worst, e)
        all_same &= torch.equal(D_ld, D_man)
    # fp16 has ~1e-3 relative precision; |ref| here is O(4), so O(4e-3) is the
    # expected fp16-accumulate rounding floor for a k=16 reduction.
    print(f"  worst max_abs vs fp64 over 8 seeds = {worst:.6e}")
    print(f"  ldmatrix == hand-assembled on all 8 seeds: {all_same}")
    ok &= all_same and worst < 5e-2

    # ---- check 3: asymmetry probe ---------------------------------------
    # A one-hot / W one-hot sweep catches any row/col transposition that a
    # symmetric random test could hide.
    print("\ncheck 3 -- one-hot sweep (catches any index transposition)")
    bad = 0
    for m in (0, 1, 7, 8, 15):
        for k in (0, 1, 7, 8, 15):
            for n in (0, 1, 7):
                A = torch.zeros(16, 16, device=dev, dtype=torch.float16)
                W = torch.zeros(8, 16, device=dev, dtype=torch.float16)
                A[m, k] = 2.0
                W[n, k] = 3.0
                D_ld, _ = ext.mma_micro(A, W)
                exp = torch.zeros(16, 8, device=dev, dtype=torch.float16)
                exp[m, n] = 6.0
                if not torch.equal(D_ld, exp):
                    bad += 1
                    if bad <= 3:
                        nz = D_ld.nonzero().tolist()
                        print(f"    MISMATCH m={m} k={k} n={n}: expected "
                              f"nonzero at [{m},{n}], got {nz}")
    print(f"  one-hot mismatches: {bad} / 75")
    ok &= (bad == 0)

    print("\n" + "=" * 78)
    print("STAGE 0a VERDICT:", "PASS -- fragment addressing verified"
          if ok else "FAIL -- do NOT proceed to a tiled kernel")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
