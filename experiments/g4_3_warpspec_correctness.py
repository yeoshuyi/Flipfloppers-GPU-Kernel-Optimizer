#!/usr/bin/env python3
"""
G4.3 Stage 0a -- correctness gate for csrc/g4_4_warpspec_gemm.cu.

Nothing in this file times anything. Per docs/PROGRESS.md step 37's Stage-0a
discipline: a low-level kernel gets a deterministic correctness gate BEFORE any
throughput number from it is trusted, because a warp-specialised kernel with a
broken barrier protocol can be fast AND wrong, and a fast-and-wrong number is
worse than no number.

The fragment addressing (ldmatrix/mma layout) is inherited verbatim from
csrc/g4_4_mma_micro.cu, which step 37 already verified (job 94,
results/g4_4_stage0a_micro_run94.log). What is NEW here, and therefore what
these checks are aimed at:

  A. the per-stage FULL/EMPTY named-barrier protocol replacing __syncthreads
     (a missed EMPTY arrival = the loader overwrites a live stage = silently
     wrong partial sums, or a hang)
  B. the shared-memory-staged 128-bit epilogue and its XOR swizzle (a wrong
     swizzle key = transposed / duplicated output chunks)
  C. the fat 64x64 consumer warp tile (NT goes 4 -> 8, new indexing)

Five checks, in increasing strength:

  1. BITWISE CONTROL. warpspec cfg[0] is BM128/BN128/BK64/stg2 with an 8-warp
     2x4 grid, plain (non-specialised) loads and the step-37 epilogue -- i.e.
     the same tile, same warp grid, same accumulation ORDER as the already
     measured g4_4_mma_gemm cfg[11]. It must be BITWISE IDENTICAL. If it is
     not, the new file is not the control it claims to be and every delta
     measured against it is meaningless.
  2. EXACT INTEGER. Inputs chosen so every partial sum is an integer
     representable exactly in fp16 -> the fp16-accumulate answer is EXACTLY the
     integer answer. Any nonzero error is a real bug, not rounding.
  3. ONE-HOT SWEEP. Catches any index transposition that checks 2/4 could hide
     (a swizzle that swaps two chunks still sums to the right total under
     symmetric random data).
  4. RANDOM FP16 vs FP64, at realistic magnitudes.
  5. SHAPE SWEEP -- every config re-run at several (M,K,N) so a tile-boundary
     bug at one shape cannot hide behind another.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")
MM_CU = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cu")
MM_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cpp")

# Divisible by every tile in the table (BM/BN up to 256, BK=64) and by the
# SPLIT=256 configs' K requirement.
SHAPES = [
    (512, 256, 512),
    (768, 512, 768),
    (256, 512, 1536),
]


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
    ws = build("g4_3_warpspec", WS_CPP, WS_CU)
    mm = build("g4_4_mma_gemm_v3", MM_CPP, MM_CU)
    ncfg = ws.num_cfg()
    print("\nwarpspec configs:")
    for c in range(ncfg):
        print(f"  [{c:2d}] {ws.cfg_name(c)}")
    print(flush=True)

    ok = True

    # ---- which configs are runnable at which shape -----------------------
    def try_run(c, inp, w, b):
        out = torch.empty(inp.size(0), w.size(0), device=dev,
                          dtype=torch.float16)
        try:
            ws.ws_gemm(c, inp, w, b, out)
            torch.cuda.synchronize()
        except Exception as e:
            return None, str(e)[:120]
        return out, None

    # =====================================================================
    print("=" * 78)
    print("CHECK 1 -- warpspec cfg[0] must be BITWISE == g4_4_mma_gemm cfg[11]")
    print("            (same tile, same 2x4 warp grid, same accumulate order)")
    g = torch.Generator(device=dev)
    bit_ok = True
    for (M, K, N) in SHAPES:
        g.manual_seed(1234 + M + K + N)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        o_ws = torch.empty(M, N, device=dev, dtype=torch.float16)
        o_mm = torch.empty(M, N, device=dev, dtype=torch.float16)
        ws.ws_gemm(0, inp, w, b, o_ws)
        mm.mma_gemm(11, inp, w, b, o_mm)
        torch.cuda.synchronize()
        same = torch.equal(o_ws, o_mm)
        nd = int((o_ws != o_mm).sum().item())
        print(f"  M={M:5d} K={K:4d} N={N:5d}  bitwise identical: {same}"
              f"   (differing elements: {nd}/{M*N})")
        bit_ok &= same
    ok &= bit_ok
    print(f"  -> CHECK 1 {'PASS' if bit_ok else 'FAIL'}", flush=True)

    # =====================================================================
    print("\n" + "=" * 78)
    print("CHECK 2 -- exact integer input; fp16 accumulate must be EXACT")
    # A[m,k] in {-2..2}, W[n,k] in {-1,0,1} -> |product| <= 2, and with K<=512
    # every partial sum is an integer of magnitude <= 1024, which fp16
    # represents exactly (integers are exact to 2048). So the expected answer
    # is EXACT, and any error at all is a bug.
    int_ok = True
    worst_by_cfg = {}
    for (M, K, N) in SHAPES:
        ii = torch.arange(M, device=dev).view(M, 1)
        kk = torch.arange(K, device=dev).view(1, K)
        nn = torch.arange(N, device=dev).view(N, 1)
        A = ((ii + kk) % 5 - 2).to(torch.float16)
        Wt = ((nn + kk * 2) % 3 - 1).to(torch.float16)
        bias = ((torch.arange(N, device=dev) % 7) - 3).to(torch.float16)
        ref = (A.double() @ Wt.double().t()) + bias.double()
        assert ref.abs().max().item() < 2048, ref.abs().max().item()
        for c in range(ncfg):
            out, err = try_run(c, A.contiguous(), Wt.contiguous(), bias)
            if out is None:
                worst_by_cfg.setdefault(c, []).append(("skip", err))
                continue
            e = (out.double() - ref).abs().max().item()
            worst_by_cfg.setdefault(c, []).append((f"{M}x{K}x{N}", e))
            if e != 0.0:
                int_ok = False
    for c in range(ncfg):
        rows = worst_by_cfg.get(c, [])
        bad = [r for r in rows if r[0] != "skip" and r[1] != 0.0]
        skipped = [r for r in rows if r[0] == "skip"]
        status = "EXACT" if not bad else f"FAIL {bad}"
        extra = f"  (skipped {len(skipped)} shape(s): {skipped[0][1]})" if skipped else ""
        print(f"  cfg[{c:2d}] {status}{extra}")
    ok &= int_ok
    print(f"  -> CHECK 2 {'PASS' if int_ok else 'FAIL'}", flush=True)

    # =====================================================================
    print("\n" + "=" * 78)
    print("CHECK 3 -- one-hot sweep (catches index transposition / swizzle "
          "chunk swaps)")
    M, K, N = 512, 256, 512
    hot_ok = True
    for c in range(ncfg):
        bad = 0
        tot = 0
        for m in (0, 1, 63, 64, 127, 255):
            for k in (0, 1, 63, 64, 255):
                for n in (0, 1, 63, 64, 511):
                    tot += 1
                    A = torch.zeros(M, K, device=dev, dtype=torch.float16)
                    Wt = torch.zeros(N, K, device=dev, dtype=torch.float16)
                    A[m, k] = 2.0
                    Wt[n, k] = 3.0
                    out, err = try_run(c, A, Wt, None)
                    if out is None:
                        bad = -1
                        break
                    exp = torch.zeros(M, N, device=dev, dtype=torch.float16)
                    exp[m, n] = 6.0
                    if not torch.equal(out, exp):
                        bad += 1
                        if bad <= 2:
                            nz = out.nonzero()
                            print(f"    cfg[{c}] MISMATCH m={m} k={k} n={n}: "
                                  f"expected nonzero at [{m},{n}], got "
                                  f"{nz[:5].tolist()} ({nz.size(0)} nonzeros)")
                if bad < 0:
                    break
            if bad < 0:
                break
        if bad < 0:
            print(f"  cfg[{c:2d}] not runnable at {M}x{K}x{N}, skipped")
            continue
        print(f"  cfg[{c:2d}] one-hot mismatches: {bad} / {tot}")
        hot_ok &= (bad == 0)
    ok &= hot_ok
    print(f"  -> CHECK 3 {'PASS' if hot_ok else 'FAIL'}", flush=True)

    # =====================================================================
    print("\n" + "=" * 78)
    print("CHECK 4/5 -- random fp16 vs fp64, every config x every shape")
    # fp16-accumulate over K terms: expected |err| ~ sqrt(K) * 2^-11 * |acc|.
    # At these magnitudes (|out| ~ 0.5) that is O(1e-3). A bug shows up as
    # orders of magnitude, not as a factor of 2, so the bar is deliberately
    # loose -- exactness is checks 2 and 3's job, this one is a sanity net.
    rnd_ok = True
    for (M, K, N) in SHAPES:
        g.manual_seed(99 + M)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        ref = (inp.double() @ w.double().t()) + b.double()
        scale = ref.abs().max().item()
        line = f"  M={M:5d} K={K:4d} N={N:5d} (|ref|max={scale:.3f}): "
        errs = []
        for c in range(ncfg):
            out, err = try_run(c, inp, w, b)
            if out is None:
                errs.append(f"[{c}]skip")
                continue
            e = (out.double() - ref).abs().max().item()
            errs.append(f"[{c}]{e:.1e}")
            if not (e < 0.15 * scale):
                rnd_ok = False
                errs[-1] += "!!"
        print(line)
        print("      " + "  ".join(errs))
    ok &= rnd_ok
    print(f"  -> CHECK 4/5 {'PASS' if rnd_ok else 'FAIL'}", flush=True)

    print("\n" + "=" * 78)
    print("G4.3 STAGE 0a VERDICT:",
          "PASS -- warp-spec kernel is correct, throughput numbers from it "
          "may be trusted" if ok else
          "FAIL -- do NOT time this kernel until fixed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
