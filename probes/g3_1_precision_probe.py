#!/usr/bin/env python3
"""
G3.1 diagnostic: WHERE does the fused-FFN divergence come from?

Probe v1 (job 39) showed every tile config producing byte-identical failure
counts, which rules out a tiling/indexing bug -- BLOCK_M does not change the
reduction order at all, and Triton's MMA steps K in fixed increments, so all
those configs really are the same arithmetic. The remaining candidates are:

  (a) tl.erf differs from torch's erf  -> isolated GELU test below
  (b) my kernel is simply wrong        -> would show as huge error vs fp64
  (c) both my kernel and cuBLAS are TF32 with DIFFERENT reduction orders, and
      what v1 measured is the (expected) divergence between two equally-valid
      TF32 answers -> both land at a similar distance from fp64

This measures all three against a float64 ground truth, so the answer is a
fact, not an inference. It also prices the accuracy/throughput ladder
(tf32 -> tf32x3 -> ieee) that CLAUDE.md's precision walk-down calls for.

Run via sbatch (jobs/g3_1_precision.sbatch). Never directly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import g3_1_kernel as gk

D = 512
NFF = 2048
ATOL = 1e-3
RTOL = 1e-2


@triton.jit
def _gelu_only_kernel(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    h = tl.load(X + off, mask=m)
    h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))
    tl.store(Y + off, h, mask=m)


def gelu_triton(x):
    y = torch.empty_like(x)
    n = x.numel()
    _gelu_only_kernel[((n + 1023) // 1024,)](x, y, n, BLOCK=1024)
    return y


def make(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, D, device="cuda", dtype=torch.float32, generator=g)
    w1 = torch.randn(NFF, D, device="cuda", generator=g) * (D ** -0.5)
    b1 = torch.randn(NFF, device="cuda", generator=g) * (D ** -0.5)
    w2 = torch.randn(D, NFF, device="cuda", generator=g) * (NFF ** -0.5)
    b2 = torch.randn(D, device="cuda", generator=g) * (NFF ** -0.5)
    return x, w1, b1, w2, b2


def reference(x, w1, b1, w2, b2):
    return F.linear(F.gelu(F.linear(x, w1, b1), approximate="none"), w2, b2)


def ref_fp64(x, w1, b1, w2, b2):
    return reference(x.double(), w1.double(), b1.double(),
                     w2.double(), b2.double())


def stats(got, ref):
    err = (got.double() - ref.double()).abs()
    ok = (err <= ATOL) | (err <= RTOL * ref.double().abs())
    return float(err.max()), int((~ok).sum()), ref.numel()


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"torch {torch.__version__}  triton {triton.__version__}")
    print(f"allow_tf32={torch.backends.cuda.matmul.allow_tf32}")

    # ---- (a) isolate GELU -------------------------------------------------
    print("\n===== GELU ISOLATION (tl.erf vs torch erf) =====")
    h = (torch.randn(1 << 20, device="cuda") * 3.0)
    gt = gelu_triton(h)
    gr = F.gelu(h, approximate="none")
    print(f"  max_abs_diff={float((gt-gr).abs().max()):.3e}  "
          f"bitwise_identical={bool(torch.equal(gt, gr))}")

    # ---- (b)/(c) attribute the GEMM error against fp64 --------------------
    print("\n===== ERROR vs float64 GROUND TRUTH =====")
    print("  (if kernel and cuBLAS are both ~equally far from fp64, v1 was")
    print("   measuring TF32 reduction-order divergence, not a bug)")
    for m in (64, 1024, 8192):
        x, w1, b1, w2, b2 = make(m, 4242 + m)
        exact = ref_fp64(x, w1, b1, w2, b2)
        rows = []
        # torch reference at TF32 ("high") and true fp32 ("highest")
        torch.set_float32_matmul_precision("high")
        r_tf32 = reference(x, w1, b1, w2, b2)
        torch.set_float32_matmul_precision("highest")
        r_fp32 = reference(x, w1, b1, w2, b2)
        torch.set_float32_matmul_precision("high")
        rows.append(("cublas_tf32", r_tf32))
        rows.append(("cublas_fp32", r_fp32))
        for prec in ("tf32", "tf32x3", "ieee"):
            try:
                rows.append((f"triton_{prec}",
                             gk.fused_ffn(x, w1, b1, w2, b2, 64, 32, 64, 8, 2,
                                           prec=prec)))
            except Exception as ex:  # noqa: BLE001
                print(f"    triton_{prec}: ERROR {str(ex)[:160]}")
        print(f"  M={m}  |y|_max={float(exact.abs().max()):.4f}  "
              f"|y|_rms={float(exact.pow(2).mean().sqrt()):.4f}")
        for name, val in rows:
            mx, nb, tot = stats(val, exact)
            print(f"    {name:14s} vs fp64: max_abs={mx:.3e}")
        # the ACTUAL grading comparison: candidate vs cublas_tf32 reference
        for name, val in rows[1:]:
            mx, nb, tot = stats(val, r_tf32)
            print(f"    {name:14s} vs cublas_tf32: max_abs={mx:.3e} "
                  f"failed={nb}/{tot} ({100.0*nb/tot:.3f}%)")

    # ---- price the ladder -------------------------------------------------
    print("\n===== TIMING (ms/call) =====")
    for m in (1024, 8192, 32768):
        x, w1, b1, w2, b2 = make(m, 7)
        t_ref = timeit(lambda: reference(x, w1, b1, w2, b2))
        out = [f"M={m:6d} torch_tf32={t_ref:.4f}"]
        for prec in ("tf32", "tf32x3", "ieee"):
            for cfg in ((64, 32, 64, 8, 2), (32, 32, 64, 8, 2)):
                try:
                    t = timeit(lambda: gk.fused_ffn(
                        x, w1, b1, w2, b2, *cfg, prec=prec))
                    out.append(f"{prec}{cfg[0]}x{cfg[1]}={t:.4f}({t_ref/t:.2f}x)")
                except Exception as ex:  # noqa: BLE001
                    out.append(f"{prec}{cfg[0]}x{cfg[1]}=ERR")
        print("  " + "  ".join(out))

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
