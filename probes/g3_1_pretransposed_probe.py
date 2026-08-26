#!/usr/bin/env python3
"""
G3.1 iteration 3: is the fused kernel's 0.72x-vs-torch ceiling structural, or
is it the cost of tl.trans on fp32 MMA operands?

Job 40 established the kernel is arithmetically correct (tf32x3/ieee match fp64
to ~2e-6) but slower than the three-kernel torch sequence at every token count.
Both dots feed their B operand through tl.trans, because nn.Linear stores
.weight as [out, in]. For fp16 that folds into the MMA shared-memory
descriptor for free; for fp32/TF32 operand layouts it may lower to a real
shared-memory shuffle -- and dot2's B tile is [512, 32] = 64 KB, the largest
thing in the kernel.

This variant pre-transposes both weights ONCE into contiguous buffers
(w1t [D, NFF], w2t [NFF, D]) so every tile load is stride-1 along the
fastest-varying axis and no tl.trans appears anywhere. Arithmetic is
unchanged; only the layout is. If the 0.72x ceiling is really tl.trans, this
moves it. If it does not move, the ceiling is the fused structure itself
(a 512-wide accumulator pinned in registers across the whole ffn_dim
reduction) and G3.1 is closed on the performance axis.

Also sweeps BLOCK_M=128 at num_warps=16 -- the only untested way to keep the
accumulator at <=128 regs/thread while raising the MMA tile.

Run via sbatch. Never directly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

D = 512
NFF = 2048
ATOL = 1e-3
RTOL = 1e-2


@triton.jit
def _ffn_pt_kernel(
    X, W1T, B1, W2T, B2, Y,
    M,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w2n, stride_w2d,
    stride_ym, stride_yd,
    D: tl.constexpr, NFF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)
    offs_bn = tl.arange(0, BLOCK_N)

    x_row = X + offs_m[:, None] * stride_xm
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for n0 in range(0, NFF, BLOCK_N):
        offs_n = n0 + offs_bn
        h = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_kk = k0 + offs_k
            a = tl.load(x_row + offs_kk[None, :] * stride_xk,
                        mask=m_mask[:, None], other=0.0)
            # w1t is [D, NFF]: tile is [BLOCK_K, BLOCK_N], stride-1 along n.
            b = tl.load(W1T + offs_kk[:, None] * stride_w1k
                            + offs_n[None, :] * stride_w1n)
            h = tl.dot(a, b, h, input_precision=PREC)

        h = h + tl.load(B1 + offs_n)[None, :]
        h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))

        # w2t is [NFF, D]: tile is [BLOCK_N, D], stride-1 along d.
        w2 = tl.load(W2T + offs_n[:, None] * stride_w2n
                         + offs_d[None, :] * stride_w2d)
        acc = tl.dot(h, w2, acc, input_precision=PREC)

    acc = acc + tl.load(B2 + offs_d)[None, :]
    tl.store(Y + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd,
             acc, mask=m_mask[:, None])


def fused_pt(x, w1t, b1, w2t, b2, bm_, bn, bk, nw, ns, prec="tf32"):
    M, Dl = x.shape
    y = torch.empty((M, Dl), device=x.device, dtype=x.dtype)
    _ffn_pt_kernel[((M + bm_ - 1) // bm_,)](
        x, w1t, b1, w2t, b2, y, M,
        x.stride(0), x.stride(1),
        w1t.stride(0), w1t.stride(1),
        w2t.stride(0), w2t.stride(1),
        y.stride(0), y.stride(1),
        D=Dl, NFF=w1t.shape[1],
        BLOCK_M=bm_, BLOCK_N=bn, BLOCK_K=bk, PREC=prec,
        num_warps=nw, num_stages=ns,
    )
    return y


def make(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, D, device="cuda", generator=g)
    w1 = torch.randn(NFF, D, device="cuda", generator=g) * (D ** -0.5)
    b1 = torch.randn(NFF, device="cuda", generator=g) * (D ** -0.5)
    w2 = torch.randn(D, NFF, device="cuda", generator=g) * (NFF ** -0.5)
    b2 = torch.randn(D, device="cuda", generator=g) * (NFF ** -0.5)
    return x, w1, b1, w2, b2


def reference(x, w1, b1, w2, b2):
    return F.linear(F.gelu(F.linear(x, w1, b1), approximate="none"), w2, b2)


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


CONFIGS = [
    (64, 32, 64, 8, 2),
    (64, 32, 64, 8, 3),
    (64, 32, 128, 8, 2),
    (64, 64, 64, 8, 2),
    (128, 32, 64, 16, 2),
    (128, 64, 64, 16, 2),
    (32, 32, 64, 4, 2),
]


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"torch {torch.__version__}  triton {triton.__version__}")

    print("\n===== CORRECTNESS of the pre-transposed variant (vs fp64) =====")
    x, w1, b1, w2, b2 = make(1024, 11)
    w1t = w1.t().contiguous()
    w2t = w2.t().contiguous()
    exact = reference(x.double(), w1.double(), b1.double(),
                      w2.double(), b2.double())
    r_tf32 = reference(x, w1, b1, w2, b2)
    for cfg in CONFIGS:
        try:
            got = fused_pt(x, w1t, b1, w2t, b2, *cfg, prec="tf32")
        except Exception as ex:  # noqa: BLE001
            print(f"  cfg{cfg}: ERROR {str(ex)[:150]}")
            continue
        e64 = float((got.double() - exact).abs().max())
        err = (got.double() - r_tf32.double()).abs()
        ok = (err <= ATOL) | (err <= RTOL * r_tf32.double().abs())
        nb = int((~ok).sum())
        print(f"  cfg{cfg}: max_abs_vs_fp64={e64:.3e}  "
              f"vs_cublas_tf32 failed={nb}/{got.numel()}")

    print("\n===== TIMING: pre-transposed vs torch (ms/call) =====")
    for m in (1024, 8192, 32768):
        x, w1, b1, w2, b2 = make(m, 5)
        w1t = w1.t().contiguous()
        w2t = w2.t().contiguous()
        t_ref = timeit(lambda: reference(x, w1, b1, w2, b2))
        print(f"  M={m}  torch_tf32={t_ref:.4f} ms")
        best = (None, 1e9)
        for cfg in CONFIGS:
            try:
                t = timeit(lambda: fused_pt(x, w1t, b1, w2t, b2, *cfg,
                                            prec="tf32"))
            except Exception:  # noqa: BLE001
                print(f"    cfg{cfg}: ERR")
                continue
            blocks = (m + cfg[0] - 1) // cfg[0]
            print(f"    cfg{cfg}: {t:.4f} ms  {t_ref/t:.3f}x  ({blocks} blocks)")
            if t < best[1]:
                best = (cfg, t)
        print(f"    -> BEST {best[0]}  {best[1]:.4f} ms  {t_ref/best[1]:.3f}x")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
