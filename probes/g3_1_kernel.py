#!/usr/bin/env python3
"""
The G3.1 fused-FFN Triton kernel, exactly as evaluated in PROGRESS.md step 19.

This lives in probes/ and NOT in benchmark.py on purpose: G3.1 was measured
and rejected (0.18x-0.869x of the three torch kernels it replaces, and its
fast tf32 mode is 3.7x less accurate than cuBLAS's own TF32). benchmark.py is
bit-identical to the state results/g2_4b_sweep_run27.log validated. Kept here
so the measurement is reproducible rather than just asserted.

Tile budget (fp32 => b=4 bytes; Ada ceiling 101,376 B = 99 KB/SM) for the best
config BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, 8 warps = 256 threads:

  dot1  A = x  [64, 64]  = 16.0 KB     dot2  A = h  [64, 32]  =  8.0 KB
        B = w1 [32, 64]  =  8.0 KB           B = w2 [512, 32] = 64.0 KB
        -> 24.0 KB/stage                     -> 72.0 KB
  peak shared ~= max(N_stage*24.0, 72.0) + ~4 KB = 76 KB <= 99 KB   OK
  accumulator acc[64, 512] = 128 KB registers = 128 regs/thread @ 256 thr

BLOCK_N=64 needs 512*64*4 = 128 KB for dot2's B operand alone and Triton
refuses it loudly ("Required: 147456, Hardware limit: 101376"). BLOCK_M=128
puts acc at 256 regs/thread, over the 255 max: it compiles but spills, 8.6x
slower. The 512-wide accumulator is the structural cost of this fusion -- the
second GEMM reduces over ffn_dim, so acc must stay [BLOCK_M, 512] live across
the whole loop, where cuBLAS is free to pick a 64- or 128-wide output tile.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ffn_kernel(
    X, W1, B1, W2, B2, Y,
    M,
    stride_xm, stride_xk,
    stride_w1n, stride_w1k,
    stride_w2d, stride_w2n,
    stride_ym, stride_yd,
    D: tl.constexpr,
    NFF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREC: tl.constexpr,
):
    # One program owns BLOCK_M tokens and produces their FULL [BLOCK_M, D]
    # output row -- token-parallel, so no grid sync and no partial sums.
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_K)
    offs_bn = tl.arange(0, BLOCK_N)

    x_row = X + offs_m[:, None] * stride_xm
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # D and NFF are powers of two and BLOCK_K | D, BLOCK_N | NFF for every
    # config used, so only the M edge ever needs masking.
    for n0 in range(0, NFF, BLOCK_N):
        offs_n = n0 + offs_bn
        h = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_kk = k0 + offs_k
            a = tl.load(
                x_row + offs_kk[None, :] * stride_xk,
                mask=m_mask[:, None],
                other=0.0,
            )
            # nn.Linear stores .weight as [out, in]; load the tile in its
            # natural (coalesced) [BLOCK_N, BLOCK_K] orientation using the
            # tensor's real strides and let tl.trans fold the transpose into
            # the MMA operand layout -- indexing it transposed directly would
            # stride the fastest-varying axis by in_features and destroy load
            # coalescing. (For fp32 operands this transpose is NOT free; see
            # probes/g3_1_pretransposed_probe.py, which prices removing it.)
            w1 = tl.load(
                W1 + offs_n[:, None] * stride_w1n + offs_kk[None, :] * stride_w1k
            )
            h = tl.dot(a, tl.trans(w1), h, input_precision=PREC)

        h = h + tl.load(B1 + offs_n)[None, :]
        # exact erf GELU -- matches F.gelu(approximate="none"), NOT tanh.
        h = 0.5 * h * (1.0 + tl.erf(h * 0.7071067811865476))

        # h stays in registers here: this is the whole point of G3.1.
        w2 = tl.load(
            W2 + offs_d[:, None] * stride_w2d + offs_n[None, :] * stride_w2n
        )
        acc = tl.dot(h, tl.trans(w2), acc, input_precision=PREC)

    acc = acc + tl.load(B2 + offs_d)[None, :]
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yd,
        acc,
        mask=m_mask[:, None],
    )


def fused_ffn(
    x2d, w1, b1, w2, b2,
    block_m=64, block_n=32, block_k=64, num_warps=8, num_stages=2,
    prec="tf32",
):
    """GELU(x @ w1.T + b1) @ w2.T + b2, fused, intermediate never in HBM.

    x2d [M, D]; w1 [NFF, D] / b1 [NFF]; w2 [D, NFF] / b2 [D] -- nn.Linear
    layout for both weights, no transposed copy materialised.
    """
    M, D = x2d.shape
    nff = w1.shape[0]
    y = torch.empty((M, D), device=x2d.device, dtype=x2d.dtype)
    _fused_ffn_kernel[((M + block_m - 1) // block_m,)](
        x2d, w1, b1, w2, b2, y,
        M,
        x2d.stride(0), x2d.stride(1),
        w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1),
        y.stride(0), y.stride(1),
        D=D, NFF=nff,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, PREC=prec,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y
