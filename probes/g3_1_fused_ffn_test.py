#!/usr/bin/env python3
"""
G3.1 standalone kernel probe -- correctness + tile-config timing for the fused
ffn_in -> GELU(exact) -> ffn_out Triton kernel, in isolation from the model.

Correctness uses this project's own disjunctive criterion:
    pass if  abs(user-ref) <= atol  OR  abs(user-ref) <= rtol*abs(ref)
with atol=1e-3, rtol=1e-2 -- the same rule benchmark.py grades the model on.

Timing compares the fused kernel against the exact torch sequence it replaces
(F.linear -> F.gelu -> F.linear) at every token count in the shape sweep, so
the M-threshold for the dispatch gate is measured, not guessed.

Run via sbatch (jobs/g3_1_probe.sbatch). Never directly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import g3_1_kernel as gk

D = 512
NFF = 2048
ATOL = 1e-3
RTOL = 1e-2

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
CONFIGS = [
    (64, 32, 64, 8, 2),
    (64, 32, 64, 8, 1),
    (64, 32, 128, 8, 2),
    (32, 32, 64, 8, 2),
    (32, 32, 128, 8, 2),
    (128, 32, 64, 8, 1),
    (16, 32, 64, 4, 2),
    (64, 64, 64, 8, 1),   # expected to blow the 99 KB shared ceiling
    (32, 64, 64, 8, 1),   # expected to blow the 99 KB shared ceiling
]

CORRECTNESS_M = [1, 7, 16, 64, 100, 127, 1024, 1000, 8192, 32768]
TIMING_M = [64, 1024, 8192, 32768]


def reference(x, w1, b1, w2, b2):
    return F.linear(F.gelu(F.linear(x, w1, b1), approximate="none"), w2, b2)


def grade(got, ref):
    err = (got - ref).abs()
    ok = (err <= ATOL) | (err <= RTOL * ref.abs())
    n_bad = int((~ok).sum())
    return n_bad, float(err.max()), int(ref.numel())


def make(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, D, device="cuda", dtype=torch.float32, generator=g)
    # weight/bias magnitudes matched to nn.Linear's default init so the
    # activation statistics resemble the real model's, not unit-variance noise.
    w1 = torch.randn(NFF, D, device="cuda", generator=g) * (D ** -0.5)
    b1 = torch.randn(NFF, device="cuda", generator=g) * (D ** -0.5)
    w2 = torch.randn(D, NFF, device="cuda", generator=g) * (NFF ** -0.5)
    b2 = torch.randn(D, device="cuda", generator=g) * (NFF ** -0.5)
    return x, w1, b1, w2, b2


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    print(f"torch {torch.__version__}  triton {getattr(gk.triton, '__version__', None)}")
    print(f"has_triton={True}  device={torch.cuda.get_device_name(0)}")
    print(f"tl.erf present: {hasattr(gk.tl, 'erf')}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    ok_configs = []

    print("\n===== CORRECTNESS =====")
    for cfg in CONFIGS:
        bmt, bn, bk, nw, ns = cfg
        worst = 0.0
        bad_total = 0
        note = ""
        try:
            for m in CORRECTNESS_M:
                x, w1, b1, w2, b2 = make(m, 1234 + m)
                ref = reference(x, w1, b1, w2, b2)
                got = gk.fused_ffn(x, w1, b1, w2, b2, bmt, bn, bk, nw, ns)
                nb, mx, tot = grade(got, ref)
                bad_total += nb
                worst = max(worst, mx)
                if nb:
                    note += f" M={m}:{nb}/{tot}"
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")[:220]
            print(f"  cfg{cfg}: COMPILE/LAUNCH ERROR: {msg}")
            continue
        status = "PASS" if bad_total == 0 else "FAIL"
        print(f"  cfg{cfg}: {status} max_abs={worst:.3e} failed={bad_total}{note}")
        if bad_total == 0:
            ok_configs.append(cfg)

    if not ok_configs:
        print("\nNo config passed -- nothing to time.")
        return 1

    print("\n===== TIMING (ms/call, single FFN, fp32/TF32) =====")
    for m in TIMING_M:
        x, w1, b1, w2, b2 = make(m, 99)
        ref_ms = timeit(lambda: reference(x, w1, b1, w2, b2))
        blocks = None
        line = [f"M={m:6d}  torch={ref_ms:.4f}"]
        best = (None, 1e9)
        for cfg in ok_configs:
            bmt, bn, bk, nw, ns = cfg
            ms = timeit(lambda: gk.fused_ffn(x, w1, b1, w2, b2, bmt, bn, bk, nw, ns))
            blocks = (m + bmt - 1) // bmt
            line.append(f"{cfg}={ms:.4f}({ms and ref_ms / ms:.2f}x,{blocks}blk)")
            if ms < best[1]:
                best = (cfg, ms)
        print("  " + "  ".join(line))
        print(f"    -> best {best[0]} {best[1]:.4f} ms  speedup {ref_ms/best[1]:.3f}x")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
