"""
G8.2 Phase 2 (run BEFORE Phase 1, deliberately) -- what tier gain survives the
FP32 carry, and does it beat cuBLAS?

Phase 0 (job 235) left five official rows shippable under the 0.00180 ceiling,
all of them needing the FINEST carry (BK32/SPLIT32, cfg 27) -- a promotion every
32 columns of K instead of every 64. That carry is 4x more frequent than the
SPLIT-64 one that already cut the tier gain from x1.53 to x1.155 at K=512.

So the question that decides whether ANY kernel work is justified is: after
paying for the carry, is the FP16-accumulate tier still ahead of cuBLAS?

The bar is not the FP32-accumulate tier. cuBLAS already runs at 91.2% of that
tier while the hand kernel reaches 55.1% of its own (step 37), so the kernel has
to win by mechanism what it gives back in efficiency. Step 37's arithmetic:
2.00 x (0.551/0.912) = 1.21 with NO carry.

This measures the real GEMM shapes of the surviving rows against F.linear
(cuBLAS), so the answer is a measured ratio rather than an extrapolation. If the
carry eats the tier, CUTLASS cannot rescue it -- CUTLASS changes efficiency, not
the number of FP32 promotions -- and the whole line closes.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = "/work"
sys.path.insert(0, ROOT)
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

SRC_CU = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cu")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cpp")
PEAK_F32ACC, PEAK_F16ACC = 165.2, 330.3

CFGS = [("accF32 tier (cfg15)", 15), ("accF16 no carry (cfg0)", 0),
        ("accF16 SPLIT64 (cfg9)", 9), ("accF16 BK32/SPLIT64 (cfg26)", 26),
        ("accF16 BK32/SPLIT32 (cfg27)", 27)]

# (label, M, K, N) -- the three fp16 causal GEMMs of the surviving rows
SHAPES = [
    ("row02 qkv     ", 128, 128, 384), ("row02 out/ffn ", 128, 128, 128),
    ("row03 qkv     ", 512, 128, 384), ("row03 out/ffn ", 512, 128, 128),
    ("row09-11 qkv  ", 8192, 128, 384), ("row09-11 o/ffn", 8192, 128, 128),
]


def build_ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    return load(name="g8_2_mma_gemm", sources=[SRC_CPP, SRC_CU],
                build_directory=bd, with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-diag-suppress", "179"], verbose=False)


def ev(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


@torch.no_grad()
def main():
    ext = build_ext()
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"cuBLAS is the bar: it already runs at 91.2% of the 165.2 TF "
          f"FP32-accum tier.\n")
    hdr = (f"{'GEMM':>15} {'M':>7} {'K':>5} {'N':>5} {'cuBLAS TF':>10}" +
           "".join(f"{n.split('(')[0].strip():>16}" for n, _ in CFGS))
    print(hdr); print("-" * len(hdr))
    verdict = {}
    for lab, M, K, N in SHAPES:
        a = torch.randn(M, K, device=DEV, dtype=torch.float16)
        w = torch.randn(N, K, device=DEV, dtype=torch.float16)
        b = torch.randn(N, device=DEV, dtype=torch.float16)
        flop = 2.0 * M * N * K
        t_cub = ev(lambda: F.linear(a, w, b))
        cells, best16 = [], 0.0
        for name, c in CFGS:
            try:
                t = ev(lambda: ext.mma_linear(c, a, w, b))
                tf = flop / (t * 1e-3) / 1e12
                cells.append(f"{tf:>9.1f} {t / t_cub:>5.2f}x")
                if "accF16" in name:
                    best16 = max(best16, flop / (t * 1e-3) / 1e12)
            except Exception:
                cells.append(f"{'n/a':>16}")
        tf_cub = flop / (t_cub * 1e-3) / 1e12
        print(f"{lab:>15} {M:>7} {K:>5} {N:>5} {tf_cub:>10.1f}" + "".join(cells))
        verdict[lab] = (tf_cub, best16)
        del a, w, b
        torch.cuda.empty_cache()

    print("\n  (each cell: achieved TFLOP/s, and time relative to cuBLAS -- "
          "<1.00x means faster)\n")
    wins = [k for k, (c, f16) in verdict.items() if f16 > c]
    print(f"  shapes where the BEST FP16-accumulate config beats cuBLAS: "
          f"{wins if wins else 'NONE'}")
    if not wins:
        print("\n  VERDICT: the carry required for accuracy eats the tier. CUTLASS\n"
              "  cannot rescue this -- it changes kernel efficiency, not the number\n"
              "  of FP32 promotions the accuracy budget demands. Phase 1 is not\n"
              "  justified and the line closes.")
    print("\nG8_2B_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
