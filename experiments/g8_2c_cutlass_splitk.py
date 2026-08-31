"""
G8.2 Phase 1 -- CUTLASS FP16-accumulate WITH split-K, at the official shapes.

Built for completeness. Phase 2 (job 236) already showed the official causal
GEMMs are not tensor-core bound -- they reach 0.6-27% of even the 165.2 TF lower
tier, and accF32 vs accF16 differ by ~1% where step 37 measured x1.43-1.53 at
K=512/N=1536. That predicts CUTLASS cannot help, because CUTLASS improves
tensor-core EFFICIENCY and efficiency is not what binds.

This tests the prediction with the best kernel available rather than resting on
it. CUTLASS reached 71.5% of its tier in step 39 (against the hand kernel's
55.1%), so if anything can convert the accumulate tier at these shapes, it is
this. Four split-K configs were added (cfg24-27, generated via
experiments/g4_6_gen_cfgs.py) -- the one cell the accumulate-tier work never
covered: CUTLASS-grade efficiency combined with the FP32 rebuild.

Controls: cfg2 (FP16-accum, same tile, no split) and cfg12 (FP32-accum twin).
The bar is cuBLAS, which already runs at 91.2% of the lower tier.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = "/work"
sys.path.insert(0, ROOT)
CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_gemm.cpp")
NUM_CFG = 28
SRC_CU = [os.path.join(ROOT, "csrc", f"g4_6_cutlass_cfg{i:02d}.cu")
          for i in range(NUM_CFG)]
DEV = torch.device("cuda")
PEAK_F32ACC, PEAK_F16ACC = 165.2, 330.3

# the fp16 causal GEMMs of the official matrix (K = d_model)
CASES = [
    ("row09-11 qkv  ", 8192, 128, 384),
    ("row09-11 o/ffn", 8192, 128, 128),
    ("row06    qkv  ", 1280000, 128, 384),
    ("row13    qkv  ", 65536, 128, 384),
    ("row08    qkv  ", 8192, 1024, 3072),   # K=1024, the one big-K official row
    ("row03    qkv  ", 512, 128, 384),
]
# cfgs of interest: split-K variants + their controls
SHOW = [2, 12, 24, 25, 26, 27]


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    t0 = time.time()
    m = load(name="g8_2c_cutlass", sources=[SRC_CPP] + SRC_CU,
             build_directory=bd, with_cuda=True,
             extra_include_paths=[os.path.join(CUTLASS, "include"),
                                  os.path.join(CUTLASS, "tools", "util", "include")],
             extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                "--expt-relaxed-constexpr",
                                "--expt-extended-lambda",
                                "-diag-suppress", "179,177,20012"],
             verbose=False)
    print(f"built {NUM_CFG} CUTLASS TUs in {time.time() - t0:.0f}s")
    return m


def ev(fn, warmup=10, iters=30):
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
    ext = build()
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    for c in SHOW:
        print(f"  cfg[{c}] = {ext.cfg_name(c)}")
    print(f"\ntier ceilings: FP32-accum {PEAK_F32ACC} TF, FP16-accum "
          f"{PEAK_F16ACC} TF. cuBLAS is the bar (91.2% of the lower tier).\n")

    hdr = f"{'GEMM':>15} {'M':>8} {'K':>5} {'N':>5} {'cuBLAS TF':>10}" + \
          "".join(f"{'cfg' + str(c):>15}" for c in SHOW)
    print(hdr); print("-" * len(hdr))
    any_win = []
    for lab, M, K, N in CASES:
        try:
            a = torch.randn(M, K, device=DEV, dtype=torch.float16)
            w = torch.randn(N, K, device=DEV, dtype=torch.float16)
            b = torch.randn(N, device=DEV, dtype=torch.float16)
            out = torch.empty(M, N, device=DEV, dtype=torch.float16)
        except RuntimeError as e:
            print(f"{lab:>15} {M:>8} skipped (alloc): {str(e)[:50]}")
            torch.cuda.empty_cache(); continue
        flop = 2.0 * M * N * K
        t_cub = ev(lambda: F.linear(a, w, b))
        tf_cub = flop / (t_cub * 1e-3) / 1e12
        cells = []
        for c in SHOW:
            try:
                nb = ext.cfg_workspace_bytes(c, M, N, K)
                ws = torch.empty(max(int(nb), 1), device=DEV, dtype=torch.uint8)
                t = ev(lambda: ext.cutlass_gemm(c, a, w, b, out, ws))
                tf = flop / (t * 1e-3) / 1e12
                cells.append(f"{tf:>8.1f} {t / t_cub:>5.2f}x")
                if t < t_cub:
                    any_win.append((lab, c, t_cub / t))
                del ws
            except Exception as e:
                cells.append(f"{('x:' + type(e).__name__)[:14]:>15}")
            torch.cuda.empty_cache()
        print(f"{lab:>15} {M:>8} {K:>5} {N:>5} {tf_cub:>10.1f}" + "".join(cells))
        del a, w, b, out
        torch.cuda.empty_cache()

    print("\n  (cell: achieved TFLOP/s, and time relative to cuBLAS -- "
          "<1.00x is faster)\n")
    print(f"  cfg2  = FP16-accum, no split   (control)")
    print(f"  cfg12 = FP32-accum twin        (control)")
    print(f"  cfg24/25 = FP16-accum + split-K 2 / 4")
    print(f"  cfg26/27 = FP16-accum + split-K 2, other tiles")
    print(f"\n  configs beating cuBLAS: {any_win if any_win else 'NONE'}")
    if not any_win:
        print("\n  VERDICT CONFIRMED: even CUTLASS-grade efficiency with a split-K\n"
              "  FP32 rebuild does not beat cuBLAS at the official shapes. The\n"
              "  binding constraint is arithmetic intensity, not kernel quality --\n"
              "  exactly as Phase 2 predicted. G8.2 closes on measurement.")
    print("\nG8_2C_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
