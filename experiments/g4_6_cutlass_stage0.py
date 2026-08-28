#!/usr/bin/env python3
"""
G4.6 Phase 0 -- CUTLASS toolchain feasibility gate.

QUESTION: can a stock CUTLASS TensorOp FP16 GEMM be compiled, unmodified,
against THIS container's toolchain (CUDA 13.1 nvcc / torch 2.13.0+cu130 /
g++ 13.3) through torch.utils.cpp_extension.load(), and does it produce sane
numbers on sm_89?  If not, G4.6 stops here exactly as G4.5 (step 38) stopped
at its own toolchain gate.

VENDORING (how to recreate .cutlass/, which is gitignored):
    cd /scratch/work && git clone --depth 1 --branch v4.7.1 \
        https://github.com/NVIDIA/cutlass.git .cutlass
Only include/ (and tools/util/include/ for the utility headers) is used --
the CUTLASS Profiler and the cutlass_library python package are NOT built.

NOTE ON THE EXAMPLE CHOSEN: the plan's starting guess
`examples/18_ampere_fp16_tensorop_gemm` does not exist in v4.7.1 (example 18
there is `18_ampere_fp64_tensorop_affine2_gemm`).  `ls examples/` was checked
against this clone; the closest live match is `examples/12_gemm_bias_relu`
(cutlass::half_t inputs, cutlass::gemm::device::Gemm, a real bias epilogue via
LinearCombinationRelu/NoBetaScaling).  Its config block is reproduced verbatim
in csrc/g4_6_cutlass_stock.cu.

GATES
  0a  compiles to a .so with zero edits to CUTLASS internals; compile time
      recorded.
  0b  runs sanely: a deterministic, exactly-representable input reproduces an
      fp64 reference EXACTLY (step 37 Stage-0a's trick), and random fp16 input
      matches fp64 within fp32-accumulate rounding.
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CU = os.path.join(ROOT, "csrc", "g4_6_cutlass_stock.cu")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_stock.cpp")


def build():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    mod = load(
        name="g4_6_cutlass_stock",
        sources=[SRC_CPP, SRC_CU],
        build_directory=build_dir,
        with_cuda=True,
        extra_include_paths=[os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools", "util", "include")],
        # TORCH_CUDA_ARCH_LIST is not exported in this container; pass the
        # gencode explicitly, as probes/g4_4_mma_micro.py does.
        # --expt-relaxed-constexpr is CUTLASS's own documented requirement.
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "--expt-relaxed-constexpr",
                           "--expt-extended-lambda",
                           "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                           "-Xptxas", "-v"],
        verbose=True,
    )
    return mod, time.time() - t0


def main():
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"torch.version.cuda {torch.version.cuda}", flush=True)
    print(f"cutlass clone: {CUTLASS}", flush=True)

    print("\n=== GATE 0a: compile ===", flush=True)
    try:
        ext, secs = build()
    except Exception as e:  # noqa: BLE001
        print("GATE 0a FAILED -- CUTLASS does not compile on this toolchain.")
        print(f"exception type: {type(e).__name__}")
        print(str(e)[-8000:])
        print("\nG4.6 VERDICT: PHASE 0 NEGATIVE (toolchain).")
        return 1
    print(f"GATE 0a PASS -- compiled in {secs:.1f}s "
          f"({secs / 60.0:.2f} min), CUTLASS {ext.cutlass_version()}",
          flush=True)

    print("\n=== GATE 0b: runs sanely ===", flush=True)
    M = N = K = 256
    ws_bytes = ext.stock_workspace_bytes(M, N, K)
    print(f"  M=N=K={M}; get_workspace_size = {ws_bytes} bytes")
    workspace = torch.empty(max(ws_bytes, 1), dtype=torch.uint8, device=dev)

    ok = True

    # ---- check 1: deterministic, exactly-representable ---------------------
    # A[m,k] = (m+k)%5 - 2 in {-2..2}; B[k,n] = (n*3+k)%4 - 1 in {-1..2}
    # products in [-4,4], K=256 terms -> |acc| <= 1024, all integers, exact in
    # both fp16 storage and the fp32 accumulator.  bias is large and positive
    # so the Relu in the stock epilogue never clips and the check stays full
    # rank (a clipped reference would hide half the output).
    mm = torch.arange(M, device=dev).view(M, 1)
    kk = torch.arange(K, device=dev).view(1, K)
    nn = torch.arange(N, device=dev).view(N, 1)
    A = ((mm + kk) % 5 - 2).to(torch.float16)          # logical [M,K]
    B = ((nn * 3 + kk) % 4 - 1).to(torch.float16)      # logical [N,K] = B^T
    bias = (4096.0 + torch.arange(M, device=dev).float())

    a_cm = A.t().contiguous()          # (K,M) contiguous == col-major [M,K]
    b_cm = B.contiguous()              # (N,K) contiguous == col-major [K,N]
    d_cm = torch.zeros(N, M, dtype=torch.float32, device=dev)
    ext.stock_gemm(a_cm, b_cm, bias, d_cm, workspace)
    D = d_cm.t()                       # logical [M,N]

    ref = torch.clamp(A.double() @ B.double().t() + bias.double().view(M, 1),
                      min=0.0)
    e1 = (D.double() - ref).abs().max().item()
    frac_clipped = (ref == 0).float().mean().item()
    print(f"  check 1 (exact integer input): max_abs vs fp64 = {e1:.6e}  "
          f"(relu-clipped fraction {frac_clipped:.3f}, want 0.000)")
    ok &= (e1 == 0.0) and (frac_clipped == 0.0)

    # ---- check 2: random fp16 vs fp64 -------------------------------------
    g = torch.Generator(device=dev)
    worst = 0.0
    for seed in range(4):
        g.manual_seed(seed)
        A = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g) * 0.5
        B = torch.randn(N, K, device=dev, dtype=torch.float16, generator=g) * 0.5
        bias = torch.full((M,), 100.0, device=dev)
        a_cm = A.t().contiguous()
        b_cm = B.contiguous()
        d_cm = torch.zeros(N, M, dtype=torch.float32, device=dev)
        ext.stock_gemm(a_cm, b_cm, bias, d_cm, workspace)
        D = d_cm.t()
        ref = torch.clamp(
            A.double() @ B.double().t() + bias.double().view(M, 1), min=0.0)
        worst = max(worst, (D.double() - ref).abs().max().item())
    print(f"  check 2 (random fp16, 4 seeds): worst max_abs vs fp64 = "
          f"{worst:.6e}  (fp16-storage/fp32-accum floor, want < 1e-1)")
    ok &= torch.isfinite(D).all().item() and worst < 1e-1

    print("\n" + "=" * 78)
    print("PHASE 0 VERDICT:",
          "PASS -- CUTLASS compiles and runs on this toolchain; proceed to "
          "Phase 1" if ok else "FAIL at gate 0b -- compiles but numerics are "
          "wrong; do NOT proceed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
