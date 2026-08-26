#!/usr/bin/env python3
"""
Stage 1.5(i): capability check before any hand-written kernel work.
Sets the cost model for Stage 2: does torch._scaled_mm support per-row
(RowWise) scales on sm_89, or per-tensor only? Does Triton's tl.dot accept
float8e4nv operands on this GPU/Triton build? If Triton works, Stage 2 is
a few days; if only raw PTX works, it's weeks with no established
cpp_extension build path in this repo.
"""
import torch
import torch.nn.functional as F

device = torch.device("cuda")
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"cc {torch.cuda.get_device_capability()}  gpu {torch.cuda.get_device_name(0)}")

# --- torch._scaled_mm: per-tensor scalar scale (known working, phase0.json) ---
try:
    a = torch.randn(64, 128, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(64, 128, device=device).to(torch.float8_e4m3fn).t()
    s = torch.ones(1, device=device, dtype=torch.float32)
    out = torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
    print(f"scaled_mm per-tensor scalar: OK, out shape {tuple(out.shape)}")
except Exception as e:
    print(f"scaled_mm per-tensor scalar: FAIL {type(e).__name__}: {e}")

# --- torch._scaled_mm: per-row (RowWise) scale tensors ---
try:
    M, K, N = 64, 128, 32
    a = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device=device).to(torch.float8_e4m3fn).t()
    scale_a = torch.ones(M, 1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, N, device=device, dtype=torch.float32)
    out = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    print(f"scaled_mm per-row (RowWise) scale: OK, out shape {tuple(out.shape)}")
except Exception as e:
    print(f"scaled_mm per-row (RowWise) scale: FAIL {type(e).__name__}: {e}")

# --- Triton: does tl.dot accept float8e4nv operands on sm_89? ---
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    M, K, N = 32, 32, 32
    a = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device=device).to(torch.float8_e4m3fn)
    c = torch.empty(M, N, device=device, dtype=torch.float32)
    _fp8_dot_kernel[(1,)](a, b, c, M, N, K, BLOCK_M=M, BLOCK_N=N, BLOCK_K=K)
    torch.cuda.synchronize()
    ref = (a.to(torch.float32) @ b.to(torch.float32))
    err = (c - ref).abs().max().item()
    print(f"triton tl.dot(float8e4nv): OK, max_abs vs fp32 ref = {err:.6f}, "
          f"triton {triton.__version__}")
except Exception as e:
    print(f"triton tl.dot(float8e4nv): FAIL {type(e).__name__}: {e}")
