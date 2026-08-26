#!/usr/bin/env python3
"""Phase 0: establish ground truth on THIS machine."""
import torch, json

r = {"torch": torch.__version__, "cuda": torch.version.cuda,
     "cc": torch.cuda.get_device_capability(), "gpu": torch.cuda.get_device_name(0)}

# --- FP8 library path (gates the whole G4/FP8 plan) --------------------
try:
    a = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(128, 64, device="cuda").to(torch.float8_e4m3fn).t()
    s = torch.ones(1, device="cuda", dtype=torch.float32)
    torch._scaled_mm(a, b, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
    r["fp8_scaled_mm"] = True
except Exception as e:
    r["fp8_scaled_mm"] = f"NO: {type(e).__name__}: {e}"

# --- Triton FP8 cast (needs PTX ISA >= 8.1 on sm_89) -------------------
try:
    import triton, triton.language as tl
    @triton.jit
    def _k(o, i, n, BS: tl.constexpr):
        off = tl.program_id(0) * BS + tl.arange(0, BS)
        tl.store(o + off, tl.load(i + off, mask=off < n).to(tl.float8e4nv),
                 mask=off < n)
    x = torch.randn(1024, device="cuda")
    y = torch.empty(1024, device="cuda", dtype=torch.float8_e4m3fn)
    _k[(1,)](y, x, 1024, BS=1024)
    r["triton_fp8"] = True
except Exception as e:
    r["triton_fp8"] = f"NO: {type(e).__name__}: {e}"

# --- cooperative launch (gates G4.1) -----------------------------------
r["cooperative_launch"] = bool(torch.cuda.get_device_properties(0).__dict__
                               .get("cooperative_launch", True))

# --- hardware limits ----------------------------------------------------
p = torch.cuda.get_device_properties(0)
r.update(sm_count=p.multi_processor_count,
         shared_per_sm_kb=p.shared_memory_per_multiprocessor / 1024,
         l2_mb=p.L2_cache_size / 1e6,
         regs_per_sm=p.regs_per_multiprocessor)

# --- achieved peak per dtype -------------------------------------------
def peak(dtype, n=8192, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(10): a @ b
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): a @ b
    e.record(); torch.cuda.synchronize()
    return 2 * n**3 / (s.elapsed_time(e) / iters * 1e-3) / 1e12

torch.backends.cuda.matmul.allow_tf32 = True
r["tf32_tflops"] = peak(torch.float32)
r["bf16_tflops"] = peak(torch.bfloat16)
r["fp16_tflops"] = peak(torch.float16)

print(json.dumps(r, indent=2))
