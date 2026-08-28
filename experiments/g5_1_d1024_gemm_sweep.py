"""
G5.1 / iteration 3 (T2) Phase 0 -- can a FP32-accumulate warp-spec GEMM beat
cuBLAS on the row-8 (d1024) causal GEMMs?

Step 43 found row 8 GEMM-bound at ~56% of the 165 TFLOPS FP16-fp32-accum
roofline (cuBLAS runs `ampere_fp16_s1688gemm_128x128 ... stages_32x1`, a
shallow pipeline). The G4.7 kernel already has an ACCF32 (fp32-accumulate)
arm; here we price the NO-GELU ACCF32 configs against cuBLAS at the exact
row-8 shapes, precision-neutral (fp32 accumulate == cuBLAS on fp16 storage).

Targets (all M=8192, K=1024; causal path, fp16 storage):
  qkv       N=3072   out_proj  N=1024   ffn_in  N=1024
ffn_out stays TF32 (fp32 `act` in) -- a warp-spec fp16 replacement there would
round the hidden and is out of scope for a precision-neutral pass.

Protocol: step 34/37/41's -- CUDA-graph replay, best-of-5.  Plus fp64 accuracy
per config so "precision-neutral" is checked, not assumed.
"""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")

# ACCF32, no GELU (fp16 out) -- from WS_CFG_LIST_G47.
ACCF32_PLAIN = [53, 57, 61, 65, 72]
# FP16-accumulate base configs for context (precision-REDUCING -- reference only)
F16_BASE = [26, 33, 37, 39, 48]

CASES = [
    ("qkv      M8192 K1024 N3072", 8192, 1024, 3072),
    ("out/ffnin M8192 K1024 N1024", 8192, 1024, 1024),
]

PEAK_F32ACC = 165.2  # TFLOPS, FP16 inputs / FP32 accumulate, GeForce Ada


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name="g5_1_ws", sources=[WS_CPP, WS_CU], build_directory=bd,
                with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-diag-suppress", "179"], verbose=False)


def graph_time(call, iters, replays, repeats=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            call()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(repeats):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / replays / iters * 1000.0)
    return best


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    ws = build()
    ncfg = ws.num_cfg()
    print(f"config table: {ncfg} entries\n")

    g = torch.Generator(device=dev)
    for label, M, K, N in CASES:
        g.manual_seed((M * 31 + K * 7 + N) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.03).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.03).half()
        o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        ref64 = (inp.double() @ w.double().T + b.double())

        iters, replays = 10, 40
        flop = 2.0 * M * K * N
        t_cublas = graph_time(lambda: torch.addmm(b, inp, w.t(), out=o16),
                              iters, replays)
        e_cublas = (torch.addmm(b, inp, w.t()).double() - ref64).abs().max().item()
        print(f"### {label}   ({flop/1e9:.2f} GFLOP)")
        print(f"  cuBLAS addmm fp16      {t_cublas:9.3f} us  "
              f"({flop/(t_cublas*1e-6)/1e12:6.1f} TF, "
              f"{flop/(t_cublas*1e-6)/1e12/PEAK_F32ACC*100:4.1f}% of f32-acc peak)"
              f"  err {e_cublas:.3e}")

        rows = []
        for tag, cfgs in (("ACCF32 (neutral)", ACCF32_PLAIN),
                          ("FP16-acc (ref only)", F16_BASE)):
            for c in cfgs:
                out = o16
                try:
                    ws.ws_gemm(c, inp, w, b, out)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    if "-2" in str(exc) or "-3" in str(exc):
                        continue
                    print(f"  cfg{c}: {str(exc)[:80]}")
                    continue
                t = graph_time(lambda c=c: ws.ws_gemm(c, inp, w, b, o16),
                               iters, replays)
                err = (ws.ws_linear(c, inp, w, b).double() - ref64).abs().max().item()
                rows.append((tag, c, t, t_cublas / t, err))
        rows.sort(key=lambda r: r[2])
        print(f"  {'arm':20s} {'cfg':>4s} {'us':>9s} {'vs cuBLAS':>10s} "
              f"{'err vs fp64':>12s}  {ws.cfg_name(rows[0][1])[:1] if rows else ''}")
        for tag, c, t, sp, err in rows:
            flag = "  <-- neutral WIN" if (tag.startswith("ACCF32") and sp > 1.0) else ""
            print(f"  {tag:20s} {c:4d} {t:9.3f} {'x%.3f' % sp:>10s} "
                  f"{err:12.3e}{flag}")
        print()
    print("interpret: an ACCF32 cfg with vs-cuBLAS > 1.03 and err ~= cuBLAS err "
          "is a precision-neutral candidate for qkv/out_proj/ffn_in at d1024.")


if __name__ == "__main__":
    main()
