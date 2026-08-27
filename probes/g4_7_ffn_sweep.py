#!/usr/bin/env python3
"""
G4.7 Stage 0b -- what is the FFN-in prize actually worth, per arm?

Gated behind probes/g4_7_ffn_gelu_correctness.py (job 131,
results/g4_7_ffn_correct_run131.log): every new config is exact on integer
inputs, the fused GELU is BIT-IDENTICAL to F.gelu(approximate="none") at 39
(config, shape) points, and the shipped cfg[0] is still bitwise equal to
g4_4_mma_gemm cfg[11].

WHAT IS BEING PRICED, AND WHY IT NEEDS THREE BASELINES, NOT ONE.

benchmark.py's ffn_in is (see benchmark.py:989-992 and :1096-1099):

    ffn_hidden_fp16 = F.linear(n2_fp16, w16, b16)          # cuBLASLt GEMM
    ffn_hidden      = ffn_hidden_fp16.to(torch.float32)    # }  one fused
    act             = F.gelu(ffn_hidden, approximate="none")  # }  inductor kernel

so the thing a fused kernel replaces is NOT just the GEMM -- it is the GEMM
PLUS a full elementwise pass over the largest tensor in the model
([tok, ffn_dim], read fp16 + written fp32). Measuring against the GEMM alone
would understate the prize; measuring against two unfused eager kernels would
overstate it. So the reference here is the GEMM plus a torch.compile'd
cast+GELU, i.e. what inductor actually emits inside the compiled region.

Arms:
  ref_gemm    torch.addmm fp16  (GEMM only -- the step-41 comparison)
  ref_chain   torch.addmm fp16 + compiled (cast->gelu)   <- THE REAL BASELINE
  ws_plain    warp-spec GEMM, fp16 out, no epilogue      (still needs the
              cast+gelu pass afterwards, so it is charged for it)
  ws_gelu     warp-spec GEMM + fused exact-erf GELU, fp32 out  <- one kernel
  and each of those split by fp16-accumulate vs ACCF32 (fp32-accumulate),
  because the judge harness runs CAUSAL shapes only and step 41 closed FP16
  ACCUMULATION on causal accuracy. If ACCF32+GELU wins, it wins with ZERO
  numerics change.

Protocol is step 34/37/41's: CUDA-graph replay, best-of-5.
"""
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")

# fp16-accumulate arms
F16_GELU = [51, 52, 55, 56, 59, 60, 63, 64]
F16_PLAIN = [26, 48, 39, 67, 37, 68, 33, 69]
# fp32-accumulate arms (precision-neutral)
F32_GELU = [54, 58, 62, 66, 71, 73, 74, 75, 76]
F32_PLAIN = [53, 57, 61, 65, 72]

# (label, M, K, N) -- ffn_in only. ffn_out is NOT a candidate: it consumes the
# fp32 activation and feeds the residual in fp32, and docs/PROGRESS.md step 27
# (G6.4a v1) closed fp16 for it on accuracy at every shape.
CASES = [
    # benchmark.py's own sweep config: d_model 512, ffn_dim 2048, 6 layers
    ("d512  default    tok1024",   1024,  512, 2048),
    ("d512  long_seq   tok8192",   8192,  512, 2048),
    ("d512  large_batch t32768",  32768,  512, 2048),
    # CLAUDE.md's official causal matrix: qkv dim 128, ffn_dim 128
    ("d128  #1  B64    tok8192",   8192,  128,  128),
    ("d128  #13 S1024  tok65536",  65536, 128,  128),
    # official matrix shape 8: qkv dim 1024, ffn_dim 1024
    ("d1024 #8  B64    tok8192",   8192, 1024, 1024),
]

PEAK_F16ACC = 330.3


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name="g4_7_ffn_gelu", sources=[WS_CPP, WS_CU],
                build_directory=bd, with_cuda=True,
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
    best = None
    for _ in range(repeats):
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best or 1e30, e0.elapsed_time(e1) / replays / iters * 1000.0)
    return best


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    ws = build()
    print("\nconfigs under test:")
    for c in sorted(set(F16_GELU + F16_PLAIN + F32_GELU + F32_PLAIN)):
        print(f"  [{c:2d}] outf32={ws.cfg_outf32(c)}  {ws.cfg_name(c)}")

    # inductor's cast+gelu, exactly as it appears inside the compiled region
    act_compiled = torch.compile(
        lambda t, o: o.copy_(F.gelu(t.float(), approximate="none")),
        dynamic=False)

    g = torch.Generator(device=dev)
    print("\n" + "=" * 120, flush=True)

    for (label, M, K, N) in CASES:
        g.manual_seed((M * 31 + K * 7 + N) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.05).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.05).half()
        o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        o32 = torch.empty(M, N, device=dev, dtype=torch.float32)

        if M * N >= 1 << 26:
            iters, replays = 3, 12
        elif M >= 8192:
            iters, replays = 10, 40
        else:
            iters, replays = 50, 200

        # warm the compiled activation up OUTSIDE the graph capture
        for _ in range(5):
            act_compiled(o16, o32)
        torch.cuda.synchronize()

        t_gemm = graph_time(lambda: torch.addmm(b, inp, w.t(), out=o16),
                            iters, replays)
        t_act = graph_time(lambda: act_compiled(o16, o32), iters, replays)

        def chain():
            torch.addmm(b, inp, w.t(), out=o16)
            act_compiled(o16, o32)

        t_chain = graph_time(chain, iters, replays)
        flop = 2.0 * M * K * N
        print(f"\n### {label}   M={M} K={K} N={N}   "
              f"({flop / 1e9:.2f} GFLOP, out {M * N * 4 / 2**20:.1f} MB fp32)")
        print(f"  BASELINE  addmm fp16              {t_gemm:9.3f} us"
              f"   ({flop / (t_gemm * 1e-6) / 1e12:6.2f} TF)")
        print(f"  BASELINE  compiled cast+gelu      {t_act:9.3f} us")
        print(f"  BASELINE  chain (what we replace) {t_chain:9.3f} us  <== the "
              f"number every fused arm is judged against")

        rows = []
        for tag, cfgs, fused in (("fp16acc  fused-gelu", F16_GELU, True),
                                 ("fp16acc  gemm-only ", F16_PLAIN, False),
                                 ("ACCF32   fused-gelu", F32_GELU, True),
                                 ("ACCF32   gemm-only ", F32_PLAIN, False)):
            for c in cfgs:
                out = o32 if ws.cfg_outf32(c) else o16
                try:
                    ws.ws_gemm(c, inp, w, b, out)
                    torch.cuda.synchronize()
                except Exception:
                    continue
                if fused:
                    t = graph_time(lambda c=c: ws.ws_gemm(c, inp, w, b, o32),
                                   iters, replays)
                    tot = t
                else:
                    t = graph_time(lambda c=c: ws.ws_gemm(c, inp, w, b, o16),
                                   iters, replays)

                    def ch(c=c):
                        ws.ws_gemm(c, inp, w, b, o16)
                        act_compiled(o16, o32)

                    tot = graph_time(ch, iters, replays)
                rows.append((tag, c, t, tot, t_chain / tot))
        rows.sort(key=lambda r: r[3])
        print(f"  {'arm':22s} {'cfg':>4s} {'kernel us':>10s} "
              f"{'total us':>9s} {'vs chain':>9s}")
        for (tag, c, t, tot, sp) in rows:
            print(f"  {tag:22s} {c:4d} {t:10.3f} {tot:9.3f} "
                  f"{'x%.3f' % sp:>9s}")
        del inp, w, b, o16, o32
        torch.cuda.empty_cache()

    print("\n" + "=" * 120)
    print("WHOLE-MODEL DILUTION -- ffn_in is 1 of 4 GEMMs per layer; 6 layers "
          "at d512. Optimized forward medians from "
          "results/g4_3_ship_verify_final_run130.log: long_seq 4.3875 ms, "
          "large_batch 16.2181 ms, default 0.5571 ms.")


if __name__ == "__main__":
    main()
