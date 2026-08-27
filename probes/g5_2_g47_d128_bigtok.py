"""
G5.2 / iteration 4 (T3) Phase 0 -- does G4.7's fused ffn_in+GELU flip to a WIN
at the memory-bound d128 big-token official rows (6: tok 1.28M, 13: tok 65536)?

Step 43: on the d128 rows the standalone GELU pass + the [tok,128] hidden
round-trip is the largest cost, and the round-trip is measured-exposed. G4.7's
kernel already fuses ffn_in GEMM + exact erf-GELU into one launch (fp32 out),
precision-neutral -- but its gate excludes d_model<512 because run132 measured
it x0.93-0.99 vs (cuBLAS + compiled cast+gelu) at d128 tok 8192 / 65536.
run132 did NOT test tok 1.28M, where both the GEMM and the GELU are deeply
memory-bound and the traffic G4.7 eliminates dominates.

Baseline = what the shipped causal path actually runs: cuBLAS `addmm` fp16 for
ffn_in + a torch.compile'd (fp16->fp32 cast + erf-GELU) -- exactly the
`triton_poi_fused__to_copy_gelu_view_2` kernel inductor emits.

Protocol: CUDA-graph replay, best-of-5.  cfg 58 (ACCF32 fused GELU) is the
shipped G4.7 config; also try 54/73/74 (other ACCF32-fused arms) and 51
(FP16-accum fused, precision-reducing -- reference only).
"""
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")

ACCF32_GELU = [58, 54, 73, 74, 66, 62]     # precision-neutral
F16_GELU = [51, 55]                        # reference only (precision-reducing)

# d128/ffn128 official rows: row1 (control), row13, row6
CASES = [
    ("row1  M8192   K128 N128", 8192,   128, 128),
    ("row13 M65536  K128 N128", 65536,  128, 128),
    ("row6  M1280000 K128 N128", 1280000, 128, 128),
]


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name="g5_2_ws", sources=[WS_CPP, WS_CU], build_directory=bd,
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
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}\n",
          flush=True)
    ws = build()

    act_compiled = torch.compile(
        lambda t, o: o.copy_(F.gelu(t.float(), approximate="none")),
        dynamic=False)

    gen = torch.Generator(device=dev)
    for label, M, K, N in CASES:
        gen.manual_seed((M * 31 + K * 7 + N) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=gen)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=gen) * 0.05).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=gen) * 0.05).half()
        o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        o32 = torch.empty(M, N, device=dev, dtype=torch.float32)

        if M >= 1 << 20:
            iters, replays = 3, 12
        elif M >= 32768:
            iters, replays = 8, 30
        else:
            iters, replays = 40, 160

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
        gib = M * N * 4 / 2**30
        print(f"### {label}   (out {gib:.2f} GiB fp32)")
        print(f"  BASELINE  cuBLAS addmm fp16     {t_gemm:10.3f} us")
        print(f"  BASELINE  compiled cast+GELU    {t_act:10.3f} us")
        print(f"  BASELINE  chain (SHIPPED path)  {t_chain:10.3f} us  <== judged against")

        rows = []
        for tag, cfgs in (("ACCF32-fused (neutral)", ACCF32_GELU),
                          ("FP16-fused (ref only)", F16_GELU)):
            for c in cfgs:
                try:
                    ws.ws_gemm(c, inp, w, b, o32)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    if "-2" in str(exc) or "-3" in str(exc):
                        continue
                    print(f"  cfg{c}: {str(exc)[:80]}")
                    continue
                t = graph_time(lambda c=c: ws.ws_gemm(c, inp, w, b, o32),
                               iters, replays)
                rows.append((tag, c, t, t_chain / t))
        rows.sort(key=lambda r: r[2])
        for tag, c, t, sp in rows:
            flag = ("  <-- NEUTRAL WIN" if tag.startswith("ACCF32") and sp > 1.03
                    else ("  (neutral, ~parity)" if tag.startswith("ACCF32") and sp > 0.98 else ""))
            print(f"  {tag:24s} cfg{c:2d}  {t:10.3f} us   x{sp:.3f}{flag}")
        print()
    print("decision: if an ACCF32-fused cfg is >1.03x the shipped chain at "
          "row 6 and/or row 13, extend _ensure_ffn_plan to admit "
          "(d_model>=128, ffn_dim>=128, tok>=<threshold>) and ship.")


if __name__ == "__main__":
    main()
