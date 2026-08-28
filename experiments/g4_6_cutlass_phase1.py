#!/usr/bin/env python3
"""
G4.6 Phase 1 -- raw throughput of CUTLASS TensorOp FP16 GEMM at the project's
real attention-GEMM shapes, in BOTH accumulate tiers.

CONTEXT (docs/PROGRESS.md step 37): the hand-written mma.sync FP16-accumulate
kernel csrc/g4_4_mma_gemm.cu reaches 181.93 TFLOPS at qkv/large_batch = 55.1%
of the 330.3 TF FP16-accumulate tier ceiling, while cuBLASLt reaches 150.69 TF
= 91.2% of its own (2x lower) 165.2 TF FP32-accumulate tier ceiling.  Step 37
named "~80% of the FP16-accumulate tier" (~264 TF) as the threshold that would
reopen this line of work.  That is this probe's KILL GATE -- not an arbitrary
number.

Step 34's measurement protocol is mandatory here: CUDA-graph replay
cross-checked against torch.profiler kernel time.  A naive host-loop timer has
already produced a false 1.96x at these exact shapes once this session.

The FP32-accumulate twins (cfg 10-13) share tile, warp shape, stages, epilogue
and layout with their FP16 counterparts -- ONLY ElementAccumulator differs --
so any win is attributable specifically to the accumulate tier (step 35's
finding-3 / step 37's own A/B discipline).  Step 37 measured that tier at
x1.43-x1.54 inside its own kernel; a CUTLASS ratio wildly different from that
is a red flag to double-check before trusting, not a result.
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_gemm.cpp")
NUM_CFG = 24
SRC_CU = [os.path.join(ROOT, "csrc", f"g4_6_cutlass_cfg{i:02d}.cu")
          for i in range(NUM_CFG)]

D = 512
QKV_N = 3 * D

CASES = [
    ("qkv      large_batch", 32768, D, QKV_N, True),    # PRIMARY judgment
    ("out_proj large_batch", 32768, D, D,     False),
    ("qkv      default    ",  1024, D, QKV_N, False),   # secondary
]

# accF16 cfg -> its accF32 twin (identical everything else)
TWIN = {0: 10, 1: 11, 2: 12, 4: 13}

PEAK_F32ACC = 165.2     # fp16 storage, FP32 accumulate (GeForce Ada half rate)
PEAK_F16ACC = 330.3     # fp16 storage, FP16 accumulate  <- the tier under test
GATE_FRAC = 0.80        # step 37's own reopening threshold
HBM_GBS = 1008.0

# step 37's measured numbers at qkv/large_batch, for direct comparison
G44_BEST_US = 283.29
G44_BEST_TF = 181.93


def build():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    t0 = time.time()
    mod = load(
        name="g4_6_cutlass_gemm",
        sources=[SRC_CPP] + SRC_CU,
        build_directory=build_dir,
        with_cuda=True,
        extra_include_paths=[os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools", "util", "include")],
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "--expt-relaxed-constexpr",
                           "--expt-extended-lambda",
                           "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                           "-Xptxas", "-v"],
        verbose=True,
    )
    return mod, time.time() - t0


def graph_time(call, iters, replays, repeats=5):
    """CUDA-graph replay timing (step 34's protocol A). Identical to
    probes/g4_4_mma_gemm_stage0c.py's, deliberately, so the two probes'
    numbers are directly comparable."""
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
        t = e0.elapsed_time(e1) / replays / iters * 1000.0
        best = t if best is None else min(best, t)
    return best


def prof_kernel_us(call, launches=50):
    """torch.profiler self_device_time (step 34's protocol B)."""
    from torch.profiler import profile, ProfilerActivity
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(launches):
            call()
        torch.cuda.synchronize()
    tot, names = 0.0, []
    for ev in prof.key_averages():
        if ev.self_device_time_total <= 0:
            continue
        low = ev.key.lower()
        if any(t in low for t in ("gemm", "cutlass", "wmma", "mma_", "xmma",
                                  "ampere", "sm80", "sm89", "kernel")):
            tot += ev.self_device_time_total
            names.append(ev.key.split("(")[0][:70])
    return tot / launches, names


def tflops(M, K, N, us):
    return 2.0 * M * K * N / (us * 1e-6) / 1e12


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda}", flush=True)

    try:
        ext, secs = build()
    except Exception as e:  # noqa: BLE001
        print("PHASE 1 BUILD FAILED")
        print(str(e)[-8000:])
        return 1
    print(f"\nbuilt {NUM_CFG} single-instantiation TUs + 1 wrapper in "
          f"{secs:.1f}s ({secs/60:.2f} min)", flush=True)

    ncfg = ext.num_cfg()
    print("\nconfigs (Sm80 tag, OpClassTensorOp, mma 16x8x16, "
          "LinearCombination/NoBetaScaling bias epilogue):")
    for c in range(ncfg):
        print(f"  [{c:2d}] {ext.cfg_name(c)}")

    g = torch.Generator(device=dev)
    gate_result = {}
    summary = []

    for name, M, K, N, judged in CASES:
        print("\n" + "=" * 78)
        print(f"=== {name}  M={M} K={K} N={N}   "
              f"{'PRIMARY/JUDGED' if judged else 'reported, not judged'}")
        flop = 2.0 * M * K * N
        bytes_min = (M * K + N * K + M * N) * 2.0
        print(f"    {flop/1e9:.2f} GFLOP, >= {bytes_min/1e6:.1f} MB compulsory "
              f"HBM traffic")
        print(f"    tier ceilings: FP32-acc {PEAK_F32ACC} TF, FP16-acc "
              f"{PEAK_F16ACC} TF; gate = {GATE_FRAC*100:.0f}% of FP16-acc = "
              f"{PEAK_F16ACC*GATE_FRAC:.1f} TF")

        g.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ref_out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ws = torch.empty(1, dtype=torch.uint8, device=dev)

        if M >= 32768:
            iters, replays = 5, 20
        elif M >= 8192:
            iters, replays = 10, 40
        else:
            iters, replays = 50, 200

        def ref_call():
            torch.addmm(b, inp, w.t(), out=ref_out)

        # Achievable-HBM reference, same harness (step 37's probe): a pure fp16
        # copy of the [M,N] output. This bounds how much of the kernel time is
        # unavoidable memory movement rather than mma issue, and is what makes
        # the "is 80% of tier even reachable here?" question answerable with a
        # measurement instead of an assertion.
        t_cp = graph_time(lambda: ref_out.copy_(out), iters, replays)
        bw = 2.0 * M * N * 2 / (t_cp * 1e-6) / 1e9
        floor_us = bytes_min / bw / 1e3
        print(f"    achievable HBM (fp16 copy of the [M,N] output): "
              f"{t_cp:.3f} us for {2*M*N*2/1e6:.1f} MB = {bw:.0f} GB/s")
        print(f"    -> compulsory traffic at that measured rate: "
              f"{floor_us:.2f} us = {flop/(floor_us*1e-6)/1e12:.1f} TF "
              f"HARD CEILING ({flop/(floor_us*1e-6)/1e12/PEAK_F16ACC*100:.1f}% "
              f"of the FP16-acc tier)")

        ref_call()
        torch.cuda.synchronize()
        ref_fp32 = (inp.float() @ w.float().t() + b.float())
        ref_scale = ref_fp32.abs().max().item()

        t_ref = graph_time(ref_call, iters, replays)
        t_ref2 = graph_time(ref_call, iters, replays)
        tf_ref = tflops(M, K, N, t_ref)
        print(f"    (pytorch reference repeatability: {t_ref:.3f} / "
              f"{t_ref2:.3f} us, spread "
              f"{abs(t_ref2-t_ref)/t_ref*100:.2f}%)")
        print(f"\n    pytorch F.linear(addmm) {t_ref:9.3f} us  {tf_ref:7.2f} TF"
              f"   = {tf_ref/PEAK_F32ACC*100:5.1f}% of the FP32-acc tier "
              f"   [THE FLOOR TO BEAT]")
        if M == 32768 and N == QKV_N:
            print(f"    step 37's hand-written cfg[11]: {G44_BEST_US:.2f} us  "
                  f"{G44_BEST_TF:.2f} TF = "
                  f"{G44_BEST_TF/PEAK_F16ACC*100:.1f}% of the FP16-acc tier")

        times = {}
        for c in range(ncfg):
            call = lambda c=c: ext.cutlass_gemm(c, inp, w, b, out, ws)
            try:
                call()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"    cfg[{c:2d}] unavailable: {str(e)[:140]}")
                continue
            if not bool(torch.isfinite(out).all()):
                print(f"    cfg[{c:2d}] NON-FINITE output, skipped")
                continue
            rel = ((out.float() - ref_fp32).abs().max().item() /
                   max(ref_scale, 1e-9))
            if rel > 0.10:
                print(f"    cfg[{c:2d}] WRONG (rel err {rel:.3e} vs fp32 "
                      f"reference), skipped -- not timed")
                continue
            t = graph_time(call, iters, replays)
            times[c] = t
            tf = tflops(M, K, N, t)
            f32 = ext.cfg_name(c).startswith("accF32")
            peak = PEAK_F32ACC if f32 else PEAK_F16ACC
            print(f"    cfg[{c:2d}] {ext.cfg_name(c):40s} {t:9.3f} us "
                  f"{tf:7.2f} TF  x{t_ref/t:.3f}  ({tf/peak*100:4.1f}% of its "
                  f"own tier)  relerr {rel:.2e}")

        f16 = {c: t for c, t in times.items()
               if not ext.cfg_name(c).startswith("accF32")}
        if not f16:
            print("    no usable FP16-accumulate config at this shape")
            continue
        bc = min(f16, key=f16.get)
        bt = f16[bc]
        btf = tflops(M, K, N, bt)
        sp = t_ref / bt
        print(f"\n    BEST FP16-accumulate cfg[{bc}] {ext.cfg_name(bc)}")
        print(f"      {bt:.3f} us   {btf:.2f} TF   x{sp:.3f} vs pytorch   "
              f"{btf/PEAK_F16ACC*100:.1f}% of the FP16-acc tier")

        bt2 = graph_time(lambda: ext.cutlass_gemm(bc, inp, w, b, out, ws),
                         iters, replays)
        print(f"      repeatability: {bt:.3f} / {bt2:.3f} us "
              f"({abs(bt2-bt)/bt*100:.2f}%)")

        pr, pn = prof_kernel_us(ref_call, 50)
        pm, pmn = prof_kernel_us(
            lambda: ext.cutlass_gemm(bc, inp, w, b, out, ws), 50)
        agree = abs(pr / pm - sp) / sp * 100
        print(f"      [prof] pytorch {pr:9.3f} us  {pn[:1]}")
        print(f"      [prof] cutlass {pm:9.3f} us  {pmn[:1]}")
        print(f"      [prof] ratio x{pr/pm:.3f}  vs graph x{sp:.3f}  -> "
              f"agree to {agree:.1f}%  "
              f"({'OK' if agree <= 3.0 else 'DISAGREE >3% -- DO NOT TRUST'})")

        best_f32 = {c: t for c, t in times.items()
                    if ext.cfg_name(c).startswith("accF32")}
        print("\n    --- THE A/B: same tile/warp/stages/epilogue, only "
              "ElementAccumulator changes ---")
        for cf16, cf32 in TWIN.items():
            if cf16 in times and cf32 in times:
                r = times[cf32] / times[cf16]
                print(f"    cfg[{cf16:2d}] accF16 {times[cf16]:9.3f} us  vs  "
                      f"cfg[{cf32:2d}] accF32 {times[cf32]:9.3f} us  -> FP16 "
                      f"accumulate is x{r:.3f}   (step 37 measured "
                      f"x1.43-x1.54 for this tier)")
                summary.append((name, cf16, r))
        if best_f32:
            bf = min(best_f32, key=best_f32.get)
            print(f"    best accF32 overall: cfg[{bf}] {best_f32[bf]:.3f} us "
                  f"{tflops(M,K,N,best_f32[bf]):.2f} TF  x"
                  f"{t_ref/best_f32[bf]:.3f} vs pytorch "
                  f"({tflops(M,K,N,best_f32[bf])/PEAK_F32ACC*100:.1f}% of the "
                  f"FP32-acc tier)")

        if judged:
            met = btf >= PEAK_F16ACC * GATE_FRAC
            gate_result = {"tflops": btf, "us": bt, "cfg": bc,
                           "cfg_name": ext.cfg_name(bc), "speedup": sp,
                           "frac_of_tier": btf / PEAK_F16ACC, "met": met,
                           "prof_ratio": pr / pm, "agree_pct": agree}
            print(f"\n    >>> KILL GATE ({GATE_FRAC*100:.0f}% of the "
                  f"FP16-accumulate tier = {PEAK_F16ACC*GATE_FRAC:.1f} TF): "
                  f"{'MET' if met else 'NOT MET'}")
            print(f"    >>> achieved {btf:.2f} TF = "
                  f"{btf/PEAK_F16ACC*100:.1f}% of tier "
                  f"(step 37's hand-written kernel: 55.1%; "
                  f"cuBLASLt: 91.2% of its own 2x-lower tier)")

    print("\n" + "=" * 78)
    print("A/B SUMMARY -- price of the FP16-accumulate tier inside CUTLASS")
    for nm, c, r in summary:
        print(f"  {nm}  cfg[{c}]  accF32/accF16 = x{r:.3f}")
    print("\n" + "=" * 78)
    if gate_result:
        print(f"PHASE 1 VERDICT at qkv/large_batch: "
              f"{gate_result['tflops']:.2f} TF "
              f"({gate_result['frac_of_tier']*100:.1f}% of the FP16-acc tier), "
              f"x{gate_result['speedup']:.3f} vs PyTorch")
        print("GATE " + ("MET -- proceed to Phase 2 (fp64 correctness + "
                         "whole-model)" if gate_result["met"] else
                         "NOT MET -- STOP, clean negative"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
