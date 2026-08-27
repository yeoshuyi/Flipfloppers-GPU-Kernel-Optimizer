#!/usr/bin/env python3
"""
G4.3 Stage 0b -- throughput sweep for csrc/g4_4_warpspec_gemm.cu.

Only run after probes/g4_3_warpspec_correctness.py returns PASS.

MEASUREMENT PROTOCOL is byte-for-byte the one docs/PROGRESS.md step 34
mandated and step 37 used (probes/g4_4_mma_gemm_stage0c.py): CUDA-graph replay,
best-of-5 repeats, cross-checked against torch.profiler kernel time.

WHAT MAKES THIS RUN COMPARABLE TO run98. Both baselines are re-measured IN
THIS SAME JOB:
  * pytorch F.linear (cuBLASLt)          -- the gate's denominator
  * g4_4_mma_gemm cfg[11]                -- step 37's champion, 283.29 us
so no number here depends on a cross-job comparison. warpspec cfg[0] is a
bitwise-identical reimplementation of g4_4 cfg[11] inside the new file
(verified in Stage 0a), so cfg[0] vs mma cfg[11] is also a live control on
whether the new translation unit changed codegen.

THE ISOLATION LADDER (see the WS_CFG_LIST comments in the .cu):
  cfg[0]      = step 37's cfg[11], replicated                  <- control
  cfg[1..3]   = + fat 64x64 consumer warp tiles ONLY
  cfg[4..5]   = + register double-buffered fragments ONLY
  cfg[6..7]   = + smem-staged 128-bit epilogue ONLY
  cfg[8..9]   = all three, still not warp-specialised
  cfg[10..17] = G4.3 warp specialisation on top
  cfg[18..21] = bigger block tiles that keep 8 warps at 64x64
  cfg[22..23] = 128x64 tile -> 4 pipeline stages (MEGAKERNEL "stages 2->4")
  cfg[24..25] = winner + the step-37 SPLIT numerics carry
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_CU = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cu")
WS_CPP = os.path.join(ROOT, "csrc", "g4_4_warpspec_gemm.cpp")
MM_CU = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cu")
MM_CPP = os.path.join(ROOT, "csrc", "g4_4_mma_gemm.cpp")

D = 512
QKV_N = 3 * D

CASES = [
    ("qkv      large_batch", 32768, D, QKV_N, True),    # PRIMARY
    ("qkv      default    ",  1024, D, QKV_N, True),    # secondary
    ("out_proj large_batch", 32768, D, D,     False),
]

PEAK_F32ACC = 165.2
PEAK_F16ACC = 330.3
HBM_GBS = 1008.0
GATE = 1.30


def build(name, cpp, cu):
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(name=name, sources=[cpp, cu], build_directory=build_dir,
                with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-Xptxas", "-v", "-diag-suppress", "179"],
                verbose=True)


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
        t = e0.elapsed_time(e1) / replays / iters * 1000.0
        best = t if best is None else min(best, t)
    return best


def prof_kernel_us(call, launches=50):
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
                                  "ampere", "sm80", "sm89", "ws_gemm")):
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
    ws = build("g4_3_warpspec", WS_CPP, WS_CU)
    mm = build("g4_4_mma_gemm_v3", MM_CPP, MM_CU)
    ncfg = ws.num_cfg()
    print("\nwarpspec configs:")
    for c in range(ncfg):
        print(f"  [{c:2d}] {ws.cfg_name(c)}")

    g = torch.Generator(device=dev)
    verdicts = []

    for name, M, K, N, judged in CASES:
        print("\n" + "=" * 100)
        print(f"=== {name}  M={M} K={K} N={N}   "
              f"{'PRIMARY/JUDGED' if judged else 'reported, not judged'}")
        flop = 2.0 * M * K * N
        bytes_min = (M * K + N * K + M * N) * 2.0
        print(f"    {flop/1e9:.2f} GFLOP, >= {bytes_min/1e6:.1f} MB compulsory "
              f"HBM traffic; tier ceilings FP32-acc {PEAK_F32ACC} TF / "
              f"FP16-acc {PEAK_F16ACC} TF")

        g.manual_seed(hash((M, K, N)) & 0x7FFFFFFF)
        inp = torch.randn(M, K, device=dev, dtype=torch.float16, generator=g)
        w = (torch.randn(N, K, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        b = (torch.randn(N, device=dev, dtype=torch.float16,
                         generator=g) * 0.02).half()
        out = torch.empty(M, N, device=dev, dtype=torch.float16)
        ref_out = torch.empty(M, N, device=dev, dtype=torch.float16)

        if M >= 32768:
            iters, replays = 5, 20
        elif M >= 8192:
            iters, replays = 10, 40
        else:
            iters, replays = 50, 200

        def ref_call():
            torch.addmm(b, inp, w.t(), out=ref_out)

        t_cp = graph_time(lambda: ref_out.copy_(out), iters, replays)
        bw = 2.0 * M * N * 2 / (t_cp * 1e-6) / 1e9
        print(f"    achievable HBM (fp16 copy of [M,N]): {t_cp:.3f} us "
              f"= {bw:.0f} GB/s -> compulsory-traffic floor "
              f"{bytes_min/bw/1e3:.2f} us "
              f"({flop/(bytes_min/bw/1e9)/1e12:.1f} TF hard ceiling)")

        t_ref = graph_time(ref_call, iters, replays)
        t_ref2 = graph_time(ref_call, iters, replays)
        tf_ref = tflops(M, K, N, t_ref)
        print(f"    (pytorch repeatability: {t_ref:.3f} / {t_ref2:.3f} us, "
              f"spread {abs(t_ref2-t_ref)/t_ref*100:.2f}%)")
        print(f"\n    pytorch F.linear(addmm) {t_ref:9.3f} us  {tf_ref:7.2f} TF"
              f"  = {tf_ref/PEAK_F32ACC*100:5.1f}% of the FP32-acc tier "
              f"   [THE FLOOR TO BEAT, gate = x{GATE}]")

        # step 37's champion, re-measured in THIS job
        t_37 = graph_time(lambda: mm.mma_gemm(11, inp, w, b, out),
                          iters, replays)
        print(f"    step-37 g4_4 cfg[11]    {t_37:9.3f} us  "
              f"{tflops(M,K,N,t_37):7.2f} TF  x{t_ref/t_37:.3f}  "
              f"({tflops(M,K,N,t_37)/PEAK_F16ACC*100:4.1f}% of its tier)"
              f"   [THE INCUMBENT]")
        print(flush=True)

        times = {}
        for c in range(ncfg):
            call = lambda c=c: ws.ws_gemm(c, inp, w, b, out)
            try:
                call()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"    cfg[{c:2d}] unavailable: {str(e)[:110]}")
                continue
            if not bool(torch.isfinite(out).all()) or out.float().std() < 1e-4:
                print(f"    cfg[{c:2d}] DEGENERATE output, skipped")
                continue
            t = graph_time(call, iters, replays)
            times[c] = t
            tf = tflops(M, K, N, t)
            print(f"    cfg[{c:2d}] {ws.cfg_name(c):78s} {t:9.3f} us "
                  f"{tf:7.2f} TF  x{t_ref/t:.3f} vs torch  "
                  f"x{t_37/t:.3f} vs step37  ({tf/PEAK_F16ACC*100:4.1f}% tier)",
                  flush=True)

        if not times:
            continue
        bc = min(times, key=times.get)
        bt = times[bc]
        sp = t_ref / bt
        print(f"\n    BEST warpspec cfg[{bc}]  {bt:.3f} us  "
              f"{tflops(M,K,N,bt):.2f} TF  x{sp:.3f} vs pytorch, "
              f"x{t_37/bt:.3f} vs step 37, "
              f"{tflops(M,K,N,bt)/PEAK_F16ACC*100:.1f}% of the FP16-acc tier")
        bt2 = graph_time(lambda: ws.ws_gemm(bc, inp, w, b, out),
                         iters, replays)
        print(f"    REPEATABILITY of the best cfg: {bt:.3f} / {bt2:.3f} us "
              f"-> x{t_ref/bt:.3f} / x{t_ref2/bt2:.3f}")

        pr, pn = prof_kernel_us(ref_call, 50)
        pm, pmn = prof_kernel_us(lambda: ws.ws_gemm(bc, inp, w, b, out), 50)
        print(f"    [prof] pytorch {pr:9.3f} us  {pn[:1]}")
        print(f"    [prof] warpspec{pm:9.3f} us  {pmn[:1]}")
        if pm > 0:
            print(f"    [prof] ratio x{pr/pm:.3f}  (graph x{sp:.3f}, agree to "
                  f"{abs(pr/pm - sp)/sp*100:.1f}%)")

        # --- the isolation table -------------------------------------------
        print("\n    --- LEVER ISOLATION (each row changes ONE thing vs the "
              "row it cites) ---")

        def pr2(label, a, bb):
            if a in times and bb in times:
                print(f"      {label:56s} cfg[{a:2d}] {times[a]:8.3f} us -> "
                      f"cfg[{bb:2d}] {times[bb]:8.3f} us   x{times[a]/times[bb]:.3f}")
        pr2("fat 64x64 consumer warp tile (2x4 -> 2x2)", 0, 1)
        pr2("  ... at 3 stages instead of 2", 1, 2)
        pr2("  ... 1x4 warp grid instead (32x32 tile)", 0, 3)
        pr2("register double-buffered fragments, 2x4 grid", 0, 4)
        pr2("register double-buffered fragments, 2x2 grid", 1, 5)
        pr2("smem-staged 128-bit epilogue, 2x4 grid", 0, 6)
        pr2("smem-staged 128-bit epilogue, 2x2 grid", 1, 7)
        pr2("all three levers stacked (no warp spec)", 0, 8)
        pr2("WARP SPEC on top of all three (+2 idle roles)", 8, 10)
        pr2("WARP SPEC on top of all three (NEXTRA=0)", 8, 12)
        pr2("  cost of the 2 idle Storer/Controller warps", 12, 10)
        pr2("  3 loader warps instead of 2", 12, 14)
        pr2("  warp spec WITHOUT regdb", 6, 16)
        pr2("bigger tile, 8 warps at 64x64: 256x128", 8, 18)
        pr2("bigger tile, 8 warps at 64x64: 128x256", 8, 19)
        pr2("4 pipeline stages (128x64 tile)", 8, 22)
        pr2("SPLIT=256 numerics carry on the best plain cfg", 8, 24)

        if judged:
            met = sp >= GATE
            print(f"\n    >>> GATE (x{GATE}): {'MET' if met else 'NOT MET'} "
                  f"(x{sp:.3f} on cfg[{bc}]; step 37 was x{t_ref/t_37:.3f})")
            verdicts.append((name, bc, sp, met))

    print("\n" + "=" * 100)
    print("VERDICT")
    for nm, bc, sp, met in verdicts:
        print(f"  {nm}  best cfg[{bc}]  x{sp:.3f}  gate x{GATE} "
              f"{'MET' if met else 'NOT MET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
