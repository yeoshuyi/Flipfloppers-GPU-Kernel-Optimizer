"""
G4.0 Phase-1, FFN block -- can the GELU between the two FFN GEMMs be absorbed
into the ffn_in GEMM's own launch (a cuBLASLt epilogue), deleting 6 of the 62
kernel launches per forward at TINY?

This is the ONLY form of "fuse the cheap surrounding elementwise ops around the
FFN GEMMs" that is physically available:
  * step 19 already built and rejected the monolithic Triton ffn_in+GELU+ffn_out
    fusion (0.180x at M=1024), so the GEMMs stay on cuBLASLt;
  * a Triton kernel cannot fuse across an opaque cuBLASLt launch, so the only
    place the GELU can go is inside the GEMM kernel, i.e. its epilogue.

Three questions, in the order that can kill the idea:

  Q1 AVAILABILITY. Does cuBLASLt's heuristic return candidates at all for
     CUBLASLT_EPILOGUE_GELU_BIAS at this shape, and does it still return the
     split-K variants that step 33's shipped 1.32-1.49x win depends on? If the
     GELU epilogue is incompatible with split-K, integrating it would trade a
     measured 1.32x GEMM win for a 1.22us launch -- a losing trade, and the
     probe must be able to see that.

  Q2 NUMERICS. Which GELU is it? The model uses F.gelu(approximate="none")
     (erf). cuBLAS does not document the form. Compared here against BOTH the
     erf and the tanh spelling in float64, so the answer is a measurement, not
     an assumption (step 30's lesson).

  Q3 SPEED. Measured under CUDA-graph replay only (step 34's lesson: at M=64
     these kernels sit under the Python dispatch floor, so an eager loop
     measures the harness, not the kernel).
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WS = 32 * 1024 * 1024
REQ = 16

# tiny: tok=64. The two FFN GEMM shapes, exactly as benchmark.py builds them.
SHAPES = [
    ("ffn_in ", 64, 2048, 512),    # (name, M, N, K)
    ("ffn_out", 64, 512, 2048),
]


def build_ext():
    from torch.utils.cpp_extension import load
    src = "/work/csrc/cublaslt_gelu.cpp"
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", "/work/.ext_build")
    os.makedirs(bd, exist_ok=True)
    os.environ.setdefault("TMPDIR", bd)
    return load(name="cublaslt_gelu", sources=[src], build_directory=bd,
                with_cuda=True, extra_ldflags=["-lcublasLt"], verbose=False)


def graph_time(fn, reps=50, replays=200, best_of=5):
    """Per-call us, measured with `reps` calls captured in one CUDA graph and
    the graph replayed `replays` times. Zero per-call CPU dispatch, which is
    what the real model pays under torch.compile(mode='reduce-overhead')."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(best_of):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / replays / reps * 1000.0)
    return best


def pick_best(ext, pid, inp, w, b, out, warmup=5, iters=30):
    best = None
    for i in range(ext.num_algos(pid)):
        try:
            t = ext.time_algo(pid, i, inp, w, b, out, warmup, iters)
        except Exception:                                        # noqa: BLE001
            continue
        if best is None or t < best[1]:
            best = (i, t)
    return best


def main() -> int:
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    ext = build_ext()
    print("ext built ok\n")

    torch.manual_seed(0)

    # ---------------- Q1 + Q2 on ffn_in (the only GEMM a GELU follows) -----
    M, N, K = 64, 2048, 512
    inp = torch.randn(M, K, device=DEV, dtype=torch.float32)
    w = torch.randn(N, K, device=DEV, dtype=torch.float32) * (K ** -0.5)
    b = torch.randn(N, device=DEV, dtype=torch.float32) * 0.1
    out = torch.empty(M, N, device=DEV, dtype=torch.float32)

    print("=== Q1: candidate availability, ffn_in M=64 N=2048 K=512 ===")
    pids = {}
    for label, epi in (("BIAS (shipped G6.6)", 1), ("GELU_BIAS", 2)):
        pid = ext.create_problem(M, N, K, True, WS, REQ, epi)
        pids[epi] = pid
        n = ext.num_algos(pid)
        print(f"  epi={epi} {label:22s}: {n} candidates")
        for i in range(n):
            print(f"      [{i:2d}] {ext.algo_info(pid, i)}")
        print()

    # ---------------- Q2: which GELU? -------------------------------------
    print("=== Q2: which GELU does CUBLASLT_EPILOGUE_GELU_BIAS compute? ===")
    pid_bias, pid_gelu = pids[1], pids[2]
    bb = pick_best(ext, pid_bias, inp, w, b, out)
    bg = pick_best(ext, pid_gelu, inp, w, b, out)
    print(f"  best BIAS candidate      idx={bb[0]}  {bb[1] * 1e3:.3f} us (eager)")
    print(f"  best GELU_BIAS candidate idx={bg[0]}  {bg[1] * 1e3:.3f} us (eager)")

    lin = torch.empty(M, N, device=DEV, dtype=torch.float32)
    ext.run(pid_bias, bb[0], inp, w, b, lin)
    fused = torch.empty(M, N, device=DEV, dtype=torch.float32)
    ext.run(pid_gelu, bg[0], inp, w, b, fused)
    torch.cuda.synchronize()

    ref_erf = F.gelu(lin, approximate="none")
    ref_tanh = F.gelu(lin, approximate="tanh")
    # float64 ground truth of both spellings, from the SAME pre-activation, so
    # the only thing being compared is the nonlinearity.
    l64 = lin.double()
    erf64 = l64 * 0.5 * (1.0 + torch.erf(l64 / math.sqrt(2.0)))
    tanh64 = l64 * 0.5 * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (l64 + 0.044715 * l64.pow(3))))

    def md(a, c):
        return (a.double() - c.double()).abs().max().item()

    print(f"  |fused - F.gelu(erf) |      = {md(fused, ref_erf):.6e}")
    print(f"  |fused - F.gelu(tanh)|      = {md(fused, ref_tanh):.6e}")
    print(f"  |fused - erf   (fp64)|      = {md(fused, erf64):.6e}")
    print(f"  |fused - tanh  (fp64)|      = {md(fused, tanh64):.6e}")
    print(f"  |F.gelu(erf) - erf  (fp64)| = {md(ref_erf, erf64):.6e}  <- torch's own error")
    print(f"  |erf(fp64) - tanh(fp64)|    = {md(erf64, tanh64):.6e}  <- the two forms differ by this much")
    print()

    # Same question, but on the ACTUAL pre-activation the model sees. The
    # magnitudes above are from randn inputs; if the two GELU forms are
    # indistinguishable only because the activations are small, that must be
    # visible rather than hidden.
    print(f"  pre-activation stats: absmax={lin.abs().max().item():.4f} "
          f"std={lin.std().item():.4f}")
    print()

    # ---------------- Q3: does it actually save time? ---------------------
    print("=== Q3: CUDA-graph timing, one ffn_in GEMM + its GELU ===")
    hidden = torch.empty(M, N, device=DEV, dtype=torch.float32)

    def unfused():
        ext.run(pid_bias, bb[0], inp, w, b, hidden)
        torch.nn.functional.gelu(hidden, approximate="none")

    def fused_fn():
        ext.run(pid_gelu, bg[0], inp, w, b, hidden)

    def gemm_only():
        ext.run(pid_bias, bb[0], inp, w, b, hidden)

    t_un = graph_time(unfused)
    t_fu = graph_time(fused_fn)
    t_gm = graph_time(gemm_only)
    print(f"  BIAS gemm alone            : {t_gm:8.3f} us")
    print(f"  BIAS gemm + separate GELU  : {t_un:8.3f} us   (GELU costs {t_un - t_gm:.3f} us)")
    print(f"  GELU_BIAS fused gemm       : {t_fu:8.3f} us")
    print(f"  fused vs unfused           : {t_un / t_fu:8.4f}x  "
          f"(saving {t_un - t_fu:.3f} us per layer, x6 layers = "
          f"{(t_un - t_fu) * 6:.2f} us/forward)")
    print()

    # ---------------- ffn_out, for completeness ---------------------------
    print("=== ffn_out M=64 N=512 K=2048: candidate set (no GELU follows it) ===")
    pid_o = ext.create_problem(64, 512, 2048, True, WS, REQ, 1)
    print(f"  BIAS: {ext.num_algos(pid_o)} candidates")
    for i in range(ext.num_algos(pid_o)):
        print(f"      [{i:2d}] {ext.algo_info(pid_o, i)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
