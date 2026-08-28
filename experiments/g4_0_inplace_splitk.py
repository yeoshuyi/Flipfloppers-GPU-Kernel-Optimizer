"""
G4.0 Phase-1, the last real launch-deletion lever at TINY.

The census (results/g4_0_census_run86.log) shows 62 kernels per tiny forward.
20 are elementwise; probe g4_0_ceiling.py priced deleting ALL of them at
8.49% (pure launch cost) to 18.07% (an over-estimate that counts the LayerNorm
arithmetic as free), and the two ways to actually delete any of them are both
closed: LayerNorm cannot be a GEMM epilogue (full-row reduction over N), and
cuBLASLt's GELU epilogue is the TANH form, 4.74e-04 away from the model's erf
GELU (probe g4_0_ffn_epilogue_probe.py, run 87).

That leaves a target the earlier passes classified as "GEMM work" and therefore
skipped: `cublasLt::splitKreduce_kernel`, 12 launches / 22.72 us / 11.5% of the
tiny forward.  It is there only because step 33's shipped algorithm choices use
split-K with an OUT-OF-PLACE reduction scheme (reduc=2), which writes partial
sums to workspace and reduces them in a SECOND kernel.  cuBLASLt also offers
CUBLASLT_REDUCTION_SCHEME_INPLACE, where the splits accumulate into the output
by atomics inside the GEMM kernel -- same tiling, same split-K win, one launch
instead of two.

That is precisely G4.0's mechanism (fewer kernel boundaries, unchanged GEMM
tiling) applied to the one place at tiny where launches are still deletable,
and it is reachable without hand-writing a GEMM -- which step 19 measured at
0.180x and steps 33/34 showed is already optimally served by cuBLASLt here.

Measured per candidate:
  * how many kernels it actually launches (profiler, not assumed)
  * time under CUDA-graph replay (step 34's lesson: at M=64 an eager loop
    measures the Python dispatch floor, not the kernel)
  * max |candidate - F.linear| and run-to-run determinism, because atomic
    float accumulation reorders the reduction.
"""

import os
import sys
import json
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WS = 32 * 1024 * 1024
REQ = 16
TOK = 64
SHAPES = [("ffn_in ", TOK, 2048, 512), ("ffn_out", TOK, 512, 2048)]
# NONE=0 INPLACE=1 COMPUTE_TYPE=2 OUTPUT_TYPE=4
MASKS = [("default (shipped)", -1), ("INPLACE only", 1),
         ("NONE+INPLACE", 3), ("all schemes", 7)]


def build_ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", "/work/.ext_build")
    os.makedirs(bd, exist_ok=True)
    os.environ.setdefault("TMPDIR", bd)
    return load(name="cublaslt_gelu", sources=["/work/csrc/cublaslt_gelu.cpp"],
                build_directory=bd, with_cuda=True,
                extra_ldflags=["-lcublasLt"], verbose=False)


def graph_us(fn, reps=40, replays=200, best_of=5):
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
    return best, g


def kernels_of(g, replays=30):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        ev = json.load(fh)["traceEvents"]
    os.unlink(path)
    per = {}
    for e in ev:
        if (e.get("cat") or "").lower() != "kernel":
            continue
        s = per.setdefault(e.get("name", "?")[:44], [0, 0.0])
        s[0] += 1
        s[1] += float(e.get("dur", 0.0))
    return per, replays


def main() -> int:
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    ext = build_ext()
    print("ext built ok\n")
    torch.manual_seed(0)

    for name, M, N, K in SHAPES:
        print(f"================ {name}  M={M} N={N} K={K} ================")
        inp = torch.randn(M, K, device=DEV, dtype=torch.float32)
        w = torch.randn(N, K, device=DEV, dtype=torch.float32) * (K ** -0.5)
        b = torch.randn(N, device=DEV, dtype=torch.float32) * 0.1
        out = torch.empty(M, N, device=DEV, dtype=torch.float32)
        ref = F.linear(inp, w, b)
        ref64 = (inp.double() @ w.double().t() + b.double())
        torch.cuda.synchronize()

        overall = []
        for label, mask in MASKS:
            try:
                pid = ext.create_problem(M, N, K, True, WS, REQ, 1, mask)
            except Exception as exc:                             # noqa: BLE001
                print(f"  mask={mask:2d} {label:18s}: create_problem failed: {exc}")
                continue
            n = ext.num_algos(pid)
            print(f"  mask={mask:2d} {label:18s}: {n} candidates")
            best = None
            for i in range(n):
                info = ext.algo_info(pid, i)
                def call(pid=pid, i=i):
                    ext.run(pid, i, inp, w, b, out)
                try:
                    us, g = graph_us(call)
                except Exception as exc:                         # noqa: BLE001
                    print(f"      [{i:2d}] FAILED: {str(exc)[:70]}")
                    continue
                per, reps = kernels_of(g)
                nk = sum(c for c, _ in per.values()) / reps / 40.0
                names = "+".join(sorted(
                    ("REDUCE" if "splitKreduce" in k
                     else "MEMSET" if "memset" in k.lower() or "Memset" in k
                     else "GEMM")
                    for k in per))
                ext.run(pid, i, inp, w, b, out)
                torch.cuda.synchronize()
                r1 = out.clone()
                ext.run(pid, i, inp, w, b, out)
                torch.cuda.synchronize()
                det = torch.equal(r1, out)
                d_ref = (r1 - ref).abs().max().item()
                d_64 = (r1.double() - ref64).abs().max().item()
                print(f"      [{i:2d}] {us:7.3f} us  {nk:4.2f} kern ({names})  "
                      f"det={'Y' if det else 'N'}  "
                      f"|-F.linear|={d_ref:.3e} |-fp64|={d_64:.3e}  {info}")
                if best is None or us < best[0]:
                    best = (us, i, nk, det, d_ref, d_64, info)
            if best is not None:
                overall.append((label, mask) + best)
            print()

        if overall:
            print(f"  --- best per mask, {name} ---")
            base = None
            for label, mask, us, i, nk, det, d_ref, d_64, info in overall:
                if mask == -1:
                    base = us
            for label, mask, us, i, nk, det, d_ref, d_64, info in overall:
                sp = f"{base / us:.4f}x" if base else "-"
                print(f"    {label:18s} idx={i:2d} {us:7.3f} us  "
                      f"{nk:4.2f} kernels  vs shipped {sp}  det={'Y' if det else 'N'}  "
                      f"|-fp64|={d_64:.3e}")
        print()

        # A pure launch-cost reference for this shape: torch's own F.linear.
        def tref():
            F.linear(inp, w, b)
        us_t, gt = graph_us(tref)
        per, reps = kernels_of(gt)
        nk = sum(c for c, _ in per.values()) / reps / 40.0
        print(f"  F.linear (PyTorch default): {us_t:7.3f} us  {nk:4.2f} kernels")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
