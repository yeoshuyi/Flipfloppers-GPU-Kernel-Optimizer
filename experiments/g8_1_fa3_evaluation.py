"""
G8.1 -- is FlashAttention-3 a possible optimization for the official matrix?

FA3 (Shah et al. 2024) gets its speed from three Hopper-only mechanisms:
`wgmma` asynchronous tensor-core issue, TMA bulk async copies, and the
warp-specialised producer/consumer overlap those enable (plus optional FP8).
This card is an RTX 4090 -- Ada, sm_89 -- which has none of them. That is the
same wall that disqualified ThunderKittens (docs/PROGRESS.md).

So the question is not "is FA3 faster" (on Hopper it is) but:
  1. can it run here at all?                       -> hardware gate, measured
  2. what does PyTorch actually dispatch today?    -> kernel name, measured
  3. if it COULD run, what is the ceiling?         -> Amdahl, per shape

(3) is the part that matters for every shape, and it is measurable without FA3:
attention's share of the forward bounds ANY attention-kernel improvement,
however good. The alternative backends were already swept in PROGRESS step 44
(`g5_0_sdpa_backend_audit_run146.log`: flash fastest-or-tied, cuDNN slower
everywhere) so they are not re-measured here.

Run via infra/slurm/g8_1_fa3_evaluation.sbatch. sbatch only.
"""
import json
import os
import sys
import tempfile
import time
from collections import defaultdict

import torch

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# label, B, S, d, H, ffn, layers
SHAPES = [
    ("row01", 64, 128, 128, 4, 128, 4),
    ("row02", 1, 128, 128, 4, 128, 4),
    ("row03", 4, 128, 128, 4, 128, 4),
    ("row04", 16, 128, 128, 4, 128, 4),
    ("row05", 128, 128, 128, 4, 128, 4),
    ("row06", 10000, 128, 128, 4, 128, 4),
    ("row07", 64, 128, 32, 4, 32, 4),
    ("row08", 64, 128, 1024, 4, 1024, 4),
    ("row09", 64, 128, 128, 1, 128, 4),
    ("row10", 64, 128, 128, 2, 128, 4),
    ("row11", 64, 128, 128, 16, 128, 4),
    ("row12", 64, 32, 128, 4, 128, 4),
    ("row13", 64, 1024, 128, 4, 128, 4),
    ("row14", 32, 100000, 1024, 16, 1024, 2),
]


def build(bs, sl, dm, nh, ff, nl):
    cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=dm,
                              num_heads=nh, ffn_dim=ff, num_layers=nl,
                              causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    del base
    return cfg, opt


def make_input(bs, sl, dm, fp16=False, seed=1234):
    if fp16:
        x = torch.empty(bs, sl, dm, dtype=torch.float16, device=DEV)
        g = torch.Generator(device=DEV).manual_seed(seed)
        for c0 in range(0, sl, 8192):
            x[:, c0:min(c0 + 8192, sl)].normal_(0.0, 1.0, generator=g)
        return x
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(bs, sl, dm, generator=g, device=DEV, dtype=torch.float32)


# --------------------------------------------------------------------------
# 1. hardware gate
# --------------------------------------------------------------------------
def check_fa3_availability():
    print("\n== 1. fa3_availability ==", flush=True)
    p = torch.cuda.get_device_properties(0)
    cc = (p.major, p.minor)
    print(f"  device            : {p.name}")
    print(f"  compute capability: sm_{cc[0]}{cc[1]}")
    print(f"  FA3 requires      : sm_90a (Hopper) -- wgmma + TMA + async "
          f"warp specialisation")
    runnable = cc >= (9, 0)
    print(f"  can FA3 run here  : {'YES' if runnable else 'NO'}   "
          f"{'PASS' if not runnable else ''}")
    print(f"  torch             : {torch.__version__}")
    for mod in ("flash_attn", "flash_attn_interface", "flash_attn_3"):
        try:
            m = __import__(mod)
            print(f"  {mod:<20}: present ({getattr(m, '__version__', '?')})")
        except Exception:
            print(f"  {mod:<20}: not installed")
    # what does torch actually dispatch for fp16 causal attention?
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    q = torch.randn(2, 4, 512, 64, device=DEV, dtype=torch.float16)
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as pr:
        F.scaled_dot_product_attention(q, q, q, is_causal=True, scale=1.0)
        torch.cuda.synchronize()
    names = [e.key for e in pr.key_averages()
             if getattr(e, "self_device_time_total", 0) > 0]
    print(f"  dispatched kernel : {names[0] if names else '?'}")
    print(f"  -> PyTorch vendors FlashAttention-2 style kernels; there is no "
          f"FA3 path on sm_89.")
    del q
    torch.cuda.empty_cache()
    return not runnable


# --------------------------------------------------------------------------
# 2. attention's share of the forward, per shape (the Amdahl input)
# --------------------------------------------------------------------------
def classify(op, kname):
    n = (op or kname or "?").lower()
    if any(t in n for t in ("efficient_attention", "flash", "fmha")):
        return "SDPA"
    if any(t in n for t in ("addmm", "aten::mm", "aten::bmm", "aten::matmul",
                            "aten::linear", "gemm", "cutlass", "s1688",
                            "s16816", "ampere_", "tensorop")):
        return "GEMM"
    return "other"


def phase_split(prof, iters):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.unlink(path)
    ext2op = {}
    for e in ev:
        if (e.get("cat") or "") == "cpu_op":
            a = e.get("args") or {}
            if a.get("External id") is not None:
                ext2op[a["External id"]] = e.get("name")
    ph = defaultdict(float)
    for e in ev:
        if (e.get("cat") or "").lower() not in ("kernel", "gpu_memcpy",
                                                "gpu_memset"):
            continue
        opn = ext2op.get((e.get("args") or {}).get("External id"))
        ph[classify(opn, e.get("name"))] += float(e.get("dur", 0.0))
    return {k: v / 1e3 / iters for k, v in ph.items()}


@torch.no_grad()
def profile_one(lab):
    """Attention's share of DEVICE time for ONE shape, in a FRESH process.

    Two things forced this design, both learned the hard way in the first run
    (job 230):
      * driving all 14 shapes through one process blows dynamo's recompile
        limit, so later shapes silently fall back to eager and the numbers stop
        being comparable;
      * under torch.compile(reduce-overhead) the forward replays as ONE CUDA
        graph, which collapses the per-kernel -> per-op correlation the split
        depends on, and everything lands in "other".
    So: one shape per process, and profile the EAGER path. The kernel mix is
    identical either way -- only the launch gaps differ, and those are not
    attention.
    """
    from torch.profiler import ProfilerActivity, profile
    spec = next(x for x in SHAPES if x[0] == lab)
    _, bs, sl, dm, nh, ff, nl = spec
    big = lab == "row14"
    _, opt = build(bs, sl, dm, nh, ff, nl)
    mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
    x = make_input(bs, sl, dm, fp16=big)
    if not big:
        # Go through forward() -- it builds the folded weights the causal path
        # needs -- but pre-empt the torch.compile so the kernels stay
        # individually attributable. Calling _optimized_forward_causal directly
        # skips the fold construction and raises (job 231).
        opt._compiled_causal = opt._optimized_forward_causal
    run = lambda: opt(x, mask)          # noqa: E731
    for _ in range(3 if not big else 1):
        run()
    torch.cuda.synchronize()
    if big:
        x = make_input(bs, sl, dm, fp16=True)   # chunked path mutates in place
        run = lambda: opt(x, mask)              # noqa: E731
    iters = 5 if not big else 1
    with profile(activities=[ProfilerActivity.CPU,
                             ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    ph = phase_split(pr, iters)
    tot = sum(ph.values())
    sd = ph.get("SDPA", 0.0)
    gm = ph.get("GEMM", 0.0)
    print(f"SHARE {lab} {tot:.6f} {sd:.6f} {gm:.6f}", flush=True)
    print(f"  {lab}: device {tot:.3f} ms   SDPA {sd:.3f} ms ({100 * sd / tot:.1f}%)"
          f"   GEMM {100 * gm / tot:.1f}%   other "
          f"{100 * (tot - sd - gm) / tot:.1f}%")
    return True


# --------------------------------------------------------------------------
# 3. the ceiling: best case from ANY faster attention kernel
# --------------------------------------------------------------------------
def amdahl(shares):
    print("\n== 3. amdahl_ceiling (what ANY faster attention could ever buy) ==",
          flush=True)
    print("  FA3's reported gain over FA2 on Hopper is ~1.5-2.0x. Applied to the")
    print("  MEASURED attention share, the whole-model ceiling per shape is:")
    print(f"  {'shape':>6} {'device ms':>10} {'SDPA %':>8} {'@1.5x':>8} "
          f"{'@2.0x':>8} {'@infinite':>10}")
    for lab in [s[0] for s in SHAPES]:
        if lab not in shares:
            print(f"  {lab:>6} {'(not measured)':>10}")
            continue
        tot, sd = shares[lab]
        f = sd / tot if tot else 0.0
        print(f"  {lab:>6} {tot:>10.3f} {100 * f:>7.1f}% "
              f"{100 * (f - f / 1.5):>7.1f}% {100 * (f - f / 2.0):>7.1f}% "
              f"{100 * f:>9.1f}%")
    print("\n  Those are ceilings on a kernel that CANNOT run on sm_89, so the")
    print("  realisable gain from FA3 on this hardware is exactly 0%.")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", action="store_true")
    ap.add_argument("--shape")
    ap.add_argument("--amdahl")
    a = ap.parse_args()

    if a.caps:
        print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
        return 0 if check_fa3_availability() else 1
    if a.shape:
        return 0 if profile_one(a.shape) else 1
    if a.amdahl:
        shares = {}
        for line in open(a.amdahl):
            if line.startswith("SHARE "):
                _, lab, tot, sd, _gm = line.split()
                shares[lab] = (float(tot), float(sd))
        amdahl(shares)
        print("\n== SUMMARY ==")
        print(f"  fa3_runnable_here      NO (sm_89 < sm_90a)")
        print(f"  shapes_profiled        {len(shares)}/14")
        ok = len(shares) == 14
        print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}"
              f"   (PASS = the probe is sound; the VERDICT is the data above)")
        print("\nG8_1_DONE", flush=True)
        return 0 if ok else 1
    print("nothing to do -- pass --caps, --shape <row>, or --amdahl <log>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
