"""
G4.0 Phase-1 census -- the FULL per-kernel breakdown of the CURRENTLY SHIPPED
forward at the TINY shape (B=1, S=64), plus default for context.

Step 20's census predates G6.4b (fp16 attention, step 28) and G6.6 (cuBLASLt
FFN GEMMs at tiny, step 33), both of which change the kernel mix at tiny
substantially:
  * G6.4b inserts fp32->fp16 casts and switches SDPA onto a flash/efficient
    backend (different kernel set than the old fmha_cutlassF_f32 path).
  * G6.6 replaces F.linear on the two FFN GEMMs with an OPAQUE custom op
    (torch.ops.g66.lt_linear), which inductor cannot fuse across -- so the
    GELU between them, and the bias/residual work around them, may now be
    standalone launches that inductor previously absorbed.

This probe lists EVERY kernel by name with per-forward count and duration, so
the "everything else" bucket (the only part a G4.0-style fusion can delete)
is enumerable rather than a single aggregate number.  It reports nothing that
requires ncu -- pure CUPTI kernel tracing, which sees inside CUDA-graph
replays.
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/work")

from benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
)

DEV = torch.device("cuda")
DTYPE = torch.float32

# MUST match benchmark.py main()'s argparse defaults.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SHAPES = [("tiny", 1, 64), ("default", 8, 128)]


def build(batch: int, seq: int):
    cfg = TransformerConfig(
        batch_size=batch, seq_len=seq, d_model=512, num_heads=8,
        ffn_dim=2048, num_layers=6, causal=False,
    )
    base = BaselineTransformer(cfg).to(DEV, DTYPE).eval()
    opt = UserOptimizedTransformer(cfg).to(DEV, DTYPE).eval()
    copy_model_weights(base, opt, strict=True)
    x, mask = generate_random_case(cfg, DEV, DTYPE, seed=1234,
                                   padding_ratio=0.0, input_scale=1.0)
    return base, opt, x, mask


def wall_ms(model, x, mask, iters=300, warmup=80) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            model(x, mask)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def census(model, x, mask, iters=20):
    from torch.profiler import ProfilerActivity, profile

    with torch.inference_mode():
        for _ in range(30):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    os.unlink(path)

    per = {}
    for ev in events:
        cat = (ev.get("cat") or "").lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        slot = per.setdefault(ev.get("name", "?"), [0, 0.0])
        slot[0] += 1
        slot[1] += float(ev.get("dur", 0.0))
    return [(n, c / iters, us / iters) for n, (c, us) in
            sorted(per.items(), key=lambda kv: -kv[1][1])]


def is_gemm(name: str) -> bool:
    low = name.lower()
    return any(t in low for t in ("gemm", "fmha", "cutlass", "flash",
                                  "attention", "implicit", "conv", "wmma",
                                  "splitk"))


def main() -> int:
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    try:
        import benchmark
        print(f"cublasLt ext available: {benchmark._lt_ext() is not None}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"cublasLt ext probe failed: {exc}")
    print()

    for name, batch, seq in SHAPES:
        base, opt, x, mask = build(batch, seq)
        bw = wall_ms(base, x, mask)
        ow = wall_ms(opt, x, mask)
        rows = census(opt, x, mask)
        tot_n = sum(c for _, c, _ in rows)
        tot_us = sum(u for _, _, u in rows)
        g_n = sum(c for n, c, _ in rows if is_gemm(n))
        g_us = sum(u for n, _, u in rows if is_gemm(n))
        o_n, o_us = tot_n - g_n, tot_us - g_us

        print(f"=== {name} (B={batch} S={seq} tok={batch * seq}) ===")
        print(f"  baseline wall   : {bw:8.4f} ms")
        print(f"  optimized wall  : {ow:8.4f} ms   speedup {bw / ow:.3f}x")
        print(f"  kernels/forward : {tot_n:6.1f}   summed {tot_us / 1000:.4f} ms "
              f"({tot_us / 1000 / ow * 100:.1f}% of wall)")
        print(f"  GEMM/attn       : {g_n:6.1f} launches  {g_us / 1000:.4f} ms "
              f"({g_us / max(tot_us, 1) * 100:.1f}%)")
        print(f"  everything else : {o_n:6.1f} launches  {o_us / 1000:.4f} ms "
              f"({o_us / max(tot_us, 1) * 100:.1f}%)  <- fusible bucket")
        print("  full kernel census (count/fwd, us/fwd, us/launch, class):")
        for n, c, u in rows:
            cls = "GEMM" if is_gemm(n) else "ELEM"
            print(f"    {c:6.2f}x {u:9.3f}us {u / max(c, 1e-9):7.3f}us/l "
                  f"[{cls}] {n[:76]}")
        print()
        del base, opt, x, mask
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
