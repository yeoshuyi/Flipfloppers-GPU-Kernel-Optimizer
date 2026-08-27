"""
G4.9 -- where is the latency on the OFFICIAL 14-row causal matrix?

Iteration 1 of the continuous loop. No build. Establishes, per official row,
where forward time actually goes on the CURRENTLY SHIPPED causal model
(G0.1c..G6.4bc + G4.7c, the latter inert here: d_model<512 on every official
row), so iteration 2+ attacks the real bottleneck.

Two views per row:
  A. EAGER per-stage timing -- cast+QKV / SDPA / FFN(in+GELU+out) x n_layers,
     called outside torch.compile so each stage's own kernel cost is isolated
     (same method + caveat as probes/stage_breakdown.py: the sum does not equal
     the graphed wall; the gap is the CUDA-graph launch-overhead saving).
  B. COMPILED-path kernel census -- CUPTI trace of the real graphed forward,
     every kernel by name with count + us/forward, bucketed GEMM/attn vs
     elementwise. (probes/g4_0_census.py pattern.)

Plus, for the memory-bound d128 rows (6, 13): a roofline mini-check on ffn_out
-- standalone FP16 GEMM time vs the [tok, ffn_dim] hidden write+read traffic
time -- to see whether fusing ffn_out (candidate 2) could pay.

Representative subset: rows 1 (small), 6 (large-batch tok 1.28M), 8 (wide
d1024), 13 (long-seq S=1024). Rows 2-5/7/9-12 track row 1; 14 OOMs the baseline.
"""
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from benchmark import (  # noqa: E402
    BaselineTransformer, TransformerConfig, UserOptimizedTransformer,
    copy_model_weights, generate_random_case,
)

DEV = torch.device("cuda")
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# name, batch, seq, d_model, heads, ffn_dim, layers
ROWS = [
    ("row1_d128_small",      64,   128, 128,  4,  128, 4),
    ("row6_d128_largebatch", 10000, 128, 128, 4,  128, 4),
    ("row8_d1024_wide",      64,   128, 1024, 4, 1024, 4),
    ("row13_d128_longseq",   64,  1024, 128,  4,  128, 4),
]


def build(b, s, d, h, f, L):
    cfg = TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                            ffn_dim=f, num_layers=L, causal=True)
    cfg.validate()
    base = BaselineTransformer(cfg).to(DEV, DTYPE).eval()
    opt = UserOptimizedTransformer(cfg).to(DEV, DTYPE).eval()
    copy_model_weights(base, opt, strict=True)
    x, mask = generate_random_case(cfg, DEV, DTYPE, seed=1234,
                                   padding_ratio=0.0, input_scale=1.0)
    return cfg, base, opt, x, mask


def wall_ms(model, x, mask, iters=200, warmup=50):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        for _ in range(iters):
            model(x, mask)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def time_fn(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def stages(cfg, opt, x):
    """Eager per-stage cost (x n_layers). Mirrors _optimized_forward_causal."""
    opt._ensure_folded_weights(DEV, DTYPE)
    for layer in opt.layers:
        opt._build_ffn_in_fold(layer, DEV, DTYPE)
        opt._build_qkv_fold(layer.attention, layer.norm1, DEV, DTYPE)
        opt._build_attn_fp16_fold(layer.attention, DEV)
    layer = opt.layers[0]
    attn = layer.attention
    nL = cfg.num_layers

    def st_cast_qkv():
        n1 = F.layer_norm(x, layer.norm1.normalized_shape, eps=layer.norm1.eps)
        n1f = n1.to(torch.float16)
        return F.linear(n1f, attn._qkv_weight_fp16, attn._qkv_bias_fp16)

    qkv = st_cast_qkv()
    q, k, v = qkv.split(attn.d_model, dim=-1)
    q = opt._split_heads_view(q, attn.num_heads, attn.head_dim)
    k = opt._split_heads_view(k, attn.num_heads, attn.head_dim)
    v = opt._split_heads_view(v, attn.num_heads, attn.head_dim)

    def st_sdpa():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                              is_causal=True, scale=1.0)

    def st_ffn():
        n2 = F.layer_norm(x, layer.norm2.normalized_shape, eps=layer.norm2.eps)
        n2f = n2.to(torch.float16)
        h = F.linear(n2f, layer._ffn_in_weight_fp16,
                     layer._ffn_in_bias_fp16).to(torch.float32)
        return layer.ffn_out(F.gelu(h, approximate="none"))

    return (time_fn(st_cast_qkv) * nL, time_fn(st_sdpa) * nL,
            time_fn(st_ffn) * nL)


def census(model, x, mask, iters=15):
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        for _ in range(25):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    ev = json.load(open(path))["traceEvents"]
    os.unlink(path)
    per = {}
    for e in ev:
        if (e.get("cat") or "").lower() not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        slot = per.setdefault(e.get("name", "?"), [0, 0.0])
        slot[0] += 1
        slot[1] += float(e.get("dur", 0.0))
    return sorted(([n, c / iters, u / iters] for n, (c, u) in per.items()),
                  key=lambda r: -r[2])


def bucket(name):
    n = name.lower()
    if any(t in n for t in ("fmha", "flash", "attention", "mha", "fused_attention")):
        return "SDPA"
    if any(t in n for t in ("gemm", "cutlass", "wmma", "s16816", "s1688",
                            "splitk", "implicit_gemm", "ampere_", "sm80_xmma",
                            "sm89")):
        return "GEMM"
    if "gelu" in n or "erf" in n:
        return "GELU"
    if any(t in n for t in ("layer_norm", "native_layer_norm", "add", "copy",
                            "elementwise", "vectorized", "triton_poi", "triton_per")):
        return "ELEM"
    return "OTHER"


def ffnout_roofline(cfg, tok):
    """d128 rows only: is the [tok, ffn_dim] hidden round-trip exposed?"""
    d, f = cfg.d_model, cfg.ffn_dim
    hid = torch.randn(tok, f, device=DEV, dtype=torch.float16)
    w = (torch.randn(d, f, device=DEV, dtype=torch.float16) * 0.05)
    b = torch.zeros(d, device=DEV, dtype=torch.float16)
    o = torch.empty(tok, d, device=DEV, dtype=torch.float16)
    t_gemm = time_fn(lambda: torch.addmm(b, hid, w.t(), out=o))
    # pure traffic: read [tok,f] fp16 + write [tok,d] fp16, no math
    src = torch.randn(tok, f, device=DEV, dtype=torch.float16)
    dst = torch.empty(tok, f, device=DEV, dtype=torch.float16)
    t_copy = time_fn(lambda: dst.copy_(src))
    bytes_rt = tok * f * 2 + tok * d * 2          # hidden read + out write
    bw = (tok * f * 2 * 2) / (t_copy * 1e-3) / 1e9   # GB/s from the copy
    t_traffic_est = bytes_rt / (bw * 1e9) * 1e3
    return t_gemm, t_copy, bw, t_traffic_est


def main():
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}\n")
    for name, b, s, d, h, f, L in ROWS:
        print(f"================ {name}  B={b} S={s} d={d} h={h} ffn={f} L={L} "
              f"tok={b * s} ================")
        cfg, base, opt, x, mask = build(b, s, d, h, f, L)
        try:
            bw_ms = wall_ms(base, x, mask, iters=100, warmup=30)
        except RuntimeError as exc:
            bw_ms = float("nan")
            print(f"  baseline wall: OOM/err ({str(exc)[:80]})")
        ow_ms = wall_ms(opt, x, mask)
        sp = bw_ms / ow_ms if bw_ms == bw_ms else float("nan")
        print(f"  baseline wall {bw_ms:9.4f} ms | optimized wall {ow_ms:9.4f} ms"
              f" | speedup {sp:.3f}x")

        t_cq, t_sd, t_ffn = stages(cfg, opt, x)
        esum = t_cq + t_sd + t_ffn
        print(f"  EAGER stages x{L}L:  cast+QKV {t_cq:8.4f}ms ({t_cq / esum * 100:4.1f}%)"
              f" | SDPA {t_sd:8.4f}ms ({t_sd / esum * 100:4.1f}%)"
              f" | FFN {t_ffn:8.4f}ms ({t_ffn / esum * 100:4.1f}%)"
              f"  [sum {esum:.4f}ms]")

        rows = census(opt, x, mask)
        tot = sum(r[2] for r in rows)
        buks = {}
        for n_, c_, u_ in rows:
            buks[bucket(n_)] = buks.get(bucket(n_), 0.0) + u_
        print(f"  COMPILED census: {sum(r[1] for r in rows):.0f} kern/fwd, "
              f"{tot / 1000:.4f}ms summed ({tot / 1000 / ow_ms * 100:.1f}% of wall)")
        for bkt in ("SDPA", "GEMM", "GELU", "ELEM", "OTHER"):
            if buks.get(bkt):
                print(f"     {bkt:6s} {buks[bkt] / 1000:9.4f}ms ({buks[bkt] / tot * 100:4.1f}%)")
        print("  top kernels (us/fwd, count/fwd, bucket, name):")
        for n_, c_, u_ in rows[:12]:
            print(f"     {u_:9.3f}us  {c_:6.2f}x  [{bucket(n_):5s}] {n_[:72]}")

        if d == 128:
            tg, tc, bwm, tt = ffnout_roofline(cfg, b * s)
            print(f"  FFN-fusion roofline (d128): ffn_out FP16 GEMM {tg:.4f}ms | "
                  f"hidden-copy {tc:.4f}ms | measured BW {bwm:.0f} GB/s | "
                  f"est [tok,{f}] round-trip {tt:.4f}ms "
                  f"({tt / max(tg, 1e-9) * 100:.0f}% of the GEMM -> "
                  f"{'EXPOSED, fuse candidate' if tt > 0.15 * tg else 'hidden, skip'})")
        print()
        del base, opt, x, mask
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
