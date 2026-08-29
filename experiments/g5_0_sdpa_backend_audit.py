"""
G5.0 / iteration 2 (T1) -- is SDPA's automatically-dispatched backend the
fastest available one on the official causal rows, and does the compiled
causal path recompile on those shapes?

The shipped causal path (G6.4bc) passes FP16 q/k/v and lets SDPA auto-dispatch
(FP16 unlocked flash/efficient; the explicit EFFICIENT_ATTENTION forcing was
removed). This checks whether forcing a specific backend beats the auto pick,
per official row -- precision-neutral, config-only. Also counts torch._dynamo
recompiles for the small rows (G4.7 diagnostics showed the causal fn can hit
recompile_limit).
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# name, batch, seq, d_model, heads   (official rows; causal, L irrelevant for SDPA)
ROWS = [
    ("row1_d128_h4_s128",   64,   128, 128,  4),
    ("row6_d128_h4_s128",   10000, 128, 128, 4),
    ("row8_d1024_h4_s128",  64,   128, 1024, 4),
    ("row11_d128_h16_s128", 64,   128, 128, 16),
    ("row13_d128_h4_s1024", 64,  1024, 128,  4),
]

BACKENDS = [
    ("FLASH", SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN", SDPBackend.CUDNN_ATTENTION),
    ("MATH", SDPBackend.MATH),
]


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


def audit_sdpa():
    print("=== SDPA backend audit (fp16, is_causal=True, scale=1.0) ===")
    for name, b, s, d, h in ROWS:
        hd = d // h
        g = torch.Generator(device=DEV).manual_seed(0)
        q = torch.randn(b, h, s, hd, device=DEV, dtype=torch.float16, generator=g)
        k = torch.randn(b, h, s, hd, device=DEV, dtype=torch.float16, generator=g)
        v = torch.randn(b, h, s, hd, device=DEV, dtype=torch.float16, generator=g)

        def call():
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=True, scale=1.0)

        try:
            t_auto = time_fn(call)
        except RuntimeError as exc:
            print(f"  {name}: auto FAILED {str(exc)[:80]}")
            continue
        ref = call().float()
        results = [("auto", t_auto, 0.0)]
        for bname, bk in BACKENDS:
            try:
                with sdpa_kernel([bk]):
                    t = time_fn(call)
                    err = (call().float() - ref).abs().max().item()
                results.append((bname, t, err))
            except (RuntimeError, Exception) as exc:  # noqa: BLE001
                results.append((bname, float("nan"), float("nan")))
        best = min((r for r in results if r[1] == r[1]), key=lambda r: r[1])
        print(f"  {name}  hd={hd}  tok={b*s}")
        for bn, t, err in results:
            tag = "  <-- auto" if bn == "auto" else ""
            win = "  ** fastest" if bn == best[0] and bn != "auto" else ""
            if t != t:
                print(f"     {bn:10s}  (unavailable)")
            else:
                print(f"     {bn:10s}  {t*1000:8.2f} us   maxdiff {err:.2e}{tag}{win}")
        if best[0] != "auto" and best[1] < t_auto * 0.97:
            print(f"     >>> {best[0]} beats auto by "
                  f"{(t_auto/best[1]-1)*100:.1f}%  (candidate)")
        print()


def audit_recompiles():
    print("=== compiled causal path recompile count (small rows) ===")
    import torch._dynamo as dyn
    for name, b, s, d, h, L in [("row1", 64, 128, 128, 4, 4),
                                 ("row4", 16, 128, 128, 4, 4),
                                 ("row12", 64, 32, 128, 4, 4)]:
        dyn.reset()
        dyn.utils.counters.clear()
        cfg = B.TransformerConfig(batch_size=b, seq_len=s, d_model=d,
                                  num_heads=h, ffn_dim=d, num_layers=L,
                                  causal=True)
        base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
        opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
        B.copy_model_weights(base, opt, strict=True)
        x, m = B.generate_random_case(cfg, DEV, torch.float32, 1234, 0.0, 1.0)
        with torch.inference_mode():
            for _ in range(60):
                opt(x, m)
            torch.cuda.synchronize()
        rc = sum(v for k, v in dyn.utils.counters.get("recompiles", {}).items())
        frames = dict(dyn.utils.counters.get("stats", {}))
        print(f"  {name}: recompiles={rc}  stats={frames}")
        del base, opt
        torch.cuda.empty_cache()
    print()


def main():
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}\n")
    audit_sdpa()
    audit_recompiles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
