"""
G7.3 / Steps 1+2 -- is FLASH faster than the mem-efficient kernel for Row 14's
per-chunk attention, and does the RAW flash op give bottom-right (prefix-causal)
alignment when Lq < Lk?

Step 0 (job 207) found the chunked forward is 86% attention:
  SDPA.past 10701 ms / 94 calls, SDPA.diag 248 ms / 100 calls, out of 12680 ms
  device time. The kernel is fmha_cutlassF_f16_aligned_64x64_rf_sm80 (the
  MEM-EFFICIENT one) at 119.7 TFLOP/s = 73% of the 165 TFLOP/s roofline, so
  ~3006 ms sits between it and peak. Nothing else is worth more than ~0.6 s.

The chunked path is on mem-efficient ONLY because it needs the returned LSE to
merge its two blocks. docs/PROGRESS.md step 44 found flash is fastest-or-tied
for fp16 causal everywhere else. So:

  A -- microbenchmark the EXACT per-chunk shapes both ways, summed over all 49
       chunks, so A1's win is known before touching benchmark.py.
  B -- does torch.ops.aten._flash_attention_forward with Lq<Lk align
       bottom-right? If yes, A2 collapses diag+past into ONE call and deletes
       the LSE merge, the fp32 widening and the contiguous copy.

sbatch only.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
import benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

Bn, H, hd, S, CQ = 32, 16, 64, 100000, 2048
NCH = (S + CQ - 1) // CQ
eff_attn = torch.ops.aten._scaled_dot_product_efficient_attention
flash_attn = torch.ops.aten._scaled_dot_product_flash_attention


def ev_time(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


# --------------------------------------------------------------------------
# A. mem-efficient vs flash on Row 14's actual per-chunk shapes
# --------------------------------------------------------------------------
@torch.no_grad()
def bench_backends():
    print("\n== A. per-chunk attention: mem-efficient vs flash ==", flush=True)
    print(f"  Row 14 chunking: {NCH} chunks of q={CQ}, B={Bn} H={H} hd={hd}")
    # One representative sample of prefix lengths; scale to the full sum after.
    probe_chunks = [1, 6, 12, 18, 24, 30, 36, 42, 48]
    tot_eff = tot_fl = tot_flbshd = 0.0
    print(f"\n  {'c0':>8} {'k_len':>8} {'mem-eff ms':>11} {'flash ms':>10} "
          f"{'flash/eff':>10}")
    for ci in probe_chunks:
        c0 = ci * CQ
        q = torch.randn(Bn, H, CQ, hd, device=DEV, dtype=torch.float16)
        k = torch.randn(Bn, H, c0, hd, device=DEV, dtype=torch.float16)
        v = torch.randn(Bn, H, c0, hd, device=DEV, dtype=torch.float16)
        t_eff = ev_time(lambda: eff_attn(q, k, v, None, True, 0.0, False,
                                         scale=1.0))
        try:
            t_fl = ev_time(lambda: flash_attn(q, k, v, 0.0, False, False,
                                              scale=1.0))
        except Exception as e:
            t_fl = float("nan")
            print(f"    flash failed: {str(e)[:100]}")
        tot_eff += t_eff
        tot_fl += t_fl
        print(f"  {c0:>8,} {c0:>8,} {t_eff:11.2f} {t_fl:10.2f} "
              f"{t_fl / t_eff:10.3f}")
        del q, k, v
        torch.cuda.empty_cache()
    # extrapolate: cost is ~linear in k_len, so scale the sampled sum by the
    # ratio of (sum of all prefix lengths) to (sum of sampled prefix lengths)
    all_sum = sum(i * CQ for i in range(1, NCH))
    smp_sum = sum(i * CQ for i in probe_chunks)
    scale = all_sum / smp_sum
    print(f"\n  sampled {len(probe_chunks)}/{NCH - 1} past-blocks; scaling by "
          f"{scale:.2f} to the full sweep, x2 layers:")
    print(f"    mem-efficient  {2 * tot_eff * scale:9.0f} ms   "
          f"(job 207 measured SDPA.past = 10701 ms)")
    print(f"    flash          {2 * tot_fl * scale:9.0f} ms   "
          f"-> {100 * (1 - tot_fl / tot_eff):+.1f}% vs mem-efficient")
    return tot_eff, tot_fl


# --------------------------------------------------------------------------
# B. is the raw flash op bottom-right aligned for Lq < Lk?
# --------------------------------------------------------------------------
@torch.no_grad()
def check_flash_bottomright():
    print("\n== B. raw _flash_attention_forward alignment for Lq < Lk ==",
          flush=True)
    b, h, hdim = 1, 2, 64
    lq, lk = 4, 10                       # c0 = 6, chunk of 4
    q = torch.randn(b, lq, h, hdim, device=DEV, dtype=torch.float16)
    k = torch.randn(b, lk, h, hdim, device=DEV, dtype=torch.float16)
    v = torch.randn(b, lk, h, hdim, device=DEV, dtype=torch.float16)

    # references, computed densely in fp32
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    scores = qh @ kh.transpose(-1, -2)
    ar = torch.arange(lq, device=DEV)
    ak = torch.arange(lk, device=DEV)
    tl = (ak[None, :] <= ar[:, None])                       # top-left
    br = (ak[None, :] <= (ar + (lk - lq))[:, None])         # bottom-right
    def ref(mask):
        sc = scores.masked_fill(~mask[None, None], float("-inf"))
        return (sc.softmax(-1) @ vh).transpose(1, 2)

    ref_tl, ref_br = ref(tl), ref(br)
    print(f"  shapes q={tuple(q.shape)} k={tuple(k.shape)} (BSHD), "
          f"Lq={lq} Lk={lk}")
    try:
        sch = str(torch.ops.aten._flash_attention_forward._schemas
                  if hasattr(torch.ops.aten._flash_attention_forward,
                             '_schemas') else
                  torch.ops.aten._flash_attention_forward.default._schema)
        print(f"  schema: {sch[:220]}")
    except Exception as e:
        print(f"  schema unavailable: {str(e)[:100]}")
    try:
        out = torch.ops.aten._flash_attention_forward(
            q, k, v, None, None, lq, lk, 0.0, True, False, scale=1.0)
        o = out[0]
        lse = out[1] if len(out) > 1 else None
        d_tl = (o.float() - ref_tl).abs().max().item()
        d_br = (o.float() - ref_br).abs().max().item()
        print(f"  out shape {tuple(o.shape)}   "
              f"lse shape {tuple(lse.shape) if lse is not None else None}")
        print(f"  max|out - TOP-LEFT ref|     = {d_tl:.3e}")
        print(f"  max|out - BOTTOM-RIGHT ref| = {d_br:.3e}")
        verdict = ("BOTTOM-RIGHT (A2 is possible)" if d_br < 1e-2 <= d_tl
                   else "TOP-LEFT (A2 impossible -- keep the 2-block split)"
                   if d_tl < 1e-2 else "NEITHER (investigate)")
        print(f"  VERDICT: {verdict}")
        return "BOTTOM-RIGHT" in verdict
    except Exception as e:
        print(f"  raw op call FAILED: {type(e).__name__}: {str(e)[:250]}")
        print(f"  VERDICT: A2 skipped -- raw flash op not usable on this stack")
        return False


@torch.no_grad()
def check_sdpa_flash_backend():
    """Can F.scaled_dot_product_attention be forced onto FLASH for these
    shapes at all? (fp32 has no flash kernel -- store=fp32 must stay
    mem-efficient, which is why every flash change goes under `if not wide`.)"""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    print("\n== C. backend availability ==", flush=True)
    for dt in (torch.float16, torch.float32):
        q = torch.randn(2, 4, 512, 64, device=DEV, dtype=dt)
        k = torch.randn(2, 4, 2048, 64, device=DEV, dtype=dt)
        v = torch.randn(2, 4, 2048, 64, device=DEV, dtype=dt)
        for name, backend in (("FLASH", SDPBackend.FLASH_ATTENTION),
                              ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel([backend]):
                    F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                                   scale=1.0)
                ok = "available"
            except Exception as e:
                ok = f"unavailable ({type(e).__name__})"
            print(f"  {str(dt).split('.')[-1]:<8} {name:<10} {ok}")
        del q, k, v
        torch.cuda.empty_cache()


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    check_sdpa_flash_backend()
    br = check_flash_bottomright()
    bench_backends()
    print(f"\n== VERDICT ==")
    print(f"  A2 (single bottom-right flash call): "
          f"{'GO' if br else 'BLOCKED -- keep the 2-block split'}")
    print("\nG7_3_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
