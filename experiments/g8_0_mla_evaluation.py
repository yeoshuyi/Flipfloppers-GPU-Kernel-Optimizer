"""
G8.0 -- is Multi-head Latent Attention (and GQA / MQA) a possible optimization
for the official 14-row matrix?

MLA (DeepSeek-V2/V3) caches ONE low-rank latent c = x W_D instead of K and V,
and absorbs the up-projections into W_Q and W_O so K is never materialised. Its
payoff is a smaller KV cache during autoregressive DECODE.

The analytic claim this probe exists to test, in PREFILL (all this harness ever
does) the S^2 terms dominate:

    standard MHA :  4 S^2 d      =  4 S^2 H head_dim
    absorbed MLA :  4 H S^2 d_c

so MLA is cheaper ONLY IF d_c < head_dim. But K and V are independent full-rank
[d,d] nn.Linear weights, so reproducing them needs d_c = d = H head_dim. The two
requirements contradict each other by exactly a factor of H. This probe measures
whether that is true rather than asserting it.

  1. rank_spectrum      -- SVD of the REAL stacked [W_K; W_V] per distinct d.
  2. accuracy_vs_rank   -- every official shape, latent rank r in {d, d/2, d/4,
     head_dim}, scored with the harness's own compare_outputs against the frozen
     BaselineTransformer. Yields the smallest r that stays in budget.
  3. speed_ab           -- a WORKING absorbed-MLA attention timed against the
     shipped path. Confirms or refutes the predicted ~H x slowdown.
  4. row14_latent       -- the only persistent K/V cache in the project; can an
     exact latent shrink it usefully?
  5. gqa_mqa            -- grouped/multi-query: expressible at all under
     strict=True weight copy? quantified, not argued.

Nothing here touches the shipped model -- the MLA path is implemented inside
this file, the way the CUTLASS phase-2 investigation was driven (PROGRESS step
40), so a failing experiment cannot leave residue.

Run via infra/slurm/g8_0_mla_evaluation.sbatch. sbatch only.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ATOL, RTOL = 2e-3, 2e-2

# label, B, S, d, H, ffn, layers
OFFICIAL = [
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
    # row 14 is handled separately -- the frozen baseline cannot run it
]


def build_models(bs, sl, dm, nh, ff, nl):
    cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=dm,
                              num_heads=nh, ffn_dim=ff, num_layers=nl,
                              causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    return cfg, base, opt


def make_input(bs, sl, dm, seed=1234):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(bs, sl, dm, generator=g, device=DEV, dtype=torch.float32)


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
# the MLA factorisation, derived from the COPIED weights (no new parameters)
# --------------------------------------------------------------------------
def mla_factors(attn, r):
    """Joint low-rank factorisation of K and V sharing ONE latent.

    nn.Linear stores [out, in] and computes x @ W.T, so K = x @ W_K.T + b_K.
    Stacking [W_K; W_V] (2d x d) and truncating its SVD to rank r gives
        c   = x @ W_D.T                 (the cached latent, [S, r])
        K   = c @ W_UK.T + b_K
        V   = c @ W_UV.T + b_V
    r = d reproduces both exactly (rank([W_K; W_V]) = d), which is precisely
    why MLA cannot compress anything here without losing accuracy.
    """
    d = attn.d_model
    W = torch.cat([attn.k_proj.weight, attn.v_proj.weight], dim=0).float()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    sq = S.clamp_min(0).sqrt()
    W_D = (sq[:, None] * Vh)                 # [r, d]
    W_U = U * sq[None, :]                    # [2d, r]
    return W_D, W_U[:d], W_U[d:]             # W_D, W_UK, W_UV


@torch.no_grad()
def mla_attention(attn, x, causal, r, absorbed=False):
    """One attention layer with K/V served from a rank-r shared latent.

    `absorbed=False` reconstructs K and V then runs the reference attention --
    this is what the accuracy sweep scores. `absorbed=True` is the actual MLA
    fast path: W_UK is folded into W_Q and scores are formed in latent space, so
    K is never materialised. The two are algebraically identical.
    """
    bsz, S, d = x.shape
    H, hd = attn.num_heads, attn.head_dim
    W_D, W_UK, W_UV = mla_factors(attn, r)
    c = F.linear(x, W_D)                                     # [bsz,S,r]  cache

    q = attn.q_proj(x).view(bsz, S, H, hd).transpose(1, 2)
    if absorbed:
        # W_Q' = W_UK_h^T @ W_Q_h  -> queries land straight in latent space,
        # scores are [S,S] contractions over r instead of over head_dim
        Wq = attn.q_proj.weight.view(H, hd, d)
        Wuk = W_UK.view(H, hd, r)
        Wq_abs = torch.einsum("hkr,hkd->hrd", Wuk, Wq)       # [H,r,d]
        qa = torch.einsum("bsd,hrd->bhsr", x, Wq_abs)
        scores = torch.matmul(qa, c.unsqueeze(1).transpose(-2, -1)) * attn.scale
    else:
        k = (F.linear(c, W_UK) + attn.k_proj.bias).view(bsz, S, H, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale

    if causal:
        m = torch.ones((S, S), device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(m, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)

    if absorbed:
        lat = torch.matmul(probs, c.unsqueeze(1))            # [b,H,S,r]
        ctx = torch.einsum("bhsr,hkr->bhsk", lat, W_UV.view(H, hd, r))
        ctx = ctx + attn.v_proj.bias.view(H, hd)[None, :, None, :]
    else:
        v = (F.linear(c, W_UV) + attn.v_proj.bias).view(bsz, S, H, hd).transpose(1, 2)
        ctx = torch.matmul(probs, v)
    ctx = ctx.transpose(1, 2).contiguous().view(bsz, S, d)
    return attn.out_proj(ctx)


@torch.no_grad()
def mla_model_forward(base, x, r, absorbed=False):
    """BaselineTransformer.forward with MLA attention swapped in."""
    causal = base.config.causal
    for layer in base.layers:
        x = x + mla_attention(layer.attention, layer.norm1(x), causal, r,
                              absorbed=absorbed)
        x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)),
                                     approximate="none"))
    return base.final_norm(x)


# --------------------------------------------------------------------------
# 1. rank spectrum of the real weights
# --------------------------------------------------------------------------
@torch.no_grad()
def check_rank_spectrum():
    print("\n== 1. rank_spectrum (SVD of the real stacked [W_K; W_V]) ==",
          flush=True)
    print("  A sharp spectral decay is what would make a small latent legal.")
    print(f"  {'d':>6} {'rank':>6} {'energy kept':>12} {'Frobenius error':>16}")
    ok = True
    for d in (32, 128, 1024):
        _, base, _ = build_models(2, 64, d, 4, d, 1)
        attn = base.layers[0].attention
        W = torch.cat([attn.k_proj.weight, attn.v_proj.weight], 0).float()
        S = torch.linalg.svdvals(W)
        tot = (S ** 2).sum()
        for r in (d, d // 2, d // 4, max(d // 8, 1)):
            kept = (S[:r] ** 2).sum() / tot
            err = (1 - kept).clamp_min(0).sqrt()
            print(f"  {d:>6} {r:>6} {100 * kept.item():>11.2f}% "
                  f"{err.item():>16.4f}")
        del base
        torch.cuda.empty_cache()
    print("  (a flat spectrum => truncation is expensive; measured, not assumed)")
    return ok


# --------------------------------------------------------------------------
# 2. accuracy vs latent rank, every scorable shape
# --------------------------------------------------------------------------
@torch.no_grad()
def check_accuracy_vs_rank():
    print("\n== 2. accuracy_vs_rank (vs the frozen BaselineTransformer) ==",
          flush=True)
    print(f"  budget: abs < {ATOL} OR rel < {RTOL}")
    print(f"  {'shape':>6} {'d':>5} {'H':>3} {'hd':>4} {'rank':>5} "
          f"{'max_abs':>11} {'failed':>18}  verdict")
    exact_ok = True
    smallest = {}
    for (lab, bs, sl, dm, nh, ff, nl) in OFFICIAL:
        hd = dm // nh
        _, base, _ = build_models(bs, sl, dm, nh, ff, nl)
        x = make_input(bs, sl, dm)
        mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
        ref = base(x, mask)
        best = None
        for r in sorted({dm, dm // 2, dm // 4, hd}, reverse=True):
            if r < 1:
                continue
            y = mla_model_forward(base, x, r)
            res = B.compare_outputs(ref, y, RTOL, ATOL)
            passed = res.failed_elements == 0
            if r == dm and not passed:
                exact_ok = False            # our own implementation is wrong
            if passed:
                best = r
            print(f"  {lab:>6} {dm:>5} {nh:>3} {hd:>4} {r:>5} "
                  f"{res.max_abs_error:>11.3e} "
                  f"{str(res.failed_elements) + '/' + str(res.total_elements):>18}"
                  f"  {'PASS' if passed else 'FAIL'}"
                  f"{'   <- exact rank, sanity check' if r == dm else ''}")
            del y
        smallest[lab] = best
        del base, x, mask, ref
        torch.cuda.empty_cache()
    print(f"\n  smallest in-budget latent rank per shape: {smallest}")
    print("  MLA only pays when that rank is BELOW head_dim (see check 3).")
    return exact_ok


# --------------------------------------------------------------------------
# 3. implemented A/B: absorbed MLA vs the shipped path
# --------------------------------------------------------------------------
@torch.no_grad()
def check_speed_ab():
    print("\n== 3. speed_ab (absorbed MLA at the exact rank vs shipped) ==",
          flush=True)
    print(f"  {'shape':>6} {'d':>5} {'H':>3} {'shipped ms':>11} "
          f"{'MLA ms':>9} {'ratio':>7} {'predicted':>10}")
    for lab, bs, sl, dm, nh, ff, nl in [
            ("row08", 64, 128, 1024, 4, 1024, 4),
            ("row11", 64, 128, 128, 16, 128, 4),
            ("row09", 64, 128, 128, 1, 128, 4),
            ("row13", 64, 1024, 128, 4, 128, 4)]:
        try:
            _, base, opt = build_models(bs, sl, dm, nh, ff, nl)
            x = make_input(bs, sl, dm)
            mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
            t_ship = ev_time(lambda: opt(x, mask))
            t_mla = ev_time(lambda: mla_model_forward(base, x, dm, absorbed=True))
            print(f"  {lab:>6} {dm:>5} {nh:>3} {t_ship:>11.3f} {t_mla:>9.3f} "
                  f"{t_mla / t_ship:>6.2f}x {'~' + str(nh) + 'x':>10}")
            del base, opt, x, mask
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  {lab:>6} ERROR: {str(e)[:110]}")
            torch.cuda.empty_cache()
    return True


# --------------------------------------------------------------------------
# 4. row 14 -- the only persistent K/V cache in the project
# --------------------------------------------------------------------------
@torch.no_grad()
def check_row14_latent():
    print("\n== 4. row14_latent (the only real KV cache here) ==", flush=True)
    Bn, S, d, H = 32, 100000, 1024, 16
    kv = 2 * Bn * S * d * 2 / 2 ** 30
    lat = Bn * S * d * 2 / 2 ** 30
    print(f"  shipped K+V cache fp16     : {kv:.2f} GB")
    print(f"  exact latent (d_c = d = {d}) : {lat:.2f} GB   -- halves it")
    tot = sum(min((i + 1) * 2048, S) for i in range(49))
    reproj = 4 * Bn * tot * d * d * 2 / 1e15
    print(f"  but re-projecting the prefix to K,V once per chunk costs "
          f"{reproj:.2f}e15 FLOPs")
    print(f"  against an attention cost of "
          f"{2 * Bn * S * S * d * 2 / 1e15:.2f}e15 FLOPs -> ~+{100 * reproj / 1.31:.0f}% work")

    # can a fused attention kernel even run at head_dim = d_c = 1024?
    print("\n  can a fused kernel run attention at head_dim = d_c = 1024?")
    for hd_test in (64, 256, 512, 1024):
        q = torch.randn(1, 2, 512, hd_test, device=DEV, dtype=torch.float16)
        k = torch.randn(1, 2, 1024, hd_test, device=DEV, dtype=torch.float16)
        try:
            torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, k, 0.0, False, False, scale=1.0)
            fl = "yes"
        except Exception as e:
            fl = f"NO ({type(e).__name__})"
        try:
            torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, k, None, False, 0.0, False, scale=1.0)
            ef = "yes"
        except Exception as e:
            ef = f"NO ({type(e).__name__})"
        print(f"    head_dim={hd_test:>5}: flash={fl:<22} mem-efficient={ef}")
        del q, k
        torch.cuda.empty_cache()
    return True


# --------------------------------------------------------------------------
# 5. GQA / MQA
# --------------------------------------------------------------------------
@torch.no_grad()
def check_gqa_mqa():
    print("\n== 5. gqa_mqa ==", flush=True)
    print("  GQA/MQA share ONE K/V across a group of heads. The baseline's "
          "per-head\n  projections are independently initialised, so no exact "
          "form exists at any\n  group size -- unlike MLA, which at least has "
          "the trivial exact d_c = d.\n  Quantify how far off the closest "
          "thing (mean-pooled K/V per group) lands:")
    print(f"  {'shape':>6} {'H':>3} {'groups':>7} {'max_abs':>11} "
          f"{'failed':>18}  verdict")
    for lab, bs, sl, dm, nh, ff, nl in [
            ("row11", 64, 128, 128, 16, 128, 4),
            ("row01", 64, 128, 128, 4, 128, 4),
            ("row08", 64, 128, 1024, 4, 1024, 4)]:
        _, base, _ = build_models(bs, sl, dm, nh, ff, nl)
        x = make_input(bs, sl, dm)
        mask = torch.ones(bs, sl, dtype=torch.bool, device=DEV)
        ref = base(x, mask)
        for groups in (nh // 2, 1):
            if groups < 1:
                continue
            y = gqa_model_forward(base, x, groups)
            res = B.compare_outputs(ref, y, RTOL, ATOL)
            print(f"  {lab:>6} {nh:>3} {groups:>7} {res.max_abs_error:>11.3e} "
                  f"{str(res.failed_elements) + '/' + str(res.total_elements):>18}"
                  f"  {'PASS' if res.failed_elements == 0 else 'FAIL'}")
            del y
        del base, x, mask, ref
        torch.cuda.empty_cache()
    return True


@torch.no_grad()
def gqa_model_forward(base, x, groups):
    causal = base.config.causal
    for layer in base.layers:
        a = layer.attention
        h = layer.norm1(x)
        bsz, S, d = h.shape
        H, hd = a.num_heads, a.head_dim
        q = a.q_proj(h).view(bsz, S, H, hd).transpose(1, 2)
        k = a.k_proj(h).view(bsz, S, H, hd).transpose(1, 2)
        v = a.v_proj(h).view(bsz, S, H, hd).transpose(1, 2)
        per = H // groups
        k = k.view(bsz, groups, per, S, hd).mean(2, keepdim=True).expand(
            bsz, groups, per, S, hd).reshape(bsz, H, S, hd)
        v = v.view(bsz, groups, per, S, hd).mean(2, keepdim=True).expand(
            bsz, groups, per, S, hd).reshape(bsz, H, S, hd)
        sc = torch.matmul(q, k.transpose(-2, -1)) * a.scale
        if causal:
            m = torch.ones((S, S), device=x.device, dtype=torch.bool).triu(1)
            sc = sc.masked_fill(m, float("-inf"))
        ctx = torch.matmul(torch.softmax(sc.float(), -1).to(x.dtype), v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, S, d)
        x = x + a.out_proj(ctx)
        x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)),
                                     approximate="none"))
    return base.final_norm(x)


def main():
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM {free / 2 ** 30:.2f}/{total / 2 ** 30:.2f} GB")
    res = {
        "rank_spectrum": check_rank_spectrum(),
        "accuracy_vs_rank": check_accuracy_vs_rank(),
        "speed_ab": check_speed_ab(),
        "row14_latent": check_row14_latent(),
        "gqa_mqa": check_gqa_mqa(),
    }
    print("\n== SUMMARY ==")
    for k, v in res.items():
        print(f"  {k:<20} {'ok' if v else 'IMPLEMENTATION ERROR'}")
    ok = all(res.values())
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}"
          f"   (PASS = the probe itself is sound; the VERDICT on MLA is the "
          f"data above)")
    print("\nG8_0_DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
