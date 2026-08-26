#!/usr/bin/env python3
"""
Resolving a real ambiguity before building anything: docs/PROGRESS.md step
22's "k=3" was tested via full dequant-recombine (implicitly capturing all
3x3=9 cross-terms' worth of accuracy), but CLAUDE.md's own G2.8 example
(2 terms/operand -> 3 GEMMs, dropping only the doubly-low A_lo*B_lo term)
implies a TRIANGULAR truncation, not literal "k terms = k GEMMs". For
k=3/operand that generalizes to 6 GEMMs (drop the 3 terms with i+j>=3),
not 3. This probe tests, with REAL torch._scaled_mm calls (per-row/
per-channel scales only -- confirmed via job 46 that's what's available;
Ada has no native per-128-tile microscaling, and Stage 1b (step 22) showed
granularity barely matters for this model anyway, so per-channel is a fair
substitute for the tile=128 numbers already measured):

  A) asymmetric-weight: weight split into 3 terms, activation single-term
     FP8 (weights are static/precomputed, so 3 terms there is "free" at
     runtime) -> exactly 3 real GEMMs, unambiguous.
  B) asymmetric-activation: activation split into 3 terms, weight single-term
     -> exactly 3 real GEMMs, the other asymmetric option.
  C) triangular-6: both sides 3 terms, keep only cross-terms with i+j<3
     (drop the 3 smallest: (1,2),(2,1),(2,2)) -> 6 real GEMMs, matching
     CLAUDE.md's own G2.8 generalization.

Reports accuracy for each against the full 9-term dequant-recombine
reference AND against BaselineTransformer directly (the real criterion),
so we know which (if any) 3-GEMM or 6-GEMM design is actually viable.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer, generate_random_case, compare_outputs

E4M3_MAX = 448.0


def per_channel_residual_terms(t: torch.Tensor, k_terms: int):
    """Per-ROW (last dim = whole channel) greedy residual FP8 split.
    Returns list of (fp8_tensor, scale_tensor[...,1]) pairs -- the scale
    broadcasts over the row, matching what torch._scaled_mm can actually
    consume (per-row/per-column scale_a/scale_b, not per-tile)."""
    residual = t.clone()
    terms = []
    for _ in range(k_terms):
        amax = residual.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = amax / E4M3_MAX
        q = (residual / scale).to(torch.float8_e4m3fn)
        dequant = q.to(torch.float32) * scale
        terms.append((q, scale))
        residual = residual - dequant
    return terms


def scaled_mm_2d(a_fp8, a_scale, b_fp8_T, b_scale_row) -> torch.Tensor:
    """a_fp8: [M,K] fp8, a_scale: [M,1] fp32 (per-row).
    b_fp8_T: [K,N] fp8 (already transposed so K is dim0),
    b_scale_row: [N,1] fp32 (per-output-row of the ORIGINAL [N,K] tensor,
    reshaped to broadcast as [1,N] for _scaled_mm's scale_b).

    Real hardware constraint found empirically: _scaled_mm with row-wise
    (per-row/per-column) scaling only supports bf16/fp16 output, not
    fp32 ("Only bf16 and fp16 high precision output types are supported
    for row-wise scaling"). Request bf16 (the GEMM's own accumulation is
    still done at whatever internal precision cuBLASLt uses regardless of
    output dtype) and upcast to fp32 IMMEDIATELY, before this term is
    summed with any others -- so multi-term summation happens in fp32,
    not compounded in bf16."""
    out = torch._scaled_mm(a_fp8, b_fp8_T, scale_a=a_scale,
                            scale_b=b_scale_row.transpose(-2, -1),
                            out_dtype=torch.bfloat16)
    return out.to(torch.float32)


def linear_via_scaled_mm(x_2d: torch.Tensor, x_terms, w_terms, mode: str) -> torch.Tensor:
    """x_2d: [M, K_in] flattened activations. w_terms: list of (q,scale) for
    weight [N, K_in] (nn.Linear layout, out=N, in=K_in). Computes the
    GEMM (no bias) via real scaled_mm calls per the requested mode."""
    out = None
    if mode == "asym_weight":  # x single-term, w k-term -> k GEMMs
        xq, xs = x_terms[0]
        for wq, ws in w_terms:
            wq_T = wq.transpose(-2, -1)
            r = scaled_mm_2d(xq, xs, wq_T, ws)
            out = r if out is None else out + r
    elif mode == "asym_activation":  # x k-term, w single-term -> k GEMMs
        wq, ws = w_terms[0]
        wq_T = wq.transpose(-2, -1)
        for xq, xs in x_terms:
            r = scaled_mm_2d(xq, xs, wq_T, ws)
            out = r if out is None else out + r
    elif mode == "triangular":  # keep i+j < k -> triangular(k) GEMMs
        k = len(x_terms)
        for i, (xq, xs) in enumerate(x_terms):
            for j, (wq, ws) in enumerate(w_terms):
                if i + j >= k:
                    continue
                wq_T = wq.transpose(-2, -1)
                r = scaled_mm_2d(xq, xs, wq_T, ws)
                out = r if out is None else out + r
    else:
        raise ValueError(mode)
    return out


def ffn_real_gemm_forward(model: BaselineTransformer, x: torch.Tensor,
                           valid_token_mask, k_terms: int, mode: str) -> torch.Tensor:
    causal = model.config.causal
    for layer in model.layers:
        x = x + layer.attention(layer.norm1(x), valid_token_mask, causal)

        n2 = layer.norm2(x)
        B, S, D = n2.shape
        n2_2d = n2.reshape(B * S, D)
        n2_terms = per_channel_residual_terms(n2_2d, k_terms)
        w1_terms = per_channel_residual_terms(layer.ffn_in.weight, k_terms)
        hidden_2d = linear_via_scaled_mm(n2_2d, n2_terms, w1_terms, mode)
        hidden = hidden_2d.reshape(B, S, -1) + layer.ffn_in.bias
        gelu_out = F.gelu(hidden, approximate="none")

        H = gelu_out.shape[-1]
        gelu_2d = gelu_out.reshape(B * S, H)
        gelu_terms = per_channel_residual_terms(gelu_2d, k_terms)
        w2_terms = per_channel_residual_terms(layer.ffn_out.weight, k_terms)
        ffn_2d = linear_via_scaled_mm(gelu_2d, gelu_terms, w2_terms, mode)
        ffn_out = ffn_2d.reshape(B, S, -1) + layer.ffn_out.bias
        x = x + ffn_out

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
    x = model.final_norm(x)
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x


def run(config, k_terms, mode, n_seeds=20):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()

    errs, true_failures = [], 0
    for seed in range(n_seeds):
        x, mask = generate_random_case(config=config, device=device, dtype=torch.float32,
                                        seed=4000 + seed, padding_ratio=0.0, input_scale=1.0)
        with torch.no_grad():
            ref = model(x, mask)
            opt = ffn_real_gemm_forward(model, x, mask, k_terms, mode)
        cmp = compare_outputs(ref, opt, rtol=0.01, atol=0.001)
        errs.append(cmp.max_abs_error)
        if not cmp.passed:
            true_failures += 1
    return {"max_abs_max": max(errs), "max_abs_mean": sum(errs) / n_seeds,
            "true_failures": true_failures}


shapes = {
    "default": TransformerConfig(batch_size=8, seq_len=128, d_model=512, num_heads=8,
                                  ffn_dim=2048, num_layers=6, causal=False),
}

print("=== real torch._scaled_mm k-term GEMM-count resolution (default shape, 20 seeds) ===")
for label, mode, k, n_gemms in [
    ("A: asym-weight (3 terms weight, 1 activation)", "asym_weight", 3, 3),
    ("B: asym-activation (3 terms activation, 1 weight)", "asym_activation", 3, 3),
    ("C: triangular (3 terms both, i+j<3)", "triangular", 3, 6),
]:
    r = run(shapes["default"], k, mode, n_seeds=20)
    print(f"  {label}  [{n_gemms} real GEMMs]")
    print(f"    max_abs_max={r['max_abs_max']:.6f}  max_abs_mean={r['max_abs_mean']:.6f}  "
          f"true_failures={r['true_failures']}/20")
