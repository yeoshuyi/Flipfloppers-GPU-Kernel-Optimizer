#!/usr/bin/env python3
"""
Stage 1a

Fixes two real methodology gaps found this iteration (see docs/PROGRESS.md
step 21 for the full debugging story):
  1. probes/g4_fp8_accuracy_precheck.py tested an ISOLATED FFN with
     SYNTHETIC weights, while steps 14/17's BF16 numbers are FULL-MODEL
     output with REAL weights -- not comparable. This probe uses real
     BaselineTransformer weights throughout.
  2. The first version of this probe quantized ONLY the FFN and tried to
     anchor against steps 14/17's WHOLE-MODEL BF16 numbers -- a scope
     mismatch (FFN-only necessarily shows less error than whole-model).
     Fixed: a separate whole_model_quantized_forward exists specifically
     for anchor validation (matching steps 14/17's scope exactly); the
     main sweep (ffn_quantized_forward) stays FFN-only, since that's what
     actually feeds Stage 1c's split-precision design -- attention stays
     in TF32 regardless (CLAUDE.md forbids FP8 in attention).
Also reports both max (the real pass/fail statistic, but noisy on a fixed,
deterministic weight tensor -- quantization grids at different bit-widths
aren't nested, so the "worst element" isn't smooth in m for one fixed
tensor) and mean (smooth, a monotonicity sanity check) error.

Anchor gate: whole-model m=8 must reproduce ~1.1e-2 (steps 14/17's real
BF16 measurement) and whole-model m=11 must reproduce ~7e-4
(results/g2_4b_sweep_run27.log's real TF32 measurement), both within 2x.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer, generate_random_case, compare_outputs


def fake_quant_mantissa(t: torch.Tensor, m_bits: int) -> torch.Tensor:
    """Round each fp32 element to the nearest value representable with only
    m_bits of mantissa (full fp32 exponent range kept) -- isolates mantissa
    width from any scale/dynamic-range choice (that's Stage 1b's question).
    Verified monotonic and correct on synthetic tensors at both moderate
    and weight-init-scale magnitude (see debugging trail in step 21)."""
    if m_bits >= 23:
        return t
    bits = t.contiguous().view(torch.int32)
    drop = 23 - m_bits
    mantissa_mask = (1 << drop) - 1
    half = 1 << (drop - 1)
    frac = bits & mantissa_mask
    round_bit = (bits >> drop) & 1
    round_up = (frac > half) | ((frac == half) & (round_bit == 1))
    add = torch.where(round_up, torch.full_like(bits, 1 << drop), torch.zeros_like(bits))
    rounded = (bits + add) & ~mantissa_mask
    return rounded.view(torch.float32)


def ffn_quantized_forward(model: BaselineTransformer, x: torch.Tensor,
                           valid_token_mask, m_bits: int) -> torch.Tensor:
    """Baseline's exact forward, except the FFN's GEMM inputs (weights +
    activations, per CLAUDE.md's 'cast down only at GEMM inputs' policy)
    are mantissa-truncated to m_bits. Attention, LayerNorm, GELU itself,
    and the residual stream stay exact fp32. This is the scope that
    matters for Stage 1c (attention is never a quantization target)."""
    causal = model.config.causal
    for layer in model.layers:
        x = x + layer.attention(layer.norm1(x), valid_token_mask, causal)

        n2 = layer.norm2(x)
        n2_q = fake_quant_mantissa(n2, m_bits)
        w1_q = fake_quant_mantissa(layer.ffn_in.weight, m_bits)
        b1_q = fake_quant_mantissa(layer.ffn_in.bias, m_bits)
        hidden = F.linear(n2_q, w1_q, b1_q)
        gelu_out = F.gelu(hidden, approximate="none")
        gelu_out_q = fake_quant_mantissa(gelu_out, m_bits)
        w2_q = fake_quant_mantissa(layer.ffn_out.weight, m_bits)
        b2_q = fake_quant_mantissa(layer.ffn_out.bias, m_bits)
        x = x + F.linear(gelu_out_q, w2_q, b2_q)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
    x = model.final_norm(x)
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x


def whole_model_quantized_forward(model: BaselineTransformer, x: torch.Tensor,
                                   valid_token_mask, m_bits: int) -> torch.Tensor:
    """Quantizes attention's Q/K/V/out_proj GEMMs too, matching steps
    14/17's whole-model BF16 scope EXACTLY -- used only for anchor
    validation, not the main sweep."""
    causal = model.config.causal
    for layer in model.layers:
        attn = layer.attention
        n1 = layer.norm1(x)
        n1_q = fake_quant_mantissa(n1, m_bits)
        batch, seq_len, _ = x.shape

        def q_proj_like(proj):
            w_q = fake_quant_mantissa(proj.weight, m_bits)
            b_q = fake_quant_mantissa(proj.bias, m_bits)
            out = F.linear(n1_q, w_q, b_q)
            return (out.view(batch, seq_len, attn.num_heads, attn.head_dim)
                    .transpose(1, 2).contiguous())

        q = q_proj_like(attn.q_proj)
        k = q_proj_like(attn.k_proj)
        v = q_proj_like(attn.v_proj)
        q_q, k_q, v_q = (fake_quant_mantissa(t, m_bits) for t in (q, k, v))

        scores = torch.matmul(q_q, k_q.transpose(-2, -1)) * attn.scale
        if valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1)
        context = torch.matmul(fake_quant_mantissa(probs, m_bits), v_q)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, attn.d_model)
        context_q = fake_quant_mantissa(context, m_bits)
        wo_q = fake_quant_mantissa(attn.out_proj.weight, m_bits)
        bo_q = fake_quant_mantissa(attn.out_proj.bias, m_bits)
        attn_out = F.linear(context_q, wo_q, bo_q)
        if valid_token_mask is not None:
            attn_out = attn_out.masked_fill(~valid_token_mask[..., None], 0)
        x = x + attn_out

        n2 = layer.norm2(x)
        n2_q = fake_quant_mantissa(n2, m_bits)
        w1_q = fake_quant_mantissa(layer.ffn_in.weight, m_bits)
        b1_q = fake_quant_mantissa(layer.ffn_in.bias, m_bits)
        hidden = F.linear(n2_q, w1_q, b1_q)
        gelu_out = fake_quant_mantissa(F.gelu(hidden, approximate="none"), m_bits)
        w2_q = fake_quant_mantissa(layer.ffn_out.weight, m_bits)
        b2_q = fake_quant_mantissa(layer.ffn_out.bias, m_bits)
        x = x + F.linear(gelu_out, w2_q, b2_q)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
    x = model.final_norm(x)
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x


def run_shape(config: TransformerConfig, m_bits_list, forward_fn, n_seeds=20):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()

    results = {}
    for m_bits in m_bits_list:
        errs = []
        true_failures = 0
        for seed in range(n_seeds):
            x, mask = generate_random_case(
                config=config, device=device, dtype=torch.float32,
                seed=2000 + seed, padding_ratio=0.0, input_scale=1.0,
            )
            with torch.no_grad():
                ref = model(x, mask)
                opt = forward_fn(model, x, mask, m_bits)
            cmp = compare_outputs(ref, opt, rtol=0.01, atol=0.001)
            errs.append(cmp.max_abs_error)
            if not cmp.passed:
                true_failures += 1
        results[m_bits] = {
            "max_abs_max": max(errs),
            "max_abs_mean": sum(errs) / n_seeds,
            "true_failures": true_failures,
        }
    return results


for name, cfg in [
    ("tiny", TransformerConfig(batch_size=1, seq_len=64, d_model=512, num_heads=8,
                                ffn_dim=2048, num_layers=6, causal=False)),
    ("default", TransformerConfig(batch_size=8, seq_len=128, d_model=512, num_heads=8,
                                   ffn_dim=2048, num_layers=6, causal=False)),
]:
    print(f"=== {name}: ANCHOR VALIDATION (whole-model quant, matches steps 14/17 scope) ===")
    anchor_res = run_shape(cfg, [8, 11], whole_model_quantized_forward, n_seeds=20)
    a8, a11 = anchor_res[8], anchor_res[11]
    print(f"  m=8  whole-model: max={a8['max_abs_max']:.6f} mean={a8['max_abs_mean']:.6f} "
          f"(expect ~1.1e-2, ratio={a8['max_abs_max']/0.011:.2f}x)")
    print(f"  m=11 whole-model: max={a11['max_abs_max']:.6f} mean={a11['max_abs_mean']:.6f} "
          f"(expect ~7e-4, ratio={a11['max_abs_max']/0.0007:.2f}x)")
    anchors_ok = (0.5 <= a8['max_abs_max']/0.011 <= 2.0) and (0.5 <= a11['max_abs_max']/0.0007 <= 2.0)
    print(f"  ANCHOR GATE: {'PASS' if anchors_ok else 'FAIL'}")
    print()

    print(f"=== {name}: FFN-ONLY sweep (feeds Stage 1c's split-precision design) ===")
    res = run_shape(cfg, list(range(3, 13)), ffn_quantized_forward, n_seeds=20)
    for m_bits, r in sorted(res.items()):
        print(f"  m={m_bits:2d}  max_abs_max={r['max_abs_max']:.6f}  "
              f"max_abs_mean={r['max_abs_mean']:.6f}  "
              f"true_failures={r['true_failures']}/20")
    print()
