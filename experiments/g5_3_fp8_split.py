#!/usr/bin/env python3
"""
Stage 1c (plan: /home/techjam2/.claude/plans/stateless-snuggling-mccarthy.md)
-- the decisive experiment. Uses REAL float8_e4m3fn hardware casts
throughout (not a simulated mantissa truncation like Stage 1a -- see
docs/PROGRESS.md step 21 for why that needed no further calibration
debugging: real casts are unambiguous).

Tests G2.8 (split/residual precision, CLAUDE.md's own explicit fallback
"use split-precision before abandoning low precision") -- never tried
this session until now. Greedy residual FP8 split with per-128-tile
dynamic scales: W ~= s0*Q0 + s1*Q1 + ... (k terms), same for activations.
approx_A @ approx_B (both dequantized, real fp32 matmul) is mathematically
equivalent to summing the k^2 cross-term FP8xFP8 products a real k-term
implementation would compute -- valid for the accuracy-gating question
Stage 1c answers (which k is enough), even though a real implementation
would do k^2 separate scaled_mm calls rather than one fp32 matmul on
the recombined approximation.

Also includes Stage 1b's granularity check (k=1, tile_size swept) folded
in since it's cheap to include in the same probe.

Gates (pre-committed, plan Part B):
  <=3 GEMMs passes all seeds/shapes with max_abs<=8e-4 -> proceed (1.9x ideal)
  4 GEMMs passes                                        -> marginal -> CLOSURE (confirmed decision)
  >=5 GEMMs needed                                       -> stop (loses to TF32 arithmetically)
  nothing passes                                         -> Stage 3 closure
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer, generate_random_case, compare_outputs

E4M3_MAX = 448.0


def tile_reshape(t: torch.Tensor, tile_size: int):
    last = t.shape[-1]
    assert last % tile_size == 0, f"{last} not divisible by {tile_size}"
    return t.reshape(*t.shape[:-1], last // tile_size, tile_size)


def quantize_residual_e4m3(t: torch.Tensor, k_terms: int, tile_size: int) -> torch.Tensor:
    """Greedy residual FP8 split with per-tile dynamic scales. Returns the
    dequantized, recombined k-term approximation (same shape as t)."""
    orig_shape = t.shape
    t_tiled = tile_reshape(t, tile_size)
    residual = t_tiled.clone()
    approx = torch.zeros_like(t_tiled)
    for _ in range(k_terms):
        amax = residual.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = amax / E4M3_MAX
        q = (residual / scale).to(torch.float8_e4m3fn)
        dequant = q.to(torch.float32) * scale
        approx = approx + dequant
        residual = residual - dequant
    return approx.reshape(orig_shape)


def ffn_split_forward(model: BaselineTransformer, x: torch.Tensor,
                       valid_token_mask, k_terms: int, tile_size: int) -> torch.Tensor:
    """FFN GEMMs use k-term FP8 split (weights AND activations); attention
    exact (CLAUDE.md: never FP8 in attention)."""
    causal = model.config.causal
    for layer in model.layers:
        x = x + layer.attention(layer.norm1(x), valid_token_mask, causal)

        n2 = layer.norm2(x)
        n2_q = quantize_residual_e4m3(n2, k_terms, tile_size)
        w1_q = quantize_residual_e4m3(layer.ffn_in.weight, k_terms, tile_size)
        hidden = F.linear(n2_q, w1_q, layer.ffn_in.bias)  # bias stays exact, negligible cost
        gelu_out = F.gelu(hidden, approximate="none")
        gelu_out_q = quantize_residual_e4m3(gelu_out, k_terms, tile_size)
        w2_q = quantize_residual_e4m3(layer.ffn_out.weight, k_terms, tile_size)
        x = x + F.linear(gelu_out_q, w2_q, layer.ffn_out.bias)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
    x = model.final_norm(x)
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x


def run(config: TransformerConfig, k_terms: int, tile_size: int, n_seeds: int = 20):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()

    errs, true_failures = [], 0
    for seed in range(n_seeds):
        x, mask = generate_random_case(config=config, device=device, dtype=torch.float32,
                                        seed=3000 + seed, padding_ratio=0.0, input_scale=1.0)
        with torch.no_grad():
            ref = model(x, mask)
            opt = ffn_split_forward(model, x, mask, k_terms, tile_size)
        cmp = compare_outputs(ref, opt, rtol=0.01, atol=0.001)
        errs.append(cmp.max_abs_error)
        if not cmp.passed:
            true_failures += 1
    return {"max_abs_max": max(errs), "max_abs_mean": sum(errs) / n_seeds,
            "true_failures": true_failures}


shapes = {
    "tiny": TransformerConfig(batch_size=1, seq_len=64, d_model=512, num_heads=8,
                               ffn_dim=2048, num_layers=6, causal=False),
    "default": TransformerConfig(batch_size=8, seq_len=128, d_model=512, num_heads=8,
                                  ffn_dim=2048, num_layers=6, causal=False),
    "long_seq": TransformerConfig(batch_size=8, seq_len=1024, d_model=512, num_heads=8,
                                   ffn_dim=2048, num_layers=6, causal=False),
}

print("=== Stage 1b (folded in): granularity check, k=1, default shape ===")
for tile_size in [512, 128, 64, 32]:
    r = run(shapes["default"], k_terms=1, tile_size=tile_size, n_seeds=20)
    print(f"  tile={tile_size:4d}  max_abs_max={r['max_abs_max']:.6f}  "
          f"max_abs_mean={r['max_abs_mean']:.6f}  true_failures={r['true_failures']}/20")
print()

print("=== Stage 1c: k-term split-precision (the decisive experiment), tile=128 ===")
for name, cfg in shapes.items():
    print(f"--- {name} ---")
    for k in [1, 2, 3, 4, 5, 6]:
        r = run(cfg, k_terms=k, tile_size=128, n_seeds=20)
        print(f"  k={k}  max_abs_max={r['max_abs_max']:.6f}  "
              f"max_abs_mean={r['max_abs_mean']:.6f}  true_failures={r['true_failures']}/20")
    print()
