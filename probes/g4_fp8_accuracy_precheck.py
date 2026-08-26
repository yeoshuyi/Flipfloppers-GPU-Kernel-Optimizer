#!/usr/bin/env python3
"""
Cheap accuracy gate for whether FP8 (as docs/MEGAKERNEL.md's G4 design
requires for shared-memory pipeline depth) is even viable for this model,
BEFORE investing in the full megakernel machinery.

BF16 already failed decisively twice this session (docs/PROGRESS.md steps
14, 17: ~11x over the 1e-3 atol budget, full-model and FFN-only). FP8
(e4m3, ~3 mantissa bits) is coarser than BF16 (~7 mantissa bits) per
quantization step, but CLAUDE.md's precision policy argues error averages
down over the K-reduction (eps/sqrt(K)) and per-channel scaling removes
the systematic part -- test that claim directly with a fair (per-channel
scaled) quantize-dequantize simulation of the FFN, not a strawman
per-tensor cast.

This simulates quantization numerically (cast to float8_e4m3fn and back,
per-channel scales on weights, per-tensor dynamic scale on activations)
rather than using torch._scaled_mm's real GEMM kernel -- if the pure
quantization error alone already fails, the real kernel can't do better,
so this is a valid, cheap first gate.
"""
import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


def quant_dequant_per_channel(w: torch.Tensor) -> torch.Tensor:
    # w: 2D [out_features, in_features]. Scale per OUTPUT channel (per row):
    # max over the in_features axis (dim=1), keepdim so it broadcasts back.
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / E4M3_MAX).clamp_min(1e-12)
    q = (w / scale).to(torch.float8_e4m3fn)
    return q.to(torch.float32) * scale


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().clamp_min(1e-12)
    scale = amax / E4M3_MAX
    q = (x / scale).to(torch.float8_e4m3fn)
    return q.to(torch.float32) * scale


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

d_model, ffn_dim = 512, 2048
torch.manual_seed(0)
ffn_in_w = torch.randn(ffn_dim, d_model, device=device) * 0.02
ffn_in_b = torch.randn(ffn_dim, device=device) * 0.02
ffn_out_w = torch.randn(d_model, ffn_dim, device=device) * 0.02
ffn_out_b = torch.randn(d_model, device=device) * 0.02

# Per-OUTPUT-channel scales: ffn_in.weight is [ffn_dim, d_model] (out,in) ->
# scale per row (dim=0, output channel). ffn_out.weight is [d_model, ffn_dim]
# (out,in) -> scale per row (dim=0) too.
ffn_in_w_q = quant_dequant_per_channel(ffn_in_w)
ffn_out_w_q = quant_dequant_per_channel(ffn_out_w)

atol, rtol = 0.001, 0.01
n_seeds = 20
true_failures = 0
max_abs_values = []

for seed in range(n_seeds):
    torch.manual_seed(1000 + seed)
    x = torch.randn(8, 128, d_model, device=device)

    ref_hidden = F.gelu(F.linear(x, ffn_in_w, ffn_in_b), approximate="none")
    ref = F.linear(ref_hidden, ffn_out_w, ffn_out_b)

    x_q = quant_dequant_per_tensor(x)
    hidden_q = F.gelu(F.linear(x_q, ffn_in_w_q, ffn_in_b), approximate="none")
    hidden_q_q = quant_dequant_per_tensor(hidden_q)
    opt = F.linear(hidden_q_q, ffn_out_w_q, ffn_out_b)

    abs_err = (opt - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-12)
    failed = ((abs_err > atol) & (rel_err > rtol)).sum().item()
    if failed > 0:
        true_failures += 1
    max_abs_values.append(float(abs_err.max().item()))

print(f"seeds={n_seeds} true_failures={true_failures} "
      f"max={max(max_abs_values):.6f} mean={sum(max_abs_values)/n_seeds:.6f} "
      f"min={min(max_abs_values):.6f}")
