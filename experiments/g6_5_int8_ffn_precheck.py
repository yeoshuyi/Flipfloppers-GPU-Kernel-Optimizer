#!/usr/bin/env python3
"""
Cheap accuracy gate for INT8 FFN (docs/CATALOGUE.md G2.7), mirroring
probes/g4_fp8_accuracy_precheck.py's exact methodology (synthetic
quantize-dequant simulation, not a real GEMM kernel -- if the pure
quantization error alone already fails, a real INT8 kernel can't do
better, so this is a valid, cheap first gate).

Motivation: G2.7 was never triggered this session (Phase-0 showed FP8
was available, so the catalogue's own "use only if FP8 unavailable"
condition never fired). FP8 failed decisively (step 18) and FP16 in the
FFN failed by a hair on a rare statistical tail (step 27) -- INT8's
uniform fixed-point quantization is a structurally different scheme
(linear steps over a calibrated range, no floating exponent) worth
checking on its own evidence rather than assumed to fail by association.
"""
import torch
import torch.nn.functional as F

INT8_MAX = 127.0


def quant_dequant_per_channel(w: torch.Tensor) -> torch.Tensor:
    # w: 2D [out_features, in_features]. Symmetric per-OUTPUT-channel
    # (per row) scale, matching the FP8 precheck's own scheme.
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / INT8_MAX).clamp_min(1e-12)
    q = torch.clamp(torch.round(w / scale), -INT8_MAX, INT8_MAX)
    return q * scale


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().clamp_min(1e-12)
    scale = amax / INT8_MAX
    q = torch.clamp(torch.round(x / scale), -INT8_MAX, INT8_MAX)
    return q * scale


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

d_model, ffn_dim = 512, 2048
torch.manual_seed(0)
ffn_in_w = torch.randn(ffn_dim, d_model, device=device) * 0.02
ffn_in_b = torch.randn(ffn_dim, device=device) * 0.02
ffn_out_w = torch.randn(d_model, ffn_dim, device=device) * 0.02
ffn_out_b = torch.randn(d_model, device=device) * 0.02

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
