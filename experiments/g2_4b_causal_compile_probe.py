#!/usr/bin/env python3
"""
Diagnostic for G2.4b: how often does torch.compile(mode="reduce-overhead")
on baseline's OWN exact causal computation actually cross the 1e-3 atol,
across many random seeds -- not just the 5 the benchmark harness samples.
"""
import torch
import sys
sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer

config = TransformerConfig(
    batch_size=8, seq_len=128, d_model=512, num_heads=8,
    ffn_dim=2048, num_layers=6, causal=True,
)
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

baseline = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
compiled = torch.compile(baseline.forward, mode="reduce-overhead")

atol = 0.001
rtol = 0.01
max_abs_values = []
fail_count = 0

for seed in range(40):
    torch.manual_seed(seed)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model,
                     device=device, dtype=torch.float32)
    with torch.no_grad():
        ref = baseline(x, None)
        opt = compiled(x, None).clone()

    abs_err = (opt.float() - ref.float()).abs()
    rel_err = abs_err / ref.float().abs().clamp_min(1e-12)
    failed = ((abs_err > atol) & (rel_err > rtol)).sum().item()
    max_abs_values.append(float(abs_err.max().item()))
    if failed > 0:
        fail_count += 1

n = len(max_abs_values)
over_atol = sum(1 for v in max_abs_values if v > atol)
print(f"seeds={n} true_failures(abs&rel both exceed)={fail_count} "
      f"max_abs_over_atol_alone={over_atol} "
      f"max={max(max_abs_values):.6f} mean={sum(max_abs_values)/n:.6f} "
      f"min={min(max_abs_values):.6f}")
