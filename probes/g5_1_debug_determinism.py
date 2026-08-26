#!/usr/bin/env python3
"""Quick check: does calling the same deterministic module twice on the
identical CUDA input give bit-identical output, with TF32 matmul enabled?"""
import torch
import sys
sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.manual_seed(1234)

config = TransformerConfig(batch_size=8, seq_len=128, d_model=512, num_heads=8,
                            ffn_dim=2048, num_layers=6, causal=False)
model = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
layer = model.layers[0]

torch.manual_seed(999)
x = torch.randn(8, 128, 512, device=device)

with torch.no_grad():
    a1 = layer.attention(layer.norm1(x), None, False)
    a2 = layer.attention(layer.norm1(x), None, False)

diff = (a1 - a2).abs()
print(f"attention called twice on identical input: max_abs_diff={diff.max().item():.10f}, "
      f"num_nonzero_diffs={(diff > 0).sum().item()}/{diff.numel()}")

with torch.no_grad():
    f1 = model(x, None)
    f2 = model(x, None)
diff2 = (f1 - f2).abs()
print(f"full model called twice on identical input: max_abs_diff={diff2.max().item():.10f}, "
      f"num_nonzero_diffs={(diff2 > 0).sum().item()}/{diff2.numel()}")
