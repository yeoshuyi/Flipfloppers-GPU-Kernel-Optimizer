#!/usr/bin/env python3
"""
Diagnostic for the G0.1 default_causal accuracy borderline failure.
Compares each SDPA backend against the baseline's manual causal attention
at B=8, S=128 (the shape that failed), across several seeds.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import sys
sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer, copy_model_weights

config = TransformerConfig(
    batch_size=8, seq_len=128, d_model=512, num_heads=8,
    ffn_dim=2048, num_layers=6, causal=True,
)
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

baseline = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()

backends = {
    "MATH": SDPBackend.MATH,
    "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "CUDNN": SDPBackend.CUDNN_ATTENTION,
}

results = {name: [] for name in backends}

for seed in range(20):
    torch.manual_seed(seed)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model,
                     device=device, dtype=torch.float32)
    with torch.no_grad():
        ref = baseline(x, None)

    for name, backend in backends.items():
        try:
            with torch.no_grad(), sdpa_kernel([backend]):
                # replicate UserOptimizedTransformer's per-layer loop but
                # force one specific backend for the SDPA call
                h = x
                for layer in baseline.layers:
                    attn = layer.attention
                    normed = layer.norm1(h)
                    q = attn._split_heads(attn.q_proj(normed))
                    k = attn._split_heads(attn.k_proj(normed))
                    v = attn._split_heads(attn.v_proj(normed))
                    context = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=True
                    )
                    context = (context.transpose(1, 2).contiguous()
                               .view(h.shape[0], h.shape[1], attn.d_model))
                    h = h + attn.out_proj(context)
                    h2 = layer.norm2(h)
                    ffn = layer.ffn_out(F.gelu(layer.ffn_in(h2), approximate="none"))
                    h = h + ffn
                out = baseline.final_norm(h)
            abs_err = (out.float() - ref.float()).abs()
            results[name].append(float(abs_err.max().item()))
        except Exception as e:
            results[name].append(f"ERROR: {type(e).__name__}: {e}")

for name, vals in results.items():
    numeric = [v for v in vals if isinstance(v, float)]
    errs = [v for v in vals if not isinstance(v, float)]
    if numeric:
        print(f"{name:10s} max_abs over 20 seeds: max={max(numeric):.6f} "
              f"mean={sum(numeric)/len(numeric):.6f} "
              f"n_over_1e-3={sum(1 for v in numeric if v > 1e-3)}/20")
    if errs:
        print(f"{name:10s} errors: {errs[:1]}")
