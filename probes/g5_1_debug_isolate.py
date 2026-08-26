#!/usr/bin/env python3
"""Isolate where the m=10-vs-m=11 non-monotonicity comes from: test the
FFN quantization on real model weights, for ONE layer, ONE seed, checking
(a) does the isolated single-layer FFN error alone show the anomaly, and
(b) does it only appear after compounding through all 6 layers."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/work")
from benchmark import TransformerConfig, BaselineTransformer, generate_random_case


def fake_quant_mantissa(t, m_bits):
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


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.manual_seed(1234)

config = TransformerConfig(batch_size=8, seq_len=128, d_model=512, num_heads=8,
                            ffn_dim=2048, num_layers=6, causal=False)
model = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
layer = model.layers[0]

x, mask = generate_random_case(config=config, device=device, dtype=torch.float32,
                                seed=2000, padding_ratio=0.0, input_scale=1.0)

with torch.no_grad():
    x1 = x + layer.attention(layer.norm1(x), mask, False)
    n2 = layer.norm2(x1)

    print("=== single-layer FFN quantization error (real weights, real input) ===")
    for m_bits in [8, 9, 10, 11, 12, 13, 14, 16, 20, 22]:
        n2_q = fake_quant_mantissa(n2, m_bits)
        w1_q = fake_quant_mantissa(layer.ffn_in.weight, m_bits)
        b1_q = fake_quant_mantissa(layer.ffn_in.bias, m_bits)
        hidden_ref = F.linear(n2, layer.ffn_in.weight, layer.ffn_in.bias)
        hidden_q = F.linear(n2_q, w1_q, b1_q)
        gelu_ref = F.gelu(hidden_ref, approximate="none")
        gelu_q = F.gelu(hidden_q, approximate="none")
        gelu_q_q = fake_quant_mantissa(gelu_q, m_bits)
        w2_q = fake_quant_mantissa(layer.ffn_out.weight, m_bits)
        b2_q = fake_quant_mantissa(layer.ffn_out.bias, m_bits)
        ffn_ref = F.linear(gelu_ref, layer.ffn_out.weight, layer.ffn_out.bias)
        ffn_q = F.linear(gelu_q_q, w2_q, b2_q)
        err = (ffn_q - ffn_ref).abs().max().item()
        # also: how much error is JUST from quantizing the weight (not activation)?
        w1_only = F.linear(n2, w1_q, b1_q)
        w1_only_err = (w1_only - hidden_ref).abs().max().item()
        print(f"  m={m_bits:2d}  full_ffn_err={err:.8f}  weight_only_err(layer1)={w1_only_err:.8f}")

    print()
    print("=== does quantizing weights ALONE (no activation quant) stay monotonic? ===")
    for m_bits in [8, 9, 10, 11, 12, 13]:
        w1_q = fake_quant_mantissa(layer.ffn_in.weight, m_bits)
        hidden_q = F.linear(n2, w1_q, layer.ffn_in.bias)
        hidden_ref = F.linear(n2, layer.ffn_in.weight, layer.ffn_in.bias)
        err = (hidden_q - hidden_ref).abs().max().item()
        print(f"  m={m_bits:2d}  weight-quant-only err={err:.8f}")
