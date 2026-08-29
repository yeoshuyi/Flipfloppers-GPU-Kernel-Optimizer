#!/usr/bin/env python3
"""Real CUDA-event stage-by-stage timing for the README's regime
breakdown section. Calls the EAGER (uncompiled) per-layer stage functions
directly with the real, already-folded weights from a real model instance
-- this measures real kernel cost per stage but does not include CUDA-graph
launch-overhead removal (G2.4/G2.4b), which the top-level Before/After
table already covers accurately via the actual compiled path. Reported
honestly as eager component timing, not a graph-replay breakdown.
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/work")
import torch_transformer_benchmark as B

SHAPES = [
    ("tiny",        dict(batch_size=1, seq_len=64), False),
    ("default",     dict(batch_size=8, seq_len=128), False),
    ("long_seq",    dict(batch_size=8, seq_len=1024), False),
    ("large_batch", dict(batch_size=256, seq_len=128), False),
    ("padded",      dict(batch_size=8, seq_len=128), False),
    ("causal",      dict(batch_size=8, seq_len=128), True),
]

def time_stage(fn, iters=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

def run(name, kw, causal):
    device = torch.device("cuda")
    config = B.TransformerConfig(batch_size=kw["batch_size"], seq_len=kw["seq_len"],
                                 d_model=512, num_heads=8, ffn_dim=2048, num_layers=6,
                                 causal=causal)
    config.validate()
    m = B.UserOptimizedTransformer(config).to(device=device, dtype=torch.float32).eval()
    x = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)
    mask = None
    if name == "padded":
        mask = torch.ones(config.batch_size, config.seq_len, device=device, dtype=torch.bool)
        mask[:, config.seq_len // 3:] = False
    no_pad = mask is None
    m._ensure_folded_weights(device, torch.float32)
    m._ensure_lt_plan(config.batch_size * config.seq_len, device, torch.float32)
    if causal:
        for layer in m.layers:
            m._build_ffn_in_fold(layer, device, torch.float32)
            m._build_qkv_fold(layer.attention, layer.norm1, device, torch.float32)
            m._build_attn_fp16_fold(layer.attention, device)
    layer = m.layers[0]; attn = layer.attention
    lt = m._lt_cur

    def stage_cast_qkv():
        n1 = F.layer_norm(x, layer.norm1.normalized_shape, eps=layer.norm1.eps)
        n1f = n1.to(torch.float16)
        if causal:
            return F.linear(n1f, attn._qkv_weight_fp16, attn._qkv_bias_fp16)
        return F.linear(n1f, attn._qkv_weight_fp16, attn._qkv_bias_fp16)

    qkv = stage_cast_qkv()
    q, k, v = qkv.split(attn.d_model, dim=-1)
    q = m._split_heads_view(q, attn.num_heads, attn.head_dim)
    k = m._split_heads_view(k, attn.num_heads, attn.head_dim)
    v = m._split_heads_view(v, attn.num_heads, attn.head_dim)

    def stage_sdpa():
        if no_pad:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  is_causal=causal, scale=1.0)
        km = mask[:, None, None, :]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=km,
                                              is_causal=False, scale=1.0)

    ctx = stage_sdpa()
    ctx2 = ctx.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], attn.d_model)

    def stage_ffn():
        n2 = F.layer_norm(x, layer.norm2.normalized_shape, eps=layer.norm2.eps)
        n2f = n2.to(torch.float16)
        h = F.linear(n2f, layer._ffn_in_weight_fp16, layer._ffn_in_bias_fp16).to(torch.float32)
        return layer.ffn_out(F.gelu(h, approximate="none"))

    t_cast_qkv = time_stage(stage_cast_qkv)
    t_sdpa = time_stage(stage_sdpa)
    t_ffn = time_stage(stage_ffn)
    n_layers = config.num_layers
    total = (t_cast_qkv + t_sdpa + t_ffn) * n_layers
    print(f"{name:12s} cast+qkv={t_cast_qkv*n_layers:.4f}ms  sdpa={t_sdpa*n_layers:.4f}ms  "
          f"ffn={t_ffn*n_layers:.4f}ms  sum({n_layers}L)={total:.4f}ms")

for name, kw, causal in SHAPES:
    run(name, kw, causal)
