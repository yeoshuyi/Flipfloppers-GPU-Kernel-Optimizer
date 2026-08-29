#!/usr/bin/env python3
"""
G6.8 follow-up -- characterise the residual ~2% seen in job 92 Part A.

Job 92 settled the stated hypothesis as a NEGATIVE for ffn_in:
  * the real compiled model's ffn_in GEMM at M=8192 is
    cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4 @ 248.02 us/launch
  * the best cuBLASLt candidate is 245.75-246.11 us -- x1.009, noise
  * under CUDA-graph capture eager F.linear(bias) and the "winning" cuBLASLt
    algo dispatch the SAME kernel (128x128_16x5_tn_align4), 244.49 vs 244.46 us,
    maxdiff 0.000e+00.  Step 34's tell: one kernel cannot be 1.12x faster than
    itself.  Run 71's 1.1177x is entirely its eager-loop bias reference
    (264.98 us here vs 244.49 us captured) -- the identical artifact step 33
    diagnosed at M=1024.
  * inductor already fuses ffn_in's bias into triton_poi_fused_addmm_gelu_view_2;
    there is no separate bias-add kernel and no addmm penalty to recover.

But enabling the WHOLE G6.6 path (both halves) at long_seq measured 1.0209x
end-to-end, bit-identical.  That is not ffn_in's GEMM (see above), so this probe
asks which half it is and whether it is stable and numerically exact:

    none  shipped F.linear for both halves
    in    cuBLASLt for ffn_in only
    out   cuBLASLt for ffn_out only
    both  cuBLASLt for both  (= raising _LT_MAX_TOKENS)

5 interleaved rounds, per-round bit-identity vs `none`, and the chosen
algorithm's config printed so a split-K (inexact) pick is visible.
"""
import os
import statistics
import sys
from typing import Optional

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch_transformer_benchmark as B  # noqa: E402

D, FF = 512, 2048
BATCH, SEQ = 8, 1024
WARMUP = 30
ITERS = 200
ROUNDS = 5


class Split(B.UserOptimizedTransformer):
    """Same model, but the G6.6 FFN branch is selectable per half.

    Only the FFN block differs from UserOptimizedTransformer._optimized_forward;
    everything above it is copied verbatim so the comparison isolates the FFN.
    """

    _lt_half = "none"

    def _optimized_forward(self, x, valid_token_mask, no_pad):
        lt = self._lt_cur
        half = self._lt_half
        for layer in self.layers:
            attn = layer.attention
            n1 = F.layer_norm(x, layer.norm1.normalized_shape,
                              eps=layer.norm1.eps)
            n1_fp16 = n1.to(torch.float16)
            qkv = F.linear(n1_fp16, attn._qkv_weight_fp16, attn._qkv_bias_fp16)
            q, k, v = qkv.split(attn.d_model, dim=-1)
            q = self._split_heads_view(q, attn.num_heads, attn.head_dim)
            k = self._split_heads_view(k, attn.num_heads, attn.head_dim)
            v = self._split_heads_view(v, attn.num_heads, attn.head_dim)
            if no_pad:
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=False, scale=1.0)
            else:
                key_keep = valid_token_mask[:, None, None, :]
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=key_keep, is_causal=False, scale=1.0)
            context = (context.transpose(1, 2).contiguous()
                       .view(x.shape[0], x.shape[1], attn.d_model))
            attn_out_fp16 = F.linear(context, attn._out_proj_weight_fp16,
                                     attn._out_proj_bias_fp16)
            attn_out = attn_out_fp16.to(torch.float32)
            if not no_pad:
                attn_out = attn_out.masked_fill(~valid_token_mask[..., None], 0)
            x = x + attn_out

            n2 = F.layer_norm(x, layer.norm2.normalized_shape,
                              eps=layer.norm2.eps)
            if lt is None or half == "none":
                ffn_hidden = F.linear(n2, layer._ffn_in_weight,
                                      layer._ffn_in_bias)
                ffn = layer.ffn_out(F.gelu(ffn_hidden, approximate="none"))
            else:
                pid_i, alg_i, pid_o, alg_o = lt
                n2f = n2.reshape(-1, n2.shape[-1])
                if half in ("in", "both"):
                    ffn_hidden = torch.ops.g66.lt_linear(
                        n2f, layer._ffn_in_weight, layer._ffn_in_bias,
                        pid_i, alg_i)
                else:
                    ffn_hidden = F.linear(n2f, layer._ffn_in_weight,
                                          layer._ffn_in_bias)
                act = F.gelu(ffn_hidden, approximate="none")
                if half in ("out", "both"):
                    ffn = torch.ops.g66.lt_linear(
                        act, layer.ffn_out.weight, layer.ffn_out.bias,
                        pid_o, alg_o).view(n2.shape)
                else:
                    ffn = F.linear(act, layer.ffn_out.weight,
                                   layer.ffn_out.bias).view(n2.shape)
            x = x + ffn
            if not no_pad:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
        x = self.final_norm(x)
        if not no_pad:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def build(device, dtype, half):
    cfg = B.TransformerConfig(batch_size=BATCH, seq_len=SEQ, d_model=D,
                              num_heads=8, ffn_dim=FF, num_layers=6,
                              causal=False)
    torch.manual_seed(1234)
    base = B.BaselineTransformer(cfg).to(device=device, dtype=dtype).eval()
    opt = Split(cfg).to(device=device, dtype=dtype).eval()
    opt._lt_half = half
    B.copy_model_weights(base, opt, strict=True)
    return cfg, opt


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev, dt = torch.device("cuda"), torch.float32
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__,
          flush=True)
    ext = B._lt_ext()
    print("ext available:", ext is not None, flush=True)

    cfg = B.TransformerConfig(batch_size=BATCH, seq_len=SEQ, d_model=D,
                              num_heads=8, ffn_dim=FF, num_layers=6,
                              causal=False)
    x, mask = B.generate_random_case(cfg, dev, dt, seed=4242,
                                     padding_ratio=0.0, input_scale=1.0)

    variants = ("none", "in", "out", "both")
    times = {v: [] for v in variants}
    ref = None
    for rnd in range(ROUNDS):
        line = [f"  round {rnd}:"]
        for v in variants:
            B._LT_MAX_TOKENS = 127 if v == "none" else 10 ** 9
            _, m = build(dev, dt, v)
            B.warmup_model(m, x, mask, WARMUP, dev)
            t = statistics.median(B.benchmark_once(m, x, mask, ITERS, dev))
            times[v].append(t)
            with torch.inference_mode():
                o = m(x, mask).clone().float()
            if v == "none":
                ref = o
                line.append(f" none {t*1000:7.1f}")
            else:
                md = (o - ref).abs().max().item()
                info = ""
                if m._lt_cur is not None and ext is not None:
                    pid_i, alg_i, pid_o, alg_o = m._lt_cur
                    if v in ("in", "both"):
                        info += " IN[" + ext.algo_info(pid_i, alg_i) + "]"
                    if v in ("out", "both"):
                        info += " OUT[" + ext.algo_info(pid_o, alg_o) + "]"
                line.append(f" {v} {t*1000:7.1f} (md={md:.2e})")
                if rnd == 0 or info:
                    line.append("\n        " + v + ":" + info)
            del m
            torch.cuda.empty_cache()
        print("".join(line), flush=True)

    B._LT_MAX_TOKENS = 127
    print("\n  variant   median_us    min_us   x vs none (median)", flush=True)
    base_med = statistics.median(times["none"])
    base_min = min(times["none"])
    for v in variants:
        med = statistics.median(times[v])
        mn = min(times[v])
        print(f"  {v:8s} {med*1000:9.1f} {mn*1000:9.1f}   "
              f"{base_med/med:7.4f}x  (min {base_min/mn:.4f}x)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
