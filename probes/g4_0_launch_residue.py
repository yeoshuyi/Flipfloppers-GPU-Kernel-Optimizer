"""
G4.0 Phase-1, launch residue -- what are the 20 non-GEMM launches at TINY, in
full, and which of them can actually be deleted?

The census (results/g4_0_census_run86.log) found, per forward at tiny:
  13 x inductor LayerNorm/residual/cast fusions  ~18.4 us
   6 x triton_poi_fused_gelu_lt_linear_2          ~7.3 us
   1 x at::native::...multi_tensor_apply_kernel   ~6.55 us   <- 3.3% of wall

The 13 inductor kernels are already the structural minimum: exactly 2 per layer
(one absorbing residual+LN before QKV, one absorbing cast+residual+LN before the
FFN) plus the entry LN and final_norm. They cannot be merged with each other
because the QKV GEMM, SDPA and out_proj sit between them, and they cannot be
merged INTO those because a LayerNorm is a full-row reduction over the GEMM's
whole N dimension, which no GEMM epilogue can express.

That leaves the multi_tensor_apply kernel, which the census could not name (the
template arguments were truncated). It is a single launch costing more than any
other elementwise kernel in the forward. Two things to establish:

  Q1 WHAT IS IT. Print the untruncated kernel name.

  Q2 IS IT DELETABLE. Hypothesis: it is inductor's cudagraph-trees copy of the
     non-static graph inputs into the graph's static input buffers. The G0.2/
     G1.1/G6.4b folded weights (_qkv_weight_fp16, _ffn_in_weight, ...) are
     PLAIN ATTRIBUTES, not nn.Parameters -- CLAUDE.md invariant 4 requires that,
     because load_state_dict(strict=True) rejects new keys -- and dynamo marks
     only real parameters/buffers as static addresses. Everything else is copied
     on every replay. torch._dynamo.mark_static_address() promises exactly this
     without adding a state_dict key. Tested by marking them and re-censusing.

Nothing here changes benchmark.py; the marking is applied from outside, to a
model instance, so the shipped file is untouched while the effect is measured.
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/work")

from benchmark import (  # noqa: E402
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
)

DEV = torch.device("cuda")
DTYPE = torch.float32
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SHAPES = [("tiny", 1, 64), ("default", 8, 128)]

# Every plain-attribute tensor the compiled region reads.
FOLDED_ATTRS_ATTN = ("_qkv_weight", "_qkv_bias", "_qkv_weight_fp16",
                     "_qkv_bias_fp16", "_out_proj_weight_fp16",
                     "_out_proj_bias_fp16")
FOLDED_ATTRS_LAYER = ("_ffn_in_weight", "_ffn_in_bias")


def build(batch, seq):
    cfg = TransformerConfig(batch_size=batch, seq_len=seq, d_model=512,
                            num_heads=8, ffn_dim=2048, num_layers=6,
                            causal=False)
    base = BaselineTransformer(cfg).to(DEV, DTYPE).eval()
    opt = UserOptimizedTransformer(cfg).to(DEV, DTYPE).eval()
    copy_model_weights(base, opt, strict=True)
    x, mask = generate_random_case(cfg, DEV, DTYPE, seed=1234,
                                   padding_ratio=0.0, input_scale=1.0)
    return opt, x, mask


def mark_static(model):
    """Pre-build the folded weights eagerly (exactly as forward() would) and
    mark every one of them as a static address BEFORE the first compiled call,
    so dynamo bakes the pointer into the graph instead of copying it per replay.
    Returns the number of tensors marked."""
    model._ensure_folded_weights(DEV, DTYPE)
    n = 0
    for layer in model.layers:
        for a in FOLDED_ATTRS_ATTN:
            t = getattr(layer.attention, a, None)
            if torch.is_tensor(t):
                torch._dynamo.mark_static_address(t)
                n += 1
        for a in FOLDED_ATTRS_LAYER:
            t = getattr(layer, a, None)
            if torch.is_tensor(t):
                torch._dynamo.mark_static_address(t)
                n += 1
    return n


def wall_ms(model, x, mask, iters=300, warmup=80):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            model(x, mask)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def census(model, x, mask, iters=20):
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        for _ in range(30):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                model(x, mask)
            torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    os.unlink(path)
    per = {}
    for ev in events:
        if (ev.get("cat") or "").lower() not in ("kernel", "gpu_memcpy",
                                                 "gpu_memset"):
            continue
        slot = per.setdefault(ev.get("name", "?"), [0, 0.0])
        slot[0] += 1
        slot[1] += float(ev.get("dur", 0.0))
    return [(n, c / iters, us / iters)
            for n, (c, us) in sorted(per.items(), key=lambda kv: -kv[1][1])]


def report(tag, rows, wall):
    tot_n = sum(c for _, c, _ in rows)
    tot_us = sum(u for _, _, u in rows)
    print(f"  [{tag}] wall {wall:.4f} ms | {tot_n:.1f} kernels | "
          f"{tot_us / 1000:.4f} ms summed")
    for n, c, u in rows:
        if "multi_tensor" in n or "copy" in n.lower() or "foreach" in n.lower():
            print(f"      {c:5.2f}x {u:8.3f}us  {n}")


def main() -> int:
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    print()

    for name, batch, seq in SHAPES:
        print(f"=== {name} (B={batch} S={seq}) ===")

        # ---- Q1: name the kernel, on the SHIPPED path -------------------
        m, x, mask = build(batch, seq)
        w0 = wall_ms(m, x, mask)
        r0 = census(m, x, mask)
        print("  SHIPPED -- full names of every non-GEMM kernel:")
        for n, c, u in r0:
            low = n.lower()
            if not any(t in low for t in ("gemm", "fmha", "flash", "cutlass",
                                          "splitk")):
                print(f"      {c:5.2f}x {u:8.3f}us  {n}")
        report("shipped", r0, w0)
        del m
        torch.cuda.empty_cache()

        # ---- Q2: same model, folded weights marked static ---------------
        m2, x2, mask2 = build(batch, seq)
        nmark = mark_static(m2)
        print(f"  marked {nmark} folded tensors as static addresses")
        w1 = wall_ms(m2, x2, mask2)
        r1 = census(m2, x2, mask2)
        report("static ", r1, w1)
        print(f"  wall: {w0:.4f} -> {w1:.4f} ms  ({w0 / w1:.4f}x)")
        n0 = sum(c for _, c, _ in r0)
        n1 = sum(c for _, c, _ in r1)
        print(f"  kernels/forward: {n0:.1f} -> {n1:.1f}")

        # correctness of the static-marked model against the shipped one
        m3, _, _ = build(batch, seq)
        with torch.inference_mode():
            a = m3(x2, mask2).float()
            b = m2(x2, mask2).float()
        print(f"  |static - shipped| max = "
              f"{(a - b).abs().max().item():.6e}")
        print()
        del m2, m3, x, mask, x2, mask2
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
