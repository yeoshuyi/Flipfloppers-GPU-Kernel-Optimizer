#!/usr/bin/env python3
"""
G6.6 diagnostic -- did the cuBLASLt plan actually FIRE at each shape?

Run 74 gave tiny 0.2390 -> 0.1980 ms (1.21x) but default 0.6502 -> 0.6483 ms
(0.3%, noise), even though the isolated block probe (run 73) predicted 1.19x on
the default-shape FFN block. Two very different explanations:

  (a) the plan fired at default and bought nothing -- meaning inductor's own
      lowering of the FULL model already avoids the addmm bias penalty that
      run 73's hand-built eager-ops-in-a-CUDA-graph comparison still paid, so
      there was never 1.19x on the table in the shipped path; or
  (b) gate 2 (must beat F.linear) rejected it at default, and the null is just
      the feature being off.

Those call for opposite conclusions, so this prints the plan directly rather
than inferring it from timings.
"""
import os
import sys

import torch

sys.path.insert(0, "/work" if os.path.isdir("/work") else
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark as bm  # noqa: E402


SHAPES = [("tiny", 1, 64), ("default", 8, 128), ("long_seq", 8, 1024),
          ("large_batch", 256, 128)]


def main():
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = bm.TransformerConfig(batch_size=8, seq_len=128, d_model=512,
                               num_heads=8, ffn_dim=2048, num_layers=6,
                               causal=False)
    model = bm.UserOptimizedTransformer(cfg).to(dev).eval()

    print("extension loaded :", bm._lt_ext() is not None, flush=True)
    print("custom op ready  :", bm._lt_register_op(), flush=True)
    print("_LT_MAX_TOKENS   :", bm._LT_MAX_TOKENS, flush=True)

    for name, B, S in SHAPES:
        tok = B * S
        x = torch.randn(B, S, cfg.d_model, device=dev)
        with torch.no_grad():
            model(x)
        plan = model._lt_plan.get(tok, "<not attempted>")
        fired = plan is not None and plan != "<not attempted>"
        print(f"  {name:12s} tok={tok:6d}  fired={fired}  plan={plan}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
