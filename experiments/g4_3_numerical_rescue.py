#!/usr/bin/env python3
"""
G4.3 numerical rescue -- price accuracy against speed for every rescue option.

The ship-verification run (results/g4_3_ship_verify_run128.log) came back FAST
and WRONG on the causal shapes:

  long_seq_causal     max_abs 0.00549904   (budget 0.002)  speedup 7.108 -> 7.559
  large_batch_causal  max_abs 0.00762591   (budget 0.002)  speedup 2.662 -> 2.828
  large_batch (non-causal) 0.00119157 -> 0.00197798, i.e. PASSING at 98.9% of
                      the budget, which CLAUDE.md's own "track the trend"
                      rule says to treat as a warning, not a pass.

CLAUDE.md's precision policy is explicit: do not revert on an accuracy failure
without first attempting numerical rescue. This probe prices the two rescues
that exist, one shape at a time, in a single job:

  1. SPLIT -- the FP32 accumulate carry already implemented in the kernel
     (MEGAKERNEL.md G4.4's mandated mitigation). FP16 accumulate WITHIN a
     chunk of SPLIT columns of K, promoted to an FP32 carry at each chunk
     boundary, so error grows as sqrt(SPLIT) instead of sqrt(K).
       cfg 48 = SPLIT 64 (carry every k-tile)  -> sqrt(512/64)  = 2.83x
       cfg 49 = SPLIT 128                      -> sqrt(512/128) = 2.00x
       cfg 50 = SPLIT 256                      -> sqrt(512/256) = 1.41x
  2. SCOPE -- which of the two projections the kernel takes at all. out_proj's
     error lands unattenuated in the residual (x = x + attn_out); qkv's passes
     through SDPA's softmax first. If out_proj is the whole problem, keeping
     it on F.linear costs only a quarter of the FLOPs.

Reported per row: max_abs from benchmark.py's own run_accuracy_tests, and the
model speedup from its own benchmark_models -- the same code paths the
shipping sweep uses, not a re-implementation.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import benchmark as B  # noqa: E402

# (label, cfg_qkv, cfg_out)
VARIANTS = [
    ("none  (F.linear both, the BEFORE state)", None, None),
    ("cfg26 both        (the FAILING state)  ",   26,   26),
    ("cfg26 qkv only    (out_proj F.linear)  ",   26, None),
    ("cfg26 out only    (qkv F.linear)       ", None,   26),
    ("split256 both     (cfg50)              ",   50,   50),
    ("split128 both     (cfg49)              ",   49,   49),
    ("split64  both     (cfg48)              ",   48,   48),
    ("split64  qkv only                      ",   48, None),
    ("cfg26 qkv + split64 out                ",   26,   48),
]

SHAPES = [
    ("large_batch_causal", 256, 128, True),   # worst max_abs in run128
    ("long_seq_causal   ", 8, 1024, True),
    ("large_batch       ", 256, 128, False),  # 98.9% of budget in run128
    ("long_seq          ", 8, 1024, False),
]

ATOL, RTOL = 0.002, 0.02
TRIALS = 8          # enough to see max_abs stabilise; the ship run uses 40


def main():
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    print(f"budget atol={ATOL} rtol={RTOL}; {TRIALS} accuracy trials per row\n")

    for (sname, bs, sl, causal) in SHAPES:
        cfg = B.TransformerConfig(batch_size=bs, seq_len=sl, d_model=512,
                                  num_heads=8, ffn_dim=2048, num_layers=6,
                                  causal=causal)
        print("=" * 104)
        print(f"=== SHAPE {sname}  B={bs} S={sl} causal={causal}  "
              f"tok={bs*sl}")
        print(f"{'variant':42s} {'max_abs':>12s} {'budget':>8s} "
              f"{'verdict':>8s} {'speedup':>9s} {'vs before':>10s}")
        print("-" * 104)
        base_speed = None
        for (label, cq, co) in VARIANTS:
            B._WS_CFG = (cq, co)
            torch.manual_seed(1234)
            baseline = B.BaselineTransformer(cfg).to(dev).eval()
            opt = B.UserOptimizedTransformer(cfg).to(dev).eval()
            opt.load_state_dict(baseline.state_dict(), strict=True)
            # Silence the harness's own printing for the accuracy pass; we
            # only want its verdict and the global max_abs it computes.
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok = B.run_accuracy_tests(
                    baseline, opt, cfg, dev, torch.float32, TRIALS, 1234,
                    0.0, 1.0, RTOL, ATOL)
            txt = buf.getvalue()
            ma = 0.0
            for line in txt.splitlines():
                if line.startswith("summary:"):
                    for tokn in line.split("|"):
                        if "max_abs" in tokn:
                            ma = float(tokn.split("=")[1])
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                B.benchmark_models(baseline, opt, cfg, dev, torch.float32,
                                   1234, 0.0, 1.0, 10, 40, 2)
            sp = None
            for line in buf2.getvalue().splitlines():
                if line.startswith("speedup"):
                    sp = float(line.split(":")[1].strip().split("x")[0])
            if base_speed is None:
                base_speed = sp
            rel = (sp / base_speed) if (sp and base_speed) else float("nan")
            print(f"{label:42s} {ma:12.8f} {ATOL:8.4f} "
                  f"{'PASS' if ok else 'FAIL':>8s} "
                  f"{(('%.3fx' % sp) if sp else '--'):>9s} "
                  f"{('%+.2f%%' % ((rel-1)*100)):>10s}", flush=True)
            del baseline, opt
            torch.cuda.empty_cache()
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
