#!/usr/bin/env python3
"""
G6.4a v2 (FP16 ffn_in only) -- re-check step 27's near-miss closure against
the new accuracy default (atol=0.002, rtol=0.02, per the updated grading
spec relayed 2026-08-27 -- see docs/PROGRESS.md's accuracy-policy note for
the provenance caveat).

Step 27 v2 (old default, atol=0.001/rtol=0.01, 40 seeds): tiny and
default_padded passed clean; default/long_seq/large_batch/long_seq_padded
still failed, but only by rare single-to-low-double-digit element counts
against 1.3M-671M element tensors (default 1/..., long_seq 6/..., large_batch
61/..., long_seq_padded 7/...) -- a near-miss, not a decisive failure, and
exactly the kind of result that is worth re-running rather than inferring
from the old summary max_abs alone.

This is NOT a modification to benchmark.py (that version was reverted via
`git checkout` and is gone). It reproduces the same isolated change --
plain FP16 for the ffn_in GEMM only, ffn_out and everything else stays
exact -- via the same monkeypatch technique already used and reviewed for
the CUTLASS near-miss recheck (g4_6_cutlass_phase2b_newbudget.py), applied
to the ONE call site this targets: benchmark.py line ~745,
`F.linear(n2, layer._ffn_in_weight, layer._ffn_in_bias)`, uniquely
identifiable by its weight shape (ffn_dim=2048, d_model=512) and fp32
input dtype (the FP16 QKV/out_proj calls in the same forward are already
fp16 in, so the dtype filter alone keeps this from touching them; ffn_out's
weight shape is the transpose, (512, 2048), so the shape filter alone would
have sufficed too -- both checked, belt and suspenders).

G6.6 (cuBLASLt FFN override) is explicitly disabled for this probe by
forcing `_lt_cur = None` every call -- isolates the ONE variable under test
(ffn_in precision) from a second, independent variable (which cuBLASLt
algorithm gets picked for this shape), matching this project's own
"isolate before compounding" discipline (step 27's own text, step 24).
Scope is intentionally the same 6 non-causal shapes step 27 used -- causal
never went through this codepath (it uses baseline's unfused ffn_in
directly, not the folded weight), and folding causal's FFN is Phase 2B-B1
of the current plan, a separate, later, independently-validated step.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch_transformer_benchmark as B  # noqa: E402

NEW_ATOL = 0.002
NEW_RTOL = 0.02
TRIALS = 40

FFN_IN_SHAPE = (2048, 512)  # (ffn_dim, d_model) -- uniquely the ffn_in GEMM

# Same 6 shapes as step 27's original v2 rigor probe -- no causal shapes,
# consistent with the fact that step 27 never exercised the causal path
# (which doesn't go through the folded _ffn_in_weight at all).
SHAPES = [
    ("tiny",             dict(batch_size=1,   seq_len=64)),
    ("default",          dict(batch_size=8,   seq_len=128)),
    ("long_seq",         dict(batch_size=8,   seq_len=1024)),
    ("large_batch",      dict(batch_size=256, seq_len=128)),
    ("default_padded",   dict(batch_size=8,   seq_len=128, padding_ratio=0.3)),
    ("long_seq_padded",  dict(batch_size=8,   seq_len=1024,
                              padding_ratio=0.3)),
]


def install_patch(enable):
    orig = F.linear
    stats = {"ffn_in": 0, "passthrough": 0}

    def patched(inp, w, bias=None):
        if (enable and inp.dtype == torch.float32 and w.dtype == torch.float32
                and bias is not None and tuple(w.shape) == FFN_IN_SHAPE
                and inp.is_cuda):
            stats["ffn_in"] += 1
            out16 = orig(inp.to(torch.float16), w.to(torch.float16),
                        bias.to(torch.float16))
            return out16.to(torch.float32)
        stats["passthrough"] += 1
        return orig(inp, w, bias)

    torch.nn.functional.linear = patched
    B.F.linear = patched
    return orig, stats


def restore(orig):
    torch.nn.functional.linear = orig
    B.F.linear = orig


def run_shape(name, kwargs, enable):
    padding_ratio = kwargs.pop("padding_ratio", 0.0)
    device = torch.device("cuda")
    dtype = torch.float32

    config = B.TransformerConfig(
        batch_size=kwargs["batch_size"], seq_len=kwargs["seq_len"],
        d_model=512, num_heads=8, ffn_dim=2048, num_layers=6, causal=False)
    config.validate()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    baseline = B.BaselineTransformer(config)
    optimized = B.UserOptimizedTransformer(config)
    B.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Force eager (bypass dynamo so the monkeypatched F.linear is seen),
    # same technique as the CUTLASS recheck probe.
    optimized._compiled_impl = optimized._optimized_forward
    # Force G6.6 off for every call -- isolates ffn_in precision as the
    # only variable under test (see module docstring).
    optimized._ensure_lt_plan = lambda *a, **k: setattr(
        optimized, "_lt_cur", None)

    orig, stats = install_patch(enable)
    try:
        passed = B.run_accuracy_tests(
            baseline=baseline, optimized=optimized, config=config,
            device=device, dtype=dtype, trials=TRIALS, seed=1234,
            padding_ratio=padding_ratio, input_scale=1.0,
            rtol=NEW_RTOL, atol=NEW_ATOL)
    finally:
        restore(orig)
    return passed, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "control"], default="fp16",
                    help="control = same harness, patch disabled")
    a = ap.parse_args()
    enable = (a.mode == "fp16")

    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    print(f"MODE = {a.mode}  (ffn_in FP16 routing "
          f"{'ENABLED' if enable else 'DISABLED'}, G6.6 forced off)",
          flush=True)
    print(f"BUDGET = atol={NEW_ATOL} rtol={NEW_RTOL}  TRIALS={TRIALS}  "
          f"(re-check of step 27 v2's near-miss under the new default)\n",
          flush=True)

    results = {}
    for name, kwargs in SHAPES:
        print("\n" + "#" * 74)
        print(f"### SHAPE: {name}  {kwargs}")
        print("#" * 74, flush=True)
        passed, stats = run_shape(name, dict(kwargs), enable)
        results[name] = (passed, stats)
        print(f"[routing] ffn_in->FP16 {stats['ffn_in']}, "
              f"passthrough {stats['passthrough']}")

    print("\n" + "=" * 74)
    print(f"G6.4a v2 NEW-BUDGET SUMMARY  (mode={a.mode}, atol={NEW_ATOL}, "
          f"rtol={NEW_RTOL}, trials={TRIALS})")
    allp = True
    for name, (passed, stats) in results.items():
        print(f"  {name:18s} {'PASS' if passed else 'FAIL'}   "
              f"(ffn_in FP16 GEMM calls: {stats['ffn_in']})")
        allp &= passed
    print(f"\nVERDICT: {'ALL SHAPES PASS' if allp else 'ACCURACY FAILURE'}")
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())
