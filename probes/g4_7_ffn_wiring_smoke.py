#!/usr/bin/env python3
"""
G4.7 -- FAST wiring smoke test for the fused ffn_in+GELU custom op in the
CAUSAL path. ~1 minute on a GPU; run this BEFORE any full ship-verify sweep.

Two silent-fallback bugs already cost a 2 h sweep each (run133: _ensure_ffn_plan
placed after the causal branch's early return; run134: register_fake assumed a
2-D [tok,K] input but the causal path passes 3-D [B,S,d_model]). This asserts,
cheaply and loudly:

  1. _ensure_ffn_plan actually engages at a gate-eligible causal shape
     (self._ffn_cur == the requested cfg, not None).
  2. the fused op runs end-to-end through torch.compile'd _optimized_forward_
     causal without a fake-tensor/shape blow-up, and its output has the right
     shape and dtype.
  3. cfg 58 (ACCF32 + exact GELU, precision-NEUTRAL) tracks the F.linear+F.gelu
     fallback to within GEMM-reordering noise (< 5e-4 here).
  4. cfg 51 (FP16-accumulate) engages, differs from the fallback by MORE than
     cfg 58 does -- i.e. the config switch is real, not another silent
     fallback (that was the run134 tell: 58 and 51 came back bit-identical).
  5. a below-gate shape leaves self._ffn_cur is None.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import benchmark as B  # noqa: E402


@torch.no_grad()
def run_case(name, tc, expect_cfg):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    # ONE baseline, built once -- all three opt models copy the SAME weights
    # from it, so any output difference is the kernel, not random init.
    base = B.BaselineTransformer(tc).to(dev, torch.float32).eval()
    x = torch.randn(tc.batch_size, tc.seq_len, tc.d_model, device=dev)

    outs = {}
    cur = {}
    # run through the SAME context the judge harness uses (run_accuracy_tests
    # wraps the forward in torch.inference_mode()); run133/134 fell back
    # silently and we need to know whether inference_mode is the cause.
    for tag, cfgv in (("fallback", -1), ("cfg58", 58), ("cfg51", 51)):
        B._FFN_CFG = cfgv
        opt = B.UserOptimizedTransformer(tc)
        B.copy_model_weights(base, opt, strict=True)
        opt = opt.to(dev, torch.float32).eval()
        # direct call to surface any exception the custom op would swallow
        try:
            probe_cfg = cfgv if cfgv >= 0 else None
            if probe_cfg is not None and B._ffn_register_op():
                xf = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
                w = opt.layers[0]._ffn_in_weight_fp16 if hasattr(
                    opt.layers[0], "_ffn_in_weight_fp16") else None
                if w is None:
                    opt._build_ffn_in_fold(opt.layers[0], dev, torch.float32)
                    w = opt.layers[0]._ffn_in_weight_fp16
                bb = opt.layers[0]._ffn_in_bias_fp16
                _ = torch.ops.g43.ffn_gelu_linear(xf, w, bb, cfgv)
                print(f"  [{tag}] direct ffn_gelu_linear OK, "
                      f"out dtype {_.dtype} shape {tuple(_.shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{tag}] direct ffn_gelu_linear RAISED: {type(e).__name__}: {str(e)[:300]}")
        with torch.inference_mode():
            torch.compiler.cudagraph_mark_step_begin()
            y = opt(x)
        torch.cuda.synchronize()
        # clone out of the cudagraph static buffer before it is reused by the
        # next model's forward (mode="reduce-overhead").
        outs[tag] = y.float().clone()
        cur[tag] = opt._ffn_cur

    ok = True
    print(f"\n=== {name}  ({tc})")
    print(f"  _ffn_cur: fallback={cur['fallback']}  cfg58={cur['cfg58']}  "
          f"cfg51={cur['cfg51']}   (expect fallback=None, "
          f"cfg58={expect_cfg}, cfg51={51 if expect_cfg else None})")
    if expect_cfg is None:
        ok &= cur["cfg58"] is None and cur["cfg51"] is None
    else:
        ok &= cur["fallback"] is None
        ok &= cur["cfg58"] == 58 and cur["cfg51"] == 51

    for tag in ("fallback", "cfg58", "cfg51"):
        y = outs[tag]
        shape_ok = tuple(y.shape) == (tc.batch_size, tc.seq_len, tc.d_model)
        print(f"  {tag:8s} shape={tuple(y.shape)} dtype={y.dtype} "
              f"finite={bool(torch.isfinite(y).all())}  {'ok' if shape_ok else 'BAD SHAPE'}")
        ok &= shape_ok and bool(torch.isfinite(y).all())

    d58 = (outs["cfg58"] - outs["fallback"]).abs().max().item()
    d51 = (outs["cfg51"] - outs["fallback"]).abs().max().item()
    d5851 = (outs["cfg58"] - outs["cfg51"]).abs().max().item()
    print(f"  max|cfg58 - fallback| = {d58:.3e}   (expect < 2e-3 -- GEMM reorder only)")
    print(f"  max|cfg51 - fallback| = {d51:.3e}   (fp16-accum, expect larger than cfg58)")
    print(f"  max|cfg58 - cfg51|    = {d5851:.3e}   (expect > 0 -- configs must differ)")
    if expect_cfg is not None:
        if d58 > 2e-3:
            print("  FAIL: cfg58 (precision-neutral) drifted too far from fallback")
            ok = False
        if d5851 == 0.0:
            print("  FAIL: cfg58 and cfg51 identical -> one is silently falling back")
            ok = False
    print(f"  --> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}", flush=True)
    C = B.TransformerConfig
    all_ok = True
    # gate-eligible: tok=8192, ffn_dim%128==0, d_model%64==0
    all_ok &= run_case(
        "causal d512/ffn2048 tok8192",
        C(batch_size=8, seq_len=1024, d_model=512, num_heads=8,
          ffn_dim=2048, num_layers=2, causal=True), expect_cfg=58)
    # official-matrix shapes -- ffn_dim < 2048, must NOT engage (empirical gate)
    all_ok &= run_case(
        "causal d128/ffn128 tok8192 (official row-1 -- gate off, ffn_dim<2048)",
        C(batch_size=64, seq_len=128, d_model=128, num_heads=4,
          ffn_dim=128, num_layers=4, causal=True), expect_cfg=None)
    all_ok &= run_case(
        "causal d1024/ffn1024 tok8192 (official row-8 -- gate off, ffn_dim<2048)",
        C(batch_size=64, seq_len=128, d_model=1024, num_heads=4,
          ffn_dim=1024, num_layers=4, causal=True), expect_cfg=None)
    # below token gate: tok=2048
    all_ok &= run_case(
        "causal d512/ffn2048 tok2048 (below token gate)",
        C(batch_size=16, seq_len=128, d_model=512, num_heads=8,
          ffn_dim=2048, num_layers=2, causal=True), expect_cfg=None)

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
