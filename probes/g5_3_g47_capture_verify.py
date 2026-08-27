"""
G5.3 / iteration 5 -- does G4.7's fused ffn_in+GELU custom op actually run
THROUGH torch.compile(mode="reduce-overhead") + CUDA-graph capture, or does it
silently fall back once the graph is captured?

Motivation: step 46. The T3 ship-verify (run150) showed d128/tok-1.28M output
bit-identical BEFORE/AFTER -> silent fallback under capture. The wiring smoke
missed it because it calls the model once (eager pre-capture warmup). This
checks the SHIPPED regime -- d512/ffn2048 causal (G4.7 step 42) -- the
capture-aware way, and re-confirms the d128 fallback.

Method, per shape:
  1. build the causal UserOptimizedTransformer, _FFN_CFG=58, run 40 forwards
     (>> the reduce-overhead warmup, so the graph is captured and subsequent
     calls are replays), then profile 15 replays -> kernel census.
  2. same with _FFN_CFG=-1 (forced fallback).
  3. ENGAGED iff: the cfg-58 census contains the warp-spec kernel symbol
     (`ws_gemm_kernel`) AND the cfg-58 replay output differs from the
     cfg--1 replay output by a small (GEMM-reorder) amount; FELL BACK iff the
     two outputs are bit-identical and no ws_gemm_kernel appears.
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/work")
import benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

CASES = [
    ("d512/ffn2048 tok8192  (G4.7 regime 1 -- SHIPPED)",
     dict(batch_size=8, seq_len=1024, d_model=512, num_heads=8,
          ffn_dim=2048, num_layers=2)),
    ("d128/ffn128  tok1.05M (T3 -- expected fallback)",
     dict(batch_size=8192, seq_len=128, d_model=128, num_heads=4,
          ffn_dim=128, num_layers=2)),
]


def build(kw, cfg_val):
    B._FFN_CFG = cfg_val
    tc = B.TransformerConfig(causal=True, **kw)
    tc.validate()
    base = B.BaselineTransformer(tc).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(tc).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    x, m = B.generate_random_case(tc, DEV, torch.float32, 1234, 0.0, 1.0)
    return opt, x, m


def run_and_census(opt, x, m, warm=40, prof_iters=15):
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode():
        for _ in range(warm):
            opt(x, m)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(prof_iters):
                opt(x, m)
            torch.cuda.synchronize()
        out = opt(x, m).float().clone()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        p = fh.name
    prof.export_chrome_trace(p)
    ev = json.load(open(p))["traceEvents"]
    os.unlink(p)
    names = {}
    for e in ev:
        if (e.get("cat") or "").lower() == "kernel":
            names[e.get("name", "?")] = names.get(e.get("name", "?"), 0) + 1
    return out, names


def main():
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}\n")
    ok = True
    for label, kw in CASES:
        print(f"================ {label} ================")
        opt58, x, m = build(kw, 58)
        cur = opt58._ffn_cur if hasattr(opt58, "_ffn_cur") else "n/a"
        # _ensure_ffn_plan runs inside forward(); trigger it once
        with torch.inference_mode():
            opt58(x, m)
        cur = opt58._ffn_cur
        o58, k58 = run_and_census(opt58, x, m)
        del opt58
        torch.cuda.empty_cache()

        optfb, x2, m2 = build(kw, -1)
        ofb, kfb = run_and_census(optfb, x2, m2)
        del optfb
        torch.cuda.empty_cache()

        has_ws = any("ws_gemm_kernel" in n for n in k58)
        has_gelu_kern_58 = any("gelu" in n.lower() for n in k58)
        diff = (o58 - ofb).abs().max().item()
        print(f"  _ffn_cur (cfg58 build)      : {cur}")
        print(f"  ws_gemm_kernel in cfg58 trace: {has_ws}")
        print(f"  standalone GELU kernel in cfg58 trace: {has_gelu_kern_58}")
        print(f"  max|cfg58_replay - fallback_replay| : {diff:.3e}")
        ws_syms = [n for n in k58 if "ws_gemm_kernel" in n]
        for s in ws_syms[:3]:
            print(f"     WS kernel: {s[:90]}  x{k58[s]}")
        if cur == 58:
            engaged = has_ws and diff > 1e-6
            verdict = "ENGAGED through capture" if engaged else \
                "*** SILENT FALLBACK under capture ***"
            print(f"  VERDICT: {verdict}")
            if not engaged:
                ok = False
        else:
            print(f"  VERDICT: gate off (_ffn_cur={cur}) -- fused path not selected here")
        print()
    print("OVERALL:", "PASS (shipped regime engages)" if ok else
          "FAIL (a selected fused regime silently falls back)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
