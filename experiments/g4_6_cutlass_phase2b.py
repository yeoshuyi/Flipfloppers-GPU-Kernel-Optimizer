#!/usr/bin/env python3
"""
G4.6 Phase 2b -- END-TO-END accuracy of the model with the two FP16 attention
GEMMs served by CUTLASS FP16-ACCUMULATE, across the full 8-shape sweep.

WHY THIS RUNS BEFORE ANY INTEGRATION WORK.  Phase 2a established that CUTLASS
computes the right answer (exact on integer input, 0/48 one-hot) but priced
FP16 accumulation at x7.2-x7.8 in GEMM-level max_abs -- 7.2e-03 against a
1e-03 atol, with ~100k gate-failing elements at a SINGLE GEMM.  Whether that
survives to the model output is a different question (the qkv GEMM feeds
softmax, which renormalises; out_proj feeds an fp32 residual stream), and it
is the only question that decides whether G4.6 can ship.  Answering it needs
no custom-op/torch.compile plumbing at all -- monkeypatching F.linear and
forcing the eager path gives the authoritative number for a fraction of the
work.  If this fails, the integration is never written.

Accuracy is judged by benchmark.py's OWN run_accuracy_tests/compare_outputs,
not a reimplementation, so the verdict is the same one the harness would give.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch_transformer_benchmark as B  # noqa: E402

CUTLASS = os.path.join(ROOT, ".cutlass")
SRC_CPP = os.path.join(ROOT, "csrc", "g4_6_cutlass_gemm.cpp")
NUM_CFG = 24
SRC_CU = [os.path.join(ROOT, "csrc", f"g4_6_cutlass_cfg{i:02d}.cu")
          for i in range(NUM_CFG)]

CFG_QKV = 6     # Phase 1 winner at qkv/large_batch (best repeatability)
CFG_OUT = 18    # Phase 1 winner at out_proj/large_batch

# The full sweep, identical to jobs/g0_1_accuracy.sbatch.
SHAPES = [
    ("tiny",             dict(batch_size=1,   seq_len=64)),
    ("default",          dict(batch_size=8,   seq_len=128)),
    ("long_seq",         dict(batch_size=8,   seq_len=1024)),
    ("large_batch",      dict(batch_size=256, seq_len=128)),
    ("default_padded",   dict(batch_size=8,   seq_len=128, padding_ratio=0.3)),
    ("default_causal",   dict(batch_size=8,   seq_len=128, causal=True)),
    ("causal_padded",    dict(batch_size=8,   seq_len=128, causal=True,
                              padding_ratio=0.3)),
    ("long_seq_padded",  dict(batch_size=8,   seq_len=1024,
                              padding_ratio=0.3)),
]


def build_ext():
    from torch.utils.cpp_extension import load
    build_dir = os.environ.get("TORCH_EXT_BUILD_DIR",
                               os.path.join(ROOT, ".ext_build"))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="g4_6_cutlass_gemm",
        sources=[SRC_CPP] + SRC_CU,
        build_directory=build_dir,
        with_cuda=True,
        extra_include_paths=[os.path.join(CUTLASS, "include"),
                             os.path.join(CUTLASS, "tools", "util", "include")],
        extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                           "--expt-relaxed-constexpr",
                           "--expt-extended-lambda",
                           "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"],
        verbose=False,
    )


def install_patch(ext, enable, route="both"):
    """Route ONLY the two fp16 attention GEMMs to CUTLASS.

    Everything else -- the whole baseline (fp32), the FFN (TF32) -- passes
    straight through to the original F.linear, so this measures exactly the
    one change under test and nothing else.
    """
    orig = torch.nn.functional.linear
    ws = torch.empty(1, dtype=torch.uint8, device="cuda")
    stats = {"qkv": 0, "out": 0, "passthrough": 0}

    def patched(inp, w, bias=None):
        if (enable and inp.dtype == torch.float16 and w.dtype == torch.float16
                and bias is not None and inp.is_cuda and inp.dim() == 3
                and inp.is_contiguous() and w.is_contiguous()):
            N, K = w.shape
            want = (N == 1536 and route in ("both", "qkv")) or \
                   (N == 512 and route in ("both", "out"))
            if K == 512 and N in (1536, 512) and want:
                cfg = CFG_QKV if N == 1536 else CFG_OUT
                stats["qkv" if N == 1536 else "out"] += 1
                flat = inp.reshape(-1, K)
                out = torch.empty(flat.shape[0], N, device=inp.device,
                                  dtype=torch.float16)
                ext.cutlass_gemm(cfg, flat, w, bias.contiguous(), out, ws)
                return out.view(inp.shape[0], inp.shape[1], N)
        stats["passthrough"] += 1
        return orig(inp, w, bias)

    torch.nn.functional.linear = patched
    B.F.linear = patched
    return orig, stats


def restore(orig):
    torch.nn.functional.linear = orig
    B.F.linear = orig


def run_shape(name, kwargs, ext, enable, route="both"):
    padding_ratio = kwargs.pop("padding_ratio", 0.0)
    causal = kwargs.pop("causal", False)
    device = torch.device("cuda")
    dtype = torch.float32

    config = B.TransformerConfig(
        batch_size=kwargs["batch_size"], seq_len=kwargs["seq_len"],
        d_model=512, num_heads=8, ffn_dim=2048, num_layers=6, causal=causal)
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

    # Force the EAGER path: forward() only wraps _optimized_forward in
    # torch.compile lazily, so pre-seeding the slot with the uncompiled bound
    # method bypasses dynamo entirely and lets a monkeypatched F.linear be
    # seen. Accuracy is unaffected by compilation; speed is not measured here.
    optimized._compiled_impl = optimized._optimized_forward

    orig, stats = install_patch(ext, enable, route)
    try:
        passed = B.run_accuracy_tests(
            baseline=baseline, optimized=optimized, config=config,
            device=device, dtype=dtype, trials=5, seed=1234,
            padding_ratio=padding_ratio, input_scale=1.0,
            rtol=1e-2, atol=1e-3)
    finally:
        restore(orig)
    return passed, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["both", "qkv", "out"], default="both",
                    help="which of the two attention GEMMs to route to "
                         "CUTLASS FP16-accumulate")
    ap.add_argument("--mode", choices=["cutlass", "control"], default="cutlass",
                    help="control = same harness, patch disabled (proves the "
                         "harness itself is not what moved the numbers)")
    a = ap.parse_args()
    enable = (a.mode == "cutlass")

    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    print(f"MODE = {a.mode}  ROUTE = {a.route}  (CUTLASS routing "
          f"{'ENABLED' if enable else 'DISABLED'})\n", flush=True)
    ext = build_ext()

    results = {}
    for name, kwargs in SHAPES:
        print("\n" + "#" * 74)
        print(f"### SHAPE: {name}  {kwargs}")
        print("#" * 74, flush=True)
        passed, stats = run_shape(name, dict(kwargs), ext, enable, a.route)
        results[name] = (passed, stats)
        print(f"[routing] qkv->CUTLASS {stats['qkv']}, "
              f"out_proj->CUTLASS {stats['out']}, "
              f"passthrough {stats['passthrough']}")

    print("\n" + "=" * 74)
    print(f"PHASE 2b SUMMARY  (mode={a.mode}, route={a.route})")
    allp = True
    for name, (passed, stats) in results.items():
        routed = stats["qkv"] + stats["out"]
        print(f"  {name:18s} {'PASS' if passed else 'FAIL'}   "
              f"(CUTLASS-routed GEMM calls: {routed})")
        allp &= passed
    print(f"\nVERDICT: {'ALL SHAPES PASS' if allp else 'ACCURACY FAILURE'}")
    if not allp and enable:
        print("An accuracy failure means the benchmark is SKIPPED entirely "
              "and the score is 0 (CLAUDE.md invariant 2). G4.6 cannot ship "
              "in this form regardless of its measured 1.57x kernel speedup.")
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())
