#!/usr/bin/env python3
"""
G6.9 Phase 3 -- end-to-end check for the ONE surviving signature:
qkv (M8192, N384, K128) fp16 -- shapes 1, 9, 10, 11.

g6_9b: F.linear runs this qkv at ~11.9us (cuBLAS ampere_s16816gemm_128x64);
the best cuBLASLt heuristic candidate (cutlass_80 64x64_32x6) runs it at
~11.5us -- a real ~3% kernel-time win (bit-identical, max|diff|=0), NOT the
21% vs idx0 (idx0 is a strawman F.linear never uses).

This measures whether that ~3% on ONE of the 4 GEMMs moves the whole causal
forward.  Static lookup: the chosen algo is captured once, outside timing;
the timed path only calls ext.run(pid, best_idx, ...).  No search / calibrate
/ sync / bench in the timed region.

Shapes: official row 1 (B64 d128 h4 S128 L4).  Matched BEFORE/AFTER on the same
compiled forward -- BEFORE = stock F.linear qkv, AFTER = cuBLASLt-best qkv.
Also the 40-trial-style accuracy delta vs the stock path.
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
F16_SRC = os.path.join(ROOT, "csrc", "cublaslt_algo_fp16.cpp")
MAX_WS = 32 * 1024 * 1024


def build():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    return load(name="g6_9c_lt_f16", sources=[F16_SRC], build_directory=bd,
                with_cuda=True, extra_ldflags=["-lcublasLt"], verbose=False)


def etime(fn, warm=25, iters=200, reps=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best


@torch.no_grad()
def main():
    ext = build()
    cfg = B.TransformerConfig(batch_size=64, seq_len=128, d_model=128,
                              num_heads=4, ffn_dim=128, num_layers=4, causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    opt._ensure_folded_weights(DEV, torch.float32)
    for l in opt.layers:
        opt._build_qkv_fold(l.attention, l.norm1, DEV, torch.float32)
        opt._build_attn_fp16_fold(l.attention, DEV)
        opt._build_ffn_in_fold(l, DEV, torch.float32)

    M, d = cfg.batch_size * cfg.seq_len, cfg.d_model
    N, K = 3 * d, d
    # STATIC LOOKUP built here, ONCE, outside any timed region:
    pid = ext.create_problem(M, N, K, True, MAX_WS, 32, 2)   # mask=2 policy-compliant
    na = ext.num_algos(pid)
    x0 = torch.randn(M, K, device=DEV, dtype=torch.float16)
    w0 = opt.layers[0].attention._qkv_weight_fp16
    b0 = opt.layers[0].attention._qkv_bias_fp16
    o0 = torch.empty(M, N, device=DEV, dtype=torch.float16)
    ts = [ext.time_algo(pid, k, x0, w0, b0, o0, 30, 200) for k in range(na)]
    BEST = min(range(na), key=lambda k: ts[k])
    print(f"qkv (M{M},N{N},K{K}) fp16: {na} algos; idx0 {ts[0]*1e3:.2f}us, "
          f"best idx{BEST} {ts[BEST]*1e3:.2f}us ({(ts[0]-ts[BEST])/ts[0]*100:+.1f}% vs idx0)")
    print(f"  best algo: {ext.algo_info(pid, BEST)}")

    # patch qkv: monkeypatch F.linear only for the qkv weight tensors
    qkv_ws = {id(l.attention._qkv_weight_fp16) for l in opt.layers}
    _orig_linear = F.linear
    _obuf = torch.empty(M, N, device=DEV, dtype=torch.float16)

    def patched_linear(inp, weight, bias=None):
        if id(weight) in qkv_ws and inp.dim() == 3 and inp.shape[-1] == K:
            flat = inp.reshape(-1, K)
            ext.run(pid, BEST, flat, weight, bias, _obuf)
            return _obuf.view(inp.shape[0], inp.shape[1], N)
        return _orig_linear(inp, weight, bias)

    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=DEV)

    fwd = opt._optimized_forward_causal
    t_before = etime(lambda: fwd(x, None, True))
    y_before = fwd(x, None, True).float().clone()

    F.linear = patched_linear
    try:
        t_after = etime(lambda: fwd(x, None, True))
        y_after = fwd(x, None, True).float().clone()
    finally:
        F.linear = _orig_linear

    r64 = base.double().to(DEV)(x.double(), None)
    e_before = (y_before.double() - r64).abs().max().item()
    e_after = (y_after.double() - r64).abs().max().item()
    print(f"\n  full _optimized_forward_causal (row 1), eager:")
    print(f"    BEFORE (stock F.linear qkv) : {t_before*1e3:8.2f} us   max|.-fp64| {e_before:.3e}")
    print(f"    AFTER  (cuBLASLt-best qkv)  : {t_after*1e3:8.2f} us   max|.-fp64| {e_after:.3e}")
    print(f"    delta                       : {(t_before-t_after)/t_before*100:+.2f}%   "
          f"max|after-before| {(y_after-y_before).abs().max().item():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
