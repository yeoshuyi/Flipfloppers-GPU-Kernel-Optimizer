"""
G5.MEGA Phase 0 -- correctness of the per-sequence fused causal megakernel
vs the shipped _optimized_forward_causal math (row-6 specialist: d128 h4 S128 L4).

Reference = model._optimized_forward_causal(x, None, True) run EAGER (uncompiled)
-- that is exactly the shipped arithmetic. Also fp64 baseline for absolute error
and the 0.002 budget check.

Small batch (B=64) -- the math is batch-independent; row 6 is B=10000.
"""
import copy
import os
import sys

import torch

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    os.makedirs(bd, exist_ok=True)
    return load(name="g5_mega_causal",
                sources=[os.path.join(ROOT, "csrc", "g5_mega_causal.cpp"),
                         os.path.join(ROOT, "csrc", "g5_mega_causal.cu")],
                build_directory=bd, with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-Xptxas", "-v", "-diag-suppress", "179"],
                verbose=True)


def stacked_weights(model):
    L = model.layers
    def st(getter):
        return torch.stack([getter(l).contiguous() for l in L], 0).contiguous()
    return dict(
        qkv_w=st(lambda l: l.attention._qkv_weight_fp16),
        qkv_b=st(lambda l: l.attention._qkv_bias_fp16),
        op_w=st(lambda l: l.attention._out_proj_weight_fp16),
        op_b=st(lambda l: l.attention._out_proj_bias_fp16),
        fi_w=st(lambda l: l._ffn_in_weight_fp16),
        fi_b=st(lambda l: l._ffn_in_bias_fp16),
        fo_w=st(lambda l: l.ffn_out.weight.float()),
        fo_b=st(lambda l: l.ffn_out.bias.float()),
        fn_w=model.final_norm.weight.detach().float().contiguous(),
        fn_b=model.final_norm.bias.detach().float().contiguous(),
    )


@torch.no_grad()
def main():
    ext = build_ext()
    cfg = B.TransformerConfig(batch_size=64, seq_len=128, d_model=128,
                              num_heads=4, ffn_dim=128, num_layers=4, causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    base64 = copy.deepcopy(base).to(torch.float64).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    opt._ensure_folded_weights(DEV, torch.float32)
    for l in opt.layers:
        opt._build_ffn_in_fold(l, DEV, torch.float32)
        opt._build_qkv_fold(l.attention, l.norm1, DEV, torch.float32)
        opt._build_attn_fp16_fold(l.attention, DEV)

    W = stacked_weights(opt)
    for k, v in W.items():
        print(f"  {k:6s} {tuple(v.shape)} {v.dtype}")
    print(f"  norm1.eps = {opt.layers[0].norm1.eps}  norm2.eps = {opt.layers[0].norm2.eps}"
          f"  final_norm.eps = {opt.final_norm.eps}", flush=True)

    ok = True
    for trial in range(5):
        x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=DEV,
                        generator=torch.Generator(device=DEV).manual_seed(100 + trial))
        ref = opt._optimized_forward_causal(x, None, True).float()          # shipped math, eager
        r64 = base64(x.double(), None)                                      # fp64 baseline

        out = torch.empty_like(x)
        ext.mega_causal_forward(x, out, W["qkv_w"], W["qkv_b"], W["op_w"],
                                W["op_b"], W["fi_w"], W["fi_b"], W["fo_w"],
                                W["fo_b"], W["fn_w"], W["fn_b"])
        torch.cuda.synchronize()

        d_ref = (out - ref).abs()
        d_64 = (out.double() - r64).abs()
        ref_d64 = (ref.double() - r64).abs()
        atol, rtol = 2e-3, 2e-2
        fail_mega = ((d_64 > atol) & (d_64 > rtol * r64.abs())).sum().item()
        fail_ref = ((ref_d64 > atol) & (ref_d64 > rtol * r64.abs())).sum().item()
        mega_err = d_64.max().item()
        ship_err = ref_d64.max().item()
        print(f"trial {trial}: max|mega-shipped_ref|={d_ref.max().item():.3e}  "
              f"max|mega-fp64|={mega_err:.3e} (fail {fail_mega})  "
              f"| shipped max|.-fp64|={ship_err:.3e} (fail {fail_ref})")
        # PASS = mega passes the budget AND is no worse than the shipped path
        # vs fp64 (a lateral rounding move, or better). It need NOT bit-match
        # the shipped path -- mega does fp32 attention where the shipped path
        # uses fp16 flash, so a ~1e-3 difference between them is expected and
        # not a bug.
        if fail_mega > 0:
            print("  FAIL: mega over the 0.002 budget vs fp64")
            ok = False
        if mega_err > ship_err + 2e-4:
            print("  FAIL: mega materially LESS accurate than the shipped path")
            ok = False

    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
