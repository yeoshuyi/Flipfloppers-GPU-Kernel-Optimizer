"""
G5.MEGA Phase 1 -- how fast is the (currently scalar) megakernel vs the shipped
compiled causal forward, at the official row-6 shape (B=10000, d128, h4, S128, L4)?

Times both via CUDA-graph replay (best-of-5). Also reports max|mega - compiled|
over the full [10000,128,128] output as a sanity check (g5_5 covers the budget).
"""
import os
import sys

import torch

sys.path.insert(0, "/work")
import benchmark as B  # noqa: E402

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_ext():
    from torch.utils.cpp_extension import load
    bd = os.environ.get("TORCH_EXT_BUILD_DIR", os.path.join(ROOT, ".ext_build"))
    return load(name="g5_mega_causal",
                sources=[os.path.join(ROOT, "csrc", "g5_mega_causal.cpp"),
                         os.path.join(ROOT, "csrc", "g5_mega_causal.cu")],
                build_directory=bd, with_cuda=True,
                extra_cuda_cflags=["-gencode=arch=compute_89,code=sm_89",
                                   "-diag-suppress", "179"], verbose=False)


def stacked(model):
    L = model.layers
    st = lambda g: torch.stack([g(l).contiguous() for l in L], 0).contiguous()
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
        fn_b=model.final_norm.bias.detach().float().contiguous())


def etime(call, iters=30, warm=15):
    # plain event timing -- do NOT wrap in torch.cuda.graph: the compiled
    # model already self-captures under reduce-overhead and a nested capture
    # errors. The megakernel is one launch; a warmed event-timed loop is
    # accurate to well under a %.
    for _ in range(warm):
        call()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(6):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        for _ in range(iters):
            call()
        e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / iters)
    return best


@torch.no_grad()
def main():
    ext = build_ext()
    cfg = B.TransformerConfig(batch_size=10000, seq_len=128, d_model=128,
                              num_heads=4, ffn_dim=128, num_layers=4, causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = B.BaselineTransformer(cfg).to(DEV, torch.float32).eval()
    opt = B.UserOptimizedTransformer(cfg).to(DEV, torch.float32).eval()
    B.copy_model_weights(base, opt, strict=True)
    opt._ensure_folded_weights(DEV, torch.float32)
    for l in opt.layers:
        opt._build_ffn_in_fold(l, DEV, torch.float32)
        opt._build_qkv_fold(l.attention, l.norm1, DEV, torch.float32)
        opt._build_attn_fp16_fold(l.attention, DEV)
    W = stacked(opt)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=DEV)
    mask = torch.ones(cfg.batch_size, cfg.seq_len, device=DEV, dtype=torch.bool)

    compiled = torch.compile(opt._optimized_forward_causal, mode="reduce-overhead")
    y_ref = compiled(x, mask, True).float().clone()
    out = torch.empty_like(x)
    ext.mega_causal_forward(x, out, W["qkv_w"], W["qkv_b"], W["op_w"], W["op_b"],
                            W["fi_w"], W["fi_b"], W["fo_w"], W["fo_b"],
                            W["fn_w"], W["fn_b"])
    torch.cuda.synchronize()
    print(f"max|mega - compiled_shipped| over [10000,128,128] = "
          f"{(out - y_ref).abs().max().item():.3e}", flush=True)

    t_ship = etime(lambda: compiled(x, mask, True))
    t_mega = etime(lambda: ext.mega_causal_forward(
        x, out, W["qkv_w"], W["qkv_b"], W["op_w"], W["op_b"], W["fi_w"],
        W["fi_b"], W["fo_w"], W["fo_b"], W["fn_w"], W["fn_b"]))
    print(f"\nrow 6 (B=10000):")
    print(f"  shipped compiled causal : {t_ship:9.3f} ms")
    print(f"  G5.MEGA (scalar)        : {t_mega:9.3f} ms   x{t_ship / t_mega:.3f}")
    print(f"\n  {'WIN' if t_mega < t_ship else 'slower -- Phase 1 optimisation needed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
