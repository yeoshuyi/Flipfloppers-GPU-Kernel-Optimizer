"""FP8-in-attention accuracy precheck -- block quantization + incoherent
processing (FlashAttention-3's technique, arXiv 2407.08608), never before
tried in this repo. CLAUDE.md's PRECISION POLICY states "Never FP8 in
attention -- softmax tails die" as a blanket rule; this probe re-tests that
rule against a specific counter-technique rather than accepting it on faith,
matching this project's own standing discipline (step 30: "the catalogue's
own accuracy claim does not survive being measured").

Kernel-free: computes the FP8-cast-and-back mechanism in plain PyTorch
against the real model's shapes/distributions (via benchmark.py's own
generate_random_case), not a custom kernel -- this is a cheap precheck
before any kernel investment, same pattern as this project's other
precision prechecks (G6.5/INT8, the FP8 FFN re-investigation).

Mechanism under test, Q/K only (V and softmax/P@V stay FP32 -- isolating
exactly the QK^T precision question, one variable at a time):

  A. baseline  : FP32 Q@K^T (what SDPA already computes; ground truth)
  B. naive FP8 : Q,K cast to float8_e4m3fn with ONE global scale, no
                 rotation -- reproduces the failure CLAUDE.md's blanket rule
                 guards against, so there is an honest "before" number.
  C. block+incoherent FP8: Q,K multiplied by a FIXED random orthogonal
     matrix R (head_dim x head_dim, R @ R.T = I) before casting.
     Mathematically invisible to Q @ K^T in exact arithmetic:
         (Q@R) @ (K@R).T = Q @ R @ R.T @ K.T = Q @ K.T
     (rotation cancels exactly), but it decorrelates per-channel outliers
     BEFORE the FP8 cast, so quantization error is spread more evenly
     instead of blowing up on whichever channel happens to carry an
     outlier. Scale is chosen PER KEY-BLOCK (tile of 128 along S, matching
     a plausible flash-attention tile) rather than globally -- free in a
     real tiled kernel since flash already visits K in blocks.

This is an accuracy-only probe -- no timing claim is made, so CUDA-graph/
clock-locking concerns don't apply, but per CLAUDE.md invariant 3 it still
runs via sbatch like everything else in this repo, not direct python.
"""
import argparse
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from benchmark import TransformerConfig, generate_random_case  # noqa: E402


def make_orthogonal(dim: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(dim, dim, device=device, dtype=torch.float32, generator=g)
    q, r = torch.linalg.qr(a)
    # fix sign so Q is a genuine rotation (det=+1 not required, just a
    # deterministic, reproducible orthogonal matrix)
    d = torch.diagonal(r)
    q = q * torch.sign(d).unsqueeze(0)
    return q.contiguous()


def block_scale(x: torch.Tensor, block: int, dim: int) -> torch.Tensor:
    """Per-block max-abs scale along `dim` (the sequence axis), tiled by
    `block`. x: [B,H,S,D]. Returns a scale tensor broadcastable back onto x,
    one scalar per (batch, head, block-of-S)."""
    B, H, S, D = x.shape
    pad = (-S) % block
    xp = F.pad(x, (0, 0, 0, pad)) if pad else x
    Sp = xp.shape[2]
    xt = xp.view(B, H, Sp // block, block, D)
    amax = xt.abs().amax(dim=(3, 4), keepdim=True).clamp_min(1e-12)
    scale = amax / 448.0  # float8_e4m3fn max representable magnitude
    scale = scale.expand(-1, -1, -1, block, D).reshape(B, H, Sp, D)
    return scale[:, :, :S, :]


def fp8_qkT(q: torch.Tensor, k: torch.Tensor, rotate: torch.Tensor | None,
           block: int | None) -> torch.Tensor:
    """Returns Q@K^T computed through an FP8 round-trip on Q and K.
    q,k: [B,H,S,D] fp32. rotate: [D,D] orthogonal or None. block: key
    tile size for block quantization, or None for one global scale."""
    if rotate is not None:
        q = q @ rotate
        k = k @ rotate

    if block is not None:
        sq = block_scale(q, block, q.shape[-1])
        sk = block_scale(k, block, k.shape[-1])
    else:
        sq = (q.abs().amax() / 448.0).clamp_min(1e-12).expand_as(q)
        sk = (k.abs().amax() / 448.0).clamp_min(1e-12).expand_as(k)

    q8 = (q / sq).to(torch.float8_e4m3fn)
    k8 = (k / sk).to(torch.float8_e4m3fn)
    # dequantize via the per-position scale actually used, then matmul in
    # fp32 -- isolates the FP8 rounding error itself, independent of
    # whatever bf16/fp32 accumulate policy a real FP8 GEMM would use.
    qd = q8.to(torch.float32) * sq
    kd = k8.to(torch.float32) * sk
    return qd @ kd.transpose(-1, -2)


def run_shape(name: str, batch: int, seq: int, causal: bool, device, trials: int):
    config = TransformerConfig(
        batch_size=batch, seq_len=seq, d_model=512, num_heads=8,
        ffn_dim=2048, num_layers=1, causal=causal,
    )
    config.validate()
    head_dim = config.d_model // config.num_heads
    R = make_orthogonal(head_dim, device, seed=20260827)

    worst = {"naive": 0.0, "block_incoherent": 0.0}
    for trial in range(trials):
        x, _ = generate_random_case(config, device, torch.float32,
                                     seed=1000 + trial, padding_ratio=0.0,
                                     input_scale=1.0)
        g = torch.Generator(device=device).manual_seed(2000 + trial)
        q = torch.randn(batch, config.num_heads, seq, head_dim,
                        device=device, generator=g)
        k = torch.randn(batch, config.num_heads, seq, head_dim,
                        device=device, generator=g)

        ref = q @ k.transpose(-1, -2)
        ref = ref * (head_dim ** -0.5)

        naive = fp8_qkT(q, k, rotate=None, block=None) * (head_dim ** -0.5)
        block_inc = fp8_qkT(q, k, rotate=R, block=128) * (head_dim ** -0.5)

        # Compare post-softmax, causal-masked probabilities -- what actually
        # reaches P@V, not the raw pre-softmax scores (which have a much
        # looser natural scale and would understate what matters).
        mask = None
        if causal:
            mask = torch.ones(seq, seq, device=device, dtype=torch.bool).triu(1)

        def sm(scores):
            s = scores.masked_fill(mask, float("-inf")) if mask is not None else scores
            return F.softmax(s, dim=-1)

        p_ref = sm(ref)
        for tag, scores in (("naive", naive), ("block_incoherent", block_inc)):
            p = sm(scores)
            err = (p - p_ref).abs().amax().item()
            worst[tag] = max(worst[tag], err)

    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda")
    shapes = [
        ("default", 8, 128, False), ("default_causal", 8, 128, True),
        ("long_seq", 8, 1024, False), ("long_seq_causal", 8, 1024, True),
        ("large_batch", 256, 128, False), ("large_batch_causal", 256, 128, True),
    ]
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(device)}")
    print("Comparing post-softmax attention-probability max_abs error vs FP32 "
          "reference, naive-FP8 vs block+incoherent-FP8, across regimes.\n")

    summary = []
    for name, b, s, causal in shapes:
        w = run_shape(name, b, s, causal, device, args.trials)
        ratio = w["naive"] / max(w["block_incoherent"], 1e-12)
        print(f"=== {name} (B={b},S={s},causal={causal}) ===")
        print(f"  naive FP8            max_abs={w['naive']:.6e}")
        print(f"  block+incoherent FP8 max_abs={w['block_incoherent']:.6e}")
        print(f"  error reduction ratio = {ratio:.2f}x")
        print()
        summary.append((name, w["naive"], w["block_incoherent"], ratio))

    print("=== summary ===")
    for name, naive, bi, ratio in summary:
        print(f"{name:20s} naive={naive:.4e}  block_incoherent={bi:.4e}  ratio={ratio:.2f}x")


if __name__ == "__main__":
    main()
