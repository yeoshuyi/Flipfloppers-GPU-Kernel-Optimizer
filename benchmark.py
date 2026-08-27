#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.002 and rtol=0.02 (2%), per the updated
grading spec relayed 2026-08-27 (communicated verbally, citing direct
judge correspondence -- not independently verified against a written
document in this environment; see docs/PROGRESS.md's accuracy-policy
update for the full provenance note). The original, more conservative
thresholds (atol=0.001, rtol=0.01) remain available via --atol/--rtol and
are still the bar this project defaults its own *engineering* validation
to wherever practical, since anything passing the tighter bar automatically
passes this one too.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# --------------------------------------------------------------------------
# G6.6: explicit cuBLASLt algorithm + BIAS epilogue for the two FFN GEMMs.
#
# Fact cited (docs/PROGRESS.md step 31): the FFN's TF32 CUTLASS GEMM
# (ffn_in/ffn_out, plain F.linear) runs at ~47% of TF32 peak with ~26%
# occupancy on the default shape.
#
# What the probes actually found (runs 71/72/73):
#   * Algorithm selection alone is a NEGATIVE at the default shape -- the best
#     of 8 cublasLtMatmulAlgoGetHeuristic candidates beats PyTorch's own pick
#     by 1.001x. cuBLASLt's default heuristic is already right there.
#   * The real gap is PyTorch's BIAS path. Same shape, same algorithm:
#     F.linear(x, w) is 32.37us, F.linear(x, w, b) is 44.62us, and cuBLASLt
#     with CUBLASLT_EPILOGUE_BIAS is 32.61us. The bias costs PyTorch 12.2us
#     and costs cuBLASLt 0.2us. It is not bandwidth (ffn_in's output is 4x
#     larger and its bias costs only 2.5us) -- PyTorch selects a different,
#     slower kernel when a bias is present.
#   * At TINY (M=64) algorithm selection IS worth 1.32x/1.49x, via split-K
#     variants the default heuristic does not pick, and the measurement is
#     genuinely GPU-bound (cpu issue 4.3us vs 7.7-30us of kernel time).
#   * Writing `F.linear(x, w) + b` instead is NOT a fix: 1.08x at M=1024 but
#     0.77x at M=8192/32768.
#
# This is deliberately NOT step 25's max-autotune failure repeated: every
# candidate here comes from cuBLASLt's own heuristic and runs the same native
# TF32 tensor-core datapath PyTorch already dispatches into. Triton's
# ALLOW_TF32 3-pass FP32 decomposition -- the thing that broke max-autotune --
# is never in the candidate set. Measured maxdiff against F.linear is 6.7e-6
# at M=1024 and bit-identical at M=8192/32768.
#
# Loaded lazily and defensively: any failure (no compiler, no cuBLASLt, older
# CUDA) leaves _LT_EXT as None and every call site falls back to F.linear.
# --------------------------------------------------------------------------
_LT_EXT = None
_LT_EXT_TRIED = False
_LT_OP_READY = False

# Regime gate, a baked constant, not a runtime autotune. This is CLAUDE.md's
# own TINY boundary (B*S < 128), and it is where the END-TO-END measurement put
# the win -- not where the isolated block probe predicted it:
#
#   shape    tok    optimized ms, run63 -> run74     block-probe prediction
#   tiny      64      0.2390 -> 0.1980  (1.21x)            1.47x
#   default 1024      0.6502 -> 0.6483  (1.003x, noise)    1.19x
#
# The default-shape prediction did not survive contact with the real model, and
# run 75 shows why it is not a mis-fire: the plan DID fire at tok=1024 (algos
# 0/0) and produced bit-identical output for no time. Run 73's baseline was
# eager ops captured in a raw CUDA graph, which still paid PyTorch's addmm bias
# penalty; inductor's lowering of the whole model already avoids it, so at the
# default shape there was never 1.19x on the table. What survives is TINY,
# where the win comes from split-K algorithms (run 75 picked idx 4/1 = splitk
# 4 and 16) that cuBLASLt's default heuristic does not choose and that are
# worth 1.32x/1.49x on the GEMMs themselves (run 71).
#
# Scoped to TINY deliberately rather than left at a wider bound: selection is
# by speed alone, so a different cuBLAS build could pick a split-K variant at a
# larger shape and shift max_abs there for no measured gain. Confining it to
# the one regime that pays confines that risk too.
_LT_MAX_TOKENS = 127
_LT_REQUESTED = 16          # cublasLtMatmulAlgoGetHeuristic candidate count
_LT_WS_BYTES = 32 * 1024 * 1024
_LT_WARMUP = 5              # one-time eager calibration, ~20ms total
_LT_ITERS = 30


def _lt_ext():
    """Build/load the cuBLASLt extension once. None means 'not available'."""
    global _LT_EXT, _LT_EXT_TRIED
    if _LT_EXT_TRIED:
        return _LT_EXT
    _LT_EXT_TRIED = True
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "csrc", "cublaslt_algo.cpp")
    if not os.path.exists(src) or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        build_dir = os.environ.get(
            "TORCH_EXT_BUILD_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ext_build"))
        os.makedirs(build_dir, exist_ok=True)
        # cpp_extension.load() needs a writable, container-visible TMPDIR;
        # the container default is not usable (docs/PROGRESS.md step 32).
        os.environ.setdefault("TMPDIR", build_dir)
        # with_cuda=True is mandatory even with no .cu source -- torch infers
        # CUDA only from the .cu extension, so without it <cuda_runtime.h> and
        # -lcudart are both missing (docs/PROGRESS.md step 32).
        _LT_EXT = load(name="cublaslt_algo", sources=[src],
                       build_directory=build_dir, with_cuda=True,
                       extra_ldflags=["-lcublasLt"], verbose=False)
    except Exception:                                        # noqa: BLE001
        _LT_EXT = None
    return _LT_EXT


def _lt_register_op() -> bool:
    """Wrap the extension in a torch.library custom op.

    Needed because the call site lives inside torch.compile(mode=
    "reduce-overhead"): a bare pybind11 call would graph-break, and a
    graph break in that region costs more than the GEMM saves. As a
    registered op with a fake impl, dynamo traces through it and inductor
    emits it as an extern call inside the same CUDA graph.
    """
    global _LT_OP_READY
    if _LT_OP_READY:
        return True
    ext = _lt_ext()
    if ext is None:
        return False
    try:
        @torch.library.custom_op("g66::lt_linear", mutates_args=())
        def lt_linear(inp: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                      pid: int, idx: int) -> torch.Tensor:
            try:
                return ext.lt_linear(pid, idx, inp, w, bias)
            except Exception:                                # noqa: BLE001
                # A problem is created for one fixed M; if dynamo ever
                # re-specialises the batch dimension the C++ side rejects the
                # call rather than computing the wrong thing. Fall back to the
                # exact op this replaces instead of aborting the run.
                return F.linear(inp, w, bias)

        @lt_linear.register_fake
        def _(inp, w, bias, pid, idx):
            return inp.new_empty((inp.shape[0], w.shape[0]))

        _LT_OP_READY = True
    except Exception:                                        # noqa: BLE001
        _LT_OP_READY = False
    return _LT_OP_READY


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    """
    Requirements:
      1. Keep the forward signature unchanged.
      2. Return a tensor with shape [batch_size, seq_len, d_model].
      3. Keep compatible parameter names, or customize copy_model_weights().

    G0.1: replace the baseline's manual
    matmul -> mask -> softmax -> matmul attention with
    F.scaled_dot_product_attention. When there is no padding, the fast path
    uses is_causal only (attn_mask=None) so SDPA can pick its flash/efficient
    backend. When valid_token_mask carries real padding, an explicit boolean
    attn_mask is unavoidable for correctness -- SDPA falls onto its math
    backend for that call only, which is inherent to the PADDED regime, not a
    lazy default around is_causal.

    CAUSAL gate: probes/g0_1_causal_backend_probe.py showed that even SDPA's
    MATH backend (algorithmically identical to the baseline's manual
    matmul->mask->softmax->matmul, just a different fused kernel) drifts past
    the 1e-3 atol on ~half of random seeds at B8_S128 causal -- flash/cuDNN
    aren't available for FP32 at all. Baseline's own TF32-matmul output is
    the accuracy reference (not a mathematically exact answer), so an
    independently-kernelled causal path can't be made to reliably match it
    within tolerance at this depth. Gate on config.causal (checked once, not
    per layer, per CLAUDE.md) and fall back to the exact baseline computation
    for causal until G1.2 (scale folding into W_Q) is in and the gap is
    re-measured.

    G0.2: fuse q_proj/k_proj/v_proj into one [d,3d] matmul
    instead of three separate [d,d] matmuls -- one GEMM launch instead of
    three, and a bigger GEMM has better arithmetic intensity than three small
    ones. The fused weight is built lazily on first forward (after
    load_state_dict + .to(device,dtype) + .eval(), so it inherits the right
    device/dtype for free) and cached as a PLAIN ATTRIBUTE on the attention
    submodule -- never nn.Parameter/nn.Buffer, so strict=True state_dict
    loading still only ever sees q_proj/k_proj/v_proj (CLAUDE.md invariant
    4). This caches a deterministic function of the frozen WEIGHTS, not of
    the input x or the output -- weights never change between calls, so
    there's nothing stale to return (CLAUDE.md invariant 1 is about caching
    on x.data_ptr()/output, not about this).

    G0.3: _split_heads_view() below skips the .contiguous() copy that
    BaselineSelfAttention._split_heads() does after view+transpose.
    Baseline needs that copy because its own torch.matmul calls want
    contiguous inputs; SDPA's fused kernels are designed to accept the
    strided [B,H,S,D] view directly (that view+transpose pattern is the
    standard MHA idiom SDPA is built around). Saves ~3 copies of
    [B,S,d_model] per layer on the way in. The baseline's own
    _split_heads is left untouched since it's shared with the frozen
    reference's manual-matmul forward, which genuinely needs the copy.

    G0.5: _mask_is_all_ones() below is the sanctioned data_ptr() cache from
    CLAUDE.md -- caches whether a mask is all-ones, never a result. Needed
    because generate_random_case() always hands back a concrete all-ones
    tensor when there's no real padding, never a literal None, so the
    unpadded shapes were silently going through the masked SDPA branch (and
    the now-provably-no-op masked_fill calls) the whole time. See forward().

    G1.2: scale folded into W_Q/b_Q inside _fused_qkv() (see the comment
    there for why it's bit-exact for this model), SDPA called with
    scale=1.0. Note this only covers the non-causal path -- the causal
    fallback still uses baseline's own unscaled q_proj/k_proj/v_proj via
    super().forward(), untouched by this fold.

    G1.1: norm1's and norm2's affine (gamma/beta) are folded into the
    linears that consume their output (_fused_qkv, _fused_ffn_in) --
    LayerNorm itself only ever computes the pure reduction now
    (F.layer_norm with weight=None, bias=None). final_norm is left alone:
    its output is the model's return value, there's no downstream linear to
    fold into (docs/CATALOGUE.md's own note on this). Unlike G1.2's
    power-of-two scale, gamma/beta are arbitrary learned floats, so this
    fold is exact in real arithmetic but not provably bit-identical in
    floating point -- verified empirically (max_abs comparison) instead,
    see PROGRESS.md.

    G2.4: forward() computes no_pad, ensures every layer's folded
    weights exist (_ensure_folded_weights, EAGER -- see
    _build_qkv_fold's docstring for why this can never move inside the
    compiled call), and self.config.causal handling itself, then
    delegates the actual per-layer loop to _optimized_forward(), lazily
    wrapped in torch.compile(mode="reduce-overhead") on first call.
    Causal still bypasses this entirely via super().forward() before
    compilation is even considered -- this diff is scoped to the
    already-optimized non-causal path only.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._mask_cache: dict = {}
        self._compiled_impl = None
        self._compiled_causal = None
        # G6.6: token-count -> (pid_in, algo_in, pid_out, algo_out) or None.
        # Keyed on a SHAPE, built from the frozen weights; never an input or
        # an output, and never a data_ptr.
        self._lt_plan: dict = {}
        self._lt_cur = None
        # G6.4b: FP16 for attention only (FFN stays exact TF32 -- G6.4a
        # closed that direction, PROGRESS.md step 27). Same accumulation-
        # precision guard as G6.4a; harmless here even though SDPA's own
        # flash/efficient kernels manage their own internal accumulation,
        # not this flag's plain-matmul path.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    @staticmethod
    def _build_qkv_fold(
        attn: "BaselineSelfAttention", norm1: nn.LayerNorm,
        device: torch.device, dtype: torch.dtype,
    ) -> None:
        # G0.2+G1.1+G1.2 folded into one weight/bias pair, cached as plain
        # attributes on attn. MUST run eagerly, never traced inside the
        # torch.compile'd region (G2.4): building a fresh tensor via
        # torch.cat inside a cudagraph'd function and then caching a
        # Python-level reference to it across calls hands out a pointer
        # into the graph's internal memory pool, which the NEXT replay
        # reclaims -- PyTorch correctly detects and raises on this
        # ("accessing tensor output of CUDAGraphs that has been
        # overwritten"), caught in this iteration's smoke test. Called from
        # forward() before the compiled call, never from inside
        # _optimized_forward.
        w = getattr(attn, "_qkv_weight", None)
        if w is not None and w.device == device and w.dtype == dtype:
            return
        w = torch.cat(
            [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
        )
        b = torch.cat(
            [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0
        )
        # G1.2: fold the attention scale into W_Q/b_Q (only Q's rows of
        # the fused weight -- torch.cat already returns fresh storage,
        # so this in-place scale can't touch attn.q_proj.weight itself).
        # Exact: this model's head_dim is always a power of two (64
        # here), so scale = head_dim**-0.5 is an exact power of two
        # (0.125 = 2^-3) -- IEEE float multiplication by a power of two
        # never rounds, so pre-scaling Q here and passing scale=1.0 to
        # SDPA is bit-identical to SDPA's default (scaling the [B,H,S,S]
        # score matrix by 0.125 after the matmul), not merely
        # "close enough." Verified empirically in PROGRESS.md step 12.
        d = attn.d_model
        w[:d].mul_(attn.scale)
        b[:d].mul_(attn.scale)
        # G1.1: absorb norm1's affine (gamma=weight, beta=bias) into
        # this (already scale-folded) weight/bias, so norm1 itself only
        # ever needs to do the mean/var reduction. y = n*gamma+beta,
        # z = y@W^T+b = n@(W*gamma)^T + (b + W@beta) -- the bias-absorb
        # must happen BEFORE the column-scale below, using this W (the
        # W@beta term doesn't involve gamma at all).
        b = b + w @ norm1.bias
        w = w * norm1.weight[None, :]
        attn._qkv_weight = w
        attn._qkv_bias = b

    @staticmethod
    def _build_attn_fp16_fold(attn: "BaselineSelfAttention", device: torch.device) -> None:
        # G6.4b: FP16 copies of the QKV and out_proj weights, built from
        # the already-folded FP32 _qkv_weight/_bias (norm1's affine + the
        # scale fold stay intact) and the untouched out_proj.weight/bias.
        # Same eager-only rule as every other weight cache this session.
        w = getattr(attn, "_qkv_weight_fp16", None)
        if w is not None and w.device == device:
            return
        attn._qkv_weight_fp16 = attn._qkv_weight.to(torch.float16)
        attn._qkv_bias_fp16 = attn._qkv_bias.to(torch.float16)
        attn._out_proj_weight_fp16 = attn.out_proj.weight.to(torch.float16)
        attn._out_proj_bias_fp16 = attn.out_proj.bias.to(torch.float16)

    @staticmethod
    def _build_ffn_in_fold(
        layer: "BaselineTransformerBlock", device: torch.device, dtype: torch.dtype,
    ) -> None:
        # G1.1, the norm2 -> ffn_in half. Same eager-only rule as above.
        w = getattr(layer, "_ffn_in_weight", None)
        if w is not None and w.device == device and w.dtype == dtype:
            return
        ffn_in = layer.ffn_in
        b = ffn_in.bias + ffn_in.weight @ layer.norm2.bias
        w = ffn_in.weight * layer.norm2.weight[None, :]
        layer._ffn_in_weight = w
        layer._ffn_in_bias = b
        # G6.4a v2: FP16 ffn_in only (ffn_out stays exact FP32/TF32 -- both-
        # GEMM FP16 failed 6/6 shapes at 40-seed rigor, PROGRESS.md step 27
        # v1). Closed at the OLD 0.001/0.01 budget by a near-miss (4/6 shapes
        # failed by rare single-to-double-digit element counts against
        # 1.3M-671M element tensors -- step 27 v2). Re-verified at the new
        # 0.002/0.02 default (40 seeds, all 6 non-causal shapes): clean pass,
        # max_abs 0.00084-0.00100 across every shape, well inside budget --
        # see docs/PROGRESS.md's Phase 2.5 update for the full log.
        layer._ffn_in_weight_fp16 = w.to(torch.float16)
        layer._ffn_in_bias_fp16 = b.to(torch.float16)

    def _ensure_folded_weights(self, device: torch.device, dtype: torch.dtype) -> None:
        for layer in self.layers:
            self._build_qkv_fold(layer.attention, layer.norm1, device, dtype)
            self._build_attn_fp16_fold(layer.attention, device)
            self._build_ffn_in_fold(layer, device, dtype)

    @staticmethod
    def _time_eager(fn, warmup: int = _LT_WARMUP, iters: int = _LT_ITERS) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters

    def _build_lt_plan(self, tok: int, device: torch.device):
        """G6.6. Pick a cuBLASLt algorithm for each of the two FFN GEMMs at this
        token count, or return None to keep F.linear.

        EAGER ONLY -- same rule and same reason as _build_qkv_fold: this runs a
        real timing loop and caches Python ints, neither of which may happen
        inside the cudagraph'd region.

        Two gates, both documented per CLAUDE.md's validity test:
          1. tok <= _LT_MAX_TOKENS (a baked constant, see its definition).
          2. the chosen algorithm must actually beat F.linear on this machine
             for BOTH halves, measured here, once. This is a one-time eager
             calibration with a correct fallback (F.linear), not per-call
             branching -- it exists so an unforeseen shape or a different
             cuBLAS build can never turn this into a regression.
        Caches a property of the SHAPE and the frozen weights. No input tensor,
        no output, and no data_ptr is involved.
        """
        ext = _lt_ext()
        if ext is None:
            return None
        layer0 = self.layers[0]
        w1, b1 = layer0._ffn_in_weight, layer0._ffn_in_bias
        w2, b2 = layer0.ffn_out.weight, layer0.ffn_out.bias
        d_model, ffn_dim = w1.shape[1], w1.shape[0]

        a1 = torch.randn(tok, d_model, device=device, dtype=torch.float32)
        a2 = torch.randn(tok, ffn_dim, device=device, dtype=torch.float32)
        o1 = torch.empty(tok, ffn_dim, device=device, dtype=torch.float32)
        o2 = torch.empty(tok, d_model, device=device, dtype=torch.float32)

        chosen = []
        for K, N, inp, w, b, out in ((d_model, ffn_dim, a1, w1, b1, o1),
                                     (ffn_dim, d_model, a2, w2, b2, o2)):
            pid = ext.create_problem(tok, N, K, True, _LT_WS_BYTES, _LT_REQUESTED)
            best = None
            for i in range(ext.num_algos(pid)):
                try:
                    t = ext.time_algo(pid, i, inp, w, b, out,
                                      _LT_WARMUP, _LT_ITERS)
                except Exception:                            # noqa: BLE001
                    continue
                if best is None or t < best[1]:
                    best = (i, t)
            if best is None:
                return None
            ref = self._time_eager(lambda: F.linear(inp, w, b))
            if best[1] >= ref:
                return None                                   # gate 2 failed
            chosen.append((pid, best[0]))
        return (chosen[0][0], chosen[0][1], chosen[1][0], chosen[1][1])

    def _ensure_lt_plan(self, tok: int, device: torch.device,
                        dtype: torch.dtype) -> None:
        # Sets self._lt_cur, read as a compile-time constant by the traced
        # region below. Must run eagerly, before the compiled call.
        if (dtype != torch.float32 or device.type != "cuda"
                or tok > _LT_MAX_TOKENS):
            self._lt_cur = None
            return
        if tok in self._lt_plan:
            self._lt_cur = self._lt_plan[tok]
            return
        plan = None
        if _lt_register_op():
            try:
                plan = self._build_lt_plan(tok, device)
            except Exception:                                # noqa: BLE001
                plan = None
        self._lt_plan[tok] = plan
        self._lt_cur = plan

    @staticmethod
    def _split_heads_view(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # G0.3: no .contiguous() -- unlike baseline's own torch.matmul,
        # SDPA's fused kernels accept the strided [B,H,S,D] view directly.
        # (BaselineSelfAttention._split_heads is left untouched: it's shared
        # with the frozen baseline's own manual-matmul forward, which does
        # need the copy for its plain torch.matmul calls.)
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    def _mask_is_all_ones(self, m: Optional[torch.Tensor]) -> bool:
        # G0.5: caches a PROPERTY of the mask (is it all-ones?), never the
        # output -- the one data_ptr() use CLAUDE.md sanctions. The harness
        # reuses one mask tensor across an entire benchmark_models() call
        # (300+ timed forwards), so this turns a per-call .all() reduction
        # into a one-time cost; a mask from a different call is a different
        # allocation (different data_ptr), so there's nothing stale here.
        if m is None:
            return True
        key = (m.data_ptr(), tuple(m.shape))
        hit = self._mask_cache.get(key)
        if hit is None:
            hit = bool(m.all())
            self._mask_cache[key] = hit
        return hit

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.causal:
            # G0.1c: SDPA (EFFICIENT_ATTENTION backend, forced explicitly --
            # automatic dispatch does not pick it for FP32) replaces the
            # manual matmul/mask/softmax/matmul loop for causal too, same
            # lever as G0.1 for non-causal. Previously closed on accuracy
            # (see docstring above) under the OLD 0.001/0.01 budget --
            # re-verified clean at the new 0.002/0.02 default, 40 seeds,
            # both causal shapes, 0 failures (probes/g0_1_causal_sdpa_newbudget.py,
            # MATH and EFFICIENT backends both pass; FLASH/CUDNN error out,
            # no FP32 kernel available on this stack -- expected, not an
            # accuracy question). No folded weights yet -- q_proj/k_proj/
            # v_proj/out_proj/norm1/norm2/ffn_in/ffn_out all still read
            # directly, isolating the attention-kernel change as the only
            # variable this iteration (one optimisation per iteration).
            # Same CUDA-graph lever as G2.4/G2.4b on top, still lazy.
            no_pad_causal = self._mask_is_all_ones(valid_token_mask)
            if self._compiled_causal is None:
                self._compiled_causal = torch.compile(
                    self._optimized_forward_causal, mode="reduce-overhead"
                )
            return self._compiled_causal(x, valid_token_mask, no_pad_causal)

        # G0.5: generate_random_case() always hands back a concrete all-ones
        # tensor when there's no real padding (never a literal None) -- so
        # without this check, every "unpadded" shape was still building and
        # passing a real attn_mask to SDPA below. no_pad short-circuits that
        # for the common case and skips the now-provably-no-op masked_fill
        # calls (an all-ones keep-mask changes nothing, but still costs a
        # full elementwise pass over the tensor if not skipped).
        #
        # Computed HERE, outside the compiled region below (G2.4): if
        # _mask_is_all_ones's data_ptr() call happened inside traced code,
        # dynamo would bake that address into the graph's guards and force
        # a recompile on every new tensor allocation -- silently defeating
        # the whole point of graph reuse. As a plain bool argument it's just
        # a specialization dimension (at most 2 graphs: no_pad True/False).
        no_pad = self._mask_is_all_ones(valid_token_mask)

        # Must also happen eagerly, outside the compiled call -- see
        # _build_qkv_fold's docstring for why.
        self._ensure_folded_weights(x.device, x.dtype)
        # G6.6: same eager-only rule -- picks the cuBLASLt algorithm for this
        # shape's two FFN GEMMs (or leaves F.linear in place). Reads the folded
        # weights above, so it has to come after them.
        self._ensure_lt_plan(x.shape[0] * x.shape[1], x.device, x.dtype)

        # G2.4: torch.compile(mode="reduce-overhead") -- CUDA graphs, to
        # remove the remaining per-kernel launch overhead in the TINY
        # regime. Compiled LAZILY on first forward (never rely on
        # --compile-user; docs/CATALOGUE.md/CLAUDE.md are explicit the
        # grader may not pass it). No formal ncu profile exists yet for
        # this (see docs/PROGRESS.md) -- citing the measured fact instead:
        # tiny's speedup jumped far more than every other regime's at each
        # of G0.2/G0.3/G0.5 (fewer GEMMs, no .contiguous(), skipped
        # masked_fill), which is itself strong evidence tiny is still
        # launch/overhead-bound post-G0/G1, exactly what CUDA graphs
        # target -- consistent with CATALOGUE.md's own "3x+ tiny" estimate
        # for G2.4 specifically (vs "1.2x default").
        if self._compiled_impl is None:
            self._compiled_impl = torch.compile(
                self._optimized_forward, mode="reduce-overhead"
            )
        return self._compiled_impl(x, valid_token_mask, no_pad)

    def _optimized_forward_causal(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        no_pad: bool,
    ) -> torch.Tensor:
        # G0.1c: unfused causal path, SDPA in place of the manual
        # matmul/mask/softmax/matmul loop. Structurally mirrors
        # BaselineTransformerBlock.forward/BaselineSelfAttention.forward
        # exactly, so this stays byte-for-byte comparable to the reference
        # everywhere except the one kernel under test.
        for layer in self.layers:
            attn = layer.attention
            n1 = layer.norm1(x)
            q = self._split_heads_view(attn.q_proj(n1), attn.num_heads, attn.head_dim)
            k = self._split_heads_view(attn.k_proj(n1), attn.num_heads, attn.head_dim)
            v = self._split_heads_view(attn.v_proj(n1), attn.num_heads, attn.head_dim)

            if no_pad:
                # No padding: is_causal=True keeps EFFICIENT_ATTENTION
                # eligible (an explicit attn_mask would kick it off, per
                # CLAUDE.md trap #3).
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    context = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=True
                    )
            else:
                # Padding present: causal restriction and key validity both
                # folded into one explicit boolean mask (True = attend),
                # is_causal=False -- SDPA rejects is_causal=True together
                # with a real attn_mask.
                seq_len = x.shape[1]
                key_keep = valid_token_mask[:, None, None, :]
                causal_ok = ~torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).triu(diagonal=1)
                allow = key_keep & causal_ok[None, None, :, :]
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    context = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=allow, is_causal=False
                    )

            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(x.shape[0], x.shape[1], attn.d_model)
            )
            attn_out = attn.out_proj(context)
            if not no_pad:
                attn_out = attn_out.masked_fill(~valid_token_mask[..., None], 0)
            x = x + attn_out

            n2 = layer.norm2(x)
            ffn = layer.ffn_out(F.gelu(layer.ffn_in(n2), approximate="none"))
            x = x + ffn
            if not no_pad:
                x = x.masked_fill(~valid_token_mask[..., None], 0)

        x = self.final_norm(x)
        if not no_pad:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    def _optimized_forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        no_pad: bool,
    ) -> torch.Tensor:
        # G6.6: constant for the whole traced region -- either None (keep
        # F.linear) or the four ints naming the two chosen cuBLASLt algorithms.
        # Chosen eagerly in forward(); dynamo bakes it in as a guard, so this
        # is a specialisation dimension, never a per-call branch.
        lt = self._lt_cur
        for layer in self.layers:
            attn = layer.attention
            # G1.1: norm1's affine is folded into _qkv_weight/_qkv_bias
            # (built eagerly by _ensure_folded_weights before this
            # compiled call, see forward()), so here we only need the pure
            # reduction (no weight=gamma/bias=beta) -- F.layer_norm(
            # weight=None, bias=None) computes exactly (x-mean)/sqrt(var+
            # eps), PyTorch's own fused kernel. attn._qkv_weight/_bias are
            # read here as already-stable tensors, never built inside this
            # compiled function (see _build_qkv_fold's docstring).
            n1 = F.layer_norm(
                x, layer.norm1.normalized_shape, eps=layer.norm1.eps
            )
            # G6.4b: QKV projection + SDPA + out_proj in FP16, FFN stays
            # exact TF32 (G6.4a closed that direction). FP16 Q/K/V also
            # unlocks SDPA's flash/memory-efficient backends, which FP32
            # can't use at all (G0.1) -- a second, independent effect
            # layered onto the precision change itself.
            n1_fp16 = n1.to(torch.float16)
            qkv = F.linear(n1_fp16, attn._qkv_weight_fp16, attn._qkv_bias_fp16)
            q, k, v = qkv.split(attn.d_model, dim=-1)
            q = self._split_heads_view(q, attn.num_heads, attn.head_dim)
            k = self._split_heads_view(k, attn.num_heads, attn.head_dim)
            v = self._split_heads_view(v, attn.num_heads, attn.head_dim)

            if no_pad:
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=False, scale=1.0
                )
            else:
                key_keep = valid_token_mask[:, None, None, :]
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=key_keep, is_causal=False, scale=1.0
                )

            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(x.shape[0], x.shape[1], attn.d_model)
            )
            attn_out_fp16 = F.linear(
                context, attn._out_proj_weight_fp16, attn._out_proj_bias_fp16
            )
            attn_out = attn_out_fp16.to(torch.float32)
            if not no_pad:
                attn_out = attn_out.masked_fill(~valid_token_mask[..., None], 0)
            x = x + attn_out

            n2 = F.layer_norm(
                x, layer.norm2.normalized_shape, eps=layer.norm2.eps
            )
            if lt is None:
                # G6.4a v2: FP16 ffn_in, cast back to FP32 immediately (same
                # pattern as G6.4b's attn_out_fp16 above) so GELU and
                # ffn_out run in exact FP32/TF32, unchanged.
                n2_fp16 = n2.to(torch.float16)
                ffn_hidden_fp16 = F.linear(n2_fp16, layer._ffn_in_weight_fp16,
                                           layer._ffn_in_bias_fp16)
                ffn_hidden = ffn_hidden_fp16.to(torch.float32)
                ffn = layer.ffn_out(F.gelu(ffn_hidden, approximate="none"))
            else:
                # G6.6: same two GEMMs, same native TF32 tensor-core datapath,
                # but through cuBLASLt with an explicitly chosen algorithm and
                # a fused BIAS epilogue. n2 is contiguous (F.layer_norm's own
                # output), so both reshapes are views, not copies.
                pid_i, alg_i, pid_o, alg_o = lt
                n2f = n2.reshape(-1, n2.shape[-1])
                ffn_hidden = torch.ops.g66.lt_linear(
                    n2f, layer._ffn_in_weight, layer._ffn_in_bias, pid_i, alg_i
                )
                act = F.gelu(ffn_hidden, approximate="none")
                ffn = torch.ops.g66.lt_linear(
                    act, layer.ffn_out.weight, layer.ffn_out.bias, pid_o, alg_o
                ).view(n2.shape)
            x = x + ffn
            if not no_pad:
                x = x.masked_fill(~valid_token_mask[..., None], 0)

        x = self.final_norm(x)
        if not no_pad:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

    
