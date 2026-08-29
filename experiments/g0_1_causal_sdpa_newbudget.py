#!/usr/bin/env python3
"""
Plan Phase 2, Stage 0 -- fresh re-measurement of SDPA-for-causal vs the
0.001/0.01 closure this project originally hit (see
probes/g0_1_causal_backend_probe.py, docs/PROGRESS.md step 5), now against
the new accuracy default (atol=0.002, rtol=0.02, per the updated grading
spec relayed 2026-08-27 -- see docs/PROGRESS.md's accuracy-policy note for
provenance). This is the single highest-value check in the current plan: if
SDPA passes cleanly here, it reopens G0.1 (and everything downstream of it
-- fused QKV, the norm folds, FP16 attention, cuBLASLt) for the causal
regime, categorically bigger than any other item in this plan.

Two upgrades over the original probe, per the plan:
  - 20 -> 40 seeds (this project's established minimum rigor bar)
  - hand-rolled abs_err.max() -> benchmark.py's own compare_outputs() via
    run_accuracy_tests(), so the verdict is the real disjunctive
    (abs<=atol OR rel<=rtol) rule, not a proxy
  - BOTH causal shapes (default_causal, causal_padded), not just B=8,S=128
    unpadded -- the plan's stated validation bar for anything touching the
    causal path

Everything stays exact FP32 (no precision reduction) -- this isolates ONE
variable only, which SDPA backend computes the reduction, from any
precision question. That's a separate, later step (Stage 2B) if this
passes.

Baseline reference is completely untouched (a separate model instance).
Only the "optimized" model's attention modules get an instance-level
forward override (types.MethodType per attn instance, not a class-level
monkeypatch) so the reference computation this is being checked against
never changes.
"""
import argparse
import sys
import types

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/work")
import torch_transformer_benchmark as B  # noqa: E402

NEW_ATOL = 0.002
NEW_RTOL = 0.02
TRIALS = 40

BACKENDS = {
    "MATH": SDPBackend.MATH,
    "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "CUDNN": SDPBackend.CUDNN_ATTENTION,
}

SHAPES = [
    ("default_causal", dict(batch_size=8, seq_len=128, padding_ratio=0.0)),
    ("causal_padded",  dict(batch_size=8, seq_len=128, padding_ratio=0.3)),
]


def make_sdpa_forward(backend):
    def sdpa_forward(self, x, valid_token_mask=None, causal=False):
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if valid_token_mask is None:
            # No padding: is_causal=True keeps flash/efficient/cudnn
            # eligible (CLAUDE.md trap #3 -- an explicit attn_mask kicks
            # SDPA off flash).
            with sdpa_kernel([backend]):
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=causal
                )
        else:
            # Padding present: causal restriction and key-validity both
            # have to be expressed as one explicit boolean mask (True =
            # attend), is_causal=False -- SDPA does not accept both
            # is_causal=True and a real attn_mask together.
            key_keep = valid_token_mask[:, None, None, :]
            allow = key_keep.expand(batch, 1, seq_len, seq_len)
            if causal:
                causal_ok = ~torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).triu(diagonal=1)
                allow = allow & causal_ok[None, None, :, :]
            with sdpa_kernel([backend]):
                context = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allow, is_causal=False
                )

        context = (
            context.transpose(1, 2).contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
    return sdpa_forward


def build_models(config, device, dtype, backend):
    baseline = B.BaselineTransformer(config)
    optimized = B.BaselineTransformer(config)  # separate instance, same arch
    B.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    fwd = make_sdpa_forward(backend)
    for layer in optimized.layers:
        layer.attention.forward = types.MethodType(fwd, layer.attention)
    return baseline, optimized


def run_shape_backend(name, kwargs, backend):
    padding_ratio = kwargs.pop("padding_ratio", 0.0)
    device = torch.device("cuda")
    dtype = torch.float32

    config = B.TransformerConfig(
        batch_size=kwargs["batch_size"], seq_len=kwargs["seq_len"],
        d_model=512, num_heads=8, ffn_dim=2048, num_layers=6, causal=True)
    config.validate()

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    baseline, optimized = build_models(config, device, dtype, backend)
    passed = B.run_accuracy_tests(
        baseline=baseline, optimized=optimized, config=config,
        device=device, dtype=dtype, trials=TRIALS, seed=1234,
        padding_ratio=padding_ratio, input_scale=1.0,
        rtol=NEW_RTOL, atol=NEW_ATOL)
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS) + ["all"],
                    default="all")
    a = ap.parse_args()
    backends = BACKENDS if a.backend == "all" else {a.backend: BACKENDS[a.backend]}

    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}",
          flush=True)
    print(f"BUDGET = atol={NEW_ATOL} rtol={NEW_RTOL}  TRIALS={TRIALS}  "
          f"(Stage 0: SDPA-for-causal recheck under the new default)\n",
          flush=True)

    results = {}
    for bname, backend in backends.items():
        for sname, kwargs in SHAPES:
            print("\n" + "#" * 74)
            print(f"### BACKEND={bname}  SHAPE={sname}  {kwargs}")
            print("#" * 74, flush=True)
            try:
                passed = run_shape_backend(sname, dict(kwargs), backend)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: {type(e).__name__}: {e}")
                passed = False
            results[(bname, sname)] = passed

    print("\n" + "=" * 74)
    print(f"STAGE 0 SUMMARY  (atol={NEW_ATOL}, rtol={NEW_RTOL}, trials={TRIALS})")
    allp = True
    for (bname, sname), passed in results.items():
        print(f"  {bname:10s} {sname:16s} {'PASS' if passed else 'FAIL'}")
        allp &= passed
    print(f"\nVERDICT: {'ALL PASS -- G0.1 REOPENS FOR CAUSAL' if allp else 'at least one backend/shape still fails'}")
    return 0 if allp else 1


if __name__ == "__main__":
    sys.exit(main())
