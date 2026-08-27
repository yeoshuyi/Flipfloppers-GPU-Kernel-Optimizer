# Causal-Path Optimization Ledger

Fast-read index of every causal-path (`config.causal=True`) optimization
attempted or planned. `docs/PROGRESS.md` has the full narrative for the
overall (mostly non-causal) session; this file is causal-specific and kept
short on purpose so a fresh session can orient in one read. Append a row
per iteration, same discipline as `PROGRESS.md`'s numbered steps.

Accuracy budget: `atol=0.002, rtol=0.02` (disjunctive), the real enforced
default in `benchmark.py --atol`/`--rtol`. Non-causal code and archive
cells are explicitly out of scope for this ledger and untouched by any of
the work below — see `CLAUDE.md`'s OFFICIAL CAUSAL EVALUATION MATRIX
section for that boundary.

| Tag | Status | Speedup (causal shape) | max_abs | Notes |
|---|---|---|---|---|
| `G0.1c` | SHIPPED | 1.80x @ default | 0.00134 | SDPA (`EFFICIENT_ATTENTION`, forced via `sdpa_kernel`) replaces baseline's manual matmul/mask/softmax/matmul loop. `MATH` also passes; `FLASH`/`CUDNN` have no FP32 kernel on this stack. |
| `G1.1c` | SHIPPED, flat | 1.80x @ default (no change) | 0.00134 (bit-identical) | norm2-affine fold into `_ffn_in_weight`/`_bias`. Exact, zero risk. No speed effect at this shape — CUDA graphs (`G2.4b`/`G0.1c`) already absorbed the launch overhead this would have saved. |
| `G6.4a_v2c` | SHIPPED | 2.01x/1.96x @ default/padded | 0.00141 | FFN-in in FP16, cast back to FP32 immediately (ffn_out/GELU stay exact). The real win of this pass. |
| `G0.2c` | SHIPPED, near-miss on speed | 1.99x @ default (flat, logged not elite) | 0.00141 (bit-identical) | QKV fused GEMM + scale-fold (provably bit-identical, `head_dim**-0.5=2^-3`) + norm1-affine-fold (empirically verified independently for causal). Correct and harmless; same launch-overhead ceiling as `G1.1c`. |
| shape sweep | VALIDATED | tiny 5.84x/5.71x, long-seq 3.99x/3.08x, large-batch 2.11x/2.03x | 0.00128-0.00163 | First causal-path validation beyond `B=8,S=128`. Proves the stack above generalizes — `default_causal`'s flat last two iterations were a property of that one small, launch-overhead-dominated shape, not a sign the stack is maxed out. |
| `G4.4` (mma.sync PTX) | DEAD END | n/a | n/a | Pure speed-gate miss (1.207x vs 1.3x gate at large_batch, non-causal). Never touched causal. No budget change revives a speed-gate miss. |
| `G4.5` (CuAssembler/SASS) | DEAD END | n/a | n/a | CUDA 13.1 `nvdisasm` vs CuAssembler fork cubin ELF container-format incompatibility (`.note.nv.tkinfo`). Toolchain blocker, not numerics. Never touched causal. |
| BF16/FP8/INT8 (causal) | DEAD END, not re-tested | n/a | n/a | Fails 5x-39x over the new budget per already-logged non-causal numbers; causal's attention reduction depth is smaller than the FFN's K=2048 where these already failed most narrowly — no mechanism to do better. |
| `G6.6c` (cuBLASLt, causal-tiny FFN) | PENDING | — | — | `_LT_MAX_TOKENS=127` gates this to `tok<=127` — only causal-tiny (`B=1,S=64`, tok=64) qualifies. Non-causal precedent (step 33): 1.32-1.49x at the FFN via a TF32 split-K variant cuBLASLt's heuristic finds and PyTorch's default misses. Mechanically small: reuse `_ensure_lt_plan`/`_build_lt_plan`, wire into `_optimized_forward_causal`'s FFN branch. |
| `G6.4bc` (FP16 attention, causal) | PENDING | — | — | Cast Q/K/V/out_proj to FP16 around SDPA, cast back immediately (same pattern as non-causal `G6.4b` and this session's `G6.4a_v2c`). Prerequisite for `G4.6c`. |
| `G4.6c` (CUTLASS FP16-accum, causal-large-batch) | PENDING, gated on `G6.4bc` | — | — | Extension + configs (`CFG_QKV=6`, `CFG_OUT=18`) already re-validated at the new budget for non-causal (`probes/g4_6_cutlass_phase2b_newbudget.py`, all 8 shapes pass). Unreachable for causal until `G6.4bc` makes causal's QKV/out_proj GEMMs FP16. Target only causal-large-batch (where non-causal history shows the win). Tightest accuracy margin of anything in this ledger — validate with care. |

## Current elite (per causal shape, `causal/fp16` family unless noted)

- default: **2.01x** (`G6.4a_v2c`, `archive/causal__fp16.json`)
- tiny: **5.84x** (validated, not yet archived as its own cell — see Plan Part 1)
- long-seq: **3.99x** (validated, not yet archived as its own cell)
- large-batch: **2.11x** (validated, not yet archived as its own cell)

## Order of remaining work (see the approved plan for full detail)

1. Archive the four causal-shape numbers above (Plan Part 1 — needs a
   `tools/archive.py` `REGIMES` schema extension: `causal-tiny`,
   `causal-long-seq`, `causal-large-batch`).
2. `G6.6c` — cuBLASLt for causal-tiny's FFN.
3. `G6.4bc` — FP16 attention for causal.
4. `G4.6c` — CUTLASS FP16-accumulate for causal-large-batch (gated on step 3).
