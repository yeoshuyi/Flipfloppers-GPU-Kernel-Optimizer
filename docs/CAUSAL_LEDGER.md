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
| `G6.6c` (cuBLASLt, causal-tiny FFN) | TRIED, REVERTED | -8.6%/-5.8% @ causal-tiny (regression) | 40-seed sweep, not noise | Wired in exactly as planned (reused `_ensure_lt_plan`/`_build_lt_plan` verbatim). Real regression: cuBLASLt-TF32 beats plain `F.linear`-TF32 (the comparison `_build_lt_plan`'s gate 2 makes, and the one non-causal's step-33 precedent was based on) but loses to `G6.4a_v2c`'s FP16 FFN, which wasn't part of that comparison. FP16 has 2x TF32's FLOPS ceiling on this hardware — once FP16-FFN exists as the alternative, cuBLASLt-TF32 is no longer competitive at this shape. Step-33's precedent predates FP16-FFN existing at all, so it doesn't transfer. Reverted cleanly; code is back to always using `G6.4a_v2c` for causal's FFN. |
| `G6.4bc` (FP16 attention, causal) | SHIPPED — biggest win this pass | default 2.71x (+37%), tiny 7.66x/8.35x (+43%/+55%), long-seq 7.10x/6.40x (+78%/+107%), large-batch 2.66x/2.66x (+26%/+31%) | 0.00157 (default), 0.00147 (tiny), 0.00161 (long-seq), **0.00182 (large-batch, 91% of the 0.002 budget — tightest margin in this ledger)** | Cast Q/K/V/out_proj to FP16 around SDPA, cast back immediately (same pattern as non-causal `G6.4b`). Removed the explicit `sdpa_kernel([EFFICIENT_ATTENTION])` forcing used by `G0.1c` — FP16 unlocks automatic flash/efficient dispatch without it, same as non-causal. long-seq's outsized gain (+78-107%) is strong evidence a real O(S)-memory flash/efficient kernel is now engaged there, not just launch-overhead reduction. **large-batch's margin is now the binding constraint for `G4.6c`** — any further precision reduction there needs real care. |
| `G4.6c` (CUTLASS FP16-accum, causal-large-batch) | TRIED, REVERTED — real accuracy FAILURE | 2.84x/2.86x (would-be speedup, moot) | **0.00763 (3.8x over the 0.002 budget), 104k/671M elements failed** | Wired in exactly as scoped (same `CFG_QKV=6`/`CFG_OUT=18` configs already proven safe for non-causal, gated to fire only at the exact causal-large-batch token count, 40-seed full-sweep validation). All other 13 shapes stayed clean (routing correctly scoped, no leakage) — only causal-large-batch failed, and by a lot: margin jumped from 0.00182 (G6.4bc alone) to 0.00763, a >4x increase, not a marginal miss. The two independent FP16-precision sources (G6.4bc's attention + CUTLASS's FP16-accumulate GEMM) compound worse together at causal-large-batch than either alone would predict — CUTLASS's own non-causal validation (clean at the same budget) does NOT transfer once it's stacked on top of an already-FP16 causal attention path. Reverted via `git checkout -- benchmark.py` to the last validated `G6.4bc` commit; verified clean (validity gate passes, working tree matches the archived elite). **Closed — do not retry without a real, new idea** (e.g. splitting which of QKV vs out_proj gets CUTLASS, or reverting `G6.4bc`'s attention FP16 first) per this project's decisive-closure precedent; the plain combination doesn't work. |

## Current elite (per causal shape, `causal*/fp16` cells)

- default: **2.71x** (`G6.4bc`, `archive/causal__fp16.json`), max_abs 0.00157
- tiny: **7.66x** (`G6.4bc`, `archive/causal-tiny__fp16.json`), max_abs 0.00147
- long-seq: **7.10x** (`G6.4bc`, `archive/causal-long-seq__fp16.json`), max_abs 0.00161
- large-batch: **2.66x** (`G6.4bc`, `archive/causal-large-batch__fp16.json`), max_abs 0.00182 (tightest margin)

**Cross-check:** `SUBMISSION.md`'s Before/After table reports causal at
2.76x (1.5473ms→0.5612ms), not 2.71x — that's a separate, later, fresh
single-job full-sweep re-verification (`jobs/final_reverify.sbatch`,
preserved at `results/final_reverify_run118.log`), not a contradiction of
the 2.71x archived here. Both were measured against the same shipped
commit; the ~1.8% gap is ordinary run-to-run/thermal variance between two
differently-timed runs, not a code difference.

## Order of remaining work (see the approved plan for full detail)

1. ~~Archive the four causal-shape numbers above~~ — DONE.
2. ~~`G6.6c` — cuBLASLt for causal-tiny's FFN~~ — TRIED, REVERTED (see row
   above). Elite numbers unaffected.
3. ~~`G6.4bc` — FP16 attention for causal~~ — SHIPPED, biggest win this
   pass (+26% to +107% depending on shape, see row above).
4. ~~`G4.6c` — CUTLASS FP16-accumulate for causal-large-batch~~ — TRIED,
   REVERTED. Real accuracy failure (0.00763, 3.8x over budget, 104k/671M
   elements) — the compounding of two independent FP16-precision sources
   is worse than either alone. Closed; elite numbers unaffected (still
   `G6.4bc`'s, see table above). This ledger's Part 2 (per the plan) is
   now exhausted — the causal path stands at 2.71x/7.66x/7.10x/2.66x
   (default/tiny/long-seq/large-batch), and every further lever this
   session identified (cuBLASLt, CUTLASS, PTX, SASS) is closed. A future
   session picking this up should treat causal-large-batch's tight margin
   (0.00182/0.002) as the binding constraint on ANY further precision
   work there, and look for genuinely new ideas rather than retrying
   these combinations.
