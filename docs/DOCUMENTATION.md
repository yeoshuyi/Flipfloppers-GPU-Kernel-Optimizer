# DOCUMENTATION.md — Transformer Kernel Optimization: Full Technical Record

This is the complete technical reference for the `UserOptimizedTransformer`
project: every optimization implemented, shipped, reverted, or closed as a
dead end, with the real measured numbers behind each. It is distinct from
`SUBMISSION.md` (the competition narrative/pitch, cross-referenced but not
duplicated here) — this document exists so a reviewer can verify every claim
against the repository's own records.

Primary sources (all numbers below are cited to one of these; none are
invented):
- `docs/PROGRESS.md` — 40-step narrative log, non-causal-focused, steps 1–40
- `docs/CAUSAL_LEDGER.md` — causal-path-specific ledger, tags `*c`
- `archive/*.json` — MAP-Elites archive (regime × family grid), the
  authoritative measured-speedup numbers
- `benchmark.py` — current shipped code, read for structure and inline
  comments (not modified, not fully read)
- `torch_transformer_benchmark.py` — GENERATED (`tools/sync_entrypoint.py`):
  judges' 2026-08-27 canonical harness + our `UserOptimizedTransformer`. A
  standalone drop-in; never hand-edited.
- `tools/verify_baseline.py` — asserts `benchmark.py`'s frozen half is
  byte-equivalent (AST) to `~/torch_transformer_benchmark.py`
- `docs/ACCURACY_BUDGET.md` — spend/benefit rule for optimisations near the
  0.002 ceiling; §8 ledger of shipped lossy steps
- `SUBMISSION.md` — competition narrative, cross-referenced only

**A note on currency.** `docs/PROGRESS.md` now runs through **step 42**
(steps 41 G4.3 non-causal ship, 42 G4.7 causal ship — both narrated in full).
An earlier gap remains: the FP16-FFN-in revival (`G6.4a_v2`) and the causal
rewrite (`G0.1c`…`G6.4bc`) were done without a "Phase 2.5" narrative section
in PROGRESS — they are verifiable in `archive/*.json`, `git log`, inline
`benchmark.py` comments, and §3.3 here. Steps 41–42 (§3.4) do have full
PROGRESS entries. `docs/ACCURACY_BUDGET.md` (§8 ledger) is the current
single source for per-optimisation precision cost.

---

## 1. Overview

`UserOptimizedTransformer` is a from-scratch-optimized re-implementation of a
frozen `BaselineTransformer` reference (18.9M params, 6 layers, `d_model=512`,
8 heads, FFN dim 2048), targeting one NVIDIA RTX 4090 (Ada Lovelace, sm_89,
CUDA 13.1, PyTorch 2.13.0+cu130). The benchmark harness enforces bit-for-bit
API compatibility (`load_state_dict(strict=True)`, same forward signature)
and scores speed only if accuracy passes on every tested shape — a failing
shape zeroes the whole benchmark, not just that shape.

The optimized model has **two entirely independent live forward paths**,
selected once per call on `self.config.causal`:
- **Non-causal**: `UserOptimizedTransformer.forward()` → lazily-compiled
  `_optimized_forward` (benchmark.py, `torch.compile(mode="reduce-overhead")`),
  dispatching internally by `B·S` (tiny / default / large-batch) and by
  padding state, per `CLAUDE.md`'s regime table.
- **Causal**: `forward()` → lazily-compiled `_optimized_forward_causal`, a
  fully separate implementation (own weight-fold builders, own SDPA backend
  selection, own FP16 gating) — see §5 for why it isn't shared.

**Current headline results** (regime, non-causal `trackA`→`fp16`-family
progression, from `archive/*.json`; causal from `archive/causal*.json` and
`docs/CAUSAL_LEDGER.md`): every one of the 6 CLAUDE.md regimes (tiny,
default, long-seq, large-batch, padded, causal) improved from its TF32-only
baseline speedup into the 2.2×–7.6× range once FP16 attention (`G6.4b`/
`G6.4bc`), the cuBLASLt tiny-FFN algorithm search (`G6.6`), and the
budget-relaxation-enabled FP16 FFN-in (`G6.4a_v2`/`G6.4a_v2c`) shipped. See
§3 for the full per-cell numbers.

---

## 2. Accuracy policy

The benchmark's correctness rule is disjunctive per output element:
`abs(user − ref) ≤ atol` **OR** `abs(user − ref) ≤ rtol · abs(ref)`. Failing
this on any tested shape scores the whole benchmark zero (`CLAUDE.md`
invariant 2) — accuracy is a hard gate, never a footnote.

**Two budgets exist in this project's history, and it matters which one a
given historical number was measured against:**

- **`atol=0.001, rtol=0.01`** (1%) — the *original* bound, and the one
  `/home/techjam2/CLAUDE.md`'s own "ACCURACY" section states. Essentially
  every step in `docs/PROGRESS.md` (steps 1–40) and the early rows of
  `docs/CAUSAL_LEDGER.md` (`G0.1c` through the causal shape-sweep row) were
  measured, gated, and reported against this tighter bound. Several closures
  in this record are budget-relative facts under *this* bound specifically —
  e.g. G6.4a's FFN-FP16 near-miss (step 27) failed here.
- **`atol=0.002, rtol=0.02`** (2%) — the enforced default, now **confirmed in
  writing**: the judges published `~/torch_transformer_benchmark.py` on
  2026-08-27 with exactly these as the `parse_args` defaults, superseding the
  earlier "relayed verbally, not independently verified" status. Our
  `benchmark.py` already matched it; `tools/verify_baseline.py` now asserts the
  frozen harness (BaselineTransformer + `compare_outputs` + timing loop +
  `parse_args`) stays byte-equivalent to that file, and repo
  `torch_transformer_benchmark.py` is generated from it by
  `tools/sync_entrypoint.py`. This is the bound that governs scoring today.
  The older, tighter 0.001/0.01 bound remains reachable via `--atol`/`--rtol`
  and is still used as the internal engineering target wherever practical
  (anything passing it passes 0.002/0.02 automatically).

**Practical effect:** the budget change is what made `G6.4a_v2` (FFN-in
FP16) and the entire `*c`-suffixed causal-path FP16 work shippable — both
were failed or borderline closures under 0.001/0.01 and clean passes once
re-verified at 0.002/0.02 (see §4, "FP16 FFN, both GEMMs" and the causal
`G6.4bc` row). Numbers in §3/§4 below are tagged with whichever budget they
were actually measured against.

---

## 3. Shipped optimizations

### 3.1 Non-causal — Track A (TF32/FP32-only precision, measured under the 0.001/0.01 budget)

Chronological, each row gated on `tools/check_validity.py` → full 8-shape
accuracy sweep → `tools/archive.py commit`. Source: `docs/PROGRESS.md` steps
5–13, `archive/*__trackA.json`.

| id | description | tiny | default | long-seq | large-batch | padded | causal |
|---|---|---|---|---|---|---|---|
| G0.1 | `F.scaled_dot_product_attention` replaces manual matmul→mask→softmax→matmul; causal gated to exact baseline fallback (accuracy) | 1.307x | 1.137x | 2.147x | 1.253x | 1.134x | 1.00x (fallback) |
| G0.2 | fused QKV: one `[d,3d]` GEMM replaces 3× `[d,d]` | 1.378x | 1.150x | 2.095x | 1.255x | 1.161x | 1.00x |
| G0.3 | drop `.contiguous()` before SDPA (strided view accepted directly) | 1.496x | 1.205x | 2.193x | 1.362x | 1.208x | 1.00x |
| G0.5 | all-ones-mask fast path (`_mask_is_all_ones`, `attn_mask=None`); found `generate_random_case()` never passes literal `None`, so this was previously dead code | **2.640x** | 1.397x | 2.300x | 1.547x | 1.211x | 1.00x |
| G1.1 | fold LayerNorm affine into consumer linear | 2.692x | 1.355x | 2.302x | 1.545x | 1.204x | 1.00x |
| G1.2 | fold attention scale into `W_Q` (`scale=head_dim**-0.5=2^-3` exact power-of-2, bit-identical) | 2.585x | 1.392x | 2.301x | 1.547x | 1.216x | 1.00x |
| G2.4 | lazy `torch.compile(mode="reduce-overhead")` — CUDA graphs | **4.649x** | 1.601x | **2.355x** (elite) | 1.609x | **1.604x** (elite) | 1.00x |
| G2.4b | `torch.compile` around baseline's *exact* causal math (causal only) | 4.595x | **1.608x** (elite) | 2.353x | **1.610x** (elite) | 1.588x | **1.747x** |

**Track A elite per cell** (`archive/*__trackA.json`): tiny 4.649x (G2.4),
default 1.608x (G2.4b), long-seq 2.355x (G2.4), large-batch 1.610x (G2.4b),
padded 1.604x (G2.4), causal 1.747x (G2.4b).

Notable within Track A: G1.2's scale-fold is proven bit-identical
(`docs/PROGRESS.md` step 11 — verified by diffing every per-trial
`max_abs`/`max_rel` line against the prior run and finding zero difference)
but delivers ~0 net speedup, because SDPA already performs the scale multiply
internally regardless of the `scale=` argument — `docs/CATALOGUE.md`'s
general estimate assumed a manual matmul implementation, which no longer
applies once SDPA owns that step. Kept anyway as a documented G4 prerequisite.

### 3.2 Non-causal — FP16 family (measured initially under 0.001/0.01; re-verified/extended under 0.002/0.02)

Source: `docs/PROGRESS.md` steps 28, 33; `archive/*__fp16.json`; and the
undocumented-in-PROGRESS.md but archive/git-verified `G6.4a_v2` round (see
overview note above).

| id | description | budget measured | tiny | default | long-seq | large-batch | padded |
|---|---|---|---|---|---|---|---|
| G6.4b | FP16 for QKV projection + SDPA + `out_proj` (FFN stays exact TF32); unlocks flash/memory-efficient SDPA backends unavailable to FP32 | 0.001/0.01 | 6.140x | 2.238x | 4.560x | 1.976x | 2.136x |
| G6.6 | cuBLASLt explicit algorithm search + bias epilogue for the FFN GEMMs, gated to TINY only (`tok ≤ 127`), correct fallback to `F.linear` | 0.001/0.01 | **7.24x** (elite) | — (no-op outside tiny) | — | — | — |
| G6.4a_v2 | FFN-in cast to FP16, GELU/ffn_out stay exact FP32/TF32 (revives step-27's near-miss; see §4) | 0.002/0.02 | 7.19x* | **2.49x** (elite) | **5.37x** (elite) | **2.45x** (elite) | **2.51x** (elite) |

\* tiny's combined-elite number (7.194x, `archive/tiny.json`) is marginally
*below* G6.6's own 7.24x elite recorded in `archive/tiny__fp16.json` — a
~0.6% difference attributed in `docs/PROGRESS.md` step 33 to session-level
clock/thermal drift (~5–8% swings measured and cross-validated there), not a
regression from G6.4a_v2 itself.

`max_abs` for G6.4b (`docs/PROGRESS.md` step 28, 40-seed/6-shape rigor,
`results/g6_4b_fp16_attn_rigor_run62.log`): 0 failures on every shape, peak
0.000906 (large_batch) — comfortably under the 0.001 atol outright, not
saved by the disjunctive `rel` clause.

The current combined non-causal elite per regime (`archive/{tiny,default,
long-seq,large-batch,padded}.json`, all `id: g6_4a_v2`, applied stack
`G0.1,G0.2,G0.3,G0.5,G1.1,G1.2,G2.4,G2.4b,G6.4b,G6.6,G6.4a_v2`, timestamped
2026-08-27T10:22:05): **tiny 7.194x, default 2.494x, long-seq 5.367x,
large-batch 2.451x, padded 2.513x.**

`SUBMISSION.md`'s Before/After table reports these same regimes very
slightly higher (tiny 7.24x/7.236x, default 2.52x, long-seq 5.37x,
large-batch 2.45x, padded 2.50x) plus causal at 2.76x vs 2.71x above —
that table is one fresh, single-job, all-shapes-together re-verification
(`jobs/final_reverify.sbatch` → `results/final_reverify_run118.log`), run
later than and separately from these incrementally-archived per-step
numbers. Both measure the same shipped commit; treat the deltas as normal
run-to-run/thermal variance between two differently-timed jobs, not as a
disagreement about what's shipped.

### 3.3 Causal path — shipped stack

Source: `docs/CAUSAL_LEDGER.md`; `archive/causal*.json`. Budget: 0.001/0.01
for the first three rows (`G0.1c`, `G1.1c`, `G6.4a_v2c`, `G0.2c`), 0.002/0.02
confirmed for the causal shape-sweep validation and everything from
`G6.4bc` onward (`docs/CAUSAL_LEDGER.md` header states 0.002/0.02 as "the
real enforced default" for the whole ledger).

| id | description | max_abs | default | tiny | long-seq | large-batch |
|---|---|---|---|---|---|---|
| G0.1c | SDPA (`EFFICIENT_ATTENTION` backend, forced via `sdpa_kernel`) replaces baseline's manual attention loop for causal | 0.00134 | 1.80x | — | — | — |
| G1.1c | norm2-affine fold into `_ffn_in_weight`/bias; exact, bit-identical, flat speed (launch overhead already absorbed by CUDA graphs) | 0.00134 (unchanged) | 1.80x | — | — | — |
| G6.4a_v2c | FFN-in FP16, cast back to FP32 immediately (ffn_out/GELU exact) — the real win of this pass | 0.00141 | 2.01x / 1.96x (padded) | — | — | — |
| G0.2c | QKV fused GEMM + scale-fold + norm1-affine-fold, causal-independently-verified | 0.00141 (unchanged) | 1.99x (near-miss, logged not elite) | — | — | — |
| shape sweep | first causal validation beyond B8_S128, confirms the stack generalizes | 0.00128–0.00163 | — | 5.84x/5.71x (padded) | 3.99x/3.08x (padded) | 2.11x/2.03x (padded) |
| **G6.4bc** | FP16 for Q/K/V/out_proj around SDPA, causal (same pattern as `G6.4b`); removes the explicit `EFFICIENT_ATTENTION` forcing — FP16 unlocks automatic flash/efficient dispatch | 0.00157 (default) / 0.00147 (tiny) / 0.00161 (long-seq) / **0.00182 (large-batch, 91% of the 0.002 budget)** | **2.71x** (+37%) | **7.66x**/8.35x padded (+43%/+55%) | **7.10x**/6.40x padded (+78%/+107%) | **2.66x**/2.66x padded (+26%/+31%) |
| **G4.7c** | fused `ffn_in` GEMM + exact-erf GELU epilogue on the G4.3 warp-spec kernel, FP32-accumulate — precision-**neutral** (`max_abs` bit-identical to G6.4bc), gated to `d_model≥512 ∧ ffn_dim≥2048 ∧ tok≥8192` so it engages only on the project's internal d512/ffn2048 causal sweep, **not** the official 14-row matrix (`ffn_dim ∈ {32,128,1024}`) | 0.00161 (long-seq) / 0.00182 (large-batch) — **unchanged from G6.4bc, exact ledger match** | — | **7.78x** (+9.5% over G6.4bc) | **2.98x** (+12.1%) | step 42; `results/g4_7_ship_verify_v2_run142.log` |

**Current causal elite** (`archive/causal*__fp16.json`): default **2.71x**,
tiny **7.66x**, long-seq **7.78x** (`causal-long-seq__fp16` `g4_7c`; was
7.10x at `g6_4bc`), large-batch **2.98x** (`causal-large-batch__fp16` `g4_7c`;
was 2.66x). The long-seq / large-batch bumps are G4.7c, and are **free of
accuracy cost** — `max_abs` is bit-identical to the pre-G4.7 40-trial record
because the fused epilogue computes the erf GELU exactly and accumulates in
FP32 (§3.4). large-batch's `max_abs` of 0.00182 is still "the tightest margin
in this ledger" — the binding constraint on any further causal-path
*precision*-spending work (see §4, `G4.6c`, and `docs/ACCURACY_BUDGET.md` §1).

long-seq's outsized +78–107% gain from `G6.4bc` is attributed to a real
mechanism, not overhead reduction: `CLAUDE.md`'s own regime table notes
attention is ~48% of the forward at `S≥1024`, and flash/memory-efficient
attention's O(S) memory scaling engages specifically at long sequences.
Causal `max_abs` values are noticeably closer to budget than non-causal's
(e.g. 0.00182 vs 0.02 headroom at large-batch) because causal masking
reduces the number of valid keys in early softmax rows, which reduces the
averaging-down of quantization error across the reduction — the same
mechanism cited throughout for why causal is consistently the tightest-margin
regime.

---

### 3.4 Warp-specialised `mma.sync` GEMM — G4.3 (non-causal) and G4.7 (causal)

The "Hand-written `mma.sync` PTX" and "CUTLASS" rows in §4 closed the
**FP16-accumulate** tier: real mechanism, but the accumulation precision is
arithmetically unaffordable at this model's budget on every shape that would
benefit. What later shipped is the **FP32-accumulate** version of the same
warp-specialised kernel — same `mma.sync.m16n8k16` datapath and
producer/consumer named-barrier pipeline, but accumulating in FP32 (identical
precision to cuBLAS on an FP16-storage GEMM), so it spends **zero** accuracy
budget.

- **G4.3 (non-causal), step 41.** Warp-specialised GEMM + a CUTLASS-grade
  128-bit shared-memory-staged epilogue, replacing cuBLASLt for the two
  attention projections (QKV, out_proj) at `tok ≥ 8192`. Ships **non-causal
  only**: `+4.75% long_seq`, `+5.38% large_batch` (`archive/*__fp16.json`,
  `results/g4_3_ship_verify_final_run130.log`). Carries a SPLIT-64 FP32 carry
  (`cfg 48`) so non-causal large_batch's `max_abs` lands at 0.00158 (79% of
  budget) rather than 0.00189 (99%). **Closed for causal** — the causal
  attention path is already at 91–95% of budget on FP16 *storage*, and the
  FP16-*accumulate* arm of this kernel adds ~5× the remaining headroom (every
  rescue on the ladder measured; `results/g4_3_numerical_rescue_run129.log`).
- **G4.7 (causal), step 42.** Extends the same kernel with an FP32-accumulate
  arm and a fused epilogue that computes `F.gelu(approximate="none")`
  **bit-identically** (verified 39/39 (cfg, shape) points,
  `results/g4_7_ffn_correct_run131.log`) — collapsing the `ffn_in` GEMM *and*
  a full elementwise cast+GELU pass into one kernel. Precision-neutral, so the
  causal budget that blocked G4.3 does not apply. Ships for causal at
  `d_model≥512 ∧ ffn_dim≥2048 ∧ tok≥8192`: **+9.5% long_seq_causal, +12.1%
  large_batch_causal**, `max_abs` bit-identical to the pre-G4.7 record. The
  microbench (`results/g4_7_ffn_sweep_run132.log`) shows the FP32-accumulate
  arm only nets a gain once `ffn_dim` is large (x0.93–x0.99 at
  `ffn_dim ∈ {128, 1024}` — half-rate FP32 `mma` on Ada is not repaid by the
  saved elementwise pass until `ffn_dim ≥ 2048`), which is why the gate
  excludes the entire official 14-row matrix. First causal-path GEMM win in
  the project. Full spend/benefit reasoning: `docs/ACCURACY_BUDGET.md` §8.

---

## 4. Attempted and reverted / closed

Every dead end on record, with the real reason it closed. None of these are
softened — a genuine negative result is reported as such.

### Precision reduction (FFN / attention / whole-model)

| Attempt | Result | Real reason | Source |
|---|---|---|---|
| **G2.1 — BF16, whole model** (QKV, SDPA, out_proj, FFN, all in BF16, residual kept FP32) | Decisive failure, reverted immediately | `max_abs` up to **0.0110 — 11x the 0.001 budget**, 13.6% of elements (22344/163840) genuinely failing the disjunctive criterion on the very first (tiny) shape; failures span O(1)-magnitude activations, not near-zero edge cases | step 14, `results/g2_1_smoke_FAILED_run28.log` |
| **G2.1b — BF16, FFN only** (attention left exact) | Decisive failure at nearly identical magnitude | `max_abs` up to 0.0112, 12.7% of elements failing (20880/163840) — same order as full-model BF16, ruling out "attention softmax was the risk" and pointing instead at the FFN's large weight reductions as the fundamental blocker | step 17, `results/g2_1b_smoke_FAILED_run37.log` |
| **G4 FP8 precheck** (per-channel weight scales, per-tensor activation scale, naive single-term FP8 on the FFN) | Decisive failure, gated the whole G4/megakernel FP8 assumption before any kernel was written | 20/20 seeds fail; mean `max_abs` **65x the 0.001 budget** (0.065174), worst case 78x — even with correct per-channel scaling | step 18, `results/g4_fp8_precheck_run38.log` |
| **G6.5 — INT8 FFN** (symmetric `[-127,127]`, per-channel weight scale, per-tensor dynamic activation scale) | Decisive failure, worse than FP8 or FP16 | 20/20 seeds fail, `max_abs` 0.0269–0.0310 — **27-31x over the 0.001 budget**; INT8's fixed linear step size has no floating exponent to auto-range O(1)-magnitude Gaussian activations | step 29, `results/g6_5_int8_precheck_FAILED_run64.log` |
| **G5.3 split-precision FP8 (G2.8)** — greedy residual FP8 split, `k` terms/operand, per-128-tile dynamic scales | **Genuinely passes accuracy at k=4** (0.00063-0.00069, 0/60 true failures) — the one precision-reduction attempt that actually worked numerically — but **closed on arithmetic, not accuracy**: `330.3/4 = 82.6` TFLOPS ideal for a 4-GEMM kernel exactly equals TF32's own peak, and torch's TF32 FFN already runs at 69% of that peak, so a hand-written kernel would need ~69% efficiency of its own tier just to break even; the one hand-written kernel this project built (G3.1) landed at 13-87% depending on shape | steps 21-22, `results/g5_3_fp8_split_run51.log` |
| **G5.5 — real-hardware re-test of k=3** (re-opened by request; checks whether the plan's "3 terms → 3 GEMMs" reading was even right) | Neither the 3-GEMM asymmetric design (either operand at 1-term) nor the 6-GEMM triangular generalization of `CLAUDE.md`'s own G2.8 example passes: asymmetric fails at ~0.09-0.10 max_abs (90-100x budget), triangular fails at 0.0072 (7x budget) | bf16-forced `_scaled_mm` row-wise output rounding compounds across 6 real GEMMs, and 6 GEMMs already gives only 55 TFLOPS ideal — *below* torch's own 57.3 TFLOPS measured TF32 baseline, so it loses arithmetically even at 100% kernel efficiency | step 24, `results/g5_5_real_gemm_result_run55.log` |
| **G6.4a v1 — FP16, both FFN GEMMs** | Smoke test (5 trials) passed, but 40-seed/6-shape rigor probe failed on **every shape** — tiny 12/40 trials (30%), default/long_seq/large_batch/padded all fail too, rare (single-to-low-double-digit) failing elements against 1.3M-671M-element tensors | closed under the 0.001/0.01 budget then in force; real speedups measured alongside the failure (default 1.894x, large_batch 2.241x, long_seq 2.784x) — a large win that had to be given up | step 27, `results/g6_4a_both_gemms_FAILED_run60.log` |
| **G6.4a v2 — FP16, FFN-in only** (narrower slice of v1) | Failure counts dropped ~10x, tiny and default_padded passed cleanly, but default/long_seq/large_batch/long_seq_padded still failed under 0.001/0.01 | closed at the time per `CLAUDE.md` invariant 6 (full sweep, not one shape); **later revived and shipped as `G6.4a_v2` once the accuracy budget loosened to 0.002/0.02** — same code, different verdict under the new budget (§3.2) | step 27, `results/g6_4a_v2_ffnin_only_FAILED_run61.log`; revival evidenced in `benchmark.py`'s inline `_build_ffn_in_fold` comment and `archive/*.json` `g6_4a_v2` entries |

### Compiler / toolchain-level

| Attempt | Result | Real reason | Source |
|---|---|---|---|
| **G6.1 — `torch.compile(mode="max-autotune")`** | Decisive accuracy failure on the first shape tested | All 5 trials fail, `max_abs` 0.00220–0.00242 (2.2-2.4x the 0.001 budget); root cause read from the autotuner's own log — it selects Triton kernels (`ALLOW_TF32=True`, a *software* 3-pass FP32 decomposition) over cuBLAS's native TF32 tensor-core path for some GEMM shapes, a different rounding pattern than the shipped `reduce-overhead` path uses uniformly | step 25, `results/g6_1_max_autotune_smoke_FAILED_run56.log` |
| **G6.2 — `cudnn.benchmark=True`** | Clean null result, not a failure | Zero accuracy risk (never affects a computed value) and confirmed empirically: every shape's speedup within ±1-2% of baseline — model has zero convolutions, nothing for cudnn's algorithm cache to act on | step 26, `results/g6_2_cudnn_benchmark_null_run57.log` |
| **G2.3 — L2 persistence via raw `ctypes`** | Not pursued | `torch.cuda.cudart()` doesn't expose `cudaDeviceSetLimit`/`cudaStreamSetAttribute`; reaching them needs `ctypes.CDLL`-ing `libcudart.so` directly with no compiler safety net for `cudaAccessPolicyWindow`'s struct layout — real crash/context-mismatch risk judged not worth the catalogued 1.1-1.5x gain | step 16 |
| **G1.6 + G2.3 — built for real (C++/pybind11 extension)** | G2.3 measured a **clean 4-6% regression**; G1.6 alone neutral (within noise) | Real hot working set is ~63MB (not 75.66MB — that double-counts dead original `nn.Parameter`s kept for `strict=True`), and the 63MB set already fits under normal LRU in 72MB L2; the persistence window can only reach `hitRatio=0.825` (device caps persisting L2 at 49.5 MiB) and just carves that away from activation traffic instead — regression, not a win | step 32, `results/g1_6_g2_3_sweep_run69.log` |
| **G6.7 — cuBLASLt algorithm search, FP16 attention GEMMs** | Clean negative | Initial "win" (1.26x/1.96x) was a measurement artifact: PyTorch's FP16 reference at M=64 is launch-bound (below both harnesses' dispatch floors), and `torch.profiler` showed both sides dispatching the *identical* kernel with `maxdiff=0.0` — one kernel cannot be faster than itself. Fair re-measurement (CUDA-graph replay + profiler kernel time) agreed to 0.4%: cuBLASLt's default heuristic is already optimal for FP16 at this shape | step 34 |
| **G6.8 — extend G6.6's cuBLASLt algorithm to `ffn_in` at LONG-SEQ (M=8192)** | Clean negative | Same measurement-artifact class as G6.7: the apparent 1.12x win lived entirely in the eager-`F.linear` reference's bias-add path, which `torch.compile`'s actual lowering already avoids by fusing the bias into `triton_poi_fused_addmm_gelu_view_2`; under graph capture, eager and the "winning" algorithm dispatch the *same kernel* (244.49 vs 244.46 µs, `maxdiff=0.0`) | step 36 |

### Kernel-fusion / megakernel track

| Attempt | Result | Real reason | Source |
|---|---|---|---|
| **G3.1 — fused FFN tile (Triton)**, `LN→ffn_in→GELU→ffn_out` fused, keeping the intermediate in registers | Built, verified numerically correct against fp64 ground truth (`tf32x3`/`ieee` modes land within ~2e-6), but **0.18x (default) to 0.869x (large_batch)** of the 3 torch kernels it replaces — a regression at every shape | torch's FFN already runs at 60-69% of the 82.6 TFLOPS TF32 roofline, so the ~48MB/forward of intermediate traffic G3.1 removes is already overlapped behind compute; fusion gives up cuBLAS's tiling freedom for no bandwidth win. Plain `tf32` mode is also 3.7x less accurate than cuBLAS's own TF32 (0.6% of elements failing in one isolated FFN) | step 19 |
| **G4.0 — two-kernel form** (checked twice: feasibility-only in step 20, built-and-measured in step 35) | Not built / gate not met either time | Step 20: summed GPU kernel duration equals wall-clock forward time at every shape (gap −0.7% to +0.0%) — no inter-kernel gap for fusion to recover. Step 35 (post-FP16-attention re-measurement, direct experiment): removing a real kernel boundary at TINY via cuBLASLt in-place split-K reduction is a **1.4-2.1x regression**, not a win — the boundary was never costing anything. `docs/MEGAKERNEL.md`'s own gate ("`>15%` launch overhead or GPU idle after CUDA Graphs") reads −0.55% to 8.49% across all four ways of asking the question, never clearing 15% except by pricing LayerNorms as "free," which they are not | steps 20, 35 |
| **G3.6 — minimax degree-7 GELU polynomial** | Closed without a GPU job, pure math | `docs/CATALOGUE.md`'s own accuracy claim (~1e-6 at degree 7) is **verifiably wrong**: a real Chebyshev-fit degree-7 polynomial over `[-5,5]` achieves max abs error **0.084 — 84x over the entire budget**; reaching the claimed 1e-6 needs a polynomial in the high 20s in degree, likely costing more arithmetic than `F.gelu`'s hardware `erf` intrinsic | step 30 |
| **G4.4 — hand-written `mma.sync.m16n8k16` PTX, FP16-accumulate GEMM** | Built, verified, Stage-0 speed gate NOT met | FP16-accumulate is architecturally the one tier cuBLAS can never reach on Ada (its FP16 path mandates `CUBLAS_COMPUTE_32F`). Kernel reached 55.1% of its own 330.3 TF tier vs cuBLASLt's 91.2% of its (2x lower) 165.2 TF tier — net **1.207x at large_batch (gate was 1.3x), 1.000x at default**. Whole-model dilution closes it even if tuned further: best case ~2.05% at large_batch, the only shape with any win at all, right at CLAUDE.md's own "<2% ⇒ stop" line | step 37 |
| **G4.5 — CuAssembler/SASS-level instruction reordering on G4.4's cfg[11]** | Clean negative at Phase 0's toolchain gate — hard blocker, never reached measurement | CuAssembler correctly encodes Ada SASS (verified: `repos.verify()` returns True, 0 error records, on the kernel's real HMMA/LDSM/LDGSTS/STG/LDG/LDGDEPBAR/DEPBAR instructions) — **not** an Ada-support problem. The wall is CUDA 13.1's `nvdisasm` rendering the `.note.nv.tkinfo` ELF section as unparseable text (`.tkinfo`/`.string` directives CuAsmParser has no model for), a **lossy** rendering that cannot be reconstructed from the `.cuasm` intermediate. Three workarounds tried (section retyping, `objcopy --remove-section`, suppressing the note) all failed. Never measured on GPU — no speedup claimed | step 38 |
| **G4.6 — CUTLASS FP16-accumulate GEMM, Phase 1 (speed)** | Phase 0 clean (no toolchain wall — header-only, compiled fresh, sidesteps G4.5's binary-parsing risk entirely); Phase 1 speed gate **narrowly missed**: best config reached 71.4-71.7% of the FP16-accumulate tier against an 80% (264.2 TF) kill gate, despite beating cuBLASLt 1.57-1.60x and the hand-written G4.4 kernel 1.30x | Diagnosed, not just measured: at 218µs this kernel already runs ~622 GB/s of compulsory traffic against a 921 GB/s ceiling; reaching 80% needs near-peak bandwidth *and* mma issue efficiency simultaneously, which CUTLASS 2.x's `device::Gemm` template (no warp-specialized pipelining — that's CUTLASS 3.x/Hopper-only) cannot deliver on Ada | step 39 |
| **G4.6 Phase 2 — CUTLASS FP16-accum, bar explicitly lowered by user instruction to accept the ~72%-of-tier, ~4.7% whole-model prize anyway** | **Closed permanently on ACCURACY, not speed.** Kernel itself verified exactly correct against fp64 ground truth (0/48 mismatches on a non-tile-aligned one-hot sweep; FP32-accumulate variant reproduces cuBLAS bit-for-bit to 7 digits) — but FP16 accumulation at K=512 costs **x7.2-x7.8** on the GEMM's own error, and FP16 *storage* alone already spends 90.6% of the 1e-3-equivalent atol budget at large_batch. **6 of 8 shapes fail** the full sweep; splitting `qkv`-only vs `out_proj`-only routing doesn't rescue it either (`qkv`-only passes 4/6 non-causal shapes but fails exactly large_batch — the only shape with any speed win) | step 40, `results/g4_6_phase2b_accuracy_run104.log` |
| **G6.6c — cuBLASLt algorithm search, causal-tiny FFN** | Tried, reverted — real regression | -8.6%/-5.8% at causal-tiny, confirmed over a 40-seed sweep (not noise). `_build_lt_plan`'s own gate only compares cuBLASLt-TF32 against plain `F.linear`-TF32 — the comparison the non-causal G6.6 precedent (step 33) was based on, before `G6.4a_v2c` (FP16 FFN) existed as a causal alternative. FP16 has 2x TF32's FLOPS ceiling on this hardware, so once FP16-FFN exists, cuBLASLt-TF32 is no longer competitive even though it still beats plain TF32 `F.linear` | `docs/CAUSAL_LEDGER.md` row 25 |
| **G4.6c — CUTLASS FP16-accumulate GEMM, causal-large-batch** | Tried, reverted — real accuracy failure | `max_abs` **0.00763 — 3.8x over the 0.002 budget**, 104k/671M elements failed. Wired in exactly as scoped (proven-safe non-causal configs, gated to fire only at causal-large-batch's token count); all 13 other shapes stayed clean, confirming correct routing. The two independent FP16-precision sources (G6.4bc's FP16 attention + CUTLASS's FP16-accumulate GEMM) compound worse together than either alone predicts — margin jumped from 0.00182 (G6.4bc alone, already the tightest in the ledger) to 0.00763, a >4x jump, not a marginal miss | `docs/CAUSAL_LEDGER.md` row 27 |

### Megakernel investigation (docs/MEGAKERNEL.md)

The megakernel design's own core precision assumption — that FP8 weights are
*required* (not merely preferred) for shared-memory pipeline depth (BF16
tiling caps at 3 `cp.async` stages, overflowing the 99KB/SM budget by 3KB;
FP8 reaches 4-5 stages) — was checked *before* any megakernel code was
written (step 18's FP8 precheck, above) and failed at 65x over budget. Since
BF16 had already failed decisively twice (steps 14, 17) and the design has
no BF16 fallback path, this closed the persistent-cooperative-kernel /
warp-specialized (G4.1+) direction outright without a build attempt. G4.0
(the two-kernel form, not requiring FP8's pipeline depth) remained the
realistic ceiling and was investigated on its own terms (see G4.0 row
above) — `docs/MEGAKERNEL.md`'s own text sanctions this outcome explicitly:
"G4.0 winning is a result, not a failure."

---

## 5. Architecture notes

**Why regimes are a compile-time dispatch, not runtime-autotuned.** `CLAUDE.md`
specifies TINY/DEFAULT/LONG-SEQ/LARGE-BATCH/PADDED/CAUSAL as distinct,
named regimes with a fixed threshold table (`B·S<128`, `S≥1024`, `B·S>16384`,
mask-not-all-ones, `config.causal`), and the shipped `forward()` follows
that structure directly rather than searching for the best kernel at
runtime. Two concrete facts in this project's own record justify that
choice: (1) `torch.compile(mode="max-autotune")` — the one runtime-search
mechanism tried — failed on accuracy specifically *because* it searches
heterogeneous kernel families (Triton's software-emulated TF32 vs cuBLAS's
native tensor-core TF32) with different rounding characteristics per shape
(§4, G6.1); a fixed, audited kernel choice per regime avoids that risk by
construction. (2) TINY is launch/dispatch-bound in a way DEFAULT/LARGE-BATCH
are not (step 15's `ncu` pass: 3.79% of TF32 peak, 11.48% occupancy at
tiny vs 23.64%/25.73% at default) — a runtime branch inside the hot path
would cost real instruction-issue slots exactly in the regime that can
least afford them, which is why `CLAUDE.md`'s solidification rule strips
`triton.autotune` and freezes winning constants per regime rather than
leaving the choice dynamic.

**Why causal has its own fully independent optimized path.** The two paths
diverge at the very first optimization tried (G0.1, step 5): SDPA replacing
baseline's manual attention. For non-causal shapes this passed accuracy
cleanly. For causal it did not — `default_causal` failed 2/5 trials right
at the 1e-3 boundary, and the root cause (isolated with a 4-backend, 20-seed
probe) is structural, not a bug: causal masking means early softmax rows sum
over very few valid keys, so there are fewer terms to average independent
kernel-rounding error down over — the *inverse* of the mechanism that lets
FP8 survive on the FFN's K=2048 reduction. Baseline's own TF32-rounded
output is the accuracy reference (not an exact answer), and *any*
independently-kernelled computation of the same causal math — SDPA
`MATH` backend included, which is algorithmically identical to baseline's
own matmul→mask→softmax→matmul — diverges from that specific rounding by an
amount already close to budget before causal masking even enters. This was
re-confirmed, not merely inherited, at every later step: `G1.1c`/`G1.2`'s
folds, `G0.2c`'s QKV fusion, and `G6.4bc`'s FP16 attention were each
independently re-verified against causal shapes rather than assumed to carry
over from the non-causal result, and causal's `max_abs` margins are
consistently the tightest in the project (large-batch causal at 0.00182 of
the 0.002 budget — 91% consumed — vs non-causal's comfortably-under-80%
margins elsewhere). Given that dynamic, sharing one code path between causal
and non-causal would mean every non-causal-motivated change (e.g. G6.6's
cuBLASLt tiny-FFN algorithm, which regressed causal-tiny by 5.8-8.6% once
`G6.4a_v2c`'s FP16 FFN existed as the real causal alternative — §4, `G6.6c`)
would need independent causal re-validation anyway; building the causal
path as its own tree (own weight-fold builders `_build_qkv_fold`/
`_build_ffn_in_fold` called explicitly in the causal branch of `forward()`,
own explicit `sdpa_kernel([EFFICIENT_ATTENTION])` backend forcing, own
`torch.compile`d `_optimized_forward_causal`) makes that required
independence explicit in the code rather than an implicit invariant someone
could accidentally break by editing the shared non-causal path.
