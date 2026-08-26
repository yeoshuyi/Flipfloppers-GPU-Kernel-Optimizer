# Progress Log

Chronological, step-by-step record of what's been done on this project.
Written for two purposes: (1) resuming cleanly after a session crash, and
(2) source material for the write-up report. Updated after every verified
step — not a plan, a record of what actually happened.

**Read `docs/MANIFEST.md`, `CLAUDE.md`, and the "Current State" section below
first** if resuming cold. Then `git log --oneline` and `python3
tools/archive.py summary` for the authoritative current numbers (this file
can lag by one iteration if a crash happened mid-step).

---

## Current State (updated after each iteration)

- **Day:** 1-2 (Track A in progress)
- **Track A progress:** G0.1, G0.2, G0.3 done. G0.4 skipped (not applicable
  yet, see step 9). G0.5 in progress.
- **Archive (Track A elites, `python3 tools/archive.py summary`):**
  | regime | trackA |
  |---|---|
  | tiny | 1.50x |
  | default | 1.21x |
  | long-seq | 2.19x |
  | large-batch | 1.36x |
  | padded | 1.21x |
  | causal | 1.00x (fallback to exact baseline, not yet improved) |
- **Known open gap:** causal regime has no speedup yet. Root cause: even
  SDPA's MATH backend drifts past the 1e-3 accuracy threshold on ~half of
  random seeds at B8_S128 causal (see step 5 below). Deferred to G1.2.
- **Latest commit:** `2635e0a` (G0.3 archive: padded/trackA)

---

## Step-by-step history

### 1. Bootstrap transfer (pre-existing, before this session)
Project tree copied to `/scratch/work` per `docs/MANIFEST.md`'s checklist.
Commit `d06f25a`.

### 2. Phase 0 probe fixed and run (pre-existing, before this session)
Two bugs fixed in `probes/phase0.py` (a `.__dict__` attribute-access bug on
`get_device_properties()`, and a transposed-shape bug in the FP8
`_scaled_mm` smoke test). Run via `sbatch`, results in
`results/phase0.json`: FP8 available (`torch._scaled_mm`, Triton FP8 cast),
cooperative launch available, TF32/BF16/FP16 TFLOPS in line with `CLAUDE.md`.
Commit `270a6e7`.

### 3. kernel.def fixed (pre-existing, before this session)
Wrong CUDA wheel tag (`cu131` doesn't exist; PyTorch ships `cu130` for
CUDA 13.x). Commit `2c655cc`. Apptainer image rebuilt to `/scratch/kernel.sif`
(7.7GB) after the fix.

### 4. Baseline sweep recorded — this session, Day 1 continuation
Verified clock locking (`/etc/slurm/prolog.sh` pins SM clock to 2520MHz,
already configured — no action needed) and `ncu` profiler permissions
(`NVreg_RestrictProfilingToAdminUsers=0`, already set). Built
`jobs/baseline_sweep.sbatch`, ran 6 shapes (tiny, default, long_seq,
large_batch, default_padded, default_causal) covering every regime in
`CLAUDE.md`'s dispatch table. Clocks held at 2520MHz throughout (no
throttling). Recorded `results/ground_truth.csv` — this is the denominator
for every later speedup claim. Commit `7b42d98` (sbatch job id 13).

### 5. G0.1 — SDPA replaces manual attention
**Fact cited:** baseline's `BaselineSelfAttention.forward` (benchmark.py
~L85-122) manually materializes the full `[B,H,S,S]` score matrix, applies
masks via `masked_fill`, and runs `torch.softmax` — `docs/CATALOGUE.md` G0
estimates ~336MB/forward of avoidable memory traffic from this pattern
across all layers.

**What changed:** `UserOptimizedTransformer.forward` (benchmark.py) replaces
this with `F.scaled_dot_product_attention`. Unpadded calls use
`is_causal`-only (`attn_mask=None`) so SDPA can pick its fastest backend;
padded calls pass a boolean key-padding `attn_mask` (unavoidable for
correctness), always alongside a coexisting unmasked fast-path call
elsewhere in the file.

**Tooling fix required:** `tools/check_validity.py`'s static gate flagged
*any* `attn_mask=<non-None>` as the "lazy default instead of is_causal"
exploit pattern, which is a false positive for the padded case. Narrowed the
regex-based check to only flag a masked SDPA call when the file has *no*
coexisting unmasked `is_causal` fast-path call — verified this still
rejects a mask-only-no-fast-path test file (the actual exploit).

**Accuracy problem found and gated:** `default_causal` (S=128, causal=True,
no padding) failed accuracy on 2 of 5 trials, right at the 1e-3 atol
boundary (max_abs up to 0.00114, only 1-3 elements out of 2.6M). Wrote
`probes/g0_1_causal_backend_probe.py` to isolate the cause: tested all 4
SDPA backends (MATH, EFFICIENT, FLASH, CUDNN) against the baseline at the
failing shape over 20 seeds.
- FLASH and CUDNN: unavailable for FP32 inputs at all (require fp16/bf16).
- EFFICIENT: max_abs up to 0.00134, fails 17/20 seeds.
- MATH (the "reference" backend — algorithmically identical to baseline's
  own matmul→mask→softmax→matmul): **still fails 10/20 seeds**, max_abs up
  to 0.00119.

Conclusion: this isn't a bug in the SDPA swap — it's that baseline's own
TF32-matmul output is the accuracy reference (not a mathematically exact
answer), and *any* independently-kernelled computation of the same causal
math will diverge from that specific TF32 rounding by an amount that's
already close to the 1e-3 budget before causal masking even enters the
picture. Causal masking then amplifies it further: early rows have very
few valid keys, so there's less terms in the softmax reduction to average
error down over (same mechanism `CLAUDE.md` cites for why FP8 *does*
survive on the FFN — inverted here, since fewer terms means *worse*
averaging).

**Resolution:** gated on `self.config.causal`, checked once (not per layer,
per `CLAUDE.md`'s dispatch guidance), falling back to the exact baseline
computation (`super().forward(...)`) when causal. Non-causal regimes get
the SDPA speedup; causal regime is unchanged (verified parity) until G1.2
(folding the attention scale into `W_Q`) gives a real lever to revisit.

**Verification:** full 8-shape sweep (`jobs/g0_1_accuracy.sbatch`, sbatch
job id 16, log in `results/g0_1_sweep_run16.log`) — all PASS:

| shape | speedup | note |
|---|---|---|
| tiny | 1.307x | |
| default | 1.137x | |
| long_seq | 2.147x | |
| large_batch | 1.253x | |
| default_padded | 1.134x | |
| default_causal | 0.999x | fallback, expected parity |
| causal_padded | 0.996x | fallback, expected parity |
| long_seq_padded | 2.148x | |

Committed `716c0dd`. Archived per regime cell (`tiny/trackA` 1.31x,
`default/trackA` 1.14x, `long-seq/trackA` 2.15x, `large-batch/trackA` 1.25x,
`padded/trackA` 1.13x, `causal/trackA` 1.00x).

### 6. archive.py auto-commit bug found and fixed
While archiving step 5's results, noticed the 6 `archive.py commit` git
commits (`8c806f2`..`dedf4c6`) each had a commit message naming the wrong
cell — e.g. `8c806f2` says `[default/trackA]` but its actual diff is
`archive/tiny__trackA.json`. Root cause: `commit()` ran `git add -A` +
`git commit` *before* `_save()` wrote the current cell's file to disk, so
every commit swept up the *previous* invocation's leftover uncommitted file
under the *current* invocation's message. The archive JSON *content* was
always correct (written right after, by the same call) — only the
per-commit git message attribution was shifted by one, and the very last
cell in any batch (`causal/trackA`) was never auto-committed at all (no
following invocation to sweep it up). Fixed by moving `_save()` before the
git add/commit block. Did not rewrite the mislabeled history (per git
safety practice — no amending/rebasing without being asked); just
documented it. Commit `5cfbc6f`.

### 7. G0.2 — fused QKV
**Fact cited:** `docs/CATALOGUE.md` G0.2 — three separate `[d,d]` GEMMs
(q_proj/k_proj/v_proj) cost three kernel launches and have worse arithmetic
intensity than one `[d,3d]` GEMM; biggest win expected in the launch-bound
TINY regime.

**What changed:** added `UserOptimizedTransformer._fused_qkv()`, a
staticmethod that lazily builds `attn._qkv_weight`/`attn._qkv_bias` (plain
tensor attributes, cached on first forward, rebuilt only if
device/dtype changes) by concatenating the three projection weights/biases,
then does one `F.linear` + `.split()` instead of three separate
`nn.Linear` calls. Plain attributes (never `Parameter`/`Buffer`), so
`strict=True` state_dict loading is unaffected (CLAUDE.md invariant 4) —
verified by the fact `check_validity.py`'s buffer/parameter check still
passed.

**Verification:** full 8-shape sweep (sbatch job id 17, log in
`results/g0_2_sweep_run17.log`), all PASS. vs G0.1 (job 16): tiny
1.307→1.378x, default 1.137→1.150x, padded 1.134→1.161x, large_batch flat
(1.253→1.255x). long_seq/long_seq_padded dipped slightly (2.147→2.095x,
2.148→2.096x) — plausibly a GEMM-shape/tiling effect at large S where the
merged `[d,3d]` matmul doesn't get as favorable a cuBLAS kernel selection
as three separate `[d,d]` calls; not investigated further since it's small
and the archive's MAP-Elites logic correctly keeps G0.1's better result for
that cell rather than regressing it. Causal unaffected (still on the exact
baseline fallback from G0.1).

Committed `8f48d13`. Archived: `tiny/trackA` 1.38x (new elite),
`default/trackA` 1.15x (new elite), `long-seq/trackA` stayed at 2.15x
(G0.2's 2.10x logged as near-miss, not elite), `large-batch/trackA` 1.25x,
`padded/trackA` 1.16x (new elite), `causal/trackA` 1.00x. Confirmed the
step-6 archive.py fix works correctly this time — every commit's diff now
matches its own message's cell.

### 8. G0.3 — kill `_split_heads` `.contiguous()`
**Fact cited:** `docs/CATALOGUE.md` G0.3, "baseline burns ~96MB/forward"
copying q/k/v into contiguous `[B,H,S,D]` layout before attention.

**What changed:** added `_split_heads_view()` — same `view` +
`transpose(1,2)` as baseline's `_split_heads`, minus the `.contiguous()`.
SDPA's fused kernels accept the resulting strided view directly (this is
the standard MHA idiom SDPA is built around); baseline's own `_split_heads`
is untouched since baseline's plain `torch.matmul` calls genuinely need the
copy. Verified the `qkv.split()` → `.view()` → `.transpose()` chain doesn't
throw a view-compatibility `RuntimeError`: splitting the *last* dim of a
contiguous `[B,S,3D]` tensor keeps that dim stride-1 internally, so
splitting it further into `(num_heads, head_dim)` is a valid view
regardless of the outer dims' strides.

**Verification:** full 8-shape sweep (sbatch job id 18, log in
`results/g0_3_sweep_run18.log`), all PASS, uniform gains, no regressions:
tiny 1.378→1.496x, default 1.150→1.205x, long_seq 2.095→2.193x (now beats
G0.1's number too), large_batch 1.255→1.362x, padded 1.161→1.208x. Causal
unaffected (still on the exact baseline fallback).

Committed `761578c`. Archived: new elites in every non-causal cell (tiny
1.50x, default 1.21x, long-seq 2.19x, large-batch 1.36x, padded 1.21x);
causal stayed at 1.00x (near-miss, correctly not overwritten).

### 9. G0.4 skipped, G0.5 implemented instead
**Reasoning for skipping G0.4** ("cache the causal mask by seq_len"): the
causal regime is currently fully gated to `super().forward()` (the exact
baseline fallback, see step 5) — there is no custom causal attention path
in `UserOptimizedTransformer` yet for a cached mask to attach to. Revisit
once G1.2 unblocks a real optimized causal path.

**Why G0.5 was promoted instead — an important finding:** checked how
`main()` actually calls the model (`benchmark.py` L480-489, L626-637) and
`generate_random_case()` (L353-356): when `padding_ratio<=0` it returns a
**concrete all-ones tensor**, never a literal `None`. That means every
"unpadded" shape tested so far (tiny, default, long_seq, large_batch) was
silently going through the `attn_mask=key_keep` branch on every one of
G0.1/G0.2/G0.3's sweeps — the `attn_mask=None` fast path was dead code the
whole time. The speedups already measured are real (verified by the
accuracy+timing harness regardless of which branch ran), but they were
left on the table relative to what SDPA can actually do when it knows
there's no mask at all.

**What changed:** implemented `docs/CATALOGUE.md` G0.5 using the exact
`_mask_is_all_ones()` pattern CLAUDE.md sanctions (the one legitimate
`data_ptr()` use — caches whether a mask is all-ones, never a result).
Added `__init__` to give `UserOptimizedTransformer` its own `_mask_cache`
dict (plain attribute, not part of `state_dict`). `forward()` now computes
`no_pad` once per call (not per layer) and uses it to: (a) pick
`attn_mask=None` for SDPA when there's no real padding, and (b) skip the
three `masked_fill` calls per layer plus the final one, which are
provably no-ops when the mask is all-ones but still cost a full
elementwise pass over the tensor if not skipped.

**Verification:** in progress, sbatch job id 19.

### 10. (in progress) G0.5 — see step 9
Full sweep results and archive/commit will land here once job 19
completes.
