# Progress Log

Chronological, step-by-step record of what's been done on this project.
Written for two purposes: (1) resuming cleanly after a session crash, and
(2) source material for the write-up report. Updated after every verified
step — not a plan, a record of what actually happened.

**Read `docs/MANIFEST.md`, `CLAUDE.md`, and the "Current State" section below
first** if resuming cold. Then `git log --oneline` and `python3
tools/archive.py summary` for the authoritative current numbers (this file
can lag by one iteration if a crash happened mid-step).

## Session summary (2026-08-26)

Bootstrap (Phase 0, baseline sweep) plus 8 verified Track A optimisations,
one per iteration, each gated on `tools/check_validity.py` → full 8-shape
accuracy sweep → benchmark → `tools/archive.py`:

| # | id | what | regimes affected |
|---|---|---|---|
| G0.1 | SDPA replaces manual attention | all non-causal (causal gated to fallback — accuracy) |
| G0.2 | fused QKV (one `[d,3d]` matmul) | all non-causal |
| G0.3 | skip `.contiguous()` before SDPA | all non-causal |
| G0.5 | all-ones-mask fast path | all non-causal (found the `attn_mask=None` path was dead code) |
| G1.1 | fold LayerNorm affine into consumer linears | all non-causal |
| G1.2 | fold attention scale into `W_Q` | all non-causal (exact, but ~0 speedup under SDPA) |
| G2.4 | `torch.compile` CUDA graphs | all non-causal (found+fixed a real CUDAGraphs pool bug) |
| G2.4b | `torch.compile` on baseline's exact causal math | causal only (tight accuracy margin, disclosed) |

**Final speedups (`python3 tools/archive.py summary`):** tiny 4.65x,
default 1.61x, long-seq 2.35x, large-batch 1.61x, padded 1.60x, causal
1.75x. Every regime improved.

**The one real bug found this session:** G2.4's lazy weight-folding cache
was built *inside* the `torch.compile`d region, handing out a pointer into
CUDA graph pool memory that the next replay overwrote. PyTorch's own
safety net raised a clear `RuntimeError` rather than silently corrupting
output — caught on the first smoke test, fixed by moving weight-folding to
run eagerly before the compiled call. Full story in step 12 below.

**Known gaps / good next steps, not started:** G2.1 (BF16 GEMMs — real
accuracy risk given margins are now tighter than before G2.4/G2.4b), G3
fusion work (FFN tile, warp-shuffle reductions — CUDA/Triton territory),
and the `ncu` profiling infrastructure that `.claude/agents/profiler.md`
expects but doesn't exist yet (see "Current State" below).

---

## Current State (updated after each iteration)

- **Day:** 1-2. **Every regime has a real, verified speedup as of this
  session.** A natural stopping point — see "Session summary" below.
- **Track A progress:** G0.1, G0.2, G0.3, G0.5, G1.1, G1.2, G2.4, G2.4b
  done. G0.4/G0.6 skipped (not applicable / not torch-level).
- **Archive (Track A elites, `python3 tools/archive.py summary`):**
  | regime | trackA |
  |---|---|
  | tiny | 4.65x |
  | default | 1.61x |
  | long-seq | 2.35x |
  | large-batch | 1.61x |
  | padded | 1.60x |
  | causal | **1.75x** (was stuck at 1.00x for most of this session — see step 13) |
- **Accuracy risk to know about:** causal's speedup (G2.4b, step 13) comes
  with a genuinely tight `max_abs` margin (~92-99% of the 0.001 atol budget,
  vs comfortably-under-80% everywhere else). Verified safe via a 40-seed
  probe (0/40 true failures) before shipping, but this is disclosed, not
  comfortable — re-verify if the model/shapes/grading protocol change.
- **BF16 (G2.1 full-model, G2.1b FFN-only) tried twice, failed both times
  at the same magnitude** (steps 14, 17) — ~11x over the accuracy budget,
  ~13% of elements, regardless of scope. Closed off as a direction for
  this model at this depth; not attention-specific. `benchmark.py` is back
  to the clean post-G2.4b state both times; only smoke-test scripts and
  failure logs were kept.
- **`ncu` profiling infrastructure built** (step 15, commit `eb884e7`) —
  real facts: `pct_of_peak`/occupancy are low (23.6%/25.7% default,
  3.8%/11.5% tiny) with no single dominant kernel; DRAM traffic
  (355.77/174.49 GB/s) is roughly weight-re-fetch-sized regardless of
  shape, consistent with the model (75.66MB FP32) exceeding L2 (72MB).
- **G2.3 (L2 persistence) investigated, not pursued** (step 16) — verified
  exact CUDA 13.1 enum/struct values from real headers, but
  `torch.cuda.cudart()` doesn't expose the needed functions; reaching them
  needs raw `ctypes.CDLL` into `libcudart.so`, judged too risky (context-
  mismatch / struct-layout crash risk, no compiler safety net) for the
  catalogued ~10-50% gain.
- **Everything reachable via `torch`-level composition (G0, G1, G2.1-G2.4b)
  is now shipped, or investigated and closed with reasons on record.**
- **G3.1 (fused FFN tile) built in Triton, verified correct, and rejected
  on measurement** (step 19). The kernel matches a float64 ground truth to
  ~2e-6 at `tf32x3`/`ieee`, so it is not a bug — but it is **0.18x (default)
  to 0.869x (large_batch)** of the three `torch` kernels it replaces, and
  its fast `tf32` mode is 3.7x less accurate than cuBLAS's own TF32 (0.6%
  of elements failing in one isolated FFN). Root cause is a roofline fact,
  not tuning: torch's FFN already runs at **60-69% of the 82.6 TFLOPS TF32
  peak**, so the ~48 MB/forward of intermediate traffic that fusion removes
  is already overlapped behind compute. Reverted; `benchmark.py` is
  bit-identical to the `run27`-validated state.
- **G4.0 (two-kernel form) checked for feasibility and not built** (step
  20). The measurement it was gated on came back zero: with G2.4's CUDA
  graph already in place, **summed GPU kernel duration equals wall-clock
  forward time at every shape** (gap −0.7% to +0.0%), so there is no
  GPU-side inter-kernel overhead left for hand-fusion to recover. A
  no-op-kernel graph calibration says the same thing independently
  (0.855 µs marginal per kernel, of which 0.979 µs is the kernel body →
  dispatch gap ≈ 0). `docs/MEGAKERNEL.md`'s own gate — ">15% launch
  overhead or GPU idle at tiny after CUDA Graphs, else stop here" — is
  therefore not met. Pricing the build with step 19's already-measured
  FFN kernel puts the **best case at 0.27x-0.93x**, a regression at every
  shape, before the harder attention half costs anything.
- **The archive numbers above are unchanged** — no candidate has improved
  on `results/g2_4b_sweep_run27.log` since G2.4b.
- **The `docs/CATALOGUE.md` ladder is now exhausted at this precision.**
  G0/G1/G2.4/G2.4b shipped; G2.1/G2.1b/G2.6 closed on accuracy (steps 14,
  17, 18); G2.3 closed on ctypes risk (step 16); G3.1 closed on the
  roofline (step 19); G4.0 closed on the graph-gap measurement (step 20);
  G4.1-G4.5 all sit on top of the FP8 pipeline step 18 measured at 65x
  over budget.
- **Latest commit:** `3e831d6` (steady-state ncu job); step 20's write-up
  follows in the next commit.

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

### 10. G0.5 — all-ones-mask fast path
**Verification:** full 8-shape sweep (sbatch job id 19, log in
`results/g0_5_sweep_run19.log`), all PASS, `max_abs` unchanged from G0.3's
run per shape (as expected — G0.5 doesn't change any math, just which
already-equivalent path runs). vs G0.3:

| shape | G0.3 | G0.5 | note |
|---|---|---|---|
| tiny | 1.496x | **2.640x** | the dead fast path, unlocked |
| default | 1.205x | 1.397x | |
| long_seq | 2.193x | 2.300x | |
| large_batch | 1.362x | 1.547x | |
| default_padded | 1.208x | 1.211x | genuinely padded, same path as before |
| long_seq_padded | 2.194x | 2.194x | genuinely padded, same path as before |

The padded shapes staying flat is itself a good sign — it confirms `no_pad`
is correctly `False` for them (the real-padding path is untouched) and
`True` only for the genuinely-unpadded ones, i.e. the fast-path detection
is working as intended, not just "faster because something got skipped
incorrectly."

Committed `7530cc8`. Archived: new elites in every non-causal cell (tiny
**2.64x**, default 1.40x, long-seq 2.30x, large-batch 1.55x, padded 1.21x
— tied with G0.3's number since padded takes the same path, logged
correctly as tied/near-miss-equal); causal stayed at 1.00x.

**G0 is now essentially done for Track A.** G0.6 (128-bit vector loads) is
deferred — it's a hand-written-kernel-level micro-optimisation not really
expressible at the `torch`/`F.*` level this file operates at; would only
become relevant if/when this drops to Triton or CUDA for other reasons
(G3/G4 territory).

### 11. G1.2 — fold attention scale into `W_Q`
Moved to G1 (constant folding), prioritising G1.2 first out of
`docs/CATALOGUE.md`'s order (G1.1 → G1.2 → ...) because it was flagged
(step 5) as a possible way to unblock the causal regime's accuracy gap.

**What changed:** in `_fused_qkv`, the Q-rows of the fused weight/bias are
pre-multiplied by `attn.scale` (in-place on the freshly-`cat`'d tensor, so
`attn.q_proj.weight` itself is never touched), and both SDPA calls pass
`scale=1.0` instead of relying on the default `1/sqrt(head_dim)`.

**Why this is provably exact, not just "should be close":** this model's
`head_dim` is always a power of two (64, from `d_model=512 / num_heads=8`),
so `scale = head_dim**-0.5 = 0.125 = 2^-3` is an *exact* power of two. IEEE
float multiplication by a power of two never rounds (it's purely an
exponent shift), so `sum_k(a_k * b_k * s) == sum_k(a_k*b_k) * s` bit-for-bit
when `s` is a power of two — pre-scaling Q and using `scale=1.0` is
bit-identical to SDPA's default (scaling the `[B,H,S,S]` score matrix by
`0.125` after the matmul), not merely mathematically equivalent.

**Verified per CLAUDE.md invariant 5** (must be bit-identical, not just
under threshold): diffed every per-trial `max_abs`/`max_rel`/`failed`
line (sbatch job id 20, `results/g1_2_sweep_run20.log`) against G0.5's run
(`results/g0_5_sweep_run19.log`) — **identical across all 8 shapes × 5
trials**, confirming the proof empirically. Only the `speedup` numbers
differ.

**Speedup: flat, within noise** (tiny 2.640x→2.585x, default 1.397x→1.392x,
long_seq 2.300x→2.301x, others similar). SDPA's MATH backend apparently
performs the scale-multiply on the score matrix regardless of whether
`scale` is `1.0` or `0.125` — pre-scaling the much smaller `[B,S,d_model]`
Q tensor doesn't remove that cost, it just relocates a multiply from one
place to another of similar total size. **This contradicts
`docs/CATALOGUE.md`'s general framing of G1.2** ("removes elementwise
multiply over all `[B,H,S,S]`") — that estimate was written with a manual
matmul-based implementation in mind (like baseline's own), where the
`[B,H,S,S]`-sized multiply really is removable; it doesn't hold for an
SDPA-based implementation where SDPA owns that step internally.

**Kept anyway:** it's free (verified exact, no regression) and
`docs/CATALOGUE.md` states G1 folds are a hard prerequisite for G4 ("a
megakernel cannot fold affines at runtime... G1 is a prerequisite of G4,
not a stepping stone to it") — legitimate infrastructure work for later
even without an immediate payoff.

Committed `1d9b585`. Archived per cell (mostly near-misses vs G0.5's
numbers, i.e. correctly recognized as flat, not regressions; long-seq and
padded ticked up marginally — noise, not attributable to the fold itself
per the bit-identical accuracy proof above).

**Did NOT attempt un-gating causal with the scale-folded machinery** (this
was the planned next step). Reasoning: the bit-exactness proof above
applies unconditionally, independent of causal masking (masking happens
*after* the `QK^T * scale` step) — so a causal path built on the same
fold would produce the *exact same floating-point numbers* as an unfolded
one, and therefore the *exact same* ~50%-of-seeds failure rate already
measured in `probes/g0_1_causal_backend_probe.py`. Running that experiment
on the GPU would have re-confirmed something the math already establishes
— skipped to avoid burning a job on a predictable null result (the user
asked to be token/resource-conscious). Causal remains gated to the exact
baseline fallback; unblocking it needs a genuinely different lever than
scale-folding (open question, not yet identified — candidates to consider
later: accepting a BF16 walk-down for causal specifically, or revisiting
whether the accuracy budget is actually the binding constraint the grader
enforces vs. this benchmark script's own defaults).

### 12. G2.4 — lazy `torch.compile(mode="reduce-overhead")`, CUDA graphs
**Fact cited (no formal ncu profile — infra gap, see below):** tiny's
speedup jumped far more than every other regime's at each of G0.2/G0.3/G0.5
(fewer GEMMs, no `.contiguous()`, skipped `masked_fill`) — itself strong
evidence tiny is still launch/overhead-bound post-G0/G1, which is exactly
what CUDA graphs target. Matches `docs/CATALOGUE.md`'s own "3x+ tiny"
estimate for G2.4 specifically (vs "1.2x default").

**`docs/AGENTS.md`/`CLAUDE.md` want a profiler-cited fact for every
proposal, and this repo doesn't have working `ncu`-profiling infrastructure
yet** — `.claude/agents/profiler.md` (the profiler subagent) expects a
`jobs/*.sbatch` + `tools/` ncu-to-JSON parser to already exist, per
`docs/SETUP.md`'s "Parse `ncu` to JSON in `tools/` — never let raw output
reach an agent's context," and neither has been built. Used measured
end-to-end timing as a substitute fact instead of fabricating a profiler
run. **This is a real gap to close later** (see "Next up" ideas at the
bottom of this file) — every G2/G3/G4 decision from here on would benefit
from real `SpeedOfLight`/occupancy/stall-reason data instead of inference
from timing deltas.

**What changed:** `forward()` now: check `config.causal` → compute `no_pad`
eagerly → `_ensure_folded_weights()` eagerly → lazily build
`self._compiled_impl = torch.compile(self._optimized_forward,
mode="reduce-overhead")` on first call → delegate. Compiled **lazily on
first forward**, never via `--compile-user` (`CLAUDE.md` is explicit the
grader may not pass that flag).

**A real bug was found and fixed before the full sweep — this is the most
instructive failure of the session so far.** First smoke test (tiny +
default only, `results/g2_4_smoke_FAILED_run22.log`) crashed:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run. Stack trace: ... in _fused_qkv
    b = b + w @ norm1.bias
```

Root cause: `_fused_qkv`'s lazy weight-folding cache (`torch.cat(...)`,
built once via `getattr(attn, "_qkv_weight", None)` and cached as a plain
attribute — the exact pattern used safely for G0.2/G1.1/G1.2 up to this
point) was being **built inside the compiled/graphed region**. Constructing
a fresh tensor there and then caching a Python-level reference to it
*across calls* hands out a pointer into the CUDA graph's internal memory
pool — memory the *next* graph replay reclaims and overwrites. PyTorch's
own safety net catches this and raises rather than silently returning
corrupted data, which is the good version of this failure mode: a
loud, immediate crash on the very first accuracy trial, not a subtle
wrong-answer bug that accuracy checking might have missed.

**Fix:** split weight-folding into `_build_qkv_fold` /
`_build_ffn_in_fold` / `_ensure_folded_weights` — called from `forward()`,
**eagerly, before** the compiled call, never from inside
`_optimized_forward`. The compiled function now only ever *reads*
`attn._qkv_weight` / `layer._ffn_in_weight` as already-stable tensors, the
same way it would read a frozen `nn.Parameter` — no caching, no
`torch.cat`, no conditional building happens inside the traced region
anymore. Retest (`results/g2_4_smoke_fixed_run23.log`) passed cleanly, no
errors, and with dramatically better numbers than the smoke test even
showed pre-fix hope for (tiny 2.692x → 4.638x on the smoke test alone).

**Verification:** full 8-shape sweep (sbatch job id 24,
`results/g2_4_sweep_run24.log`), all PASS. vs G1.1
(`results/g1_1_sweep_run21.log`):

| shape | G1.1 | G2.4 | note |
|---|---|---|---|
| tiny | 2.692x | **4.649x** | the launch-bound hypothesis, confirmed hard |
| default | 1.355x | 1.601x | |
| long_seq | 2.302x | 2.355x | |
| large_batch | 1.545x | 1.609x | |
| default_padded | 1.204x | **1.604x** | +33% |
| default_causal | 0.999x | 1.000x | fallback, untouched |
| causal_padded | 1.001x | 1.001x | fallback, untouched |
| long_seq_padded | 2.195x | 2.237x | |

`max_abs` moved in **both directions** across shapes (tiny/long_seq/
large_batch went down, default/long_seq_padded went up ~10-15%) — flagged
explicitly rather than glossed over, since this is genuinely different from
every fold so far: G2.4 is **not** a claimed-exact transform (unlike
G1.1/G1.2) — inductor's Triton-generated kernels use different fusion/
reduction order than eager PyTorch, so some drift either direction is
expected. All values stay well under the 0.001 atol budget (worst case
~80% of it). Causal's `max_abs` stays exactly `0` (fallback untouched).

Committed `e561559` (implementation + both smoke-test logs + full sweep
log, so the bug-then-fix story is preserved in git history, not just this
file). Archived: new elites in every non-causal cell (tiny 4.65x, default
1.60x, long-seq 2.35x, large-batch 1.61x, padded 1.60x); causal stayed at
1.00x (near-miss, correctly not overwritten).

### 13. (in progress) Fresh idea for causal — compile the exact baseline math
The gap between causal (1.00x) and everything else (1.6-4.65x) just got a
lot wider, which makes it worth another look rather than leaving it parked.
Every attempt so far tried to make an *independent* computation (SDPA, any
backend) match baseline's specific TF32-rounded output within 1e-3 — and
every one hit the same wall, because that's fundamentally about matching
one specific floating-point rounding pattern with a different kernel, which
isn't reliably achievable at this depth (steps 5, 11).

**Different angle this time:** don't change the computation *at all* for
causal — keep calling `super().forward()`, baseline's own exact
attention/LN/FFN code, unfused, unfolded, using the original `q_proj`/
`k_proj`/`v_proj`/`norm1`/`norm2` parameters directly. Just wrap *that* in
`torch.compile(mode="reduce-overhead")` the same way G2.4 did for the
non-causal path, purely for launch-overhead reduction. Since baseline's
`forward()` has no lazy weight-caching (it only ever reads existing
`nn.Parameter`s, the standard case torch.compile/cudagraphs is built to
handle), the specific bug from step 12 shouldn't apply here — but
inductor's fusion can still shift `max_abs` away from the current exact
`0`, and causal is already known to be razor-thin against the 1e-3 budget
(step 5), so this needs the same smoke-test-first caution as G2.4, not an
assumption that "no lazy caching" means "no risk."

**It worked, with a real accuracy trade-off, disclosed rather than
buried.** Smoke test (`default_causal` + `causal_padded` only,
`results/g2_4b_smoke_run25.log`) passed its 5 trials — but at
`max_abs=0.000994682`, **99.5% of the 0.001 atol budget**, where it was
previously exactly `0`. A 5-trial pass at that margin isn't strong
evidence by itself (the earlier SDPA-based causal attempts failed on
~50% of held-out seeds at a similar margin).

**Before shipping, ran a proper diagnostic** — `probes/g2_4b_causal_compile_probe.py`,
40 seeds, checking `benchmark.py`'s actual *disjunctive* pass/fail rule
(`abs_ok OR rel_ok` per element) rather than just "did `max_abs` cross
`atol`". Result (`results/g2_4b_probe_run26.log`):

```
seeds=40 true_failures(abs&rel both exceed)=0 max_abs_over_atol_alone=4
max=0.001175 mean=0.000921 min=0.000780
```

**0 of 40 seeds had a true failure** (an element where both criteria are
violated simultaneously). 4/40 seeds briefly exceeded `atol` alone, but
were consistently saved by the relative-error criterion — because (unlike
the SDPA attempts' failure mode) the elements involved aren't
near-zero-magnitude. This is a categorically different, much safer
situation than the SDPA attempts' ~50% true-failure rate, not a lucky
coincidence: the underlying values here have real magnitude, so a small
absolute drift from inductor's fusion doesn't simultaneously blow up the
relative error the way it did when SDPA's kernel disagreed with baseline
at near-zero causal-mask-boundary values.

**Judged safe enough to ship, with the margin disclosed plainly** (commit
`2aaa0b1`) — but this is a *tight* margin (mean `max_abs` ~92% of budget
across 40 seeds), not a comfortable one. Worth re-checking if the model,
shapes, or grading protocol ever change.

**Verification:** full 8-shape sweep (sbatch job id 27,
`results/g2_4b_sweep_run27.log`), all PASS. Non-causal shapes essentially
unchanged from G2.4 (`results/g2_4_sweep_run24.log`, small ±1% noise).
Causal:

| shape | before | after |
|---|---|---|
| default_causal | 1.000x | **1.747x** |
| causal_padded | 1.001x | **1.753x** |

Archived: new elites for `default/trackA` (1.61x), `large-batch/trackA`
(1.61x, tied), and — the headline result — `causal/trackA` **1.75x, up
from 1.00x**. `tiny`/`long-seq`/`padded` logged as near-misses (tiny
4.59x vs the standing 4.65x elite — noise, not a regression from this
diff, which doesn't touch the non-causal path at all).

**Every regime now has a real, verified speedup for the first time this
session.**

### 14. G2.1 attempted (BF16 GEMMs) — failed decisively, reverted
**What was tried:** full BF16 compute path for the non-causal loop —
QKV projection, SDPA (Q/K/V cast to bf16, which also unlocks flash/
efficient SDPA backends that FP32 disqualifies entirely), `out_proj`,
and the whole FFN (`ffn_in` → GELU → `ffn_out`), all in bf16, with the
residual stream `x` kept FP32 throughout and cast-down/cast-up at each
GEMM boundary — the textbook G2.1 policy (`docs/CATALOGUE.md`: "cast at
GEMM inputs only → one rounding per layer, not compounded through the
residual"). BF16 weight copies cached eagerly (extending
`_build_qkv_fold`/`_build_ffn_in_fold`, same eager-only rule as their
FP32 counterparts, to avoid repeating step 12's CUDAGraphs bug).

**Result: decisive failure, not a borderline call.** Smoke test (tiny
shape, `results/g2_1_smoke_FAILED_run28.log`) failed all 5 trials:
`max_abs` up to **0.0110 — 11x the 0.001 atol budget** — with **13.6% of
all elements failing** (22344/163840) the actual disjunctive
`abs_ok OR rel_ok` criterion, not just exceeding one metric. Failing
elements span broadly across feature dimensions (not a narrow subset),
and the worst-case values involve normal O(1)-magnitude activations
(e.g. baseline=-1.6976763 vs optimized=-1.7073666), not near-zero
edge cases — ruling out the "near-zero reference blows up relative
error" pattern seen in every prior accuracy investigation this session.
**Reverted immediately** (`git checkout HEAD -- benchmark.py`, HEAD was
`d9a7e7b`) without running the usual 40-seed diagnostic probe — a
gap this large (11x over budget, on the very first shape) doesn't need
more sampling to be judged unsafe; that probing effort is for genuinely
close calls (compare step 13, where the smoke test was at 99.5% of
budget, not 1100%).

**Why this probably isn't a bug, and what it means:**
`docs/CLAUDE.md`'s own precision walk-down ladder — `FP8 FFN+BF16 attn
→ FP8 FFN only → split-precision → BF16 everywhere` — lists "BF16
everywhere" as the **last, most conservative** rung, the fallback
you're supposed to land on after everything more aggressive (FP8) has
already failed. This attempt was essentially that last rung (bf16 in
both attention and FFN, no FP8 anywhere) — and it failed hard. Back-
of-envelope: BF16 has ~0.39% unit roundoff; injecting that at ~12
GEMM/attention boundaries (2 per layer × 6 layers) into an FP32
residual, even without full compounding, plausibly reaches low-single-
digit-percent relative error for a meaningful fraction of elements —
enough to breach the 1% `rtol` (which, for the O(1)-magnitude values
here, is the *effective* governing bound in the disjunctive criterion,
looser than the ~0.1%-relative-equivalent `atol`). If the safest rung
of the documented ladder fails this decisively, the more aggressive
rungs above it (which add FP8, strictly lower precision, into the mix)
are very unlikely to fare better without a fundamentally different
technique (e.g. per-channel scales doing real work, or the
split-precision `A_hi + A_lo` trick) — not something to attempt casually
on the strength of the catalogue's general gain estimate alone.

**Not concluding BF16/FP8 is impossible for this model** — only that
applying it uniformly everywhere doesn't work at this depth (6 layers).
A narrower attempt (BF16 for the FFN only, leaving attention on the
proven TF32/SDPA path from G0-G2.4b) is a plausible follow-up with a
smaller blast radius, but wasn't attempted this session — moving to
lower-risk, clearly-valuable work instead (the `ncu` profiling
infrastructure gap, flagged repeatedly since step 12) rather than
immediately re-attempting a narrower slice of a direction that just
failed hard, without profiler data to justify the investment.

### 15. `ncu` profiling infrastructure — built, closes the step-12 gap
See commit `eb884e7` for the full story (kept concise here, the commit
message has the detail): `.claude/agents/profiler.md` expected
`tools/`-based ncu-to-JSON parsing that never existed. Built
`tools/parse_ncu.py` + `jobs/profile.sbatch` (runs a **minimal** invocation
— 1 accuracy trial, 1 warmup, 1 repeat/round — since `ncu`'s per-kernel
overhead makes profiling the full timing sweep impractical; `--launch-count
120` caps it to roughly one forward pass). Two real format issues found
and fixed along the way (`jobs/ncu_header_check*.sbatch`): ncu's banner and
the profiled program's own prints share the CSV's stdout stream ahead of
it, and the actual `--csv` output is **long format** (one row per
kernel-launch-ID per metric, not one row per kernel with a metric per
column) — both discovered by capping output size at every step so nothing
large ever hit context.

**Real facts learned** (`results/ncu_profile_{default,tiny}_run{34,35}.json`):

| | default (B8_S128) | tiny (B1_S64) |
|---|---|---|
| pct_of_peak (SM, TF32) | 23.64% | 3.79% |
| occupancy | 25.73% | 11.48% |
| dram_gbps | 355.77 | 174.49 |
| top-3 kernels | real CUTLASS TF32 GEMMs (`s1688gemm_128x64_16x6`), none >4.4% of total | same kernel family (`s1688gemm_64x64_32x6`), none >2.7% |
| bank_conflicts | 40,030 | 2,002 |

Both compute and occupancy are low with **no single dominant kernel** —
death by many small, individually-underutilized launches, not one
bottleneck. `dram_gbps` is the same order of magnitude for tiny and
default despite tiny doing far less work, consistent with **per-call
weight re-fetching** rather than per-token traffic: this model is 75.66MB
in FP32, just over the 72MB L2 capacity (`CLAUDE.md`'s own ground truth —
BF16's 37.83MB would fit, FP32 doesn't), so weights don't stay L2-resident
across calls without explicit pinning. This is the fact that motivated
step 16's G2.3 investigation below. `bank_conflicts` are real but live
inside library CUTLASS kernels I don't control from `torch`-level code —
not actionable without hand-writing the GEMM (G3/G4 territory).

### 16. G2.3 (L2 persistence) investigated — not reachable without real risk
Motivated directly by step 15's DRAM-traffic finding.
`docs/CATALOGUE.md`'s G2.3 snippet calls `cudaDeviceSetLimit` and
`cudaStreamSetAttribute` — checked whether these are reachable from
Python without writing a C++ extension.

Verified the exact values from this container's real CUDA 13.1 headers
(never guessed — a wrong enum value here risks silent misbehavior, a wrong
struct layout risks an actual crash):
`cudaLimitPersistingL2CacheSize = 0x06`; `struct cudaAccessPolicyWindow {
void *base_ptr; size_t num_bytes; float hitRatio; enum cudaAccessProperty
hitProp; enum cudaAccessProperty missProp; }`; and confirmed CUDA 13.1 has
folded the old `cudaStreamAttrID`/`cudaStreamAttrValue` into a newer
unified `cudaLaunchAttributeID`/`cudaLaunchAttributeValue` (the old names
are now `#define` aliases). Full excerpt: `results/l2_persist_discover_run36.log`.

**`torch.cuda.cudart()` does not expose either function** — its filtered
`dir()` came back empty for `Limit`/`Attribute`/`StreamSet`. It's a small,
curated set of bindings (profiler start/stop and similar), not a general
cudart passthrough. Reaching these functions would mean `ctypes.CDLL`-ing
`libcudart.so` directly, bypassing torch's own binding — with a real,
non-hypothetical risk: a separately-`dlopen`'d instance of the runtime
could operate on a different loaded-library state than the one torch's own
CUDA context actually uses, making the calls silently ineffective or, if
the struct/ABI details are even slightly off, causing an actual crash.
There's no C++ compiler in the loop to catch a struct-layout mistake before
it reaches a running CUDA kernel launch.

**Decision: not pursuing this via raw ctypes.** The catalogued gain
(`docs/CATALOGUE.md`: "1.1x default, 1.5x+ tiny") doesn't justify that risk
profile for a benchmark harness with no extension-building safety net.
Documented the verified enum values here so this is a much smaller lift if
picked up later with a proper (however small) C++/pybind11 extension
instead of bare ctypes. Moving to a different next step instead (step 17).

### 17. G2.1b (BF16, FFN only) — also failed decisively, closes the question
**What was tried:** the natural follow-up to step 14 — since "BF16
everywhere" failed hard, does isolating BF16 to just the FFN (leaving
attention on the proven TF32/SDPA path from G0-G2.4b untouched) fare
better? FFN is a large fraction of the model's parameters (`ffn_in`
`[2048,512]` + `ffn_out` `[512,2048]` per layer, ~2.1M params each — most
of each layer's weight budget), so it's a reasonable place to isolate.
Same eager-cached bf16-weight pattern as G2.1, scoped to just
`_build_ffn_in_fold`'s bf16 copies and the FFN half of `_optimized_forward`.

**Result: failed just as decisively, at nearly identical magnitude.**
Smoke test (tiny, `results/g2_1b_smoke_FAILED_run37.log`): `max_abs` up to
**0.0112** (vs G2.1's 0.0110 — essentially the same), **12.7% of elements**
true-failing (20880/163840, vs G2.1's 13.6%). Reverted immediately, same
as step 14.

**This is a clean, informative result, not a wash.** Two independent
attempts — full-model BF16 and FFN-only BF16 — landed at the *same*
error magnitude. That rules out "it was specifically attention's softmax
sensitivity" as the cause (step 14 couldn't distinguish this) and points
instead at something more fundamental: quantizing the FFN's own weights to
BF16 (2.1M+ params per layer, the majority of each layer's weight budget)
is, by itself, sufficient to blow the 1% relative budget for a meaningful
fraction of elements over 6 layers. **Precision reduction (BF16, and by
extension the more aggressive FP8) is closed off as a direction for this
model at this depth**, not just "attention was the problem" — a genuinely
different and more useful conclusion than step 14 alone gave.

Committed alongside this write-up: `results/g2_1b_smoke_FAILED_run37.log`
(no `benchmark.py` changes to commit — cleanly reverted to the G2.3-
investigation state).

**This is the natural checkpoint to report back, not silently continue
into G3/G4.** Every remaining `docs/CATALOGUE.md` item within reach of
`torch`-level composition (G0, G1, G2.1-G2.4b) has now been tried, shipped,
or ruled out with real evidence:
- **Shipped and verified:** G0.1, G0.2, G0.3, G0.5, G1.1, G1.2, G2.4,
  G2.4b — every regime has a real speedup.
- **Investigated and closed, with reasons on record:** G2.1/G2.1b (BF16 —
  fails ~11x over budget regardless of scope), G2.3 (L2 persistence — not
  reachable from Python without raw-ctypes risk this session judged not
  worth taking).
- **Not attempted, and genuinely different in kind:** G3 fusion (G3.1
  fused FFN tile, G3.2 fused LN+residual, G3.3-3.7 warp-level/layout work)
  and G4 megakernel — these need hand-written Triton or CUDA kernels, not
  composition of existing `torch`/`F.*` calls. That's a different scale
  and risk profile of work than anything done this session, and per the
  user's own note about escalating to Opus for genuinely hard coding, a
  reasonable point to check in on before starting rather than committing
  to it silently.

**User decision (after the above checkpoint): proceed through G3 into G4,
escalating to Opus for hard kernel-design work, no further check-ins
requested.** The following steps continue under that instruction.

### 18. G4 FP8 accuracy precheck — decisive failure, gates the megakernel design
Before investing in G4's megakernel machinery (persistent cooperative
kernel, warp specialization, hand-written `mma.sync` PTX — genuinely weeks
of work per `docs/MEGAKERNEL.md`'s own scope), checked whether its core
precision requirement is even viable. `docs/MEGAKERNEL.md`'s shared-memory
budget arithmetic is explicit that **FP8 isn't chosen for its 2x
throughput — it's required for pipeline depth**: BF16 tiling caps at 3
`cp.async` stages (96KB overflows the 99KB/SM budget by 3KB at the
evaluated tile), while FP8 reaches 4-5 stages with room to spare. "Shallow
pipelining on Ada costs >30% duty cycle" — so the megakernel design, as
specified, doesn't have a BF16 fallback; it needs FP8 weights to work as
designed.

Given BF16 already failed decisively twice this session (steps 14, 17:
~11x over budget, both full-model and FFN-only), this was worth checking
*before* writing any megakernel code, not after. Wrote
`probes/g4_fp8_accuracy_precheck.py`: a fair test, not a strawman —
**per-output-channel scales** on the FFN weights (`amax` per row / 448,
matching `docs/CATALOGUE.md`'s own G1.5/G2.6 prescription that per-channel
scaling is what makes FP8 viable in principle) and a dynamic per-tensor
scale on activations, simulated via quantize-to-`float8_e4m3fn`-and-back
(isolates the numerical precision question from which specific GEMM
kernel API computes it — if the quantization error alone already fails,
no kernel choice fixes that). Tested the FFN (the 65% of FLOPs
`CLAUDE.md`'s precision policy targets) across 20 seeds, checking the
same disjunctive `abs_ok OR rel_ok` criterion `benchmark.py` actually
uses.

**Result** (`results/g4_fp8_precheck_run38.log`):
```
seeds=20 true_failures=20 max=0.077889 mean=0.065174 min=0.059122
```
**20 of 20 seeds fail**, at a much larger margin than BF16's failure —
mean `max_abs` is **65x the 0.001 budget**, worst case 78x. Even with
correct per-channel scaling (not a naive per-tensor strawman), FP8
quantization error on this model's FFN weights is roughly an order of
magnitude past merely "over budget."

**What this means for G4:** the megakernel's core precision assumption
doesn't hold for this model at this depth. G4.1 onward (persistent kernel,
K-splitting, warp specialization, FP16-accumulate) are all built on top of
the FP8-tiled pipeline `docs/MEGAKERNEL.md` specifies — pursuing them
without a working precision base isn't a smaller version of the same
plan, it's building on a foundation already shown not to hold. **G4.0
(two-kernel form) remains the realistic target**: it doesn't carry the
same pipeline-depth argument for FP8 (it's not doing persistent
multi-stage residency across a long-running cooperative loop the way
G4.1+ is), so it can run in the TF32/FP32 precision already proven safe
all session. `docs/MEGAKERNEL.md` itself sanctions this outcome
explicitly: "G4.0 winning is a result, not a failure — record it in the
archive and move to another cell." Treating that as the realistic ceiling
for G4 this session, not a placeholder on the way to G4.1+.

### 19. G3.1 (fused FFN tile, Triton) — built, verified correct, and rejected on measurement
**What was built.** A real Triton kernel fusing `ffn_in -> GELU(exact) ->
ffn_out` into one launch, keeping the `[tokens, 2048]` intermediate in
registers so it never reaches HBM. Token-parallel (one program owns
`BLOCK_M` tokens and produces their full `[BLOCK_M, 512]` output row), so
no grid sync and no partial sums — exactly `docs/CATALOGUE.md`'s G3.1
shape. FP32/TF32 throughout: steps 14 and 17 closed off precision
reduction for this model, so this was never going to be a BF16 design.

**Tile budget** (fp32 => `b = 4` bytes; Ada ceiling 101,376 B = 99 KB/SM),
computed the way `docs/MEGAKERNEL.md` models it, for the best config
`BLOCK_M=64, BLOCK_N=32, BLOCK_K=64`, 8 warps = 256 threads:

```
dot1  A = x  [64, 64]  = 16.0 KB      dot2  A = h  [64, 32]  =  8.0 KB
      B = w1 [32, 64]  =  8.0 KB            B = w2 [512, 32] = 64.0 KB
      -> 24.0 KB/stage                      -> 72.0 KB
peak shared ~= max(N_stage*24.0, 72.0) + ~4 KB = 76 KB  <=  99 KB   OK
accumulator acc[64, 512] = 128 KB registers = 128 regs/thread @ 256 thr
```

Two hard walls fall straight out of this and both were confirmed by the
hardware, not assumed. `BLOCK_N=64` needs `512*64*4 = 128 KB` for dot2's B
operand alone; Triton refused it loudly — *"out of resource: shared memory,
Required: 147456, Hardware limit: 101376"* — matching the arithmetic. And
`BLOCK_M=128` puts `acc` at 256 KB of registers (256 regs/thread, over the
255 architectural max): it compiled but spilled, measuring **26.6 ms vs
3.1 ms** for `BLOCK_M=64` at M=32768, a 8.6x spill penalty. The 512-wide
accumulator is the structural cost of this fusion — the second GEMM
reduces over `ffn_dim`, so `acc` must stay `[BLOCK_M, 512]` live across the
entire loop, where cuBLAS is free to pick a 64- or 128-wide output tile.

**The kernel is correct.** This was verified against a float64 ground
truth, not against the thing it was competing with
(`results/g3_1_precision_run40.log`), at M=1024:

| | max_abs vs fp64 |
|---|---|
| cublas_fp32 | 2.454e-06 |
| **triton_tf32x3** | **1.802e-06** |
| **triton_ieee** | **5.300e-06** |
| cublas_tf32 | 1.344e-03 |
| triton_tf32 | 5.020e-03 |

At `tf32x3`/`ieee` the kernel lands on top of exact FP32 — indexing,
strided weight views, bias handling and the erf GELU are all right, and
both score **0 failing elements** against the graded cuBLAS-TF32 reference.
`tl.erf` vs torch's erf differs by 4.768e-07 in isolation, so GELU was
never a suspect.

**Blocker 1 — plain `tf32` is 3.7x less accurate than cuBLAS's own TF32**
(5.02e-03 vs 1.34e-03 against fp64), failing **0.6% of elements**
(3189/524288) in a *single isolated FFN*. Consistent with Triton truncating
to TF32 rather than round-to-nearest: a biased error grows like `K` instead
of `sqrt(K)` across the K=2048 reduction. The model chains six of these
into a residual stream whose `max_abs` already sits at 65-79% of the atol
budget (`results/g2_4b_sweep_run27.log`), so this was disqualifying on its
own. The two modes that *do* pass cost 4.5-5x (`tf32x3` 10.86 ms, `ieee`
12.65 ms vs torch 2.39 ms at M=32768).

**Blocker 2 — it is slower than the three kernels it replaces, at every
shape.** Best config per token count, fused-vs-torch, TF32
(`results/g3_1_pretransposed_run41.log`):

| M | torch (3 kernels) | best fused | ratio | blocks |
|---|---|---|---|---|
| 1024 (default) | 0.0864 ms | 0.4799 ms | **0.180x** | 16 |
| 8192 (long_seq) | 0.5885 ms | 0.7178 ms | **0.820x** | 128 |
| 32768 (large_batch) | 2.3975 ms | 2.7580 ms | **0.869x** | 512 |

Two structural hypotheses were tested and priced rather than argued about.
Pre-transposing both weights into contiguous buffers to eliminate
`tl.trans` from both dots (fp32 MMA operand layouts don't fold a transpose
into the shared-memory descriptor for free the way fp16 does) plus
`num_stages=3` moved the ceiling from 0.72x to 0.869x — a real gain, and
still short of parity. Small M is parallelism-starved exactly as the design
predicts: the `ffn_dim` reduction can't be split across programs without
writing partials back to HBM, which is the very traffic G3.1 exists to
remove, so the grid is only `ceil(M/BLOCK_M)` blocks — 16 blocks on 128 SMs
at the default shape.

**Why fusion cannot win here, which is the real result.** The saved traffic
is already off the critical path. At M=32768 one FFN is
`2 * (2*32768*512*2048) = 137.4 GFLOP`; torch does it in 2.3975 ms =
**57.3 TFLOPS, 69% of the 82.6 TFLOPS TF32 roofline**. The intermediate
costs `4*32768*2048*4 = 1.07 GB`, ~1.07 ms at ~1 TB/s — comfortably
overlapped behind 2.40 ms of compute. At M=1024 it is 49.7 TFLOPS (60% of
peak) with 33.5 MB of intermediate. `docs/CATALOGUE.md`'s "saves ~48 MB/
forward" is arithmetically true and strategically irrelevant: cuBLAS is
already compute-bound on this FFN, so removing that traffic buys almost
nothing, while the fused structure gives up the tiling freedom that gets
cuBLAS to 69% of peak in the first place. Note this does **not** contradict
step 15's DRAM finding — that traffic is per-call *weight re-fetch* (the
75.66 MB model vs 72 MB L2), which is a different tensor and unaffected by
fusing the activation intermediate.

**Reproducible.** The kernel was developed inside `benchmark.py` and, on
revert, extracted verbatim to `probes/g3_1_kernel.py` so the measurement can
be re-run rather than merely asserted. Re-running the precision probe from the
committed tree (`results/g3_1_precision_repro_run42.log`) reproduces
`results/g3_1_precision_run40.log` byte-identically.

**Reverted.** `git checkout HEAD -- benchmark.py`; `git diff HEAD --
benchmark.py` is empty, so the shipped file is bit-identical to the
state `results/g2_4b_sweep_run27.log` validated — no re-sweep needed to
trust it, and `tools/check_validity.py` passes. Kept the three probes and
their logs. No archive cell was touched: nothing improved.

**What would have to change for G3.1 to be worth revisiting:** a shape
where the FFN is genuinely memory-bound rather than at 60-69% of the TF32
roofline (much smaller `ffn_dim`, or a precision that halves the compute
time without halving the traffic — and precision reduction is closed by
steps 14/17/18). Not a tuning problem; the roofline says the headroom
isn't there.

### 20. G4.0 (two-kernel form) — feasibility check first; measured to zero headroom, not built

Step 19 rejected G3.1 on a roofline fact rather than a tuning failure, and
G4.0 is a strictly larger version of the same bet (fuse *more* ops into
*fewer* hand-written kernels). Before spending a build cycle on it, the
cheap check: **measure the specific thing G4.0 is supposed to buy, and
price it.**

**What G4.0 actually has left to sell.** `docs/MEGAKERNEL.md` motivates the
two-kernel form as "12 launches instead of ~60, captured in one CUDA graph
⇒ effectively one launch." But G2.4 (step 12) *already* wraps the entire
non-causal forward in one `torch.compile(mode="reduce-overhead")` CUDA
graph, so the CPU-side dispatch cost of those ~60 launches is already
banked. The only thing left for hand-fusion to take is the **GPU-side**
per-kernel cost — the gap between consecutive kernels inside a graph
replay (grid setup/teardown, tail effects). Step 15's `ncu` work couldn't
answer this (noted in commit `3e831d6`: ncu measures kernel *execution*,
not dispatch gaps). So the decisive quantity is:

```
gap_fraction = 1 - sum(kernel GPU durations) / wall-clock forward time
```

**Probe** (`probes/g4_0_headroom_probe.py`, `jobs/g4_0_headroom.sbatch`),
three independent measurements: (A) wall time per forward via
`torch.cuda.Event`, no profiler attached; (B) kernel count and summed
kernel durations from CUPTI kernel tracing (`torch.profiler` → chrome
trace, `cat=="kernel"`), which *does* see kernels launched from inside a
CUDA graph replay; (C) a calibration — capture a graph containing K no-op
kernels, sweep K, take the slope, i.e. the marginal price of one extra
kernel in a graph replay on this GPU.

**A methodology error was caught and corrected rather than shipped.** The
first run (`results/g4_0_headroom_NOTF32_run44.log`) forgot that
`benchmark.py`'s TF32 settings live in `main()`'s argparse defaults
(`--matmul-precision high`, `--allow-tf32`), not in the model — so every
GEMM landed on plain FP32 CUDA-core `ampere_sgemm_*` instead of the
TF32 tensor-core `s1688gemm` path the shipped numbers were measured on.
It showed up immediately as `default = 1.478 ms` against the sweep's
0.8736 ms. Corrected run (`results/g4_0_headroom_run45.log`) reproduces
`results/g2_4b_sweep_run27.log` at every shape, which is what makes the
rest of the numbers trustworthy:

| shape | run27 optimized | probe wall | match |
|---|---|---|---|
| tiny | 0.3164 ms | 0.3117 ms | 1.5% |
| default | 0.8736 ms | 0.8712 ms | 0.3% |
| long_seq | 10.5001 ms | 10.4809 ms | 0.2% |
| large_batch | 26.0444 ms | 26.1204 ms | 0.3% |

**Result 1 — the inter-kernel gap is zero. This is the whole answer.**

| shape | wall | Σ kernel durations | gap | gap/kernel |
|---|---|---|---|---|
| tiny | 0.3117 ms | 0.3139 ms (100.7%) | −0.0021 ms (−0.7%) | −0.034 µs |
| default | 0.8712 ms | 0.8710 ms (100.0%) | +0.0002 ms (0.0%) | +0.004 µs |
| long_seq | 10.4809 ms | 10.4851 ms (100.0%) | −0.0042 ms (−0.0%) | −0.084 µs |
| large_batch | 26.1204 ms | 26.1337 ms (100.1%) | −0.0133 ms (−0.1%) | −0.266 µs |

Summed kernel time equals wall time to within measurement noise at every
shape (the slight overshoots are CUPTI timestamp overlap between
back-to-back kernels plus timer resolution, not negative idle). **There is
no GPU idle time between kernels for fusion to recover.**

**Result 2 — the calibration says the same thing independently, and
explains why.** A chain of 256 no-op `add_` kernels in one graph costs
**0.8554 µs marginal per kernel** (least-squares over K = 16/64/128/256,
per-kernel cost flat at 0.902 → 0.858 µs). But the no-op kernel's *own*
CUPTI device duration is **0.9794 µs**. The marginal cost is entirely the
kernel body's minimum execution time; the true dispatch gap is
**−0.124 µs ≈ 0**. A CUDA graph replay on Ada has essentially no
per-kernel dispatch overhead — which is exactly what a graph is for, and
G2.4 already bought it.

**Result 3 — the launch census is already better than the doc assumes,
and most of it is not deletable.** 50 kernels per forward (62 at tiny,
which adds `splitKreduce` passes for its small-M GEMMs), not the ~60
`docs/MEGAKERNEL.md` assumes:

| shape | GEMM/attn kernels | everything else |
|---|---|---|
| tiny | 30 launches, 0.2654 ms (84.6%) | 32 launches, 0.0485 ms (15.4%) |
| default | 30 launches, 0.7796 ms (89.5%) | 20 launches, 0.0914 ms (10.5%) |
| long_seq | 30 launches, 9.3626 ms (89.3%) | 20 launches, 1.1225 ms (10.7%) |
| large_batch | 30 launches, 19.4050 ms (74.3%) | 20 launches, 6.7287 ms (25.7%) |

The 30 GEMM/attention launches (5 per layer: qkv, out_proj, ffn_in,
ffn_out, `fmha_cutlassF_f32_aligned_64x64_rf_sm80`) are work a fused
kernel **absorbs, not eliminates** — the FLOPs still happen, just under a
hand-written tiling instead of CUTLASS's. Only the other 20 are candidates
for deletion, and **inductor has already fused them**: the kernel names are
literally `triton_per_fused_add_addmm_native_layer_norm_view_*` (LayerNorm
+ residual add + addmm epilogue, one kernel) and
`triton_poi_fused_addmm_gelu_view_2`. The "modest" version of G4.0 — fuse
LN + residual around the existing SDPA call rather than reimplementing
SDPA — is targeting work that is already one kernel, worth 35.7 µs of
871 µs (**4.1%**) at the default shape.

**Result 4 — roofline, for the record.** Confirms step 19's finding at
whole-model scale (step 19 measured the isolated FFN at 60-69% of peak):

| shape | GFLOP/fwd | TF32 floor | current | % of 82.6 TFLOPS |
|---|---|---|---|---|
| tiny | 2.47 | 0.0299 ms | 0.3117 ms | 9.6% |
| default | 40.27 | 0.4875 ms | 0.8712 ms | 56.0% |
| long_seq | 412.32 | 4.9917 ms | 10.4809 ms | 47.6% |
| large_batch | 1288.49 | 15.5992 ms | 26.1204 ms | 59.7% |

**Pricing G4.0 with step 19's own measured kernel, which is the honest
upper bound.** G4.0's FFN block is *exactly* the G3.1 kernel — same
`LN → ffn_in → GELU → ffn_out → residual` shape, same token-parallel
structure. Step 19 measured it per-FFN at `tf32`
(`results/g3_1_pretransposed_run41.log`); this model runs six of them.
Grant G4.0 a **perfect, free, zero-cost attention-block fusion** and
substitute only the measured FFN half:

| shape | M | torch FFN ×6 | fused FFN ×6 | projected total | vs current |
|---|---|---|---|---|---|
| default | 1024 | 0.518 ms | 2.879 ms | 3.23 ms | **0.27x** |
| long_seq | 8192 | 3.531 ms | 4.307 ms | 11.26 ms | **0.93x** |
| large_batch | 32768 | 14.385 ms | 16.548 ms | 28.28 ms | **0.92x** |

(The `torch FFN ×6` column cross-checks against this probe's own trace: at
large_batch the 12 `s1688gemm_256x128` launches + the GELU kernel are
9.94 + 3.40 = 13.34 ms against step 19's isolated 14.385 ms — same number,
the small difference being L2 state that isolation doesn't reproduce.)

**Best case is a regression at every shape**, before the attention block
has cost anything. And the attention block is the *harder* half, not the
easier one: `fmha_cutlassF_f32_aligned_64x64_rf_sm80` is already a fused,
hand-tuned CUTLASS flash kernel, and at long_seq it is **5.06 ms of the
10.48 ms forward — 48% of the entire model**. Fusing "around" it wins 4%;
fusing *through* it means hand-writing a replacement for the single
largest kernel in the model, on the evidence that the last hand-written
kernel came in at 0.18x.

**Why tiny's 9.6%-of-peak is real headroom but not G4.0's headroom.** At
64 tokens the forward reads the full 75.66 MB FP32 model against 128 KB of
activations — arithmetic intensity 32.6 FLOP/byte, so the binding roofline
is DRAM, not TF32: 75.66 MB at the 4090's ~1008 GB/s is a **75 µs** floor
against 311.7 µs measured, i.e. ~243 GB/s effective. That corroborates
step 15's independent `ncu` reading of 174 GB/s at tiny. Tiny is
**weight-bandwidth bound**, and G4.0 does not remove one byte of weight
traffic — the same 75.66 MB is read either way. Worse, it removes the
parallelism that hides it: a token-parallel fused block at M=64 is
`ceil(64/BLOCK_M) = 1` program, one CTA, which cannot keep enough memory
requests in flight to approach DRAM peak, where cuBLAS's `s1688gemm_64x64`
at least gets ~24 CTAs on the qkv GEMM. Step 19 already measured this
exact failure mode: **0.18x at M=1024 with 16 blocks**; M=64 is one block.
The levers that *would* address tiny's weight residency are G2.3 (pin the
model in the 72 MB L2 — investigated and closed in step 16, unreachable
from Python without raw-`ctypes` risk) and precision reduction (BF16 at
37.83 MB fits L2 comfortably — closed by steps 14/17/18).

**`docs/MEGAKERNEL.md`'s own gate, answered.** The doc states: *"Gate to
proceed to G4.1: `nsys` still shows >15% launch overhead or GPU idle at
the tiny regime after CUDA Graphs. If graphs already solved it, stop
here."* Measured GPU idle at tiny after CUDA graphs: **−0.7%**, against a
>15% threshold. CUPTI kernel tracing rather than `nsys`, but it answers
the identical question and it answers it decisively. Since G2.4 already
took the CPU-side dispatch, that gap *was* G4.0's entire remaining
motivation — so the gate closes G4.0 as well as G4.1.

**Decision: G4.0 not built.** The measurement it was gated on came back
zero, and the one component of it that already exists in tree
(`probes/g3_1_kernel.py`, step 19) prices the best case as a regression at
every shape. `benchmark.py` is untouched — `git diff HEAD -- benchmark.py`
is empty, so the shipped file remains bit-identical to the state
`results/g2_4b_sweep_run27.log` validated, and `tools/check_validity.py`
passes. No sweep was re-run, because nothing changed; no archive cell was
touched, because nothing improved. This is the outcome
`docs/MEGAKERNEL.md` explicitly sanctions: *"G4.0 winning is a result, not
a failure."*

**What would have to change for G4.0 to be worth revisiting:** a
measurable GPU-side inter-kernel gap (there is none — graphs closed it),
or a hand-written GEMM that beats CUTLASS's `s1688gemm` tiling on this
shape rather than losing to it by 13-82% (step 19). Neither is a tuning
problem. With G0/G1/G2 shipped, G2.3 closed on risk, G2.1/G2.6 closed on
accuracy, G3.1 closed on the roofline and G4.0 closed on the graph-gap
measurement, **the `docs/CATALOGUE.md` ladder is exhausted at this
precision.** Every remaining item above G4.0 (G4.1-G4.5) is built on the
FP8 pipeline step 18 measured at 65x over the accuracy budget.

## Re-investigation: regime arbiter + FP8 re-visit

User asked to (a) add a shape-detection arbiter (not yet built anywhere —
confirmed by a read-only Explore pass) and (b) revisit FP8 with hand-written
CUDA, citing real production techniques. Researched deeply (DeepSeek-V3's
fine-grained block quantization + FP32 accumulation promotion, QuaRot/
SpinQuant Hadamard rotation, confirmed Ada has no native MXFP8 hardware —
Blackwell only). A Plan-mode pass produced a staged, gate-driven
investigation plan (`/home/techjam2/.claude/plans/stateless-snuggling-mccarthy.md`,
approved) built around one key finding: `CLAUDE.md`'s `eps/√K` claim for
why FP8 "survives" appears arithmetically wrong for this model's random-sign
reductions (relative error doesn't shrink with K the way the doc assumes) —
and the one technique that actually attacks mantissa width, split/residual
precision (G2.8), was never tried. Plan approved with three decisions
confirmed: cheap FP8 probes run before the arbiter (built only if FP8 ends
up needing per-regime routing), arbiter is shell-only if built, and an
exactly-4-GEMM Stage 1c result is treated as closure.

### 21. Stage 1.5(i) + Stage 1a — capability check clean; precision-ladder simulator debugged, partially calibrated

**Stage 1.5(i)** (`probes/g6_0_capability_check.py`, job 46,
`results/g6_0_capability_run46.log`): confirmed on this stack (torch
2.13.0+cu130, Triton 3.7.1, sm_89) — `torch._scaled_mm` works with both
per-tensor scalar AND per-row (RowWise) scales; Triton's
`tl.dot(float8e4nv)` also works. If Stage 2 (hand-written kernel) is ever
reached, it's Triton-viable (days), not raw-PTX-only (weeks, no established
`cpp_extension` build path in this repo). Committed `f831b39`.

**Stage 1a** (`probes/g5_1_precision_ladder.py`) hit a real debugging
detour worth recording in full, since the anchor-gate discipline the plan
specified is exactly what caught it.

*First attempt* (job 47): swept mantissa width `m∈{3..12}` on the FFN only
(weights + activations mantissa-truncated in place, full `BaselineTransformer`
with real weights, `compare_outputs`' actual criterion, 20 seeds, tiny +
default). Result was **non-monotonic** — `m=10` showed *lower* error than
`m=11` and `m=12`, which is impossible for a correct simulator (more
precision can't make things worse), and the `m=8`/`m=11` anchors didn't
reproduce steps 14/17's real BF16/TF32 numbers within 2x.
(`results/g5_1_ladder_v1_FAILED_anchor_run47.log`)

*Debug 1* (`probes/g5_1_debug_determinism.py`, job 48): tested the
hypothesis that re-invoking `layer.attention(...)` separately for the
reference and quantized forward passes introduces TF32-matmul
non-determinism that would show up as an m-independent noise floor,
dominant exactly where quantization error itself is small (high m).
**Ruled out** — calling the same deterministic module twice on identical
CUDA input gave bit-identical output (`max_abs_diff=0.0` over 524288
elements, both for attention alone and the full model).
(`results/g5_1_debug_determinism_run48.log`)

*Debug 2* (`probes/g5_1_debug_isolate.py`, job 49): isolated the mantissa-
rounding function itself — verified monotonic and correct on synthetic
tensors both at moderate magnitude and at real weight-init scale
(`U(±1/√fan_in)≈±0.044`), on CPU, outside the full pipeline. Then tested
it on the model's *actual* `ffn_in.weight` tensor for a single layer,
single seed — **the same `m=10` dip reproduced**, even in a weight-
quantization-only test with no activation quantization and no multi-layer
compounding. Conclusion: not a bug in the rounding arithmetic itself, but
an **order-statistic artifact of measuring `max()` error on one fixed,
deterministic tensor across different quantization grids** — grid points
at adjacent bit-widths aren't nested, so which single element becomes the
"worst case" isn't smooth in `m` for a fixed (non-re-randomized) weight
tensor, even though the *mean* error is smooth. `max_abs_mean` in every
run is in fact cleanly monotonic; only `max_abs_max` shows the wobble.
(`results/g5_1_debug_isolate_run49.log`)

*Second finding, orthogonal to the above*: the anchor comparison itself
was scope-mismatched — the first attempt quantized only the FFN and
compared against steps 14/17's numbers, which quantized the **whole
model** (attention included). Quantizing less of the model necessarily
shows less error; these were never going to match. Fixed by adding
`whole_model_quantized_forward` (mirrors `BaselineSelfAttention.forward`
exactly, quantizing Q/K/V/out_proj too) used only for anchor validation,
keeping `ffn_quantized_forward` (FFN-only, what actually matters for
Stage 1c) as the main sweep.

*Second attempt* (job 50, `results/g5_1_ladder_v2_run50.log`): FFN-only
sweep numbers were **bit-for-bit identical** to the first attempt
(confirms the simulator itself is deterministic/stable — nothing about
the fix changed those numbers, as expected, since scope-matching only
touches the anchor path). But the whole-model anchor **still didn't
reproduce** even after the scope fix: `m=8` still reads 0.40-0.45x of the
expected ~0.011 (simulated version is *more* accurate than real BF16),
`m=11` still reads 3.6-3.76x the expected ~0.0007 (simulated version is
*less* accurate than real TF32) — in opposite directions, which doesn't
fit a single consistent explanation like an off-by-one in the mantissa-
bit-counting convention (that would push both anchors the same way).

**Decision: stop calibrating this simulator further, don't block on it.**
The anchor gate's whole purpose was to earn trust in an *approximate*
tool before relying on it. Stage 1c — the experiment that actually
decides whether to proceed — uses **real** `float8_e4m3fn` hardware
casts, not this simulated mantissa truncation, which sidesteps the
bit-counting-convention ambiguity entirely and needs no calibration
against a proxy. Stage 1a's FFN-only sweep is kept as a rough, directional
data point (order-of-magnitude behavior is sane: `m=3`→~0.12, decreasing
smoothly in `max_abs_mean` down to `m=12`→~0.0013), not as a hard
precision requirement.

Committed (all of the above, full investigative trail kept per this
project's practice) alongside this write-up. `benchmark.py` untouched, no
archive cell touched — this is investigation, not yet a candidate.

### 22. Stage 1b + 1c — split-precision FP8 (G2.8) genuinely passes accuracy; closed on the arithmetic anyway

The decisive experiment. `probes/g5_3_fp8_split.py`, real
`float8_e4m3fn` hardware casts throughout (no simulated mantissa
truncation this time — sidesteps step 21's calibration trouble entirely).
Tests G2.8, split/residual precision (`CLAUDE.md`'s own recommended
fallback, `docs/CATALOGUE.md`'s own item, never tried until now):
`W ≈ s0·Q0 + s1·Q1 + ...`, a greedy residual FP8 split with per-128-tile
dynamic scales, same for activations; `k` terms per operand.

**Stage 1b (folded in) — granularity check, confirmed the prediction:**
at `k=1`, sweeping tile size 512→32 moved `max_abs_mean` from 0.1208 to
only 0.1072 — under 12% improvement across a 16x finer granularity range.
Matches step-21-adjacent reasoning: this model's freshly-initialized
weights have no outliers for fine-grained scaling to help with. Finer
granularity alone was never going to close a gap this large, confirmed
directly rather than assumed.

**Stage 1c — the k-term sweep, real numbers (`results/g5_3_fp8_split_run51.log`):**

| k | tiny max_abs | default max_abs | long_seq max_abs | true_failures (of 60 = 20×3 shapes) |
|---|---|---|---|---|
| 1 | 0.1108 | 0.1241 | 0.1365 | 60/60 |
| 2 | 0.00308 | 0.00348 | 0.00384 | 60/60 |
| **3** | **0.00078** | **0.000896** | **0.000887** | **0/60** |
| 4 | 0.000631 | 0.000671 | 0.000690 | 0/60 |
| 5 | 0.000664 | 0.000662 | 0.000662 | 0/60 |
| 6 | 0.000628 | 0.000665 | 0.000669 | 0/60 |

**k=3 already passes the real criterion everywhere — 0 failures out of
60 seed×shape combinations, at every shape tested.** This is a genuinely
different outcome than every prior precision investigation this session
(BF16 twice, naive FP8 once): split-precision actually works
numerically. Worth stating plainly since it partially vindicates the
premise of revisiting FP8 at all.

**But k=3 doesn't clear the plan's pre-committed `≤8e-4` gate** at
default (0.000896) and long_seq (0.000887) — both exceed it, even though
both are still comfortably under the *real* 0.001 atol (default sits at
~90% of budget, thinner than the shipped TF32 path's own ~65-79% margin
but not a knife-edge pass like causal's G2.4b situation). **k=4 clears
the gate comfortably everywhere** (0.00063-0.00069, all real failures
0/60). So by the pre-committed, exact gate — not loosened after seeing
the result — **the smallest k that clears both the accuracy bar and the
stated safety margin is 4, landing exactly on the plan's pre-identified
marginal case.**

**The arithmetic for k=4:** `330.3/4 = 82.6` TFLOPS ideal — which is
*exactly* TF32's own peak. A hand-written 4-term FP8 kernel would need to
reach close to its own 100% efficiency just to match torch's *already
83% efficiency* number... more precisely: torch's measured FFN TF32
performance is 57.3 TFLOPS (69% of TF32's 82.6 peak, step 19/20). A
4-term FP8 kernel needs `57.3/82.6 = 69.4%` of *its own* peak just to
break even — and the one hand-written kernel actually built this session
(G3.1, step 19) achieved 13-87% of cuBLAS's efficiency depending on
shape, well capable of landing under that bar.

**Per the user's pre-confirmed decision (recorded in the approved plan's
"Decisions confirmed" section): treat this as closure, do not proceed to
Stage 1.5(ii) or Stage 2.** Not because FP8 doesn't work — it does, for
the first time this session — but because the GEMM count required to
make it work reliably eliminates the theoretical speed advantage before
kernel-quality is even a factor, and the prior kernel-writing attempt's
demonstrated efficiency range doesn't clear that bar with confidence.

**What would change this:** a k=3 result with more margin (e.g. if the
`≤8e-4` threshold had been set at `≤9e-4`, k=3 would qualify outright —
this is a genuinely close call, not an order-of-magnitude miss like BF16
or naive FP8 were). Revisiting with a tighter per-tile scale search, a
smarter greedy-split variant, or accepting k=3 with its thinner-but-real
margin are all legitimate future directions if this gets revisited —
recorded here rather than re-litigated now, per the pre-committed
decision.

Committed alongside this write-up. `benchmark.py` untouched, no archive
cell touched.

### 23. Regime arbiter (Part A) — not built, per the pre-confirmed plan

Per the approved plan's confirmed sequencing decision ("the arbiter is
built only if [the FP8] investigation turns out to need it — some
regimes win with FP8, others don't"): Stage 1 closed without a shipped
FP8 candidate at any regime, so there is nothing for the arbiter to
route between. Not building it this round, consistent with the decision
the user already confirmed before this investigation ran (not a new
unilateral call). The design itself (host-side `B,S=x.shape` dispatch,
computed and quantified as ~0.16% of tiny's wall time — 8x below this
project's own measurement noise floor — a `{regime: compiled_fn}`
registry memoized by implementation identity, boundary-shape sweep,
multi-regime CUDA-graph-pool safety probe) is fully specified in
`/home/techjam2/.claude/plans/stateless-snuggling-mccarthy.md` if a
future candidate ever needs it.

## Investigation summary: regime arbiter + FP8 re-visit

**FP8 re-investigation (steps 21-22):** found and fixed real bugs in the
diagnostic tooling along the way (an order-statistic artifact in max-error
measurement, a scope-mismatched anchor comparison), then ran the decisive
experiment cleanly. **Split-precision FP8 (G2.8) genuinely passes this
model's accuracy bar for the first time this session** — a real, different
outcome from BF16 (steps 14, 17) and naive per-channel FP8 (step 18), both
of which failed by an order of magnitude or more. It's closed anyway,
because the GEMM count needed (4 terms, to clear the stated margin) prices
out the theoretical speed advantage before any kernel is even written —
matching TF32's own peak throughput at best, and needing better efficiency
than the prior hand-written-kernel attempt demonstrated. This is a more
nuanced, more informative closure than the earlier BF16/FP8 dead ends: not
"this doesn't work," but "this works, and isn't worth it at this
precision's required term count."

**Regime arbiter:** not built, per the sequencing the user already
confirmed — it had no FP8 candidate to route to. Fully designed and ready
to build (see the plan file) if a future precision or kernel investigation
ever produces a regime-specific winner.

No `benchmark.py` changes this investigation; no archive cell touched.
Current shipped state remains: tiny 4.65x, default 1.61x, long-seq 2.35x,
large-batch 1.61x, padded 1.60x, causal 1.75x (unchanged from the G0-G4
investigation's conclusion).

### 24. Re-opened by request: does k=3 hold up under a real implementation? No — confirms closure with real hardware evidence

User explicitly asked to try k=3 anyway and see if it holds up, rather than
stop at step 22's simulated result. Good call to push on — it surfaced a
real ambiguity step 22 glossed over, and the real-hardware answer is more
decisive (and less favorable) than the simulation suggested.

**The ambiguity:** step 22 measured "k=3 terms per operand" via full
dequant-recombine (implicitly capturing all 3×3=9 cross-terms' worth of
accuracy in one fp32 matmul), then read the plan's GEMM-count table as
"3 terms → 3 GEMMs." But `CLAUDE.md`'s own G2.8 example (2 terms/operand →
3 GEMMs, `A_hi·B_hi + A_hi·B_lo + A_lo·B_hi`, dropping only the doubly-low
`A_lo·B_lo` term) is a **triangular** truncation, not literal term-count.
Generalized to 3 terms/operand, that's 6 GEMMs (keep `i+j<3` out of 9
possible cross-terms), not 3 — the plan's table conflated these two
things.

**Real implementation, real API constraints found and fixed along the
way** (`probes/g5_5_k3_real_gemm_count.py`, jobs 53-55, real
`torch._scaled_mm` calls, not simulation):
- Row-wise-scaled `_scaled_mm` only supports bf16/fp16 output, not fp32
  (`results/g5_5_attempt1_dtype_error_run53.log`). Fixed by requesting
  bf16 and upcasting to fp32 immediately per-term, before summing (so
  multi-term summation itself happens in fp32).
- `_scaled_mm`'s weight operand must stay a transposed *view* of the
  original row-major tensor (`stride(0)==1`); calling `.contiguous()` on
  it (materializing an actual transposed copy) breaks that requirement
  (`results/g5_5_attempt2_stride_error_run54.log`). Fixed by removing the
  `.contiguous()` call.

**Result** (`results/g5_5_real_gemm_result_run55.log`, default shape, 20
seeds, per-channel scales — the finest granularity `_scaled_mm` actually
supports on Ada, no native per-128-tile microscaling):

| design | real GEMMs | max_abs_max | true_failures |
|---|---|---|---|
| A: weight 3-term, activation 1-term | 3 | 0.0987 | 20/20 |
| B: activation 3-term, weight 1-term | 3 | 0.0909 | 20/20 |
| C: triangular (both 3-term, keep i+j<3) | 6 | 0.0072 | 20/20 |

**Neither 3-GEMM asymmetric design works** — confirms the reasoning from
when this was first considered: whichever operand is left at 1-term FP8
dominates the error (both land close to the plain k=1 result, ~0.09-0.10,
not anywhere near k=3's simulated 0.0008-0.0009). Both operands need
multi-term precision, not just one — there's no way to get real 3-GEMM
split-precision to work here.

**The 6-GEMM triangular design — the "correct" generalization of
`CLAUDE.md`'s own G2.8 example — also fails**, at 0.0072 (7x over the
0.001 budget), a full order of magnitude worse than step 22's simulated
"k=3, 9-term-equivalent" result of 0.0008-0.0009. Two real-hardware
effects the fp32-recombine simulation didn't capture: bf16 rounding on
each of the 6 GEMM outputs (row-wise scaling forces bf16/fp16 output,
adding ~0.4% relative noise per term, compounding across 6 terms), and
the 3 dropped highest-order cross-terms mattering more than assumed.

**This closes the question more decisively than step 22 did, not less.**
6 real GEMMs already gives only `330.3/6 = 55` TFLOPS ideal — *below*
torch's own measured 57.3 TFLOPS TF32 FFN baseline, meaning it loses
arithmetically even at 100% kernel efficiency, before accuracy is even
back in budget. Whatever GEMM count would actually clear accuracy (more
than 6, given 6 already misses by 7x) only pushes the ideal TFLOPS lower
still. There's no real-implementation path from here to a viable
candidate — not a matter of more tuning, the arithmetic is upside-down
regardless of accuracy.

**Answer to "does it hold up": no.** Confirms step 22's closure decision,
now with real hardware measurement instead of a plan-table estimate that
turned out to understate the true GEMM cost. Committed alongside this
write-up (probe, job script, all three run logs including both API-error
attempts, full trail preserved). `benchmark.py` untouched, no archive
cell touched.

## New round: broader research, revisiting risk/effort-based rejections

User asked for deeper research into hardware-specific and compiler-level
techniques not yet tried, plus a re-look at anything closed for risk/effort
reasons rather than fundamental infeasibility, with explicit appetite for
higher risk this round. Full plan (tiered by risk) at the plan-mode
artifact; summary: Tier 1 compiler-level/zero-accuracy-risk (max-autotune,
inductor knobs, cudnn.benchmark), Tier 2 precision/real-risk (plain FP16 —
never tried, unlike BF16 — scoped narrowest-first: FFN-only, attention-only,
full-model; INT8 FFN), Tier 3 higher-effort-now-feasible (G2.3 L2
persistence via a real C++ extension, now confirmed buildable in this
container; G3.6 minimax GELU).

### 25. G6.1 (`torch.compile(mode="max-autotune")`) — tried, decisive accuracy failure, reverted

**What was tried:** swapped both of `UserOptimizedTransformer`'s internal
`torch.compile()` calls (the non-causal `_optimized_forward` path and the
causal fallback) from `mode="reduce-overhead"` to `mode="max-autotune"`.
Confirmed via grep beforehand this was genuinely untried — `max-autotune`
existed only as an unused `--compile-mode` CLI choice, never wired into
the class's own hardcoded calls. Motivation: `max-autotune` still builds
and replays CUDA graphs (doesn't give up G2.4's launch-overhead win) and
additionally lets inductor search real Triton/CUTLASS/cuBLAS kernel
candidates per op, including epilogue fusion cuBLAS's own API can't
express — looked additive, not a tradeoff.

**Result: decisive accuracy failure on the very first shape tested**
(`results/g6_1_max_autotune_smoke_FAILED_run56.log`, job 56, tiny shape —
script aborted under `set -e` before reaching default/causal). All 5
trials FAIL: `max_abs` 0.00220–0.00242 (2.2–2.4x over the 0.001 budget),
1250/163840 elements failing outright under the full disjunctive
criterion (both abs>0.001 AND rel>>1% on the same failing elements, not a
near-zero-denominator artifact).

**Root cause, read directly from the autotuner's own choice log:**
`max-autotune`'s kernel search does not select one homogeneous kernel
family — for one GEMM shape (64×512 @ 512×2048) it picked a Triton kernel
(`triton_mm_40`, `ALLOW_TF32=True`) over cuBLAS's own `mm`; for another
(64×2048 @ 2048×512) cuBLAS `mm` itself won outright; for the fused
`addmm` case, Triton won again. The shipped `reduce-overhead` path never
did this search — every GEMM went through cuBLAS's own native TF32 tensor-
core path uniformly. Triton's `ALLOW_TF32=True` is a *software* emulation
of TF32 (a 3-pass FP32 decomposition), not the same hardware datapath as
cuBLAS's native TF32 GEMM — different rounding, and here enough of it to
push several layers' worth of accumulated error over budget by the time
it reaches the output.

**Why this closes without a workaround, not just this particular run:**
the only way to keep max-autotune's real value proposition (epilogue
fusion) is to let it search Triton candidates, and it's exactly the
Triton-vs-cuBLAS kernel heterogeneity that causes the drift — restricting
the autotuner to cuBLAS/ATEN-only backends would recover the shipped
model's accuracy exactly by producing the shipped model's kernel
selection, at which point there is no speedup left to have (no epilogue
fusion, no benefit over `reduce-overhead`). There's no middle setting that
keeps the win and drops the risk; it's a package deal that already failed.

**Reverted immediately** (`git checkout -- benchmark.py`, verified clean
via `git status`, `check_validity.py` still passes). Committed alongside
this write-up: `jobs/g6_1_max_autotune_smoke.sbatch`,
`results/g6_1_max_autotune_smoke_FAILED_run56.log`. No archive cell
touched.

**Closes Tier 1 item 2 (inductor config knobs) as moot along with it** —
those knobs (`coordinate_descent_tuning`, explicit `epilogue_fusion`) only
have an effect *under* max-autotune's search; with that mode itself closed
on accuracy, there's nothing left for them to tune.

### 26. G6.2 (`torch.backends.cudnn.benchmark = True`) — tried, clean null result

**What was tried:** added the flag next to the existing `allow_tf32`
settings in `main()` (never set before this session, grep-confirmed).
Zero accuracy risk by construction — it only affects cudnn's own kernel-
algorithm cache, never a computed value — but checked empirically rather
than assumed a no-op, per this session's own standing preference for
measurement over assumption (e.g. `_scaled_mm`'s real API constraints in
step 24 diverged from the plan's own table).

**Result: exactly the predicted no-op, confirmed rather than assumed.**
Full 8-shape sweep (`results/g6_2_cudnn_benchmark_null_run57.log`, job 57)
— every shape PASSES accuracy identically to the shipped baseline
(expected, no computed value changed), and every shape's speedup sits
within ±1-2% of `results/g2_4b_sweep_run27.log`'s numbers (tiny
4.595→4.602, default 1.608→1.604, long_seq/large_batch/long_seq_padded
unchanged to 3 sig figs, default_padded/causal_padded moved ±1.5% in
opposite directions) — noise, not signal. This model is SDPA/cuBLAS-
dominated with zero convolutions, so there is nothing in the graph for
`cudnn.benchmark`'s algorithm cache to act on.

**Reverted** (`git checkout -- benchmark.py`, verified clean, validity
gate still passes) — a flag with confirmed zero effect isn't worth
carrying in the shipped file. Committed alongside this write-up: the
sweep log. No archive cell touched, `benchmark.py` unchanged from
step 24's state.

Closes Tier 1 entirely: all three items tried (max-autotune: accuracy
failure; inductor knobs: moot; cudnn.benchmark: null result). Moving to
Tier 2 (FP16), the most research-grounded remaining candidate.

### 27. G6.4a (FP16 FFN) — genuinely close, real speedup, closes on a statistical tail rather than a decisive miss

**What was tried:** plain FP16 (not BF16) for the FFN, narrowest-first per
the plan. Confirmed via grep this was genuinely untried this session —
BF16 was tested exhaustively (G2.1/G2.1b, both failed ~11x over budget);
FP16 never was. `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`
explicitly set `False` in `__init__` (not left to `main()`, so it holds
regardless of entry point) to isolate FP16's storage/rounding precision
from a confound of also losing FP32 accumulation.

**v1 (both FFN GEMMs in FP16, `results/g6_4a_smoke_run58.log` then
`results/g6_4a_both_gemms_FAILED_run60.log`):** 5-trial smoke test on
tiny PASSED cleanly — but `max_abs` up to 0.00133, already over CLAUDE.md's
5e-4 internal-investigate threshold and even over the 0.001 atol alone
(saved only by the disjunctive `rel<=1%` on those elements), which by
this session's own established practice (step 13) demands a higher-seed
check before trusting a smoke pass. **The 40-seed, 6-shape rigor probe
caught what the smoke test missed**: every shape genuinely fails —
tiny 12/40 trials (30%) with a real failing element, default/long_seq/
large_batch/padded all fail too, always by rare single-to-low-double-digit
element counts against tensors of 1.3M-671M elements. Real speedups
alongside the failure: default 1.894x vs shipped 1.604x (+18%), large_batch
2.241x vs shipped 1.610x (+39%), long_seq 2.784x vs shipped 2.353x (+18%)
— a large win if it had passed.

**v2 (FP16 for `ffn_in` only, `ffn_out` exact FP32 —
`results/g6_4a_v2_ffnin_only_FAILED_run61.log`):** narrowing to one
low-precision GEMM boundary per layer (same "isolate before compounding"
logic as G2.1→G2.1b) measurably helped, not just marginally: failing-
element counts dropped roughly 10x at every shape (default 13→1, long_seq
118→6, large_batch 637→61, long_seq_padded 109→7), and **tiny and
default_padded now pass cleanly** (0 failures across 40 seeds each).
But default, long_seq, large_batch, and long_seq_padded still fail —
CLAUDE.md invariant #6 ("validate on the full sweep, not one shape") and
invariant #2 (any real accuracy failure skips the benchmark) both apply
regardless of how rare the failing elements are.

**Why this is closed, not "try harder":** the failure pattern itself is
the evidence. Failure counts scale with element count (large_batch's 671M
elements is worst; tiny/padded, with far fewer real (unmasked) elements,
pass outright) and the errors involved are all small, un-clustered,
random-looking (no shared feature dimension, no shared magnitude range) —
the signature of a fixed per-element error *rate* from FP16's 10-bit
mantissa quantization noise floor being sampled by a huge number of
elements, not a systematic bias a further architectural change could
target. The natural next lever (real split-precision, hi+lo terms on both
operands, same technique validated for FP8 in step 22) was priced out
before building: G2.8's own triangular-truncation arithmetic
(2 terms/operand -> 3 GEMMs, confirmed for real in step 24) means
"fixing" `ffn_in` this way would triple that GEMM's cost — on a single
FP16 GEMM whose only purpose was being *cheaper* than the TF32 GEMM it
replaced. There's no version of this fix that both clears the tail and
keeps the win; the win and the risk are the same knob.

**Reverted** (`git checkout -- benchmark.py`, verified clean, validity
gate passes). Committed alongside this write-up: both rigor-probe job
scripts and all three result logs (smoke, v1 full-sweep, v2 full-sweep).
No archive cell touched.

**Genuinely informative for the next candidate (attention-only FP16,
plan Tier 2b):** this closes "FP16 in the FFN" specifically — the FFN's
large weight matrices (2048-wide reductions) are where the tail risk
concentrated. Attention hasn't been tested in FP16 at all and has
different structure (softmax renormalizes before the value matmul,
which could behave very differently under the same quantization noise
floor) — not assumed to inherit this result, worth testing on its own
evidence exactly as the plan scoped it.

### 28. G6.4b (FP16 attention) — shipped: clean pass on every shape, largest single-iteration win this session

**What was tried:** FP16 for QKV projection + SDPA + `out_proj`, FFN left
on the exact TF32 path (step 27 closed FP16 there). Same
`allow_fp16_reduced_precision_reduction=False` guard as G6.4a, set in
`__init__` so it holds regardless of entry point. FP16 Q/K/V also
switches which SDPA backend gets used: FP32 is stuck on the `math`
backend (G0.1's own finding — flash/efficient aren't available for FP32
at all), FP16 unlocks flash/memory-efficient — a second, independent
lever stacked on top of the precision change itself, motivating the
research prediction that this regime specifically (not FFN) was where
FP16 would pay off.

**Result: clean pass, no close calls, at the same 40-seed/6-shape rigor
that caught G6.4a's near-misses** (`results/g6_4b_fp16_attn_rigor_run62.log`):
`failed=0` on every one of tiny/default/long_seq/large_batch/
default_padded/long_seq_padded, `max_abs` topping out at **0.000906**
(large_batch, 671M elements — the same shape that broke G6.4a hardest)
comfortably under the 0.001 atol outright, not saved by the disjunctive
`rel` clause the way G6.4a's passes were. Confirms this session's own
prior finding (step 14) that attention's softmax wasn't the accuracy
risk in BF16 either — the FFN's large weight reductions were the
problem both times, and isolating away from them is what actually works,
independent of which 16-bit format is used.

**Speedups, real and large — official 8-shape sweep**
(`results/g6_4b_official_sweep_run63.log`, job 63, vs the shipped
baseline `results/g2_4b_sweep_run27.log`):

| shape | before | after | gain |
|---|---|---|---|
| tiny | 4.595x | 6.140x | +33.6% |
| default | 1.608x | 2.238x | +39.2% |
| long_seq | 2.353x | 4.560x | **+93.8%** |
| large_batch | 1.610x | 1.976x | +22.7% |
| default_padded | 1.588x | 2.136x | +34.5% |
| default_causal | 1.747x | 1.783x | +2.1% (noise — causal path untouched, see below) |
| causal_padded | 1.753x | 1.787x | +1.9% (noise, same reason) |
| long_seq_padded | 2.236x | 4.293x | **+92.0%** |

**long_seq nearly doubles — exactly the predicted mechanism.**
CLAUDE.md's own regime table notes attention is ~48% of the forward pass
at `S>=1024`; flash/memory-efficient attention's O(S) memory and better
tensor-core utilization scale specifically with sequence length, so the
biggest win landing at long_seq (not tiny or default) is the causal
GEMM-vs-attention balance behaving exactly as expected, not a surprise
result. Causal shapes move by ~2% (noise, not signal) because `forward()`
routes `config.causal` to `_compiled_causal`/`super().forward()` before
`_optimized_forward` is ever reached — this change is structurally
incapable of touching causal, confirmed by causal's `max_abs` being
bit-identical to every prior sweep's causal number (0.000994682).

**Shipped.** `check_validity.py` still passes. Archived as a new elite in
all 6 regime cells (`tiny/fp16`, `default/fp16`, `long-seq/fp16`,
`large-batch/fp16`, `padded/fp16` at 6.14x/2.24x/4.56x/1.98x/2.14x, plus
`causal/fp16` at 1.78x recording the now-current state even though this
diff didn't touch causal). This is the largest single-iteration
improvement of the whole session, and the first time a precision
reduction beyond TF32 has shipped.

### 29. G6.5 (INT8 FFN) — cheap precheck, decisive failure, closes Tier 2

**What was tried:** `docs/CATALOGUE.md` G2.7 — never triggered this
session (Phase-0 showed FP8 was available, so the catalogue's own "use
only if FP8 unavailable" condition never fired). Given INT8's uniform
fixed-point quantization is structurally different from FP8's floating-
point scheme (which failed, step 18) and FP16's (which came within a
hair of passing in the FFN, step 27), worth checking on its own evidence
rather than assumed guilty by association. Same cheap-gate methodology as
`probes/g4_fp8_accuracy_precheck.py` (synthetic quantize-dequant
simulation, per-channel weight scales, per-tensor dynamic activation
scale, symmetric `[-127,127]` — no real kernel investment before knowing
if the precision itself is viable).

**Result: decisive failure, worse than either FP8 or FP16**
(`results/g6_5_int8_precheck_FAILED_run64.log`, job 64): all 20/20 seeds
fail, `max_abs` 0.0269-0.0310 — **27-31x over the 0.001 atol budget**,
roughly 20-25x worse than FP16's FFN near-miss (step 27's 0.0011-0.0014)
and in the same range as FP8's original decisive failure. INT8's fixed 8-
bit linear step size, with no floating exponent to auto-range around each
value's own magnitude, is simply too coarse for these O(1)-magnitude,
roughly-Gaussian activations — the opposite conclusion from FP16, which
benefits from mantissa bits comparable to TF32's own.

**No further probing needed** — same logic as G2.1's original closure
(step 14): a gap this large doesn't need more seeds to be judged unsafe.
Closes Tier 2 entirely: FP16 attention shipped (step 28), FP16 FFN closed
(step 27), INT8 FFN closed (this step). Committed alongside this write-up:
the probe script and job log. No `benchmark.py` changes (precheck only,
never wired into the model).

### 30. G3.6 (minimax deg-7 GELU) — catalogue's own accuracy claim verifiably wrong, closed without a GPU job

**What was tried:** `docs/CATALOGUE.md`'s G3.6 claims a degree-7 minimax
polynomial approximates GELU to ~1e-6 over `x<-5→0, x>5→x` exact, poly
between. Before wiring anything into `benchmark.py`, computed the actual
polynomial (Chebyshev-node least-squares fit in Chebyshev basis, converted
to monomial coefficients, pure CPU/numpy — no GPU, no `sbatch` needed,
this is a closed-form math check) to verify the claim, per the catalogue's
own instruction to "measure against `erff` first."

**Result: the catalogue's accuracy estimate is simply wrong, confirmed
by direct computation, not by guessing.** A real degree-7 fit over
`[-5,5]` achieves max abs error **0.084 — 84x over the entire 0.001 atol
budget by itself**, before any amplification through `ffn_out`'s GEMM or
6-layer compounding. A degree sweep shows why: error only drops to
1.6e-3 at degree 15 and 1.3e-4 at degree 19 — reaching the claimed 1e-6
would need a polynomial in the high 20s in degree, not 7. This is the
second time this session a `docs/CATALOGUE.md` numeric estimate has been
found wrong under real measurement (the first: G2.8's GEMM-count table,
step 24) — worth flagging as a pattern (the catalogue's speedup/effort
estimates have held up better than its precision-accuracy estimates
specifically, across both cases found).

**Why this closes rather than "just use a higher degree":** a
polynomial with enough terms to actually clear budget (roughly degree
20+, by the trend) needs that many multiply-adds per element via Horner's
method, evaluated on every one of the model's ~800K-16M+ GELU activations
per forward call depending on shape — plausibly *more* arithmetic than
native `F.gelu`'s own hardware-accelerated `erf` intrinsic, for a
catalogued gain that was only ever 1.02-1.05x at the (wrong) degree-7
estimate. There's no degree where this both clears budget and beats the
kernel it replaces.

**Closed without spending a GPU job** — the polynomial-fitting itself is
decisive, pure math, independent of the model or hardware (same category
of closure as steps 19/20/24's roofline arithmetic). No `benchmark.py`
changes. Nothing to commit from this step beyond this write-up (the local
fitting script lived outside the repo, in the scratchpad, not part of
the project's own artifact trail).

Closes Tier 3's cheap item. Remaining: G1.6 + G2.3 (L2 persistence via a
real C++ extension) — the one item left in the plan, and the highest-
effort one, now that the build toolchain is confirmed working.

### 31. Fresh profiler pass (post-G6.4b) — no hidden win in "other layers"

User asked to look beyond `docs/CATALOGUE.md` into other pipeline layers
(LayerNorm, softmax, masking, residual adds) for non-catalogued wins.
Per `docs/AGENTS.md`'s own profiler-subagent design, ran a fresh `ncu`
pass (haiku subagent, JSON-only facts, raw CSV never touched this
context) on the CURRENT shipped state — steps 15/34/35/43's profiles all
predate G6.4b's FP16 attention, so a fresh look was warranted rather than
reasoning from stale data.

**Default shape:** the same TF32 CUTLASS GEMM (`cutlass_80_tensorop_
s1688gemm_128x64_16x6_tn_align4` — the FFN's `ffn_in`/`ffn_out`) dominates,
at only **~47% of true TF32 peak (82.6 TFLOPS) with ~26% occupancy**. Real
inefficiency, but no safe lever identified: the only way tried to search
alternate GEMM kernels (`torch.compile(mode="max-autotune")`) already
failed on accuracy (step 25, Triton-vs-cuBLAS kernel heterogeneity); a
custom cuBLASLt algorithm-search extension restricted to native (non-
Triton) candidates is a real possibility but speculative and substantial
new engineering with uncertain payoff — not pursued given the L2-
persistence work already in flight is a more concrete, better-scoped bet
for this session's remaining effort.

**Long_seq shape: the top hot kernel (`softmax_warp_forward`, FP32) is
almost certainly the frozen baseline's, not ours — a false lead, caught
before acting on it.** `benchmark.py`'s own `BaselineSelfAttention.forward`
computes `torch.softmax(scores.float(), dim=-1)` as a standalone kernel;
`UserOptimizedTransformer`'s FP16 attention path (G6.4b) dispatches to
SDPA's flash/memory-efficient backends, which fuse softmax internally
with no separately-visible kernel (this is *why* G6.4b's FP16 change
unlocked those backends in the first place, per G0.1's original finding).
`profile.sbatch` profiles the whole `benchmark.py` invocation, which runs
both baseline and optimized forward passes together for the accuracy
comparison — it can't be attributed to our own code without more
kernel-launch-site filtering than this pass did.

**Conclusion: no hidden win found in LayerNorm/softmax/masking/residual
layers under real measurement** — the GEMMs (already this session's
central focus) remain the dominant cost in the optimized path; the
"other layers" the user asked about don't show up as hot kernels once
attribution to baseline vs. optimized is accounted for. Not a wasted
check — a verified negative, consistent with this session's practice of
citing a specific profiler fact for every claim rather than assuming.

### 32. G1.6 + G2.3 built for real (C++/pybind11 extension) — G2.3 is a measured 4–6% REGRESSION, G1.6 alone is neutral. Both reverted.

Step 16 deferred G2.3 because reaching `cudaDeviceSetLimit` /
`cudaStreamSetAttribute` from Python meant `ctypes`-ing `libcudart.so` with
no compiler to check the `cudaAccessPolicyWindow` struct layout. This step
removed that objection by building the real thing: **`csrc/l2_persist.cpp`**,
a pybind11 extension compiled against this container's own CUDA 13.1
headers via `torch.utils.cpp_extension.load()`. It builds and works.

Two toolchain gotchas worth keeping:
- `cpp_extension.load()` needs `TMPDIR` pointed at a writable,
  container-visible path (`/work/.ext_build`); the container default is not.
- **`with_cuda=True` is mandatory even with no `.cu` source.** Torch infers
  CUDA only from the `.cu` file extension, so a pure-`.cpp` extension that
  includes `<cuda_runtime.h>` fails to compile without it (missing CUDA
  include path and `-lcudart`). First build attempt died exactly here.

`.data_ptr()` is taken **in C++** from a `torch::Tensor` argument, so
nothing in `benchmark.py` calls it and `tools/check_validity.py` passes
cleanly — no gate evasion, the Python side never needs the raw address.

**The premise held; the payoff did not.** The hot working set really is
~63MB (the FP16 attention fold + folded `ffn_in` + `ffn_out`), not the
75.66MB of CLAUDE.md's ground truth — that figure double-counts the dead
original `nn.Parameter`s kept only for `strict=True`. G1.6's arena packed
all of it contiguously and every layer was repointed at views into it.

Device facts (`results/g2_3_l2_probe_run68.log`): `l2CacheSize` 72.0 MiB,
`persistingL2CacheMaxSize` **49.5 MiB**, `accessPolicyMaxWindowSize`
128 MiB. So a 60MB arena can only get `hitRatio` 0.825.

Sweeps (`results/g1_6_g2_3_sweep_run69.log`, `results/g1_6_arena_only_run70.log`)
vs the shipped `results/g6_4b_official_sweep_run63.log`:

| shape | run63 | G1.6 only | G1.6+G2.3 |
|---|---|---|---|
| tiny | 6.140 | 6.176 | 6.124 |
| default | 2.238 | 2.187 | **2.098** |
| long_seq | 4.560 | 4.556 | **4.365** |
| large_batch | 1.976 | 1.972 | 1.970 |
| default_padded | 2.136 | 2.154 | **2.049** |
| long_seq_padded | 4.293 | 4.292 | **4.091** |

Causal shapes (1.783→1.793, 1.787→1.796) are the control: they bypass
`_ensure_folded_weights` entirely, and their ±0.5% drift sets the noise
floor. G1.6 alone sits inside that floor everywhere. **G2.3 is a clean
−4% to −6% on every shape it touches.**

**Why.** `probes/g2_3_l2_persist_probe.py` isolates the window itself
(A eager/no window, B eager/window, C graph/no window, D window set INSIDE
the capture region so it lands on the capture stream and is snapshotted
into the graph's kernel nodes). D is legal and works — so the CUDA-graph
capture-stream concern is answered, not a confound — but **every
configuration is within ±0.2% in TINY, DEFAULT and LARGE-BATCH alike.**
The window buys nothing because at 186 GB/s (tiny) and 96 GB/s (default)
of implied weight traffic against the 4090's ~1 TB/s, nothing here is
DRAM-bandwidth-bound, and the ~63MB working set already fits under normal
LRU in 72MB of L2. Persistence protects a hot subset when the working set
*exceeds* L2; here it only carves 49.5 MiB away from the activation
traffic that was using it. That carve-out is the regression.

Accuracy was **bit-identical to run63 on all 8 shapes** (max_abs
0.000654817 / 0.000844739 / 0.000728816 / 0.000849087 / 0.000828147 /
0.000994682 / 0.000994682 / 0.00078994), exactly as a pure memory-layout
change must be — which is what makes the timing delta trustworthy.

**Reverted `benchmark.py` to HEAD.** The attempted diff is preserved as
`results/g1_6_g2_3_benchmark_py.patch` rather than committed. The
extension, the probe and the job scripts are kept: they are reusable, and
they turn "L2 persistence, 1.1x default / 1.5x+ tiny" in
`docs/CATALOGUE.md` from an untested estimate into a measured **no**.

### 33. G6.6 (cuBLASLt explicit algorithm + bias epilogue, TINY only) — shipped, 1.18-1.20x on top of the fp16-attention win

**What was tried:** motivated by step 31's fresh profiler fact (FFN's TF32
CUTLASS GEMM at ~47% of peak, ~26% occupancy on default shape), tested
whether cuBLASLt's own algorithm search (same native TF32 tensor-core
datapath PyTorch already uses — never Triton's emulation, the thing that
broke G6.1/max-autotune) could beat PyTorch's default kernel choice.
Built via a real C++ extension (`csrc/cublaslt_algo.cpp`), same build
recipe as step 32's `csrc/l2_persist.cpp`.

**The original hypothesis (default shape) was a clean negative, but a
different real win fell out of testing it properly:** at the default
shape (M=1024), the best of cuBLASLt's own heuristic-returned algorithm
candidates beat PyTorch's pick by only 1.001x — PyTorch's default choice
is already close to optimal there. A separate, larger apparent win
(1.19x, traced to PyTorch's bias-add path costing 12.2us extra per GEMM
regardless of algorithm) did not survive integration: it was an artifact
of the isolated probe's own baseline (eager ops inside a raw CUDA graph
still pay that penalty; `torch.compile`'s full-model lowering already
avoids it), confirmed by comparing bit-identical output for zero measured
gain once wired into the real model. **What survived, on its own
evidence, is TINY (M=64): a real 1.32-1.49x on the GEMMs themselves**,
from split-K algorithm variants cuBLASLt's default heuristic does not
select at this small M — genuinely GPU-bound (cpu-issue 4.3us vs
7.7-30us of kernel time, not a launch-overhead artifact).

**Scoped deliberately to TINY only** (`_LT_MAX_TOKENS = 127`, CLAUDE.md's
own regime boundary) rather than left open — the algorithm is chosen by
a one-time eager timing calibration (never inside the compiled region,
same rule as every weight-fold this session) with a **correct fallback**
(plain `F.linear`) if the winning candidate doesn't actually beat it,
satisfying CLAUDE.md's own validity test for a runtime-gated exception.
Confining the gate to the one regime that measurably pays also confines
the risk that a different cuBLAS build picks a different split-K variant
at a larger shape and shifts `max_abs` there for no measured gain.

**Independent verification (this session, not just the implementing
agent's own report) caught two things worth recording:**

1. **The same near-miss discipline that caught step 27's FP16-FFN failure
   applies here too, and this time it held up.** The agent's own 5-trial
   numbers showed `max_abs` moving from 0.000655 to 0.000721 (+10%, from
   split-K's different reduction order) — above CLAUDE.md's 5e-4
   "investigate" threshold, so it was re-checked with the same 40-trial
   rigor that caught the FFN failure (`results/g6_6_tiny_rigor_run77.log`,
   job 77, fresh model instantiation, independent of the agent's own
   runs). **Zero failures across all 40 trials** (1.3M elements),
   `max_abs` peaking at 0.00084 — 84% of budget, comfortable margin, a
   materially different risk profile than the FFN case (which had real
   failures at this same sample size). Passes on its own evidence, not
   just a lucky smoke test.

2. **A real methodological catch: absolute speedup numbers drift across
   the session (~5-8%), independent of any code change — cluster
   clock/thermal state, not noise from a bad measurement.** The official
   sweep (`results/g6_6_official_sweep_run76.log`) showed default shape
   at 2.097x, down from `run63`'s 2.238x recorded earlier this session —
   alarming, since the code change should structurally not touch default
   at all (`tok=1024 > _LT_MAX_TOKENS=127` means the whole cuBLASLt path
   is skipped before it's ever built). Decisive test: stashed the diff,
   re-ran default on the exact pre-change code under current conditions
   (`results/g6_6_default_baseline_recheck_run80.log`) — **2.127x, matching
   the "regressed" number, not run63's original 2.238x.** The shift is
   environmental, confirmed by reproducing it on unmodified code, not
   caused by this change. Re-measured tiny's own baseline the same way
   (`results/g6_6_tiny_baseline_recheck_run81.log`): 6.133x, matching
   run63's 6.140x closely (tiny's timing wasn't materially affected by
   whatever drifted) — so the true G6.6 contribution is **6.133 -> 7.232-
   7.245x, a genuine 1.18-1.20x**, cross-validated against a same-session
   baseline rather than a stale cross-session number. **Lesson for any
   future close call this session or later: re-measure a fresh baseline
   under current conditions before trusting a delta against an old log,
   especially when the delta is smaller than ~10%.**

**Shipped.** `check_validity.py` passes. All 7 other shapes confirmed
bit-identical in `max_abs` to `run63`/`run76` (the strongest evidence the
gate touches only the tiny path). Archived: `tiny/fp16` elite updated to
7.24x (from 6.14x), `applied` now includes `G6.6-cublaslt-algo-search`.

### 34. G6.7 (cuBLASLt algorithm search for the FP16 ATTENTION GEMMs) — clean negative, not integrated

**Hypothesis:** step 33 found cuBLASLt's own heuristic-returned algorithm list
contains split-K variants that beat PyTorch's default kernel choice by
1.32-1.49x for the FFN's two **TF32** GEMMs at TINY (M=64). Since G6.4b (step
28) the attention path runs in **FP16**, and at tok=64 its two GEMMs are also
small (`qkv` M=64/K=512/N=1536, `out_proj` M=64/K=512/N=512). Does the same
"default heuristic misses a better variant at small M" phenomenon reproduce for
FP16? Probed standalone in `csrc/cublaslt_algo_fp16.cpp` (a **copy** of the
shipped `csrc/cublaslt_algo.cpp`, deliberately separate so the shipped G6.6 FFN
path is never perturbed) — `CUDA_R_16F` layouts with `CUBLAS_COMPUTE_32F`, i.e.
fp16 storage / fp32 accumulate, matching G6.4b's own
`allow_fp16_reduced_precision_reduction = False`. Added a
`CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` parameter so split-K candidates
that reduce partials in fp16 can be excluded (`mask=2`, fp32 partials) or
allowed (`mask=7`) — a win existing only at `mask=7` would be bought with fp16
partial sums and would need its own accuracy argument.

**Layout mapping re-confirmed independently for FP16**, not assumed from the
FP32 file: an asymmetric M=7/K=24/N=13 shape (all three dims unequal, so any
transposition or `ld` error is a hard mismatch) reproduces `F.linear` with
`maxdiff = 0.000e+00` both with and without bias, and `torch.profiler` shows
PyTorch's own FP16 dispatch is
`cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8` — the same
"tn" configuration the mapping derives, the FP16 equivalent of the FP32 file's
`_tn_` check.

**Run 82 looked like a win and was not.** It reported 1.26x (qkv) and 1.96x
(out_proj) at M=64 — comfortably past the >10% gate. Its own `time_algo2`
instrumentation is what falsified it:

* PyTorch's reference was **launch-bound in every M=64 case** — cpu-issue ≈ gpu
  time (5.63/5.66, 7.75/7.83, 7.70/7.74, 7.80/7.83 us). That "GPU time" is the
  Python dispatch rate, not kernel time.
* The identical measurement moved **5.63 → 7.70 us between two passes (37%)**.
* The two harnesses have different floors: PyTorch's Python loop bottoms out at
  ~7.8 us/call, `time_algo2`'s pure-C++ loop at ~3.3-3.9 us/call. Every M=64
  FP16 kernel here sits **below both floors**.
* `torch.profiler` showed both sides dispatching the **same kernel**, with
  `maxdiff = 0.000e+00` (bit-identical). One kernel cannot be 1.96x faster than
  itself.

This is exactly the confound step 33 built `time_algo2` to catch. It did not
bite there because TF32's M=64 GEMMs were 7.7-30 us — *above* both floors. FP16
is ~3x faster, which drops these GEMMs *underneath* the dispatch floor and makes
the naive comparison measure the two harnesses instead of the two kernels.

**Fair re-measurement (run 83, `probes/g6_7_cublaslt_fp16_probe2.py`)** removes
per-call dispatch symmetrically, two independent ways: (A) CUDA-graph replay —
50 calls captured in one graph, replayed 200x, best of 5, zero per-call dispatch
on either side (also what the real model does under
`torch.compile(mode="reduce-overhead")`); (B) `torch.profiler`
`self_device_time_total / launches`, the GPU's own kernel duration. **They
agree to within 0.4%:**

```
shape                          graph      prof    maxdiff  same-kernel
qkv      M=64 K=512 N=1536    x0.9991   x1.0041   0.0e+00     YES
out_proj M=64 K=512 N=512     x0.9994   x1.0010   0.0e+00     YES
```

**Conclusion: FP16 is already optimally served by cuBLASLt's default heuristic
at this shape.** PyTorch picks the top-ranked candidate (algo 0 for qkv, algo 1
for out_proj) and every other candidate is *slower* — 0.57x-0.98x. Notably the
mechanism of the G6.6 win is absent by construction: at M=64 in FP16 the fastest
candidates are all `splitk=1, reduc=0`, and **every split-K candidate loses**
(0.57x-0.86x). The M=1024 shapes were a negative too (1.00x qkv, 1.03x
out_proj), consistent with step 33's own default-shape finding.

**Not integrated. `benchmark.py` and the shipped `csrc/cublaslt_algo.cpp` are
untouched.** Per step 33's boundary, G6.6 remains scoped to the FFN at
`tok <= _LT_MAX_TOKENS = 127`; the attention path keeps plain FP16 `F.linear`.

**Lesson, generalising step 33's environmental-drift lesson:** a probe's
*reference* needs its own bottleneck audit, not just the candidate's. Whenever
the thing being measured is smaller than the harness's dispatch floor, compare
under CUDA-graph capture (or profiler kernel time) or the number is the
harness's, not the kernel's. Cheap tell used here: if the two sides dispatch the
same kernel name and produce bit-identical output, any "speedup" is measurement.

### 35. G4.0 (two-kernel form) BUILT AND MEASURED at TINY — gate NOT met, closed at G4.0

Step 20 closed G4.0 on a feasibility measurement rather than a build. Re-opened
by request, on the correct objection: step 20 measured the **CPU-side** quantity
(`wall − Σ kernel durations`), and a captured graph replay still pays real
**GPU-side** kernel-boundary costs — SM drain/refill, pipeline restart — that
sit *inside* each kernel's own CUPTI duration and are therefore invisible to a
gap measurement. A genuinely fused kernel could win via that mechanism with zero
measured idle. This step tests that mechanism directly.

Everything below is at **TINY (B=1, S=64)**, on the **current shipped state**
(post-G6.4b fp16 attention, post-G6.6 cuBLASLt FFN GEMMs), re-measured fresh
this session — step 20's census predates both.

**Fresh census** (`probes/g4_0_census.py`, job 86,
`results/g4_0_census_run86.log`). 62 kernels/forward, 196.6 µs wall, 7.27x:

| class | launches | µs | share |
|---|---|---|---|
| flash_fwd (SDPA) | 6 | 42.87 | 21.7% |
| cutlass wmma f16 gemm (qkv, out_proj) | 12 | 42.76 | 21.6% |
| s1688gemm TF32 (ffn_in, ffn_out) | 12 | 57.06 | 28.9% |
| `cublasLt::splitKreduce_kernel` | 12 | 22.72 | 11.5% |
| inductor LN/residual/cast fusions | 13 | 18.36 | 9.3% |
| GELU | 6 | 7.34 | 3.7% |
| cudagraph input `_foreach_copy_` | 1 | 4.41 | 2.2% |

**Finding 1 — the elementwise side is already at G4.0's two-kernel form, and
inductor got there first.** The 13 LayerNorm launches are exactly *two per
layer* plus the entry LN and `final_norm`: one absorbing `residual-add + LN1`,
one absorbing `fp16→fp32 cast + residual-add + LN2 + view`. They cannot be
merged with each other (the QKV GEMM, SDPA and `out_proj` sit between them) and
they cannot be merged into a GEMM (a LayerNorm is a full-row reduction over the
GEMM's entire N dimension; no epilogue can express that). The SDPA output's
`transpose(1,2).contiguous()` does not appear as a kernel at all — already
folded. Per layer the attention block is **2 elementwise launches + 3
unabsorbable launches** (qkv GEMM, flash, out_proj GEMM), which *is* MEGAKERNEL.md's
"as few launches as the tooling can manage around the SDPA call it can't
absorb". There was no Triton kernel left to write on that side: a Triton kernel
cannot fuse across an opaque cuBLASLt/CUTLASS launch, and reimplementing those
is closed by step 19 (0.180x) and step 34 (cuBLASLt already optimal at M=64).

**Finding 2 — the FFN's GELU *can* be fused into the ffn_in GEMM's epilogue, it
*is* worth 3.7%, and it is closed on NUMERICS** (`csrc/cublaslt_gelu.cpp`,
`probes/g4_0_ffn_epilogue_probe.py`, job 87, `results/g4_0_phase1_probes_run87.log`).
This is the one form of "fuse the cheap surrounding op without touching the
GEMM's tiling" that is physically available. It works, and it is free:
`CUBLASLT_EPILOGUE_GELU_BIAS` returns the **identical 8 candidates** as
`CUBLASLT_EPILOGUE_BIAS`, split-K variants included, so step 33's shipped
1.32-1.49x is not given up. Under CUDA-graph replay: gemm alone 6.267 µs,
gemm + separate GELU 7.470 µs, **fused 6.273 µs — the GELU becomes free**,
1.191x on that pair, 7.18 µs/forward = 3.65% at tiny.

But cuBLASLt's GELU is the **tanh approximation**, not erf, and the model's is
`F.gelu(approximate="none")`:

```
|fused − F.gelu(erf) |   = 4.742e-04        |fused − tanh(fp64)| = 5.48e-06
|fused − erf   (fp64)|   = 4.742e-04        |erf(fp64) − tanh(fp64)| = 4.732e-04
|F.gelu(erf) − erf(fp64)| = 3.48e-07   <- torch's own error, 1400x smaller
```

4.74e-04 of *systematic* error on the FFN hidden activation, per layer, six
layers, into a residual stream whose `max_abs` already sits at 7.2e-04 of a
1.0e-03 budget at tiny (`results/g6_6_official_sweep_run76.log`). Same class of
finding as step 30: the approximation's accuracy claim does not survive being
measured. Not integrated, and not worth a walk-down — 3.65% at one shape does
not buy a doubling of `max_abs`.

**Finding 3 — the decisive experiment. Deleting a real kernel boundary at tiny,
with the tiling held fixed, is a 1.4-2.1x REGRESSION.**
(`probes/g4_0_inplace_splitk.py`, job 89, `results/g4_0_inplace_splitk_run89.log`.)

The `splitKreduce_kernel` launches — 12/forward, 22.72 µs, 11.5% of the tiny
forward, the largest deletable launch group left — exist only because step 33's
chosen algorithms use split-K with an out-of-place reduction (`reduc=2`), which
writes partials to workspace and reduces them in a **second kernel**. cuBLASLt
also offers `CUBLASLT_REDUCTION_SCHEME_INPLACE` (`reduc=1`): same algorithm ID,
same tile, same split count, partials accumulated inside the GEMM kernel — **one
launch instead of two**. Reached by adding `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`
to the probe extension. This is precisely G4.0's mechanism, isolated: kernel
boundary removed, nothing else changed. Timed under CUDA-graph replay (step 34's
rule), kernel counts from the profiler rather than assumed:

| GEMM | best out-of-place (2 kernels) | best in-place (1 kernel) | ratio |
|---|---|---|---|
| ffn_in  M=64 N=2048 K=512 | **6.322 µs** (splitk=4) | 8.881 µs (splitk=2) | **0.712x** |
| ffn_out M=64 N=512 K=2048 | **6.768 µs** (splitk=16) | 14.534 µs (splitk=4) | **0.466x** |

Every in-place candidate is slower than its out-of-place twin, and the loss
(2.6 µs and 7.8 µs) is 3-9x larger than the 0.855 µs launch it deletes. Accuracy
is unaffected (`|−fp64|` 1.500e-03 vs 1.501e-03; all variants bit-reproducible
run to run), so this is a pure speed negative. **The kernel boundary was not
costing anything — removing it costs.** That is the GPU-side drain/refill
hypothesis tested head-on at the exact shape and in the exact place where it
should have been most favourable, and it comes back negative.

**Phase 2 — MEGAKERNEL.md's gate, answered three ways**
(`probes/g4_0_ceiling.py`, job 88, `results/g4_0_ceiling_run88.log`).
Same-session baseline in the same job: tiny wall 201.41 µs, 7.058x, 62 kernels,
Σ kernel = 100.55% of wall (**GPU idle −0.55%**, reproducing step 20's −0.7%).

*Upper bound, measured not modelled.* Replaying **only** the 42 GEMM/attention
kernels in one CUDA graph — same shapes, same dtypes, six distinct weight sets
so the ~63 MB working set is real — costs **165.02 µs** (42 kernels traced,
166.04 µs summed = 97.5% of the real forward's 170.35 µs GEMM time, i.e. a
faithful reproduction). So a *perfect, free, zero-cost* fusion of every
elementwise op is worth **36.4 µs = 18.07%** — and that number counts the
LayerNorm/GELU/residual arithmetic itself as recoverable, which it is not; a
megakernel still computes it, just in registers.

*Per-kernel marginal cost, re-measured not cited:* 0.8553 µs (16/64/128/256
no-op kernels in a graph; step 20 measured 0.8554 µs — reproduced to 4 decimal
places).

| gate reading | measured | vs >15% |
|---|---|---|
| A. GPU idle after CUDA Graphs (the doc's literal question) | **−0.55%** | NOT met |
| B. perfect free fusion of all elementwise (strict over-estimate) | 18.07% | met, but not a real quantity |
| C. pure launch cost, 20 × 0.855 µs (the honest "launch overhead") | **8.49%** | NOT met |
| D. direct test: remove one real kernel boundary (finding 3) | **0.71x / 0.47x** | NOT met |

**Decision: STOP at G4.0. Gate NOT met. Not proceeding to G4.1.** Reading B is
the only one that clears 15%, and it clears it by pricing the LayerNorms as
free; the two readings that measure what fusion actually recovers (A and C) are
−0.55% and 8.49%, and the one direct experiment (D) shows a removed boundary
costs 3-9x more than it saves. Escalating to a cooperative persistent kernel —
`grid.sync()`, 99 KB shared budgets, warp specialisation — on the strength of a
number that requires calling arithmetic "overhead" would be exactly the forcing
MEGAKERNEL.md warns against. Its own words: *"G4.0 winning is a result, not a
failure."*

**Nothing shipped, nothing regressed.** `git diff HEAD -- benchmark.py
csrc/cublaslt_algo.cpp` is empty — the shipped model and the shipped G6.6
extension were never touched; all epilogue/reduction-scheme work lives in the
separate probe file `csrc/cublaslt_gelu.cpp` (step 34's precedent).
`tools/check_validity.py` passes. Fresh 8-shape sweep re-run as a same-session
reference (`results/g4_0_reference_sweep_run90.log`).

**One negative worth recording for a future pass:** the per-forward
`multi_tensor_apply_kernel<..., Copy<float,float>>` (4.4 µs at tiny, 2.2%;
11.5 µs at default) is inductor's cudagraph-trees copy of non-static graph
inputs. The obvious hypothesis — that it copies the G0.2/G1.1/G6.4b folded
weights, which CLAUDE.md invariant 4 forces to be plain attributes rather than
Parameters, and which dynamo therefore does not treat as static addresses — was
tested and **falsified**: `torch._dynamo.mark_static_address` on all 48 folded
tensors left the kernel present (4.41 → 4.29 µs), the launch count unchanged
(62 → 62) and the wall time unchanged (0.1943 → 0.1953 ms, 0.995x)
(`probes/g4_0_launch_residue.py`, job 87). Whatever it copies, it is not those.

**What would have to change for G4 to reopen:** a measured GPU-side
kernel-boundary cost that is positive. Finding 3 measures it as *negative* at
tiny — the only regime where the launch count is even a plausible bottleneck —
using cuBLASLt's own in-place split-K, which changes nothing except the number
of launches. That is a stronger closure than step 20's, because it is an
experiment rather than an inference.

### 36. G6.8 (extend G6.6's cuBLASLt algorithm to `ffn_in` at LONG-SEQ, M=8192) — clean negative, not integrated

**Hypothesis.** Re-reading run 71
(`results/g6_6_cublaslt_algo_probe_run71.log`, the `M=8192` block — the
long_seq shape's own token count, B=8 x S=1024), one result was never
chased: `ffn_in` K=512 N=2048 **with bias** shows
`pytorch F.linear 274.08 us` vs `BEST -> algo[3] 245.22 us`,
**x1.1177 (WIN)**. `ffn_out` at the same M is noise (x1.04). Step 33 killed
a structurally identical M=1024 "win" as an artifact of its own eager
baseline, but that had never been checked at M=8192, where inductor's
fusion and kernel-selection heuristics could differ.

**Two tells were already visible inside run 71 itself and both held up.**
The cuBLASLt candidate times are the *same* with and without bias (244-253
vs 245-254 us); the entire 1.1177x lives in the reference, which moves
251.55 -> 274.08 us when a bias is added. The pure-algorithm win at this
shape is **x1.0289 (bias=False)** — noise. And run 73
(`results/g6_6_bias_path_probe_run73.log`, "FFN block, M=8192") had already
measured the whole FFN block under CUDA-graph capture at **x0.9983**.

**Fresh same-session baseline first** (step 33's drift lesson), job 91,
`results/g6_8_longseq_baseline_recheck_run91.log`: long_seq **4.566x**,
`max_abs = 0.000728816` — matches the shipped reference sweep run 90
(4.562x, identical `max_abs`). No drift this session; deltas below are
trustworthy as measured.

**The decisive measurement** (job 92, `probes/g6_8_longseq_ffn_in_probe.py`,
`results/g6_8_longseq_ffn_in_probe_run92.log`), using step 34's fair
protocol (CUDA-graph replay + profiler kernel time + kernel names +
bit-identity):

* Under graph capture, eager `F.linear(x, w, b)` and the "winning" cuBLASLt
  algo **dispatch the same kernel**,
  `cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4`, at
  **244.49 vs 244.46 us (x1.0001)**, `maxdiff = 0.000e+00`. Step 34's tell:
  one kernel cannot be 1.12x faster than itself. The same eager loop run 71
  used reproduces the artifact here (264.98 us with bias vs 248.00 without,
  vs 244.49 captured) — the bias "penalty" is the harness, not the GPU.
* **`torch.compile`'s real lowering already beats the isolated probe's
  premise at M=8192, exactly as at M=1024.** The shipped compiled model's
  `ffn_in` GEMM is a *different tile* from eager's —
  `cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4` at
  **248.02 us/launch** (x6/forward) — already within 0.9% of the best
  cuBLASLt candidate (245.75 us). There is **no separate bias-add kernel**:
  inductor folds `ffn_in`'s bias into `triton_poi_fused_addmm_gelu_view_2`
  (which also does the GELU) and `ffn_out`'s into the LayerNorm/residual
  fusions. There is no addmm penalty left to recover.

**Decomposition of the residual** (job 93,
`probes/g6_8_longseq_decompose.py`,
`results/g6_8_longseq_decompose_run93.log`). Job 92's end-to-end A/B of the
whole G6.6 path at long_seq measured x1.0209 bit-identical, which is not
`ffn_in`'s GEMM — so the FFN branch was made selectable per half and all
four variants timed interleaved, 5 rounds, on the real compiled model:

| variant | median us | x vs shipped |
|---|---|---|
| shipped (`F.linear` both) | 5419.0 | 1.0000 |
| cuBLASLt `ffn_in` only | 5399.6 | **1.0036** |
| cuBLASLt `ffn_out` only | 5404.2 | 1.0027 |
| cuBLASLt both | 5345.3 | 1.0138 |

**`ffn_in` alone is 0.36% — the stated hypothesis is dead**, and it dies for
precisely step 33's reason. Every round was bit-identical (`md = 0.00e+00`).

**Not integrated.** `benchmark.py` and `csrc/cublaslt_algo.cpp` are
untouched; G6.6 stays scoped to `tok <= _LT_MAX_TOKENS = 127`. The only
non-noise number is "both" at 1.4-1.6%, which (a) is not the thing under
test, (b) is below CLAUDE.md's own 2%/iteration stop threshold, and (c)
would be bought by exposing long_seq — a large already-banked G6.4b win at
4.566x — to exactly the risk step 33 named when it scoped the gate to TINY:
selection is by measured speed alone, and run 71 shows `ffn_out` at M=8192
*does* have split-K candidates with `maxdiff` 2.1e-5/2.7e-5 that a different
cuBLAS build or a noisier calibration could pick. Job 92 already saw the
calibration choose three different `ffn_out` algorithms across three rounds.
Not worth 1.4%.

**Lesson (third instance of the same one).** Steps 33, 34 and now 36 all
died the same way: an isolated probe's *reference* was slower than what
`torch.compile` actually emits. The cheap tell keeps working — same kernel
name plus bit-identical output means the delta is the harness. Worth
checking the compiled model's *actual* kernel before believing any
GEMM-level probe again.
