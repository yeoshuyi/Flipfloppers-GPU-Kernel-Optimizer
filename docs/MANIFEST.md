# MANIFEST — Transfer Bundle

Everything needed to run the project. Copy the whole tree to
`/scratch/work/` on `ubuntu-makers`.

Post-restructure layout (2026-08-28). Top-level `README.md` is the front door;
`docs/ARCHITECTURE.md` maps concepts to `benchmark.py` line ranges.

```
README.md                    front door — problem, result, layout, quickstart
benchmark.py                  entry point + source of truth (frozen baseline + our model + harness)
torch_transformer_benchmark.py  GENERATED drop-in (judges' harness + our model). Never hand-edit.
run_eval.sh · Makefile        standardized eval

CLAUDE.md                    ~1.9k tok  invariants, ground truth, regime dispatch, loop
RESUME.md                    session cursor (survives token limits)

.claude/
  agents/profiler.md         ~0.3k tok  the ONE subagent (isolated context)
  settings.json              permissions — see docs/AGENTS.md §8

docs/                        LOAD ON DEMAND — never all at once
  ARCHITECTURE.md            concept → benchmark.py line range + the 2 sync guards
  DOCUMENTATION.md           full technical record (every opt shipped/reverted/closed)
  PROGRESS.md · FINAL_SCORECARD.md · PARETO_FRONTIER_ANALYSIS.md
  CATALOGUE.md · DIAGNOSIS.md · ACCURACY_BUDGET.md · MEGAKERNEL.md · SETUP.md · AGENTS.md
  MANIFEST.md (this file) · README.md · CAUSAL_LEDGER.md

csrc/                        hand-written CUDA / C++ / inline-PTX  (csrc/README.md — 3 files build at eval)
tools/
  check_validity.py          static gate — replaces the adversary agent (0 tokens)
  archive.py                 MAP-Elites — replaces the archivist agent (0 tokens)
  slurm.py                   submit() / poll() — never block on srun
  verify_baseline.py         AST-diff benchmark.py's frozen half vs ~/torch_transformer_benchmark.py
  sync_entrypoint.py         regenerate torch_transformer_benchmark.py
  parse_ncu.py               ncu --csv → compact JSON for the profiler subagent

experiments/                 64 g0–g6 investigation drivers  (experiments/README.md)
  phase0.py                  capability probe — RUN THIS FIRST
infra/
  apptainer/kernel.def + build.sh    reproducible image
  slurm/*.sbatch                     batch scripts (Apptainer + exclusive GPU)
  run_container.sh · package.sh · verify_submission.sh
results/logs/                120 Slurm job receipts       results/artifacts/  ncu JSON, ground_truth.csv, .patch
archive/                     MAP-Elites elite-config store  (archive/README.md)
```

**Context budget:** ~1.9k always + 0.4–2.4k on demand. Worst case if one session
touches everything ≈ 8.5k. Typical G0–G2 iteration ≈ 3.6k.

---

## Transfer checklist

```
[ ] copy tree to /scratch/work/
[ ] create .claude/settings.json from docs/AGENTS.md §8
[ ] configure Slurm + prolog clock locking      docs/SETUP.md §2
[ ] build Apptainer image                       docs/SETUP.md §3
[ ] enable ncu perf counters + reboot           docs/SETUP.md §3
[ ] sbatch infra/slurm/phase0.sbatch            <- GATES EVERYTHING
[ ] record baseline sweep -> results/artifacts/ground_truth.csv
[ ] build Track A, verify it passes             <- non-negotiable safety net
[ ] bootstrap-validate the loop on G3.2         docs/AGENTS.md §6
[ ] then start Track B
```

---

## Coverage — every optimisation discussed

### G0 Structural (6)
`G0.1` SDPA + `is_causal` · `G0.2` fused QKV · `G0.3` kill `.contiguous()`
transposes · `G0.4` cached causal mask · `G0.5` all-ones mask path ·
`G0.6` 128-bit vector loads

### G1 Constant folding — exact, prerequisite for G4 (6)
`G1.1` LayerNorm affine absorption · `G1.2` scale → `W_Q` · `G1.3` `log2(e)` →
`exp2` · `G1.4` `ldmatrix` pre-swizzle · `G1.5` per-channel scales ·
`G1.6` single weight arena

### G2 Precision + residency (8)
`G2.1` BF16 + FP32 residual · `G2.2` FP32 softmax/LN · `G2.3` L2 persistence ·
`G2.4` CUDA Graphs · `G2.5` FP32 output cast-back · `G2.6` FP8 FFN per-channel ·
`G2.7` INT8 FFN (fallback if FP8 unavailable) · `G2.8` split-precision
(Ootomo & Yokota)

### G3 Fusion + layout (7)
`G3.1` fused FFN tile · `G3.2` fused LN+residual · `G3.3` warp-shuffle
reductions (Welford) · `G3.4` XOR swizzle · `G3.5` `cp.async` pipeline ·
`G3.6` minimax GELU · `G3.7` `__launch_bounds__` sweep

### G4 Megakernel (6)
`G4.0` two-kernel form · `G4.1` persistent cooperative · `G4.2` K-dimension
split · `G4.3` warp specialisation · `G4.4` `mma.sync` FP16 accumulate ·
`G4.5` gated softmax max-subtraction skip

**33 total.**

### Cross-cutting
- Multi-regime dispatch: TINY / DEFAULT / LONG-SEQ / LARGE-BATCH / PADDED /
  CAUSAL — `CLAUDE.md`
- Track A safety net — `docs/CATALOGUE.md`, `docs/SETUP.md §9`
- Solidification (strip autotune) — `CLAUDE.md`
- MAP-Elites 2D archive — `CLAUDE.md`, `tools/archive.py`
- Phase 0 capability probe — `experiments/phase0.py`, `docs/SETUP.md §6`
- Measurement protocol + clock locking — `docs/SETUP.md §2, §7`
- Agent roles, limits, best practices — `docs/AGENTS.md`

---

## The four numbers

- 18,915,328 params → **37.8 MB BF16** vs **72 MB L2** — the model lives on-chip
  (FP32 at 75.66 MB misses by 5%)
- 40.27 GFLOP/forward → **0.244 ms** BF16 floor, **0.122 ms** FP8
- **82.6 → 165 → 330** TFLOPS — 4× headroom before any fusion work
- **24.1 KB/SM** — one FP8 layer → 4 pipeline stages (BF16 gets 3)

## The four traps

1. `strict=True` rejects fused params → **plain attributes**
2. GeForce FP32-accumulate is **half rate** → 660 TFLOPS is FP16-accum only
3. Explicit `attn_mask` kicks SDPA off flash → **`is_causal=True`**
4. Accuracy failure **skips the benchmark entirely** → correctness is a gate

## The one rule that matters most

**Never let raw `ncu` output reach the main context.** It is 25k–100k tokens.
The profiler subagent exists for exactly this. Use it.
