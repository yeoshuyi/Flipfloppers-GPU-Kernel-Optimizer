# Causal-Path Optimization Ledger — MERGED into `docs/PROGRESS.md`

This file's content has been merged into `docs/PROGRESS.md`, under the
heading **"## CAUSAL PATH LEDGER (merged from `docs/CAUSAL_LEDGER.md`)"**
near the end of the file (after step 40 / the G4.6 Phase 2 closure) — that
section is now the single source of truth for causal-path work, replacing
this standalone file.

Kept as a short stub, not deleted, so existing references to
`docs/CAUSAL_LEDGER.md` (in `CLAUDE.md`'s load-on-demand list,
`DOCUMENTATION.md`'s citations, and elsewhere) still resolve to something
useful: read this, then jump to that section of `docs/PROGRESS.md`.

**tl;dr of what's there:** causal-path elites — default 2.71x / tiny 7.66x
(`G6.4bc`); long-seq **7.78x** / large-batch **2.98x** (`G4.7c`, PROGRESS
step 42 — fused `ffn_in`+exact-erf-GELU epilogue on the warp-spec kernel,
FP32-accumulate, **precision-neutral**, `max_abs` bit-identical to `G6.4bc`
at 0.00161 / 0.00182; gated `d_model≥512 ∧ ffn_dim≥2048 ∧ tok≥8192`, so it
does not engage on any official 14-row matrix shape) (`archive/causal*__fp16.json`).
`G6.6c` (cuBLASLt) and `G4.6c` (CUTLASS FP16-accum) were both tried and
reverted — real regression / real accuracy failure respectively, not
shipped. `G4.4`/`G4.5` (PTX/SASS) never touched causal, dead ends for
unrelated reasons. Full detail, per-shape max_abs, and the "order of
remaining work" note are in `docs/PROGRESS.md`.
