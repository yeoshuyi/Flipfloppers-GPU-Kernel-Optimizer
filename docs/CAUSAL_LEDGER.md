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

**tl;dr of what's there:** causal-path elites, all tag `G6.4bc` — default
2.71x, tiny 7.66x, long-seq 7.10x, large-batch 2.66x (`archive/causal*.json`).
`G6.6c` (cuBLASLt) and `G4.6c` (CUTLASS FP16-accum) were both tried and
reverted — real regression / real accuracy failure respectively, not
shipped. `G4.4`/`G4.5` (PTX/SASS) never touched causal, dead ends for
unrelated reasons. Full detail, per-shape max_abs, and the "order of
remaining work" note are in `docs/PROGRESS.md`.
