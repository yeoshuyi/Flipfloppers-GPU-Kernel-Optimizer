# RESUME — submission repo: restructure + README regen DONE

Plan: `~/.claude/plans/crispy-cooking-pine.md`. No git remote — everything
committed on `master`; `dist/techjam2_<hash>.tar.gz` is the artefact to push.

## STATUS: complete. Working tree clean.

Two tasks done this session on top of the converged optimization work:

1. **Restructure** (13 commits, `d8326bc..c83b869`): `jobs/`→`infra/slurm/`,
   `probes/`→`experiments/`, `results/*.log`→`results/logs/`,
   `kernel.def`→`infra/apptainer/`, `DOCUMENTATION.md`→`docs/`. New: `README.md`,
   `Makefile`, `run_eval.sh`, `infra/*.sh`, `docs/ARCHITECTURE.md`, per-dir READMEs.
   Verified job 173: kernels engage from repo root, row 1/13 numbers unchanged.

2. **README regeneration** (`eb4b341`, `1fb7100`): `README.md` is now the single
   comprehensive submission doc (results-first, UVP bullets, accepted/rejected
   optimization tables, 3 Mermaid flowcharts, roofline argument, reproduce,
   Devpost tool/library list, limitations, narrative in `<details>`).
   `tools/make_figures.py` (stdlib SVG) → `assets/{latency_breakdown,pareto_accuracy,roofline}.svg`.
   `SUBMISSION.md` **retired** (user: "super outdated"); prose harvested, refs
   repointed. `LICENSE` = MIT © Yeo Shu Yi (user added, `22c5b1d`).
   Team/contributions section left out — **user hand-writes it**.

## FROZEN — never touched

`benchmark.py`, `torch_transformer_benchmark.py`, `csrc/*` runtime triad,
`tools/` (except new `make_figures.py`), `archive/`. Guards pass:
`verify_baseline` (20 frozen symbols), `sync_entrypoint --check`, `check_validity`.

## To push (user runs)

```bash
cd /scratch/work
git branch -m master main
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
make package && bash infra/verify_submission.sh dist/techjam2_*.tar.gz   # optional final check
```
`dist/`, `.ext_build/`, `.cutlass/`, `.sass/` are gitignored — not pushed.

## Open / for the user

- README "Team & contributions" — omitted by request; add before submitting.
- README is ~4,600 words (comprehensive; absorbed SUBMISSION.md). Trim if desired.
- Verify the 3 Mermaid blocks render in GitHub preview (syntax error → raw code box).
- `docs/PROGRESS.md` still cites pre-restructure paths + one SUBMISSION.md ref
  — intentional historical narrative; README carries the layout-change note.

## Prior work (frozen)

Optimization loop converged at PROGRESS step 52. Shipped causal stack unchanged
since step 42 (run142). `docs/FINAL_SCORECARD.md`: Σ 383.4→60.8 ms, 6.3×,
geomean 7.7×, 13/13 pass. `docs/PARETO_FRONTIER_ANALYSIS.md`: "the stack is at
the frontier."
