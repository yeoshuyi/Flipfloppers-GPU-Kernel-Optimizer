# RESUME — repo restructure: DONE

Plan: `~/.claude/plans/crispy-cooking-pine.md`. No git remote — restructure
committed on `master`; clean tarball at `dist/` for the user to push.

## STATUS: complete (P1–P11). Working tree clean.

Deliverable: **`dist/techjam2_<hash>.tar.gz`** (604 K, 409 entries, no build
artefacts) — rebuild any time with `make package`; check with `make verify`.

Rebuild+verify last run: `verify_submission: PASS` — entrypoints parse, frozen
symbols intact, csrc triad present, archive clean, `lt=True ws=True` (custom
kernels engage from the repo root).

## What changed (13 commits, `d8326bc..HEAD`)

| Move | Commit |
|---|---|
| `kernel.def` → `infra/apptainer/` | `630ff5d` |
| `results/*.log` → `results/logs/`, `{json,csv,patch}` → `results/artifacts/` | `12b0598` |
| `jobs/` → `infra/slurm/` (+ `tools/slurm.py`) | `96710e0` |
| `probes/` → `experiments/` (+ 61 sbatch sed) | `81b49e8` |
| `DOCUMENTATION.md` → `docs/` | `5c7f864` |
| new: `README.md` `Makefile` `run_eval.sh` `infra/*.sh` `docs/ARCHITECTURE.md` + per-dir READMEs + `.gitignore` | `a2a761a` |
| doc-tree rewrites (MANIFEST / SETUP / docs-README) | `edf2d8b` |
| `csrc/` REGISTER-PRESSURE/OCCUPANCY header blocks (4 kernels) | `3cd826a` |
| verify smoke (job 173) + package.sh polish | `00e4792`, `fbaa6e5` |

## FROZEN — untouched, verified engaging

`benchmark.py`, `torch_transformer_benchmark.py`, `csrc/*` (runtime triad),
`tools/`, `archive/` — all at repo root. `verify_baseline` (20 frozen symbols),
`sync_entrypoint --check`, `check_validity` all pass. Job 173: row 1 4.86×,
row 13 31.71×, `max_abs 0.0013676` — identical to run168.

## If more work is asked

- The optimization loop is CONVERGED (PROGRESS step 52). `benchmark.py` stays at
  the step-42/run142 shipped state — do not modify it.
- `docs/RESUME.md` does NOT exist — RESUME.md was intentionally kept at repo root
  (dev-methodology artifact, part of the agentic-log story).
- Open item flagged in the plan: no `LICENSE` file. User to choose / add at push time.
- `docs/PROGRESS.md` and `SUBMISSION.md` still cite some pre-restructure paths
  (`probes/…`, `jobs/…`) as historical narrative — intentional, README carries the
  layout-change note.

## Prior work (frozen)

Optimization loop converged at PROGRESS step 52. Shipped causal stack unchanged
since step 42 (run142). Final scorecard + Pareto analysis: `docs/FINAL_SCORECARD.md`,
`docs/PARETO_FRONTIER_ANALYSIS.md`. Σ 383.4→60.8 ms, 6.3×, geomean 7.7×, 13/13 pass.
