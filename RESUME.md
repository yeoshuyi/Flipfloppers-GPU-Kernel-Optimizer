# RESUME — repo restructure for competition submission

Plan: `~/.claude/plans/crispy-cooking-pine.md` (REWRITTEN for this task).
No git remote — restructure + commit on `master`, produce `dist/techjam2_<ver>.tar.gz`.
Mode: **auto** — work through phases without stopping; commit each phase.

---

## HARD CONSTRAINT

FROZEN LOCATIONS — never move / never split: `benchmark.py`,
`torch_transformer_benchmark.py`, `csrc/`, `tools/`, `archive/`.
Moving `benchmark.py` or `csrc/` = **silent** kernel disengagement
(`_lt_ext()`/`_ws_ext()` return `None` under `os.path.exists` guards).
The judges' path (`torch_transformer_benchmark.py` + `./csrc` at root) is untouched
by this restructure — only internal dev tooling paths move.

## TARGET LAYOUT

`jobs/`→`infra/slurm/` · `kernel.def`→`infra/apptainer/` · `probes/`→`experiments/`
· `results/*.log`→`results/logs/` · `results/{json,csv,patch}`→`results/artifacts/`
· `DOCUMENTATION.md`→`docs/`. Keep `RESUME.md`, `README.md`(new), `SUBMISSION.md`,
`CLAUDE.md` at root. New: `README.md`, `Makefile`, `run_eval.sh`, `docs/ARCHITECTURE.md`,
`csrc/README.md`, `experiments/README.md`, `archive/README.md`, `infra/*.sh`.
No `src/`, no `agent_logs/` (resolved in README + docs/ARCHITECTURE.md).

## EXECUTION CHECKLIST

- [x] **P1** `0a7046f` track row-14 receipts + RESUME plan
- [x] **P2** `630ff5d` kernel.def → infra/apptainer/; docs/SETUP.md:93
- [x] **P3** `12b0598` results/ → logs/ + artifacts/; 2 cp lines repointed
- [x] **P4** `96710e0` jobs/ → infra/slurm/; tools/slurm.py:6
- [x] **P5** `81b49e8` probes/ → experiments/; 61 sbatch sed'd; GATE passed (0 `/work/probes/` left)
- [x] **P6** `5c7f864` DOCUMENTATION.md → docs/ (RESUME.md kept at root)
- [x] **P7** `a2a761a` README.md, Makefile, run_eval.sh, infra/*.sh, per-dir READMEs, .gitignore +=
- [x] **P8** `edf2d8b` doc-tree rewrites (MANIFEST, SETUP §8/§4/§5, docs/README Loop)
- [x] **P9** `3cd826a` csrc REGISTER PRESSURE / OCCUPANCY blocks (4 kernels + 2 cross-refs)
- [~] **P10** VERIFY — guards green (verify_baseline / sync --check / check_validity all pass;
      both entrypoints parse; `grep -rl /work/probes/ infra/slurm/` = 0; slurm.py → infra/slurm).
      GPU spot-check: **job 173** `results/logs/restructure_smoke_run173.log` — kernels-found
      assertion + rows 1 & 13 via torch_transformer_benchmark.py vs run168 (4.96× / 31.76×).
- [ ] **P11** `make package` → `dist/techjam2_<ver>.tar.gz`; `bash infra/verify_submission.sh dist/*.tar.gz`

## RESUME NOTES

- Each `git mv` + its ref-fixes = ONE commit; `master` never broken.
- P5 sed list = the 61 files in the Plan-agent report (any `infra/slurm/*.sbatch` with `/work/probes/`).
- Verification `verify_baseline.py` needs `~/torch_transformer_benchmark.py` — present in THIS env.
- On resume: `git log --oneline -12` shows which P-commits landed; continue from the first unticked box.
- Pre-restructure numbers to preserve: row1 4.96×, row13 31.76×, Σ 383.4→60.8 ms (docs/FINAL_SCORECARD.md).

## Prior work (frozen, pre-restructure)

Optimization loop CONVERGED at PROGRESS step 52. Shipped causal stack unchanged since
step 42 (run142): G0.1c/G1.1c/G6.4a_v2c/G0.2c/G6.4bc + G4.7c (d512-only, inert on the
official 14-row matrix). G6.9 (cuBLASLt algo-selection) rejected-as-marginal (step 50).
Final scorecard + Pareto analysis delivered (steps 51-52). `benchmark.py` must stay at
step-42 state — this task does not touch it.
