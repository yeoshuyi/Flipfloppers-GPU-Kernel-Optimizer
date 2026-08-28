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

## EXECUTION CHECKLIST  (tick as done; commit per phase)

- [ ] **P1** `git add jobs/row14_extreme.sbatch results/row14_extreme_run172.log` → commit "chore: track row-14 receipts"
- [ ] **P2** `git mv kernel.def infra/apptainer/kernel.def`; fix `docs/SETUP.md:93` → commit
- [ ] **P3** split `results/`: `*.log`→`results/logs/`, `{*.json,*.csv,*.patch}`→`results/artifacts/`;
      fix `jobs/final_scorecard.sbatch:18` + `jobs/row14_extreme.sbatch:38` cp targets → commit
- [ ] **P4** `git mv jobs infra/slurm`; fix `tools/slurm.py:6` (`jobs/bench.sbatch`→`infra/slurm/bench.sbatch`) → commit
- [ ] **P5** `git mv probes experiments && rm -rf probes`;
      `sed -i 's#/work/probes/#/work/experiments/#g'` across the 61 `infra/slurm/*.sbatch`;
      fix `experiments/phase0.sbatch` self-path; harden the 2 hardcoded `/work/csrc/cublaslt_gelu.cpp` probes;
      GATE: `grep -rl '/work/probes/' infra/slurm/` empty → commit
- [ ] **P6** `git mv DOCUMENTATION.md docs/DOCUMENTATION.md`; grep-fix refs → commit  (RESUME.md stays at root)
- [ ] **P7** new files: `README.md`, `Makefile`, `run_eval.sh`(+x), `infra/apptainer/build.sh`,
      `infra/run_container.sh`, `infra/package.sh`, `infra/verify_submission.sh`,
      `csrc/README.md`, `experiments/README.md`, `archive/README.md`, `docs/ARCHITECTURE.md`;
      `.gitignore` += `/dist/ *.tar.gz .claude/settings.local.json *.ncu-rep *.nsys-rep *.cubin *.o *.so .pytest_cache/ .venv/ .DS_Store` → commit
- [ ] **P8** doc-tree rewrites: `docs/MANIFEST.md` tree §, `docs/SETUP.md` §8/§3/§5,
      `CLAUDE.md` LOOP step 5 + jobs/ refs, `docs/README.md` Files+Loop blocks;
      layout-change note in `README.md` + `docs/MANIFEST.md`. NO mass rewrite of PROGRESS/DOCUMENTATION/SUBMISSION → commit
- [ ] **P9** CUDA doc headers: `csrc/g4_4_mma_gemm.cu` (full), `csrc/g5_mega_causal.cu` (expand),
      `csrc/g4_6_cutlass_gemm.cuh` (add), `csrc/g4_4_warpspec_gemm.cu` (reg/occupancy subsection) → commit
- [ ] **P10** VERIFY (see plan Verification §) — `verify_baseline.py`, `sync_entrypoint.py --check`,
      `check_validity.py`, `grep -rl '/work/probes/' infra/slurm/` empty, kernels-found apptainer check,
      2-row spot-check vs `results/logs/official_causal_sweep_run168.log`
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
