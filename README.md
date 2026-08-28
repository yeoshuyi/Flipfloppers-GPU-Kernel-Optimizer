# Optimizing a Frozen-Baseline Transformer on an RTX 4090

TikTok TechJam submission. The task: make `UserOptimizedTransformer` produce the
same outputs as a **frozen** `BaselineTransformer` (within `atol=0.002` /
`rtol=0.02`, disjunctive per element) while running as fast as possible on a
single consumer GPU.

- **Hardware:** one NVIDIA RTX 4090 (Ada Lovelace, `sm_89`), 24 GB, 72 MB L2.
  No Hopper — no TMA, no `wgmma`, no thread-block clusters.
- **Stack:** CUDA 13.1 · PyTorch (cu13x) · Triton · Apptainer for reproducible runs.
- **Scoring:** CLAUDE.md's official 14-row causal evaluation matrix. Accuracy is
  a hard gate — one failing shape zeroes the benchmark.

## Result

Official 14-row causal matrix, shipped vs. the FP32 baseline (full table in
[`docs/FINAL_SCORECARD.md`](docs/FINAL_SCORECARD.md)):

| | |
|---|---|
| Σ latency, 13 scored rows | **383.4 ms → 60.8 ms** (6.3×) |
| Geometric-mean speedup | **≈ 7.7×** (range 1.9× wide-`d` to 31.8× long-seq) |
| Accuracy | 13/13 pass, `failed = 0`; tightest margin `max_abs = 0.00211` (row 7, cleared on the rtol arm) |
| Row 14 (`S=100000`) | not scorable — the FP32 baseline OOMs a 24 GB card before any math |

Where the remaining time goes and why it is at the accuracy-constrained
roofline: [`docs/PARETO_FRONTIER_ANALYSIS.md`](docs/PARETO_FRONTIER_ANALYSIS.md).

## Run the evaluation

```bash
bash infra/apptainer/build.sh          # builds /scratch/kernel.sif (one time)
./run_eval.sh                          # rows 1-13 through torch_transformer_benchmark.py
#   ENTRY=benchmark.py ./run_eval.sh   # run the human-edited file instead
#   RUN_ROW14=1 ./run_eval.sh          # attempt row 14 (expect OOM)
```

`make help` lists the shortcuts (`make eval` / `check` / `entrypoint` /
`container` / `package` / `verify`).

## Layout

```
benchmark.py                     entry point + source of truth: frozen BaselineTransformer,
                                 our UserOptimizedTransformer, and the scoring harness, in one file
torch_transformer_benchmark.py   GENERATED drop-in = judges' canonical harness + our model
                                 (tools/sync_entrypoint.py; do not hand-edit)
run_eval.sh · Makefile           standardized eval

csrc/                            hand-written CUDA / C++ / inline-PTX kernels  (see csrc/README.md)
tools/                           verify_baseline · sync_entrypoint · check_validity · archive · parse_ncu · slurm
experiments/                     64 g0-g6 investigation drivers (was probes/)  (see experiments/README.md)
infra/
  apptainer/  kernel.def + build.sh     reproducible image
  slurm/      *.sbatch                  batch scripts (was jobs/)
  run_container.sh · package.sh · verify_submission.sh
results/
  logs/                          120 Slurm job stdout receipts — every number in the docs traces here
  artifacts/                     ncu JSON summaries, ground_truth.csv, one .patch
archive/                         MAP-Elites elite-config store  (see archive/README.md)
docs/                            ARCHITECTURE · PROGRESS (52-step log) · DOCUMENTATION · FINAL_SCORECARD
                                 · PARETO_FRONTIER_ANALYSIS · ACCURACY_BUDGET · SETUP · CATALOGUE · ...
CLAUDE.md                        the agent's operating manual — this repo was built as an agentic log
```

## Where is the model code?

All of it — model definitions, custom-op dispatch (`_lt_ext`/`_ws_ext`/
`_ffn_register_op`), the regime gates, and the G1 weight-precompute — lives in
the single file **`benchmark.py`**, on purpose:

- `tools/sync_entrypoint.py` splices our `UserOptimizedTransformer` into the
  judges' harness **by text markers** to produce `torch_transformer_benchmark.py`.
- `tools/verify_baseline.py` **AST-diffs** the 20 frozen baseline symbols in
  `benchmark.py` against the judges' canonical script on every change.
- `benchmark.py` locates its CUDA sources relative to its own path and guards
  them with `os.path.exists` — a moved/renamed `csrc/` returns `None` and the
  custom kernels **silently** fall back to eager (still "passes", just slow).

Splitting or relocating the file breaks all three. The conceptual module map
(concept → line range) is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Where are the agent logs?

No MCP or telemetry capture was set up. The verifiable record is:

- `results/logs/*.log` — 120 Slurm job receipts (accuracy sweeps, benchmarks, profiles).
- [`docs/PROGRESS.md`](docs/PROGRESS.md) — the 52-step chronological engineering log.
- [`docs/RESUME.md`](docs/RESUME.md) — the session cursor used to survive token limits.
- [`docs/AGENTS.md`](docs/AGENTS.md) + `.claude/agents/profiler.md` — the one profiler
  subagent and why the other roles were replaced by scripts
  (`tools/check_validity.py`, `tools/archive.py`).

## Verify a package

```bash
make package                                   # -> dist/techjam2_<ver>.tar.gz
bash infra/verify_submission.sh dist/techjam2_*.tar.gz
```

`tools/verify_baseline.py` and `tools/sync_entrypoint.py --check` need the
judges' canonical `~/torch_transformer_benchmark.py`; the packaging scripts warn
and continue if it is absent on a fresh clone.

---

*Layout note (2026-08-28): `probes/` → `experiments/`, `jobs/` → `infra/slurm/`,
`results/*.log` → `results/logs/`, `kernel.def` → `infra/apptainer/`,
`DOCUMENTATION.md` → `docs/`. Older narrative in `docs/PROGRESS.md` and
`SUBMISSION.md` may cite the pre-restructure paths.*
