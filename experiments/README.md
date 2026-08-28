# `experiments/` — investigation drivers

64 standalone scripts (was `probes/`). These are **not** the scoring harness —
that is `benchmark.py` / `torch_transformer_benchmark.py` at the repo root.
Each script isolates one question: a correctness check against an fp64
reference, a micro-benchmark, a config sweep, an accuracy pre-check, or a
profiling run. Findings feed `docs/PROGRESS.md`.

## Naming

`gN_<step>[letter]_<topic>.py` maps to the G-stage taxonomy in
`docs/CATALOGUE.md` (generations g0–g6). Plus a few fixed-name utilities:

| Script | Produces |
|---|---|
| `phase0.py` (`infra/slurm/phase0.sbatch`) | day-1 GPU capability ground truth — run first |
| `final_scorecard.py` | the table in `docs/FINAL_SCORECARD.md` |
| `stage_breakdown.py` | per-stage latency decomposition |
| `g4_9_official_profile.py` | official-matrix kernel census |
| `g4_6_gen_cfgs.py` | **generates** `csrc/g4_6_cutlass_cfg00..23.cu` |

## Running one

```bash
infra/run_container.sh python3 /work/experiments/<name>.py
# or the matching batch script:
sbatch infra/slurm/<name>.sbatch
```

## `sys.path` contract — do not nest this directory deeper

Scripts add the repo root to `sys.path` so they can `import benchmark`. Three
patterns are in use, and **all of them assume `experiments/` is a direct child
of the repo root**:

| Pattern | Code | Resolves because |
|---|---|---|
| A (~22 files) | `sys.path.insert(0, "/work")` | the container binds the repo at `/work`, `benchmark.py` is at `/work/benchmark.py` |
| B (~35 files) | `sys.path.insert(0, dirname(dirname(abspath(__file__))))` | parent-of-parent = repo root **only at depth 1** |
| C (2 files) | `sys.path.insert(0, dirname(abspath(__file__)))` | sibling-probe imports within this dir |

Moving `experiments/` under another directory breaks pattern B for ~35 scripts.
A same-depth rename (as `probes/` → `experiments/` was) is safe.

`.ext_build/`, `.cutlass/`, `.sass/` are expected at the repo root (all
gitignored). Recreate `.cutlass/` / `.sass/` per the headers of
`g4_6_cutlass_stage0.py` / `g4_5_sass_roundtrip.py`.
